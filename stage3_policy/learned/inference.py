"""Select a trained LeRobot ACT motor policy, or report exactly why the scripted fallback is used.

    python -m stage3_policy.learned.inference            # checks models.policy_checkpoint in configs/default.yaml

The symbolic plan always comes from the rule-based planner. An ACT policy would
sit BELOW it, inside the executor: during one skill (e.g. "pick up the mug with
arm B") it maps each 25 Hz observation to 12 joint position targets. No executor
calls this yet; that needs the Stage 4 contract in stage3_policy/CONTRACT_PROPOSAL.md.

ActMotorPolicy follows the lerobot 0.6.1 inference path (ACTPolicy.from_pretrained,
make_pre_post_processors, prepare_observation_for_inference). It has NOT been run:
no trained checkpoint exists and lerobot is not installed in the Stage 3 environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stage3_policy.learned import schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
REQUIRED_CHECKPOINT_FILES = ("config.json", "model.safetensors")
LEARNED, FALLBACK = "learned_act", "scripted_primitives"


@dataclass(frozen=True)
class PolicySelection:
    mode: str  # LEARNED or FALLBACK
    reason: str
    checkpoint_dir: Path | None
    policy: ActMotorPolicy | None = None


def configured_checkpoint(config_path: Path = DEFAULT_CONFIG) -> str | None:
    """models.policy_checkpoint from the shared config (a path string, not proof a model exists)."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = (config.get("models") or {}).get("policy_checkpoint")
    return str(value) if value else None


def select_motor_policy(checkpoint: str | Path | None, *, device: str = "cpu", load: bool = True) -> PolicySelection:
    def fallback(reason: str, directory: Path | None = None) -> PolicySelection:
        return PolicySelection(FALLBACK, reason, directory)

    if checkpoint is None:
        return fallback("no ACT checkpoint is configured")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path
    directory = path.parent if path.suffix == ".safetensors" else path
    if not directory.is_dir():
        return fallback(f"checkpoint directory {directory} does not exist; a path in a config file is not a trained model")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (directory / name).is_file()]
    if missing:
        return fallback(f"{directory} is missing {missing}", directory)
    missing_modules = [name for name in ("torch", "lerobot") if importlib.util.find_spec(name) is None]
    if missing_modules:
        return fallback(f"{missing_modules} not installed in this interpreter", directory)
    if not load:
        return PolicySelection(LEARNED, "checkpoint files and dependencies are present (not loaded)", directory)
    try:
        policy = ActMotorPolicy.load(directory, device=device)
    except Exception as exc:  # a broken checkpoint must not take down the scripted executor
        return fallback(f"loading {directory} failed: {type(exc).__name__}: {exc}", directory)
    return PolicySelection(LEARNED, f"loaded ACT checkpoint from {directory}", directory, policy)


class ActMotorPolicy:
    """One ACT checkpoint for one skill: observation -> 12 joint position targets (rad)."""

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, device: Any) -> None:
        self._policy, self._pre, self._post, self._device = policy, preprocessor, postprocessor, device

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> ActMotorPolicy:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = ACTPolicy.from_pretrained(str(directory))
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(directory))
        return cls(policy, preprocessor, postprocessor, torch.device(device))

    def reset(self) -> None:
        """Call at the start of every skill execution (clears ACT's queued action chunk)."""
        self._policy.reset()

    def act(self, joint_positions: Sequence[float], overhead_rgb: Any, instruction: str) -> list[float]:
        import numpy as np
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray(joint_positions, dtype=np.float32)
        image = np.asarray(overhead_rgb)
        if state.shape != (len(schema.MOTOR_NAMES),):
            raise ValueError(f"expected {len(schema.MOTOR_NAMES)} joint positions in {schema.MOTOR_NAMES} order, got {state.shape}")
        if image.shape != schema.IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"expected a uint8 {schema.IMAGE_SHAPE} RGB frame, got {image.dtype} {image.shape}")

        observation = {schema.STATE_KEY: state, schema.IMAGE_KEY: image}
        batch = prepare_observation_for_inference(observation, self._device, task=instruction, robot_type=schema.ROBOT_TYPE)
        action = self._post(self._policy.select_action(self._pre(batch)))
        values = [float(v) for v in action.squeeze(0).to("cpu").tolist()]
        if len(values) != len(schema.MOTOR_NAMES) or not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"policy returned an invalid action {values!r}")
        return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stage3_policy.learned.inference", description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="checkpoint directory (default: models.policy_checkpoint in configs/default.yaml)")
    parser.add_argument("--no-load", action="store_true", help="check files and dependencies without loading weights")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or configured_checkpoint()
    selection = select_motor_policy(checkpoint, load=not args.no_load)
    print(f"configured checkpoint: {checkpoint}")
    print(f"motor policy mode: {selection.mode}")
    print(f"reason: {selection.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
