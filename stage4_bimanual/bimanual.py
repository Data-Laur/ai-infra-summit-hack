"""Stage 4: bimanual execution of Actions with two SO-101 arms in MuJoCo."""

# TODO(Azeem): replace stub with real implementation

from dataclasses import dataclass
from typing import Any

from common.types import Action, ExecutionResult, SceneState


@dataclass
class FakeSim:
    """Placeholder for the MuJoCo sim handle until the real scene loader lands."""

    seed: int


def reset_scene(seed: int) -> FakeSim:
    """Reset and randomize the scene for a seed (ranges from configs/default.yaml)."""
    return FakeSim(seed=seed)


def get_camera_frame(sim: Any) -> Any | None:
    """Return the tabletop camera frame (np.ndarray in the real implementation)."""
    return None


def execute(actions: list[Action], sim: Any | None = None) -> ExecutionResult:
    """Pretend to run every action on the dual-arm sim and report success."""
    final_scene = SceneState(
        objects={
            "plate": (0.20, 0.10, 0.02),
            "mug": (0.30, -0.10, 0.05),
            "water_bottle": (-0.15, 0.20, 0.11),
            "spoon": (0.05, -0.25, 0.01),
            "fork": (0.08, -0.25, 0.01),
        },
        drawers={"top_drawer": "open"},
    )
    return ExecutionResult(
        action_results={a.step_id: True for a in actions},
        success=True,
        final_scene=final_scene,
        error=None,
    )
