"""Record physics-based demonstrations of one skill into a LeRobotDataset v3 (one dataset per skill).

    SEEDS=101 SPLIT=train .venv-learned/bin/python3 record_episodes.py                 # arm B picks the mug
    SKILL=open_drawer ARM=A SEEDS=101,102 SPLIT=train .venv-learned/bin/python3 record_episodes.py
    SKILL=pick ARM=A OBJECT=fork PRE="open_drawer:A" SEEDS=101 .venv-learned/bin/python3 record_episodes.py
    SKILL=place ARM=B OBJECT=fork XY=0.12,0.07 PRE="open_drawer:A pick:A:fork place:A:fork:0:0 pick:B:fork" ...

PRE lists setup skills (same skill:arm[:object[:x:y]] syntax as stage4_bimanual.review_skill) that
run before recording starts; only the target skill's frames are written. Runs in .venv-learned
(needs lerobot's writer and mujoco). Datasets land in data/butler_demos/<skill>/.
"""

import json
import os
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage3_policy.learned import schema
from stage4_bimanual import primitives as P
from stage4_bimanual.review_skill import run_spec

INSTRUCTIONS = {
    "pick": "pick up the {object} with arm {arm}",
    "place": "place the {object} on the table with arm {arm}",
    "open_drawer": "open the top drawer with arm {arm}",
}
JITTER_M = 0.02


def get_or_create_dataset(root: Path, repo_id: str):
    if (root / "meta" / "info.json").exists():
        return LeRobotDataset.resume(repo_id=repo_id, root=str(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id=repo_id, fps=schema.FPS, root=str(root), robot_type=schema.ROBOT_TYPE,
                                 features=schema.lerobot_features(use_videos=False), use_videos=False)


def record_episode(sim: P.Sim, renderer, seed: int, skill: str, arm_letter: str, obj: str | None, xy, pre: list[str]):
    """Run the setup skills unrecorded, then the target skill with every frame logged."""
    arm = sim.arms[arm_letter]
    jitter_obj = obj if skill == "pick" else None
    sim.reset(seed, jitter={jitter_obj: JITTER_M} if jitter_obj else None)
    sim.frame_hook = None
    for spec in pre:
        ok, _ = run_spec(sim, spec)
        if not ok:
            return [], False, {"setup_failed": spec}

    frames = []

    def log(s):
        renderer.update_scene(s.data, camera=schema.CAMERA_NAME)
        frames.append({
            "state": s.all_joint_qpos(schema.MOTOR_NAMES),
            "action": np.array(s.data.ctrl[:len(schema.MOTOR_NAMES)], dtype=np.float32),
            "image": renderer.render().copy(),
        })
    sim.frame_hook = log
    try:
        ok, metrics = run_spec(sim, ":".join(str(x) for x in [skill, arm_letter] + ([obj] if obj else []) + (list(xy) if xy else [])))
    finally:
        sim.frame_hook = None
    return frames, ok, metrics


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "101").split(",")]
    split = os.environ.get("SPLIT", "train")
    skill = os.environ.get("SKILL", "pick")
    arm_letter = os.environ.get("ARM", "B").upper()
    obj = os.environ.get("OBJECT", "mug") if skill != "open_drawer" else None
    xy = tuple(float(v) for v in os.environ["XY"].split(",")) if skill == "place" else None
    pre = os.environ.get("PRE", "").split()
    instruction = INSTRUCTIONS[skill].format(object=obj, arm=arm_letter)
    root = Path("data/butler_demos") / skill

    sim = P.Sim()
    import mujoco
    renderer = mujoco.Renderer(sim.model, height=schema.IMAGE_SHAPE[0], width=schema.IMAGE_SHAPE[1])
    dataset = get_or_create_dataset(root, f"local/butler_demos_{skill}")
    meta_path = root / "meta" / "butler_episodes.json"
    episodes_meta = json.loads(meta_path.read_text())["episodes"] if meta_path.exists() else []

    for seed in seeds:
        episode_index = dataset.meta.total_episodes
        frames, success, metrics = record_episode(sim, renderer, seed, skill, arm_letter, obj, xy, pre)
        if not frames:
            print(f"seed {seed}: setup failed ({metrics}); skipped")
            continue
        n = len(frames)
        for fi, fr in enumerate(frames):
            dataset.add_frame({
                schema.STATE_KEY: fr["state"], schema.ACTION_KEY: fr["action"], schema.IMAGE_KEY: fr["image"],
                schema.SUCCESS_KEY: np.array([success and fi == n - 1]), schema.DONE_KEY: np.array([fi == n - 1]),
                "task": instruction,
            })
        dataset.save_episode()
        episodes_meta.append({
            "episode_index": episode_index, "seed": seed, "skill": skill, "arm": arm_letter,
            "object": obj or "top_drawer", "instruction": instruction, "split": split,
            "success": success, "physics_only": True,
            **{k: round(float(v), 4) for k, v in metrics.items()},
        })
        print(f"episode {episode_index} seed={seed} split={split} skill={skill} frames={n} success={success} "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    dataset.finalize()
    meta_path.write_text(json.dumps({"episodes": episodes_meta}, indent=2))
    print("dataset now has", dataset.meta.total_episodes, "episodes,", dataset.meta.total_frames, "frames at", root)


if __name__ == "__main__":
    main()
