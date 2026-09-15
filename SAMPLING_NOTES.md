# Pick-skill demonstration recorder — what this is and how to use it

Records physics-only `pick` episodes (arm B, mug) into a LeRobotDataset v3 directory that
passes `stage3_policy.learned.validate_dataset` with status VALID (tested: 2 episodes, 1 train / 1 val).

## Files
- `bimanual_scene_weld.xml` — copy of `assets/bimanual_scene.xml` plus one toggleable weld
  equality constraint (`b_grasp_mug`, site-based). Put it in `assets/`.
- `record_pick_episodes.py` — run from the repo root.

## Run (from repo root, needs mujoco + pyarrow + numpy)
    SEEDS=101,102,103 SPLIT=train python record_pick_episodes.py
    SEEDS=201,202     SPLIT=val   python record_pick_episodes.py
    python -m stage3_policy.learned.validate_dataset --root data/butler_demos/pick

Append-mode: each run adds episodes to the existing dataset. Seeds 0–9 are reserved for
evaluation — never use them here. Keep train/val seeds disjoint. Target 50+ successful
train episodes per the validator's warning threshold.

## Three bugs found on the way (worth telling Stage 4)
1. **qpos vs dof addressing.** `model.jnt_qposadr` and `model.jnt_dofadr` are different arrays once
   free-floating objects exist earlier in the model (7 qpos slots vs 6 dof slots each). Writes to
   `data.qpos` must use `jnt_qposadr`; Jacobian columns use `jnt_dofadr`. Mixing them makes joint
   commands silently do nothing. Any IK/control code in the repo should be checked for this.
2. **Weld anchor.** `<weld body1=... body2=...>` welds body *origins*. The gripper body origin is
   ~11 cm from the fingertip, so a body weld fights a huge offset and fails. Use
   `<weld site1="b_gripperframe" site2="mug_site">` instead — welds the actual fingertip.
3. **Gripper overshoot.** A close command near the joint limit lets physics overshoot past it and
   the validator rejects the frame. Command 0.05 rad, not -0.15. The weld carries the grasp;
   the gripper value only needs to be plausible and in-range.

## What's NOT done
- Images are declared in `meta/info.json` but not written to disk (validator doesn't check them;
  LeRobot training will). Add PNG/video writing before real training.
- `open_drawer` is not solved: the D-handle needs orientation-aware IK, not just position.
- `pick plate` requires the drawer open first (plate starts inside it). Init `drawer_slide` open
  for standalone plate episodes.
- Per-skill weld constraints are needed for the other objects (one `<weld>` each).
- ~110 s/episode here is software rendering with no GPU; expect much faster on the Intel box.
