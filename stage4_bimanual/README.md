# stage4_bimanual — physics-based table-setting primitives (branch `stage4-lauren`)

`bimanual.py` is still the original stub (`execute()` steps physics and reports success). The
real motion lives in `primitives.py`: scripted, physics-based skills for the two SO-101 arms in
`assets/bimanual_scene_weld.xml`, with ground-truth success checks. They are the "scripted
fallback" of the policy stage; the learned ACT policy is trained on demonstrations they record.

## What works (seed 101, `python -m stage4_bimanual.review_skill ...`)

| Skill | Arm | Result |
|---|---|---|
| `open_drawer` | A | hooks the D-handle, pulls the slide 15.4 cm of 16, 0.5 mm worst contact |
| `pick` mug (table) | B | 10 cm lift, ~4° tilt, <1 mm gripper/object penetration |
| `pick` plate / fork / spoon (from the open drawer) | A or B | real fingertip contact, 8–10 cm lift, 2–4° tilt |
| `place` | A, B | lowers until the object rests on something, releases; 0.4–2.2 cm from target, flat |
| full sequence | A+B | 15/15: mug → drawer → plate, fork, spoon each taken out by A, handed over at (-0.02, 0.02), set by B |

Full run (2 min 43 s of sim time, 4 076 frames): `outputs/review/table_setting_demo.mp4`, made with

```bash
.venv-learned/bin/python -m stage4_bimanual.review_skill pick:B:mug place:B:mug:0.04:0.34 open_drawer:A \
  pick:A:plate place:A:plate:-0.02:0.02:free pick:B:plate place:B:plate:0.12:0.16 \
  pick:A:fork  place:A:fork:-0.02:0.02:free  pick:B:fork  place:B:fork:0.12:0.07 \
  pick:A:spoon place:A:spoon:-0.02:0.02:free pick:B:spoon place:B:spoon:0.12:0.25
```

(`:free` = hand-off placement without yaw alignment; the review writes a slow-motion mp4 and a
key-frame sheet under `outputs/review/`.)

## How a grasp works, and what "success" means

- Approach from 15 cm above; **guarded descent** stops at the first fingertip contact (>0.8 mm)
  with the object or its support, so nothing is pressed into the table or rim.
- A body **weld** (`<arm>_grasp_<object>` in the scene XML) is activated with the relative pose
  captured at that instant — the object is never yanked or rotated. The drawer uses a `connect`
  (ball joint) so the wrist can rotate while pulling.
- IK runs on a scratch `MjData` (never teleports the live sim), holds the gripper's approach axis
  on vertical moves (5-DOF arm: 3 position + 2 rotation constraints), and long carries are
  waypointed. Joint speed is capped at 0.5 rad/s: the sts3215 servo model saturates at 3.35 N·m.
- `success` is computed from simulator state, never self-reported: pick = touched the object
  AND lifted >5 cm AND worst penetration <3 mm; drawer = opened >12 cm; place = within 3 cm,
  <3 cm above the table, <10° tilt. Every metric is logged per episode.

## Layout (why the hand-off)

Arm A can only place within ~0.40 m of its base, where the drawer and bottle already sit, so
drawer items go to a hand-off zone at (-0.02, 0.02) — reachable by both arms, clear of the
bottle — and arm B sets the
table: plate (0.12, 0.16), fork (0.12, 0.07), spoon (0.12, 0.25), mug (0.04, 0.34), placemat
centred on (0.12, 0.16). Set the plate before the utensils (a plate lowered onto a fork tips over).

## Scene changes in `assets/bimanual_scene_weld.xml` (vs `bimanual_scene.xml`)

- Visual meshes have `mass="0"`. **Bug also present in `bimanual_scene.xml`:** without it MuJoCo
  gives the meshes default density — the mug simulates at 0.94 kg and the bottle at 0.61 kg,
  which saturates the elbow servo. One attribute on each visual mesh geom fixes it.
- Spoon and fork start inside the drawer tray, lying flat, with **box** collision geoms (their
  mesh hulls sink through the 8 mm tray floor). Note MuJoCo re-orients mesh geoms to principal
  axes at compile time, so body quaternions must be derived from `geom_quat`, not guessed.
- Inactive welds for every arm/object pair, `connect` hooks for the drawer, a visual-only placemat.

## Recording demonstrations for ACT (runs in `.venv-learned`)

```bash
SKILL=pick ARM=B OBJECT=mug SEEDS=101,102 SPLIT=train .venv-learned/bin/python3 record_episodes.py
SKILL=open_drawer ARM=A SEEDS=101 SPLIT=train .venv-learned/bin/python3 record_episodes.py
SKILL=pick ARM=A OBJECT=fork PRE="open_drawer:A" SEEDS=101 .venv-learned/bin/python3 record_episodes.py
.venv/bin/python -m stage3_policy.learned.validate_dataset --root data/butler_demos/pick
```

One LeRobotDataset v3 per skill under `data/butler_demos/<skill>/`, written with lerobot's own
writer (images embedded), plus `meta/butler_episodes.json` with the per-episode metrics.
Seeds 0–9 are evaluation seeds and must not be used.

## Still open

- `execute()` in `bimanual.py` does not call these primitives yet (Stage 3 → Stage 4 contract).
- `pour` is not implemented; the bottle is decoration.
- Utensils yaw-align to the table edge within ~10°; carries can tilt them mid-air (they land flat).
- Only seed 101 has been run end-to-end; the 10-seed evaluation (`scripts/evaluate.py`) is next.
- Placement targets are hard-coded in the sequence; the planner's `planner_config.yaml` slots
  should replace them once Stage 3/4 agree the contract.
