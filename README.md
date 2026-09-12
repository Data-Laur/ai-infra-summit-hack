# ai-infra-summit-hack

Two simulated SO-101 arms set a dinner table from a natural-language voice
command — Intel Physical AI Online Challenge (lablab.ai AI Infra Summit
Hackathon, Sept 10–16, 2026).

## Overview

Voice command in, table set. Example command:

> "Open the top drawer, pick up the plate with arm A, place it on the table,
> pick up the mug with arm B, pour water into the mug with arm A."

**Voice input is powered by Speechmatics** (ASR) with Claude for instruction
parsing — see `stage1_voice/`.

TODO: 2–3 sentence pitch once the demo works.

## Architecture

```
voice (Speechmatics + Claude)          stage1_voice      -> Task
  -> perception (OpenCV)               stage2_perception -> SceneState
  -> policy (LeRobot VLA + fallback)   stage3_policy     -> list[Action]
  -> bimanual execution (MuJoCo)       stage4_bimanual   -> ExecutionResult
  -> verify / recover                  stage6_verify     -> VerifyResult
  -> evaluation over seeds             stage7_eval       -> EvalReport

standalone: OpenVINO benchmark         stage5_openvino
sim helpers (MuJoCo):                  stage4_bimanual.reset_scene / get_camera_frame
```

All inter-stage types are pydantic models in [`common/types.py`](common/types.py).
Signatures are pinned in [`CONTRACTS.md`](CONTRACTS.md).

## Simulation Environment (MuJoCo)

TODO(Azeem): how to load the dual SO-101 dinner-table scene from assets/,
which SO-101 model files are used (source: TheRobotStudio/SO-ARM100), and
how configs/default.yaml drives environment randomization via
stage4_bimanual.reset_scene(seed).

All 3D assets in assets/ are free / openly licensed; sources and licenses
are listed in assets/README.md.

## Setup & Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
pip install pydantic pyyaml        # enough to run the stub pipeline today
pip install -r requirements.txt    # full stack, needed for real implementations
```

## How to Reproduce the Demo

```bash
python scripts/run_pipeline.py                     # full pipeline, default command
python scripts/run_pipeline.py --command "..."     # your own command
python scripts/run_pipeline.py --seed 3            # randomized scene for a given seed
python scripts/evaluate.py --seeds 10              # success rate over 10 seeds
python scripts/benchmark.py                        # OpenVINO benchmark table
```

TODO: exact reproduction steps for the submitted demo video.

## VLA / Policy Choice

TODO(Bidipta/Azeem): which VLA we fine-tuned via LeRobot, why, and how the
rule-based fallback decides when to take over.

## Bimanual Coordination Strategy

TODO(Azeem): dependency-ordered execution (`TaskStep.depends_on`),
complementary actions via `requires_hold` (one arm holds, the other acts),
hand-off protocol between arms.

## Training Approach

TODO(Azeem collects, Bidipta trains): dataset collection in sim, training
config, checkpoints.

## Robustness & Randomization

TODO(Abdullah): randomization ranges (see `configs/default.yaml`), verify →
replan/retry loop from stage6.

## OpenVINO Optimization & Intel Hardware Mapping

TODO(Lauren): conversion path (PyTorch → OpenVINO IR), precision (FP16/INT8),
which parts run on Intel Core Ultra CPU/GPU/NPU.

## Benchmark Results

TODO(Lauren): table from `python scripts/benchmark.py` on Intel hardware.

## Evaluation Results (success rate over 10 seeds)

TODO(Abdullah): output of `python scripts/evaluate.py --seeds 10`.

## Demo Video

TODO: link (file or upload lives in `docs/`).

The video shows, per the challenge requirements:

- the natural-language command and the randomized initial scene, for each of 10 seeds
- both SO-101 arms executing the task, including at least one hand-off / complementary action
- the final table state and a summarized success rate over the 10 seeds
- the OpenVINO benchmark running on Intel hardware with before/after numbers

## Team

| Who | Owns |
|-----|------|
| Alex | stage1_voice, integration & submission (scripts/, contracts) |
| Lauren | stage2_perception, stage5_openvino |
| Bidipta | stage3_policy (with Azeem) |
| Azeem | stage3_policy, stage4_bimanual |
| Abdullah | stage6_verify, stage7_eval |
| Everyone | docs/ (demo video & slides) |

## Working in this repo

- One branch per stage (`stage1-voice`, `stage3-policy`, ...). PRs into `main`
  are merged by the integration owner (Alex).
- Keep your code inside your own `stageN_*` folder.
- Don't change another stage's interface or a shared model in
  `common/types.py` without agreeing it in [`CONTRACTS.md`](CONTRACTS.md) first.
- `main` must always pass `python scripts/run_pipeline.py` — if your stage
  isn't ready, its stub stays in place.
