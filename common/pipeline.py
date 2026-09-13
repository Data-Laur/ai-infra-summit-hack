"""Single-run pipeline logic, shared by scripts/run_pipeline.py and stage7_eval.

Recovery: if verify() returns replan=True, re-run plan -> execute -> verify
up to max_retries (from configs/default.yaml), then report FAIL. Each retry
plans from the scene observed after the previous attempt, and a run succeeds
only when execution reports success AND verify accepts the scene.
"""

from pathlib import Path

import yaml

from common.types import RunResult

_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK_MAX_RETRIES = 2


def _config_max_retries() -> int:
    try:
        cfg = yaml.safe_load((_ROOT / "configs" / "default.yaml").read_text())
        return int(cfg["max_retries"])
    except Exception:
        return _FALLBACK_MAX_RETRIES


def run_once(command: str, seed: int = 0, max_retries: int | None = None) -> RunResult:
    """Run the full pipeline once for a seed, with the verify->replan recovery loop."""
    from stage1_voice import parse_text
    from stage2_perception import perceive
    from stage3_policy import PlanningError, plan
    from stage4_bimanual import execute, get_camera_frame, reset_scene
    from stage6_verify import verify

    if max_retries is None:
        max_retries = _config_max_retries()

    log: list[str] = []

    log.append(f"[sim]      stage4_bimanual.reset_scene(seed={seed})")
    sim = reset_scene(seed)

    log.append(f'[voice]    stage1_voice.parse_text("{command}")')
    task = parse_text(command)
    log.append(f"           -> Task with {len(task.steps)} steps:")
    for step in task.steps:
        deps = f" (after {step.depends_on})" if step.depends_on else ""
        log.append(f"              {step.id}. {step.action.value:<12} arm {step.arm}{deps}")

    log.append("[perceive] stage2_perception.perceive(get_camera_frame(sim))")
    scene = perceive(get_camera_frame(sim))
    log.append(f"           -> {len(scene.objects)} objects: {', '.join(scene.objects)}")
    log.append(f"           -> drawers: {dict(scene.drawers)}")

    attempts = 0
    success = False
    while True:
        attempts += 1
        log.append(f"[plan]     stage3_policy.plan(task, scene)  (attempt {attempts})")
        try:
            actions = plan(task, scene)
        except PlanningError as exc:
            log.append(f"           -> planning refused ({type(exc).__name__}): {exc}")
            break
        log.append(f"           -> {len(actions)} executable actions")

        log.append("[execute]  stage4_bimanual.execute(actions, sim)")
        result = execute(actions, sim)
        ok = sum(result.action_results.values())
        log.append(f"           -> {ok}/{len(actions)} actions succeeded")
        if result.error:
            log.append(f"           -> error: {result.error}")

        log.append("[verify]   stage6_verify.verify(perceive(get_camera_frame(sim)), task)")
        scene_after = perceive(get_camera_frame(sim))
        verdict = verify(scene_after, task)
        log.append(f"           -> ok={verdict.ok} replan={verdict.replan} ({verdict.details})")

        if verdict.ok and result.success:
            success = True
            break
        if verdict.ok:
            log.append("           -> execution reported failure, so verify ok is not counted as success")
        if verdict.replan and attempts < max_retries:
            log.append(f"           -> replan requested: retrying from the new observation ({attempts + 1}/{max_retries})")
            scene = scene_after
            continue
        break

    return RunResult(success=success, attempts=attempts, task=task, log=log)
