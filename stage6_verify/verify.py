"""Stage 6: verify scene after execution against the task intent."""

# TODO(Abdullah): replace stub with real implementation

from common.types import SceneState, Task, VerifyResult


def verify(scene_after: SceneState, task: Task) -> VerifyResult:
    """Re-perceive after placement and compare scene to the plan (stub: always ok)."""
    return VerifyResult(
        ok=True,
        replan=False,
        details="stub: real scene-vs-plan comparison pending",
    )
