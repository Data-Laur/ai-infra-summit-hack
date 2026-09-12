"""Stage 3: rule-based policy — Task + SceneState -> ordered executable Actions."""

# TODO(Bidipta+Azeem): replace stub with real implementation (LeRobot VLA fine-tune with rule-based fallback)

from common.types import Action, SceneState, Task


def plan(task: Task, scene: SceneState) -> list[Action]:
    """Map each task step to one executable Action, preserving step order."""
    actions: list[Action] = []
    for step in task.steps:
        obj = step.object or step.target or step.source
        pose = scene.objects.get(obj, (0.0, 0.0, 0.0)) if obj else (0.0, 0.0, 0.0)
        actions.append(
            Action(
                step_id=step.id,
                action=step.action,
                arm=step.arm,
                object=obj,
                target_pose=pose,
            )
        )
    return actions
