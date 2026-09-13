"""Verify post-action scene state and request recovery when possible."""

from math import dist
from typing import Iterable, Mapping, Sequence

from common.types import ActionType, SceneState, Task, VerifyResult


POSITION_TOLERANCE_M = 0.01


def position_within_tolerance(
    actual: Sequence[float],
    expected: Sequence[float],
    tolerance_m: float = POSITION_TOLERANCE_M,
) -> bool:
    """Return whether two 3D positions are within the configured tolerance."""
    if len(actual) != len(expected):
        return False
    return dist(tuple(actual), tuple(expected)) <= tolerance_m


def _required_objects(task: Task) -> set[str]:
    required: set[str] = set()
    for step in task.steps:
        if step.object:
            required.add(step.object)
        if step.source:
            required.add(step.source)
        if step.into:
            required.add(step.into)
    return required


def _drawer_requirements(task: Task) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for step in task.steps:
        if step.action == ActionType.OPEN_DRAWER:
            requirements[step.target or "top_drawer"] = "open"
        elif step.action == ActionType.CLOSE_DRAWER:
            requirements[step.target or "top_drawer"] = "closed"
    return requirements


def _missing_objects(scene: SceneState, required: Iterable[str]) -> list[str]:
    return sorted(name for name in required if name not in scene.objects)


def verify(
    scene_after: SceneState,
    task: Task,
    expected_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_m: float = POSITION_TOLERANCE_M,
) -> VerifyResult:
    """Compare the observed scene with requirements implied by ``task``.

    Missing objects, incorrect drawer state, and position errors are treated as
    recoverable because the integration pipeline can re-perceive and retry.
    ``expected_positions`` is optional because the current shared Task contract
    does not carry destination coordinates.
    """
    if not task.steps:
        return VerifyResult(ok=True, details="no task steps to verify")

    failures: list[str] = []

    missing = _missing_objects(scene_after, _required_objects(task))
    if missing:
        failures.append(f"missing objects: {', '.join(missing)}")

    for drawer, expected_state in _drawer_requirements(task).items():
        actual_state = scene_after.drawers.get(drawer)
        if actual_state != expected_state:
            failures.append(
                f"{drawer} is {actual_state or 'missing'}, expected {expected_state}"
            )

    if expected_positions:
        for name, expected in expected_positions.items():
            actual = scene_after.objects.get(name)
            if actual is None:
                continue
            if not position_within_tolerance(actual, expected, tolerance_m):
                failures.append(f"{name} is outside the {tolerance_m:.3f} m tolerance")

    if failures:
        return VerifyResult(
            ok=False,
            replan=True,
            details="; ".join(failures),
        )

    return VerifyResult(ok=True, replan=False, details="scene satisfies task requirements")
