"""Stage 3 policy package: plan(task, scene) -> list[Action].

PlanningError is exported so callers can report planning failures without
catching unrelated exceptions.
"""

from stage3_policy.errors import PlanningError
from stage3_policy.policy import plan

__all__ = ["plan", "PlanningError"]
