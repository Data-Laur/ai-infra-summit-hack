"""Perception package for Part 2."""

from typing import Any

from common.types import SceneState


def perceive(image: Any | None = None) -> SceneState:
	"""Return the legacy planner-facing scene representation."""
	return SceneState(
		objects={
			"plate": (0.30, 0.00, 0.02),
			"mug": (0.20, 0.15, 0.05),
			"water_bottle": (0.40, -0.10, 0.10),
			"spoon": (0.25, -0.20, 0.01),
			"fork": (0.25, -0.25, 0.01),
		},
		drawers={"top_drawer": "closed"},
	)


__all__ = ["perceive"]
