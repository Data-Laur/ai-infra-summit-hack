"""Public Stage 2 perception API."""

from typing import Any

from common.types import SceneState

from .scene_pipeline import ScenePipeline


IMAGE_CORNERS = [(80.0, 60.0), (560.0, 60.0), (560.0, 420.0), (80.0, 420.0)]
TABLE_CORNERS = [(0.0, 0.0), (0.8, 0.0), (0.8, 0.6), (0.0, 0.6)]


def perceive(image: Any | None = None) -> SceneState:
	"""Detect scene objects from a camera image for downstream planning.

	The default calibration matches the 640x480 tabletop validation camera. A
	marker-free image returns an empty scene rather than synthetic positions.
	"""
	if image is None:
		raise ValueError("A camera image is required for real Stage 2 perception.")

	pipeline = ScenePipeline()
	pipeline.compute_homography_from_points(IMAGE_CORNERS, TABLE_CORNERS)
	detected = pipeline.detect_from_image(image)

	objects = {
		item.name: (item.x, item.y, item.z)
		for item in detected.objects
	}
	drawers = {
		"top_drawer": "open" if detected.drawer.open_fraction > 0.0 else "closed"
	}
	return SceneState(objects=objects, drawers=drawers)


__all__ = ["perceive"]
