"""Stage 1: natural-language command -> structured Task (stub)."""

# TODO(Alex): replace stub with real implementation (Speechmatics ASR -> Claude parser)

from common.types import ActionType, HoldRequirement, Task, TaskStep


def parse_text(text: str) -> Task:
    """Parse a natural-language command into a Task. Stub: returns a fixed example."""
    return Task(
        command=text,
        steps=[
            TaskStep(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            TaskStep(id=2, action=ActionType.PICK, arm="A", object="plate", depends_on=[1]),
            TaskStep(
                id=3,
                action=ActionType.PLACE,
                arm="A",
                object="plate",
                destination="table",
                depends_on=[2],
            ),
            TaskStep(id=4, action=ActionType.PICK, arm="B", object="mug"),
            TaskStep(
                id=5,
                action=ActionType.POUR,
                arm="A",
                source="water_bottle",
                into="mug",
                depends_on=[4],
                requires_hold=HoldRequirement(arm="B", object="mug"),
            ),
        ],
        constraints=["keep_glasses_away_from_edge"],
    )


def parse_command(audio_path: str) -> Task:
    """Transcribe an audio file and parse it into a Task. Not implemented yet."""
    raise NotImplementedError("Speechmatics integration pending")
