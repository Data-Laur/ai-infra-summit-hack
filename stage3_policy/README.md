# stage3_policy — Owner: Bidipta + Azeem
Turns a parsed `Task` + `SceneState` into an ordered `list[Action]` (rule-based stub; later LeRobot VLA with rule fallback).
Input: `Task`, `SceneState` from `common.types`. Output: one `Action` per step, order preserved.
Run standalone: `python3 -c "from common.types import Task, SceneState; from stage3_policy import plan; print(plan(Task(command='x', steps=[]), SceneState(objects={}, drawers={})))"`
