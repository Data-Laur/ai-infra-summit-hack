# stage4_bimanual — owner: Azeem
Executes planned Actions with two SO-101 arms in MuJoCo (drawer opening, hand-offs, coordinated moves). Currently a stub — no MuJoCo import.
Input: `list[Action]` (+ optional sim handle). Output: `ExecutionResult` with per-step results and final `SceneState`.
Test standalone:
`python3 -c "from common.types import Action, ActionType; from stage4_bimanual import execute; print(execute([Action(step_id=1, action=ActionType.PICK, arm='A', object='plate')]))"`
