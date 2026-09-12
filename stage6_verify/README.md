# stage6_verify — owner: Abdullah
Re-perceives after placement, compares the scene to the plan, and triggers replan/retry (stubbed: always ok).
Input: post-execution `SceneState` + original `Task`. Output: `VerifyResult`.
Run standalone: `python3 -c "from common.types import SceneState, Task; from stage6_verify import verify; print(verify(SceneState(objects={}, drawers={}), Task(command='x', steps=[])))"`
