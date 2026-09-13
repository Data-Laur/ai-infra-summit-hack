"""UNIT INTEGRATION tests for common.pipeline.run_once recovery logic.

Every stage is replaced by a controlled double, so these tests show how run_once
wires stage outputs together. They are not physical validation and say nothing
about whether the simulated robot can do the task.
"""

import sys
import types

import pytest

from common.pipeline import run_once
from common.types import Action, ActionType, ExecutionResult, SceneState, Task, TaskStep, VerifyResult

TASK = Task(command="pick mug", steps=[TaskStep(id=1, action=ActionType.PICK, arm="B", object="mug")])


class DoublePlanningError(Exception):
    """Stands in for stage3_policy.PlanningError."""


def _scene(x: float) -> SceneState:
    return SceneState(objects={"mug": (x, 0.18, 0.70)}, drawers={"top_drawer": "open"})


@pytest.fixture
def install_stages(monkeypatch):
    """Replace every stage module with scripted doubles; returns a record of calls."""

    def install(*, observations, execution_success=(), verdicts=(), plan_error=None):
        observations, execution_success, verdicts = list(observations), list(execution_success), list(verdicts)
        calls = {"plan_scenes": [], "executions": 0}

        def plan(task, scene):
            calls["plan_scenes"].append(scene)
            if plan_error is not None:
                raise plan_error
            return [Action(step_id=1, action=ActionType.PICK, arm="B", object="mug", target_pose=scene.objects["mug"])]

        def execute(actions, sim=None):
            calls["executions"] += 1
            ok = execution_success.pop(0)
            return ExecutionResult(
                action_results={a.step_id: ok for a in actions},
                success=ok,
                final_scene=_scene(0.0),
                error=None if ok else "double: grasp slipped",
            )

        doubles = {
            "stage1_voice": {"parse_text": lambda text: TASK},
            "stage2_perception": {"perceive": lambda image=None: observations.pop(0)},
            "stage3_policy": {"plan": plan, "PlanningError": DoublePlanningError},
            "stage4_bimanual": {"execute": execute, "get_camera_frame": lambda sim: None, "reset_scene": lambda seed: object()},
            "stage6_verify": {"verify": lambda scene_after, task: verdicts.pop(0)},
        }
        for name, attributes in doubles.items():
            module = types.ModuleType(name)
            for attribute, value in attributes.items():
                setattr(module, attribute, value)
            monkeypatch.setitem(sys.modules, name, module)
        return calls

    return install


def test_retry_plans_from_the_post_execution_observation(install_stages):
    initial, after_first, after_second = _scene(0.06), _scene(0.10), _scene(0.10)
    calls = install_stages(
        observations=[initial, after_first, after_second],
        execution_success=[True, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
    )
    result = run_once("pick mug", max_retries=2)

    assert result.success and result.attempts == 2
    assert calls["plan_scenes"] == [initial, after_first]


def test_failed_execution_is_not_success_even_if_verify_accepts(install_stages):
    install_stages(observations=[_scene(0.06), _scene(0.06)], execution_success=[False], verdicts=[VerifyResult(ok=True)])
    result = run_once("pick mug", max_retries=2)

    assert not result.success and result.attempts == 1
    assert any("not counted as success" in line for line in result.log)


def test_failed_execution_can_recover_through_a_replan(install_stages):
    install_stages(
        observations=[_scene(0.06), _scene(0.07), _scene(0.07)],
        execution_success=[False, True],
        verdicts=[VerifyResult(ok=False, replan=True), VerifyResult(ok=True)],
    )
    result = run_once("pick mug", max_retries=2)
    assert result.success and result.attempts == 2


def test_planning_refusal_reports_fail_without_executing(install_stages):
    calls = install_stages(observations=[_scene(0.06)], plan_error=DoublePlanningError("'mug' is not in the observed scene"))
    result = run_once("pick mug", max_retries=2)

    assert not result.success and result.attempts == 1
    assert calls["executions"] == 0
    assert any("planning refused" in line and "not in the observed scene" in line for line in result.log)


def test_retries_stop_at_max_retries(install_stages):
    calls = install_stages(
        observations=[_scene(0.06), _scene(0.06), _scene(0.06), _scene(0.06)],
        execution_success=[True, True, True],
        verdicts=[VerifyResult(ok=False, replan=True)] * 3,
    )
    result = run_once("pick mug", max_retries=3)

    assert not result.success and result.attempts == 3
    assert calls["executions"] == 3
