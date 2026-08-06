"""L4 · DAG construction, parallelism and failure containment."""

from __future__ import annotations

import asyncio

import pytest

from alm.config import Settings
from alm.core.errors import CyclicPlanError, OrchestrationError
from alm.experts.registry import ExpertRegistry
from alm.orchestration.context import ExecutionContext
from alm.orchestration.dag import build_dag
from alm.orchestration.executor import DAGExecutor
from alm.protocol.answer import ExpertAnswer
from alm.protocol.task import ExecutionPlan, SubTask, TaskEnvelope


def _plan(*subtasks: SubTask) -> ExecutionPlan:
    return ExecutionPlan(subtasks=list(subtasks), session_id="ses-dag")


# -- DAG construction --------------------------------------------------------


def test_independent_subtasks_form_one_parallel_wave():
    a = SubTask(description="a", expert_id="x")
    b = SubTask(description="b", expert_id="y")
    dag = build_dag(_plan(a, b))

    assert dag.depth == 1
    assert dag.width == 2
    assert len(dag.levels()[0]) == 2


def test_chained_subtasks_form_sequential_waves():
    a = SubTask(description="a", expert_id="x")
    b = SubTask(description="b", expert_id="y", depends_on=[a.subtask_id])
    c = SubTask(description="c", expert_id="z", depends_on=[b.subtask_id])
    dag = build_dag(_plan(a, b, c))

    assert dag.depth == 3
    assert dag.width == 1
    assert dag.levels() == [[a.subtask_id], [b.subtask_id], [c.subtask_id]]


def test_join_point_waits_for_both_branches():
    a = SubTask(description="a", expert_id="x")
    b = SubTask(description="b", expert_id="y")
    join = SubTask(description="join", expert_id="z", depends_on=[a.subtask_id, b.subtask_id])
    dag = build_dag(_plan(a, b, join))

    assert dag.depth == 2
    assert dag.width == 2
    assert dag.levels()[1] == [join.subtask_id]
    assert [t.subtask_id for t in dag.terminal_nodes()] == [join.subtask_id]


def test_empty_plan_is_rejected():
    with pytest.raises(OrchestrationError):
        build_dag(_plan())


def test_self_dependency_is_rejected():
    a = SubTask(description="a", expert_id="x")
    a.depends_on = [a.subtask_id]
    with pytest.raises(CyclicPlanError):
        build_dag(_plan(a))


def test_cycle_is_detected_and_reported():
    a = SubTask(description="a", expert_id="x")
    b = SubTask(description="b", expert_id="y", depends_on=[a.subtask_id])
    a.depends_on = [b.subtask_id]
    with pytest.raises(CyclicPlanError, match="cycle"):
        build_dag(_plan(a, b))


def test_dangling_dependencies_are_ignored_not_fatal():
    a = SubTask(description="a", expert_id="x", depends_on=["ghost"])
    dag = build_dag(_plan(a))
    assert dag.depth == 1


def test_dag_renders_waves():
    a = SubTask(description="a", expert_id="alpha")
    b = SubTask(description="b", expert_id="beta")
    rendering = build_dag(_plan(a, b)).render()
    assert "wave 1" in rendering
    assert "parallel" in rendering


# -- execution ---------------------------------------------------------------


class _StubAgent:
    def __init__(self, expert_id: str, *, delay: float = 0.0, fail: bool = False) -> None:
        self.spec = type(
            "Spec",
            (),
            {"id": expert_id, "authority": {}, "verifier_rules": []},
        )()
        self.id = expert_id
        self.domain = "test"
        self.delay = delay
        self.fail = fail
        self.seen_upstream: dict | None = None

    async def execute(self, subtask, task, *, upstream=None, meter=None):  # noqa: ANN001
        self.seen_upstream = upstream
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("expert exploded")
        return ExpertAnswer(
            subtask_id=subtask.subtask_id,
            expert_id=self.id,
            domain=self.domain,
            content=f"answer from {self.id}",
            confidence=0.8,
        )

    def verify(self, answer):  # noqa: ANN001
        from alm.experts.verifier import VerificationResult

        return VerificationResult(expert_id=self.id, answer_id=answer.answer_id, passed=True)


class _StubRegistry(ExpertRegistry):
    def __init__(self, agents: dict[str, _StubAgent]) -> None:  # noqa: D107
        self._agents = agents  # type: ignore[assignment]

    def get(self, expert_id: str):  # noqa: ANN001
        return self._agents.get(expert_id)


def _executor(agents: dict[str, _StubAgent], **settings_overrides) -> DAGExecutor:
    settings = Settings(**settings_overrides)
    return DAGExecutor(_StubRegistry(agents), settings=settings)


async def test_independent_steps_run_concurrently():
    agents = {
        "a": _StubAgent("a", delay=0.15),
        "b": _StubAgent("b", delay=0.15),
    }
    a = SubTask(description="a", expert_id="a")
    b = SubTask(description="b", expert_id="b")
    dag = build_dag(_plan(a, b))

    loop = asyncio.get_running_loop()
    started = loop.time()
    report = await _executor(agents).execute(dag, TaskEnvelope(intent_text="q"))
    elapsed = loop.time() - started

    assert len(report.successful) == 2
    # Serial execution would take ~0.30s; concurrent should be close to 0.15s.
    assert elapsed < 0.28


async def test_parallelism_is_bounded_by_settings():
    agents = {name: _StubAgent(name, delay=0.1) for name in "abcd"}
    subtasks = [SubTask(description=n, expert_id=n) for n in "abcd"]
    dag = build_dag(_plan(*subtasks))

    loop = asyncio.get_running_loop()
    started = loop.time()
    await _executor(agents, max_parallelism=2).execute(dag, TaskEnvelope(intent_text="q"))
    elapsed = loop.time() - started

    # Four 0.1s steps at concurrency 2 must take at least two batches.
    assert elapsed >= 0.18


async def test_downstream_receives_structured_upstream_context():
    agents = {"a": _StubAgent("a"), "b": _StubAgent("b")}
    a = SubTask(description="a", expert_id="a")
    b = SubTask(description="b", expert_id="b", depends_on=[a.subtask_id])
    dag = build_dag(_plan(a, b))

    await _executor(agents).execute(dag, TaskEnvelope(intent_text="q"))

    upstream = agents["b"].seen_upstream
    assert upstream is not None
    assert upstream["upstream"][0]["expert_id"] == "a"
    assert "answer from a" in upstream["upstream"][0]["content"]


async def test_failure_skips_dependents_but_not_siblings():
    agents = {
        "broken": _StubAgent("broken", fail=True),
        "dependent": _StubAgent("dependent"),
        "sibling": _StubAgent("sibling"),
    }
    broken = SubTask(description="broken", expert_id="broken")
    dependent = SubTask(
        description="dependent", expert_id="dependent", depends_on=[broken.subtask_id]
    )
    sibling = SubTask(description="sibling", expert_id="sibling")
    dag = build_dag(_plan(broken, dependent, sibling))

    report = await _executor(agents, step_max_retries=0).execute(
        dag, TaskEnvelope(intent_text="q")
    )

    statuses = {a.expert_id: a.status for a in report.answers}
    assert statuses["broken"] == "error"
    assert statuses["dependent"] == "skipped"
    assert statuses["sibling"] == "success", "an independent branch must still run"
    assert report.skipped and report.failed


async def test_unregistered_expert_falls_back_rather_than_failing():
    called = {}

    async def fallback(subtask, task, upstream, meter):  # noqa: ANN001
        called["hit"] = True
        return ExpertAnswer(
            subtask_id=subtask.subtask_id,
            expert_id="orchestrator",
            content="handled by the orchestrator",
        )

    executor = DAGExecutor(
        _StubRegistry({}), settings=Settings(), fallback_handler=fallback
    )
    subtask = SubTask(description="a", expert_id="missing-expert")
    report = await executor.execute(build_dag(_plan(subtask)), TaskEnvelope(intent_text="q"))

    assert called.get("hit") is True
    assert report.answers[0].is_fallback


async def test_missing_fallback_handler_reports_a_useful_error():
    executor = DAGExecutor(_StubRegistry({}), settings=Settings())
    subtask = SubTask(description="a", expert_id="", is_fallback=True)
    report = await executor.execute(build_dag(_plan(subtask)), TaskEnvelope(intent_text="q"))

    assert report.answers[0].status == "error"
    assert "orchestrator model" in report.answers[0].error


async def test_timeout_is_enforced():
    agents = {"slow": _StubAgent("slow", delay=1.0)}
    subtask = SubTask(description="slow", expert_id="slow")
    report = await _executor(
        agents, step_timeout_seconds=0.05, step_max_retries=0
    ).execute(build_dag(_plan(subtask)), TaskEnvelope(intent_text="q"))

    assert report.answers[0].status == "error"
    assert "timed out" in report.answers[0].error


# -- execution context -------------------------------------------------------


def test_context_only_passes_declared_dependencies():
    context = ExecutionContext("ses-ctx")
    a = SubTask(description="a", expert_id="a")
    b = SubTask(description="b", expert_id="b")
    consumer = SubTask(description="c", expert_id="c", depends_on=[a.subtask_id])

    context.record(
        ExpertAnswer(subtask_id=a.subtask_id, expert_id="a", content="from a", confidence=0.9)
    )
    context.record(
        ExpertAnswer(subtask_id=b.subtask_id, expert_id="b", content="from b", confidence=0.9)
    )

    payload = context.inputs_for(consumer)
    experts = [item["expert_id"] for item in payload["upstream"]]
    assert experts == ["a"], "an expert must not see what it did not depend on"


def test_context_reports_missing_dependencies():
    context = ExecutionContext("ses-missing")
    upstream = SubTask(description="a", expert_id="a")
    consumer = SubTask(description="b", expert_id="b", depends_on=[upstream.subtask_id])

    assert context.missing_dependencies(consumer) == [upstream.subtask_id]

    context.record(
        ExpertAnswer(subtask_id=upstream.subtask_id, expert_id="a", status="error")
    )
    assert context.missing_dependencies(consumer) == [upstream.subtask_id]
