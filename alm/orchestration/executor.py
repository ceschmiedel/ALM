"""L4 · DAG executor.

Runs the waves of an :class:`~alm.orchestration.dag.ExecutionDAG`: everything
independent goes concurrently, everything dependent waits for its predecessor's
structured output.

Three behaviours are worth stating explicitly, because they are the difference
between an orchestrator and a loop over a list:

* **Parallelism is bounded.** ``ALM_MAX_PARALLELISM`` caps concurrent expert
  calls, so a wide plan cannot saturate a shared GPU and turn a latency win into
  a queue.
* **A failed step does not fail the plan.** Its dependents are skipped with a
  recorded reason, independent branches continue, and arbitration receives a
  partial-but-attributed set of answers. A federation that collapses on one
  expert error is worse than the monolith it replaces.
* **Every step is timed and accounted.** Per-step latency and cost roll up into
  the metrics the evaluation harness compares against the baseline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from alm.config import Settings, get_settings
from alm.core.telemetry import Stopwatch, UsageMeter, alm_span
from alm.experts.registry import ExpertRegistry
from alm.orchestration.context import ExecutionContext
from alm.orchestration.dag import ExecutionDAG
from alm.protocol.answer import ExpertAnswer
from alm.protocol.task import SubTask, TaskEnvelope
from alm.protocol.trace import ExecutionTrace, Layer

logger = logging.getLogger(__name__)


class ExecutionReport:
    """Outcome of running one DAG."""

    __slots__ = ("answers", "context", "dag", "wall_ms", "skipped", "failed", "waves")

    def __init__(
        self,
        *,
        answers: list[ExpertAnswer],
        context: ExecutionContext,
        dag: ExecutionDAG,
        wall_ms: float,
        skipped: list[dict[str, Any]],
        failed: list[dict[str, Any]],
        waves: int,
    ) -> None:
        self.answers = answers
        self.context = context
        self.dag = dag
        self.wall_ms = wall_ms
        self.skipped = skipped
        self.failed = failed
        self.waves = waves

    @property
    def successful(self) -> list[ExpertAnswer]:
        return [a for a in self.answers if a.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "waves": self.waves,
            "wall_ms": round(self.wall_ms, 3),
            "answers": len(self.answers),
            "successful": len(self.successful),
            "failed": self.failed,
            "skipped": self.skipped,
            "topology": self.dag.to_dict(),
        }


#: Signature of the handler used when a subtask has no matched expert.
FallbackHandler = Any  # Callable[[SubTask, TaskEnvelope, dict], Awaitable[ExpertAnswer]]


class DAGExecutor:
    """Executes a plan wave by wave."""

    def __init__(
        self,
        experts: ExpertRegistry,
        *,
        settings: Settings | None = None,
        fallback_handler: FallbackHandler | None = None,
    ) -> None:
        self.experts = experts
        self.settings = settings or get_settings()
        #: Invoked for subtasks the router could not assign to an expert.
        self.fallback_handler = fallback_handler

    async def execute(
        self,
        dag: ExecutionDAG,
        task: TaskEnvelope,
        *,
        trace: ExecutionTrace | None = None,
        meter: UsageMeter | None = None,
        context: ExecutionContext | None = None,
    ) -> ExecutionReport:
        """Run the graph and return every answer, successful or not."""
        execution_context = context or ExecutionContext(
            task.session_id, tenant_id=task.tenant_id
        )
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallelism))
        stopwatch = Stopwatch()

        answers: list[ExpertAnswer] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        levels = dag.levels()

        if trace is not None:
            trace.emit(
                Layer.L4_ORCHESTRATION,
                (
                    f"built execution DAG: {len(dag)} step(s) in {dag.depth} wave(s), "
                    f"max parallelism {dag.width}"
                ),
                category="plan",
                detail=dag.to_dict(),
            )

        for wave_index, level in enumerate(levels, start=1):
            runnable: list[SubTask] = []
            for subtask_id in level:
                subtask = dag.nodes[subtask_id]
                missing = execution_context.missing_dependencies(subtask)
                if missing:
                    reason = (
                        f"skipped: {len(missing)} upstream step(s) produced no usable output"
                    )
                    skipped.append(
                        {
                            "subtask_id": subtask.subtask_id,
                            "expert_id": subtask.expert_id,
                            "missing": missing,
                            "reason": reason,
                        }
                    )
                    answer = ExpertAnswer(
                        subtask_id=subtask.subtask_id,
                        expert_id=subtask.expert_id,
                        domain=subtask.domain,
                        capability_id=subtask.capability_id,
                        status="skipped",
                        error=reason,
                    )
                    answers.append(answer)
                    if trace is not None:
                        trace.emit(
                            Layer.L4_ORCHESTRATION,
                            f"{subtask.expert_id or 'orchestrator'} {reason}",
                            category="warning",
                            subtask_id=subtask.subtask_id,
                            expert_id=subtask.expert_id,
                            detail={"missing": missing},
                        )
                    continue
                runnable.append(subtask)

            if not runnable:
                continue

            if trace is not None and len(runnable) > 1:
                trace.emit(
                    Layer.L4_ORCHESTRATION,
                    (
                        f"wave {wave_index}/{len(levels)}: running "
                        f"{len(runnable)} expert(s) in parallel"
                    ),
                    category="wave",
                    detail={"experts": [st.expert_id or "orchestrator" for st in runnable]},
                )

            results = await asyncio.gather(
                *(
                    self._run_step(
                        subtask,
                        task,
                        execution_context,
                        semaphore,
                        trace=trace,
                        meter=meter,
                    )
                    for subtask in runnable
                ),
                return_exceptions=False,
            )

            for answer in results:
                answers.append(answer)
                execution_context.record(answer)
                if not answer.ok:
                    failed.append(
                        {
                            "subtask_id": answer.subtask_id,
                            "expert_id": answer.expert_id,
                            "status": answer.status,
                            "error": answer.error,
                        }
                    )

        wall_ms = stopwatch.stop()
        if trace is not None:
            successful = len([a for a in answers if a.ok])
            trace.emit(
                Layer.L4_ORCHESTRATION,
                f"execution complete: {successful}/{len(answers)} step(s) succeeded",
                category="complete",
                duration_ms=wall_ms,
                detail={"failed": len(failed), "skipped": len(skipped)},
            )

        return ExecutionReport(
            answers=answers,
            context=execution_context,
            dag=dag,
            wall_ms=wall_ms,
            skipped=skipped,
            failed=failed,
            waves=len(levels),
        )

    # -- single step -------------------------------------------------------

    async def _run_step(
        self,
        subtask: SubTask,
        task: TaskEnvelope,
        context: ExecutionContext,
        semaphore: asyncio.Semaphore,
        *,
        trace: ExecutionTrace | None,
        meter: UsageMeter | None,
    ) -> ExpertAnswer:
        upstream = context.inputs_for(subtask)

        async with semaphore:
            with alm_span(
                f"alm.expert.{subtask.expert_id or 'fallback'}",
                attributes={
                    "session_id": task.session_id,
                    "subtask_id": subtask.subtask_id,
                    "domain": subtask.domain,
                },
            ):
                stopwatch = Stopwatch()
                if trace is not None:
                    trace.emit(
                        Layer.L3_EXPERTS,
                        f"{subtask.expert_id or 'orchestrator'} starting: "
                        f"{subtask.description[:100]}",
                        category="start",
                        subtask_id=subtask.subtask_id,
                        expert_id=subtask.expert_id,
                    )

                answer = await self._invoke(subtask, task, upstream, meter)
                elapsed = stopwatch.stop()

                if trace is not None:
                    if answer.ok:
                        trace.emit(
                            Layer.L3_EXPERTS,
                            (
                                f"{answer.expert_id or 'orchestrator'} answered "
                                f"(confidence {answer.confidence:.2f}"
                                f"{'' if answer.calibrated else ', uncalibrated'}, "
                                f"{len(answer.citations)} citation(s))"
                            ),
                            category="answer",
                            subtask_id=subtask.subtask_id,
                            expert_id=answer.expert_id,
                            model_id=answer.model_id,
                            duration_ms=elapsed,
                            detail={
                                "confidence": answer.confidence,
                                "raw_confidence": answer.raw_confidence,
                                "calibrated": answer.calibrated,
                                "cost_usd": answer.cost_usd,
                                "tokens": answer.prompt_tokens + answer.completion_tokens,
                                "summary": answer.summary(160),
                            },
                        )
                    else:
                        trace.emit(
                            Layer.L3_EXPERTS,
                            f"{answer.expert_id or 'orchestrator'} failed: {answer.error}",
                            category="error",
                            subtask_id=subtask.subtask_id,
                            expert_id=answer.expert_id,
                            duration_ms=elapsed,
                            detail={"status": answer.status},
                        )
                return answer

    async def _invoke(
        self,
        subtask: SubTask,
        task: TaskEnvelope,
        upstream: dict[str, Any],
        meter: UsageMeter | None,
    ) -> ExpertAnswer:
        """Dispatch one subtask, with timeout and retry."""
        if subtask.is_fallback or not subtask.expert_id:
            return await self._run_fallback(subtask, task, upstream, meter)

        agent = self.experts.get(subtask.expert_id)
        if agent is None:
            logger.warning(
                "Plan references unregistered expert %r; escalating that step",
                subtask.expert_id,
            )
            return await self._run_fallback(subtask, task, upstream, meter)

        attempts = max(1, self.settings.step_max_retries + 1)
        last_error = ""
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    agent.execute(subtask, task, upstream=upstream, meter=meter),
                    timeout=self.settings.step_timeout_seconds,
                )
            except TimeoutError:
                last_error = (
                    f"expert timed out after {self.settings.step_timeout_seconds:.0f}s"
                )
                logger.warning("Expert %s timed out (attempt %d)", subtask.expert_id, attempt + 1)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Expert %s raised on attempt %d: %s", subtask.expert_id, attempt + 1, exc
                )

        return ExpertAnswer(
            subtask_id=subtask.subtask_id,
            expert_id=subtask.expert_id,
            domain=subtask.domain,
            capability_id=subtask.capability_id,
            status="error",
            error=last_error or "expert failed",
        )

    async def _run_fallback(
        self,
        subtask: SubTask,
        task: TaskEnvelope,
        upstream: dict[str, Any],
        meter: UsageMeter | None,
    ) -> ExpertAnswer:
        if self.fallback_handler is None:
            return ExpertAnswer(
                subtask_id=subtask.subtask_id,
                expert_id="",
                domain=subtask.domain,
                status="error",
                is_fallback=True,
                error=(
                    "this step needs the orchestrator, but no orchestrator model is "
                    "registered — add one with `alm model add --tier orchestrator`"
                ),
            )
        answer = await self.fallback_handler(subtask, task, upstream, meter)
        answer.is_fallback = True
        return answer
