"""The evaluation harness — the controlled experiment.

> The ALM thesis is falsifiable, and should be treated as such. Without an
> evaluation harness, ALM is faith; with one, it is engineering.

The harness runs the same battery of tasks in two configurations — the
monolithic LLM with a good prompt and RAG, and the federation — and compares
them across all six metrics at once.  It is willing to return a verdict against
the federation, which is the only thing that makes a verdict *for* it worth
anything.

It also closes a loop that nothing else can: the labelled outcomes it produces
are exactly the data confidence calibration needs, so
:meth:`EvaluationHarness.fit_calibrations` turns an evaluation run into
better-behaved arbitration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from alm.core.ids import new_id
from alm.core.telemetry import UsageMeter
from alm.evaluation.baseline import MonolithicBaseline
from alm.evaluation.dataset import EvalCase, EvalDataset, load_datasets
from alm.evaluation.graders import grade
from alm.evaluation.metrics import (
    ArmMetrics,
    CaseResult,
    Comparison,
    aggregate,
    compare,
)
from alm.experts.calibration import CalibrationStore, fit_calibration
from alm.federation.runtime import FederationRuntime
from alm.persistence.database import session_scope
from alm.persistence.models import EvalCaseRow, EvalRunRow
from alm.protocol.task import TaskEnvelope

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, CaseResult], None]


class EvaluationRun:
    """One arm's results, aggregated and persisted."""

    __slots__ = ("run_id", "arm", "dataset", "results", "metrics", "domain")

    def __init__(
        self,
        *,
        run_id: str,
        arm: str,
        dataset: str,
        domain: str,
        results: list[CaseResult],
        metrics: ArmMetrics,
    ) -> None:
        self.run_id = run_id
        self.arm = arm
        self.dataset = dataset
        self.domain = domain
        self.results = results
        self.metrics = metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "dataset": self.dataset,
            "domain": self.domain,
            "metrics": self.metrics.model_dump(mode="json"),
            "cases": [r.model_dump(mode="json") for r in self.results],
        }


class EvaluationHarness:
    """Runs the federation and the baseline over the same dataset."""

    def __init__(
        self,
        runtime: FederationRuntime,
        *,
        pack_name: str = "",
    ) -> None:
        self.runtime = runtime
        self.pack_name = pack_name
        self.baseline = MonolithicBaseline(runtime.serving, runtime.retriever)

    # -- arms --------------------------------------------------------------

    async def run_federation(
        self,
        dataset: EvalDataset,
        *,
        repeats: int = 1,
        on_progress: ProgressCallback | None = None,
        persist: bool = True,
    ) -> EvaluationRun:
        """Run every case through the full federation."""
        results: list[CaseResult] = []
        cases = list(dataset.cases) * max(1, repeats)

        for index, case in enumerate(cases, start=1):
            result = await self._run_federation_case(case)
            results.append(result)
            if on_progress is not None:
                on_progress(index, len(cases), result)

        run = EvaluationRun(
            run_id=new_id("run"),
            arm="federation",
            dataset=dataset.source or dataset.name,
            domain=dataset.domain,
            results=results,
            metrics=aggregate("federation", results),
        )
        if persist:
            self._persist(run)
        return run

    async def run_baseline(
        self,
        dataset: EvalDataset,
        *,
        repeats: int = 1,
        on_progress: ProgressCallback | None = None,
        persist: bool = True,
    ) -> EvaluationRun:
        """Run every case through the monolithic baseline."""
        results: list[CaseResult] = []
        cases = list(dataset.cases) * max(1, repeats)

        for index, case in enumerate(cases, start=1):
            result = await self._run_baseline_case(case)
            results.append(result)
            if on_progress is not None:
                on_progress(index, len(cases), result)

        run = EvaluationRun(
            run_id=new_id("run"),
            arm="baseline",
            dataset=dataset.source or dataset.name,
            domain=dataset.domain,
            results=results,
            metrics=aggregate("baseline", results),
        )
        if persist:
            self._persist(run)
        return run

    async def run_comparison(
        self,
        dataset: EvalDataset,
        *,
        repeats: int = 1,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[EvaluationRun, EvaluationRun, Comparison]:
        """Run both arms over the same set and compare them."""
        federation = await self.run_federation(
            dataset, repeats=repeats, on_progress=on_progress
        )
        baseline = await self.run_baseline(
            dataset, repeats=repeats, on_progress=on_progress
        )
        return federation, baseline, compare(federation.metrics, baseline.metrics)

    # -- single cases ------------------------------------------------------

    async def _run_federation_case(self, case: EvalCase) -> CaseResult:
        task = TaskEnvelope(
            intent_text=case.question,
            tenant_id=self.runtime.tenant_id,
            principal=case.principal,
            claims=list(case.claims),
            source="eval",
        )
        try:
            result = await self.runtime.run(task, persist=False)
        except Exception as exc:
            logger.warning("Federation case %s crashed: %s", case.id, exc)
            return CaseResult(
                case_id=case.id,
                domain=case.domain,
                question=case.question,
                expected=case.expected,
                error=str(exc),
            )

        graded = await grade(case, result.answer, serving=self.runtime.serving)
        return CaseResult(
            case_id=case.id,
            domain=case.domain,
            question=case.question,
            answer=result.answer,
            expected=case.expected,
            score=graded.score,
            correct=graded.correct,
            grader=graded.grader,
            grader_reason=graded.reason,
            latency_ms=result.metrics.wall_ms,
            cost_usd=result.metrics.cost_usd,
            tokens=result.metrics.total_tokens,
            used_fallback=result.metrics.used_fallback,
            trace_complete=result.trace_is_complete(),
            experts=result.experts,
            error=result.error if result.status == "error" else "",
            detail={
                "confidence": result.confidence,
                "conflicts": result.metrics.conflicts,
                "strategy": result.plan.strategy if result.plan else "",
                "grader_detail": graded.detail,
            },
        )

    async def _run_baseline_case(self, case: EvalCase) -> CaseResult:
        meter = UsageMeter()
        answer = await self.baseline.answer(case.question, meter=meter)
        graded = await grade(case, answer.answer, serving=self.runtime.serving)
        return CaseResult(
            case_id=case.id,
            domain=case.domain,
            question=case.question,
            answer=answer.answer,
            expected=case.expected,
            score=graded.score,
            correct=graded.correct,
            grader=graded.grader,
            grader_reason=graded.reason,
            latency_ms=answer.latency_ms,
            cost_usd=answer.cost_usd,
            tokens=answer.tokens,
            used_fallback=False,
            # The monolith has no decomposition, no arbitration and no per-agent
            # governance to record. Its trail is a prompt and a completion.
            trace_complete=False,
            experts=["monolithic-llm"],
            error=answer.error,
            detail={"citations": answer.citations},
        )

    # -- calibration -------------------------------------------------------

    async def fit_calibrations(
        self, dataset: EvalDataset, *, persist: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Fit per-expert confidence calibration from labelled outcomes.

        Runs each case, pairs the expert's raw confidence with whether the
        federation's answer was actually correct, and fits Platt scaling. This
        is what turns "confidence" from a number a model emitted into something
        arbitration is entitled to act on.
        """
        samples: dict[str, list[tuple[float, bool]]] = {}

        for case in dataset.cases:
            task = TaskEnvelope(
                intent_text=case.question,
                tenant_id=self.runtime.tenant_id,
                principal=case.principal,
                claims=list(case.claims),
                source="eval",
            )
            try:
                result = await self.runtime.run(task, persist=False)
            except Exception:
                continue
            graded = await grade(case, result.answer, serving=self.runtime.serving)
            for answer in result.answers:
                if answer.ok and answer.expert_id:
                    samples.setdefault(answer.expert_id, []).append(
                        (answer.raw_confidence, graded.correct)
                    )

        store = CalibrationStore(self.runtime.tenant_id)
        report: dict[str, dict[str, Any]] = {}
        for expert_id, pairs in samples.items():
            calibration = fit_calibration(
                expert_id, [c for c, _ in pairs], [o for _, o in pairs]
            )
            if persist and calibration.fitted:
                store.save(calibration)
            report[expert_id] = calibration.to_dict()

        self.runtime.calibrations.invalidate()
        return report

    # -- persistence -------------------------------------------------------

    def _persist(self, run: EvaluationRun) -> None:
        try:
            with session_scope() as session:
                session.add(
                    EvalRunRow(
                        run_id=run.run_id,
                        tenant_id=self.runtime.tenant_id,
                        name=f"{run.arm}:{run.dataset}",
                        arm=run.arm,
                        pack=self.pack_name,
                        dataset=run.dataset,
                        domain=run.domain,
                        metrics=run.metrics.model_dump(mode="json"),
                        config={
                            "router_threshold": self.runtime.settings.router_confidence_threshold,
                            "arbitration": self.runtime.settings.arbitration_strategies,
                            "embedder": self.runtime.embedder.signature,
                        },
                        case_count=len(run.results),
                    )
                )
                for result in run.results:
                    session.add(
                        EvalCaseRow(
                            run_id=run.run_id,
                            case_id=result.case_id,
                            domain=result.domain,
                            score=result.score,
                            correct=result.correct,
                            latency_ms=result.latency_ms,
                            cost_usd=result.cost_usd,
                            used_fallback=result.used_fallback,
                            trace_complete=result.trace_complete,
                            answer=result.answer[:8000],
                            expected=result.expected[:4000],
                            detail=result.detail,
                        )
                    )
        except Exception:  # pragma: no cover
            logger.warning("Failed to persist evaluation run %s", run.run_id, exc_info=True)


def load_pack_datasets(paths: Sequence[str | Path], *, domain: str = "") -> EvalDataset:
    """Load a pack's evaluation files, optionally narrowed to one domain."""
    dataset = load_datasets(paths)
    return dataset.filter_domain(domain) if domain else dataset


def list_runs(tenant_id: str = "default", limit: int = 20) -> list[dict[str, Any]]:
    """Recent evaluation runs, newest first."""
    from sqlalchemy import select

    with session_scope() as session:
        rows = session.execute(
            select(EvalRunRow)
            .where(EvalRunRow.tenant_id == tenant_id)
            .order_by(EvalRunRow.created_at.desc())
            .limit(limit)
        ).scalars().all()
        return [
            {
                "run_id": r.run_id,
                "arm": r.arm,
                "domain": r.domain,
                "dataset": r.dataset,
                "cases": r.case_count,
                "metrics": dict(r.metrics or {}),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]
