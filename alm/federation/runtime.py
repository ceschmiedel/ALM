"""The federation runtime — L1 through L5, wired.

One request flows: LIP structures the intent (L1) → the Context Graph decomposes
and routes it (L2) → the plan becomes a DAG (L4) → Expert Agents execute with
their own models and their own retrieval (L3) → arbitration and synthesis
produce the answer with its trail (L5).  Governance sits across all of it, and
nothing touches context outside it.

The runtime is also where the architecture's honesty lives.  It records the
fallback rate, the cost and the latency of every run; it captures failures and
escalations as feedback for the next distillation cycle; and when it degrades —
no orchestrator, no embedder, an uncalibrated expert — it says so in the trace
instead of quietly producing a worse answer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alm.arbitration.arbiter import Arbiter
from alm.arbitration.strategies import build_strategies
from alm.arbitration.synthesis import Synthesizer
from alm.cmrag.embeddings import Embedder, get_embedder
from alm.cmrag.retriever import CMRAGRetriever
from alm.cmrag.store import ChunkStore
from alm.config import Settings, get_settings
from alm.core.errors import ALMError
from alm.core.ids import session_id as new_session_id
from alm.core.telemetry import Stopwatch, UsageMeter, alm_span
from alm.experts.calibration import CalibrationStore
from alm.experts.pack import DomainPack, load_pack
from alm.experts.registry import ExpertRegistry
from alm.federation.result import FederationResult, RunMetrics
from alm.governance.ibac import IBACEngine
from alm.governance.policies import PolicySet
from alm.graph.store import ContextGraph
from alm.models.registry import ModelRegistry
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.orchestration.context import ExecutionContext
from alm.orchestration.dag import build_dag
from alm.orchestration.executor import DAGExecutor
from alm.persistence.database import init_db, session_scope
from alm.persistence.models import FeedbackRow, SessionRow
from alm.protocol.answer import ExpertAnswer
from alm.protocol.task import SubTask, TaskEnvelope
from alm.protocol.trace import ExecutionTrace, Layer, TraceEvent
from alm.router.router import CognitiveRouter

logger = logging.getLogger(__name__)

_FALLBACK_SYSTEM = """\
You are the orchestrator of an ALM federation. No domain specialist covered this
request with sufficient confidence, so it has been escalated to you.

Answer it directly and completely. If retrieved context is supplied, prefer it
over your general knowledge and cite what you used. If you cannot answer
reliably, say so — an honest gap is more useful than a confident guess.
"""


class FederationRuntime:
    """The whole federation, as one object."""

    def __init__(
        self,
        *,
        tenant_id: str = "default",
        settings: Settings | None = None,
        graph: ContextGraph | None = None,
        models: ModelRegistry | None = None,
        embedder: Embedder | None = None,
        chunk_store: ChunkStore | None = None,
        ibac: IBACEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tenant_id = tenant_id

        self.embedder = embedder if embedder is not None else get_embedder(self.settings)
        self.graph = graph or ContextGraph(tenant_id, embedder=self.embedder.embed)
        if not self.graph.has_embedder:
            self.graph.set_embedder(self.embedder.embed)

        self.models = models or ModelRegistry(tenant_id)
        self.serving = ServingRouter(self.models)

        self.chunk_store = chunk_store or ChunkStore(tenant_id)
        self.retriever = CMRAGRetriever(self.chunk_store, self.embedder)

        self.ibac = ibac or IBACEngine(
            PolicySet(),
            tenant_id=tenant_id,
            enabled=self.settings.governance_enabled,
            default_effect=self.settings.governance_default_decision,
        )
        self.calibrations = CalibrationStore(tenant_id)

        self.experts = ExpertRegistry(
            self.graph,
            self.serving,
            retriever=self.retriever,
            ibac=self.ibac,
            calibrations=self.calibrations,
        )
        self.router = CognitiveRouter(self.graph, self.serving, settings=self.settings)
        self.executor = DAGExecutor(
            self.experts,
            settings=self.settings,
            fallback_handler=self._orchestrator_fallback,
        )
        self.arbiter = Arbiter(
            build_strategies(
                self.settings.arbitration_strategies,
                graph=self.graph,
                serving=self.serving,
                confidence_margin=self.settings.conflict_confidence_margin,
                judge_enabled=self.settings.judge_enabled,
            ),
            experts=self.experts,
        )
        self.synthesizer = Synthesizer(self.serving)

        self.experts.refresh()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_database(
        cls, *, tenant_id: str = "default", settings: Settings | None = None
    ) -> FederationRuntime:
        """Build from what is already installed — the production path."""
        init_db()
        runtime = cls(tenant_id=tenant_id, settings=settings)
        runtime.load_policies_from_graph()
        return runtime

    @classmethod
    def from_pack(
        cls,
        path: str | Path,
        *,
        tenant_id: str = "default",
        settings: Settings | None = None,
        install: bool = True,
        index_corpus: bool = True,
    ) -> FederationRuntime:
        """Build a runtime from a domain pack, installing it if needed."""
        init_db()
        pack = load_pack(path)
        runtime = cls(tenant_id=tenant_id or pack.tenant, settings=settings)
        if install:
            pack.install(
                graph=runtime.graph,
                registry=runtime.models,
                store=runtime.chunk_store,
                embedder=runtime.embedder,
                index_corpus=index_corpus,
            )
        runtime.apply_pack_policies(pack)
        runtime.experts.refresh()
        runtime.router.invalidate()
        return runtime

    def apply_pack_policies(self, pack: DomainPack) -> None:
        """Load a pack's IBAC policies into the running engine."""
        if len(pack.policies):
            self.ibac.load(pack.policies)
            logger.info("Loaded %d IBAC policy(ies) from pack %s", len(pack.policies), pack.name)

    def load_policies_from_graph(self) -> int:
        """Load policies persisted as graph nodes (installed by earlier runs)."""
        from alm.governance.policies import Policy
        from alm.graph.models import NodeKind

        loaded = 0
        for node in self.graph.find_nodes(NodeKind.POLICY):
            raw = node.attributes.get("policy")
            if isinstance(raw, dict):
                try:
                    self.ibac.add_policy(Policy.model_validate(raw))
                    loaded += 1
                except Exception:
                    logger.warning("Invalid persisted policy %r", node.key, exc_info=True)
        return loaded

    # -- main entry point --------------------------------------------------

    async def run(
        self,
        intent: str | TaskEnvelope,
        *,
        context: dict[str, Any] | None = None,
        principal: str = "",
        claims: list[str] | None = None,
        requested_outputs: list[str] | None = None,
        session_id: str = "",
        on_event: Callable[[TraceEvent], None] | None = None,
        persist: bool = True,
    ) -> FederationResult:
        """Run one intent through the federation and return the full result."""
        task = self._as_task(
            intent,
            context=context,
            principal=principal,
            claims=claims,
            requested_outputs=requested_outputs,
            session_id=session_id,
        )
        trace = ExecutionTrace(session_id=task.session_id)
        if on_event is not None:
            trace.subscribe(on_event)

        meter = UsageMeter()
        self.serving.reset()
        stopwatch = Stopwatch()
        result = FederationResult(session_id=task.session_id, task=task, trace=trace)

        with alm_span(
            "alm.federation.run",
            attributes={"session_id": task.session_id, "tenant": self.tenant_id},
        ):
            try:
                await self._run_inner(task, trace, meter, result, stopwatch)
            except ALMError as exc:
                result.status = "error"
                result.error = exc.message
                result.answer = f"The federation could not complete this request: {exc.message}"
                trace.emit(
                    Layer.L1_INTERFACE, f"run failed: {exc.message}", category="error",
                    detail=exc.to_dict(),
                )
                logger.warning("Federation run %s failed: %s", task.session_id, exc.message)
            except Exception as exc:  # pragma: no cover - defensive
                result.status = "error"
                result.error = str(exc)
                result.answer = "The federation could not complete this request."
                trace.emit(
                    Layer.L1_INTERFACE, f"run failed: {exc}", category="error"
                )
                logger.exception("Federation run %s crashed", task.session_id)

        result.metrics.wall_ms = stopwatch.stop()
        self._finalise_metrics(result, meter)

        if persist:
            self._persist(result)
        self._capture_feedback(result)
        return result

    async def _run_inner(
        self,
        task: TaskEnvelope,
        trace: ExecutionTrace,
        meter: UsageMeter,
        result: FederationResult,
        stopwatch: Stopwatch,
    ) -> None:
        # -- L1 · admission -------------------------------------------------
        trace.emit(
            Layer.L1_INTERFACE,
            f"intent admitted: {task.intent_text[:120]}",
            category="intent",
            detail={
                "task_id": task.task_id,
                "tenant": task.tenant_id,
                "principal": task.principal or "anonymous",
                "source": task.source,
            },
        )

        admission = self.ibac.admit_intent(
            intent_text=task.intent_text,
            principal=task.principal,
            claims=task.claims,
            session_id=task.session_id,
            context=task.context,
        )
        if not admission.allowed:
            result.status = "denied"
            result.error = admission.reason
            result.answer = f"This request was refused by governance: {admission.reason}"
            trace.emit(
                Layer.GV_GOVERNANCE,
                f"intent denied: {admission.reason}",
                category="denied",
                detail={"policies": admission.matched_policies},
            )
            return

        if not self.graph.has_embedder:
            trace.emit(
                Layer.CX_CONTEXT,
                "no embedder configured; routing and retrieval are lexical only",
                category="warning",
            )

        # -- L2 · routing ---------------------------------------------------
        routing_watch = Stopwatch()
        outcome = await self.router.route(task, trace=trace, meter=meter)
        result.metrics.routing_ms = routing_watch.stop()
        result.plan = outcome.plan
        result.metrics.used_fallback = outcome.plan.is_fallback

        # -- L4 · orchestration + L3 · execution ----------------------------
        dag = build_dag(outcome.plan)
        execution_watch = Stopwatch()
        report = await self.executor.execute(
            dag,
            task,
            trace=trace,
            meter=meter,
            context=ExecutionContext(task.session_id, tenant_id=task.tenant_id),
        )
        result.metrics.execution_ms = execution_watch.stop()
        result.answers = report.answers
        result.metrics.dag_width = dag.width
        result.metrics.dag_depth = dag.depth
        result.metrics.experts_invoked = len(report.answers)
        result.metrics.experts_succeeded = len(report.successful)

        # -- L5 · arbitration and synthesis ---------------------------------
        arbitration_watch = Stopwatch()
        arbitration = await self.arbiter.arbitrate(
            report.answers,
            intent_text=task.intent_text,
            primary_domain=outcome.classification.primary,
            trace=trace,
            meter=meter,
        )
        result.arbitration = arbitration
        result.contributions = arbitration.contributions

        synthesis = await self.synthesizer.synthesize(
            report.answers,
            arbitration,
            intent_text=task.intent_text,
            requested_outputs=task.requested_outputs,
            trace=trace,
            meter=meter,
        )
        result.metrics.arbitration_ms = arbitration_watch.stop()

        result.answer = synthesis.answer
        result.citations = synthesis.citations
        result.confidence = synthesis.confidence
        result.synthesis_method = synthesis.method
        result.metrics.conflicts = len(arbitration.conflicts)
        result.metrics.conflicts_resolved = len(arbitration.resolutions)

        # -- GV · emission --------------------------------------------------
        emission = self.ibac.may_emit(
            domains=outcome.classification.domains,
            entities=[e for c in result.citations for e in c.entities],
            sensitivity=_max_sensitivity(result),
            principal=task.principal,
            claims=task.claims,
            session_id=task.session_id,
        )
        if not emission.allowed:
            result.status = "denied"
            result.error = emission.reason
            result.answer = (
                "An answer was produced but governance blocked its release: "
                f"{emission.reason}"
            )
            trace.emit(
                Layer.GV_GOVERNANCE,
                f"artifact emission denied: {emission.reason}",
                category="denied",
                detail={"policies": emission.matched_policies},
            )
            return

        result.status = "success" if report.successful else "error"
        if not report.successful:
            result.error = "no expert produced a usable answer"

    # -- fallback ----------------------------------------------------------

    async def _orchestrator_fallback(
        self,
        subtask: SubTask,
        task: TaskEnvelope,
        upstream: dict[str, Any],
        meter: UsageMeter | None,
    ) -> ExpertAnswer:
        """Run an unrouted subtask on the orchestrator.

        This is the guarantee that the federation is never worse than the
        monolithic baseline: at worst, the request is answered by the same
        frontier model the baseline would have used.
        """
        answer = ExpertAnswer(
            subtask_id=subtask.subtask_id,
            expert_id="orchestrator",
            domain=subtask.domain,
            tier=str(ModelTier.ORCHESTRATOR),
            is_fallback=True,
        )

        # The orchestrator still gets retrieval — sovereign context is useful
        # regardless of which model consumes it.
        retrieval = None
        if self.retriever is not None:
            access_filter = self.ibac.context_filter(
                expert_id="orchestrator",
                domain=subtask.domain,
                principal=task.principal,
                claims=task.claims,
                session_id=task.session_id,
            )
            retrieval = self.retriever.with_access_filter(access_filter).retrieve(
                task.intent_text,
                domains=[subtask.domain] if subtask.domain else None,
                top_k=8,
            )
            answer.citations = retrieval.citations()

        blocks = retrieval.as_context_blocks() if retrieval else []
        sections = [f"## Request\n{subtask.description}"]
        upstream_items = upstream.get("upstream") or []
        if upstream_items:
            sections.append(
                "## Findings from specialists\n"
                + "\n\n".join(
                    f"### {item['expert_id']} ({item['domain']})\n{item.get('content', '')}"
                    for item in upstream_items
                )
            )
        if blocks:
            sections.append(
                "## Retrieved context\n"
                + "\n\n".join(f"[{b['index']}] {b['title']}\n{b['text']}" for b in blocks)
            )

        request = GenerationRequest(
            messages=[
                Message(role="system", content=_FALLBACK_SYSTEM),
                Message(role="user", content="\n\n".join(sections)),
            ],
            temperature=0.1,
            max_tokens=1200,
            metadata={
                "question": subtask.description,
                "context_chunks": blocks,
                "max_sentences": 5,
            },
        )
        try:
            generated = await self.serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                meter=meter,
                component="federation.fallback",
            )
        except Exception as exc:
            answer.status = "error"
            answer.error = f"orchestrator fallback failed: {exc}"
            return answer

        answer.content = generated.text.strip()
        answer.model_id = generated.model_id
        answer.tier = generated.tier
        answer.prompt_tokens = generated.prompt_tokens
        answer.completion_tokens = generated.completion_tokens
        answer.latency_ms = generated.latency_ms
        answer.cost_usd = generated.cost_usd
        # The orchestrator has no fitted calibration; treat it as a strong but
        # unvalidated source rather than an authority.
        answer.raw_confidence = 0.7 if answer.content else 0.0
        answer.confidence = answer.raw_confidence
        answer.calibrated = False
        answer.status = "success" if answer.content else "error"
        if not answer.content:
            answer.error = "orchestrator returned an empty answer"
        return answer

    # -- helpers -----------------------------------------------------------

    def _as_task(
        self,
        intent: str | TaskEnvelope,
        *,
        context: dict[str, Any] | None,
        principal: str,
        claims: list[str] | None,
        requested_outputs: list[str] | None,
        session_id: str,
    ) -> TaskEnvelope:
        if isinstance(intent, TaskEnvelope):
            if session_id:
                intent.session_id = session_id
            if not intent.session_id:
                intent.session_id = new_session_id()
            return intent
        return TaskEnvelope(
            session_id=session_id or new_session_id(),
            intent_text=intent,
            context=context or {},
            requested_outputs=requested_outputs or [],
            tenant_id=self.tenant_id,
            principal=principal,
            claims=claims or [],
        )

    def _finalise_metrics(self, result: FederationResult, meter: UsageMeter) -> None:
        summary = meter.summary()
        metrics: RunMetrics = result.metrics
        metrics.cost_usd = summary["total_cost_usd"]
        metrics.total_tokens = summary["total_tokens"]
        metrics.model_calls = summary["calls"]
        metrics.inference_ms = summary["inference_latency_ms"]
        metrics.by_tier = summary["by_tier"]
        metrics.by_model = summary["by_model"]
        metrics.escalations = self.serving.escalation_count()
        if any(a.is_fallback for a in result.answers):
            metrics.used_fallback = True

    def _persist(self, result: FederationResult) -> None:
        try:
            with session_scope() as session:
                row = session.get(SessionRow, result.session_id)
                if row is None:
                    row = SessionRow(session_id=result.session_id)
                    session.add(row)
                row.tenant_id = self.tenant_id
                row.principal = result.task.principal
                row.intent_text = result.task.intent_text
                row.status = result.status
                row.strategy = result.plan.strategy if result.plan else ""
                row.router_confidence = result.plan.router_confidence if result.plan else 0.0
                row.used_fallback = result.metrics.used_fallback
                row.answer = result.answer
                row.plan = result.plan.model_dump(mode="json") if result.plan else {}
                row.answers = [a.model_dump(mode="json") for a in result.answers]
                row.arbitration = (
                    result.arbitration.to_dict() if result.arbitration else {}
                )
                row.trace = result.trace.to_dict()
                row.metrics = result.metrics.model_dump(mode="json")
                row.error = result.error
                from alm.core.ids import utc_now

                row.completed_at = utc_now()
        except Exception:  # pragma: no cover - persistence must not break a run
            logger.warning("Failed to persist session %s", result.session_id, exc_info=True)

    def _capture_feedback(self, result: FederationResult) -> None:
        """Record escalations and failures for the next distillation cycle.

        This is the loop that makes the federation improve with use: every case
        an expert failed or handed to the LLM is exactly the material the
        teacher should be generating training data about.
        """
        if not self.settings.audit_enabled:
            return
        entries: list[dict[str, Any]] = []

        if result.metrics.used_fallback and result.plan is not None:
            entries.append(
                {
                    "kind": "escalation",
                    "expert_id": "",
                    "domain": result.plan.domains[0] if result.plan.domains else "",
                    "expected": "",
                    "actual": result.answer[:2000],
                    "detail": {
                        "reason": result.plan.fallback_reason,
                        "router_confidence": result.plan.router_confidence,
                    },
                }
            )
        for answer in result.answers:
            if answer.status in {"error", "skipped"} and answer.expert_id:
                entries.append(
                    {
                        "kind": "failure",
                        "expert_id": answer.expert_id,
                        "domain": answer.domain,
                        "expected": "",
                        "actual": answer.error[:2000],
                        "detail": {"subtask_id": answer.subtask_id, "status": answer.status},
                    }
                )

        if not entries:
            return
        try:
            with session_scope() as session:
                for entry in entries:
                    session.add(
                        FeedbackRow(
                            tenant_id=self.tenant_id,
                            session_id=result.session_id,
                            intent_text=result.task.intent_text,
                            **entry,
                        )
                    )
        except Exception:  # pragma: no cover
            logger.debug("Failed to capture feedback", exc_info=True)

    # -- lifecycle ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Inventory and degradation report — backs `alm doctor` and `/v1/health`."""
        orchestrator = self.models.orchestrator()
        heuristic_experts = [
            spec.id
            for spec in self.experts.specs()
            if spec.model and (self.models.get(spec.model) or _null_spec()).backend
            in {"heuristic", "scripted"}
        ]
        return {
            "tenant": self.tenant_id,
            "graph": self.graph.stats().model_dump(),
            "experts": len(self.experts),
            "models": self.models.serving_summary(),
            "corpus": self.chunk_store.stats(),
            "embedder": self.embedder.signature,
            "semantic_routing": self.graph.has_embedder,
            "orchestrator_configured": orchestrator is not None,
            "fallback_available": orchestrator is not None
            and self.settings.fallback_enabled,
            "governance_enabled": self.ibac.enabled,
            "policies": len(self.ibac.policies),
            "uncalibrated_experts": [
                spec.id
                for spec in self.experts.specs()
                if not self.calibrations.get(spec.id).fitted
            ],
            "experts_on_placeholder_backends": heuristic_experts,
        }

    async def close(self) -> None:
        await self.models.close()


def _null_spec():
    from alm.models.spec import ModelSpec

    return ModelSpec(model_id="", backend="")


def _max_sensitivity(result: FederationResult) -> str:
    from alm.governance.policies import SENSITIVITY_ORDER

    best = "public"
    for answer in result.answers:
        for citation in answer.citations:
            level = str(citation.metadata.get("sensitivity", "")) or ""
            if level in SENSITIVITY_ORDER and SENSITIVITY_ORDER.index(
                level
            ) > SENSITIVITY_ORDER.index(best):
                best = level
    return best
