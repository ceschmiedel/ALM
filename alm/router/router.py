"""L2 · The cognitive router.

This is where the federation is won or lost.  The router takes the structured
task from L1 and answers three questions: **what does this decompose into**,
**which Expert Agent resolves each part**, and **with what confidence**.  An
error here contaminates everything downstream — a subtask sent to the wrong
specialist cannot be rescued by a good specialist.

Which is why the last question matters as much as the first two.  The router
must know when it does not know.  Below a confidence threshold, or when nothing
matches a declared capability, the correct move is to escalate to the
orchestrator LLM rather than guess.

That escalation is designed, not conceded.  It guarantees the federation is
never worse than the monolithic baseline: in the worst case it hands the task
back to the LLM; in the common case it answers locally, cheaply and inside the
perimeter.

The **fallback rate is a health metric**, and the runtime reports it as one:
too high means expert coverage is insufficient; too low may mean the router is
gambling on cases it should be escalating.
"""

from __future__ import annotations

import logging

from alm.config import Settings, get_settings
from alm.core.telemetry import UsageMeter
from alm.graph.store import ContextGraph
from alm.models.serving import ServingRouter
from alm.protocol.task import EntityRef, ExecutionPlan, SubTask, TaskEnvelope
from alm.protocol.trace import ExecutionTrace, Layer
from alm.router.classifier import Classification, IntentClassifier
from alm.router.decomposer import TaskDecomposer
from alm.router.matcher import CapabilityMatcher

logger = logging.getLogger(__name__)


class RoutingOutcome:
    """An execution plan plus the reasoning that produced it."""

    __slots__ = ("plan", "classification", "decomposition_strategy", "escalated", "reason")

    def __init__(
        self,
        plan: ExecutionPlan,
        classification: Classification,
        decomposition_strategy: str,
        escalated: bool,
        reason: str,
    ) -> None:
        self.plan = plan
        self.classification = classification
        self.decomposition_strategy = decomposition_strategy
        self.escalated = escalated
        self.reason = reason


class CognitiveRouter:
    """Classifies, decomposes, matches and decides when to escalate."""

    def __init__(
        self,
        graph: ContextGraph,
        serving: ServingRouter | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.graph = graph
        self.serving = serving
        self.settings = settings or get_settings()
        self.matcher = CapabilityMatcher(graph)
        self.classifier = IntentClassifier(graph, self.matcher, serving)
        self.decomposer = TaskDecomposer(
            graph,
            self.matcher,
            serving,
            max_subtasks=self.settings.router_max_subtasks,
        )

    # -- routing -----------------------------------------------------------

    async def route(
        self,
        task: TaskEnvelope,
        *,
        trace: ExecutionTrace | None = None,
        meter: UsageMeter | None = None,
    ) -> RoutingOutcome:
        """Produce an execution plan for a task."""
        threshold = self.settings.router_confidence_threshold

        classification = await self.classifier.classify(task.intent_text)
        if trace is not None:
            trace.emit(
                Layer.L2_ROUTER,
                (
                    f"classified into {', '.join(classification.domains) or 'no domain'} "
                    f"(confidence {classification.confidence:.2f}, via {classification.method})"
                ),
                category="classification",
                detail={
                    "domains": classification.domains,
                    "scores": classification.scores,
                    "entities": classification.entities,
                    "method": classification.method,
                    "reason": classification.reason,
                },
            )

        decomposition = await self.decomposer.decompose(task, classification, meter=meter)

        if trace is not None and decomposition.subtasks:
            trace.emit(
                Layer.L2_ROUTER,
                (
                    f"decomposed into {len(decomposition.subtasks)} subtask(s) "
                    f"via {decomposition.strategy} strategy"
                ),
                category="decomposition",
                detail={
                    "strategy": decomposition.strategy,
                    "reason": decomposition.reason,
                    "subtasks": [
                        {
                            "id": st.subtask_id,
                            "expert": st.expert_id or "(fallback)",
                            "capability": st.capability_id,
                            "confidence": round(st.confidence, 4),
                            "depends_on": st.depends_on,
                        }
                        for st in decomposition.subtasks
                    ],
                },
            )

        matched = [st for st in decomposition.subtasks if not st.is_fallback]
        confidence = _plan_confidence(decomposition.subtasks, classification)

        # -- escalation decision -------------------------------------------
        escalate_reason = ""
        if not decomposition.subtasks:
            escalate_reason = "no capability in the graph covers this request"
        elif not matched:
            escalate_reason = "no subtask matched a declared capability"
        elif confidence < threshold:
            escalate_reason = (
                f"routing confidence {confidence:.2f} is below the "
                f"{threshold:.2f} threshold"
            )

        if escalate_reason:
            if not self.settings.fallback_enabled:
                # Without a fallback the federation must still answer with what
                # it has; refusing would be worse than a low-confidence route.
                if decomposition.subtasks:
                    if trace is not None:
                        trace.emit(
                            Layer.L2_ROUTER,
                            f"low confidence ({confidence:.2f}) but fallback is disabled; "
                            "proceeding with the best available experts",
                            category="warning",
                            detail={"reason": escalate_reason},
                        )
                else:
                    raise _no_route_error(task, escalate_reason)
            else:
                return self._escalate(task, classification, confidence, escalate_reason, trace)

        plan = ExecutionPlan(
            task_id=task.task_id,
            session_id=task.session_id,
            subtasks=decomposition.subtasks,
            router_confidence=confidence,
            strategy="single_expert" if len(matched) == 1 else "federated",
            domains=classification.domains,
        )

        problems = plan.validate_dependencies()
        if problems:
            logger.warning("Plan %s has dependency problems: %s", plan.plan_id, problems)
            for subtask in plan.subtasks:
                subtask.depends_on = [
                    d for d in subtask.depends_on if plan.by_id(d) is not None
                ]

        return RoutingOutcome(
            plan,
            classification,
            decomposition.strategy,
            escalated=False,
            reason=decomposition.reason,
        )

    # -- fallback ----------------------------------------------------------

    def _escalate(
        self,
        task: TaskEnvelope,
        classification: Classification,
        confidence: float,
        reason: str,
        trace: ExecutionTrace | None,
    ) -> RoutingOutcome:
        """Hand the whole task to the orchestrator as a single fallback step."""
        if trace is not None:
            trace.emit(
                Layer.L2_ROUTER,
                f"escalating to the orchestrator: {reason}",
                category="fallback",
                detail={
                    "reason": reason,
                    "confidence": round(confidence, 4),
                    "threshold": self.settings.router_confidence_threshold,
                    "domains": classification.domains,
                },
            )
        logger.info("Router escalating to orchestrator: %s", reason)

        subtask = SubTask(
            description=task.intent_text,
            domain=classification.primary,
            candidate_domains=classification.domains,
            expert_id="",
            capability_id="",
            entities=[
                EntityRef(entity_type=e, source="router") for e in classification.entities
            ],
            requested_outputs=list(task.requested_outputs),
            confidence=confidence,
            rationale=reason,
            is_fallback=True,
        )
        plan = ExecutionPlan(
            task_id=task.task_id,
            session_id=task.session_id,
            subtasks=[subtask],
            router_confidence=confidence,
            strategy="fallback_orchestrator",
            fallback_reason=reason,
            domains=classification.domains,
        )
        return RoutingOutcome(
            plan, classification, "fallback", escalated=True, reason=reason
        )

    # -- introspection -----------------------------------------------------

    def explain(self, text: str, *, top_k: int = 5) -> dict[str, object]:
        """Dry-run routing signals for a request, without executing anything.

        Backs `alm expert test` and the routing panel in the API: the router is
        auditable in the same way its output is.
        """
        classification = self.classifier.classify_with_graph(text)
        matches = self.matcher.match(
            text,
            domain_scores=classification.scores,
            entities=classification.entities,
            top_k=top_k,
        )
        return {
            "text": text,
            "entities_detected": classification.entities,
            "domains": classification.domains,
            "domain_scores": classification.scores,
            "classification_confidence": classification.confidence,
            "semantic_available": self.graph.has_embedder,
            "matches": [m.model_dump() for m in matches],
            "threshold": self.settings.router_confidence_threshold,
            "would_escalate": (
                not matches or matches[0].score < self.settings.router_confidence_threshold
            ),
        }

    def invalidate(self) -> None:
        """Drop cached ontology terms after the graph changes."""
        self.matcher.invalidate()


def _plan_confidence(subtasks: list[SubTask], classification: Classification) -> float:
    """Confidence in the plan as a whole.

    Weakest-link dominated, because one misrouted step is enough to make the
    synthesised answer wrong: a plan is only as trustworthy as its worst
    assignment.  Classification then enters as *corroboration* rather than as a
    third of the average — a saturating boost that can raise a decent plan but
    can never rescue a bad one, since it scales with the room left above the
    core score.

    Subtasks the router could not assign at all are a direct penalty: a plan
    half of which needs the orchestrator is not a confident federation route.
    """
    routed = [st for st in subtasks if not st.is_fallback]
    if not routed:
        return 0.0

    weakest = min(st.confidence for st in routed)
    mean = sum(st.confidence for st in routed) / len(routed)
    core = 0.6 * weakest + 0.4 * mean

    corroborated = core + (1.0 - core) * 0.35 * classification.confidence

    unmatched = len(subtasks) - len(routed)
    penalty = 1.0 - 0.5 * unmatched / max(len(subtasks), 1)

    return round(max(0.0, min(1.0, corroborated * penalty)), 4)


def _no_route_error(task: TaskEnvelope, reason: str):
    from alm.core.errors import NoCapabilityMatchError

    return NoCapabilityMatchError(
        f"cannot route this request and fallback is disabled: {reason}",
        intent_text=task.intent_text[:200],
        reason=reason,
    )
