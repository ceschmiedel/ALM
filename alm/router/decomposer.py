"""Task decomposition — turning one request into a dependency graph of subtasks.

Few enterprise decisions are a single step. The legal opinion feeds the
financial analysis, which feeds the risk assessment. Treating that as a set of
loose parallel calls produces incoherent answers, so decomposition's real
output is not a list of subtasks but a list of subtasks *plus their
dependencies*.

Two strategies, chosen by how much structure the graph can already supply:

**Capability-driven** (no model call).  Matched capabilities become subtasks and
the order comes from the graph: explicit ``depends_on`` edges, plus inference
from ``produces``/``operates_on`` overlap — a capability that reads an entity
type another capability writes must run after it. This handles the common case
for free and is fully deterministic.

**Planner-driven** (orchestrator model).  For composite or ambiguous requests,
the frontier model proposes subtasks and dependencies in natural language, and
every proposed subtask is then matched back onto a declared capability. The
model suggests structure; the graph decides who executes. A subtask the graph
cannot match becomes an explicit fallback step rather than a silent guess.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from alm.core.jsonutil import extract_json_object
from alm.core.telemetry import UsageMeter
from alm.graph.models import CapabilityDeclaration
from alm.graph.store import ContextGraph
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.protocol.task import EntityRef, SubTask, TaskEnvelope
from alm.router.classifier import Classification
from alm.router.matcher import CapabilityMatcher

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """\
You are the planning layer of an ALM federation. You do not answer the request.
You break it into the smallest set of subtasks that specialist agents can solve
independently, and you declare the dependencies between them.

Available capabilities:
{capabilities}

Return ONLY a JSON object:
{{
  "subtasks": [
    {{
      "id": "s1",
      "description": "<self-contained instruction for one specialist>",
      "capability_id": "<one of the ids above, or empty if none fits>",
      "depends_on": ["<id of a subtask whose output this one needs>"]
    }}
  ],
  "reason": "<one sentence on how you split it>"
}}

Rules:
- Prefer FEWER subtasks. A single-domain request is one subtask.
- Each description must stand alone: a specialist sees only its own subtask
  plus the outputs of its declared dependencies.
- Declare a dependency only when the subtask genuinely needs the other's output.
  Independent subtasks run in parallel, so a spurious dependency costs latency.
- Never invent a capability_id. Leave it empty if nothing fits.
- At most {max_subtasks} subtasks.
"""

# Signals that a request is likely composite and worth a planner call.
_COMPOSITE_MARKERS = re.compile(
    r"\b(and then|then|after that|as well as|compare|both|also|"
    r"e depois|então|além disso|compare|ambos|também)\b",
    re.IGNORECASE,
)


class Decomposition:
    """The result of decomposing a task."""

    __slots__ = ("subtasks", "strategy", "reason", "unmatched")

    def __init__(
        self,
        subtasks: list[SubTask],
        strategy: str,
        reason: str = "",
        unmatched: int = 0,
    ) -> None:
        self.subtasks = subtasks
        self.strategy = strategy
        self.reason = reason
        #: Subtasks the planner proposed that no capability covered.
        self.unmatched = unmatched


class TaskDecomposer:
    """Produces subtasks and their dependencies."""

    #: A capability must score at least this fraction of the best match to be
    #: invited into the plan.
    RELATIVE_MATCH_BAND = 0.55

    def __init__(
        self,
        graph: ContextGraph,
        matcher: CapabilityMatcher,
        serving: ServingRouter | None = None,
        *,
        max_subtasks: int = 8,
        match_threshold: float = 0.2,
    ) -> None:
        self.graph = graph
        self.matcher = matcher
        self.serving = serving
        self.max_subtasks = max_subtasks
        self.match_threshold = match_threshold

    # -- entry point -------------------------------------------------------

    async def decompose(
        self,
        task: TaskEnvelope,
        classification: Classification,
        *,
        meter: UsageMeter | None = None,
        prefer_planner: bool | None = None,
    ) -> Decomposition:
        capabilities = self._candidate_capabilities(classification)
        if not capabilities:
            return Decomposition([], "none", "no capability is declared for these domains")

        use_planner = (
            self._should_plan(task, classification, capabilities)
            if prefer_planner is None
            else prefer_planner
        )

        if use_planner and self.serving is not None:
            planned = await self._plan_with_model(
                task, classification, capabilities, meter=meter
            )
            if planned is not None and planned.subtasks:
                return planned
            logger.info("Planner unavailable or empty; falling back to capability routing")

        return self._plan_from_capabilities(task, classification, capabilities)

    # -- strategy selection ------------------------------------------------

    def _should_plan(
        self,
        task: TaskEnvelope,
        classification: Classification,
        capabilities: Sequence[CapabilityDeclaration],
    ) -> bool:
        """Decide whether this request is worth an orchestrator call.

        Simple, single-domain requests are the majority and must not pay for
        planning.  A planner call is justified when the request spans domains,
        reads as multi-part, or the graph is genuinely unsure who should act.
        """
        if classification.is_multi_domain:
            return True
        if _COMPOSITE_MARKERS.search(task.intent_text):
            return True
        if len(task.requested_outputs) > 1:
            return True
        if classification.confidence < 0.35:
            return True
        return False

    def _candidate_capabilities(
        self, classification: Classification
    ) -> list[CapabilityDeclaration]:
        if not classification.domains:
            return self.graph.capabilities()
        selected: list[CapabilityDeclaration] = []
        for domain in classification.domains:
            selected.extend(self.graph.capabilities(domain))
        return selected or self.graph.capabilities()

    # -- capability-driven -------------------------------------------------

    def _plan_from_capabilities(
        self,
        task: TaskEnvelope,
        classification: Classification,
        capabilities: Sequence[CapabilityDeclaration],
    ) -> Decomposition:
        matches = self.matcher.match(
            task.intent_text,
            domain_scores=classification.scores,
            entities=classification.entities,
            capabilities=capabilities,
            top_k=self.max_subtasks * 2,
            min_score=self.match_threshold,
        )
        if not matches:
            return Decomposition(
                [], "none", "no capability scored above the routing threshold"
            )

        # Keep only experts close to the leader.  A marginal third specialist
        # adds cost and noise, and — because plan confidence is weakest-link —
        # it would drag an otherwise sound plan below the escalation threshold.
        # Not inviting it is different from failing to find it.
        best = matches[0].score
        cutoff = max(self.match_threshold, best * self.RELATIVE_MATCH_BAND)
        matches = [m for m in matches if m.score >= cutoff]

        # One subtask per expert: two capabilities of the same expert are one
        # visit, not two, and merging them keeps the plan honest about cost.
        by_expert: dict[str, list] = {}
        for match in matches:
            by_expert.setdefault(match.expert_id, []).append(match)

        declarations = {c.capability_id: c for c in capabilities}
        subtasks: list[SubTask] = []
        for expert_id, expert_matches in list(by_expert.items())[: self.max_subtasks]:
            best = expert_matches[0]
            declaration = declarations.get(best.capability_id)
            subtasks.append(
                SubTask(
                    description=self._describe(task, declaration),
                    retrieval_query=task.intent_text,
                    domain=best.domain,
                    expert_id=expert_id,
                    capability_id=best.capability_id,
                    entities=[
                        EntityRef(entity_type=e, source="router")
                        for e in classification.entities
                    ],
                    requested_outputs=list(task.requested_outputs),
                    confidence=best.score,
                    rationale=best.reason,
                    tier_hint=declaration.tier_hint if declaration else "",
                )
            )

        self._wire_dependencies(subtasks, declarations)
        return Decomposition(
            subtasks,
            "capability",
            f"matched {len(subtasks)} expert(s) directly from the capability graph",
        )

    @staticmethod
    def _describe(task: TaskEnvelope, declaration: CapabilityDeclaration | None) -> str:
        """Frame the request for one specialist without losing the original ask."""
        if declaration is None or not declaration.description:
            return task.intent_text
        return (
            f"{declaration.description.rstrip('.')}, as it applies to the following "
            f"request:\n\n{task.intent_text}"
        )

    def _wire_dependencies(
        self,
        subtasks: list[SubTask],
        declarations: dict[str, CapabilityDeclaration],
    ) -> None:
        """Derive execution order from the graph, not from list position."""
        by_capability = {st.capability_id: st for st in subtasks}

        for subtask in subtasks:
            declared = self.matcher.declared_dependencies(subtask.capability_id)
            for dependency in declared:
                upstream = by_capability.get(dependency)
                if upstream is not None and upstream.subtask_id != subtask.subtask_id:
                    if upstream.subtask_id not in subtask.depends_on:
                        subtask.depends_on.append(upstream.subtask_id)

        # Inferred: a consumer of an entity type must follow its producer.
        for consumer in subtasks:
            consumer_declaration = declarations.get(consumer.capability_id)
            if consumer_declaration is None:
                continue
            for producer in subtasks:
                if producer.subtask_id == consumer.subtask_id:
                    continue
                producer_declaration = declarations.get(producer.capability_id)
                if producer_declaration is None:
                    continue
                if self.matcher.infer_dependency(producer_declaration, consumer_declaration):
                    if producer.subtask_id not in consumer.depends_on:
                        consumer.depends_on.append(producer.subtask_id)

        _break_cycles(subtasks)

    # -- planner-driven ----------------------------------------------------

    async def _plan_with_model(
        self,
        task: TaskEnvelope,
        classification: Classification,
        capabilities: Sequence[CapabilityDeclaration],
        *,
        meter: UsageMeter | None = None,
    ) -> Decomposition | None:
        listing = "\n".join(
            f"- {c.capability_id} [{c.domain}]: {c.description}"
            + (f" (e.g. {c.examples[0]})" if c.examples else "")
            for c in capabilities
        )
        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=_PLANNER_SYSTEM.format(
                        capabilities=listing, max_subtasks=self.max_subtasks
                    ),
                ),
                Message(role="user", content=task.intent_text),
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=800,
        )
        try:
            result = await self.serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                meter=meter,
                component="router.planner",
                allow_escalation=False,
            )
        except Exception as exc:
            logger.info("Planner call failed (%s)", exc)
            return None

        parsed = extract_json_object(result.text)
        if not parsed or not isinstance(parsed.get("subtasks"), list):
            return None

        declarations = {c.capability_id: c for c in capabilities}
        subtasks: list[SubTask] = []
        id_map: dict[str, str] = {}
        unmatched = 0

        for raw in parsed["subtasks"][: self.max_subtasks]:
            if not isinstance(raw, dict):
                continue
            description = str(raw.get("description", "")).strip()
            if not description:
                continue

            capability_id = str(raw.get("capability_id", "")).strip()
            declaration = declarations.get(capability_id)

            if declaration is None:
                # The planner proposed a step but named no usable capability.
                # Re-match on the description before giving up on it.
                rematched = self.matcher.match(
                    description,
                    domain_scores=classification.scores,
                    capabilities=capabilities,
                    top_k=1,
                    min_score=self.match_threshold,
                )
                if rematched:
                    declaration = declarations.get(rematched[0].capability_id)
                    capability_id = rematched[0].capability_id

            subtask = SubTask(
                description=description,
                retrieval_query=description,
                domain=declaration.domain if declaration else "",
                expert_id=declaration.expert_id if declaration else "",
                capability_id=capability_id if declaration else "",
                entities=[
                    EntityRef(entity_type=e, source="router")
                    for e in classification.entities
                ],
                confidence=0.0,
                rationale="planned by the orchestrator",
                is_fallback=declaration is None,
            )
            if declaration is None:
                unmatched += 1
                subtask.rationale = (
                    "planned by the orchestrator; no declared capability covers it"
                )
            else:
                scored = self.matcher.match(
                    description,
                    domain_scores=classification.scores,
                    capabilities=[declaration],
                    top_k=1,
                )
                subtask.confidence = scored[0].score if scored else 0.5

            id_map[str(raw.get("id", f"s{len(subtasks) + 1}"))] = subtask.subtask_id
            subtasks.append(subtask)

        # Translate the planner's own ids into ours.
        for raw, subtask in zip(parsed["subtasks"][: len(subtasks)], subtasks, strict=False):
            for dependency in raw.get("depends_on", []) or []:
                mapped = id_map.get(str(dependency))
                if mapped and mapped != subtask.subtask_id:
                    subtask.depends_on.append(mapped)

        _break_cycles(subtasks)
        return Decomposition(
            subtasks,
            "planner",
            str(parsed.get("reason", "decomposed by the orchestrator")),
            unmatched=unmatched,
        )


def _break_cycles(subtasks: list[SubTask]) -> None:
    """Remove only the dependency edges that actually close a cycle.

    A positional heuristic ("a dependency on a later subtask is a cycle") is
    wrong here: the capability path emits subtasks ordered by match score, not
    by execution order, so a legitimate producer→consumer edge would be
    discarded whenever the consumer happened to match more strongly. That is
    exactly the dependency the ontology exists to derive.

    So this does real cycle detection and drops the back edge it finds. A cycle
    is a planning error, and dropping one edge lets the plan run rather than
    failing the whole request; the DAG validates again before execution.
    """
    known = {st.subtask_id for st in subtasks}
    by_id = {st.subtask_id: st for st in subtasks}

    for subtask in subtasks:
        subtask.depends_on = list(
            dict.fromkeys(
                dependency
                for dependency in subtask.depends_on
                if dependency in known and dependency != subtask.subtask_id
            )
        )

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(known, WHITE)

    def visit(node_id: str) -> None:
        colour[node_id] = GREY
        for dependency in list(by_id[node_id].depends_on):
            if colour[dependency] == GREY:
                logger.debug(
                    "Dropping cyclic dependency %s → %s", dependency, node_id
                )
                by_id[node_id].depends_on.remove(dependency)
                continue
            if colour[dependency] == WHITE:
                visit(dependency)
        colour[node_id] = BLACK

    for subtask_id in list(known):
        if colour[subtask_id] == WHITE:
            visit(subtask_id)
