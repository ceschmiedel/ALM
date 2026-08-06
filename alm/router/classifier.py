"""Intent classification — the cheapest step in the pipeline.

The router's first move is to decide *where a request goes*, not to answer it.
That job belongs on the smallest model in the federation: a micro-SLM, or no
model at all.

Two paths, in order of preference:

* **graph classification** (no model) — score the request against domain and
  capability nodes using embeddings and the ontology vocabulary. Free,
  deterministic, and surprisingly hard to beat when the ontology is good.
* **micro-SLM classification** — a sub-1B model picks domains from a list. Used
  when the graph signal is weak or ambiguous, which is exactly where a model
  earns its cost.

Classification never resolves the task. It only decides which specialists are
worth waking up.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from alm.core.jsonutil import extract_json_object
from alm.graph.models import NodeKind
from alm.graph.store import ContextGraph
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.router.matcher import CapabilityMatcher, tokenise

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """\
You route requests to specialist agents in a domain federation. You do not
answer the request; you only decide which domains it belongs to.

Available domains:
{domains}

Return ONLY a JSON object:
{{"domains": ["<domain>", ...], "confidence": <0..1>, "reason": "<one sentence>"}}

Rules:
- List every domain genuinely involved, most relevant first. Most requests touch one.
- If no listed domain fits, return an empty list and a low confidence. That is a
  useful answer, not a failure — the request will be escalated.
"""


class Classification(BaseModel):
    """Which domains a request belongs to, and how sure the router is."""

    domains: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    method: str = "graph"
    reason: str = ""
    entities: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> str:
        return self.domains[0] if self.domains else ""

    @property
    def is_multi_domain(self) -> bool:
        return len(self.domains) > 1


class IntentClassifier:
    """Maps a request onto one or more domains."""

    def __init__(
        self,
        graph: ContextGraph,
        matcher: CapabilityMatcher,
        serving: ServingRouter | None = None,
    ) -> None:
        self.graph = graph
        self.matcher = matcher
        self.serving = serving

    # -- graph path --------------------------------------------------------

    def classify_with_graph(self, text: str) -> Classification:
        """Score domains from ontology vocabulary, capability matches and embeddings."""
        domains = self.graph.find_nodes(NodeKind.DOMAIN)
        if not domains:
            return Classification(method="graph", reason="no domains registered")

        entities = self.matcher.detect_entities(text)
        query_tokens = tokenise(text)
        scores: dict[str, float] = {d.key: 0.0 for d in domains}

        # 1 · capability matches are the most direct evidence a domain applies.
        for match in self.matcher.match(text, entities=entities, top_k=12):
            scores[match.domain] = max(scores.get(match.domain, 0.0), match.score)

        # 2 · entity types belonging to a domain reinforce it.
        for entity in entities:
            node = self.graph.get_node(NodeKind.ENTITY_TYPE, entity)
            if node is not None and node.domain in scores:
                scores[node.domain] = min(1.0, scores[node.domain] + 0.15)

        # 3 · the domain's own description and label.
        for domain in domains:
            domain_tokens = tokenise(f"{domain.key} {domain.label} {domain.description}")
            if domain_tokens and query_tokens:
                overlap = len(query_tokens & domain_tokens) / len(query_tokens)
                scores[domain.key] = min(1.0, scores[domain.key] + 0.2 * overlap)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0][1] if ranked else 0.0
        if best <= 0.0:
            return Classification(
                scores=scores,
                method="graph",
                confidence=0.0,
                entities=entities,
                reason="no domain matched the request",
            )

        # Keep domains close to the leader; a distant second is noise, not a
        # second domain.
        selected = [name for name, score in ranked if score >= max(best * 0.6, 0.12)]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = _confidence_from_margin(best, runner_up)

        return Classification(
            domains=selected,
            scores={k: round(v, 6) for k, v in scores.items()},
            confidence=confidence,
            method="graph",
            entities=entities,
            reason=(
                f"best domain '{ranked[0][0]}' scored {best:.2f}"
                + (f" against runner-up {runner_up:.2f}" if runner_up else "")
            ),
        )

    # -- model path --------------------------------------------------------

    async def classify_with_model(self, text: str) -> Classification | None:
        """Ask the micro-SLM tier.  Returns ``None`` when no model is available."""
        if self.serving is None:
            return None
        domains = self.graph.find_nodes(NodeKind.DOMAIN)
        if not domains:
            return None

        listing = "\n".join(
            f"- {d.key}: {d.description or d.label}" for d in domains
        )
        request = GenerationRequest(
            messages=[
                Message(role="system", content=_CLASSIFIER_SYSTEM.format(domains=listing)),
                Message(role="user", content=text),
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=200,
        )
        try:
            result = await self.serving.generate(
                request,
                tier=ModelTier.MICRO_SLM,
                component="router.classifier",
                allow_escalation=False,
            )
        except Exception as exc:
            logger.info("Micro-SLM classification unavailable (%s); using graph only", exc)
            return None

        parsed = extract_json_object(result.text)
        if not parsed:
            logger.info("Router model returned unparseable classification; using graph only")
            return None

        known = {d.key for d in domains}
        selected = [str(d) for d in parsed.get("domains", []) if str(d) in known]
        confidence = min(max(float(parsed.get("confidence", 0.0) or 0.0), 0.0), 1.0)

        if not selected:
            # A reported confidence with no domain attached has no referent —
            # confident about *what*? Matching nothing is a low-confidence
            # result by definition, and it must be allowed to escalate.
            confidence = 0.0

        return Classification(
            domains=selected,
            scores={d: confidence for d in selected},
            confidence=confidence,
            method="micro_slm",
            reason=str(parsed.get("reason", "")),
            entities=self.matcher.detect_entities(text),
        )

    # -- combined ----------------------------------------------------------

    async def classify(
        self, text: str, *, model_threshold: float = 0.45
    ) -> Classification:
        """Classify with the graph, escalating to the micro-SLM when unsure.

        The escalation is the point: the cheap path handles the common case, and
        a model is only paid for when the free signal is genuinely ambiguous.
        """
        graph_result = self.classify_with_graph(text)
        if graph_result.confidence >= model_threshold and graph_result.domains:
            return graph_result

        model_result = await self.classify_with_model(text)
        if model_result is None:
            graph_result.reason += " (no router model available to disambiguate)"
            return graph_result
        if not model_result.domains:
            model_result.reason = model_result.reason or "router model matched no domain"
            return model_result

        # Both spoke.  Merge, preferring the model's ordering but keeping any
        # domain the graph was confident about.
        merged = list(model_result.domains)
        for domain in graph_result.domains:
            if domain not in merged and graph_result.scores.get(domain, 0.0) >= 0.5:
                merged.append(domain)
        scores = dict(graph_result.scores)
        for domain in model_result.domains:
            scores[domain] = max(scores.get(domain, 0.0), model_result.confidence)

        return Classification(
            domains=merged,
            scores=scores,
            confidence=max(graph_result.confidence, model_result.confidence),
            method="graph+micro_slm",
            reason=f"graph: {graph_result.reason}; model: {model_result.reason}",
            entities=graph_result.entities or model_result.entities,
        )


def _confidence_from_margin(best: float, runner_up: float) -> float:
    """Confidence from absolute strength, *modulated* by the gap to the runner-up.

    The margin scales the best score rather than being added to it. Adding them
    would let a request that matches nothing well still score confidently just
    because nothing else matched it either — an uncontested weak match is not
    evidence, it is the absence of evidence, and it must escalate.

    Conversely, a strong match that a second capability matches equally well is
    a coin flip, so the margin discounts it.
    """
    if best <= 0.0:
        return 0.0
    margin = max(0.0, (best - runner_up) / best)
    return round(min(1.0, min(best, 1.0) * (0.65 + 0.35 * margin)), 4)


def domain_list(graph: ContextGraph) -> Sequence[str]:
    return [n.key for n in graph.find_nodes(NodeKind.DOMAIN)]
