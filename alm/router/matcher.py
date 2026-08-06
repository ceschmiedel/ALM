"""Capability matching — the graph as the router.

The tempting implementation is imperative: a chain of ``if`` statements mapping
phrases to experts.  It does not scale and it cannot be defended — every new
expert edits the router, and nobody can say why a request went where it went.

The pattern used here is declarative.  Capabilities and their relationships to
ontology entity types live as nodes and edges in the Context Graph, and the
router *queries* them.  Adding an Expert Agent becomes adding a node.  That
property is what turns routing into a proprietary, evolving asset instead of a
thicket of conditionals.

Three complementary signals are combined, because each fails differently:

* **semantic** — embedding similarity, finds paraphrase, blind to identifiers;
* **lexical** — token overlap with the declaration's vocabulary and examples,
  catches the domain jargon embeddings smear together;
* **entity** — ontology entity types mentioned in the request against the ones
  the capability declares it operates on. This is the signal that is actually
  *about the domain* rather than about the words.

Every score is reported separately in the trace, so a bad route can be
diagnosed instead of guessed at.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence

from alm.graph.models import CapabilityDeclaration, CapabilityMatch, Node, NodeKind
from alm.graph.store import ContextGraph

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]{3,}")

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "you", "your", "our", "its", "their", "which",
    "what", "when", "where", "who", "how", "why", "should", "would", "could",
    "can", "any", "all", "some", "such", "than", "then", "there", "here",
    "into", "about", "does", "did", "will", "shall", "must", "under", "over",
    "que", "para", "com", "uma", "dos", "das", "por", "como", "mais", "nao",
    "não", "sobre", "entre", "quando", "onde", "qual", "quais", "seu", "sua",
    "esse", "essa", "isso", "pelo", "pela", "está", "esta", "são",
}


def tokenise(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS}


class CapabilityMatcher:
    """Scores a request against every capability declared in the graph."""

    #: Weights of the three signals.  Entity match is weighted below the others
    #: because it is high-precision but low-recall: many valid requests never
    #: name an entity type explicitly.
    WEIGHT_SEMANTIC = 0.45
    WEIGHT_LEXICAL = 0.35
    WEIGHT_ENTITY = 0.20

    def __init__(self, graph: ContextGraph) -> None:
        self.graph = graph
        self._entity_index: dict[str, list[str]] | None = None

    # -- entity detection --------------------------------------------------

    def _entity_terms(self) -> dict[str, list[str]]:
        """Map each ontology entity type to the surface forms that signal it."""
        if self._entity_index is None:
            index: dict[str, list[str]] = {}
            for node in self.graph.find_nodes(NodeKind.ENTITY_TYPE):
                terms = [node.key.lower(), (node.label or "").lower()]
                terms.extend(str(k).lower() for k in node.attributes.get("keywords", []))
                # Split CamelCase so "EmploymentContract" also matches "contract".
                terms.extend(
                    part.lower()
                    for part in re.findall(r"[A-Z][a-z]+|[a-z]+", node.key)
                    if len(part) > 3
                )
                index[node.key] = sorted({t for t in terms if t and len(t) > 2})
            self._entity_index = index
        return self._entity_index

    def detect_entities(self, text: str) -> list[str]:
        """Ontology entity types the request appears to be about."""
        lowered = (text or "").lower()
        found: list[str] = []
        for entity, terms in self._entity_terms().items():
            if any(re.search(rf"\b{re.escape(term)}s?\b", lowered) for term in terms):
                found.append(entity)
        return found

    def invalidate(self) -> None:
        """Drop cached ontology terms after the graph changes."""
        self._entity_index = None

    # -- matching ----------------------------------------------------------

    def match(
        self,
        text: str,
        *,
        domain_scores: dict[str, float] | None = None,
        entities: Sequence[str] | None = None,
        capabilities: Sequence[CapabilityDeclaration] | None = None,
        top_k: int = 8,
        min_score: float = 0.0,
    ) -> list[CapabilityMatch]:
        """Rank capabilities for a request, best first."""
        declarations = list(
            capabilities if capabilities is not None else self.graph.capabilities()
        )
        if not declarations:
            return []

        detected = list(entities) if entities is not None else self.detect_entities(text)
        query_tokens = tokenise(text)
        semantic = self._semantic_scores(text, declarations)

        matches: list[CapabilityMatch] = []
        for declaration in declarations:
            lexical = self._lexical_score(query_tokens, declaration)
            entity = self._entity_score(detected, declaration)
            semantic_score = semantic.get(declaration.capability_id, 0.0)

            base = self._blend(
                semantic=semantic_score if semantic else None,
                lexical=lexical,
                entity=entity if detected else None,
            )

            # A domain the classifier considers irrelevant should not win on a
            # coincidental word match; a domain it favours gets a modest lift.
            domain_weight = 1.0
            if domain_scores:
                domain_weight = 0.6 + 0.4 * domain_scores.get(declaration.domain, 0.0)

            authority = self.graph.authority_weight(
                declaration.expert_id, declaration.domain
            )
            score = base * domain_weight

            if score < min_score:
                continue

            matches.append(
                CapabilityMatch(
                    capability_id=declaration.capability_id,
                    expert_id=declaration.expert_id,
                    domain=declaration.domain,
                    score=round(min(score, 1.0), 6),
                    semantic_score=round(semantic_score, 6),
                    lexical_score=round(lexical, 6),
                    entity_score=round(entity, 6),
                    authority=authority,
                    reason=_describe(semantic_score, lexical, entity, detected, declaration),
                )
            )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    # -- signal blending ---------------------------------------------------

    @classmethod
    def _blend(
        cls,
        *,
        semantic: float | None,
        lexical: float,
        entity: float | None,
    ) -> float:
        """Combine only the signals that are actually available.

        A missing signal is normalised away rather than scored as zero.  The
        distinction matters: most requests never name an ontology entity
        explicitly, and treating that silence as evidence *against* every
        capability would push the whole federation below the escalation
        threshold and turn the fallback from a safety net into the default path.

        Agreement is then rewarded. Two independent signals both pointing at the
        same capability is much stronger evidence than one signal pointing twice
        as hard, and a linear sum cannot express that.
        """
        contributions: list[tuple[float, float]] = [(cls.WEIGHT_LEXICAL, lexical)]
        if semantic is not None:
            contributions.append((cls.WEIGHT_SEMANTIC, semantic))
        if entity is not None:
            contributions.append((cls.WEIGHT_ENTITY, entity))

        total_weight = sum(weight for weight, _ in contributions)
        if total_weight <= 0:
            return 0.0
        blended = sum(weight * value for weight, value in contributions) / total_weight

        strong = sum(1 for _, value in contributions if value >= 0.4)
        if strong >= 2:
            # Saturating boost: moves a corroborated match toward 1.0 without
            # ever exceeding it.
            blended += 0.25 * (1.0 - blended)
        return min(blended, 1.0)

    # -- individual signals ------------------------------------------------

    def _semantic_scores(
        self, text: str, declarations: Sequence[CapabilityDeclaration]
    ) -> dict[str, float]:
        """Cosine similarity per capability, or empty when no embedder is set.

        Negative cosine is clamped to zero rather than rescaled onto ``[0,1]``.
        Rescaling would hand every unrelated capability a baseline of 0.5 —
        which is not a weak match, it is no match, and the difference is exactly
        what the router needs to see.
        """
        if not self.graph.has_embedder:
            return {}
        ranked = self.graph.semantic_search(
            text, kinds=[NodeKind.CAPABILITY], top_k=max(len(declarations), 1)
        )
        if not ranked:
            return {}
        return {node.key: max(0.0, score) for node, score in ranked}

    @staticmethod
    def _lexical_score(query_tokens: set[str], declaration: CapabilityDeclaration) -> float:
        """Overlap with the declaration's vocabulary, examples weighted highest.

        Examples are the strongest lexical evidence available: they are written
        in the phrasing real requests use, which a description rarely is.
        """
        if not query_tokens:
            return 0.0

        keyword_tokens = tokenise(" ".join(declaration.keywords))
        example_tokens = tokenise(" ".join(declaration.examples))
        description_tokens = tokenise(
            f"{declaration.capability_id.replace('_', ' ')} {declaration.description}"
        )

        def overlap(tokens: set[str]) -> float:
            """Length-independent overlap.

            Dividing by the query length alone would punish long questions: a
            fifteen-word request that hits five of a capability's keywords is
            *strong* evidence, but scores 0.33 under pure coverage. Real
            requests in a regulated domain are long, so that bias would push the
            federation into fallback exactly where it should be most confident.

            Taking the better of coverage and a saturating count keeps short
            queries honest while letting accumulated evidence speak.
            """
            if not tokens or not query_tokens:
                return 0.0
            hits = len(query_tokens & tokens)
            if not hits:
                return 0.0
            coverage = hits / len(query_tokens)
            saturating = 1.0 - math.exp(-hits / 2.5)
            return max(coverage, saturating)

        score = (
            0.45 * overlap(example_tokens)
            + 0.35 * overlap(keyword_tokens)
            + 0.20 * overlap(description_tokens)
        )
        # Reward density: matching 3 of 4 query terms against a tight keyword
        # list is stronger evidence than matching 3 of 4 against a long essay.
        if keyword_tokens and (query_tokens & keyword_tokens):
            score += 0.15 * (len(query_tokens & keyword_tokens) / len(keyword_tokens))
        return min(score, 1.0)

    @staticmethod
    def _entity_score(detected: Sequence[str], declaration: CapabilityDeclaration) -> float:
        """How well this capability's scope lines up with what the request is about.

        Two readings, and the more favourable one wins: the share of the
        *request's* entities this capability covers, and the share of the
        *capability's* declared scope the request touches. Using only the second
        would punish a capability for declaring a broad scope, which is the
        opposite of the incentive the ontology should create.
        """
        wanted = set(declaration.operates_on)
        present = set(detected)
        if not wanted or not present:
            return 0.0
        overlap = len(wanted & present)
        if not overlap:
            return 0.0
        return max(overlap / len(present), overlap / len(wanted))

    # -- graph-derived dependencies ---------------------------------------

    def declared_dependencies(self, capability_id: str) -> list[str]:
        """Capabilities this one declares an explicit graph dependency on."""
        node = self.graph.get_node(NodeKind.CAPABILITY, capability_id)
        if node is None:
            return []
        from alm.graph.models import Relation

        dependencies: list[str] = []
        for edge in self.graph.edges_from(node.node_id, Relation.DEPENDS_ON):
            target = self.graph.get_node_by_id(edge.target_id)
            if target is not None and target.kind == NodeKind.CAPABILITY:
                dependencies.append(target.key)
        return dependencies

    def infer_dependency(
        self, producer: CapabilityDeclaration, consumer: CapabilityDeclaration
    ) -> bool:
        """True when ``consumer`` reads an entity type ``producer`` writes.

        This is how a plan acquires order without anybody writing the order
        down: a risk assessment that operates on ``FinancialExposure`` depends
        on the finance capability that produces it.
        """
        if producer.capability_id == consumer.capability_id:
            return False
        produced = set(producer.produces)
        return bool(produced and produced & set(consumer.operates_on))


def _describe(
    semantic: float,
    lexical: float,
    entity: float,
    detected: Sequence[str],
    declaration: CapabilityDeclaration,
) -> str:
    """One line explaining why this capability scored as it did."""
    parts: list[str] = []
    if semantic >= 0.6:
        parts.append("semantically close to the capability description")
    if lexical >= 0.25:
        parts.append("shares vocabulary with its keywords/examples")
    shared = set(detected) & set(declaration.operates_on)
    if shared:
        parts.append(f"request mentions {', '.join(sorted(shared))}")
    if not parts:
        parts.append("weak match on all signals")
    return "; ".join(parts)


def domain_nodes(graph: ContextGraph) -> list[Node]:
    return graph.find_nodes(NodeKind.DOMAIN)
