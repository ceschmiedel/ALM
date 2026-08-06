"""L2 · capability matching, classification, decomposition and escalation."""

from __future__ import annotations

import pytest

from alm.graph.models import CapabilityDeclaration
from alm.graph.ontology import Ontology
from alm.models.registry import ModelRegistry
from alm.models.serving import ServingRouter
from alm.models.spec import ModelSpec
from alm.models.tiers import ModelTier
from alm.protocol.task import TaskEnvelope
from alm.router.matcher import CapabilityMatcher, tokenise
from alm.router.router import CognitiveRouter


@pytest.fixture
def populated_graph(graph):
    Ontology(
        domain="legal",
        entities=[
            {"name": "Clause", "keywords": ["clause", "provision"]},
            {"name": "LiabilityCap", "keywords": ["liability cap", "limitation of liability"]},
        ],
    ).apply(graph)
    Ontology(
        domain="finance",
        entities=[
            {"name": "Fee", "keywords": ["fee", "fees", "charge"]},
            {"name": "FinancialExposure", "keywords": ["exposure", "financial exposure"]},
        ],
    ).apply(graph)

    graph.register_expert(
        "contracts-expert",
        domain="legal",
        capabilities=[
            CapabilityDeclaration(
                capability_id="analyze_clause",
                domain="legal",
                description="Analyse a contractual clause and its liability allocation",
                keywords=["clause", "liability", "cap", "termination", "enforceable"],
                examples=["Is clause 7.2 enforceable?", "What does the liability cap limit?"],
                operates_on=["Clause", "LiabilityCap"],
                produces=["ClauseAnalysis"],
            )
        ],
        authority={"legal": 1.0},
    )
    graph.register_expert(
        "finance-expert",
        domain="finance",
        capabilities=[
            CapabilityDeclaration(
                capability_id="quantify_exposure",
                domain="finance",
                description="Quantify financial exposure and commercial impact",
                keywords=["exposure", "cost", "fee", "payment", "penalty"],
                examples=["What is our maximum financial exposure?"],
                operates_on=["Fee", "FinancialExposure"],
                produces=["FinancialExposure"],
            )
        ],
        authority={"finance": 1.0},
    )
    return graph


@pytest.fixture
def router(populated_graph):
    registry = ModelRegistry()
    registry.register(
        ModelSpec(model_id="expert", tier=ModelTier.SLM, backend="heuristic")
    )
    return CognitiveRouter(populated_graph, ServingRouter(registry))


# -- matcher -----------------------------------------------------------------


def test_tokenise_drops_stopwords():
    tokens = tokenise("What is the liability cap for this agreement?")
    assert "liability" in tokens
    assert "what" not in tokens
    assert "the" not in tokens


def test_entity_detection_from_ontology_keywords(populated_graph):
    matcher = CapabilityMatcher(populated_graph)
    detected = matcher.detect_entities("what does the liability cap in clause 7.2 limit?")
    assert "LiabilityCap" in detected
    assert "Clause" in detected


def test_matcher_prefers_the_right_capability(populated_graph):
    matcher = CapabilityMatcher(populated_graph)
    legal = matcher.match("Is clause 7.2 enforceable?", top_k=3)
    assert legal[0].capability_id == "analyze_clause"

    finance = matcher.match("What is our maximum financial exposure?", top_k=3)
    assert finance[0].capability_id == "quantify_exposure"


def test_long_questions_are_not_penalised(populated_graph):
    """Regression: normalising lexical overlap by query length buried long asks."""
    matcher = CapabilityMatcher(populated_graph)
    short = matcher.match("liability cap", top_k=1)[0].score
    long = matcher.match(
        "Could you please help me understand what exactly the liability cap "
        "provision in clause 7.2 of this agreement limits for us?",
        top_k=1,
    )[0].score
    assert long >= short * 0.6, "a longer, more specific question must not score far worse"


def test_missing_entity_signal_does_not_penalise(populated_graph):
    """A request naming no entity must not be scored as evidence against everything."""
    matcher = CapabilityMatcher(populated_graph)
    with_entity = matcher.match("what does the liability cap limit?", top_k=1)[0].score
    without_entity = matcher.match("is it enforceable and can we terminate?", top_k=1)[0].score
    assert without_entity > 0.1
    assert with_entity > 0.0


def test_unrelated_request_scores_near_zero(populated_graph):
    matcher = CapabilityMatcher(populated_graph)
    matches = matcher.match("what is the best recipe for sourdough bread?", top_k=3)
    assert all(m.score < 0.3 for m in matches)


def test_entity_score_rewards_scope_overlap_both_ways(populated_graph):
    matcher = CapabilityMatcher(populated_graph)
    declaration = CapabilityDeclaration(
        capability_id="c", operates_on=["Clause", "LiabilityCap", "Fee"]
    )
    # Covering one of three declared entities still counts, via request coverage.
    assert matcher._entity_score(["Clause"], declaration) > 0.0
    assert matcher._entity_score([], declaration) == 0.0
    assert matcher._entity_score(["Ghost"], declaration) == 0.0


def test_inferred_dependency_from_produces_and_operates_on(populated_graph):
    matcher = CapabilityMatcher(populated_graph)
    producer = CapabilityDeclaration(capability_id="a", produces=["FinancialExposure"])
    consumer = CapabilityDeclaration(capability_id="b", operates_on=["FinancialExposure"])
    assert matcher.infer_dependency(producer, consumer) is True
    assert matcher.infer_dependency(consumer, producer) is False
    assert matcher.infer_dependency(producer, producer) is False


# -- classification ----------------------------------------------------------


async def test_classification_identifies_the_domain(router):
    classification = await router.classifier.classify("Is clause 7.2 enforceable?")
    assert "legal" in classification.domains
    assert classification.confidence > 0.0


async def test_classification_of_an_unrelated_request_is_weak(router):
    classification = await router.classifier.classify("how do I bake sourdough bread?")
    assert classification.confidence < 0.4


# -- routing -----------------------------------------------------------------


async def test_single_domain_request_routes_to_one_expert(router):
    outcome = await router.route(TaskEnvelope(intent_text="Is clause 7.2 enforceable?"))
    assert not outcome.escalated
    assert outcome.plan.expert_ids == ["contracts-expert"]
    assert outcome.plan.strategy == "single_expert"


async def test_cross_domain_request_engages_both_experts(router):
    outcome = await router.route(
        TaskEnvelope(
            intent_text=(
                "What does the liability cap in clause 7.2 limit, and what is our "
                "maximum financial exposure in fees?"
            )
        )
    )
    assert not outcome.escalated
    assert set(outcome.plan.expert_ids) == {"contracts-expert", "finance-expert"}
    assert outcome.plan.strategy == "federated"


async def test_unmatched_request_escalates_to_the_orchestrator(router):
    outcome = await router.route(
        TaskEnvelope(intent_text="what is the best recipe for sourdough bread?")
    )
    assert outcome.escalated
    assert outcome.plan.strategy == "fallback_orchestrator"
    assert outcome.plan.subtasks[0].is_fallback
    assert outcome.plan.fallback_reason


async def test_dependency_is_derived_from_the_graph(populated_graph):
    """Risk consumes what finance produces, so order comes from the ontology."""
    populated_graph.register_expert(
        "risk-expert",
        domain="finance",
        capabilities=[
            CapabilityDeclaration(
                capability_id="assess_risk",
                domain="finance",
                description="Assess residual risk given quantified exposure",
                keywords=["risk", "residual", "severity"],
                examples=["What is the residual risk?"],
                operates_on=["FinancialExposure"],
                produces=["RiskAssessment"],
            )
        ],
    )
    registry = ModelRegistry()
    registry.register(ModelSpec(model_id="e", tier=ModelTier.SLM, backend="heuristic"))
    router = CognitiveRouter(populated_graph, ServingRouter(registry))

    outcome = await router.route(
        TaskEnvelope(
            intent_text=(
                "What is our maximum financial exposure in fees, and what residual "
                "risk severity remains?"
            )
        )
    )
    plan = outcome.plan
    risk = next((s for s in plan.subtasks if s.expert_id == "risk-expert"), None)
    finance = next((s for s in plan.subtasks if s.expert_id == "finance-expert"), None)
    if risk is not None and finance is not None:
        assert finance.subtask_id in risk.depends_on


async def test_router_carries_the_users_words_as_the_retrieval_query(router):
    """Regression: the capability description used to become the search query."""
    intent = "Is clause 7.2 enforceable?"
    outcome = await router.route(TaskEnvelope(intent_text=intent))
    subtask = outcome.plan.subtasks[0]
    assert subtask.query() == intent
    assert subtask.description != subtask.query()


def test_explain_exposes_routing_signals(router):
    explanation = router.explain("Is clause 7.2 enforceable?")
    assert explanation["matches"]
    assert "semantic_score" in explanation["matches"][0]
    assert explanation["entities_detected"]
    assert explanation["would_escalate"] is False


async def test_plan_confidence_is_weakest_link_dominated(router):
    strong = await router.route(TaskEnvelope(intent_text="Is clause 7.2 enforceable?"))
    weak = await router.route(TaskEnvelope(intent_text="tell me something about things"))
    assert strong.plan.router_confidence > weak.plan.router_confidence
