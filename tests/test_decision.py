"""The decision contract, its generative backend, and the four layers that use it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alm.arbitration.conflict import Conflict, ConflictKind
from alm.arbitration.strategies import JudgeStrategy
from alm.decision import Choice, DecisionEngine, DecisionRequest, Noul, Score
from alm.decision.generative import parse_answers, render_system, render_user
from alm.evaluation.dataset import EvalCase
from alm.evaluation.graders import grade_rubric
from alm.governance.ibac import IBACEngine
from alm.governance.policies import Effect, EvaluationPoint, Policy
from alm.graph.models import Node, NodeKind
from alm.models.serving import ServingRouter
from alm.models.spec import ModelSpec
from alm.models.tiers import ModelTier
from alm.protocol.answer import ExpertAnswer
from alm.router.classifier import IntentClassifier
from alm.router.matcher import CapabilityMatcher


def _serving(registry, response, *, tier=ModelTier.ORCHESTRATOR) -> ServingRouter:
    """A serving router whose only model returns ``response`` as its JSON."""
    registry.register(
        ModelSpec(
            model_id="scripted",
            tier=tier,
            backend="scripted",
            params={"responses": [{"match": ".*", "response": response}]},
        )
    )
    return ServingRouter(registry)


def _request(**questions) -> DecisionRequest:
    return DecisionRequest(state="the state", questions=questions)


_COLOUR = Choice(instructions="Which colour?", options={"red": "warm", "blue": "cold"})
_COLOURS = Choice(instructions="Which colours?", options={"red": "", "blue": ""}, multi=True)
_QUALITY = Score(instructions="How good?", levels=["bad", "fair", "good"])
_TRUE = Noul(instructions="It is true.")


# -- contract ----------------------------------------------------------------


def test_requests_reject_empty_questions_options_and_scales():
    with pytest.raises(ValidationError):
        DecisionRequest(state="x", questions={})
    with pytest.raises(ValidationError):
        Choice(instructions="?", options={})
    with pytest.raises(ValidationError):
        Score(instructions="?", levels=["only one"])


def test_prompt_lists_every_question_with_its_labels_and_format():
    request = DecisionRequest(
        state={"ticket": 7},
        questions={"colour": _COLOUR, "quality": _QUALITY, "ok": _TRUE},
        preamble="You are a judge.",
    )
    system = render_system(request)
    assert system.startswith("You are a judge.")
    assert '"colour": {"choice"' in system
    assert '"score": <0..2>' in system
    assert '"ok": {"probability"' in system

    user = render_user(request)
    assert '"ticket": 7' in user
    assert "- red: warm" in user
    assert "- 2: good" in user


# -- parsing -----------------------------------------------------------------


def test_choice_keeps_only_known_labels():
    answers = parse_answers(
        _request(q=_COLOUR), {"q": {"choice": "red", "confidence": 0.9, "reason": "warm"}}
    )
    assert answers["q"].choice == "red"
    assert answers["q"].selected == ["red"]
    assert answers["q"].confidence == 0.9
    assert answers["q"].reason == "warm"


def test_unknown_choice_is_empty_and_carries_no_confidence():
    answers = parse_answers(_request(q=_COLOUR), {"q": {"choice": "green", "confidence": 0.99}})
    assert answers["q"].choice == ""
    assert answers["q"].confidence == 0.0


def test_multi_choice_preserves_order_and_drops_unknowns_and_duplicates():
    answers = parse_answers(
        _request(q=_COLOURS),
        {"q": {"selected": ["blue", "green", "red", "blue"], "confidence": 0.7}},
    )
    assert answers["q"].selected == ["blue", "red"]
    assert answers["q"].choice == "blue"


def test_score_is_clamped_and_normalised():
    answers = parse_answers(_request(q=_QUALITY), {"q": {"score": 1}})
    assert answers["q"].score == 1.0
    assert answers["q"].normalised == 0.5
    assert parse_answers(_request(q=_QUALITY), {"q": {"score": 9}})["q"].normalised == 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [({"probability": 0.8}, 0.8), (True, 1.0), ("no", 0.0), (0.3, 0.3), ({"probability": 7}, 1.0)],
)
def test_noul_accepts_numbers_booleans_and_words(raw, expected):
    assert parse_answers(_request(q=_TRUE), {"q": raw})["q"].probability == expected


def test_a_single_question_may_be_answered_unwrapped():
    answers = parse_answers(_request(q=_COLOUR), {"choice": "blue", "confidence": 0.6})
    assert answers["q"].choice == "blue"


def test_missing_and_malformed_answers_become_empty_defaults():
    answers = parse_answers(
        _request(a=_COLOUR, b=_QUALITY, c=_TRUE), {"a": None, "b": {"score": "high"}}
    )
    assert answers["a"].choice == ""
    assert answers["b"].score == 0.0
    assert answers["c"].probability == 0.0


# -- engine ------------------------------------------------------------------


async def test_engine_answers_several_questions_in_one_call(registry):
    serving = _serving(
        registry,
        {
            "colour": {"choice": "blue", "confidence": 0.8, "reason": "cold"},
            "quality": {"score": 2, "confidence": 0.7},
            "ok": {"probability": 0.9},
        },
    )
    result = await DecisionEngine(serving).decide(
        _request(colour=_COLOUR, quality=_QUALITY, ok=_TRUE), tier=ModelTier.ORCHESTRATOR
    )
    assert result.parsed
    assert result.method == "generative"
    assert result.calibrated is False
    assert result.choice("colour").choice == "blue"
    assert result.score("quality").normalised == 1.0
    assert result.noul("ok").value is True


async def test_engine_marks_unparseable_output(registry):
    serving = _serving(registry, "I would rather write prose.")
    result = await DecisionEngine(serving).decide(_request(q=_COLOUR), tier="orchestrator")
    # The scripted backend wraps bare prose as {"answer": ...}: parseable JSON
    # that answers nothing, which must read as an empty judgment.
    assert result.choice("q").choice == ""
    assert result.choice("q").confidence == 0.0


# -- the four call sites -----------------------------------------------------


def _domains(graph) -> None:
    graph.upsert_node(Node(kind=NodeKind.DOMAIN, key="legal", description="contracts"))
    graph.upsert_node(Node(kind=NodeKind.DOMAIN, key="finance", description="money"))


async def test_classifier_reads_a_ranked_domain_selection(graph, registry):
    _domains(graph)
    serving = _serving(
        registry,
        {"domains": {"selected": ["finance", "hr", "legal"], "confidence": 0.82, "reason": "both"}},
        tier=ModelTier.MICRO_SLM,
    )
    classifier = IntentClassifier(graph, CapabilityMatcher(graph), serving)
    result = await classifier.classify_with_model("what is the exposure under the contract?")
    assert result is not None
    assert result.domains == ["finance", "legal"]
    assert result.scores == {"finance": 0.82, "legal": 0.82}
    assert result.confidence == 0.82
    assert result.method == "micro_slm"
    assert result.reason == "both"


async def test_classifier_matching_no_domain_has_zero_confidence(graph, registry):
    _domains(graph)
    serving = _serving(
        registry, {"domains": {"selected": [], "confidence": 0.9}}, tier=ModelTier.MICRO_SLM
    )
    classifier = IntentClassifier(graph, CapabilityMatcher(graph), serving)
    result = await classifier.classify_with_model("what is the weather?")
    assert result is not None
    assert result.domains == []
    assert result.confidence == 0.0


def _conflict_and_answers():
    answers = {
        "a1": ExpertAnswer(expert_id="legal-expert", domain="legal", content="cap is 12 months"),
        "a2": ExpertAnswer(expert_id="finance-expert", domain="finance", content="cap is 24 months"),
    }
    for answer_id, answer in answers.items():
        answer.answer_id = answer_id
    conflict = Conflict(
        kind=ConflictKind.FIELD_DISAGREEMENT, participants=["a1", "a2"], description="cap length"
    )
    return conflict, answers


async def test_judge_resolves_to_the_named_expert(registry):
    serving = _serving(
        registry, {"winner": {"choice": "finance-expert", "confidence": 0.8, "reason": "cites it"}}
    )
    conflict, answers = _conflict_and_answers()
    resolution = await JudgeStrategy(serving).resolve(conflict, answers, context={})
    assert resolution is not None
    assert resolution.winner_expert == "finance-expert"
    assert resolution.losers == ["legal-expert"]
    assert resolution.reason == "cites it"


async def test_judge_leaves_the_conflict_open_on_an_unknown_expert(registry):
    serving = _serving(registry, {"winner": {"choice": "nobody", "confidence": 0.9}})
    conflict, answers = _conflict_and_answers()
    assert await JudgeStrategy(serving).resolve(conflict, answers, context={}) is None


async def test_rubric_grader_scores_on_the_scale_and_asks_correctness_separately(registry):
    serving = _serving(
        registry,
        {"quality": {"score": 3, "reason": "minor omission"}, "correct": {"probability": 0.9}},
    )
    case = EvalCase(id="c", question="q?", expected="reference", grader="rubric")
    result = await grade_rubric(case, "an answer", serving=serving)
    assert result.score == 0.75
    assert result.correct is True
    assert result.reason == "minor omission"
    assert result.detail["model_graded"] is True


def _ibac_with_deny_policy() -> IBACEngine:
    engine = IBACEngine()
    engine.policies.add(
        Policy(
            id="no-salaries",
            description="Individual salaries must never be disclosed.",
            effect=Effect.DENY,
            evaluation_points=[EvaluationPoint.INTENT_ADMISSION],
            # No selector an intent can match: only the semantic pass can apply it.
            resources=["entity:salary"],
        )
    )
    return engine


async def test_semantic_ibac_denies_and_names_the_policy(registry):
    serving = _serving(
        registry,
        {
            "violates": {"probability": 0.95, "reason": "asks for one person's pay"},
            "policies": {"selected": ["no-salaries", "invented"], "confidence": 0.9},
        },
    )
    decision = await _ibac_with_deny_policy().evaluate_intent_semantic(
        intent_text="how much does Ana earn?", serving=serving
    )
    assert decision.allowed is False
    assert decision.matched_policies == ["no-salaries"]
    assert decision.reason == "asks for one person's pay"


async def test_semantic_ibac_keeps_the_deterministic_allow_when_not_violated(registry):
    serving = _serving(
        registry, {"violates": {"probability": 0.1}, "policies": {"selected": []}}
    )
    decision = await _ibac_with_deny_policy().evaluate_intent_semantic(
        intent_text="what is the headcount?", serving=serving
    )
    assert decision.allowed is True
