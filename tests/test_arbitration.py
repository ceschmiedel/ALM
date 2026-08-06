"""L5 · conflict detection, the strategy chain, synthesis and verification."""

from __future__ import annotations

import pytest

from alm.arbitration.arbiter import Arbiter
from alm.arbitration.conflict import ConflictDetector, ConflictKind
from alm.arbitration.strategies import (
    AuthorityStrategy,
    ConfidenceStrategy,
    VerifierStrategy,
    build_strategies,
)
from alm.arbitration.synthesis import Synthesizer
from alm.experts.spec import VerifierRule
from alm.experts.verifier import AnswerVerifier
from alm.graph.models import CapabilityDeclaration
from alm.protocol.answer import Citation, ExpertAnswer


def _answer(expert_id: str, **overrides) -> ExpertAnswer:
    defaults = dict(
        expert_id=expert_id,
        domain=overrides.pop("domain", expert_id),
        content=f"finding from {expert_id}",
        confidence=0.7,
        calibrated=True,
        status="success",
    )
    defaults.update(overrides)
    return ExpertAnswer(**defaults)


# -- conflict detection ------------------------------------------------------


def test_numeric_disagreement_is_detected():
    detector = ConflictDetector()
    conflicts = detector.detect(
        [
            _answer("legal", structured={"liability_cap_months": 12}),
            _answer("finance", structured={"liability_cap_months": 24}),
        ]
    )
    assert len(conflicts) == 1
    assert conflicts[0].kind is ConflictKind.NUMERIC_DISAGREEMENT
    assert conflicts[0].field == "liability_cap_months"


def test_numbers_within_tolerance_do_not_conflict():
    detector = ConflictDetector(numeric_tolerance=0.05)
    conflicts = detector.detect(
        [
            _answer("a", structured={"amount": 1000}),
            _answer("b", structured={"amount": 1020}),
        ]
    )
    assert conflicts == []


def test_boolean_disagreement_is_a_verdict_conflict():
    conflicts = ConflictDetector().detect(
        [
            _answer("a", structured={"enforceable": True}),
            _answer("b", structured={"enforceable": False}),
        ]
    )
    assert conflicts[0].kind is ConflictKind.VERDICT_DISAGREEMENT


def test_prose_fields_that_merely_differ_are_not_conflicts():
    """Regression: an embedded number used to turn two paragraphs into a conflict."""
    conflicts = ConflictDetector().detect(
        [
            _answer("legal", structured={"finding": "Clause 7.2 caps liability at 12 months."}),
            _answer("finance", structured={"finding": "Fees under clause 3 total EUR 1.2m."}),
        ]
    )
    assert conflicts == [], "differently-worded narrative fields are not a disagreement"


def test_verdict_vocabulary_fields_do_conflict_on_strings():
    conflicts = ConflictDetector().detect(
        [
            _answer("a", structured={"severity": "high"}),
            _answer("b", structured={"severity": "low"}),
        ]
    )
    assert conflicts and conflicts[0].field == "severity"


def test_complementary_lists_are_a_merge_not_a_conflict():
    conflicts = ConflictDetector().detect(
        [
            _answer("a", structured={"controls": ["backup"]}),
            _answer("b", structured={"controls": ["insurance"]}),
        ]
    )
    assert conflicts == []


def test_single_answer_cannot_conflict():
    assert ConflictDetector().detect([_answer("solo")]) == []


def test_failed_answers_are_excluded_from_detection():
    conflicts = ConflictDetector().detect(
        [
            _answer("a", structured={"x": 1}),
            _answer("b", structured={"x": 99}, status="error"),
        ]
    )
    assert conflicts == []


# -- strategies --------------------------------------------------------------


async def test_verifier_strategy_favours_the_answer_that_passes():
    from alm.experts.verifier import VerificationResult

    good = _answer("good", structured={"x": 1})
    bad = _answer("bad", structured={"x": 2})
    conflict = ConflictDetector().detect([good, bad])[0]

    verifications = {
        good.answer_id: VerificationResult(passed=True),
        bad.answer_id: VerificationResult(
            passed=False,
            violations=[
                {"rule_id": "r", "rule_type": "required_fields", "message": "missing"}
            ],
        ),
    }
    resolution = await VerifierStrategy().resolve(
        conflict,
        {good.answer_id: good, bad.answer_id: bad},
        context={"verifications": verifications},
    )
    assert resolution is not None
    assert resolution.winner_expert == "good"
    assert resolution.strategy == "verifier"


async def test_confidence_strategy_refuses_uncalibrated_scores():
    """An unvalidated confidence must not be allowed to win an argument."""
    high = _answer("high", confidence=0.95, calibrated=False, structured={"x": 1})
    low = _answer("low", confidence=0.4, calibrated=True, structured={"x": 2})
    conflict = ConflictDetector().detect([high, low])[0]

    resolution = await ConfidenceStrategy(margin=0.1).resolve(
        conflict, {high.answer_id: high, low.answer_id: low}, context={}
    )
    assert resolution is None


async def test_confidence_strategy_needs_a_real_margin():
    a = _answer("a", confidence=0.72, structured={"x": 1})
    b = _answer("b", confidence=0.70, structured={"x": 2})
    conflict = ConflictDetector().detect([a, b])[0]

    assert (
        await ConfidenceStrategy(margin=0.15).resolve(
            conflict, {a.answer_id: a, b.answer_id: b}, context={}
        )
        is None
    )


async def test_confidence_strategy_decides_on_a_clear_margin():
    a = _answer("a", confidence=0.9, structured={"x": 1})
    b = _answer("b", confidence=0.5, structured={"x": 2})
    conflict = ConflictDetector().detect([a, b])[0]

    resolution = await ConfidenceStrategy(margin=0.15).resolve(
        conflict, {a.answer_id: a, b.answer_id: b}, context={}
    )
    assert resolution is not None and resolution.winner_expert == "a"


async def test_authority_strategy_defers_to_the_primary_domain(graph):
    graph.register_domain("legal")
    graph.register_domain("finance")
    graph.register_expert("legal-expert", domain="legal", authority={"legal": 1.0})
    graph.register_expert(
        "finance-expert", domain="finance", authority={"finance": 1.0, "legal": 0.3}
    )

    legal = _answer("legal-expert", domain="legal", structured={"x": 1})
    finance = _answer("finance-expert", domain="finance", structured={"x": 2})
    conflict = ConflictDetector().detect([legal, finance])[0]

    resolution = await AuthorityStrategy(graph).resolve(
        conflict,
        {legal.answer_id: legal, finance.answer_id: finance},
        context={"primary_domain": "legal"},
    )
    assert resolution is not None
    assert resolution.winner_expert == "legal-expert"
    assert "authority" in resolution.reason


def test_build_strategies_skips_the_judge_without_serving(graph):
    strategies = build_strategies(
        ["verifier", "confidence", "authority", "judge"], graph=graph, serving=None
    )
    assert [s.name for s in strategies] == ["verifier", "confidence", "authority"]


def test_build_strategies_ignores_unknown_names(graph):
    strategies = build_strategies(["verifier", "telepathy"], graph=graph, serving=None)
    assert [s.name for s in strategies] == ["verifier"]


# -- the arbiter -------------------------------------------------------------


async def test_arbiter_records_contributions_and_supersession(graph):
    graph.register_domain("legal")
    graph.register_domain("finance")
    graph.register_expert("legal-expert", domain="legal", authority={"legal": 1.0})
    graph.register_expert(
        "finance-expert", domain="finance", authority={"finance": 1.0, "legal": 0.2}
    )

    legal = _answer("legal-expert", domain="legal", structured={"months": 12})
    finance = _answer("finance-expert", domain="finance", structured={"months": 24})

    arbiter = Arbiter(build_strategies(["authority"], graph=graph, serving=None))
    result = await arbiter.arbitrate(
        [legal, finance], intent_text="how long?", primary_domain="legal"
    )

    assert result.had_conflict
    assert result.fully_resolved
    assert legal.answer_id in result.accepted_answer_ids
    assert finance.answer_id in result.superseded_answer_ids

    superseded = next(c for c in result.contributions if not c.accepted)
    assert superseded.superseded_by == "legal-expert"
    assert "authority" in superseded.reason
    assert result.audit_lines()


async def test_unresolvable_conflict_is_reported_not_hidden(graph):
    a = _answer("a", confidence=0.7, structured={"x": 1})
    b = _answer("b", confidence=0.7, structured={"x": 2})

    arbiter = Arbiter([ConfidenceStrategy(margin=0.5)])
    result = await arbiter.arbitrate([a, b], intent_text="q")

    assert result.had_conflict
    assert not result.fully_resolved
    assert any("UNRESOLVED" in line for line in result.audit_lines())


async def test_no_conflict_keeps_every_contribution():
    arbiter = Arbiter([])
    result = await arbiter.arbitrate([_answer("a"), _answer("b")], intent_text="q")
    assert not result.had_conflict
    assert len(result.accepted_answer_ids) == 2


# -- synthesis ---------------------------------------------------------------


async def test_single_accepted_answer_passes_through_unchanged():
    answer = _answer("solo", content="the definitive finding", citations=[Citation(title="src")])
    arbitration = await Arbiter([]).arbitrate([answer], intent_text="q")

    synthesis = await Synthesizer(None).synthesize(
        [answer], arbitration, intent_text="q"
    )
    assert synthesis.method == "single_expert"
    assert synthesis.answer == "the definitive finding"


async def test_deterministic_synthesis_without_a_model():
    answers = [_answer("a", content="legal view"), _answer("b", content="finance view")]
    arbitration = await Arbiter([]).arbitrate(answers, intent_text="q")

    synthesis = await Synthesizer(None).synthesize(answers, arbitration, intent_text="q")
    assert synthesis.method == "deterministic"
    assert "legal view" in synthesis.answer
    assert "finance view" in synthesis.answer
    assert set(synthesis.contributing_experts) == {"a", "b"}


async def test_synthesis_reports_failure_honestly():
    failed = [_answer("a", status="error", error="backend down")]
    arbitration = await Arbiter([]).arbitrate(failed, intent_text="q")

    synthesis = await Synthesizer(None).synthesize(failed, arbitration, intent_text="q")
    assert synthesis.method == "none"
    assert synthesis.confidence == 0.0
    assert "could not produce an answer" in synthesis.answer


async def test_unresolved_conflict_reduces_confidence():
    a = _answer("a", confidence=0.8, structured={"x": 1})
    b = _answer("b", confidence=0.8, structured={"x": 2})
    arbitration = await Arbiter([ConfidenceStrategy(margin=0.9)]).arbitrate([a, b], intent_text="q")

    synthesis = await Synthesizer(None).synthesize([a, b], arbitration, intent_text="q")
    assert synthesis.confidence < 0.8


async def test_citations_are_merged_and_deduplicated():
    shared = Citation(chunk_id="c1", title="doc", score=0.4)
    better = Citation(chunk_id="c1", title="doc", score=0.9)
    answers = [
        _answer("a", citations=[shared]),
        _answer("b", citations=[better, Citation(chunk_id="c2", title="other", score=0.5)]),
    ]
    arbitration = await Arbiter([]).arbitrate(answers, intent_text="q")
    synthesis = await Synthesizer(None).synthesize(answers, arbitration, intent_text="q")

    ids = [c.chunk_id for c in synthesis.citations]
    assert ids == ["c1", "c2"]
    assert synthesis.citations[0].score == 0.9


# -- verifier ----------------------------------------------------------------


def test_required_fields_rule():
    verifier = AnswerVerifier([VerifierRule(id="r", type="required_fields", fields=["finding"])])
    assert not verifier.verify(_answer("a", structured={})).passed
    assert verifier.verify(_answer("a", structured={"finding": "x"})).passed


def test_requires_citation_rule():
    verifier = AnswerVerifier([VerifierRule(id="r", type="requires_citation")])
    assert not verifier.verify(_answer("a")).passed
    assert verifier.verify(_answer("a", citations=[Citation(title="src")])).passed


def test_numeric_range_rule():
    verifier = AnswerVerifier(
        [VerifierRule(id="r", type="numeric_range", fields=["months"], minimum=0, maximum=120)]
    )
    assert verifier.verify(_answer("a", structured={"months": 24})).passed
    assert not verifier.verify(_answer("a", structured={"months": 500})).passed


def test_numeric_range_ignores_prose():
    """Regression: stripping digits from prose used to produce phantom values."""
    verifier = AnswerVerifier(
        [VerifierRule(id="r", type="numeric_range", fields=["finding"], maximum=10)]
    )
    result = verifier.verify(
        _answer("a", structured={"finding": "Clause 7.2 of the 2025 agreement applies."})
    )
    assert result.passed


def test_numeric_consistency_rule():
    verifier = AnswerVerifier(
        [
            VerifierRule(
                id="r", type="numeric_consistency", fields=["total", "a", "b"], tolerance=0.01
            )
        ]
    )
    assert verifier.verify(_answer("x", structured={"total": 30, "a": 10, "b": 20})).passed
    assert not verifier.verify(_answer("x", structured={"total": 99, "a": 10, "b": 20})).passed


def test_regex_rule_enforces_a_vocabulary():
    verifier = AnswerVerifier(
        [
            VerifierRule(
                id="r", type="regex_match", fields=["severity"], pattern="^(low|high)$"
            )
        ]
    )
    assert verifier.verify(_answer("a", structured={"severity": "high"})).passed
    assert not verifier.verify(_answer("a", structured={"severity": "quite bad"})).passed


def test_forbidden_terms_rule():
    verifier = AnswerVerifier([VerifierRule(id="r", type="forbidden_terms", terms=["guarantee"])])
    assert not verifier.verify(_answer("a", content="we guarantee full recovery")).passed


def test_unknown_rule_type_warns_rather_than_failing():
    verifier = AnswerVerifier([VerifierRule(id="r", type="telepathy")])
    result = verifier.verify(_answer("a"))
    assert result.passed
    assert result.warnings


def test_failed_execution_fails_verification():
    verifier = AnswerVerifier([])
    result = verifier.verify(_answer("a", status="error", error="boom"))
    assert not result.passed


@pytest.mark.parametrize("value,expected", [(0.9, True), (0.2, False)])
def test_min_confidence_rule(value, expected):
    verifier = AnswerVerifier([VerifierRule(id="r", type="min_confidence", minimum=0.5)])
    assert verifier.verify(_answer("a", confidence=value)).passed is expected


def test_capability_declaration_embedding_text_includes_examples():
    declaration = CapabilityDeclaration(
        capability_id="analyze_clause",
        description="Analyse clauses",
        examples=["Is clause 7.2 enforceable?"],
        keywords=["liability"],
    )
    text = declaration.text_for_embedding()
    assert "enforceable" in text
    assert "liability" in text
