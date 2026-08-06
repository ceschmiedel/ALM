"""§11 · the controlled experiment, its graders and its verdict."""

from __future__ import annotations

import json

import pytest

from alm.core.errors import EvaluationError
from alm.evaluation.dataset import EvalCase, EvalDataset, load_datasets
from alm.evaluation.graders import grade, grade_keywords, grade_numeric
from alm.evaluation.harness import EvaluationHarness
from alm.evaluation.metrics import (
    ArmMetrics,
    CaseResult,
    aggregate,
    compare,
    compute_consistency,
    percentile,
    render_comparison,
)


def _case(**overrides) -> EvalCase:
    defaults = dict(id="c1", domain="legal", question="q?", expected="a")
    defaults.update(overrides)
    return EvalCase(**defaults)


def _result(**overrides) -> CaseResult:
    defaults = dict(case_id="c", domain="legal", score=1.0, correct=True, latency_ms=10.0)
    defaults.update(overrides)
    return CaseResult(**defaults)


# -- datasets ----------------------------------------------------------------


def test_dataset_loads_jsonl(tmp_path):
    path = tmp_path / "legal.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"question": "q1", "expected": "a1", "keywords": ["a1"]},
                {"question": "q2", "expected": "a2", "keywords": ["a2"]},
            ]
        ),
        encoding="utf-8",
    )
    dataset = EvalDataset.from_jsonl(path)
    assert len(dataset) == 2
    assert dataset.cases[0].id == "legal-001"
    assert dataset.cases[0].domain == "legal"


def test_dataset_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        '# a comment\n\n{"question": "q", "expected": "a"}\n', encoding="utf-8"
    )
    assert len(EvalDataset.from_jsonl(path)) == 1


def test_dataset_reports_the_offending_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(EvaluationError, match="line"):
        EvalDataset.from_jsonl(path)


def test_missing_dataset_raises():
    with pytest.raises(EvaluationError, match="not found"):
        EvalDataset.from_jsonl("/nonexistent/file.jsonl")


def test_datasets_merge_and_filter(tmp_path):
    for name, domain in (("legal", "legal"), ("finance", "finance")):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps({"question": "q", "expected": "a", "domain": domain}) + "\n",
            encoding="utf-8",
        )
    merged = load_datasets([tmp_path / "legal.jsonl", tmp_path / "finance.jsonl"])
    assert len(merged) == 2
    assert set(merged.domains()) == {"legal", "finance"}
    assert len(merged.filter_domain("legal")) == 1


# -- graders -----------------------------------------------------------------


def test_keyword_grader_scores_by_ratio():
    case = _case(keywords=["twelve", "fees"], min_keyword_ratio=0.5)
    both = grade_keywords(case, "capped at twelve months of fees")
    assert both.correct and both.score == 1.0

    one = grade_keywords(case, "capped at twelve months")
    assert one.score == 0.5 and one.correct

    none = grade_keywords(case, "nothing relevant")
    assert not none.correct


def test_keyword_grader_ignores_accents_and_case():
    case = _case(keywords=["Notificação"], min_keyword_ratio=1.0)
    assert grade_keywords(case, "a notificacao foi enviada").correct


def test_numeric_grader_respects_tolerance():
    case = _case(grader="numeric", expected_number=1200000, tolerance=0.01)
    assert grade_numeric(case, "the exposure is EUR 1200000").correct
    assert not grade_numeric(case, "the exposure is EUR 900000").correct
    assert not grade_numeric(case, "no numbers here").correct


async def test_grade_dispatches_by_case_type():
    exact = await grade(_case(grader="exact", expected="hello"), "Hello")
    assert exact.correct

    contains = await grade(_case(grader="contains", expected="twelve"), "it is twelve months")
    assert contains.correct


async def test_rubric_grader_degrades_without_a_model():
    result = await grade(_case(grader="rubric", keywords=["a"]), "a", serving=None)
    assert "orchestrator model" in result.reason


# -- metrics -----------------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 100.0
    assert percentile([], 0.95) == 0.0


def test_consistency_penalises_spread():
    uniform = [_result(score=0.8) for _ in range(5)]
    erratic = [_result(score=s) for s in (0.0, 1.0, 0.0, 1.0, 0.0)]
    assert compute_consistency(uniform) > compute_consistency(erratic)


def test_aggregate_computes_the_six_metrics():
    results = [
        _result(correct=True, latency_ms=10, cost_usd=0.001, trace_complete=True),
        _result(correct=False, score=0.0, latency_ms=100, cost_usd=0.001, used_fallback=True),
    ]
    metrics = aggregate("federation", results)
    assert metrics.cases == 2
    assert metrics.domain_accuracy == 0.5
    assert metrics.fallback_rate == 0.5
    assert metrics.auditability == 0.5
    assert metrics.cost_per_query == pytest.approx(0.001)
    assert metrics.latency_p95_ms >= 10


def test_comparison_declares_for_the_federation_when_it_wins():
    federation = ArmMetrics(
        arm="federation",
        cases=10,
        domain_accuracy=0.9,
        cost_per_query=0.0001,
        latency_p95_ms=100,
        consistency=0.9,
        auditability=1.0,
    )
    baseline = ArmMetrics(
        arm="baseline",
        cases=10,
        domain_accuracy=0.8,
        cost_per_query=0.003,
        latency_p95_ms=300,
        consistency=0.85,
        auditability=0.0,
    )
    comparison = compare(federation, baseline)
    assert comparison.federation_wins
    assert comparison.verdict == "federation"


def test_comparison_declares_for_the_baseline_when_accuracy_falls():
    """The harness must be willing to return a verdict against the federation."""
    federation = ArmMetrics(
        arm="federation",
        cases=10,
        domain_accuracy=0.6,
        cost_per_query=0.0,
        latency_p95_ms=50,
        consistency=0.9,
        auditability=1.0,
    )
    baseline = ArmMetrics(
        arm="baseline",
        cases=10,
        domain_accuracy=0.9,
        cost_per_query=0.01,
        latency_p95_ms=400,
        consistency=0.8,
        auditability=0.0,
    )
    comparison = compare(federation, baseline)
    assert not comparison.federation_wins
    assert comparison.verdict == "baseline"
    assert "RAG over the shared model" in comparison.recommendation


def test_comparison_can_be_inconclusive():
    federation = ArmMetrics(
        arm="federation",
        cases=10,
        domain_accuracy=0.9,
        cost_per_query=0.05,
        latency_p95_ms=900,
        consistency=0.4,
        auditability=1.0,
    )
    baseline = ArmMetrics(
        arm="baseline",
        cases=10,
        domain_accuracy=0.85,
        cost_per_query=0.001,
        latency_p95_ms=100,
        consistency=0.9,
        auditability=0.0,
    )
    assert compare(federation, baseline).verdict == "inconclusive"


def test_render_comparison_includes_the_verdict():
    federation = ArmMetrics(arm="federation", cases=5, domain_accuracy=0.9, auditability=1.0)
    baseline = ArmMetrics(arm="baseline", cases=5, domain_accuracy=0.8)
    rendered = render_comparison(compare(federation, baseline), domain="legal")
    assert "VERDICT" in rendered
    assert "domain accuracy" in rendered
    assert "auditability" in rendered


# -- harness -----------------------------------------------------------------


async def test_harness_runs_both_arms(runtime, tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "question": "What are the payment terms?",
                    "expected": "net 30",
                    "keywords": ["net 30"],
                    "min_keyword_ratio": 1.0,
                    "domain": "legal",
                },
                {
                    "question": "What does clause 7.2 cap liability at?",
                    "expected": "twelve months of fees",
                    "keywords": ["twelve"],
                    "min_keyword_ratio": 1.0,
                    "domain": "legal",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvalDataset.from_jsonl(path)
    harness = EvaluationHarness(runtime, pack_name="demo")
    federation, baseline, comparison = await harness.run_comparison(dataset)

    assert federation.metrics.cases == 2
    assert baseline.metrics.cases == 2
    # Auditability is where the federation wins by construction.
    assert federation.metrics.auditability > baseline.metrics.auditability
    assert comparison.verdict in {"federation", "baseline", "inconclusive"}
    await runtime.close()


async def test_harness_persists_runs(runtime, tmp_path):
    from alm.evaluation import list_runs

    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"question": "What are the payment terms?", "expected": "net 30"}) + "\n",
        encoding="utf-8",
    )
    await EvaluationHarness(runtime).run_federation(EvalDataset.from_jsonl(path))

    runs = list_runs(runtime.tenant_id)
    assert runs and runs[0]["arm"] == "federation"
    await runtime.close()
