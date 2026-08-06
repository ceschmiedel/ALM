"""MLOps · versioning, eval-gated promotion, rollback, drift and triggers."""

from __future__ import annotations

import pytest

from alm.core.errors import PromotionBlockedError
from alm.distillation.pipeline import DistillationPipeline
from alm.mlops.drift import DriftDetector, assess_promotion
from alm.mlops.versions import VersionRegistry, summarise
from alm.models.spec import ModelSpec


@pytest.fixture
def versions(registry) -> VersionRegistry:
    registry.register(ModelSpec(model_id="legal-lora", backend="heuristic"))
    return VersionRegistry()


# -- versioning --------------------------------------------------------------


def test_record_is_idempotent(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.8})
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.85})
    history = versions.versions("legal-lora")
    assert len(history) == 1
    assert history[0].metrics["domain_accuracy"] == 0.85


def test_promotion_requires_a_recorded_evaluation(versions):
    versions.record("legal-lora", "1")
    with pytest.raises(PromotionBlockedError, match="no recorded"):
        versions.promote("legal-lora", "1")


def test_promotion_succeeds_with_an_evaluation(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.8})
    promoted = versions.promote("legal-lora", "1")
    assert promoted.status == "active"
    assert versions.active("legal-lora").version == "1"


def test_promotion_is_refused_when_it_would_regress(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.9})
    versions.promote("legal-lora", "1")
    versions.record("legal-lora", "2", metrics={"domain_accuracy": 0.7})

    with pytest.raises(PromotionBlockedError, match="regression"):
        versions.promote("legal-lora", "2")


def test_force_bypasses_the_gate_explicitly(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.9})
    versions.promote("legal-lora", "1")
    versions.record("legal-lora", "2")
    promoted = versions.promote("legal-lora", "2", force=True)
    assert promoted.status == "active"


def test_promotion_retires_the_incumbent(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.8})
    versions.promote("legal-lora", "1")
    versions.record("legal-lora", "2", metrics={"domain_accuracy": 0.9})
    versions.promote("legal-lora", "2")

    history = {v.version: v.status for v in versions.versions("legal-lora")}
    assert history["1"] == "retired"
    assert history["2"] == "active"


def test_rollback_reactivates_the_previous_version(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.8})
    versions.promote("legal-lora", "1")
    versions.record("legal-lora", "2", metrics={"domain_accuracy": 0.9})
    versions.promote("legal-lora", "2")

    rolled = versions.rollback("legal-lora")
    assert rolled.version == "1"
    assert versions.active("legal-lora").version == "1"


def test_rollback_without_history_is_refused(versions):
    with pytest.raises(PromotionBlockedError):
        versions.rollback("legal-lora")


def test_promotion_updates_the_model_registry(versions, registry):
    versions.record(
        "legal-lora", "3", metrics={"domain_accuracy": 0.9}, artifact_uri="/adapters/v3"
    )
    versions.promote("legal-lora", "3")
    spec = registry.require("legal-lora")
    assert spec.version == "3"
    assert spec.adapter_uri == "/adapters/v3"


def test_summarise_reports_active_and_candidates(versions):
    versions.record("legal-lora", "1", metrics={"domain_accuracy": 0.8})
    versions.promote("legal-lora", "1")
    versions.record("legal-lora", "2")

    summary = summarise(versions, ["legal-lora"])
    assert summary["legal-lora"]["active"] == "1"
    assert summary["legal-lora"]["candidates"] == ["2"]


# -- drift -------------------------------------------------------------------


def test_drift_needs_history():
    report = DriftDetector().detect("nobody")
    assert not report.drifted
    assert "not enough history" in report.reasons[0]


def test_drift_requires_volume_before_judging():
    from alm.core.ids import utc_now
    from alm.persistence.database import session_scope
    from alm.persistence.models import MetricSnapshotRow

    with session_scope() as session:
        for accuracy in (0.9, 0.5):
            session.add(
                MetricSnapshotRow(
                    tenant_id="default",
                    expert_id="e",
                    window_start=utc_now(),
                    window_end=utc_now(),
                    accuracy=accuracy,
                    sample_count=3,
                )
            )
    report = DriftDetector().detect("e")
    assert not report.drifted
    assert "insufficient volume" in report.reasons[0]


def test_drift_is_flagged_on_a_real_decline():
    from alm.core.ids import utc_now
    from alm.persistence.database import session_scope
    from alm.persistence.models import MetricSnapshotRow

    with session_scope() as session:
        # Oldest first; detect() reads newest-first.
        for accuracy, fallback in ((0.95, 0.05), (0.60, 0.40)):
            session.add(
                MetricSnapshotRow(
                    tenant_id="default",
                    expert_id="drifty",
                    window_start=utc_now(),
                    window_end=utc_now(),
                    accuracy=accuracy,
                    fallback_rate=fallback,
                    sample_count=50,
                )
            )
    report = DriftDetector().detect("drifty")
    assert report.drifted
    assert report.reasons


# -- promotion triggers ------------------------------------------------------


def test_no_trigger_means_stay_on_rag():
    assessment = assess_promotion(expert_id="e", monthly_calls=10)
    assert not assessment.recommend_dedicated
    assert "RAG over the shared model" in assessment.recommendation


def test_sovereignty_alone_justifies_a_dedicated_model():
    assessment = assess_promotion(expert_id="e", sovereignty_required=True, monthly_calls=1)
    assert assessment.recommend_dedicated
    assert "sovereignty" in assessment.confirmed()


def test_volume_trigger():
    assessment = assess_promotion(
        expert_id="e", monthly_calls=100_000, volume_threshold=50_000
    )
    assert "volume" in assessment.confirmed()


def test_latency_trigger_needs_both_numbers():
    unmet = assess_promotion(
        expert_id="e", monthly_calls=1, latency_requirement_ms=200, observed_p95_ms=800
    )
    assert "latency" in unmet.confirmed()

    unknown = assess_promotion(expert_id="e", monthly_calls=1, latency_requirement_ms=200)
    assert "latency" not in unknown.confirmed()


def test_accuracy_trigger_requires_a_measured_ceiling():
    assessment = assess_promotion(
        expert_id="e", monthly_calls=1, rag_accuracy=0.6, accuracy_target=0.85
    )
    assert "accuracy" in assessment.confirmed()
    assert "LoRA adapter" in assessment.recommendation


def test_assessment_renders():
    rendering = assess_promotion(expert_id="e", monthly_calls=1).render()
    assert "Promotion check" in rendering
    assert "sovereignty" in rendering


# -- distillation ------------------------------------------------------------


def test_seed_requires_an_indexed_corpus(runtime):
    from alm.core.errors import DistillationError

    pipeline = DistillationPipeline(
        serving=runtime.serving, chunk_store=runtime.chunk_store
    )
    with pytest.raises(DistillationError, match="no indexed corpus"):
        pipeline.seed("nonexistent-domain")


def test_seed_returns_corpus_passages(runtime):
    pipeline = DistillationPipeline(
        serving=runtime.serving, chunk_store=runtime.chunk_store
    )
    seeds = pipeline.seed("legal")
    assert seeds
    assert "text" in seeds[0]


def test_filter_rejects_bad_examples(runtime):
    from alm.distillation.pipeline import TrainingExample

    pipeline = DistillationPipeline(
        serving=runtime.serving, chunk_store=runtime.chunk_store, dataset="test"
    )
    pipeline._persist(
        [
            TrainingExample(
                prompt="What is the cap?",
                completion="The aggregate liability cap is twelve months of fees paid.",
                domain="legal",
            ),
            TrainingExample(prompt="Short?", completion="yes", domain="legal"),
            TrainingExample(
                prompt="Refuse?",
                completion="I cannot answer that question about the contract terms.",
                domain="legal",
            ),
        ]
    )
    report = pipeline.filter("legal")
    assert report["reviewed"] == 3
    assert report["accepted"] == 1
    assert report["rejected"] == 2
    assert set(report["rejection_reasons"]) <= {
        "answer too short to be informative",
        "teacher refused or hedged instead of answering",
        "duplicate",
    }


def test_export_splits_train_and_eval_deterministically(runtime, tmp_path):
    from alm.distillation.pipeline import TrainingExample

    pipeline = DistillationPipeline(
        serving=runtime.serving, chunk_store=runtime.chunk_store, dataset="test"
    )
    pipeline._persist(
        [
            TrainingExample(
                prompt=f"Question number {i} about the agreement?",
                completion=f"Answer number {i} describing the contractual position clearly.",
                domain="legal",
            )
            for i in range(40)
        ]
    )
    pipeline.filter("legal")

    first = pipeline.export("legal", tmp_path / "legal.jsonl", eval_ratio=0.25)
    assert first["train"] + first["eval"] == 40
    assert first["eval"] > 0

    second = pipeline.export("legal", tmp_path / "legal2.jsonl", eval_ratio=0.25)
    # The split is content-hash based, so it is stable across exports.
    assert second["eval"] == first["eval"]


def test_feedback_becomes_seed_material(runtime):
    from alm.persistence.database import session_scope
    from alm.persistence.models import FeedbackRow

    with session_scope() as session:
        session.add(
            FeedbackRow(
                tenant_id="default",
                session_id="s1",
                domain="legal",
                kind="escalation",
                intent_text="A question the federation had to escalate.",
            )
        )

    pipeline = DistillationPipeline(
        serving=runtime.serving, chunk_store=runtime.chunk_store, dataset="test"
    )
    report = pipeline.ingest_feedback("legal")
    assert report["captured"] == 1
    assert report["by_kind"]["escalation"] == 1

    # Consumed feedback is not re-ingested.
    assert pipeline.ingest_feedback("legal")["captured"] == 0


def test_training_availability_is_reported_honestly():
    from alm.distillation.training import training_available

    assert isinstance(training_available(), bool)
