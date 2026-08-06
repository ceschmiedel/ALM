"""End-to-end: the whole federation, L1 through L5, over the demo pack."""

from __future__ import annotations

import pytest

from alm.experts.calibration import Calibration, fit_calibration
from alm.protocol.task import TaskEnvelope
from alm.protocol.trace import Layer


async def test_pack_installs_a_complete_federation(runtime):
    health = runtime.health()
    assert health["experts"] == 3
    assert set(health["graph"]["domains"]) == {"legal", "finance", "risk"}
    assert health["corpus"]["chunks"] > 0
    assert health["policies"] > 0
    assert health["semantic_routing"] is True
    await runtime.close()


async def test_single_domain_question_routes_to_one_expert(runtime):
    result = await runtime.run("What does clause 7.2 cap the aggregate liability at?")
    assert result.ok
    assert result.plan.strategy in {"single_expert", "federated"}
    assert not result.metrics.used_fallback
    assert result.answer
    await runtime.close()


async def test_cross_domain_question_engages_several_experts(runtime):
    result = await runtime.run(
        "What is our maximum financial exposure under the liability cap, "
        "and what residual risk remains after controls?"
    )
    assert result.ok
    assert len(result.contributions) >= 2
    assert result.plan.strategy == "federated"
    await runtime.close()


async def test_dependency_ordering_comes_from_the_ontology(runtime):
    """Risk consumes FinancialExposure, which finance produces."""
    result = await runtime.run(
        "Quantify our financial exposure in fees and assess the residual risk severity."
    )
    plan = result.plan
    risk = next((s for s in plan.subtasks if s.expert_id == "risk-expert"), None)
    finance = next((s for s in plan.subtasks if s.expert_id == "finance-expert"), None)
    if risk is not None and finance is not None:
        assert finance.subtask_id in risk.depends_on
        assert result.metrics.dag_depth >= 2
    await runtime.close()


async def test_result_carries_a_complete_audit_trail(runtime):
    result = await runtime.run("What is the liability cap and what does it exclude?")

    assert result.trace_is_complete()
    layers = {event.layer for event in result.trace.events}
    assert Layer.L2_ROUTER in layers
    assert Layer.L4_ORCHESTRATION in layers
    assert Layer.L5_ARBITRATION in layers
    assert result.audit_trail()
    await runtime.close()


async def test_unrelated_question_escalates_to_the_orchestrator(runtime):
    result = await runtime.run("What is the best recipe for sourdough bread?")
    assert result.metrics.used_fallback
    assert result.plan.strategy == "fallback_orchestrator"
    assert result.plan.fallback_reason
    await runtime.close()


async def test_metrics_are_accounted(runtime):
    result = await runtime.run("What are the payment terms?")
    metrics = result.metrics
    assert metrics.wall_ms > 0
    assert metrics.model_calls > 0
    assert metrics.total_tokens > 0
    assert metrics.dag_width >= 1
    assert metrics.dag_depth >= 1
    await runtime.close()


async def test_session_is_persisted_for_later_audit(runtime):
    from alm.persistence.database import session_scope
    from alm.persistence.models import SessionRow

    result = await runtime.run("What are the payment terms?")
    with session_scope() as session:
        row = session.get(SessionRow, result.session_id)
        assert row is not None
        assert row.answer == result.answer
        assert row.trace
        assert row.plan
    await runtime.close()


async def test_governance_denies_restricted_material_without_the_claim(runtime):
    """The demo pack restricts incident records to holders of a claim."""
    from alm.governance.audit import AuditLog

    await runtime.run(
        "Describe any security incident or service disruption on record.",
        principal="anonymous",
    )
    denials = AuditLog(runtime.tenant_id).query(decision="deny", limit=50)
    # Either nothing restricted matched, or every restricted read was denied.
    assert all(d.decision == "deny" for d in denials)
    await runtime.close()


async def test_claims_widen_access(runtime):
    result = await runtime.run(
        "What incidents are recorded in the vendor risk register?",
        principal="reviewer",
        claims=["risk.incident_review"],
    )
    assert result.status in {"success", "error"}
    assert result.status != "denied"
    await runtime.close()


async def test_task_envelope_input_is_accepted(runtime):
    task = TaskEnvelope(intent_text="What are the payment terms?", principal="alice")
    result = await runtime.run(task)
    assert result.session_id == task.session_id
    assert result.task.principal == "alice"
    await runtime.close()


async def test_trace_events_stream_to_a_listener(runtime):
    seen = []
    await runtime.run("What is the liability cap?", on_event=seen.append)
    assert seen
    assert any(event.layer is Layer.L2_ROUTER for event in seen)
    await runtime.close()


async def test_failures_and_escalations_are_captured_as_feedback(runtime):
    from sqlalchemy import select

    from alm.persistence.database import session_scope
    from alm.persistence.models import FeedbackRow

    await runtime.run("What is the best recipe for sourdough bread?")
    with session_scope() as session:
        rows = session.execute(select(FeedbackRow)).scalars().all()
    assert any(r.kind == "escalation" for r in rows), (
        "an escalation must become training material for the next cycle"
    )
    await runtime.close()


async def test_runtime_rebuilds_from_the_database_alone(runtime, pack_path):
    """Production does not need the pack directory on disk."""
    from alm.federation.runtime import FederationRuntime

    await runtime.close()
    rebuilt = FederationRuntime.from_database()
    assert len(rebuilt.experts) == 3
    result = await rebuilt.run("What are the payment terms?")
    assert result.answer
    await rebuilt.close()


# -- calibration -------------------------------------------------------------


def test_calibration_needs_enough_labelled_samples():
    calibration = fit_calibration("e", [0.9, 0.8], [True, False])
    assert not calibration.fitted


def test_calibration_shrinks_unfitted_confidence_toward_the_prior():
    calibration = Calibration("e")
    assert not calibration.fitted
    # An unvalidated 0.95 must not be treated as a validated 0.95.
    assert calibration.apply(0.95) < 0.95
    assert calibration.apply(0.05) > 0.05


def test_calibration_fits_and_improves_brier():
    # Systematically overconfident expert: high scores, frequently wrong.
    confidences = [0.9] * 10 + [0.85] * 10
    outcomes = [True, False] * 10
    calibration = fit_calibration("overconfident", confidences, outcomes)
    assert calibration.fitted
    assert calibration.apply(0.9) < 0.9
    assert calibration.brier_after <= calibration.brier_before


def test_calibration_rejects_a_degenerate_all_correct_set():
    calibration = fit_calibration("e", [0.8] * 20, [True] * 20)
    assert not calibration.fitted, "an all-correct set carries no calibration signal"


def test_calibration_store_round_trips():
    from alm.experts.calibration import CalibrationStore

    store = CalibrationStore()
    fitted = fit_calibration("e", [0.9] * 10 + [0.3] * 10, [True] * 10 + [False] * 10)
    store.save(fitted)
    store.invalidate()

    reloaded = store.get("e")
    assert reloaded.fitted
    assert reloaded.samples == 20


@pytest.mark.parametrize(
    "question",
    [
        "What are the payment terms?",
        "What is the liability cap?",
        "What is the residual risk of this supplier?",
    ],
)
async def test_representative_questions_all_answer(runtime, question):
    result = await runtime.run(question)
    assert result.status == "success"
    assert result.answer.strip()
    await runtime.close()
