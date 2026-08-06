"""L1 · envelopes, task objects and the trace."""

from __future__ import annotations

import pytest

from alm.protocol.answer import Citation, ExpertAnswer
from alm.protocol.envelope import (
    IntentPayload,
    LIPEnvelope,
    MessageType,
    SenderInfo,
    SenderKind,
    build_envelope,
    parse_payload,
)
from alm.protocol.lip import (
    complete_envelope,
    intent_envelope,
    task_from_execute,
    task_from_intent,
    upstream_context,
)
from alm.protocol.task import EntityRef, ExecutionPlan, SubTask, TaskEnvelope
from alm.protocol.trace import ExecutionTrace, Layer


def test_envelope_roundtrips_through_json():
    envelope = build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id="req-1"),
        "ses-1",
        IntentPayload(intent_text="hello"),
    )
    restored = LIPEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored.message_type is MessageType.INTENT
    assert restored.session_id == "ses-1"
    assert parse_payload(restored).intent_text == "hello"


def test_task_from_intent_carries_governance_identity():
    envelope = build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id="req-1", oidc_subject="alice@acme"),
        "ses-2",
        IntentPayload(
            intent_text="review the contract",
            context={"tenant_id": "acme", "entities": [{"entity_type": "Contract"}]},
            ibac_claims_requested=["legal.read"],
        ),
    )
    task = task_from_intent(envelope)
    assert task.principal == "alice@acme"
    assert task.tenant_id == "acme"
    assert task.claims == ["legal.read"]
    assert task.entities[0].entity_type == "Contract"
    assert task.source == "lip"


def test_task_from_intent_rejects_wrong_message_type():
    envelope = build_envelope(
        MessageType.OFFER,
        SenderInfo(kind=SenderKind.AGENT, id="a"),
        "ses",
        {"capability_id": "x"},
    )
    with pytest.raises(ValueError, match="expected an intent"):
        task_from_intent(envelope)


def test_task_from_execute_requires_intent_text():
    envelope = build_envelope(
        MessageType.EXECUTE,
        SenderInfo(kind=SenderKind.COORDINATOR, id="coord"),
        "ses",
        {"execution_plan": {}},
    )
    with pytest.raises(ValueError, match="no intent text"):
        task_from_execute(envelope)


def test_empty_intent_is_rejected():
    with pytest.raises(ValueError):
        TaskEnvelope(intent_text="   ")


def test_subtask_query_prefers_retrieval_query_over_framing():
    subtask = SubTask(
        description="Analyse a contractual clause, as it applies to: what is the cap?",
        retrieval_query="what is the cap?",
    )
    assert subtask.query() == "what is the cap?"


def test_subtask_query_falls_back_to_description():
    assert SubTask(description="find the cap").query() == "find the cap"


def test_plan_reports_dangling_and_self_dependencies():
    first = SubTask(description="a")
    second = SubTask(description="b", depends_on=[first.subtask_id, "ghost"])
    third = SubTask(description="c")
    third.depends_on = [third.subtask_id]

    plan = ExecutionPlan(subtasks=[first, second, third])
    problems = plan.validate_dependencies()
    assert any("unknown subtask" in p for p in problems)
    assert any("depends on itself" in p for p in problems)


def test_plan_expert_ids_are_deduplicated_in_order():
    plan = ExecutionPlan(
        subtasks=[
            SubTask(description="a", expert_id="x"),
            SubTask(description="b", expert_id="y"),
            SubTask(description="c", expert_id="x"),
        ]
    )
    assert plan.expert_ids == ["x", "y"]


def test_upstream_context_preserves_provenance_not_prose():
    answers = [
        ExpertAnswer(
            expert_id="legal",
            domain="legal",
            content="Clause 7.2 caps liability.",
            confidence=0.8,
            entities=[EntityRef(entity_type="Clause")],
        ),
        ExpertAnswer(expert_id="broken", status="error", error="boom"),
    ]
    payload = upstream_context(answers)
    assert len(payload["upstream"]) == 1, "failed answers must not be passed downstream"
    item = payload["upstream"][0]
    assert item["expert_id"] == "legal"
    assert item["confidence"] == 0.8
    assert item["entities"][0]["entity_type"] == "Clause"


def test_trace_emits_notifies_and_renders():
    trace = ExecutionTrace(session_id="ses")
    seen = []
    unsubscribe = trace.subscribe(seen.append)

    trace.emit(Layer.L2_ROUTER, "classified", detail={"domains": ["legal"]})
    assert len(seen) == 1

    unsubscribe()
    trace.emit(Layer.L5_ARBITRATION, "resolved")
    assert len(seen) == 1, "unsubscribed listener must stop receiving events"

    assert len(trace.events) == 2
    assert "classified" in trace.render()
    assert len(trace.by_layer(Layer.L2_ROUTER)) == 1


def test_broken_listener_does_not_break_the_run():
    trace = ExecutionTrace(session_id="ses")

    def explode(event):
        raise RuntimeError("listener failure")

    trace.subscribe(explode)
    trace.emit(Layer.L1_INTERFACE, "still recorded")
    assert len(trace.events) == 1


def test_complete_envelope_places_answer_first():
    envelope = complete_envelope(
        agent_id="alm",
        session_id="ses",
        answer="the answer",
        artifacts=[{"type": "expert_answer"}],
        metadata={"confidence": 0.9},
    )
    assert envelope.message_type is MessageType.COMPLETE
    assert envelope.payload["artifacts"][0]["content"] == "the answer"
    assert envelope.payload["metadata"]["confidence"] == 0.9


def test_intent_envelope_serialises_entities():
    task = TaskEnvelope(
        intent_text="review",
        entities=[EntityRef(entity_type="Contract", key="acme")],
        tenant_id="acme",
    )
    envelope = intent_envelope(task)
    assert envelope.payload["context"]["entities"][0]["entity_type"] == "Contract"
    assert envelope.payload["context"]["tenant_id"] == "acme"


def test_citation_defaults_are_safe():
    citation = Citation()
    assert citation.score == 0.0
    assert citation.entities == []
