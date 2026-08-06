"""Conversions between LIP envelopes and ALM task objects.

This is the seam between the bus and the federation.  An ``intent`` arriving
from a LIP coordinator becomes a :class:`TaskEnvelope`; a completed federation
run becomes a ``complete`` message whose artifacts carry the answer *and* the
provenance, so downstream agents inherit the audit trail instead of receiving
an unattributed paragraph.

Context passing between experts uses the same representation.  When step B
consumes step A's output, what crosses the boundary is a structured payload
with entity references and the producing expert's identity — the degradation
you get from threading free text through many hops is what this avoids.
"""

from __future__ import annotations

from typing import Any

from alm.core.telemetry import current_trace_ids
from alm.protocol.answer import ExpertAnswer
from alm.protocol.envelope import (
    CompletePayload,
    EventPayload,
    IntentPayload,
    LIPEnvelope,
    MessageType,
    SenderInfo,
    SenderKind,
    TraceContext,
    build_envelope,
)
from alm.protocol.task import EntityRef, TaskEnvelope
from alm.protocol.trace import TraceEvent


def inject_trace_context() -> TraceContext:
    """Capture the active OTel span into a propagatable trace context."""
    trace_id, span_id = current_trace_ids()
    return TraceContext(trace_id=trace_id, span_id=span_id)


# ---------------------------------------------------------------------------
# Inbound: LIP → ALM
# ---------------------------------------------------------------------------


def task_from_intent(
    envelope: LIPEnvelope,
    *,
    tenant_id: str = "default",
) -> TaskEnvelope:
    """Convert an inbound ``intent`` envelope into a :class:`TaskEnvelope`."""
    if envelope.message_type != MessageType.INTENT:
        raise ValueError(
            f"expected an intent envelope, got {envelope.message_type!r}"
        )
    intent = IntentPayload.model_validate(envelope.payload)
    context = dict(intent.context)

    entities = [
        EntityRef.model_validate(item)
        for item in context.pop("entities", [])
        if isinstance(item, dict)
    ]

    return TaskEnvelope(
        session_id=envelope.session_id or "",
        intent_text=intent.intent_text,
        context=context,
        requested_outputs=list(intent.requested_outputs),
        entities=entities,
        tenant_id=str(context.get("tenant_id", tenant_id)),
        principal=envelope.sender.oidc_subject or envelope.sender.id,
        claims=list(intent.ibac_claims_requested),
        source="lip",
    )


def task_from_execute(
    envelope: LIPEnvelope,
    *,
    tenant_id: str = "default",
) -> TaskEnvelope:
    """Convert an ``execute`` envelope into a task.

    A LIP coordinator authorises execution after negotiation; the intent text
    and context travel inside ``execution_plan``.
    """
    plan = envelope.payload.get("execution_plan", {}) or {}
    context = dict(plan.get("context", {}) or {})
    intent_text = (
        plan.get("intent_text")
        or context.get("intent_text")
        or plan.get("description")
        or ""
    )
    if not intent_text:
        raise ValueError("execute envelope carries no intent text to act on")

    return TaskEnvelope(
        session_id=envelope.session_id or "",
        intent_text=str(intent_text),
        context=context,
        requested_outputs=list(plan.get("requested_outputs", []) or []),
        tenant_id=str(context.get("tenant_id", tenant_id)),
        principal=envelope.sender.oidc_subject or envelope.sender.id,
        claims=list(envelope.payload.get("authorized_scopes", []) or []),
        source="lip",
    )


# ---------------------------------------------------------------------------
# Outbound: ALM → LIP
# ---------------------------------------------------------------------------


def answer_artifact(answer: ExpertAnswer) -> dict[str, Any]:
    """Render one expert answer as a LIP artifact with its provenance."""
    return {
        "type": "expert_answer",
        "expert_id": answer.expert_id,
        "domain": answer.domain,
        "subtask_id": answer.subtask_id,
        "content": answer.content,
        "structured": answer.structured,
        "confidence": answer.confidence,
        "model_id": answer.model_id,
        "tier": answer.tier,
        "citations": [c.model_dump(mode="json") for c in answer.citations],
        "status": answer.status,
    }


def complete_envelope(
    *,
    agent_id: str,
    session_id: str,
    answer: str,
    artifacts: list[dict[str, Any]],
    metadata: dict[str, Any],
    status: str = "success",
) -> LIPEnvelope:
    """Build the ``complete`` message that returns a federation result."""
    payload = CompletePayload(
        status=status,
        artifacts=[{"type": "answer", "content": answer}, *artifacts],
        metadata=metadata,
    )
    return build_envelope(
        MessageType.COMPLETE,
        SenderInfo(kind=SenderKind.AGENT, id=agent_id),
        session_id,
        payload,
        inject_trace_context(),
    )


def event_envelope(agent_id: str, event: TraceEvent) -> LIPEnvelope:
    """Forward an ALM trace event as a LIP ``event`` message."""
    payload = EventPayload(
        category=event.category,
        phase=str(event.layer),
        summary=event.summary,
        detail=event.detail,
        agent_id=event.expert_id or agent_id,
        progress=event.progress,
    )
    return build_envelope(
        MessageType.EVENT,
        SenderInfo(kind=SenderKind.AGENT, id=agent_id),
        event.session_id,
        payload,
        inject_trace_context(),
    )


def intent_envelope(
    task: TaskEnvelope,
    *,
    requester_id: str = "alm",
) -> LIPEnvelope:
    """Publish a task onto a bus as an ``intent`` (federation acting as client)."""
    payload = IntentPayload(
        intent_text=task.intent_text,
        context={
            **task.context,
            "tenant_id": task.tenant_id,
            "entities": [e.model_dump(mode="json") for e in task.entities],
        },
        requested_outputs=list(task.requested_outputs),
        ibac_claims_requested=list(task.claims),
    )
    return build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id=requester_id, oidc_subject=task.principal),
        task.session_id,
        payload,
        inject_trace_context(),
    )


# ---------------------------------------------------------------------------
# Inter-expert context passing
# ---------------------------------------------------------------------------


def upstream_context(answers: list[ExpertAnswer]) -> dict[str, Any]:
    """Package predecessor answers as structured input for a successor expert.

    Deliberately not a concatenated string: each contribution keeps its
    producer, its domain, its confidence and its entity references, so the
    consuming expert (and later the arbiter) can tell the sources apart.
    """
    return {
        "upstream": [
            {
                "expert_id": a.expert_id,
                "domain": a.domain,
                "subtask_id": a.subtask_id,
                "content": a.content,
                "structured": a.structured,
                "confidence": a.confidence,
                "entities": [e.model_dump(mode="json") for e in a.entities],
            }
            for a in answers
            if a.ok
        ]
    }
