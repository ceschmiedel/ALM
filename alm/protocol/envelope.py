"""L1 · Liquid Interface Protocol envelope.

ALM does not invent a new transport. Layer 1 of the architecture *is* the
Liquid Interface Protocol (LIP): the same envelope the Agentic Bus uses to
carry intent, offers, execution and completion between agents.  Keeping the
envelope wire-identical means an ALM federation can be dropped onto an
existing bus as a provider agent, and an Agentic Bus coordinator can route
work into the federation without a translation layer.

Reference implementation of the protocol:
https://github.com/draiven-io/agentic-bus

The performative message types are unchanged from LIP §4.1.1:

``intent``
    Articulates a high-level objective and instantiates an interaction context.
``offer``
    Declares a capability relevant to the expressed intention.
``accept`` / ``reject``
    Negotiated agreement, or structured refusal.
``execute``
    Authorises execution under the negotiated terms.
``complete``
    Signals termination of execution and carries artifacts.
``dissolve``
    Invalidates the interaction context; mandates cleanup.
``event``
    Informational progress notification; carries no performative weight.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from alm.core.ids import utc_now_iso


class MessageType(StrEnum):
    """Performative message types defined by LIP §4.1.1."""

    INTENT = "intent"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    EXECUTE = "execute"
    COMPLETE = "complete"
    DISSOLVE = "dissolve"
    EVENT = "event"


class SenderKind(StrEnum):
    """Runtime role of the message sender."""

    REQUESTER = "requester"
    COORDINATOR = "coordinator"
    AGENT = "agent"


class SenderInfo(BaseModel):
    """Identity of the message sender."""

    kind: SenderKind
    id: str
    oidc_subject: str = ""


class TraceContext(BaseModel):
    """W3C Trace-Context compatible propagation fields."""

    trace_id: str = ""
    span_id: str = ""


class LIPEnvelope(BaseModel):
    """Common envelope wrapping every message on the wire.

    Field-for-field compatible with the Agentic Bus envelope so that messages
    can be exchanged with a LIP coordinator without conversion.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    message_type: MessageType
    timestamp: str = Field(default_factory=utc_now_iso)
    sender: SenderInfo
    trace: TraceContext = Field(default_factory=TraceContext)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------


class IntentPayload(BaseModel):
    """Payload for ``message_type='intent'``."""

    intent_text: str
    context: dict[str, Any] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    ibac_claims_requested: list[str] = Field(default_factory=list)
    assigned_agent_id: str = ""


class OfferPayload(BaseModel):
    """Payload for ``message_type='offer'``.

    An ALM federation offers a *domain coverage* rather than a single function:
    the capability describes which ontology domains the federation can reason
    over, and the composition plan describes which experts would be engaged.
    """

    capability_id: str
    capability_description: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    estimated_cost: float | None = None
    estimated_latency: float | None = None
    required_scopes: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    composition_plan: dict[str, Any] = Field(default_factory=dict)
    participating_agents: list[str] = Field(default_factory=list)


class AcceptPayload(BaseModel):
    """Payload for ``message_type='accept'``."""

    accepted_offers: list[str] = Field(default_factory=list)
    composition_plan: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    approval_note: str = ""


class RejectPayload(BaseModel):
    """Payload for ``message_type='reject'``."""

    rejected_offers: list[str] = Field(default_factory=list)
    reason: str = ""
    renegotiation_hint: dict[str, Any] = Field(default_factory=dict)
    renegotiate: bool = False


class ExecutePayload(BaseModel):
    """Payload for ``message_type='execute'``."""

    execution_plan: dict[str, Any] = Field(default_factory=dict)
    authorized_scopes: list[str] = Field(default_factory=list)


class CompletePayload(BaseModel):
    """Payload for ``message_type='complete'``."""

    status: str = "success"
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DissolvePayload(BaseModel):
    """Payload for ``message_type='dissolve'``."""

    reason: str = "session_complete"


class EventPayload(BaseModel):
    """Payload for ``message_type='event'`` — progress and status notifications."""

    category: str = "info"
    phase: str = ""
    summary: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    agent_id: str = ""
    step_index: int | None = None
    progress: float | None = None


PAYLOAD_TYPES: dict[MessageType, type[BaseModel]] = {
    MessageType.INTENT: IntentPayload,
    MessageType.OFFER: OfferPayload,
    MessageType.ACCEPT: AcceptPayload,
    MessageType.REJECT: RejectPayload,
    MessageType.EXECUTE: ExecutePayload,
    MessageType.COMPLETE: CompletePayload,
    MessageType.DISSOLVE: DissolvePayload,
    MessageType.EVENT: EventPayload,
}


def build_envelope(
    message_type: MessageType,
    sender: SenderInfo,
    session_id: str,
    payload: BaseModel | dict[str, Any],
    trace: TraceContext | None = None,
) -> LIPEnvelope:
    """Build a fully populated :class:`LIPEnvelope`."""
    payload_dict = payload.model_dump() if isinstance(payload, BaseModel) else payload
    return LIPEnvelope(
        session_id=session_id,
        message_type=message_type,
        sender=sender,
        trace=trace or TraceContext(),
        payload=payload_dict,
    )


def parse_payload(envelope: LIPEnvelope) -> BaseModel:
    """Validate ``envelope.payload`` against the schema for its message type."""
    model = PAYLOAD_TYPES[envelope.message_type]
    return model.model_validate(envelope.payload)
