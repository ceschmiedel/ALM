"""The Liquid Interface Protocol bridge."""

from __future__ import annotations

import pytest

from alm.bridge.lip_agent import LIPBridge
from alm.protocol.envelope import (
    ExecutePayload,
    IntentPayload,
    LIPEnvelope,
    MessageType,
    SenderInfo,
    SenderKind,
    build_envelope,
)


class _RecordingConnection:
    """Captures what the bridge would put on the wire."""

    def __init__(self) -> None:
        self.sent: list[LIPEnvelope] = []

    async def send(self, raw: str) -> None:
        self.sent.append(LIPEnvelope.model_validate_json(raw))

    async def close(self) -> None:
        return None


@pytest.fixture
def bridge(runtime):
    instance = LIPBridge(runtime, agent_id="alm-test", stream_events=False)
    instance._connection = _RecordingConnection()
    return instance


def test_registration_advertises_one_capability_per_domain(bridge):
    payload = bridge.registration_payload()
    ids = {c["capability_id"] for c in payload["capabilities"]}
    assert ids == {"alm.legal", "alm.finance", "alm.risk"}
    assert payload["mode"] == "persistent"
    assert set(payload["supported_data_domains"]) == {"legal", "finance", "risk"}


def test_registration_declares_the_experts_behind_each_capability(bridge):
    payload = bridge.registration_payload()
    legal = next(c for c in payload["capabilities"] if c["capability_id"] == "alm.legal")
    assert "contracts-expert" in legal["operational_constraints"]["experts"]
    assert legal["operational_constraints"]["auditable"] is True
    assert legal["expected_artifacts"]


async def test_registration_is_sent_on_connect(bridge):
    await bridge._register()
    sent = bridge._connection.sent
    assert len(sent) == 1
    assert sent[0].session_id == "__registration__"
    assert "registration" in sent[0].payload


async def test_matching_intent_produces_an_offer(bridge):
    envelope = build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id="req"),
        "ses-1",
        IntentPayload(intent_text="What does clause 7.2 cap the liability at?"),
    )
    await bridge.handle(envelope)

    offers = [e for e in bridge._connection.sent if e.message_type is MessageType.OFFER]
    assert len(offers) == 1
    assert offers[0].payload["capability_id"].startswith("alm.")
    assert offers[0].payload["participating_agents"]


async def test_unmatched_intent_is_declined_silently(bridge):
    envelope = build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id="req"),
        "ses-2",
        IntentPayload(intent_text="what is the best recipe for sourdough bread?"),
    )
    await bridge.handle(envelope)
    # Offering on work the federation would only escalate misrepresents it.
    assert not [e for e in bridge._connection.sent if e.message_type is MessageType.OFFER]


async def test_execute_returns_a_complete_with_provenance(bridge):
    envelope = build_envelope(
        MessageType.EXECUTE,
        SenderInfo(kind=SenderKind.COORDINATOR, id="coord"),
        "ses-3",
        ExecutePayload(
            execution_plan={
                "intent_text": "What does clause 7.2 cap the liability at?",
                "context": {},
            }
        ),
    )
    await bridge.handle(envelope)

    completes = [
        e for e in bridge._connection.sent if e.message_type is MessageType.COMPLETE
    ]
    assert len(completes) == 1
    payload = completes[0].payload
    assert payload["status"] == "success"
    assert payload["artifacts"][0]["type"] == "answer"

    trail = next(a for a in payload["artifacts"] if a["type"] == "audit_trail")
    assert "contributions" in trail
    assert "arbitration" in trail
    assert payload["metadata"]["experts"] is not None


async def test_dissolve_is_handled_without_error(bridge):
    envelope = build_envelope(
        MessageType.DISSOLVE,
        SenderInfo(kind=SenderKind.COORDINATOR, id="coord"),
        "ses-4",
        {"reason": "session_complete"},
    )
    await bridge.handle(envelope)
    assert not bridge._connection.sent


async def test_malformed_execute_does_not_crash_the_bridge(bridge):
    envelope = build_envelope(
        MessageType.EXECUTE,
        SenderInfo(kind=SenderKind.COORDINATOR, id="coord"),
        "ses-5",
        ExecutePayload(execution_plan={}),
    )
    # _safe_handle swallows the error so one bad message cannot drop the bus link.
    await bridge._safe_handle(envelope)


async def test_offer_reports_whether_it_would_escalate(bridge):
    envelope = build_envelope(
        MessageType.INTENT,
        SenderInfo(kind=SenderKind.REQUESTER, id="req"),
        "ses-6",
        IntentPayload(intent_text="What is the liability cap and the payment terms?"),
    )
    await bridge.handle(envelope)
    offer = next(e for e in bridge._connection.sent if e.message_type is MessageType.OFFER)
    assert "would_escalate" in offer.payload["constraints"]
    assert "routing_confidence" in offer.payload["constraints"]
