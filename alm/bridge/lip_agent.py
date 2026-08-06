"""LIP bridge — publish the federation as a provider agent on an Agentic Bus.

Layer 1 of ALM *is* the Liquid Interface Protocol, so an ALM federation does not
need an adapter to join a bus: it speaks the envelope natively.  This module
connects to a coordinator over WebSocket, registers one capability per domain,
offers on matching intents, and returns ``complete`` messages whose artifacts
carry the provenance chain — so downstream agents inherit the audit trail rather
than an unattributed paragraph.

Reference coordinator: https://github.com/draiven-io/agentic-bus

There is no dependency on the ``agentic-bus`` package. The protocol is
implemented against the wire format, which means the bridge also works with any
other coordinator that speaks LIP.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from alm.core.errors import ALMError
from alm.federation.result import FederationResult
from alm.federation.runtime import FederationRuntime
from alm.protocol.envelope import (
    LIPEnvelope,
    MessageType,
    OfferPayload,
    SenderInfo,
    SenderKind,
    build_envelope,
)
from alm.protocol.lip import (
    answer_artifact,
    complete_envelope,
    event_envelope,
    inject_trace_context,
    task_from_execute,
    task_from_intent,
)
from alm.protocol.trace import TraceEvent

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the bridge actually runs
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]


class LIPBridge:
    """Runs the federation as a LIP provider agent."""

    def __init__(
        self,
        runtime: FederationRuntime,
        *,
        coordinator_uri: str = "ws://localhost:8765",
        agent_id: str = "alm-federation",
        version: str = "0.1.0",
        stream_events: bool = True,
    ) -> None:
        self.runtime = runtime
        self.coordinator_uri = coordinator_uri
        self.agent_id = agent_id
        self.version = version
        self.stream_events = stream_events
        self._connection: Any = None
        self._running = False

    # -- capability declaration -------------------------------------------

    def registration_payload(self) -> dict[str, Any]:
        """What this federation advertises to the bus.

        One capability per domain rather than one per Expert Agent: from the
        bus's perspective the federation is a single competent participant, and
        which specialists it engages internally is its own business. That is the
        boundary that lets the federation evolve without renegotiating with
        every other agent on the bus.
        """
        capabilities: list[dict[str, Any]] = []
        for domain in sorted(self.runtime.experts.domains()):
            agents = self.runtime.experts.for_domain(domain)
            examples: list[str] = []
            entity_types: list[str] = []
            for agent in agents:
                for capability in agent.spec.capabilities:
                    examples.extend(capability.examples[:2])
                    entity_types.extend(capability.operates_on)
            capabilities.append(
                {
                    "capability_id": f"alm.{domain}",
                    "description": (
                        f"Governed {domain} analysis by an ALM federation of "
                        f"{len(agents)} domain specialist(s), with an auditable "
                        f"decision trail."
                    ),
                    "required_scopes": sorted(
                        {s for a in agents for s in a.spec.required_scopes}
                    ),
                    "supported_data_domains": [domain],
                    "operational_constraints": {
                        "experts": [a.id for a in agents],
                        "entity_types": sorted(set(entity_types)),
                        "examples": examples[:6],
                        "auditable": True,
                        "sovereign": True,
                    },
                    "expected_artifacts": ["answer", "expert_answer", "audit_trail"],
                    "output_schema": _ANSWER_SCHEMA,
                }
            )

        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "mode": "persistent",
            "capabilities": capabilities,
            "semantic_description": (
                "ALM federation: a governed set of domain models orchestrated by a "
                "Context Graph. Decomposes a request across domain specialists, "
                "arbitrates disagreement explicitly, and returns the answer with "
                "the reasoning that produced it."
            ),
            "supported_data_domains": sorted(self.runtime.experts.domains()),
        }

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Connect and register with the coordinator."""
        if websockets is None:  # pragma: no cover
            raise ALMError(
                "the LIP bridge needs the 'websockets' package, which ships with ALM; "
                "reinstall with `pip install -e .`"
            )
        auth = json.dumps({"sub": self.agent_id, "iss": "alm"})
        self._connection = await websockets.connect(
            self.coordinator_uri,
            additional_headers={"Authorization": f"Bearer {auth}"},
        )
        self._running = True
        logger.info("Bridge %s connected to %s", self.agent_id, self.coordinator_uri)
        await self._register()

    async def stop(self) -> None:
        self._running = False
        if self._connection is not None:
            with contextlib.suppress(Exception):
                await self._connection.close()
            self._connection = None
        logger.info("Bridge %s disconnected", self.agent_id)

    async def run_forever(self) -> None:
        """Connect and serve until cancelled."""
        await self.start()
        try:
            async for raw in self._connection:
                try:
                    envelope = LIPEnvelope.model_validate_json(raw)
                except Exception:
                    logger.warning("Discarding malformed envelope from the bus")
                    continue
                # Each message gets its own task so a long federation run does
                # not block the bus connection.
                asyncio.create_task(self._safe_handle(envelope))
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._running:
                logger.exception("Bridge connection lost")
        finally:
            await self.stop()

    # -- message handling --------------------------------------------------

    async def _safe_handle(self, envelope: LIPEnvelope) -> None:
        try:
            await self.handle(envelope)
        except Exception:
            logger.exception(
                "Failed to handle %s for session %s",
                envelope.message_type,
                envelope.session_id,
            )

    async def handle(self, envelope: LIPEnvelope) -> None:
        """Route one inbound envelope."""
        if envelope.message_type == MessageType.INTENT:
            await self._handle_intent(envelope)
        elif envelope.message_type == MessageType.EXECUTE:
            await self._handle_execute(envelope)
        elif envelope.message_type == MessageType.DISSOLVE:
            logger.debug("Session %s dissolved", envelope.session_id)
        else:
            logger.debug("Ignoring %s from the bus", envelope.message_type)

    async def _handle_intent(self, envelope: LIPEnvelope) -> None:
        """Offer on an intent the federation can actually cover.

        The offer is not unconditional: the router is asked, without executing
        anything, whether any declared capability matches. Offering on requests
        the federation would only escalate would waste the bus's negotiation and
        misrepresent what this participant is for.
        """
        task = task_from_intent(envelope, tenant_id=self.runtime.tenant_id)
        explanation = self.runtime.router.explain(task.intent_text, top_k=3)
        matches = explanation.get("matches") or []

        # `explain` ranks every declared capability, including ones that barely
        # scored, so a non-empty list is not evidence of competence. The router's
        # own escalation verdict is: below the threshold, this federation would
        # simply hand the work to a general LLM, and claiming domain coverage it
        # does not have is worse for the bus than staying quiet.
        if not matches or explanation.get("would_escalate"):
            logger.info(
                "Declining intent %s: no declared capability covers it with confidence",
                envelope.session_id,
            )
            return

        best = matches[0]
        domain = best.get("domain", "")
        offer = OfferPayload(
            capability_id=f"alm.{domain}" if domain else "alm.federation",
            capability_description=(
                f"ALM federation covering {domain or 'multiple domains'} with "
                f"an auditable decision trail"
            ),
            constraints={
                "routing_confidence": best.get("score", 0.0),
                "would_escalate": explanation.get("would_escalate", False),
                "governed": self.runtime.ibac.enabled,
            },
            expected_artifacts=["answer", "expert_answer", "audit_trail"],
            required_scopes=[],
            output_schema=_ANSWER_SCHEMA,
            participating_agents=sorted({m.get("expert_id", "") for m in matches if m}),
            composition_plan={
                "candidates": matches,
                "entities_detected": explanation.get("entities_detected", []),
            },
        )
        await self._send(
            build_envelope(
                MessageType.OFFER,
                self._sender(),
                envelope.session_id,
                offer,
                inject_trace_context(),
            )
        )

    async def _handle_execute(self, envelope: LIPEnvelope) -> None:
        """Run the federation and return the answer with its provenance."""
        task = task_from_execute(envelope, tenant_id=self.runtime.tenant_id)

        def forward(event: TraceEvent) -> None:
            if self.stream_events:
                asyncio.create_task(self._send(event_envelope(self.agent_id, event)))

        result = await self.runtime.run(task, on_event=forward if self.stream_events else None)
        await self._send(self._completion(result, envelope.session_id))

    def _completion(self, result: FederationResult, session_id: str) -> LIPEnvelope:
        artifacts: list[dict[str, Any]] = [
            answer_artifact(answer) for answer in result.answers if answer.ok
        ]
        artifacts.append(
            {
                "type": "audit_trail",
                # The whole point of putting a federation on a bus: the answer
                # arrives with the reasoning that produced it, not just the text.
                "contributions": [c.model_dump(mode="json") for c in result.contributions],
                "arbitration": result.arbitration.to_dict() if result.arbitration else {},
                "trail": result.audit_trail(),
            }
        )
        return complete_envelope(
            agent_id=self.agent_id,
            session_id=session_id or result.session_id,
            answer=result.answer,
            artifacts=artifacts,
            metadata={
                "confidence": result.confidence,
                "strategy": result.plan.strategy if result.plan else "",
                "experts": result.experts,
                "used_fallback": result.metrics.used_fallback,
                "cost_usd": result.metrics.cost_usd,
                "latency_ms": result.metrics.wall_ms,
                "citations": len(result.citations),
            },
            status="success" if result.ok else "error",
        )

    # -- transport ---------------------------------------------------------

    def _sender(self) -> SenderInfo:
        return SenderInfo(kind=SenderKind.AGENT, id=self.agent_id)

    async def _register(self) -> None:
        """Announce capabilities using the registration envelope LIP expects."""
        await self._send(
            build_envelope(
                MessageType.COMPLETE,
                self._sender(),
                "__registration__",
                {"registration": self.registration_payload()},
                inject_trace_context(),
            )
        )
        logger.info(
            "Registered %d domain capability(ies) with the bus",
            len(self.registration_payload()["capabilities"]),
        )

    async def _send(self, envelope: LIPEnvelope) -> None:
        if self._connection is None:
            logger.debug("Cannot send %s: not connected", envelope.message_type)
            return
        try:
            await self._connection.send(envelope.model_dump_json())
        except Exception:
            logger.warning("Failed to send %s to the bus", envelope.message_type)


_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The synthesised answer."},
        "confidence": {"type": "number"},
        "experts": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "object"}},
        "audit_trail": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence"],
}
