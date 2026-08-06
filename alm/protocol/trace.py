"""Execution trace — the auditability guarantee of the federation.

In a monolithic model the reasoning that produced an answer is not
recoverable.  In a federation every transition is a message between named
components, so the trail can be complete by construction rather than
reconstructed after the fact.  This module is what makes that concrete: every
layer emits :class:`TraceEvent` records into a single :class:`ExecutionTrace`
that ships with the answer.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from alm.core.ids import new_id, utc_now_iso


class Layer(StrEnum):
    """Architecture layer that emitted an event."""

    L1_INTERFACE = "L1"
    L2_ROUTER = "L2"
    L3_EXPERTS = "L3"
    L4_ORCHESTRATION = "L4"
    L5_ARBITRATION = "L5"
    CX_CONTEXT = "CX"
    GV_GOVERNANCE = "GV"


_LAYER_LABELS = {
    Layer.L1_INTERFACE: "interface",
    Layer.L2_ROUTER: "router",
    Layer.L3_EXPERTS: "experts",
    Layer.L4_ORCHESTRATION: "orchestration",
    Layer.L5_ARBITRATION: "arbitration",
    Layer.CX_CONTEXT: "context",
    Layer.GV_GOVERNANCE: "governance",
}


class TraceEvent(BaseModel):
    """A single, immutable observation about the run."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)
    layer: Layer
    category: str = "info"
    summary: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)

    subtask_id: str = ""
    expert_id: str = ""
    model_id: str = ""
    duration_ms: float | None = None
    progress: float | None = None

    def line(self) -> str:
        label = _LAYER_LABELS.get(self.layer, str(self.layer))
        who = f" [{self.expert_id}]" if self.expert_id else ""
        took = f" ({self.duration_ms:.0f}ms)" if self.duration_ms is not None else ""
        return f"{self.layer:>2} {label:<13}{who} {self.summary}{took}"


TraceListener = Callable[[TraceEvent], None]


class ExecutionTrace(BaseModel):
    """Ordered collection of trace events for one federation session."""

    session_id: str = ""
    events: list[TraceEvent] = Field(default_factory=list)

    # Listeners are runtime-only (streaming to a WebSocket, a CLI spinner, a
    # test collector) and never serialised with the trace.
    _listeners: list[TraceListener] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def subscribe(self, listener: TraceListener) -> Callable[[], None]:
        """Register a listener; returns a function that unsubscribes it."""
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    def emit(
        self,
        layer: Layer,
        summary: str,
        *,
        category: str = "info",
        detail: dict[str, Any] | None = None,
        subtask_id: str = "",
        expert_id: str = "",
        model_id: str = "",
        duration_ms: float | None = None,
        progress: float | None = None,
    ) -> TraceEvent:
        """Append an event and notify listeners."""
        event = TraceEvent(
            session_id=self.session_id,
            layer=layer,
            category=category,
            summary=summary,
            detail=detail or {},
            subtask_id=subtask_id,
            expert_id=expert_id,
            model_id=model_id,
            duration_ms=duration_ms,
            progress=progress,
        )
        self.events.append(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # a broken listener must not break the run
                pass
        return event

    # -- queries -----------------------------------------------------------

    def by_layer(self, layer: Layer) -> list[TraceEvent]:
        return [e for e in self.events if e.layer == layer]

    def errors(self) -> list[TraceEvent]:
        return [e for e in self.events if e.category == "error"]

    def render(self, *, include_detail: bool = False) -> str:
        """Render the trail as human-readable text (used by the CLI)."""
        lines: list[str] = []
        for event in self.events:
            lines.append(event.line())
            if include_detail and event.detail:
                for key, value in event.detail.items():
                    lines.append(f"      · {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "events": [e.model_dump(mode="json") for e in self.events],
        }
