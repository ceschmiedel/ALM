"""The result of a federation run.

Everything an auditor, an evaluation harness or a debugging engineer needs is
in one object: the answer, the plan that produced it, every expert answer, how
arbitration decided, the full layer-by-layer trace, and the cost and latency
that the comparison against a monolithic baseline turns on.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from alm.arbitration.arbiter import ArbitrationResult
from alm.core.ids import utc_now_iso
from alm.protocol.answer import Citation, Contribution, ExpertAnswer
from alm.protocol.task import ExecutionPlan, TaskEnvelope
from alm.protocol.trace import ExecutionTrace


class RunMetrics(BaseModel):
    """The measured cost of one run, in every sense that matters."""

    wall_ms: float = 0.0
    inference_ms: float = 0.0
    routing_ms: float = 0.0
    execution_ms: float = 0.0
    arbitration_ms: float = 0.0

    cost_usd: float = 0.0
    total_tokens: int = 0
    model_calls: int = 0

    experts_invoked: int = 0
    experts_succeeded: int = 0
    dag_width: int = 0
    dag_depth: int = 0

    used_fallback: bool = False
    escalations: int = 0
    conflicts: int = 0
    conflicts_resolved: int = 0

    by_tier: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)

    @property
    def parallel_speedup(self) -> float:
        """How much the DAG's parallelism saved against running steps serially.

        Serial inference time over wall-clock: 1.0 means no benefit, 2.0 means
        the plan took half the time a sequential run would have.
        """
        if self.wall_ms <= 0:
            return 1.0
        return round(max(1.0, self.inference_ms / self.wall_ms), 3)


class FederationResult(BaseModel):
    """The complete, auditable outcome of running one intent."""

    session_id: str
    task: TaskEnvelope
    answer: str = ""
    status: str = Field(default="success", description="success | denied | error")
    error: str = ""

    plan: ExecutionPlan | None = None
    answers: list[ExpertAnswer] = Field(default_factory=list)
    arbitration: ArbitrationResult | None = None
    contributions: list[Contribution] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    confidence: float = 0.0
    synthesis_method: str = ""
    trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    created_at: str = Field(default_factory=utc_now_iso)

    # -- convenience -------------------------------------------------------

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @property
    def experts(self) -> list[str]:
        return [c.expert_id for c in self.contributions if c.accepted]

    @property
    def used_fallback(self) -> bool:
        return self.metrics.used_fallback

    def answer_for(self, expert_id: str) -> ExpertAnswer | None:
        for answer in self.answers:
            if answer.expert_id == expert_id:
                return answer
        return None

    def audit_trail(self) -> list[str]:
        """The decision record: who contributed what, and why it prevailed."""
        return self.arbitration.audit_lines() if self.arbitration else []

    def trace_is_complete(self) -> bool:
        """Whether every layer that ran left a record.

        This is the auditability metric the evaluation harness scores: a
        federation earns its claim to be auditable by construction only if the
        trail actually covers routing, execution and arbitration.
        """
        from alm.protocol.trace import Layer

        required = {Layer.L2_ROUTER, Layer.L4_ORCHESTRATION, Layer.L5_ARBITRATION}
        present = {event.layer for event in self.trace.events}
        return required.issubset(present)

    # -- rendering ---------------------------------------------------------

    def summary(self) -> str:
        parts = [
            f"session   {self.session_id}",
            f"status    {self.status}",
            f"strategy  {self.plan.strategy if self.plan else '-'}",
            f"experts   {', '.join(self.experts) or '-'}",
            f"confidence {self.confidence:.2f}",
            f"latency   {self.metrics.wall_ms:.0f} ms",
            f"cost      ${self.metrics.cost_usd:.6f}",
        ]
        if self.metrics.used_fallback:
            parts.append("fallback  yes (escalated to the orchestrator)")
        return "\n".join(parts)

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
            "confidence": round(self.confidence, 4),
            "synthesis_method": self.synthesis_method,
            "intent_text": self.task.intent_text,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "answers": [a.model_dump(mode="json") for a in self.answers],
            "contributions": [c.model_dump(mode="json") for c in self.contributions],
            "citations": [c.model_dump(mode="json") for c in self.citations],
            "arbitration": self.arbitration.to_dict() if self.arbitration else None,
            "audit_trail": self.audit_trail(),
            "metrics": self.metrics.model_dump(mode="json"),
            "created_at": self.created_at,
        }
        if include_trace:
            data["trace"] = self.trace.to_dict()
        return data
