"""Expert output contracts.

An :class:`ExpertAnswer` is deliberately more than a string. Arbitration (L5)
cannot weigh two answers unless each one carries the confidence its producer
assigns to it, the evidence it used, and the model that produced it. The
monolithic alternative resolves contradictions invisibly inside a latent space;
here the inputs to that resolution are on the record.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from alm.core.ids import new_id, utc_now_iso
from alm.protocol.task import EntityRef


class Citation(BaseModel):
    """A retrieved fragment an expert used to justify its answer."""

    chunk_id: str = ""
    document_id: str = ""
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    domain: str = ""
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpertAnswer(BaseModel):
    """The result of one Expert Agent executing one subtask."""

    answer_id: str = Field(default_factory=lambda: new_id("ans"))
    subtask_id: str = ""
    expert_id: str = ""
    domain: str = ""
    capability_id: str = ""

    content: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    entities: list[EntityRef] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    # Confidence the expert reports, and the value after calibration.  They are
    # kept apart on purpose: a model that is confident and wrong is worse than
    # one that is uncertain and honest, and only the calibrated number should
    # ever drive arbitration.
    raw_confidence: float = 0.0
    confidence: float = 0.0
    calibrated: bool = False

    # Accounting, per answer, so cost and latency roll up per session.
    model_id: str = ""
    tier: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    is_fallback: bool = False
    status: str = Field(default="success", description="success | error | skipped")
    error: str = ""
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def summary(self, limit: int = 240) -> str:
        text = self.content.strip().replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"


class Contribution(BaseModel):
    """How one expert's answer entered the final synthesis.

    This is the row a regulated-industry auditor actually reads: who said what,
    with how much confidence, and whether it survived arbitration.
    """

    expert_id: str
    domain: str = ""
    subtask_id: str = ""
    answer_id: str = ""
    confidence: float = 0.0
    weight: float = 0.0
    accepted: bool = True
    superseded_by: str = ""
    reason: str = ""
