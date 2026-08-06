"""Drift detection and the promotion triggers.

Two questions this module answers, both of which the architecture insists must
be settled by measurement rather than intuition:

**Has an expert drifted?** Accuracy falling or fallback rising over time is the
signal that the domain moved and the model needs recycling. Detection compares a
recent window against a baseline window and reports the delta, rather than
alerting on a single bad day.

**Should this agent get its own model?** An Expert Agent graduates from "RAG over
a shared model" to "dedicated model" when at least one business trigger is
confirmed by evaluation — sovereignty, volume, latency or accuracy. The check is
objective on purpose: promoting on intuition is how a federation acquires MLOps
load it cannot justify.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import timedelta

from pydantic import BaseModel, Field
from sqlalchemy import select

from alm.core.ids import utc_now
from alm.evaluation.metrics import percentile
from alm.persistence.database import session_scope
from alm.persistence.models import MetricSnapshotRow, SessionRow

logger = logging.getLogger(__name__)


class DriftReport(BaseModel):
    """Comparison of a recent window against a baseline window."""

    expert_id: str = ""
    domain: str = ""
    baseline_samples: int = 0
    recent_samples: int = 0

    accuracy_delta: float = 0.0
    fallback_delta: float = 0.0
    latency_delta_ms: float = 0.0
    confidence_delta: float = 0.0

    drifted: bool = False
    reasons: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        if not self.drifted:
            return f"{self.expert_id or 'federation'}: no drift detected"
        return f"{self.expert_id or 'federation'}: {'; '.join(self.reasons)}"


class DriftDetector:
    """Compares recent behaviour with an earlier baseline."""

    def __init__(
        self,
        tenant_id: str = "default",
        *,
        accuracy_drop: float = 0.05,
        fallback_rise: float = 0.10,
        latency_rise_ratio: float = 0.5,
        min_samples: int = 20,
    ) -> None:
        self.tenant_id = tenant_id
        self.accuracy_drop = accuracy_drop
        self.fallback_rise = fallback_rise
        self.latency_rise_ratio = latency_rise_ratio
        self.min_samples = min_samples

    def snapshot(self, *, window_hours: int = 24) -> list[MetricSnapshotRow]:
        """Record current per-expert health from recent sessions."""
        window_start = utc_now() - timedelta(hours=window_hours)
        with session_scope() as session:
            sessions = session.execute(
                select(SessionRow).where(
                    SessionRow.tenant_id == self.tenant_id,
                    SessionRow.created_at >= window_start,
                )
            ).scalars().all()

            by_expert: dict[str, list[SessionRow]] = {}
            for row in sessions:
                for answer in row.answers or []:
                    expert_id = str(answer.get("expert_id") or "")
                    if expert_id:
                        by_expert.setdefault(expert_id, []).append(row)

            created: list[MetricSnapshotRow] = []
            for expert_id, rows in by_expert.items():
                latencies = [
                    float((r.metrics or {}).get("wall_ms", 0.0) or 0.0) for r in rows
                ]
                confidences = [
                    float(a.get("confidence", 0.0) or 0.0)
                    for r in rows
                    for a in (r.answers or [])
                    if a.get("expert_id") == expert_id
                ]
                snapshot = MetricSnapshotRow(
                    tenant_id=self.tenant_id,
                    expert_id=expert_id,
                    window_start=window_start,
                    window_end=utc_now(),
                    # Production accuracy is unlabelled; the honest proxy is the
                    # share of runs that completed without escalating.
                    accuracy=round(
                        sum(1 for r in rows if not r.used_fallback) / len(rows), 4
                    ),
                    fallback_rate=round(
                        sum(1 for r in rows if r.used_fallback) / len(rows), 4
                    ),
                    p95_latency_ms=percentile(latencies, 0.95),
                    mean_confidence=round(
                        sum(confidences) / len(confidences), 4
                    ) if confidences else 0.0,
                    sample_count=len(rows),
                )
                session.add(snapshot)
                created.append(snapshot)
            return created

    def detect(self, expert_id: str, *, lookback: int = 10) -> DriftReport:
        """Compare the newest snapshot against the mean of earlier ones."""
        with session_scope() as session:
            snapshots = session.execute(
                select(MetricSnapshotRow)
                .where(
                    MetricSnapshotRow.tenant_id == self.tenant_id,
                    MetricSnapshotRow.expert_id == expert_id,
                )
                .order_by(MetricSnapshotRow.created_at.desc())
                .limit(lookback + 1)
            ).scalars().all()

        if len(snapshots) < 2:
            return DriftReport(
                expert_id=expert_id,
                reasons=["not enough history to judge drift"],
            )

        recent = snapshots[0]
        baseline = snapshots[1:]
        baseline_samples = sum(s.sample_count for s in baseline)

        if recent.sample_count < self.min_samples or baseline_samples < self.min_samples:
            return DriftReport(
                expert_id=expert_id,
                recent_samples=recent.sample_count,
                baseline_samples=baseline_samples,
                reasons=[
                    f"insufficient volume to judge drift "
                    f"({recent.sample_count} recent, {baseline_samples} baseline; "
                    f"{self.min_samples} needed)"
                ],
            )

        def mean(attribute: str) -> float:
            values = [getattr(s, attribute) for s in baseline]
            return sum(values) / len(values) if values else 0.0

        accuracy_delta = round(recent.accuracy - mean("accuracy"), 4)
        fallback_delta = round(recent.fallback_rate - mean("fallback_rate"), 4)
        latency_baseline = mean("p95_latency_ms")
        latency_delta = round(recent.p95_latency_ms - latency_baseline, 3)
        confidence_delta = round(recent.mean_confidence - mean("mean_confidence"), 4)

        reasons: list[str] = []
        if accuracy_delta <= -self.accuracy_drop:
            reasons.append(
                f"completion rate fell {abs(accuracy_delta):.1%} against baseline"
            )
        if fallback_delta >= self.fallback_rise:
            reasons.append(f"fallback rate rose {fallback_delta:.1%}")
        if latency_baseline > 0 and latency_delta / latency_baseline >= self.latency_rise_ratio:
            reasons.append(
                f"p95 latency rose {latency_delta / latency_baseline:.0%}"
            )

        return DriftReport(
            expert_id=expert_id,
            baseline_samples=baseline_samples,
            recent_samples=recent.sample_count,
            accuracy_delta=accuracy_delta,
            fallback_delta=fallback_delta,
            latency_delta_ms=latency_delta,
            confidence_delta=confidence_delta,
            drifted=bool(reasons),
            reasons=reasons,
        )

    def detect_all(self, expert_ids: Sequence[str]) -> list[DriftReport]:
        return [self.detect(expert_id) for expert_id in expert_ids]


# ---------------------------------------------------------------------------
# Promotion triggers
# ---------------------------------------------------------------------------


class PromotionTrigger(BaseModel):
    """Whether one business trigger for a dedicated model is confirmed."""

    name: str
    confirmed: bool = False
    evidence: str = ""
    value: float | None = None
    threshold: float | None = None


class PromotionAssessment(BaseModel):
    """The verdict on whether an expert should get its own model."""

    expert_id: str
    triggers: list[PromotionTrigger] = Field(default_factory=list)
    recommend_dedicated: bool = False
    recommendation: str = ""

    def confirmed(self) -> list[str]:
        return [t.name for t in self.triggers if t.confirmed]

    def render(self) -> str:
        lines = [f"Promotion check · {self.expert_id}", ""]
        for trigger in self.triggers:
            mark = "✓" if trigger.confirmed else "·"
            lines.append(f"  {mark} {trigger.name:<12} {trigger.evidence}")
        lines.append("")
        lines.append(f"  {self.recommendation}")
        return "\n".join(lines)


def assess_promotion(
    *,
    expert_id: str,
    domain: str = "",
    tenant_id: str = "default",
    sovereignty_required: bool = False,
    monthly_calls: int | None = None,
    volume_threshold: int = 50_000,
    latency_requirement_ms: float | None = None,
    observed_p95_ms: float | None = None,
    rag_accuracy: float | None = None,
    accuracy_target: float | None = None,
    window_hours: int = 24 * 30,
) -> PromotionAssessment:
    """Evaluate the four triggers for promoting an expert to a dedicated model.

    An agent moves from RAG-over-a-shared-model to its own model when at least
    one of these is confirmed *by measurement*:

    * **sovereignty** — the domain handles data that cannot leave the perimeter;
    * **volume** — the agent is called often enough that aggregate LLM cost
      exceeds the cost of training and serving a dedicated one;
    * **latency** — the task needs real-time response an external call cannot
      guarantee;
    * **accuracy** — RAG has hit its ceiling and still errs systematically on
      domain behaviour.
    """
    triggers: list[PromotionTrigger] = []

    triggers.append(
        PromotionTrigger(
            name="sovereignty",
            confirmed=sovereignty_required,
            evidence=(
                "domain handles data that must not leave the perimeter"
                if sovereignty_required
                else "no residency constraint declared for this domain"
            ),
        )
    )

    observed_calls = monthly_calls
    if observed_calls is None:
        observed_calls = _count_expert_calls(expert_id, tenant_id, window_hours)
    triggers.append(
        PromotionTrigger(
            name="volume",
            confirmed=observed_calls >= volume_threshold,
            evidence=(
                f"{observed_calls:,} call(s) in the window against a "
                f"{volume_threshold:,} threshold"
            ),
            value=float(observed_calls),
            threshold=float(volume_threshold),
        )
    )

    latency_confirmed = (
        latency_requirement_ms is not None
        and observed_p95_ms is not None
        and observed_p95_ms > latency_requirement_ms
    )
    triggers.append(
        PromotionTrigger(
            name="latency",
            confirmed=latency_confirmed,
            evidence=(
                f"observed p95 {observed_p95_ms:.0f}ms exceeds the "
                f"{latency_requirement_ms:.0f}ms requirement"
                if latency_confirmed
                else "no unmet real-time requirement measured"
            ),
            value=observed_p95_ms,
            threshold=latency_requirement_ms,
        )
    )

    accuracy_confirmed = (
        rag_accuracy is not None
        and accuracy_target is not None
        and rag_accuracy < accuracy_target
    )
    triggers.append(
        PromotionTrigger(
            name="accuracy",
            confirmed=accuracy_confirmed,
            evidence=(
                f"RAG accuracy {rag_accuracy:.3f} is below the "
                f"{accuracy_target:.3f} target"
                if accuracy_confirmed
                else "RAG has not been shown to hit its ceiling"
            ),
            value=rag_accuracy,
            threshold=accuracy_target,
        )
    )

    confirmed = [t for t in triggers if t.confirmed]
    if confirmed:
        recommendation = (
            f"Promote to a dedicated model — {len(confirmed)} trigger(s) confirmed: "
            f"{', '.join(t.name for t in confirmed)}. Start with a LoRA adapter; "
            f"most experts stop there."
        )
    else:
        recommendation = (
            "Keep this agent as RAG over the shared model. No business trigger is "
            "confirmed, and a dedicated model would add MLOps load without a "
            "measured return."
        )

    return PromotionAssessment(
        expert_id=expert_id,
        triggers=triggers,
        recommend_dedicated=bool(confirmed),
        recommendation=recommendation,
    )


def _count_expert_calls(expert_id: str, tenant_id: str, window_hours: int) -> int:
    since = utc_now() - timedelta(hours=window_hours)
    with session_scope() as session:
        rows = session.execute(
            select(SessionRow.answers).where(
                SessionRow.tenant_id == tenant_id,
                SessionRow.created_at >= since,
            )
        ).scalars().all()
    return sum(
        1
        for answers in rows
        for answer in (answers or [])
        if str(answer.get("expert_id") or "") == expert_id
    )
