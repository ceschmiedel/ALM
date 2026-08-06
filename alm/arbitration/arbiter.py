"""The arbiter — runs verification, detects conflicts, applies the strategy chain.

The output is not just a winner. It is a record: which experts contributed,
which answers were superseded and by what reasoning, and which disagreements
could not be settled. For an enterprise client in a regulated vertical, that
record is the non-negotiable part — an answer without a trail is not adoptable.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from alm.arbitration.conflict import Conflict, ConflictDetector
from alm.arbitration.strategies import ArbitrationStrategy, Resolution
from alm.core.telemetry import UsageMeter
from alm.experts.registry import ExpertRegistry
from alm.experts.verifier import VerificationResult
from alm.protocol.answer import Contribution, ExpertAnswer
from alm.protocol.trace import ExecutionTrace, Layer

logger = logging.getLogger(__name__)


class ArbitrationResult(BaseModel):
    """Everything L5 decided, and why."""

    contributions: list[Contribution] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list)
    verifications: dict[str, VerificationResult] = Field(default_factory=dict)
    accepted_answer_ids: list[str] = Field(default_factory=list)
    superseded_answer_ids: list[str] = Field(default_factory=list)
    unresolved: list[Conflict] = Field(default_factory=list)

    @property
    def had_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def fully_resolved(self) -> bool:
        return not self.unresolved

    def audit_lines(self) -> list[str]:
        """Human-readable decision record."""
        lines = [
            f"{c.expert_id} ({c.domain}) contributed with confidence {c.confidence:.2f}, "
            f"weight {c.weight:.2f}"
            + ("" if c.accepted else f" — superseded by {c.superseded_by}: {c.reason}")
            for c in self.contributions
        ]
        lines.extend(r.line() for r in self.resolutions)
        lines.extend(f"UNRESOLVED — {c.summary()}" for c in self.unresolved)
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributions": [c.model_dump(mode="json") for c in self.contributions],
            "conflicts": [c.model_dump(mode="json") for c in self.conflicts],
            "resolutions": [r.model_dump(mode="json") for r in self.resolutions],
            "verifications": {
                k: v.model_dump(mode="json") for k, v in self.verifications.items()
            },
            "accepted": self.accepted_answer_ids,
            "superseded": self.superseded_answer_ids,
            "unresolved": [c.model_dump(mode="json") for c in self.unresolved],
        }


class Arbiter:
    """Applies verification and the strategy chain to a set of answers."""

    def __init__(
        self,
        strategies: list[ArbitrationStrategy],
        *,
        experts: ExpertRegistry | None = None,
        detector: ConflictDetector | None = None,
    ) -> None:
        self.strategies = strategies
        self.experts = experts
        self.detector = detector or ConflictDetector()

    async def arbitrate(
        self,
        answers: list[ExpertAnswer],
        *,
        intent_text: str = "",
        primary_domain: str = "",
        trace: ExecutionTrace | None = None,
        meter: UsageMeter | None = None,
    ) -> ArbitrationResult:
        result = ArbitrationResult()
        by_id = {a.answer_id: a for a in answers}

        # 1 · verification, before anything is weighed
        result.verifications = self._verify(answers, trace)

        # 2 · conflict detection
        conflicts = self.detector.detect(answers)
        result.conflicts = conflicts

        if trace is not None:
            if conflicts:
                trace.emit(
                    Layer.L5_ARBITRATION,
                    f"detected {len(conflicts)} conflict(s) between specialists",
                    category="conflict",
                    detail={"conflicts": [c.summary() for c in conflicts]},
                )
            else:
                trace.emit(
                    Layer.L5_ARBITRATION,
                    "no conflicts between specialists",
                    category="conflict",
                )

        # 3 · resolution
        context: dict[str, Any] = {
            "intent_text": intent_text,
            "primary_domain": primary_domain,
            "verifications": result.verifications,
            "meter": meter,
        }
        superseded: dict[str, tuple[str, str]] = {}

        for conflict in conflicts:
            resolution = await self._resolve(conflict, by_id, context, trace)
            if resolution is None:
                result.unresolved.append(conflict)
                if trace is not None:
                    trace.emit(
                        Layer.L5_ARBITRATION,
                        f"conflict left unresolved: {conflict.summary()}",
                        category="warning",
                        detail={"conflict": conflict.model_dump(mode="json")},
                    )
                continue

            result.resolutions.append(resolution)
            for answer_id in conflict.participants:
                if answer_id != resolution.winner_answer_id:
                    superseded[answer_id] = (
                        resolution.winner_expert,
                        f"{resolution.strategy}: {resolution.reason}",
                    )

        # 4 · contributions
        result.contributions = self._contributions(
            answers, superseded, result.verifications, primary_domain
        )
        result.accepted_answer_ids = [
            a.answer_id for a in answers if a.ok and a.answer_id not in superseded
        ]
        result.superseded_answer_ids = list(superseded)

        if trace is not None and result.resolutions:
            trace.emit(
                Layer.L5_ARBITRATION,
                f"resolved {len(result.resolutions)} conflict(s)",
                category="resolution",
                detail={"resolutions": [r.line() for r in result.resolutions]},
            )

        return result

    # -- internals ---------------------------------------------------------

    def _verify(
        self, answers: list[ExpertAnswer], trace: ExecutionTrace | None
    ) -> dict[str, VerificationResult]:
        verifications: dict[str, VerificationResult] = {}
        if self.experts is None:
            return verifications

        for answer in answers:
            agent = self.experts.get(answer.expert_id) if answer.expert_id else None
            if agent is None:
                continue
            verification = agent.verify(answer)
            verifications[answer.answer_id] = verification
            if not verification.passed and trace is not None:
                trace.emit(
                    Layer.L5_ARBITRATION,
                    f"{answer.expert_id} failed verification: {verification.summary()}",
                    category="verification",
                    expert_id=answer.expert_id,
                    detail={
                        "violations": [
                            v.model_dump(mode="json") for v in verification.violations
                        ]
                    },
                )
        return verifications

    async def _resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        context: dict[str, Any],
        trace: ExecutionTrace | None,
    ) -> Resolution | None:
        """Walk the chain until a strategy decides."""
        for strategy in self.strategies:
            try:
                resolution = await strategy.resolve(conflict, answers, context=context)
            except Exception:
                logger.warning(
                    "Arbitration strategy %s raised; continuing the chain",
                    strategy.name,
                    exc_info=True,
                )
                continue
            if resolution is not None:
                if trace is not None:
                    trace.emit(
                        Layer.L5_ARBITRATION,
                        resolution.line(),
                        category="resolution",
                        expert_id=resolution.winner_expert,
                        detail={
                            "strategy": resolution.strategy,
                            "field": conflict.field,
                            "losers": resolution.losers,
                        },
                    )
                return resolution
        return None

    def _contributions(
        self,
        answers: list[ExpertAnswer],
        superseded: dict[str, tuple[str, str]],
        verifications: dict[str, VerificationResult],
        primary_domain: str,
    ) -> list[Contribution]:
        contributions: list[Contribution] = []
        for answer in answers:
            if not answer.ok:
                continue
            weight = answer.confidence
            if self.experts is not None:
                agent = self.experts.get(answer.expert_id)
                if agent is not None and primary_domain:
                    weight *= agent.spec.authority.get(primary_domain, 0.5)
            verification = verifications.get(answer.answer_id)
            if verification is not None and not verification.passed:
                weight *= 0.4  # failed its own domain checks

            winner, reason = superseded.get(answer.answer_id, ("", ""))
            contributions.append(
                Contribution(
                    expert_id=answer.expert_id or "orchestrator",
                    domain=answer.domain,
                    subtask_id=answer.subtask_id,
                    answer_id=answer.answer_id,
                    confidence=answer.confidence,
                    weight=round(weight, 6),
                    accepted=answer.answer_id not in superseded,
                    superseded_by=winner,
                    reason=reason,
                )
            )
        contributions.sort(key=lambda c: (c.accepted, c.weight), reverse=True)
        return contributions
