"""Arbitration strategies, cheapest and most deterministic first.

Four ways to settle a disagreement, applied as a chain:

**Verifier** — a dedicated check against the domain's own rules (a total must
equal the sum of its parts; a clause cannot contradict another). Cheap and
deterministic wherever the rules are clear, and it produces a reason a person
can act on. Always tried first.

**Calibrated confidence** — favour the higher confidence, but only *calibrated*
confidence and only when the margin is real. A model that is confident and wrong
is worse than one that is uncertain and honest, so an uncalibrated score is
explicitly not allowed to win an argument on its own.

**Domain authority** — on the boundary between two domains, the expert whose
domain is primary for the question carries more weight. The weights live in the
Context Graph, not in code, so they are inspectable and tunable per tenant.

**LLM judge** — for genuinely ambiguous conflicts, the orchestrator sees both
answers with their context and decides, with a justification. Powerful, but it
reintroduces cost and latency: reserved for what the previous three could not
settle.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from alm.arbitration.conflict import Conflict
from alm.core.jsonutil import extract_json_object, truncate
from alm.core.telemetry import UsageMeter
from alm.experts.verifier import VerificationResult
from alm.graph.store import ContextGraph
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.protocol.answer import ExpertAnswer

logger = logging.getLogger(__name__)


class Resolution(BaseModel):
    """How one conflict was settled."""

    conflict: Conflict
    strategy: str
    winner_expert: str = ""
    winner_answer_id: str = ""
    losers: list[str] = Field(default_factory=list)
    reason: str = ""
    resolved: bool = True
    cost_usd: float = 0.0

    def line(self) -> str:
        if not self.resolved:
            return f"unresolved ({self.strategy}): {self.conflict.summary()}"
        return (
            f"{self.winner_expert} prevailed over {', '.join(self.losers)} "
            f"on '{self.conflict.field or 'conclusion'}' via {self.strategy} — {self.reason}"
        )


class ArbitrationStrategy(ABC):
    """Base class for a single arbitration mechanism."""

    name = "strategy"

    @abstractmethod
    async def resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        *,
        context: dict[str, Any],
    ) -> Resolution | None:
        """Return a resolution, or ``None`` to defer to the next strategy."""


class VerifierStrategy(ArbitrationStrategy):
    """An answer that fails its domain's own rules loses to one that passes."""

    name = "verifier"

    def __init__(self, verifications: dict[str, VerificationResult] | None = None) -> None:
        self.verifications = verifications or {}

    async def resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        *,
        context: dict[str, Any],
    ) -> Resolution | None:
        verifications = context.get("verifications", self.verifications)
        passed: list[ExpertAnswer] = []
        failed: list[ExpertAnswer] = []

        for answer_id in conflict.participants:
            answer = answers.get(answer_id)
            if answer is None:
                continue
            result = verifications.get(answer_id)
            if result is not None and not result.passed:
                failed.append(answer)
            else:
                passed.append(answer)

        # Only decisive when exactly one side survives verification.
        if not failed or not passed:
            return None

        winner = max(passed, key=lambda a: a.confidence)
        failure_reasons = "; ".join(
            verifications[a.answer_id].summary()
            for a in failed
            if a.answer_id in verifications
        )
        return Resolution(
            conflict=conflict,
            strategy=self.name,
            winner_expert=winner.expert_id,
            winner_answer_id=winner.answer_id,
            losers=[a.expert_id for a in failed],
            reason=f"the other answer failed domain verification ({failure_reasons})",
        )


class ConfidenceStrategy(ArbitrationStrategy):
    """Favour higher calibrated confidence when the margin is meaningful."""

    name = "confidence"

    def __init__(self, margin: float = 0.15) -> None:
        self.margin = margin

    async def resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        *,
        context: dict[str, Any],
    ) -> Resolution | None:
        candidates = [answers[a] for a in conflict.participants if a in answers]
        if len(candidates) < 2:
            return None

        ranked = sorted(candidates, key=lambda a: a.confidence, reverse=True)
        best, runner_up = ranked[0], ranked[1]

        if not best.calibrated:
            # Refusing to decide here is the point: an unvalidated score is not
            # evidence, and pretending otherwise is how a federation inherits a
            # confident-and-wrong expert's mistakes.
            logger.debug(
                "Confidence strategy declined: %s is not calibrated", best.expert_id
            )
            return None

        gap = best.confidence - runner_up.confidence
        if gap < self.margin:
            return None

        return Resolution(
            conflict=conflict,
            strategy=self.name,
            winner_expert=best.expert_id,
            winner_answer_id=best.answer_id,
            losers=[a.expert_id for a in ranked[1:]],
            reason=(
                f"calibrated confidence {best.confidence:.2f} exceeds "
                f"{runner_up.confidence:.2f} by {gap:.2f} (threshold {self.margin:.2f})"
            ),
        )


class AuthorityStrategy(ArbitrationStrategy):
    """Defer to the expert whose domain is primary for the question."""

    name = "authority"

    def __init__(self, graph: ContextGraph, *, min_gap: float = 0.1) -> None:
        self.graph = graph
        self.min_gap = min_gap

    async def resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        *,
        context: dict[str, Any],
    ) -> Resolution | None:
        candidates = [answers[a] for a in conflict.participants if a in answers]
        if len(candidates) < 2:
            return None

        primary_domain = (
            conflict.domain
            or context.get("primary_domain")
            or candidates[0].domain
        )
        weighted = [
            (answer, self.graph.authority_weight(answer.expert_id, primary_domain))
            for answer in candidates
        ]
        weighted.sort(key=lambda pair: pair[1], reverse=True)

        (best, best_weight), (_, runner_weight) = weighted[0], weighted[1]
        if best_weight - runner_weight < self.min_gap:
            return None

        return Resolution(
            conflict=conflict,
            strategy=self.name,
            winner_expert=best.expert_id,
            winner_answer_id=best.answer_id,
            losers=[a.expert_id for a, _ in weighted[1:]],
            reason=(
                f"'{best.expert_id}' holds authority {best_weight:.2f} over domain "
                f"'{primary_domain}' against {runner_weight:.2f}"
            ),
        )


_JUDGE_SYSTEM = """\
You are the arbitration layer of an ALM federation. Two domain specialists have
reached incompatible conclusions. Decide which one prevails.

Judge on the evidence each answer cites and on domain competence — not on tone,
length or fluency. If neither is adequately supported, say so.

Return ONLY a JSON object:
{"winner": "<expert_id>", "reason": "<one or two sentences>", "confidence": <0..1>}
"""


class JudgeStrategy(ArbitrationStrategy):
    """Last resort: the orchestrator decides, with a justification."""

    name = "judge"

    def __init__(self, serving: ServingRouter, *, enabled: bool = True) -> None:
        self.serving = serving
        self.enabled = enabled

    async def resolve(
        self,
        conflict: Conflict,
        answers: dict[str, ExpertAnswer],
        *,
        context: dict[str, Any],
    ) -> Resolution | None:
        if not self.enabled:
            return None

        candidates = [answers[a] for a in conflict.participants if a in answers]
        if len(candidates) < 2:
            return None

        meter: UsageMeter | None = context.get("meter")
        rendered = "\n\n".join(
            f"### {a.expert_id} (domain: {a.domain}, confidence: {a.confidence:.2f})\n"
            f"{truncate(a.content, 1500)}\n"
            f"Cited sources: "
            + (", ".join(c.title or c.chunk_id for c in a.citations) or "none")
            for a in candidates
        )
        request = GenerationRequest(
            messages=[
                Message(role="system", content=_JUDGE_SYSTEM),
                Message(
                    role="user",
                    content=(
                        f"Original request: {context.get('intent_text', '')}\n\n"
                        f"Point of disagreement: {conflict.description or conflict.summary()}\n\n"
                        f"{rendered}\n\n"
                        f"Valid expert_id values: "
                        f"{', '.join(a.expert_id for a in candidates)}"
                    ),
                ),
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=400,
        )

        try:
            result = await self.serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                meter=meter,
                component="arbitration.judge",
                allow_escalation=False,
            )
        except Exception as exc:
            logger.info("Judge unavailable (%s); conflict left unresolved", exc)
            return None

        parsed = extract_json_object(result.text) or {}
        winner_id = str(parsed.get("winner", "")).strip()
        winner = next((a for a in candidates if a.expert_id == winner_id), None)
        if winner is None:
            logger.info("Judge named an unknown expert %r; conflict left unresolved", winner_id)
            return None

        return Resolution(
            conflict=conflict,
            strategy=self.name,
            winner_expert=winner.expert_id,
            winner_answer_id=winner.answer_id,
            losers=[a.expert_id for a in candidates if a.answer_id != winner.answer_id],
            reason=str(parsed.get("reason", "adjudicated by the orchestrator")),
            cost_usd=result.cost_usd,
        )


def build_strategies(
    names: list[str],
    *,
    graph: ContextGraph,
    serving: ServingRouter | None,
    confidence_margin: float = 0.15,
    judge_enabled: bool = True,
) -> list[ArbitrationStrategy]:
    """Build the configured strategy chain, skipping any that cannot run."""
    built: list[ArbitrationStrategy] = []
    for name in names:
        key = name.strip().lower()
        if key == "verifier":
            built.append(VerifierStrategy())
        elif key == "confidence":
            built.append(ConfidenceStrategy(margin=confidence_margin))
        elif key == "authority":
            built.append(AuthorityStrategy(graph))
        elif key == "judge":
            if serving is not None:
                built.append(JudgeStrategy(serving, enabled=judge_enabled))
        else:
            logger.warning("Unknown arbitration strategy %r; ignoring", name)
    return built
