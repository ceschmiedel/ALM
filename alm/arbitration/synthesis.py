"""Synthesis — composing the final answer, with the trail attached.

Once conflicts are settled, the answer is composed from the accepted
contributions.  For an enterprise client the non-negotiable requirement is not
the prose: it is that the answer arrives with **who contributed what, and why
one line prevailed over another**.  That explainability is not decoration; in a
regulated vertical it is what makes the decision auditable and therefore
adoptable.

Two composition paths, and both produce the same provenance:

* **model synthesis** — the orchestrator weaves the contributions into one
  coherent answer.  Better prose, costs a call.
* **deterministic composition** — contributions are ordered by weight and
  assembled into sections.  Free, reproducible, and always available, which
  matters because it means the federation still answers when the frontier model
  is unreachable.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from alm.arbitration.arbiter import ArbitrationResult
from alm.core.jsonutil import truncate
from alm.core.telemetry import UsageMeter
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier
from alm.protocol.answer import Citation, ExpertAnswer
from alm.protocol.trace import ExecutionTrace, Layer

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = """\
You are the synthesis layer of an ALM federation. Several domain specialists have
answered parts of one request. Compose their findings into a single coherent answer.

Rules:
- Use ONLY what the specialists reported. Add no facts of your own.
- Attribute substantive claims to the specialist that made them.
- Where a disagreement was arbitrated, state the conclusion that prevailed and
  note that the alternative was considered and why it did not prevail.
- If the specialists could not answer part of the request, say so plainly.
- Answer the original request directly. No preamble.
"""


class SynthesisResult(BaseModel):
    """The final answer plus its provenance."""

    answer: str = ""
    method: str = "deterministic"
    citations: list[Citation] = Field(default_factory=list)
    contributing_experts: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    cost_usd: float = 0.0
    provenance: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "contributing_experts": self.contributing_experts,
            "citations": [c.model_dump(mode="json") for c in self.citations],
            "provenance": self.provenance,
        }


class Synthesizer:
    """Composes the final answer from arbitrated contributions."""

    def __init__(self, serving: ServingRouter | None = None) -> None:
        self.serving = serving

    async def synthesize(
        self,
        answers: list[ExpertAnswer],
        arbitration: ArbitrationResult,
        *,
        intent_text: str,
        requested_outputs: list[str] | None = None,
        trace: ExecutionTrace | None = None,
        meter: UsageMeter | None = None,
        prefer_model: bool = True,
    ) -> SynthesisResult:
        accepted = [
            a
            for a in answers
            if a.ok and a.answer_id in set(arbitration.accepted_answer_ids)
        ]
        if not accepted:
            return self._empty(answers, arbitration)

        # A single accepted answer needs no synthesis; passing it through a model
        # would only risk paraphrasing away the specialist's precision.
        if len(accepted) == 1 and not arbitration.resolutions:
            return self._passthrough(accepted[0], arbitration)

        result: SynthesisResult | None = None
        if prefer_model and self.serving is not None:
            result = await self._synthesize_with_model(
                accepted, arbitration, intent_text, requested_outputs or [], meter
            )
        if result is None:
            result = self._compose(accepted, arbitration, intent_text)

        result.citations = _merge_citations(accepted)
        result.contributing_experts = [a.expert_id or "orchestrator" for a in accepted]
        result.confidence = _aggregate_confidence(accepted, arbitration)
        result.provenance = arbitration.audit_lines()

        if trace is not None:
            trace.emit(
                Layer.L5_ARBITRATION,
                (
                    f"synthesised the final answer from {len(accepted)} contribution(s) "
                    f"via {result.method} (confidence {result.confidence:.2f})"
                ),
                category="synthesis",
                detail={
                    "experts": result.contributing_experts,
                    "citations": len(result.citations),
                    "resolutions": len(arbitration.resolutions),
                },
            )
        return result

    # -- model path --------------------------------------------------------

    async def _synthesize_with_model(
        self,
        accepted: list[ExpertAnswer],
        arbitration: ArbitrationResult,
        intent_text: str,
        requested_outputs: list[str],
        meter: UsageMeter | None,
    ) -> SynthesisResult | None:
        rendered = "\n\n".join(
            f"### {a.expert_id or 'orchestrator'} — {a.domain} "
            f"(confidence {a.confidence:.2f})\n{truncate(a.content, 2000)}"
            for a in accepted
        )
        arbitration_note = ""
        if arbitration.resolutions:
            arbitration_note = "\n\n## Arbitration outcomes\n" + "\n".join(
                f"- {r.line()}" for r in arbitration.resolutions
            )
        outputs_note = (
            f"\n\nThe requester asked specifically for: {', '.join(requested_outputs)}."
            if requested_outputs
            else ""
        )

        request = GenerationRequest(
            messages=[
                Message(role="system", content=_SYNTHESIS_SYSTEM),
                Message(
                    role="user",
                    content=(
                        f"## Original request\n{intent_text}\n\n"
                        f"## Specialist findings\n{rendered}"
                        f"{arbitration_note}{outputs_note}"
                    ),
                ),
            ],
            temperature=0.1,
            max_tokens=1200,
            metadata={
                "question": intent_text,
                "context_chunks": [{"text": a.content} for a in accepted],
                "max_sentences": 6,
            },
        )
        try:
            result = await self.serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                meter=meter,
                component="arbitration.synthesis",
                allow_escalation=False,
            )
        except Exception as exc:
            logger.info("Model synthesis unavailable (%s); composing deterministically", exc)
            return None

        if not result.text.strip():
            return None

        return SynthesisResult(
            answer=result.text.strip(), method="model", cost_usd=result.cost_usd
        )

    # -- deterministic path ------------------------------------------------

    def _compose(
        self,
        accepted: list[ExpertAnswer],
        arbitration: ArbitrationResult,
        intent_text: str,
    ) -> SynthesisResult:
        """Assemble sections ordered by arbitration weight."""
        weights = {c.answer_id: c.weight for c in arbitration.contributions}
        ordered = sorted(
            accepted, key=lambda a: weights.get(a.answer_id, a.confidence), reverse=True
        )

        sections: list[str] = []
        for answer in ordered:
            heading = f"**{answer.expert_id or 'orchestrator'}** ({answer.domain})"
            sections.append(f"{heading}\n{answer.content.strip()}")

        body = "\n\n".join(sections)
        if arbitration.resolutions:
            body += "\n\n**Arbitration**\n" + "\n".join(
                f"- {r.line()}" for r in arbitration.resolutions
            )
        if arbitration.unresolved:
            body += "\n\n**Unresolved disagreements**\n" + "\n".join(
                f"- {c.summary()}" for c in arbitration.unresolved
            )
        return SynthesisResult(answer=body, method="deterministic")

    # -- degenerate cases --------------------------------------------------

    @staticmethod
    def _passthrough(answer: ExpertAnswer, arbitration: ArbitrationResult) -> SynthesisResult:
        return SynthesisResult(
            answer=answer.content.strip(),
            method="single_expert",
            citations=list(answer.citations),
            contributing_experts=[answer.expert_id or "orchestrator"],
            confidence=answer.confidence,
            provenance=arbitration.audit_lines(),
        )

    @staticmethod
    def _empty(answers: list[ExpertAnswer], arbitration: ArbitrationResult) -> SynthesisResult:
        """Report failure honestly rather than inventing an answer."""
        errors = [a.error for a in answers if a.error]
        detail = f" ({'; '.join(errors[:3])})" if errors else ""
        return SynthesisResult(
            answer=(
                "The federation could not produce an answer: no specialist returned a "
                f"usable result{detail}."
            ),
            method="none",
            confidence=0.0,
            provenance=arbitration.audit_lines(),
        )


def _merge_citations(answers: list[ExpertAnswer]) -> list[Citation]:
    """Union of cited fragments, best score per fragment, strongest first."""
    best: dict[str, Citation] = {}
    for answer in answers:
        for citation in answer.citations:
            key = citation.chunk_id or f"{citation.document_id}:{citation.snippet[:40]}"
            current = best.get(key)
            if current is None or citation.score > current.score:
                best[key] = citation
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def _aggregate_confidence(
    answers: list[ExpertAnswer], arbitration: ArbitrationResult
) -> float:
    """Confidence in the composed answer.

    Weight-averaged over contributions, then penalised for anything left
    unresolved — an answer built on an unsettled disagreement genuinely deserves
    less trust, and saying so is more useful than a reassuring number.
    """
    if not answers:
        return 0.0
    weights = {c.answer_id: c.weight for c in arbitration.contributions}
    total_weight = sum(max(weights.get(a.answer_id, a.confidence), 1e-6) for a in answers)
    weighted = sum(
        a.confidence * max(weights.get(a.answer_id, a.confidence), 1e-6) for a in answers
    )
    base = weighted / total_weight if total_weight else 0.0
    if arbitration.unresolved:
        base *= max(0.5, 1.0 - 0.2 * len(arbitration.unresolved))
    return round(min(max(base, 0.0), 1.0), 4)
