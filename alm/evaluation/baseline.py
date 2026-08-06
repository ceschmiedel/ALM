"""The monolithic baseline — the arm the federation must beat.

This is deliberately a *good* baseline, not a straw man.  It is the architecture
ALM proposes to replace, given every advantage that is fair: the frontier model,
a carefully written prompt, and RAG over the whole corpus.  The comparison is
only worth anything if the baseline is the thing a competent team would actually
build.

What it structurally cannot do is the point of the exercise: it has no per-agent
governance, no explicit arbitration, and no decision trail — so it scores near
zero on auditability by construction, and that is a real property of the
architecture rather than a handicap imposed here.
"""

from __future__ import annotations

import logging
from typing import Any

from alm.cmrag.retriever import CMRAGRetriever
from alm.core.telemetry import Stopwatch, UsageMeter
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier

logger = logging.getLogger(__name__)

_BASELINE_SYSTEM = """\
You are an expert enterprise analyst covering legal, financial and risk matters.
Answer the question using the retrieved context.

- Ground every claim in the retrieved context; cite fragments by their [n] index.
- If the context does not support an answer, say so rather than speculating.
- Be precise with numbers, dates and clause references.
"""


class BaselineResult:
    """One baseline answer with its accounting."""

    __slots__ = ("answer", "latency_ms", "cost_usd", "tokens", "citations", "error")

    def __init__(
        self,
        *,
        answer: str = "",
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        tokens: int = 0,
        citations: int = 0,
        error: str = "",
    ) -> None:
        self.answer = answer
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd
        self.tokens = tokens
        self.citations = citations
        self.error = error


class MonolithicBaseline:
    """One frontier model, one good prompt, RAG over everything."""

    def __init__(
        self,
        serving: ServingRouter,
        retriever: CMRAGRetriever | None = None,
        *,
        top_k: int = 8,
        max_context_chars: int = 8000,
    ) -> None:
        self.serving = serving
        self.retriever = retriever
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    async def answer(
        self,
        question: str,
        *,
        domains: list[str] | None = None,
        meter: UsageMeter | None = None,
    ) -> BaselineResult:
        """Answer one question the way the monolithic architecture would."""
        stopwatch = Stopwatch()

        blocks: list[dict[str, Any]] = []
        if self.retriever is not None:
            # Retrieval is unscoped on purpose: without a Context Graph there is
            # no principled way to narrow it per agent, and the extra noise is a
            # genuine property of the architecture being compared.
            retrieval = self.retriever.retrieve(
                question, domains=domains, top_k=self.top_k
            )
            blocks = retrieval.as_context_blocks(self.max_context_chars)

        context_section = (
            "\n\n".join(f"[{b['index']}] {b['title']}\n{b['text']}" for b in blocks)
            or "No context retrieved."
        )
        request = GenerationRequest(
            messages=[
                Message(role="system", content=_BASELINE_SYSTEM),
                Message(
                    role="user",
                    content=f"## Context\n{context_section}\n\n## Question\n{question}",
                ),
            ],
            temperature=0.0,
            max_tokens=1000,
            metadata={
                "question": question,
                "context_chunks": blocks,
                "max_sentences": 5,
            },
        )

        try:
            result = await self.serving.generate(
                request,
                tier=ModelTier.ORCHESTRATOR,
                meter=meter,
                component="eval.baseline",
            )
        except Exception as exc:
            logger.warning("Baseline call failed: %s", exc)
            return BaselineResult(latency_ms=stopwatch.stop(), error=str(exc))

        return BaselineResult(
            answer=result.text.strip(),
            latency_ms=stopwatch.stop(),
            cost_usd=result.cost_usd,
            tokens=result.total_tokens,
            citations=len(blocks),
        )
