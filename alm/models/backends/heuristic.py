"""Extractive backend — runs the whole pipeline without a language model.

This is **not a language model** and does not pretend to be one.  It is a
deterministic extractive responder: given a question and retrieved context, it
scores candidate sentences by lexical overlap with the question and returns the
best ones, with a confidence derived from that overlap.

It exists for two honest reasons:

* **The demo runs with no API key and no GPU.**  Routing, retrieval, DAG
  execution, arbitration, governance and evaluation are all real code paths
  exercised end to end; only the generation step is extractive.
* **Evaluation needs a fixed point.**  A deterministic generator makes the
  consistency metric measurable and makes harness regressions attributable to
  the orchestration rather than to sampling noise.

Never register it as an expert's model in production — `alm doctor` warns when
it finds one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from alm.core.jsonutil import dumps
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD_RE = re.compile(r"[\wÀ-ÿ]{3,}")
_NUMBER_RE = re.compile(r"-?\d[\d.,]*")

# Function words carry no discriminative signal for overlap scoring.
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have",
    "has", "had", "not", "但", "you", "your", "our", "its", "their", "which", "what",
    "when", "where", "who", "how", "why", "shall", "will", "would", "should", "can",
    "could", "may", "might", "must", "any", "all", "some", "such", "than", "then",
    "there", "here", "into", "onto", "upon", "under", "over", "about",
    "que", "para", "com", "uma", "dos", "das", "por", "como", "mais", "nao", "não",
    "sobre", "entre", "quando", "onde", "qual", "quais", "seu", "sua", "esse", "essa",
}


def _tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if w.lower() not in _STOPWORDS
    }


def _sentences(text: str) -> list[str]:
    parts = [s.strip(" \t-•*") for s in _SENTENCE_RE.split(text or "")]
    return [s for s in parts if len(s) > 20]


class HeuristicBackend(ChatBackend):
    """Deterministic extractive responder over the supplied context."""

    name = "heuristic"

    #: How many sentences to include in a free-text answer.
    default_sentences = 3

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        question = self._question(request)
        context = self._context(request)
        query_tokens = _tokens(question)

        ranked = self._rank(context, query_tokens)
        top = [sentence for sentence, _ in ranked[: self._limit(request)]]
        confidence = self._confidence(ranked, query_tokens)

        if request.json_mode:
            text = dumps(self._structured(request, top, confidence, question), indent=2)
        else:
            text = self._prose(top, question)

        return GenerationResult(
            text=text,
            finish_reason="stop",
            raw={
                "extractive": True,
                "candidates_considered": len(ranked),
                "confidence": confidence,
            },
        )

    # -- input extraction --------------------------------------------------

    def _question(self, request: GenerationRequest) -> str:
        explicit = request.metadata.get("question")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return request.prompt_text()

    def _context(self, request: GenerationRequest) -> list[str]:
        """Prefer structured context passed by the caller over prompt scraping."""
        chunks = request.metadata.get("context_chunks")
        if isinstance(chunks, list) and chunks:
            out: list[str] = []
            for chunk in chunks:
                text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
                out.extend(_sentences(str(text or "")))
            if out:
                return out
        return _sentences(request.prompt_text())

    def _limit(self, request: GenerationRequest) -> int:
        return int(request.metadata.get("max_sentences", self.default_sentences))

    # -- scoring -----------------------------------------------------------

    @staticmethod
    def _rank(sentences: Sequence[str], query_tokens: set[str]) -> list[tuple[str, float]]:
        if not query_tokens:
            return [(s, 0.0) for s in sentences]
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()
        for sentence in sentences:
            key = sentence[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            tokens = _tokens(sentence)
            if not tokens:
                continue
            overlap = len(tokens & query_tokens)
            if overlap == 0:
                continue
            # Normalise by the query so long sentences do not win by volume.
            scored.append((sentence, overlap / len(query_tokens)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    @staticmethod
    def _confidence(ranked: Sequence[tuple[str, float]], query_tokens: set[str]) -> float:
        """Coverage of the question by the best evidence found.

        Deliberately conservative: with no supporting context the score is low,
        which is what should happen — an extractive responder with nothing to
        extract from genuinely does not know.
        """
        if not ranked or not query_tokens:
            return 0.1
        best = ranked[0][1]
        breadth = min(len(ranked), 5) / 5.0
        return round(min(0.95, 0.25 + 0.55 * min(best, 1.0) + 0.20 * breadth), 4)

    # -- output shaping ----------------------------------------------------

    @staticmethod
    def _prose(sentences: Sequence[str], question: str) -> str:
        if not sentences:
            return (
                "No supporting passage in the retrieved context addresses this question."
            )
        body = " ".join(s.rstrip(".") + "." for s in sentences)
        return body

    def _structured(
        self,
        request: GenerationRequest,
        sentences: Sequence[str],
        confidence: float,
        question: str,
    ) -> dict[str, Any]:
        """Fill the requested schema with extracted material.

        Types are honoured so that downstream validation, verifier rules and
        arbitration all receive the shapes they expect.
        """
        schema = request.response_schema or {}
        properties: dict[str, Any] = schema.get("properties", {}) if schema else {}
        answer_text = self._prose(sentences, question)

        if not properties:
            return {"answer": answer_text, "confidence": confidence}

        out: dict[str, Any] = {}
        for field, definition in properties.items():
            field_type = definition.get("type", "string")
            lowered = field.lower()
            choices = definition.get("enum")

            if choices:
                # A closed vocabulary must be honoured, or downstream verifier
                # rules and cross-expert aggregation break on a value that is
                # merely plausible prose.
                out[field] = self._closest_choice(choices, sentences)
            elif lowered in {"confidence", "certainty", "score"}:
                out[field] = confidence
            elif field_type == "array":
                out[field] = list(sentences)
            elif field_type in {"number", "integer"}:
                numbers = _NUMBER_RE.findall(" ".join(sentences))
                value = 0.0
                if numbers:
                    try:
                        value = float(numbers[0].replace(",", ""))
                    except ValueError:
                        value = 0.0
                out[field] = int(value) if field_type == "integer" else value
            elif field_type == "boolean":
                out[field] = bool(sentences)
            elif field_type == "object":
                out[field] = {}
            else:
                out[field] = answer_text
        return out

    @staticmethod
    def _closest_choice(choices: list[Any], sentences: Sequence[str]) -> Any:
        """Pick the enum member the extracted text actually supports."""
        if not choices:
            return ""
        haystack = " ".join(sentences).lower()
        best = None
        best_hits = 0
        for choice in choices:
            hits = haystack.count(str(choice).lower())
            if hits > best_hits:
                best, best_hits = choice, hits
        # No evidence for any member: return the first, which by convention is
        # the least alarming, rather than inventing a severity.
        return best if best is not None else choices[0]

    async def health(self) -> bool:
        return True
