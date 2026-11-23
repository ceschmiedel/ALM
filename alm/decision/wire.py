"""The ``/v1/systemone`` wire format.

Hosted decision models and the open tooling around them (SDKs, benchmarks)
speak one request shape — a state plus typed questions — so ALM answers it too.
A benchmark or client written for that shape can point at an ALM server and
measure the federation's own decision model without an adapter.
"""

from __future__ import annotations

from typing import Any

from alm.decision.bench import question_from_wire
from alm.decision.spec import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


def request_from_wire(payload: dict[str, Any]) -> DecisionRequest:
    """Build a :class:`DecisionRequest` from a wire payload.  Raises ``ValueError``."""
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of id → question")
    typed = {}
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {question_id!r} must be an object")
        try:
            typed[str(question_id)] = question_from_wire(question)
        except ValueError as exc:
            raise ValueError(f"question {question_id!r}: {exc}") from exc
    return DecisionRequest(state=payload.get("state", ""), questions=typed)


def answers_to_wire(request: DecisionRequest, result: DecisionResult) -> dict[str, Any]:
    """Render typed answers in the wire shape, one entry per question id."""
    out: dict[str, Any] = {}
    for question_id, question in request.questions.items():
        answer = result.answers.get(question_id)
        if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
            out[question_id] = {
                "type": "choice",
                "choice": answer.choice,
                "probabilities": answer.probabilities,
                "confidence": answer.confidence,
            }
        elif isinstance(question, Score) and isinstance(answer, ScoreAnswer):
            out[question_id] = {
                "type": "score",
                "score": answer.score,
                "probabilities": {str(i): p for i, p in enumerate(answer.probabilities)},
                "confidence": answer.confidence,
                "legend": {str(i): level for i, level in enumerate(question.levels)},
            }
        elif isinstance(answer, NoulAnswer):
            out[question_id] = {"type": "noul", "noul": answer.probability}
    return out
