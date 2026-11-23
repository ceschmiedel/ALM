"""Generative backend — a chat model asked for JSON, read back as typed answers.

This is the fallback implementation of the decision contract, and the one every
installation already has: any registered chat model can serve it.  All the
questions of a request go out in one call.

It is honest about what it is.  The confidences and probabilities it returns
are the model *reporting* a number, not a distribution read from the model, so
results are marked ``calibrated=False``.  Parsing is tolerant because small
models comply with a JSON format unevenly: an answer that cannot be read
becomes the empty default rather than an exception, and the caller decides what
an empty judgment means.
"""

from __future__ import annotations

from typing import Any

from alm.core.jsonutil import dumps, extract_json_object
from alm.core.telemetry import UsageMeter
from alm.decision.spec import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    empty_answer,
)
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, Message
from alm.models.tiers import ModelTier

_FORMAT_SYSTEM = """\
You answer typed questions about the state you are given. You do not write
prose; each answer is consumed by code.

Return ONLY a JSON object with exactly one key per question id:
{formats}

Rules:
- Use only the labels listed for a question, spelled exactly as given.
- "confidence" and "probability" are numbers between 0 and 1.
- "reason" is one or two sentences.
"""

_TRUE_WORDS = {"true", "yes", "y", "sim"}
_FALSE_WORDS = {"false", "no", "n", "nao", "não"}


class GenerativeDecisionBackend:
    """Serves :class:`DecisionRequest` through the federation's chat models."""

    method = "generative"

    def __init__(self, serving: ServingRouter) -> None:
        self.serving = serving

    async def decide(
        self,
        request: DecisionRequest,
        *,
        tier: ModelTier | str | None = None,
        model_id: str = "",
        meter: UsageMeter | None = None,
        component: str = "",
    ) -> DecisionResult:
        generation = GenerationRequest(
            messages=[
                Message(role="system", content=render_system(request)),
                Message(role="user", content=render_user(request)),
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=request.max_tokens or 150 + 120 * len(request.questions),
            metadata=dict(request.metadata),
        )
        result = await self.serving.generate(
            generation,
            model_id=model_id,
            tier=tier,
            meter=meter,
            component=component,
            allow_escalation=False,
        )
        parsed = extract_json_object(result.text)
        return DecisionResult(
            answers=parse_answers(request, parsed or {}),
            parsed=bool(parsed),
            method=self.method,
            calibrated=False,
            model_id=result.model_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
        )


# -- prompt ------------------------------------------------------------------


def _format_line(question_id: str, question: Choice | Score | Noul) -> str:
    key = dumps(question_id)
    if isinstance(question, Choice) and question.multi:
        body = '{"selected": ["<label>", ...], "confidence": <0..1>, "reason": "..."}'
    elif isinstance(question, Choice):
        body = '{"choice": "<label>", "confidence": <0..1>, "reason": "..."}'
    elif isinstance(question, Score):
        body = f'{{"score": <0..{question.top}>, "confidence": <0..1>, "reason": "..."}}'
    else:
        body = '{"probability": <0..1>, "reason": "..."}'
    return f"  {key}: {body}"


def render_system(request: DecisionRequest) -> str:
    formats = "{\n" + ",\n".join(
        _format_line(qid, q) for qid, q in request.questions.items()
    ) + "\n}"
    system = _FORMAT_SYSTEM.format(formats=formats)
    return f"{request.preamble.strip()}\n\n{system}" if request.preamble.strip() else system


def _render_question(question_id: str, question: Choice | Score | Noul) -> str:
    lines = [f"### {question_id}", question.instructions.strip()]
    if isinstance(question, Choice):
        lines.append(
            "Labels (list every one that applies, most relevant first; an empty list is valid):"
            if question.multi
            else "Labels (pick exactly one):"
        )
        lines.extend(
            f"- {label}: {description}" if description else f"- {label}"
            for label, description in question.options.items()
        )
    elif isinstance(question, Score):
        lines.append("Scale:")
        lines.extend(f"- {index}: {level}" for index, level in enumerate(question.levels))
    else:
        lines.append("Answer with the probability that this is true.")
    return "\n".join(lines)


def render_user(request: DecisionRequest) -> str:
    state = request.state if isinstance(request.state, str) else dumps(request.state, indent=2)
    questions = "\n\n".join(_render_question(qid, q) for qid, q in request.questions.items())
    return f"## State\n{state}\n\n## Questions\n{questions}"


# -- parsing -----------------------------------------------------------------


def _unit(value: Any) -> float:
    try:
        return min(max(float(value or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _as_mapping(raw: Any, primary: str) -> dict[str, Any]:
    """Accept a bare value where an object was asked for."""
    if isinstance(raw, dict):
        return raw
    return {} if raw is None else {primary: raw}


def _parse_choice(question: Choice, raw: Any) -> ChoiceAnswer:
    data = _as_mapping(raw, "selected" if question.multi else "choice")
    named = data.get("selected")
    if named is None:
        named = data.get("choice", [])
    if not isinstance(named, list):
        named = [named]

    selected: list[str] = []
    for label in (str(item).strip() for item in named):
        if label in question.options and label not in selected:
            selected.append(label)
    if not question.multi:
        selected = selected[:1]

    reported = data.get("probabilities")
    probabilities = (
        {str(k): _unit(v) for k, v in reported.items() if str(k) in question.options}
        if isinstance(reported, dict)
        else {}
    )

    return ChoiceAnswer(
        choice=selected[0] if selected else "",
        selected=selected,
        probabilities=probabilities,
        # A confidence with no label attached has no referent — confident about
        # *what*?  Naming nothing valid is a zero-confidence answer by definition.
        confidence=_unit(data.get("confidence")) if selected else 0.0,
        reason=str(data.get("reason", "") or ""),
    )


def _parse_score(question: Score, raw: Any) -> ScoreAnswer:
    data = _as_mapping(raw, "score")
    try:
        score = float(data.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = min(max(score, 0.0), float(question.top))
    return ScoreAnswer(
        score=round(score, 4),
        normalised=round(score / question.top, 4),
        confidence=_unit(data.get("confidence")),
        reason=str(data.get("reason", "") or ""),
    )


def _parse_noul(raw: Any) -> NoulAnswer:
    data = _as_mapping(raw, "probability")
    value = data.get("probability", data.get("noul", data.get("value")))
    if isinstance(value, bool):
        probability = 1.0 if value else 0.0
    elif isinstance(value, str) and value.strip().lower() in _TRUE_WORDS:
        probability = 1.0
    elif isinstance(value, str) and value.strip().lower() in _FALSE_WORDS:
        probability = 0.0
    else:
        probability = _unit(value)
    return NoulAnswer(probability=probability, reason=str(data.get("reason", "") or ""))


def parse_answers(request: DecisionRequest, parsed: dict[str, Any]) -> dict[str, Answer]:
    """Read a model's JSON into typed answers, one per question.

    A missing or malformed entry yields that question's empty answer.  With a
    single question, a model that returns the answer object itself instead of
    wrapping it under the question id is understood.
    """
    answers: dict[str, Answer] = {}
    sole = len(request.questions) == 1
    for question_id, question in request.questions.items():
        if question_id in parsed:
            raw: Any = parsed[question_id]
        elif sole:
            raw = parsed
        else:
            raw = None

        if raw is None:
            answers[question_id] = empty_answer(question)
        elif isinstance(question, Choice):
            answers[question_id] = _parse_choice(question, raw)
        elif isinstance(question, Score):
            answers[question_id] = _parse_score(question, raw)
        else:
            answers[question_id] = _parse_noul(raw)
    return answers
