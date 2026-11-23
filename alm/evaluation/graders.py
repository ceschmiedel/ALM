"""Graders — turning an answer into a score.

Deterministic graders are preferred wherever the domain admits them: they cost
nothing, they are reproducible across runs, and a regression is attributable to
the system rather than to a judging model that drifted.  The rubric grader
exists for open-ended cases, and it is honest about being the weakest link —
it reports that the score came from a model.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from alm.core.jsonutil import truncate
from alm.decision import DecisionEngine, DecisionRequest, Noul, Score
from alm.evaluation.dataset import EvalCase
from alm.models.serving import ServingRouter
from alm.models.tiers import ModelTier

logger = logging.getLogger(__name__)

_FULL_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class GradeResult(BaseModel):
    """The score for one case."""

    score: float = 0.0
    correct: bool = False
    grader: str = ""
    reason: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


def _normalise(text: str) -> str:
    """Casefold and strip accents so grading is not defeated by diacritics."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def grade_keywords(case: EvalCase, answer: str) -> GradeResult:
    """Fraction of required keywords present in the answer."""
    keywords = case.keywords or [w for w in case.expected.split() if len(w) > 6]
    if not keywords:
        return GradeResult(
            grader="keywords", reason="no keywords defined; case is ungradeable"
        )

    haystack = _normalise(answer)
    hits = [k for k in keywords if _normalise(k) in haystack]
    ratio = len(hits) / len(keywords)
    return GradeResult(
        score=round(ratio, 4),
        correct=ratio >= case.min_keyword_ratio,
        grader="keywords",
        reason=f"matched {len(hits)}/{len(keywords)} keyword(s)",
        detail={"matched": hits, "missing": [k for k in keywords if k not in hits]},
    )


def grade_exact(case: EvalCase, answer: str) -> GradeResult:
    match = _normalise(answer) == _normalise(case.expected)
    return GradeResult(
        score=1.0 if match else 0.0,
        correct=match,
        grader="exact",
        reason="exact match" if match else "does not match the expected answer",
    )


def grade_contains(case: EvalCase, answer: str) -> GradeResult:
    match = _normalise(case.expected) in _normalise(answer)
    return GradeResult(
        score=1.0 if match else 0.0,
        correct=match,
        grader="contains",
        reason="expected text present" if match else "expected text absent",
    )


def grade_numeric(case: EvalCase, answer: str) -> GradeResult:
    """Whether the expected number appears within tolerance."""
    if case.expected_number is None:
        return GradeResult(grader="numeric", reason="case declares no expected_number")

    candidates = [
        float(m) for m in _FULL_NUMBER_RE.findall((answer or "").replace(",", ""))
    ]
    if not candidates:
        return GradeResult(grader="numeric", reason="answer contains no number")

    target = case.expected_number
    scale = max(abs(target), 1e-9)
    closest = min(candidates, key=lambda v: abs(v - target))
    relative = abs(closest - target) / scale
    correct = relative <= case.tolerance
    return GradeResult(
        score=1.0 if correct else max(0.0, round(1.0 - relative, 4)),
        correct=correct,
        grader="numeric",
        reason=(
            f"closest number {closest} vs expected {target} "
            f"({relative:.2%} off, tolerance {case.tolerance:.2%})"
        ),
        detail={"closest": closest, "candidates": candidates[:10]},
    )


_RUBRIC_PREAMBLE = """\
You are grading one answer against a reference. Judge only whether the answer is
substantively correct for the domain — not its style, length or fluency."""

#: The ordered scale the rubric grader scores on, worst first.
RUBRIC_LEVELS = [
    "incorrect — contradicts or misses the reference",
    "mostly incorrect — a relevant fragment, wrong conclusion",
    "partially correct — right direction, material gaps or errors",
    "mostly correct — matches the reference with minor omissions",
    "fully correct — substantively equivalent to the reference",
]


async def grade_rubric(
    case: EvalCase, answer: str, *, serving: ServingRouter
) -> GradeResult:
    """Model-graded, for cases no deterministic rule can score.

    Two independent judgments in one call: where the answer sits on the scale,
    and whether it is correct.  They are asked separately because "how good" and
    "good enough" are different questions, and deriving one from the other
    hides a threshold inside the grader.
    """
    rubric = case.rubric or "Is the answer substantively correct?"
    request = DecisionRequest(
        state={
            "question": case.question,
            "reference_answer": case.expected,
            "rubric": rubric,
            "answer_to_grade": truncate(answer, 3000),
        },
        questions={
            "quality": Score(
                instructions="How well does the answer satisfy the rubric against the reference?",
                levels=RUBRIC_LEVELS,
            ),
            "correct": Noul(
                instructions="The answer to grade is substantively correct per the rubric.",
            ),
        },
        preamble=_RUBRIC_PREAMBLE,
        max_tokens=300,
    )
    try:
        result = await DecisionEngine(serving).decide(
            request,
            tier=ModelTier.ORCHESTRATOR,
            component="eval.rubric",
        )
    except Exception as exc:
        logger.info("Rubric grading unavailable (%s)", exc)
        return GradeResult(
            grader="rubric", reason=f"grader model unavailable: {exc}"
        )

    quality = result.score("quality")
    return GradeResult(
        score=quality.normalised,
        correct=result.noul("correct").value,
        grader="rubric",
        reason=quality.reason or result.noul("correct").reason,
        detail={"model_graded": True, "method": result.method},
    )


async def grade(
    case: EvalCase, answer: str, *, serving: ServingRouter | None = None
) -> GradeResult:
    """Dispatch to the grader the case asks for."""
    kind = (case.grader or "keywords").strip().lower()
    if kind == "exact":
        return grade_exact(case, answer)
    if kind == "contains":
        return grade_contains(case, answer)
    if kind == "numeric":
        return grade_numeric(case, answer)
    if kind == "rubric":
        if serving is None:
            # Degrade to keywords rather than silently scoring zero, and say so.
            fallback = grade_keywords(case, answer)
            fallback.reason += " (rubric grading needs an orchestrator model)"
            return fallback
        return await grade_rubric(case, answer, serving=serving)
    return grade_keywords(case, answer)
