"""Decision benchmark — accuracy and calibration over labelled typed decisions.

A decision model is only as useful as its probabilities are honest, so the
benchmark reports calibration next to accuracy:

* **accuracy** — the most probable label is the expected one.
* **Brier** — squared error of the whole distribution against the truth, summed
  over the labels (0 is perfect; a uniform guess over two labels scores 0.5).
* **ECE** — expected calibration error of the top label: predictions are put in
  ten equal-width confidence bins and the gap between mean confidence and
  accuracy is averaged, weighted by bin size.  0.10 means the stated confidence
  is off by ten points on average.

Items use the JevBench record shape (MIT), so public decision-model benchmarks
and ALM's own sets run through the same code::

    {"id": "...", "state": "...", "labels": ["no", "yes"], "expected": "no",
     "question": {"type": "noul", "instructions": "...", "criteria": {...}}}

``choice`` criteria map label → description, ``score`` criteria list the levels
lowest first (labels are their indices), ``noul`` labels are ``no``/``yes``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alm.decision.engine import DecisionEngine
from alm.decision.spec import Choice, DecisionRequest, DecisionResult, Noul, Score

NOUL_LABELS = ["no", "yes"]


class BenchItem(BaseModel):
    """One labelled decision."""

    id: str
    state: Any
    question: dict[str, Any]
    labels: list[str] = Field(default_factory=list)
    expected: str
    family: str = ""

    @property
    def type(self) -> str:
        return str(self.question.get("type", ""))


def question_from_wire(question: dict[str, Any]) -> Choice | Score | Noul:
    """Build a typed question from the ``/v1/systemone`` wire shape."""
    kind = str(question.get("type", "")).lower()
    instructions = str(question.get("instructions", ""))
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("a choice question needs criteria: an object of label → description")
        return Choice(
            instructions=instructions,
            options={str(k): str(v or "") for k, v in criteria.items()},
        )
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("a score question needs criteria: an ordered list of at least two levels")
        return Score(instructions=instructions, levels=[str(level) for level in criteria])
    if kind == "noul":
        meaning = criteria if isinstance(criteria, dict) else {}
        return Noul(
            instructions=instructions,
            when_true=str(meaning.get("true", "") or ""),
            when_false=str(meaning.get("false", "") or ""),
        )
    raise ValueError(f"unknown question type {question.get('type')!r}; expected choice, score or noul")


def load_items(path: str | Path) -> list[BenchItem]:
    """Read a JSON Lines benchmark file."""
    items: list[BenchItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["expected"] = str(record["expected"])
        kind = str(record.get("question", {}).get("type", ""))
        if not record.get("labels"):
            criteria = record["question"].get("criteria")
            if kind == "noul":
                record["labels"] = NOUL_LABELS
            elif kind == "score":
                record["labels"] = [str(i) for i in range(len(criteria))]
            else:
                record["labels"] = list(criteria)
        record["labels"] = [str(label) for label in record["labels"]]
        items.append(BenchItem.model_validate(record))
    return items


def probabilities_for(item: BenchItem, result: DecisionResult, question_id: str = "decision") -> dict[str, float]:
    """The distribution over the item's labels, or ``{}`` when the backend gave none."""
    if item.type == "noul":
        p = result.noul(question_id).probability
        return {"yes": p, "no": 1.0 - p}
    if item.type == "score":
        distribution = result.score(question_id).probabilities
        return {str(i): p for i, p in enumerate(distribution)}
    return dict(result.choice(question_id).probabilities)


# -- metrics -----------------------------------------------------------------


def brier(probabilities: dict[str, float], expected: str, labels: Sequence[str]) -> float:
    """Multi-class Brier score: squared error summed over the label set."""
    return sum(
        (float(probabilities.get(label, 0.0)) - (1.0 if label == expected else 0.0)) ** 2
        for label in labels
    )


def expected_calibration_error(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> float:
    """Top-label ECE over ``(confidence, correct)`` pairs, equal-width bins."""
    counts = [0] * bins
    confidence_sum = [0.0] * bins
    correct_sum = [0] * bins
    for confidence, correct in pairs:
        confidence = min(max(float(confidence), 0.0), 1.0)
        index = min(int(confidence * bins), bins - 1)
        counts[index] += 1
        confidence_sum[index] += confidence
        correct_sum[index] += int(correct)
    total = sum(counts)
    if not total:
        return 0.0
    return sum(
        (counts[i] / total) * abs(correct_sum[i] / counts[i] - confidence_sum[i] / counts[i])
        for i in range(bins)
        if counts[i]
    )


class BenchRow(BaseModel):
    id: str
    type: str
    family: str = ""
    expected: str
    predicted: str = ""
    confidence: float = 0.0
    correct: bool = False
    brier: float | None = None
    error: str = ""


class BenchSummary(BaseModel):
    n: int = 0
    errors: int = 0
    accuracy: float = 0.0
    brier: float | None = None
    ece: float | None = None
    #: Accuracy of always answering the most common expected label.
    majority_baseline: float = 0.0


class BenchReport(BaseModel):
    method: str = ""
    model_id: str = ""
    overall: BenchSummary = Field(default_factory=BenchSummary)
    by_type: dict[str, BenchSummary] = Field(default_factory=dict)
    latency_ms_p50: float = 0.0
    prompt_tokens: int = 0
    rows: list[BenchRow] = Field(default_factory=list)


def summarise(rows: Sequence[BenchRow]) -> BenchSummary:
    answered = [r for r in rows if not r.error]
    if not rows:
        return BenchSummary()
    scored = [r for r in answered if r.brier is not None]
    expected = [r.expected for r in rows]
    majority = max(expected.count(label) for label in set(expected)) / len(rows)
    return BenchSummary(
        n=len(rows),
        errors=len(rows) - len(answered),
        # An item the backend could not answer is a wrong answer, not a skipped one.
        accuracy=round(sum(r.correct for r in answered) / len(rows), 4),
        brier=round(sum(r.brier for r in scored) / len(scored), 4) if scored else None,
        ece=round(expected_calibration_error((r.confidence, r.correct) for r in scored), 4)
        if scored
        else None,
        majority_baseline=round(majority, 4),
    )


async def run_bench(
    items: Sequence[BenchItem],
    engine: DecisionEngine,
    *,
    tier: str | None = None,
    model_id: str = "",
) -> BenchReport:
    """Run every item through the engine and score it."""
    rows: list[BenchRow] = []
    latencies: list[float] = []
    report = BenchReport()
    for item in items:
        row = BenchRow(id=item.id, type=item.type, family=item.family, expected=item.expected)
        try:
            request = DecisionRequest(
                state=item.state, questions={"decision": question_from_wire(item.question)}
            )
            started = time.perf_counter()
            result = await engine.decide(request, tier=tier, model_id=model_id, component="decision.bench")
            latencies.append((time.perf_counter() - started) * 1000.0)
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"[:300]
            rows.append(row)
            continue

        report.method = result.method
        report.model_id = result.model_id
        report.prompt_tokens += result.prompt_tokens
        probabilities = probabilities_for(item, result)
        if item.type == "noul":
            row.predicted = "yes" if probabilities["yes"] >= 0.5 else "no"
            row.confidence = max(probabilities.values())
        elif item.type == "score":
            # A score is right when its most probable level is the expected one.
            row.predicted = (
                max(probabilities, key=probabilities.__getitem__)
                if probabilities
                else str(round(result.score("decision").score))
            )
            row.confidence = result.score("decision").confidence
        else:
            row.predicted = result.choice("decision").choice
            row.confidence = result.choice("decision").confidence
        row.correct = row.predicted == item.expected
        if probabilities:
            row.brier = round(brier(probabilities, item.expected, item.labels), 6)
        rows.append(row)

    report.rows = rows
    report.overall = summarise(rows)
    report.by_type = {
        kind: summarise([r for r in rows if r.type == kind])
        for kind in sorted({r.type for r in rows})
    }
    if latencies:
        report.latency_ms_p50 = round(sorted(latencies)[len(latencies) // 2], 1)
    return report
