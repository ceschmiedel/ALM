"""The six metrics the ALM thesis is judged on.

The federation must win **on the set**, not on one number.  Each metric exists
because it captures a way the architecture could be quietly failing:

``domain_accuracy``
    Quality on the real task, against a domain gold standard. The sovereign
    metric — everything else is a cost of achieving it.
``cost_per_query``
    Total infrastructure cost per answer, including serving the federation, not
    just a token price. This is where a federation of small models is supposed
    to win, and where a naive comparison flatters it.
``latency_p95``
    The 95th percentile, not the mean. The tail is what a user actually feels,
    and averaging hides exactly the behaviour that makes a system unusable.
``fallback_rate``
    Share of requests escalated to the LLM. Measures expert coverage. Too high
    means the specialists are not carrying the load; too low can mean the
    router is gambling on cases it should escalate.
``consistency``
    Variation in behaviour between runs and between experts. A risk specific to
    federations that a monolith simply does not have.
``auditability``
    Completeness of the decision trail. The one dimension where the federation
    wins by construction — and it still has to be measured, not asserted.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field


class CaseResult(BaseModel):
    """The outcome of one evaluation case in one arm."""

    case_id: str
    domain: str = ""
    question: str = ""
    answer: str = ""
    expected: str = ""

    score: float = 0.0
    correct: bool = False
    grader: str = ""
    grader_reason: str = ""

    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    used_fallback: bool = False
    trace_complete: bool = False
    experts: list[str] = Field(default_factory=list)
    error: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class ArmMetrics(BaseModel):
    """Aggregated metrics for one arm of the experiment."""

    arm: str = ""
    cases: int = 0

    domain_accuracy: float = 0.0
    mean_score: float = 0.0
    cost_per_query: float = 0.0
    total_cost_usd: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    fallback_rate: float = 0.0
    consistency: float = 0.0
    auditability: float = 0.0

    tokens_per_query: float = 0.0
    errors: int = 0
    by_domain: dict[str, float] = Field(default_factory=dict)

    def to_row(self) -> dict[str, float]:
        return {
            "domain accuracy": self.domain_accuracy,
            "cost per query": self.cost_per_query,
            "latency p95 (ms)": self.latency_p95_ms,
            "fallback rate": self.fallback_rate,
            "consistency": self.consistency,
            "auditability": self.auditability,
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with the tens-of-cases sets a domain
    evaluation realistically has, interpolation invents a value between two
    observations and reads as more precise than the data supports.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))
    return round(ordered[index], 3)


def compute_consistency(results: Sequence[CaseResult]) -> float:
    """How uniformly the arm behaves across cases.

    Derived from the spread of per-case scores: an arm that is excellent on some
    cases and useless on others is less trustworthy than one that is uniformly
    good, even at the same mean. Reported as ``1 - normalised stdev``.
    """
    scores = [r.score for r in results]
    if len(scores) < 2:
        return 1.0
    spread = statistics.pstdev(scores)
    # Scores live in [0,1], where the maximum possible stdev is 0.5.
    return round(max(0.0, 1.0 - min(spread / 0.5, 1.0)), 4)


def compute_auditability(results: Sequence[CaseResult]) -> float:
    """Share of answers that arrived with a complete decision trail."""
    if not results:
        return 0.0
    return round(sum(1 for r in results if r.trace_complete) / len(results), 4)


def aggregate(arm: str, results: Sequence[CaseResult]) -> ArmMetrics:
    """Roll per-case results into the six comparison metrics."""
    if not results:
        return ArmMetrics(arm=arm)

    latencies = [r.latency_ms for r in results]
    scored = [r for r in results if not r.error]

    by_domain: dict[str, list[float]] = {}
    for result in results:
        by_domain.setdefault(result.domain or "-", []).append(1.0 if result.correct else 0.0)

    return ArmMetrics(
        arm=arm,
        cases=len(results),
        domain_accuracy=round(sum(1 for r in results if r.correct) / len(results), 4),
        mean_score=round(sum(r.score for r in results) / len(results), 4),
        total_cost_usd=round(sum(r.cost_usd for r in results), 8),
        cost_per_query=round(sum(r.cost_usd for r in results) / len(results), 8),
        latency_mean_ms=round(sum(latencies) / len(latencies), 3),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        fallback_rate=round(sum(1 for r in results if r.used_fallback) / len(results), 4),
        consistency=compute_consistency(scored),
        auditability=compute_auditability(results),
        tokens_per_query=round(sum(r.tokens for r in results) / len(results), 2),
        errors=sum(1 for r in results if r.error),
        by_domain={
            domain: round(sum(values) / len(values), 4)
            for domain, values in by_domain.items()
        },
    )


class Comparison(BaseModel):
    """Federation against baseline, and the verdict that follows."""

    federation: ArmMetrics
    baseline: ArmMetrics
    deltas: dict[str, float] = Field(default_factory=dict)
    wins: dict[str, bool] = Field(default_factory=dict)
    verdict: str = ""
    recommendation: str = ""

    @property
    def federation_wins(self) -> bool:
        return self.verdict == "federation"


#: Metrics where a lower value is better.
_LOWER_IS_BETTER = {"cost per query", "latency p95 (ms)"}

#: Metrics that decide the verdict.  Fallback rate is reported but excluded: it
#: is a coverage diagnostic, and the baseline has no fallback to compare with.
_DECIDING = {
    "domain accuracy",
    "cost per query",
    "latency p95 (ms)",
    "consistency",
    "auditability",
}


def compare(federation: ArmMetrics, baseline: ArmMetrics) -> Comparison:
    """Compare the two arms and state the honest conclusion.

    The rule is the one the architecture commits to: if the federation does not
    win **on the set**, the correct decision is to keep that agent as RAG over
    the shared model.  ALM applies where it wins, not everywhere on principle —
    so this function is willing to return a verdict against the federation.
    """
    fed_row = federation.to_row()
    base_row = baseline.to_row()

    deltas: dict[str, float] = {}
    wins: dict[str, bool] = {}
    for metric, fed_value in fed_row.items():
        base_value = base_row.get(metric, 0.0)
        deltas[metric] = round(fed_value - base_value, 8)
        if metric not in _DECIDING:
            continue
        if metric in _LOWER_IS_BETTER:
            wins[metric] = fed_value <= base_value
        else:
            wins[metric] = fed_value >= base_value

    won = sum(1 for metric, value in wins.items() if value)
    total = len(wins)
    accuracy_held = wins.get("domain accuracy", False)

    if accuracy_held and won >= total - 1:
        verdict = "federation"
        recommendation = (
            "The federation wins on the set. Keep the dedicated experts and "
            "continue promoting domains that meet a business trigger."
        )
    elif not accuracy_held:
        verdict = "baseline"
        recommendation = (
            "The federation does not hold accuracy against the baseline. Keep these "
            "agents as RAG over the shared model — ALM applies where it wins, not "
            "everywhere on principle."
        )
    else:
        verdict = "inconclusive"
        recommendation = (
            "Accuracy holds but the federation does not win on the set. Investigate "
            "the losing metrics before promoting any further domain."
        )

    return Comparison(
        federation=federation,
        baseline=baseline,
        deltas=deltas,
        wins=wins,
        verdict=verdict,
        recommendation=recommendation,
    )


def render_comparison(comparison: Comparison, *, domain: str = "") -> str:
    """Render the comparison as the table `alm eval run --compare` prints."""
    fed = comparison.federation
    base = comparison.baseline
    header = (
        f"DOMAIN {domain or 'all'} · {fed.cases} cases · federation vs baseline\n"
    )
    lines = [
        header,
        f"  {'metric':<24}{'federation':>14}{'baseline':>14}{'delta':>12}",
        "  " + "─" * 62,
    ]

    fed_row = fed.to_row()
    base_row = base.to_row()
    for metric, fed_value in fed_row.items():
        base_value = base_row.get(metric, 0.0)
        won = comparison.wins.get(metric)
        mark = "" if won is None else ("  ✓" if won else "  ✗")

        if metric == "cost per query":
            fed_text, base_text = f"${fed_value:.6f}", f"${base_value:.6f}"
            delta = (
                f"{(fed_value - base_value) / base_value * 100:+.1f}%"
                if base_value
                else "—"
            )
        elif metric == "latency p95 (ms)":
            fed_text, base_text = f"{fed_value:,.0f}", f"{base_value:,.0f}"
            delta = (
                f"{(fed_value - base_value) / base_value * 100:+.1f}%"
                if base_value
                else "—"
            )
        elif metric == "fallback rate":
            fed_text, base_text, delta = f"{fed_value:.3f}", "—", "—"
        else:
            fed_text, base_text = f"{fed_value:.3f}", f"{base_value:.3f}"
            delta = f"{fed_value - base_value:+.3f}"

        lines.append(f"  {metric:<24}{fed_text:>14}{base_text:>14}{delta:>12}{mark}")

    lines.append("")
    lines.append(f"  VERDICT  {comparison.verdict}")
    lines.append(f"  {comparison.recommendation}")
    return "\n".join(lines)
