"""Model tiers.

Not every agent needs the same size of model, and routing by tier is what keeps
each request on the cheapest hardware that can actually answer it: micro-SLMs on
CPU, SLMs on the shared base, the frontier model only on fallback.

The tier is a *routing* concept, not a claim about a specific checkpoint's
parameter count — it says what class of work the model is expected to carry.
"""

from __future__ import annotations

from enum import StrEnum


class ModelTier(StrEnum):
    """The four tiers the federation routes across."""

    MICRO_SLM = "micro_slm"
    SLM = "slm"
    SMALL = "small"
    ORCHESTRATOR = "orchestrator"


TIER_GUIDANCE: dict[ModelTier, dict[str, str]] = {
    ModelTier.MICRO_SLM: {
        "size": "< 1B",
        "use_for": (
            "Narrow, deterministic tasks: classification, field extraction, routing, "
            "normalisation."
        ),
        "runs_on": "CPU or a modest GPU, at very low latency.",
    },
    ModelTier.SLM: {
        "size": "1B – 4B",
        "use_for": (
            "The standard domain expert: in-domain reasoning, structured generation, "
            "answering queries within scope."
        ),
        "runs_on": "A single GPU.",
    },
    ModelTier.SMALL: {
        "size": "4B – 8B",
        "use_for": "Domains needing longer reasoning or richer generation, still in scope.",
        "runs_on": "Still viable on-premise.",
    },
    ModelTier.ORCHESTRATOR: {
        "size": "frontier",
        "use_for": (
            "Complex planning, ambiguous cases, hard arbitration, and generating "
            "training data. Used sparingly — not on every request."
        ),
        "runs_on": "A hosted API, or a large self-hosted model where data must not leave.",
    },
}

# Ordered cheapest-first; the cascade escalates along this order.
TIER_ORDER: tuple[ModelTier, ...] = (
    ModelTier.MICRO_SLM,
    ModelTier.SLM,
    ModelTier.SMALL,
    ModelTier.ORCHESTRATOR,
)


def tier_rank(tier: ModelTier | str) -> int:
    """Position in the escalation order (0 = cheapest)."""
    try:
        return TIER_ORDER.index(ModelTier(str(tier)))
    except ValueError:
        return len(TIER_ORDER)


def next_tier(tier: ModelTier | str) -> ModelTier | None:
    """The next tier up, or ``None`` when already at the top."""
    rank = tier_rank(tier)
    if rank + 1 >= len(TIER_ORDER):
        return None
    return TIER_ORDER[rank + 1]


def parse_tier(value: str | ModelTier | None, default: ModelTier = ModelTier.SLM) -> ModelTier:
    """Parse a tier name leniently, accepting hyphens and mixed case."""
    if value is None or value == "":
        return default
    if isinstance(value, ModelTier):
        return value
    normalised = str(value).strip().lower().replace("-", "_")
    aliases = {
        "micro": ModelTier.MICRO_SLM,
        "microslm": ModelTier.MICRO_SLM,
        "micro_slm": ModelTier.MICRO_SLM,
        "slm": ModelTier.SLM,
        "small": ModelTier.SMALL,
        "orchestrator": ModelTier.ORCHESTRATOR,
        "llm": ModelTier.ORCHESTRATOR,
        "frontier": ModelTier.ORCHESTRATOR,
        "teacher": ModelTier.ORCHESTRATOR,
    }
    if normalised in aliases:
        return aliases[normalised]
    raise ValueError(
        f"unknown model tier {value!r}; expected one of {', '.join(t.value for t in ModelTier)}"
    )
