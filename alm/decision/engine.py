"""The decision engine — one entry point for every typed judgment.

Callers build a :class:`~alm.decision.spec.DecisionRequest` and name the tier
that should be able to answer it; the engine hands it to a backend.  The only
backend today is the generative one, so introducing the engine changes how the
question is phrased to the model and nothing about who answers it.

The point of the indirection is what it makes possible afterwards: a backend
that reads a real distribution from a local model, or a hosted decision model,
plugs in here and every judgment in the federation — routing, arbitration,
grading, governance — moves to it at once, measurable against the fallback.
"""

from __future__ import annotations

from typing import Protocol

from alm.core.telemetry import UsageMeter
from alm.decision.generative import GenerativeDecisionBackend
from alm.decision.spec import DecisionRequest, DecisionResult
from alm.models.serving import ServingRouter
from alm.models.tiers import ModelTier


class DecisionBackend(Protocol):
    """Anything that can turn typed questions into typed answers."""

    method: str

    async def decide(
        self,
        request: DecisionRequest,
        *,
        tier: ModelTier | str | None = None,
        model_id: str = "",
        meter: UsageMeter | None = None,
        component: str = "",
    ) -> DecisionResult: ...


class DecisionEngine:
    """Routes decision requests to the configured backend."""

    def __init__(
        self,
        serving: ServingRouter,
        *,
        backend: DecisionBackend | None = None,
    ) -> None:
        self.serving = serving
        self.backend: DecisionBackend = backend or GenerativeDecisionBackend(serving)

    async def decide(
        self,
        request: DecisionRequest,
        *,
        tier: ModelTier | str | None = None,
        model_id: str = "",
        meter: UsageMeter | None = None,
        component: str = "",
    ) -> DecisionResult:
        """Answer every question in the request.

        Backend failures propagate: each caller already knows what its own
        degraded path is (fall back to the graph, leave a conflict unresolved,
        keep the deterministic decision), and that is not the engine's call.
        """
        return await self.backend.decide(
            request, tier=tier, model_id=model_id, meter=meter, component=component
        )
