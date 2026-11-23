"""The decision engine — one entry point for every typed judgment.

Callers build a :class:`~alm.decision.spec.DecisionRequest` and name the tier
that should be able to answer it; the engine hands it to a backend:

* ``generative`` — a chat model asked for JSON.  Works everywhere.
* ``readout`` — next-token probabilities read in one forward pass.  Needs a
  model whose serving backend exposes them.

``ALM_DECISION_BACKEND`` picks the default.  In readout mode the engine still
falls back to the generative backend for a request readout cannot serve (a
hosted model with no logprobs, more options than there are label tokens), so
switching it on never removes a judgment the federation could make before.
"""

from __future__ import annotations

import logging
from typing import Protocol

from alm.config import Settings, get_settings
from alm.core.telemetry import UsageMeter
from alm.decision.generative import GenerativeDecisionBackend
from alm.decision.readout import ReadoutConfig, ReadoutDecisionBackend, ReadoutUnsupportedError
from alm.decision.spec import DecisionRequest, DecisionResult
from alm.models.serving import ServingRouter
from alm.models.tiers import ModelTier

logger = logging.getLogger(__name__)


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


def readout_config(settings: Settings) -> ReadoutConfig:
    return ReadoutConfig(
        max_permutations=settings.decision_permutations,
        temperature={
            "choice": settings.decision_temperature_choice,
            "score": settings.decision_temperature_score,
            "noul": settings.decision_temperature_noul,
        },
    )


#: Readout backends hold loaded providers, so one is kept per serving router.
_READOUT: dict[int, ReadoutDecisionBackend] = {}


def _readout_backend(serving: ServingRouter, settings: Settings) -> ReadoutDecisionBackend:
    backend = _READOUT.get(id(serving))
    if backend is None or backend.serving is not serving:
        backend = ReadoutDecisionBackend(serving, config=readout_config(settings))
        _READOUT[id(serving)] = backend
    return backend


class DecisionEngine:
    """Routes decision requests to the configured backend."""

    def __init__(
        self,
        serving: ServingRouter,
        *,
        backend: DecisionBackend | None = None,
        mode: str | None = None,
    ) -> None:
        self.serving = serving
        self.settings = get_settings()
        self.mode = (mode or self.settings.decision_backend or "generative").strip().lower()
        self.generative = GenerativeDecisionBackend(serving)
        self.backend = backend

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
        if self.backend is not None:
            return await self.backend.decide(
                request, tier=tier, model_id=model_id, meter=meter, component=component
            )
        if self.mode == "readout":
            try:
                return await _readout_backend(self.serving, self.settings).decide(
                    request,
                    tier=tier,
                    model_id=model_id or self.settings.decision_model,
                    meter=meter,
                    component=component,
                )
            except ReadoutUnsupportedError as exc:
                logger.info("Readout cannot serve %s (%s); using the generative backend", component, exc)
        return await self.generative.decide(
            request, tier=tier, model_id=model_id, meter=meter, component=component
        )
