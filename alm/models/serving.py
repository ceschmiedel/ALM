"""Serving policy — tier routing and the escalation cascade.

Two mechanisms, both cheap and both load-bearing:

**Tier routing.** Every call names the *smallest* tier that should be able to
answer it. Micro-SLMs classify and extract, SLMs carry the domain work, the
frontier model is reserved for planning and hard arbitration. Each request then
uses the minimum necessary rather than the maximum available.

**Escalation.** When the chosen tier cannot answer — unreachable backend, empty
output, or a caller-supplied quality predicate that fails — the request climbs
one tier and retries. This is the cascade pattern that FrugalGPT shows cuts cost
without losing quality, and it is also the safety property of the whole
architecture: at worst a request ends up at the frontier model, which is exactly
the monolithic baseline. The federation cannot be worse than what it replaces.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from alm.core.errors import BackendError, ModelNotConfiguredError
from alm.core.telemetry import UsageMeter
from alm.models.registry import ModelRegistry
from alm.models.spec import GenerationRequest, GenerationResult
from alm.models.tiers import ModelTier, next_tier, parse_tier

logger = logging.getLogger(__name__)

#: Return ``True`` when a result is good enough to stop escalating.
QualityCheck = Callable[[GenerationResult], bool]


def _non_empty(result: GenerationResult) -> bool:
    return bool(result.text and result.text.strip())


class ServingRouter:
    """Resolves models by tier and runs the escalation cascade."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.escalations: list[dict[str, Any]] = []

    # -- resolution --------------------------------------------------------

    def resolve(
        self,
        *,
        model_id: str = "",
        tier: ModelTier | str | None = None,
    ) -> str:
        """Return the model id to call, preferring an explicit binding.

        Falls back from an unknown explicit model to the tier, because an
        expert whose dedicated adapter has been retired should degrade to the
        shared model of its tier rather than fail the whole run.
        """
        if model_id and self.registry.exists(model_id):
            return model_id
        if model_id:
            logger.warning("Model %r not registered; resolving by tier instead", model_id)

        resolved_tier = parse_tier(tier) if tier is not None else ModelTier.SLM
        spec = self.registry.for_tier(resolved_tier)
        if spec is None:
            # Any tier is better than no answer; prefer the closest one up.
            climb = next_tier(resolved_tier)
            while climb is not None:
                spec = self.registry.for_tier(climb)
                if spec is not None:
                    break
                climb = next_tier(climb)
        if spec is None:
            raise ModelNotConfiguredError(
                f"no model registered for tier {resolved_tier!r}. "
                "Register one with `alm model add`, or install a domain pack that declares it.",
                tier=str(resolved_tier),
                requested_model=model_id,
            )
        return spec.model_id

    # -- execution ---------------------------------------------------------

    async def generate(
        self,
        request: GenerationRequest,
        *,
        model_id: str = "",
        tier: ModelTier | str | None = None,
        meter: UsageMeter | None = None,
        component: str = "",
        allow_escalation: bool = True,
        quality_check: QualityCheck | None = None,
    ) -> GenerationResult:
        """Generate, escalating up the tiers when the current one cannot deliver."""
        check = quality_check or _non_empty
        resolved_id = self.resolve(model_id=model_id, tier=tier)
        spec = self.registry.require(resolved_id)
        current_tier = spec.tier
        attempted: set[str] = set()
        last_error: Exception | None = None

        while True:
            attempted.add(resolved_id)
            try:
                result = await self.registry.generate(
                    resolved_id, request, meter=meter, component=component
                )
                if check(result):
                    return result
                reason = "quality check failed"
            except BackendError as exc:
                last_error = exc
                reason = str(exc)
                result = None  # type: ignore[assignment]

            if not allow_escalation:
                if last_error is not None:
                    raise last_error
                return result  # type: ignore[return-value]

            upper = next_tier(current_tier)
            candidate = None
            while upper is not None:
                spec_up = self.registry.for_tier(upper)
                if spec_up is not None and spec_up.model_id not in attempted:
                    candidate = spec_up
                    break
                upper = next_tier(upper)

            if candidate is None:
                if last_error is not None:
                    raise last_error
                return result  # type: ignore[return-value]

            self.escalations.append(
                {
                    "from_model": resolved_id,
                    "from_tier": str(current_tier),
                    "to_model": candidate.model_id,
                    "to_tier": str(candidate.tier),
                    "reason": reason,
                    "component": component,
                }
            )
            logger.info(
                "Escalating %s → %s (%s): %s",
                resolved_id,
                candidate.model_id,
                candidate.tier,
                reason,
            )
            resolved_id = candidate.model_id
            current_tier = candidate.tier

    # -- introspection -----------------------------------------------------

    def escalation_count(self) -> int:
        return len(self.escalations)

    def reset(self) -> None:
        self.escalations.clear()

    def topology(self) -> dict[str, Any]:
        """Serving layout, including how much multi-adapter sharing is in play."""
        summary = self.registry.serving_summary()
        summary["escalations"] = len(self.escalations)
        return summary
