"""Model registry — the inventory the federation routes across.

Registering a model is a data operation, like registering a capability: the
router asks the registry for "the model for this tier" or "the model this
expert is bound to", and the registry answers from the database.  Swapping an
expert from a shared SLM to its own LoRA adapter is therefore a registry
update, not a deployment.

Cost accounting deserves a note.  Prices are **not** baked in, because a table
of vendor prices in source code is wrong the week after it is written and,
worse, says nothing about self-hosted serving — which is where most of a
federation's cost actually sits.  Set prices explicitly per model, and use
:func:`selfhosted_cost_per_1k` to convert a GPU-hour rate into the same unit so
that on-premise and hosted models are comparable in one number.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from alm.core.errors import ConfigurationError, ModelNotConfiguredError
from alm.core.telemetry import CallRecord, UsageMeter
from alm.models.backends import ChatBackend, create_backend
from alm.models.spec import GenerationRequest, GenerationResult, ModelSpec
from alm.models.tiers import ModelTier, parse_tier
from alm.persistence.database import session_scope
from alm.persistence.models import ModelRow

logger = logging.getLogger(__name__)


def selfhosted_cost_per_1k(
    gpu_hourly_usd: float,
    tokens_per_second: float,
    *,
    utilisation: float = 0.7,
) -> float:
    """Convert a GPU-hour rate into cost per 1 000 tokens.

    ``utilisation`` is the fraction of wall-clock the GPU actually spends
    generating for this model; continuous batching pushes it up, an idle
    dedicated GPU pushes it down.  Comparing a federation against a hosted LLM
    on token price alone is the mistake this function exists to prevent.
    """
    if tokens_per_second <= 0 or utilisation <= 0:
        return 0.0
    effective_tps = tokens_per_second * utilisation
    cost_per_second = gpu_hourly_usd / 3600.0
    return round((cost_per_second / effective_tps) * 1000.0, 10)


def _row_to_spec(row: ModelRow) -> ModelSpec:
    params = dict(row.params or {})
    return ModelSpec(
        model_id=row.model_id,
        tier=parse_tier(row.tier),
        backend=row.backend,
        model_name=row.model_name,
        base_model=row.base_model,
        adapter=row.adapter,
        adapter_uri=row.adapter_uri,
        endpoint=row.endpoint,
        api_key=str(params.pop("api_key", "")),
        params=params,
        context_window=row.context_window,
        max_output_tokens=int(params.get("max_output_tokens", 1024) or 1024),
        cost_per_1k_input=row.cost_per_1k_input,
        cost_per_1k_output=row.cost_per_1k_output,
        description=row.description,
        status=row.status,
        version=row.active_version,
        tenant_id=row.tenant_id,
        tier_default=bool(row.tier_default),
    )


class ModelRegistry:
    """Tenant-scoped registry of models, with cached backend instances."""

    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id
        self._backends: dict[str, ChatBackend] = {}

    # -- registration ------------------------------------------------------

    def register(self, spec: ModelSpec) -> ModelSpec:
        """Insert or update a model."""
        spec.tenant_id = self.tenant_id
        params = dict(spec.params)
        if spec.api_key:
            params["api_key"] = spec.api_key
        params.setdefault("max_output_tokens", spec.max_output_tokens)

        with session_scope() as session:
            row = session.get(ModelRow, spec.model_id)
            if row is None:
                row = ModelRow(model_id=spec.model_id, tenant_id=self.tenant_id)
                session.add(row)
            row.tier = str(spec.tier)
            row.backend = spec.backend
            row.model_name = spec.model_name
            row.base_model = spec.base_model
            row.adapter = spec.adapter
            row.adapter_uri = spec.adapter_uri
            row.endpoint = spec.endpoint
            row.params = params
            row.context_window = spec.context_window
            row.cost_per_1k_input = spec.cost_per_1k_input
            row.cost_per_1k_output = spec.cost_per_1k_output
            row.status = spec.status
            row.active_version = spec.version
            row.description = spec.description
            row.tier_default = spec.tier_default

        if spec.tier_default:
            self.set_tier_default(spec.model_id)

        self._backends.pop(spec.model_id, None)  # force re-instantiation
        logger.info(
            "Model registered: %s (tier=%s backend=%s%s)",
            spec.model_id,
            spec.tier,
            spec.backend,
            f" adapter={spec.adapter} on {spec.base_model}" if spec.is_adapter else "",
        )
        return spec

    def register_many(self, specs: Iterable[ModelSpec]) -> list[ModelSpec]:
        return [self.register(spec) for spec in specs]

    def set_tier_default(self, model_id: str) -> ModelSpec:
        """Make ``model_id`` the model its tier resolves to.

        Exactly one model per tier carries the flag, so "assign this model to
        the SLM tier" means what it says instead of competing with whatever
        else happens to be registered there.
        """
        with session_scope() as session:
            row = session.get(ModelRow, model_id)
            if row is None or row.tenant_id != self.tenant_id:
                raise ModelNotConfiguredError(
                    f"model {model_id!r} is not registered", model_id=model_id
                )
            siblings = session.execute(
                select(ModelRow).where(
                    ModelRow.tenant_id == self.tenant_id,
                    ModelRow.tier == row.tier,
                )
            ).scalars().all()
            for sibling in siblings:
                sibling.tier_default = sibling.model_id == model_id
            session.flush()
            logger.info("Model %s is now the default for tier %s", model_id, row.tier)
            return _row_to_spec(row)

    def remove(self, model_id: str) -> bool:
        with session_scope() as session:
            row = session.get(ModelRow, model_id)
            if row is None or row.tenant_id != self.tenant_id:
                return False
            session.delete(row)
        self._backends.pop(model_id, None)
        return True

    # -- lookup ------------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec | None:
        with session_scope() as session:
            row = session.get(ModelRow, model_id)
            if row is None or row.tenant_id != self.tenant_id:
                return None
            return _row_to_spec(row)

    def require(self, model_id: str) -> ModelSpec:
        spec = self.get(model_id)
        if spec is None:
            raise ModelNotConfiguredError(
                f"model {model_id!r} is not registered; run `alm model list` to see what is",
                model_id=model_id,
            )
        return spec

    def list(
        self,
        *,
        tier: ModelTier | str | None = None,
        status: str | None = "active",
    ) -> list[ModelSpec]:
        with session_scope() as session:
            stmt = select(ModelRow).where(ModelRow.tenant_id == self.tenant_id)
            if tier is not None:
                stmt = stmt.where(ModelRow.tier == str(parse_tier(tier)))
            if status is not None:
                stmt = stmt.where(ModelRow.status == status)
            rows = session.execute(stmt.order_by(ModelRow.model_id)).scalars().all()
            return [_row_to_spec(r) for r in rows]

    def exists(self, model_id: str) -> bool:
        return self.get(model_id) is not None

    # -- tier resolution ---------------------------------------------------

    def for_tier(self, tier: ModelTier | str) -> ModelSpec | None:
        """The model to use for a tier.

        An explicit ``ALM_*_MODEL`` setting wins; otherwise the single
        registered model of that tier, or the first by id when several exist.
        """
        from alm.config import get_settings

        resolved = parse_tier(tier)
        settings = get_settings()
        preferred = {
            ModelTier.ORCHESTRATOR: settings.orchestrator_model,
            ModelTier.MICRO_SLM: settings.router_model,
            ModelTier.SLM: settings.default_expert_model,
        }.get(resolved, "")
        if preferred:
            spec = self.get(preferred)
            if spec is not None:
                return spec
            logger.warning(
                "Configured model %r for tier %s is not registered; falling back",
                preferred,
                resolved,
            )

        candidates = self.list(tier=resolved)
        if not candidates:
            return None
        # An explicit assignment wins over registration order. Without this the
        # tier resolves alphabetically, so a pack's placeholder would keep
        # serving after the operator assigned a real model to the tier.
        for candidate in candidates:
            if candidate.tier_default:
                return candidate
        return candidates[0]

    def orchestrator(self) -> ModelSpec | None:
        """The frontier model used for planning, hard arbitration and teaching."""
        return self.for_tier(ModelTier.ORCHESTRATOR)

    def router_model(self) -> ModelSpec | None:
        """The cheap classifier the router uses before anything expensive runs."""
        return self.for_tier(ModelTier.MICRO_SLM) or self.for_tier(ModelTier.SLM)

    def default_expert_model(self) -> ModelSpec | None:
        return self.for_tier(ModelTier.SLM) or self.for_tier(ModelTier.SMALL)

    # -- backends ----------------------------------------------------------

    def backend(self, model_id: str) -> ChatBackend:
        """Return the cached backend instance for a model."""
        cached = self._backends.get(model_id)
        if cached is not None:
            return cached
        backend = create_backend(self.require(model_id))
        self._backends[model_id] = backend
        return backend

    async def generate(
        self,
        model_id: str,
        request: GenerationRequest,
        *,
        meter: UsageMeter | None = None,
        component: str = "",
    ) -> GenerationResult:
        """Run a request against a model and account for it."""
        backend = self.backend(model_id)
        result = await backend.generate(request)
        if meter is not None:
            meter.record(
                CallRecord(
                    model_id=result.model_id or model_id,
                    tier=result.tier,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency_ms=result.latency_ms,
                    cost_usd=result.cost_usd,
                    component=component,
                )
            )
        return result

    async def health(self) -> dict[str, bool]:
        """Reachability of every registered model, for `alm doctor`."""
        out: dict[str, bool] = {}
        for spec in self.list():
            try:
                out[spec.model_id] = await self.backend(spec.model_id).health()
            except Exception:
                out[spec.model_id] = False
        return out

    async def close(self) -> None:
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Backend close failed", exc_info=True)
        self._backends.clear()

    # -- serving topology --------------------------------------------------

    def adapter_groups(self) -> dict[str, list[str]]:
        """Map each shared base model to the adapters served on top of it.

        This is the number that decides whether the federation is affordable:
        ``N`` experts over one base cost *base + N adapters*, not ``N`` models.
        """
        groups: dict[str, list[str]] = {}
        for spec in self.list():
            if spec.is_adapter:
                groups.setdefault(spec.base_model, []).append(spec.model_id)
        return {base: sorted(ids) for base, ids in groups.items()}

    def serving_summary(self) -> dict[str, Any]:
        specs = self.list()
        groups = self.adapter_groups()
        adapters = sum(len(v) for v in groups.values())
        standalone = [s.model_id for s in specs if not s.is_adapter]
        return {
            "models": len(specs),
            "by_tier": {
                tier.value: len([s for s in specs if s.tier == tier]) for tier in ModelTier
            },
            "shared_bases": len(groups),
            "adapters": adapters,
            "adapter_groups": groups,
            "standalone_models": standalone,
            # How many full model loads multi-adapter serving avoids.
            "loads_saved": max(0, adapters - len(groups)),
        }


def spec_from_mapping(model_id: str, data: dict[str, Any]) -> ModelSpec:
    """Build a :class:`ModelSpec` from a pack's YAML mapping."""
    if not model_id:
        raise ConfigurationError("model entries need an id")
    return ModelSpec(
        model_id=model_id,
        tier=parse_tier(data.get("tier")),
        backend=str(data.get("backend", "heuristic")),
        model_name=str(data.get("model") or data.get("model_name") or ""),
        base_model=str(data.get("base_model", "")),
        adapter=str(data.get("adapter", "")),
        adapter_uri=str(data.get("adapter_uri", "")),
        endpoint=str(data.get("endpoint", "")),
        api_key=str(data.get("api_key", "")),
        params=dict(data.get("params", {}) or {}),
        context_window=int(data.get("context_window", 8192)),
        max_output_tokens=int(data.get("max_output_tokens", 1024)),
        cost_per_1k_input=float(data.get("cost_per_1k_input", 0.0)),
        cost_per_1k_output=float(data.get("cost_per_1k_output", 0.0)),
        description=str(data.get("description", "")),
    )
