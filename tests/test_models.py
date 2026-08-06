"""L3 · model registry, backends, tiers and the escalation cascade."""

from __future__ import annotations

import pytest

from alm.core.errors import (
    BackendError,
    BackendUnavailableError,
    ConfigurationError,
    ModelNotConfiguredError,
)
from alm.core.telemetry import UsageMeter
from alm.models.backends import available_backends, create_backend, resolve_backend_class
from alm.models.backends.base import ChatBackend, CircuitBreaker
from alm.models.registry import ModelRegistry, selfhosted_cost_per_1k
from alm.models.serving import ServingRouter
from alm.models.spec import GenerationRequest, GenerationResult, Message, ModelSpec, estimate_tokens
from alm.models.tiers import ModelTier, next_tier, parse_tier, tier_rank


def _request(text: str = "What is the liability cap?") -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content=text)],
        metadata={"question": text},
    )


class _AlwaysFails(ChatBackend):
    name = "always-fails"

    async def _generate(self, request):  # noqa: ANN001
        raise BackendError("simulated outage", backend=self.name)


class _Empty(ChatBackend):
    name = "empty"

    async def _generate(self, request):  # noqa: ANN001
        return GenerationResult(text="")


# -- tiers -------------------------------------------------------------------


def test_parse_tier_accepts_aliases():
    assert parse_tier("micro-slm") is ModelTier.MICRO_SLM
    assert parse_tier("LLM") is ModelTier.ORCHESTRATOR
    assert parse_tier(None) is ModelTier.SLM
    with pytest.raises(ValueError):
        parse_tier("enormous")


def test_tier_order_escalates_upward():
    assert tier_rank(ModelTier.MICRO_SLM) < tier_rank(ModelTier.ORCHESTRATOR)
    assert next_tier(ModelTier.MICRO_SLM) is ModelTier.SLM
    assert next_tier(ModelTier.ORCHESTRATOR) is None


# -- registry ----------------------------------------------------------------


def test_register_and_retrieve(registry: ModelRegistry):
    registry.register(ModelSpec(model_id="m1", tier=ModelTier.SLM, backend="heuristic"))
    assert registry.exists("m1")
    assert registry.require("m1").backend == "heuristic"
    with pytest.raises(ModelNotConfiguredError):
        registry.require("ghost")


def test_adapter_grouping_reports_saved_loads(registry: ModelRegistry):
    for name in ("legal", "finance", "risk"):
        registry.register(
            ModelSpec(
                model_id=f"{name}-lora",
                tier=ModelTier.SLM,
                backend="vllm",
                base_model="llama-3.2-3b",
                adapter=f"{name}-v1",
            )
        )
    registry.register(ModelSpec(model_id="standalone", backend="heuristic"))

    summary = registry.serving_summary()
    assert summary["shared_bases"] == 1
    assert summary["adapters"] == 3
    # Three adapters over one base means two full model loads avoided.
    assert summary["loads_saved"] == 2
    assert summary["standalone_models"] == ["standalone"]


def test_adapter_is_addressed_by_its_own_name_on_the_wire():
    spec = ModelSpec(
        model_id="legal", backend="vllm", base_model="llama-3.2-3b", adapter="legal-v3"
    )
    assert spec.is_adapter
    assert spec.served_name == "legal-v3"


def test_served_name_falls_back_through_the_chain():
    assert ModelSpec(model_id="x", model_name="gpt-4o").served_name == "gpt-4o"
    assert ModelSpec(model_id="x", base_model="b").served_name == "b"
    assert ModelSpec(model_id="only-id").served_name == "only-id"


def test_cost_is_computed_per_thousand_tokens():
    spec = ModelSpec(model_id="m", cost_per_1k_input=0.5, cost_per_1k_output=1.5)
    assert spec.cost_for(1000, 1000) == pytest.approx(2.0)
    assert spec.cost_for(0, 0) == 0.0


def test_selfhosted_cost_conversion():
    cost = selfhosted_cost_per_1k(2.0, 1000.0, utilisation=0.5)
    # $2/h over 500 effective tokens/s → $0.000001111.. per 1k
    assert cost == pytest.approx(2.0 / 3600.0 / 500.0 * 1000.0)
    assert selfhosted_cost_per_1k(2.0, 0.0) == 0.0


def test_api_key_is_not_leaked_into_params_listing(registry: ModelRegistry):
    registry.register(ModelSpec(model_id="secret", backend="heuristic", api_key="sk-123"))
    spec = registry.require("secret")
    assert spec.api_key == "sk-123"
    assert "api_key" not in spec.params


# -- backends ----------------------------------------------------------------


def test_backend_registry_resolves_known_names():
    assert "heuristic" in available_backends()
    assert resolve_backend_class("ollama").name == "ollama"
    with pytest.raises(ConfigurationError, match="unknown backend"):
        resolve_backend_class("nonexistent")


async def test_heuristic_backend_extracts_from_supplied_context():
    backend = create_backend(ModelSpec(model_id="h", backend="heuristic"))
    request = GenerationRequest(
        messages=[Message(role="user", content="What is the liability cap?")],
        metadata={
            "question": "What is the liability cap?",
            "context_chunks": [
                {
                    "text": (
                        "Clause 7.2 states the aggregate liability cap is twelve months "
                        "of fees. Payment terms are net 30 days from invoice."
                    )
                }
            ],
        },
    )
    result = await backend.generate(request)
    assert "liability" in result.text.lower()
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


async def test_heuristic_backend_honours_enum_in_schema():
    backend = create_backend(ModelSpec(model_id="h", backend="heuristic"))
    request = GenerationRequest(
        messages=[Message(role="user", content="How severe is the risk?")],
        json_mode=True,
        response_schema={
            "type": "object",
            "properties": {"severity": {"type": "string", "enum": ["low", "high"]}},
        },
        metadata={
            "question": "How severe is the risk?",
            "context_chunks": [{"text": "The residual risk of this engagement is high."}],
        },
    )
    result = await backend.generate(request)
    import json

    assert json.loads(result.text)["severity"] in {"low", "high"}


async def test_heuristic_backend_admits_when_it_has_nothing():
    backend = create_backend(ModelSpec(model_id="h", backend="heuristic"))
    result = await backend.generate(
        GenerationRequest(
            messages=[Message(role="user", content="zzzz")],
            metadata={"question": "zzzz", "context_chunks": [{"text": "unrelated"}]},
        )
    )
    assert "no supporting passage" in result.text.lower()


async def test_scripted_backend_matches_rules_in_order():
    spec = ModelSpec(
        model_id="s",
        backend="scripted",
        params={
            "responses": [
                {"match": "liability", "response": "capped at twelve months"},
                {"match": ".*", "response": "fallback"},
            ]
        },
    )
    backend = create_backend(spec)
    hit = await backend.generate(_request("what about liability?"))
    assert hit.text == "capped at twelve months"
    miss = await backend.generate(_request("something else entirely"))
    assert miss.text == "fallback"


# -- circuit breaker ---------------------------------------------------------

def test_circuit_breaker_opens_then_half_opens():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=0.0)
    breaker.record_failure()
    assert not breaker.is_open
    breaker.record_failure()
    # Cooldown is zero, so the very next check half-opens for a probe.
    assert breaker.is_open is False
    breaker.record_success()
    assert not breaker.is_open


async def test_backend_trips_the_breaker_after_repeated_failures():
    backend = _AlwaysFails(ModelSpec(model_id="bad", backend="always-fails"))
    for _ in range(3):
        with pytest.raises(BackendError):
            await backend.generate(_request())
    with pytest.raises(BackendUnavailableError):
        await backend.generate(_request())


# -- serving cascade ---------------------------------------------------------


async def test_serving_escalates_to_the_next_tier_on_failure(registry: ModelRegistry):
    from alm.models.backends import register_backend

    register_backend("always-fails", _AlwaysFails)
    registry.register(ModelSpec(model_id="small", tier=ModelTier.SLM, backend="always-fails"))
    registry.register(
        ModelSpec(model_id="big", tier=ModelTier.ORCHESTRATOR, backend="heuristic")
    )

    router = ServingRouter(registry)
    result = await router.generate(
        _request(),
        tier=ModelTier.SLM,
        component="test",
    )
    assert result.model_id == "big"
    assert router.escalation_count() == 1
    assert router.escalations[0]["from_tier"] == "slm"


async def test_serving_escalates_on_empty_output(registry: ModelRegistry):
    from alm.models.backends import register_backend

    register_backend("empty", _Empty)
    registry.register(ModelSpec(model_id="quiet", tier=ModelTier.SLM, backend="empty"))
    registry.register(
        ModelSpec(model_id="loud", tier=ModelTier.ORCHESTRATOR, backend="heuristic")
    )

    router = ServingRouter(registry)
    result = await router.generate(
        _request(),
        tier=ModelTier.SLM,
        component="test",
    )
    assert result.model_id == "loud"


async def test_serving_raises_when_no_model_exists(registry: ModelRegistry):
    router = ServingRouter(registry)
    with pytest.raises(ModelNotConfiguredError):
        await router.generate(_request(), tier=ModelTier.SLM)


def test_serving_falls_back_from_unknown_model_to_tier(registry: ModelRegistry):
    registry.register(ModelSpec(model_id="real", tier=ModelTier.SLM, backend="heuristic"))
    router = ServingRouter(registry)
    assert router.resolve(model_id="retired-adapter", tier=ModelTier.SLM) == "real"


async def test_usage_meter_accumulates_per_tier(registry: ModelRegistry):
    registry.register(
        ModelSpec(
            model_id="m",
            tier=ModelTier.SLM,
            backend="heuristic",
            cost_per_1k_input=1.0,
            cost_per_1k_output=1.0,
        )
    )
    meter = UsageMeter()
    router = ServingRouter(registry)
    await router.generate(_request(), model_id="m", meter=meter, component="test")

    summary = meter.summary()
    assert summary["calls"] == 1
    assert summary["total_cost_usd"] > 0
    assert "slm" in summary["by_tier"]


def test_estimate_tokens_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3
