"""L3 · Model federation — tiers, registry, backends and serving policy."""

from alm.models.backends import (
    ChatBackend,
    available_backends,
    create_backend,
    register_backend,
)
from alm.models.registry import ModelRegistry, selfhosted_cost_per_1k, spec_from_mapping
from alm.models.serving import ServingRouter
from alm.models.spec import (
    GenerationRequest,
    GenerationResult,
    Message,
    ModelSpec,
    estimate_tokens,
)
from alm.models.tiers import TIER_GUIDANCE, ModelTier, next_tier, parse_tier, tier_rank

__all__ = [
    "TIER_GUIDANCE",
    "ChatBackend",
    "GenerationRequest",
    "GenerationResult",
    "Message",
    "ModelRegistry",
    "ModelSpec",
    "ModelTier",
    "ServingRouter",
    "available_backends",
    "create_backend",
    "estimate_tokens",
    "next_tier",
    "parse_tier",
    "register_backend",
    "selfhosted_cost_per_1k",
    "spec_from_mapping",
    "tier_rank",
]
