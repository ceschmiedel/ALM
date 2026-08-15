"""Request and response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Run one intent through the federation."""

    intent: str = Field(..., description="The request, in natural language.")
    context: dict[str, Any] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    principal: str = Field(default="", description="Identity IBAC evaluates against.")
    claims: list[str] = Field(default_factory=list)
    session_id: str = ""
    include_trace: bool = True


class AskResponse(BaseModel):
    """The answer, with everything needed to audit it."""

    session_id: str
    status: str
    answer: str
    confidence: float
    experts: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    citations: list[dict[str, Any]] = Field(default_factory=list)
    audit_trail: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    error: str = ""


class ExplainRequest(BaseModel):
    """Dry-run the router without executing anything."""

    intent: str
    top_k: int = 5


class SearchRequest(BaseModel):
    """Query CMRAG directly."""

    query: str
    domains: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    top_k: int = 6
    expert_id: str = Field(
        default="",
        description="Apply this expert's IBAC filter, as a real retrieval would.",
    )


class GraphSearchRequest(BaseModel):
    query: str
    kinds: list[str] = Field(default_factory=list)
    domain: str = ""
    top_k: int = 10


class ModelCreateRequest(BaseModel):
    """Register a model in the federation registry."""

    model_id: str
    tier: str = "slm"
    backend: str = "heuristic"
    model_name: str = ""
    base_model: str = ""
    adapter: str = ""
    adapter_uri: str = ""
    endpoint: str = ""
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    description: str = ""
    tier_default: bool = Field(
        default=False,
        description="Make this the model its tier resolves to, displacing any other.",
    )


class ModelProbeRequest(BaseModel):
    """Check a model actually answers before committing it to a tier.

    Registering a hosted fallback with a wrong key otherwise fails silently
    until the first escalation — the worst possible moment to discover it.
    """

    model_id: str = Field(default="", description="Probe an already-registered model.")
    # Or probe an unregistered candidate before saving it:
    backend: str = ""
    model_name: str = ""
    endpoint: str = ""
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class CapabilityInput(BaseModel):
    """One capability of an agent being created through the console."""

    id: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    operates_on: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)


class AgentCreateRequest(BaseModel):
    """Create an Expert Agent without authoring a pack on disk.

    A pack remains the reproducible way to ship a domain; this is the
    interactive way to build one up. Both land in the same place — an EXPERT
    node on the Context Graph carrying the full specification.
    """

    id: str
    domain: str
    label: str = ""
    description: str = ""
    model: str = Field(default="", description="Model id from the registry.")
    tier: str = "slm"

    capabilities: list[CapabilityInput] = Field(default_factory=list)
    authority: dict[str, float] = Field(default_factory=dict)
    retrieval_domains: list[str] = Field(default_factory=list)
    retrieval_top_k: int = 6

    system_prompt: str = ""
    answer_language: str = ""
    temperature: float = 0.0
    max_tokens: int = 900
    enabled: bool = True

    create_domain: bool = Field(
        default=True,
        description=(
            "Register the domain (and any entity types the capabilities name) "
            "if it does not exist yet, so an agent can be created on a fresh "
            "install with no pack."
        ),
    )


class AgentUpdateRequest(BaseModel):
    """Patch an existing agent. Only the fields provided are changed."""

    label: str | None = None
    description: str | None = None
    model: str | None = None
    tier: str | None = None
    enabled: bool | None = None
    authority: dict[str, float] | None = None
    retrieval_domains: list[str] | None = None
    retrieval_top_k: int | None = None
    system_prompt: str | None = None
    answer_language: str | None = None
    capabilities: list[CapabilityInput] | None = None


class EvalRequest(BaseModel):
    """Run the controlled experiment.

    Either ``dataset`` (a specific JSONL file) or ``pack`` (every evaluation
    file the pack declares) must be given — the same shortcut the CLI's
    ``alm eval run --pack`` offers.
    """

    dataset: str = Field(default="", description="Path to a JSONL evaluation set.")
    pack: str = Field(default="", description="Path to a pack; uses all its eval files.")
    domain: str = ""
    compare: bool = Field(
        default=True, description="Also run the monolithic baseline arm."
    )
    repeats: int = 1


class PromotionCheckRequest(BaseModel):
    """Evaluate the business triggers for a dedicated model."""

    expert_id: str
    sovereignty_required: bool = False
    monthly_calls: int | None = None
    volume_threshold: int = 50_000
    latency_requirement_ms: float | None = None
    observed_p95_ms: float | None = None
    rag_accuracy: float | None = None
    accuracy_target: float | None = None


class PackInstallRequest(BaseModel):
    path: str
    index_corpus: bool = True


class HealthResponse(BaseModel):
    """Inventory *and* degradation — what is configured and what is missing."""

    status: str = "ok"
    version: str = ""
    tenant: str = "default"
    experts: int = 0
    domains: list[str] = Field(default_factory=list)
    models: dict[str, Any] = Field(default_factory=dict)
    corpus: dict[str, Any] = Field(default_factory=dict)
    semantic_routing: bool = False
    orchestrator_configured: bool = False
    fallback_available: bool = False
    governance_enabled: bool = False
    warnings: list[str] = Field(default_factory=list)
