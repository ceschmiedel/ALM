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


class EvalRequest(BaseModel):
    """Run the controlled experiment."""

    dataset: str = Field(..., description="Path to a JSONL evaluation set.")
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
