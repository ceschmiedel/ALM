"""Model specifications and the generation request/response contract.

A :class:`ModelSpec` is what the registry stores and what a backend is
constructed from.  The field that carries the architecture's economics is
``adapter``: when several specs share a ``base_model`` and differ only by
adapter, they are served from one loaded base, and GPU cost stops scaling with
the number of experts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from alm.models.tiers import ModelTier

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """One chat message."""

    role: Role
    content: str


class ModelSpec(BaseModel):
    """Everything needed to reach and account for one model."""

    model_id: str
    tier: ModelTier = ModelTier.SLM
    backend: str = "heuristic"
    model_name: str = Field(default="", description="Name the backend expects.")

    # Multi-adapter serving: many experts, one base, one GPU.
    base_model: str = ""
    adapter: str = ""
    adapter_uri: str = ""

    endpoint: str = ""
    api_key: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 8192
    max_output_tokens: int = 1024

    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    description: str = ""
    status: str = "active"
    version: str = "1"
    tenant_id: str = "default"

    @property
    def served_name(self) -> str:
        """The name to send to the backend.

        On an OpenAI-compatible server that hosts LoRA adapters (vLLM), each
        adapter is addressed by its own name while sharing the base weights —
        so the adapter name, when present, *is* the model name on the wire.
        """
        return self.adapter or self.model_name or self.base_model or self.model_id

    @property
    def is_adapter(self) -> bool:
        return bool(self.adapter and self.base_model)

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            (prompt_tokens / 1000.0) * self.cost_per_1k_input
            + (completion_tokens / 1000.0) * self.cost_per_1k_output,
            10,
        )


class GenerationRequest(BaseModel):
    """A single inference call."""

    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    stop: list[str] = Field(default_factory=list)

    # When set, the backend is asked for JSON and the caller parses it.  Small
    # models comply unevenly, which is why parsing is tolerant downstream.
    json_mode: bool = False
    response_schema: dict[str, Any] | None = None

    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_text(self) -> str:
        """Flatten to text — used for token estimation and by text-only backends."""
        return "\n\n".join(f"{m.role}: {m.content}" for m in self.messages)


class GenerationResult(BaseModel):
    """The outcome of one inference call, with accounting attached."""

    text: str = ""
    model_id: str = ""
    served_name: str = ""
    tier: str = ""
    backend: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    finish_reason: str = "stop"
    truncated: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer dependency.

    Backends that report real usage always win over this; the estimate exists so
    that cost and context accounting still work with backends that report
    nothing.  It is intentionally slightly conservative (over-counts a little)
    so budgets are not silently exceeded.
    """
    if not text:
        return 0
    words = text.split()
    # ~1.3 tokens per whitespace word, with a character-based floor for
    # languages and payloads (JSON, code) that do not split on spaces.
    return max(int(len(words) * 1.3) + 1, len(text) // 4)
