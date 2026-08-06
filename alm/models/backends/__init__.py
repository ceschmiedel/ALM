"""Inference backends and the name → class registry.

Backends are resolved by the string in ``ModelSpec.backend``.  Optional
backends (``transformers``) are imported lazily so that the base install never
requires torch.
"""

from __future__ import annotations

from collections.abc import Callable

from alm.core.errors import ConfigurationError
from alm.models.backends.anthropic import AnthropicBackend
from alm.models.backends.base import ChatBackend, CircuitBreaker
from alm.models.backends.google import GoogleBackend
from alm.models.backends.heuristic import HeuristicBackend
from alm.models.backends.ollama import OllamaBackend
from alm.models.backends.openai_compat import (
    AzureOpenAIBackend,
    OpenAIBackend,
    OpenAICompatBackend,
    VLLMBackend,
)
from alm.models.backends.scripted import ScriptedBackend
from alm.models.spec import ModelSpec


def _transformers_backend() -> type[ChatBackend]:
    from alm.models.backends.transformers_local import TransformersBackend

    return TransformersBackend


_BACKENDS: dict[str, type[ChatBackend] | Callable[[], type[ChatBackend]]] = {
    "openai": OpenAIBackend,
    "openai_compat": OpenAICompatBackend,
    "vllm": VLLMBackend,
    "azure": AzureOpenAIBackend,
    "anthropic": AnthropicBackend,
    "google": GoogleBackend,
    "gemini": GoogleBackend,
    "ollama": OllamaBackend,
    "heuristic": HeuristicBackend,
    "scripted": ScriptedBackend,
    "transformers": _transformers_backend,
    "local": _transformers_backend,
}


def available_backends() -> list[str]:
    """Names accepted in ``ModelSpec.backend``."""
    return sorted(_BACKENDS)


def register_backend(name: str, backend_class: type[ChatBackend]) -> None:
    """Register a third-party backend under ``name``."""
    _BACKENDS[name.strip().lower()] = backend_class


def resolve_backend_class(name: str) -> type[ChatBackend]:
    entry = _BACKENDS.get((name or "").strip().lower())
    if entry is None:
        raise ConfigurationError(
            f"unknown backend {name!r}; available: {', '.join(available_backends())}",
            backend=name,
        )
    if isinstance(entry, type):
        return entry
    return entry()


def create_backend(spec: ModelSpec) -> ChatBackend:
    """Instantiate the backend a spec asks for."""
    return resolve_backend_class(spec.backend)(spec)


__all__ = [
    "AnthropicBackend",
    "AzureOpenAIBackend",
    "ChatBackend",
    "CircuitBreaker",
    "GoogleBackend",
    "HeuristicBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatBackend",
    "ScriptedBackend",
    "VLLMBackend",
    "available_backends",
    "create_backend",
    "register_backend",
    "resolve_backend_class",
]
