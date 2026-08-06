"""Backend interface and shared machinery.

Every backend implements one coroutine, :meth:`ChatBackend._generate`.  The
public :meth:`ChatBackend.generate` wraps it with the concerns that are the
same everywhere and that the federation depends on: timing, token and cost
accounting, and a circuit breaker so one unreachable endpoint cannot stall
every expert bound to it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from alm.core.errors import BackendError, BackendUnavailableError
from alm.models.spec import (
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Trip after repeated failures; retry after a cooldown.

    A federation multiplies the number of endpoints in the path of a single
    answer.  Without this, one dead backend turns into N slow timeouts per
    request instead of one fast, explicit failure that routing can react to.
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            # Half-open: allow one probe through.
            self._opened_at = None
            self._failures = self.threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


class ChatBackend(ABC):
    """Abstract inference backend for one :class:`ModelSpec`."""

    #: Registry key used in ``ModelSpec.backend``.
    name: str = "base"
    #: Whether this backend can produce embeddings as well as completions.
    supports_embeddings: bool = False

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.breaker = CircuitBreaker()
        self._concurrency = asyncio.Semaphore(
            int(spec.params.get("max_concurrency", 8) or 8)
        )

    # -- subclass contract -------------------------------------------------

    @abstractmethod
    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Perform the call.  Return a result with ``text`` set.

        Populate ``prompt_tokens``/``completion_tokens`` when the provider
        reports them; leave them at zero to let the wrapper estimate.
        """

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise BackendError(
            f"backend {self.name!r} does not support embeddings", backend=self.name
        )

    async def health(self) -> bool:
        """Whether the backend is reachable.  Best-effort; never raises."""
        return not self.breaker.is_open

    async def close(self) -> None:  # noqa: B027 - optional hook, not every backend holds resources
        """Release resources (HTTP clients, loaded weights)."""

    # -- public API --------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a request with timing, accounting and circuit breaking."""
        if self.breaker.is_open:
            raise BackendUnavailableError(
                f"backend {self.name!r} for model {self.spec.model_id!r} is circuit-open",
                backend=self.name,
                model_id=self.spec.model_id,
            )

        started = time.perf_counter()
        try:
            async with self._concurrency:
                result = await self._generate(request)
        except BackendError:
            self.breaker.record_failure()
            raise
        except Exception as exc:
            self.breaker.record_failure()
            raise BackendError(
                f"{self.name} backend failed for model {self.spec.model_id!r}: {exc}",
                backend=self.name,
                model_id=self.spec.model_id,
            ) from exc

        self.breaker.record_success()

        result.latency_ms = (time.perf_counter() - started) * 1000.0
        result.model_id = self.spec.model_id
        result.served_name = self.spec.served_name
        result.tier = str(self.spec.tier)
        result.backend = self.name

        if result.prompt_tokens <= 0:
            result.prompt_tokens = estimate_tokens(request.prompt_text())
        if result.completion_tokens <= 0:
            result.completion_tokens = estimate_tokens(result.text)
        result.cost_usd = self.spec.cost_for(result.prompt_tokens, result.completion_tokens)
        return result

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, if this backend supports it."""
        if not texts:
            return []
        return await self._embed(texts)

    # -- helpers for subclasses -------------------------------------------

    def _resolved_max_tokens(self, request: GenerationRequest) -> int:
        return int(request.max_tokens or self.spec.max_output_tokens or 1024)

    def _merged_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Spec-level params merged with call-level overrides."""
        params = {
            k: v
            for k, v in self.spec.params.items()
            if k not in {"max_concurrency", "responses", "timeout"}
        }
        params.update(extra or {})
        return params

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model_id={self.spec.model_id!r} name={self.spec.served_name!r}>"
