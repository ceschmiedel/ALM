"""Ollama backend — the shortest path to a sovereign local federation.

Ollama is the default recommendation for getting an ALM federation running with
real models on a laptop or a single on-premise box: a micro-SLM for the router,
an SLM per domain expert, and embeddings from the same daemon, with no data
leaving the machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from alm.config import get_settings
from alm.core.errors import BackendError
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult


class OllamaBackend(ChatBackend):
    """Chat and embeddings against a local Ollama daemon."""

    name = "ollama"
    supports_embeddings = True

    def __init__(self, spec) -> None:  # noqa: ANN001
        super().__init__(spec)
        self._client: httpx.AsyncClient | None = None

    def base_url(self) -> str:
        return (self.spec.endpoint or get_settings().ollama_base_url).rstrip("/")

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = float(self.spec.params.get("timeout", 180.0))
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": self._resolved_max_tokens(request),
        }
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = request.stop
        if request.seed is not None:
            options["seed"] = request.seed
        options.update(self._merged_params())

        payload: dict[str, Any] = {
            "model": self.spec.served_name,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": options,
        }
        # Ollama enforces JSON natively, and will honour a JSON Schema when one
        # is available — worth using, because schema adherence is exactly where
        # small models are weakest.
        if request.json_mode:
            payload["format"] = request.response_schema or "json"

        response = await self._http().post(f"{self.base_url()}/api/chat", json=payload)
        if response.status_code >= 400:
            raise BackendError(
                f"ollama returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.name,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        data = response.json()
        message = data.get("message") or {}

        return GenerationResult(
            text=(message.get("content") or "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            finish_reason=str(data.get("done_reason") or "stop"),
            raw={"total_duration_ns": data.get("total_duration", 0)},
        )

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.spec.params.get("embedding_model") or self.spec.served_name
        response = await self._http().post(
            f"{self.base_url()}/api/embed", json={"model": model, "input": list(texts)}
        )
        if response.status_code >= 400:
            raise BackendError(
                f"ollama embeddings returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                backend=self.name,
                status_code=response.status_code,
            )
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        return [list(vector) for vector in (embeddings or [])]

    async def health(self) -> bool:
        if self.breaker.is_open:
            return False
        try:
            response = await self._http().get(f"{self.base_url()}/api/tags")
            return response.status_code < 500
        except Exception:
            return False
