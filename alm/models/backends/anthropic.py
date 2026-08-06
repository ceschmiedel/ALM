"""Anthropic backend — typically the orchestrator tier.

Used for planning, hard arbitration and as the teacher in the distillation
pipeline, not on every request.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from alm.core.errors import BackendError, ConfigurationError
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult


class AnthropicBackend(ChatBackend):
    """Messages API client."""

    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    api_version = "2023-06-01"

    def __init__(self, spec) -> None:  # noqa: ANN001
        super().__init__(spec)
        self._client: httpx.AsyncClient | None = None

    def base_url(self) -> str:
        return (self.spec.endpoint or self.default_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        key = self.spec.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ConfigurationError(
                f"model {self.spec.model_id!r} needs ANTHROPIC_API_KEY",
                model_id=self.spec.model_id,
            )
        return {
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": str(self.spec.params.get("api_version", self.api_version)),
        }

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=float(self.spec.params.get("timeout", 120.0))
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        # The Messages API takes the system prompt out of band.
        system_parts = [m.content for m in request.messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        if not turns:
            turns = [{"role": "user", "content": "\n\n".join(system_parts) or ""}]
            system_parts = []

        payload: dict[str, Any] = {
            "model": self.spec.served_name,
            "messages": turns,
            "max_tokens": self._resolved_max_tokens(request),
            "temperature": request.temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        payload.update(self._merged_params())

        response = await self._http().post(
            f"{self.base_url()}/messages", json=payload, headers=self._headers()
        )
        if response.status_code >= 400:
            raise BackendError(
                f"anthropic returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.name,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        data = response.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ).strip()
        usage = data.get("usage") or {}

        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=str(data.get("stop_reason") or "stop"),
            truncated=data.get("stop_reason") == "max_tokens",
            raw={"id": data.get("id", ""), "usage": usage},
        )
