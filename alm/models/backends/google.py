"""Google Gemini backend (generative language API)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from alm.core.errors import BackendError, ConfigurationError
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult


class GoogleBackend(ChatBackend):
    """Gemini ``generateContent`` client."""

    name = "google"
    supports_embeddings = True
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, spec) -> None:  # noqa: ANN001
        super().__init__(spec)
        self._client: httpx.AsyncClient | None = None

    def base_url(self) -> str:
        return (self.spec.endpoint or self.default_base_url).rstrip("/")

    def _api_key(self) -> str:
        key = self.spec.api_key or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ConfigurationError(
                f"model {self.spec.model_id!r} needs GOOGLE_API_KEY",
                model_id=self.spec.model_id,
            )
        return key

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
        system_parts = [m.content for m in request.messages if m.role == "system"]
        contents = [
            {
                "role": "user" if m.role == "user" else "model",
                "parts": [{"text": m.content}],
            }
            for m in request.messages
            if m.role != "system"
        ]
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]}]
            system_parts = []

        generation_config: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": self._resolved_max_tokens(request),
        }
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.stop:
            generation_config["stopSequences"] = request.stop
        if request.json_mode:
            generation_config["responseMimeType"] = "application/json"
            if request.response_schema:
                generation_config["responseSchema"] = request.response_schema

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        url = f"{self.base_url()}/models/{self.spec.served_name}:generateContent"
        response = await self._http().post(
            url, json=payload, headers={"x-goog-api-key": self._api_key()}
        )
        if response.status_code >= 400:
            raise BackendError(
                f"google returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.name,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        data = response.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
        usage = data.get("usageMetadata") or {}

        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            finish_reason=str(
                (candidates[0].get("finishReason") if candidates else "") or "stop"
            ).lower(),
            raw={"usage": usage},
        )

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.spec.params.get("embedding_model") or "text-embedding-004"
        url = f"{self.base_url()}/models/{model}:batchEmbedContents"
        payload = {
            "requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        response = await self._http().post(
            url, json=payload, headers={"x-goog-api-key": self._api_key()}
        )
        if response.status_code >= 400:
            raise BackendError(
                f"google embeddings returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                backend=self.name,
                status_code=response.status_code,
            )
        return [
            list(item.get("values", []))
            for item in response.json().get("embeddings", [])
        ]
