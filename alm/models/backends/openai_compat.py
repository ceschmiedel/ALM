"""OpenAI-compatible backend — covers OpenAI, vLLM, Azure and any gateway.

One implementation serves three very different deployments because they speak
the same wire format:

* **OpenAI / any hosted gateway** — the orchestrator tier.
* **vLLM** — the on-premise workhorse.  This is where multi-adapter serving
  happens: the server loads one base model and registers LoRA adapters under
  their own names, so ``spec.adapter`` is simply the model name on the wire and
  dozens of experts share a single GPU-resident base.
* **Azure OpenAI** — same protocol, different URL shape; see
  :class:`AzureOpenAIBackend`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import httpx

from alm.config import get_settings
from alm.core.errors import BackendError, ConfigurationError
from alm.models.backends.base import ChatBackend
from alm.models.spec import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)


class OpenAICompatBackend(ChatBackend):
    """Chat completions over the OpenAI HTTP API shape."""

    name = "openai"
    supports_embeddings = True

    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    requires_api_key = True

    def __init__(self, spec) -> None:  # noqa: ANN001 - spec typed on the base class
        super().__init__(spec)
        self._client: httpx.AsyncClient | None = None
        self._json_mode_supported = True

    # -- wiring ------------------------------------------------------------

    def base_url(self) -> str:
        return (self.spec.endpoint or self.default_base_url).rstrip("/")

    def api_key(self) -> str:
        return self.spec.api_key or os.getenv(self.api_key_env, "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif self.requires_api_key:
            raise ConfigurationError(
                f"model {self.spec.model_id!r} needs an API key: set {self.api_key_env} "
                f"or store one on the model with `alm model add --api-key`",
                model_id=self.spec.model_id,
            )
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = float(self.spec.params.get("timeout", 120.0))
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- generation --------------------------------------------------------

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.served_name,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": self._resolved_max_tokens(request),
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.json_mode and self._json_mode_supported:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self._merged_params())
        return payload

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url()}/chat/completions"
        response = await self._http().post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise BackendError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.name,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        return response.json()

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._payload(request)
        try:
            data = await self._post_chat(payload)
        except BackendError as exc:
            # Servers that do not implement JSON mode reject the whole request.
            # Retry once in plain mode rather than declaring the expert broken;
            # tolerant parsing downstream handles the un-guaranteed output.
            if request.json_mode and self._json_mode_supported and exc.details.get(
                "status_code"
            ) in {400, 404, 422}:
                logger.info(
                    "Endpoint for %s rejected JSON mode; retrying without it",
                    self.spec.model_id,
                )
                self._json_mode_supported = False
                payload.pop("response_format", None)
                data = await self._post_chat(payload)
            else:
                raise

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(
                f"{self.name} returned no choices for model {self.spec.model_id!r}",
                backend=self.name,
                model_id=self.spec.model_id,
            )
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}

        return GenerationResult(
            text=(message.get("content") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
            truncated=choices[0].get("finish_reason") == "length",
            raw={"id": data.get("id", ""), "usage": usage},
        )

    # -- embeddings --------------------------------------------------------

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.spec.params.get("embedding_model") or self.spec.served_name
        url = f"{self.base_url()}/embeddings"
        response = await self._http().post(
            url, json={"model": model, "input": list(texts)}, headers=self._headers()
        )
        if response.status_code >= 400:
            raise BackendError(
                f"{self.name} embeddings returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                backend=self.name,
                status_code=response.status_code,
            )
        data = response.json()
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [list(item["embedding"]) for item in items]

    async def health(self) -> bool:
        if self.breaker.is_open:
            return False
        try:
            response = await self._http().get(
                f"{self.base_url()}/models", headers=self._headers()
            )
            return response.status_code < 500
        except Exception:
            return False


class VLLMBackend(OpenAICompatBackend):
    """vLLM server — the on-premise, multi-adapter path.

    Identical protocol; the difference is operational.  Point ``endpoint`` at
    the server, set ``base_model`` to the loaded base and ``adapter`` to the
    LoRA name the server registered, and every expert with the same base shares
    one set of GPU-resident weights.
    """

    name = "vllm"
    requires_api_key = False
    api_key_env = "VLLM_API_KEY"

    def base_url(self) -> str:
        return (self.spec.endpoint or get_settings().vllm_base_url).rstrip("/")


class OpenAIBackend(OpenAICompatBackend):
    """Hosted OpenAI (or a drop-in gateway via ``ALM_OPENAI_BASE_URL``)."""

    name = "openai"

    def base_url(self) -> str:
        return (
            self.spec.endpoint or get_settings().openai_base_url or self.default_base_url
        ).rstrip("/")


class AzureOpenAIBackend(OpenAICompatBackend):
    """Azure OpenAI — same payloads, deployment-scoped URLs and an api-key header."""

    name = "azure"
    api_key_env = "AZURE_OPENAI_API_KEY"

    def base_url(self) -> str:
        endpoint = self.spec.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if not endpoint:
            raise ConfigurationError(
                "Azure backend needs an endpoint: set AZURE_OPENAI_ENDPOINT or pass "
                "--endpoint when registering the model",
                model_id=self.spec.model_id,
            )
        return endpoint.rstrip("/")

    def _api_version(self) -> str:
        return str(
            self.spec.params.get("api_version")
            or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )

    def _headers(self) -> dict[str, str]:
        key = self.api_key()
        if not key:
            raise ConfigurationError(
                "Azure backend needs AZURE_OPENAI_API_KEY", model_id=self.spec.model_id
            )
        return {"Content-Type": "application/json", "api-key": key}

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        deployment = self.spec.params.get("deployment") or self.spec.served_name
        url = (
            f"{self.base_url()}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self._api_version()}"
        )
        payload.pop("model", None)
        response = await self._http().post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise BackendError(
                f"azure returned HTTP {response.status_code}: {response.text[:400]}",
                backend=self.name,
                status_code=response.status_code,
                model_id=self.spec.model_id,
            )
        return response.json()

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        deployment = (
            self.spec.params.get("embedding_deployment")
            or self.spec.params.get("deployment")
            or self.spec.served_name
        )
        url = (
            f"{self.base_url()}/openai/deployments/{deployment}/embeddings"
            f"?api-version={self._api_version()}"
        )
        response = await self._http().post(
            url, json={"input": list(texts)}, headers=self._headers()
        )
        if response.status_code >= 400:
            raise BackendError(
                f"azure embeddings returned HTTP {response.status_code}: {response.text[:300]}",
                backend=self.name,
                status_code=response.status_code,
            )
        items = sorted(response.json().get("data", []), key=lambda d: d.get("index", 0))
        return [list(item["embedding"]) for item in items]

    async def health(self) -> bool:
        return not self.breaker.is_open
