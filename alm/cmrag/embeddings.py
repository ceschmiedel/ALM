"""Embedding providers.

Four providers cover the deployment spectrum without forcing a dependency on
any of them:

``hash``
    Deterministic, offline, zero-dependency signed-hashing projection.  Real
    lexical similarity, no semantics — enough to run and test the whole
    pipeline, and reproducible, which matters for the evaluation harness.
``ollama`` / ``sentence``
    Local embeddings, nothing leaves the perimeter.  The sovereign default.
``openai``
    A hosted or self-hosted OpenAI-compatible embeddings endpoint.

Whichever is chosen, the vector dimension is fixed per index: changing the
provider requires re-indexing, and :class:`Embedder.signature` exists so the
runtime can detect that instead of silently comparing incompatible vectors.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from alm.config import Settings, get_settings
from alm.core.errors import BackendError, ConfigurationError

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+")


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


class Embedder(ABC):
    """Maps texts to fixed-dimension vectors."""

    name: str = "embedder"

    def __init__(self, dim: int, model: str = "") -> None:
        self.dim = dim
        self.model = model

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts (synchronous; providers batch internally)."""

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0] if vectors else [0.0] * self.dim

    @property
    def signature(self) -> str:
        """Identity of the vector space; stored alongside every indexed chunk."""
        return f"{self.name}:{self.model or 'default'}:{self.dim}"

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)


class HashEmbedder(Embedder):
    """Signed hashing of word and character n-grams — deterministic and offline.

    This is the feature-hashing trick, not a learned model: similarity here is
    lexical overlap projected into a fixed space.  It will not recognise that
    "indemnity" and "hold harmless" are related.  It *will* run anywhere, cost
    nothing, and give byte-identical vectors across machines and runs, which is
    what makes the demo and the consistency metric reproducible.
    """

    name = "hash"

    def __init__(self, dim: int = 384, model: str = "", char_ngrams: int = 4) -> None:
        super().__init__(dim=dim, model=model or "signed-hashing")
        self.char_ngrams = char_ngrams

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            lowered = (text or "").lower()
            tokens = _TOKEN_RE.findall(lowered)

            for token in tokens:
                index, sign = self._bucket(f"w:{token}")
                vector[index] += sign
            # Bigrams capture a little word order, which pure bag-of-words loses.
            for a, b in zip(tokens, tokens[1:], strict=False):
                index, sign = self._bucket(f"b:{a}_{b}")
                vector[index] += sign * 0.6
            # Character n-grams give partial credit for morphology and typos.
            joined = " ".join(tokens)
            n = self.char_ngrams
            for i in range(max(0, len(joined) - n + 1)):
                index, sign = self._bucket(f"c:{joined[i : i + n]}")
                vector[index] += sign * 0.3

            out.append(_l2_normalise(vector))
        return out


class OllamaEmbedder(Embedder):
    """Local embeddings from an Ollama daemon (e.g. ``nomic-embed-text``)."""

    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", dim: int = 768, base_url: str = "") -> None:
        super().__init__(dim=dim, model=model)
        self.base_url = (base_url or get_settings().ollama_base_url).rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise BackendError(
                f"ollama embeddings unreachable at {self.base_url}: {exc}",
                backend="ollama",
            ) from exc
        if response.status_code >= 400:
            raise BackendError(
                f"ollama embeddings returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                backend="ollama",
                status_code=response.status_code,
            )
        data = response.json()
        vectors = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else [])
        result = [list(map(float, v)) for v in vectors]
        if result:
            self.dim = len(result[0])
        return result


class OpenAIEmbedder(Embedder):
    """OpenAI-compatible embeddings endpoint (hosted or self-hosted)."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "",
        api_key: str = "",
    ) -> None:
        super().__init__(dim=dim, model=model)
        self.base_url = (base_url or get_settings().openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI embeddings need OPENAI_API_KEY (or switch "
                "ALM_EMBEDDING_PROVIDER to 'ollama' or 'hash')"
            )
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": list(texts)},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise BackendError(
                f"openai embeddings unreachable: {exc}", backend="openai"
            ) from exc
        if response.status_code >= 400:
            raise BackendError(
                f"openai embeddings returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                backend="openai",
                status_code=response.status_code,
            )
        items = sorted(response.json().get("data", []), key=lambda d: d.get("index", 0))
        result = [list(map(float, item["embedding"])) for item in items]
        if result:
            self.dim = len(result[0])
        return result


class SentenceTransformerEmbedder(Embedder):
    """Fully local embeddings via sentence-transformers.

    Requires the optional extra::

        pip install "draiven-alm[embeddings]"
    """

    name = "sentence"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384) -> None:
        super().__init__(dim=dim, model=model)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                'sentence-transformers is not installed: pip install "draiven-alm[embeddings]"'
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


_PROVIDERS = {
    "hash": HashEmbedder,
    "ollama": OllamaEmbedder,
    "openai": OpenAIEmbedder,
    "sentence": SentenceTransformerEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
}


def available_embedders() -> list[str]:
    return sorted(_PROVIDERS)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Build the configured embedder.

    Falls back to :class:`HashEmbedder` when a remote provider cannot be
    constructed, and says so loudly — retrieval quality drops, but the
    federation keeps answering, which is the right trade for an outage.
    """
    resolved = settings or get_settings()
    provider = (resolved.embedding_provider or "hash").strip().lower()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ConfigurationError(
            f"unknown embedding provider {provider!r}; available: "
            f"{', '.join(available_embedders())}"
        )

    kwargs: dict[str, object] = {}
    if resolved.embedding_model:
        kwargs["model"] = resolved.embedding_model
    if provider in {"hash"}:
        kwargs["dim"] = resolved.embedding_dim

    try:
        return factory(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning(
            "Embedding provider %r unavailable (%s); falling back to deterministic "
            "hash embeddings — retrieval quality will be lexical only",
            provider,
            exc,
        )
        return HashEmbedder(dim=resolved.embedding_dim)
