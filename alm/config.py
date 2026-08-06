"""Runtime configuration for the ALM federation.

Every knob is resolvable from an environment variable prefixed with ``ALM_``
(a ``.env`` file in the working directory is loaded automatically).  Nothing
here reaches the network at import time: :func:`get_settings` is a pure
resolution of environment into a frozen value object, so importing ``alm``
never requires a configured provider.

The federation degrades honestly.  If no orchestrator model is configured the
runtime still answers using the domain experts and simply reports that the
fallback path is unavailable, instead of pretending it succeeded.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:  # pragma: no cover - dotenv is a hard dependency but keep import defensive
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Resolved runtime settings.

    Attributes map one-to-one onto ``ALM_*`` environment variables; see
    ``.env.example`` for the full annotated list.
    """

    model_config = {"frozen": True}

    # -- storage -----------------------------------------------------------
    database_url: str = "sqlite:///alm.db"
    home: Path = Field(default_factory=lambda: Path(".alm"))
    packs_dir: Path = Field(default_factory=lambda: Path("packs"))

    # -- api ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8800
    api_token: str = ""

    # -- model tiers -------------------------------------------------------
    orchestrator_model: str = ""
    router_model: str = ""
    default_expert_model: str = ""

    # -- embeddings --------------------------------------------------------
    embedding_provider: str = "hash"
    embedding_model: str = ""
    embedding_dim: int = 384

    # -- backend endpoints -------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000/v1"
    openai_base_url: str = ""

    # -- routing behaviour -------------------------------------------------
    router_confidence_threshold: float = 0.55
    router_max_subtasks: int = 8
    fallback_enabled: bool = True

    # -- orchestration -----------------------------------------------------
    max_parallelism: int = 8
    step_timeout_seconds: float = 120.0
    step_max_retries: int = 1

    # -- arbitration -------------------------------------------------------
    arbitration_strategies: list[str] = Field(
        default_factory=lambda: ["verifier", "confidence", "authority", "judge"]
    )
    judge_enabled: bool = True
    conflict_confidence_margin: float = 0.15

    # -- governance --------------------------------------------------------
    governance_enabled: bool = True
    governance_default_decision: str = "allow"
    audit_enabled: bool = True

    # -- observability -----------------------------------------------------
    telemetry_enabled: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment."""
        strategies = _env(
            "ALM_ARBITRATION_STRATEGIES", "verifier,confidence,authority,judge"
        )
        return cls(
            database_url=_env("ALM_DATABASE_URL", "sqlite:///alm.db"),
            home=Path(_env("ALM_HOME", ".alm")),
            packs_dir=Path(_env("ALM_PACKS_DIR", "packs")),
            api_host=_env("ALM_API_HOST", "0.0.0.0"),
            api_port=_env_int("ALM_API_PORT", 8800),
            api_token=_env("ALM_API_TOKEN", ""),
            orchestrator_model=_env("ALM_ORCHESTRATOR_MODEL", ""),
            router_model=_env("ALM_ROUTER_MODEL", ""),
            default_expert_model=_env("ALM_DEFAULT_EXPERT_MODEL", ""),
            embedding_provider=_env("ALM_EMBEDDING_PROVIDER", "hash"),
            embedding_model=_env("ALM_EMBEDDING_MODEL", ""),
            embedding_dim=_env_int("ALM_EMBEDDING_DIM", 384),
            ollama_base_url=_env("ALM_OLLAMA_BASE_URL", "http://localhost:11434"),
            vllm_base_url=_env("ALM_VLLM_BASE_URL", "http://localhost:8000/v1"),
            openai_base_url=_env("ALM_OPENAI_BASE_URL", ""),
            router_confidence_threshold=_env_float("ALM_ROUTER_CONFIDENCE_THRESHOLD", 0.55),
            router_max_subtasks=_env_int("ALM_ROUTER_MAX_SUBTASKS", 8),
            fallback_enabled=_env_bool("ALM_FALLBACK_ENABLED", True),
            max_parallelism=_env_int("ALM_MAX_PARALLELISM", 8),
            step_timeout_seconds=_env_float("ALM_STEP_TIMEOUT_SECONDS", 120.0),
            step_max_retries=_env_int("ALM_STEP_MAX_RETRIES", 1),
            arbitration_strategies=[s.strip() for s in strategies.split(",") if s.strip()],
            judge_enabled=_env_bool("ALM_JUDGE_ENABLED", True),
            conflict_confidence_margin=_env_float("ALM_CONFLICT_CONFIDENCE_MARGIN", 0.15),
            governance_enabled=_env_bool("ALM_GOVERNANCE_ENABLED", True),
            governance_default_decision=_env("ALM_GOVERNANCE_DEFAULT_DECISION", "allow"),
            audit_enabled=_env_bool("ALM_AUDIT_ENABLED", True),
            telemetry_enabled=_env_bool("ALM_TELEMETRY_ENABLED", False),
            log_level=_env("ALM_LOG_LEVEL", "INFO"),
        )

    def redacted(self) -> dict[str, Any]:
        """Return settings as a dict with secrets masked, for display."""
        data = self.model_dump(mode="json")
        if data.get("api_token"):
            data["api_token"] = "***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings.from_env()


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests and the CLI after ``.env`` edits."""
    get_settings.cache_clear()
