"""Shared fixtures.

Every test gets an isolated SQLite database in a temp directory, so the suite
never touches a developer's working database and tests cannot leak state into
each other through the Context Graph.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the runtime at a fresh database for each test."""
    db_path = tmp_path / "alm-test.db"
    monkeypatch.setenv("ALM_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ALM_HOME", str(tmp_path / ".alm"))
    monkeypatch.setenv("ALM_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("ALM_EMBEDDING_DIM", "256")
    monkeypatch.setenv("ALM_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("ALM_LOG_LEVEL", "ERROR")
    monkeypatch.delenv("ALM_API_TOKEN", raising=False)
    monkeypatch.delenv("ALM_ROUTER_MODEL", raising=False)
    monkeypatch.delenv("ALM_ORCHESTRATOR_MODEL", raising=False)
    monkeypatch.delenv("ALM_DEFAULT_EXPERT_MODEL", raising=False)

    from alm.config import reset_settings_cache
    from alm.persistence.database import init_db, reset_engine

    reset_settings_cache()
    reset_engine()
    init_db()

    yield db_path

    reset_engine()
    reset_settings_cache()


@pytest.fixture
def graph():
    from alm.cmrag.embeddings import HashEmbedder
    from alm.graph.store import ContextGraph

    return ContextGraph("default", embedder=HashEmbedder(dim=256).embed)


@pytest.fixture
def embedder():
    from alm.cmrag.embeddings import HashEmbedder

    return HashEmbedder(dim=256)


@pytest.fixture
def registry():
    from alm.models.registry import ModelRegistry

    return ModelRegistry()


@pytest.fixture
def pack_path() -> Path:
    """The bundled demo pack — the suite's end-to-end fixture."""
    path = Path(__file__).resolve().parent.parent / "packs" / "demo-enterprise"
    if not path.exists():  # pragma: no cover
        pytest.skip("demo pack is not present")
    return path


@pytest.fixture
def runtime(pack_path: Path):
    """A federation runtime with the demo pack installed."""
    from alm.federation.runtime import FederationRuntime

    return FederationRuntime.from_pack(pack_path)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("ALM_LOG_LEVEL", "ERROR")
