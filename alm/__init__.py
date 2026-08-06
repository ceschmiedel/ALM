"""ALM — Agent Language Model.

A governed federation of domain models orchestrated by a Context Graph.

Instead of routing every agent through one monolithic LLM, ALM gives each
Expert Agent its own model (a micro-SLM, an SLM, or a LoRA adapter over a
shared base) and moves the intelligence into the orchestration: an explicit
cognitive router (L2), a dependency DAG (L4), and an explicit arbitration and
synthesis layer (L5) — all reading and writing a Context Graph, all governed
by IBAC.

Typical embedded use::

    from alm import FederationRuntime, load_pack

    runtime = FederationRuntime.from_pack("packs/acme-contracts")
    result = await runtime.run("Is clause 7.2 of the Acme MSA enforceable?")
    print(result.answer)
    print(result.trace.render())

The public surface is intentionally small; everything else is reachable
through the submodules (``alm.graph``, ``alm.router``, ``alm.experts``, …).
"""

from __future__ import annotations

__version__ = "0.1.0"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from alm.config import Settings, get_settings
    from alm.experts.pack import DomainPack, load_pack
    from alm.federation.result import FederationResult
    from alm.federation.runtime import FederationRuntime

__all__ = [
    "__version__",
    "DomainPack",
    "FederationResult",
    "FederationRuntime",
    "Settings",
    "get_settings",
    "load_pack",
]

_LAZY: dict[str, tuple[str, str]] = {
    "DomainPack": ("alm.experts.pack", "DomainPack"),
    "FederationResult": ("alm.federation.result", "FederationResult"),
    "FederationRuntime": ("alm.federation.runtime", "FederationRuntime"),
    "Settings": ("alm.config", "Settings"),
    "get_settings": ("alm.config", "get_settings"),
    "load_pack": ("alm.experts.pack", "load_pack"),
}


def __getattr__(name: str) -> Any:
    """Import public symbols lazily so ``import alm`` stays cheap."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'alm' has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    return getattr(import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)
