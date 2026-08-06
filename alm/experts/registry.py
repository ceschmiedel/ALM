"""Expert registry — rebuilding the federation from the Context Graph.

Because installation stores each expert's full specification on its graph node,
the runtime can reconstruct every Expert Agent from the database alone.  A
production deployment therefore does not need the pack directory on disk, and
adding an expert to a running federation is a graph write followed by a
:meth:`ExpertRegistry.refresh` — no restart, no code change.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from alm.cmrag.retriever import CMRAGRetriever
from alm.experts.agent import ExpertAgent
from alm.experts.calibration import CalibrationStore
from alm.experts.spec import ExpertSpec
from alm.governance.ibac import IBACEngine
from alm.graph.models import NodeKind
from alm.graph.store import ContextGraph
from alm.models.serving import ServingRouter

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """Live collection of :class:`ExpertAgent` instances, keyed by id."""

    def __init__(
        self,
        graph: ContextGraph,
        serving: ServingRouter,
        *,
        retriever: CMRAGRetriever | None = None,
        ibac: IBACEngine | None = None,
        calibrations: CalibrationStore | None = None,
    ) -> None:
        self.graph = graph
        self.serving = serving
        self.retriever = retriever
        self.ibac = ibac
        self.calibrations = calibrations or CalibrationStore(graph.tenant_id)
        self._agents: dict[str, ExpertAgent] = {}

    # -- construction ------------------------------------------------------

    def refresh(self) -> int:
        """Rebuild every agent from the graph.  Returns how many were loaded."""
        agents: dict[str, ExpertAgent] = {}
        for node in self.graph.find_nodes(NodeKind.EXPERT):
            raw = node.attributes.get("spec")
            if not isinstance(raw, dict):
                logger.warning(
                    "Expert node %r carries no specification; skipping. Reinstall the "
                    "pack that declared it.",
                    node.key,
                )
                continue
            try:
                spec = ExpertSpec.model_validate(raw)
            except Exception:
                logger.warning("Expert %r has an invalid specification", node.key, exc_info=True)
                continue
            if not spec.enabled:
                continue
            agents[spec.id] = self._build(spec)

        self._agents = agents
        logger.info("Expert registry loaded %d agent(s)", len(agents))
        return len(agents)

    def add(self, spec: ExpertSpec) -> ExpertAgent:
        """Register an agent directly, without going through a pack."""
        agent = self._build(spec)
        self._agents[spec.id] = agent
        return agent

    def _build(self, spec: ExpertSpec) -> ExpertAgent:
        return ExpertAgent(
            spec,
            serving=self.serving,
            retriever=self.retriever,
            ibac=self.ibac,
            calibrations=self.calibrations,
        )

    # -- access ------------------------------------------------------------

    def get(self, expert_id: str) -> ExpertAgent | None:
        return self._agents.get(expert_id)

    def require(self, expert_id: str) -> ExpertAgent:
        agent = self._agents.get(expert_id)
        if agent is None:
            known = ", ".join(sorted(self._agents)) or "none"
            raise KeyError(f"expert {expert_id!r} is not registered (known: {known})")
        return agent

    def for_capability(self, capability_id: str) -> ExpertAgent | None:
        for agent in self._agents.values():
            if agent.spec.capability(capability_id) is not None and any(
                c.id == capability_id for c in agent.spec.capabilities
            ):
                return agent
        return None

    def for_domain(self, domain: str) -> list[ExpertAgent]:
        return [a for a in self._agents.values() if a.domain == domain]

    def ids(self) -> list[str]:
        return sorted(self._agents)

    def domains(self) -> list[str]:
        return sorted({a.domain for a in self._agents.values()})

    def specs(self) -> list[ExpertSpec]:
        return [a.spec for a in self._agents.values()]

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[ExpertAgent]:
        return iter(self._agents.values())

    def __contains__(self, expert_id: object) -> bool:
        return expert_id in self._agents

    # -- introspection -----------------------------------------------------

    def inventory(self) -> list[dict[str, object]]:
        """Per-expert summary used by the CLI and the API."""
        out = []
        for agent in sorted(self._agents.values(), key=lambda a: a.id):
            calibration = self.calibrations.get(agent.id)
            out.append(
                {
                    "id": agent.id,
                    "domain": agent.domain,
                    "label": agent.spec.label,
                    "model": agent.spec.model,
                    "tier": str(agent.spec.tier),
                    "capabilities": [c.id for c in agent.spec.capabilities],
                    "retrieval_domains": agent.spec.retrieval.domains,
                    "verifier_rules": len(agent.spec.verifier_rules),
                    "authority": agent.spec.authority,
                    "calibrated": calibration.fitted,
                    "calibration_samples": calibration.samples,
                }
            )
        return out
