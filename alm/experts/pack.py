"""Domain packs — a whole vertical as data.

A pack declares everything that makes a federation domain-competent: the
ontology, the Expert Agents and their models, the corpus CMRAG indexes, the
IBAC policies, and the evaluation set that decides whether any of it is
actually better than the baseline.  Installing a pack writes all of that into
the Context Graph, the model registry and the corpus store.

```
packs/my-domain/
├── pack.yaml
├── ontology/legal.yaml
├── experts/contracts.yaml
├── corpus/legal/*.md
├── policies/ibac.yaml
└── eval/legal.jsonl
```

Validation runs before installation and is strict on purpose: a capability
pointing at an entity type the ontology does not define, or an expert bound to
an unregistered model, would show up later as inexplicably bad routing.  Better
to fail at `alm pack validate`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from alm.cmrag.embeddings import Embedder
from alm.cmrag.ingest import CorpusIngestor, EntityAnnotator
from alm.cmrag.store import ChunkStore
from alm.core.errors import PackError
from alm.experts.spec import ExpertSpec
from alm.governance.policies import PolicySet
from alm.graph.ontology import Ontology
from alm.graph.store import ContextGraph
from alm.models.registry import ModelRegistry, spec_from_mapping
from alm.models.spec import ModelSpec

logger = logging.getLogger(__name__)

PACK_FILE = "pack.yaml"


class PackDefaults(BaseModel):
    """Model bindings a pack expects, by role."""

    expert_model: str = ""
    router_model: str = ""
    orchestrator_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""


class DomainPack(BaseModel):
    """A loaded, not-yet-installed domain pack."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    tenant: str = "default"
    root: Path = Field(default_factory=Path)

    ontologies: list[Ontology] = Field(default_factory=list)
    experts: list[ExpertSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    policies: PolicySet = Field(default_factory=PolicySet)
    defaults: PackDefaults = Field(default_factory=PackDefaults)

    corpus_dir: Path | None = None
    eval_files: list[Path] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> DomainPack:
        """Read a pack directory (or its ``pack.yaml``) from disk."""
        root = Path(path)
        if root.is_file():
            manifest_path = root
            root = root.parent
        else:
            manifest_path = root / PACK_FILE
        if not manifest_path.exists():
            raise PackError(
                f"no {PACK_FILE} found at {root}", path=str(root)
            )

        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise PackError(f"invalid YAML in {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise PackError(f"{manifest_path} must contain a mapping")

        name = str(manifest.get("name") or root.name)

        ontologies = [
            Ontology.from_yaml(_resolve(root, entry))
            for entry in _as_list(manifest.get("ontologies") or manifest.get("ontology"))
        ]
        if not ontologies:
            ontologies = [
                Ontology.from_yaml(p) for p in sorted((root / "ontology").glob("*.y*ml"))
            ]

        expert_entries = _as_list(manifest.get("experts"))
        if expert_entries:
            experts = [ExpertSpec.from_yaml(_resolve(root, e)) for e in expert_entries]
        else:
            experts = [
                ExpertSpec.from_yaml(p) for p in sorted((root / "experts").glob("*.y*ml"))
            ]

        models = [
            spec_from_mapping(model_id, data or {})
            for model_id, data in (manifest.get("models") or {}).items()
        ]

        policies = PolicySet()
        policy_entries = _as_list(manifest.get("policies"))
        if not policy_entries and (root / "policies").exists():
            policy_entries = [
                str(p.relative_to(root)) for p in sorted((root / "policies").glob("*.y*ml"))
            ]
        for entry in policy_entries:
            policies = policies.merge(PolicySet.from_yaml(_resolve(root, entry)))

        corpus_value = manifest.get("corpus", "corpus")
        corpus_dir = _resolve(root, corpus_value) if corpus_value else None
        if corpus_dir is not None and not corpus_dir.exists():
            corpus_dir = None

        eval_entries = _as_list(manifest.get("eval"))
        if not eval_entries and (root / "eval").exists():
            eval_entries = [
                str(p.relative_to(root)) for p in sorted((root / "eval").glob("*.jsonl"))
            ]
        eval_files = [_resolve(root, e) for e in eval_entries]

        return cls(
            name=name,
            version=str(manifest.get("version", "0.1.0")),
            description=str(manifest.get("description", "")),
            tenant=str(manifest.get("tenant", "default")),
            root=root,
            ontologies=ontologies,
            experts=experts,
            models=models,
            policies=policies,
            defaults=PackDefaults.model_validate(manifest.get("defaults", {}) or {}),
            corpus_dir=corpus_dir,
            eval_files=eval_files,
        )

    # -- validation --------------------------------------------------------

    def validate(self, *, registry: ModelRegistry | None = None) -> list[str]:
        """Return every problem found.  Empty means the pack is installable."""
        problems: list[str] = []

        if not self.ontologies:
            problems.append("pack declares no ontology; the router has nothing to match on")
        if not self.experts:
            problems.append("pack declares no experts")

        entity_names: set[str] = set()
        domains: set[str] = set()
        for ontology in self.ontologies:
            problems.extend(
                f"ontology '{ontology.domain}': {p}"
                for p in ontology.validate_consistency()
            )
            entity_names.update(ontology.entity_names())
            domains.add(ontology.domain)

        seen_experts: set[str] = set()
        seen_capabilities: dict[str, str] = {}
        declared_models = {m.model_id for m in self.models}

        for expert in self.experts:
            if expert.id in seen_experts:
                problems.append(f"duplicate expert id '{expert.id}'")
            seen_experts.add(expert.id)

            if expert.domain not in domains:
                problems.append(
                    f"expert '{expert.id}' declares domain '{expert.domain}', "
                    f"which no ontology in this pack defines"
                )
            if not expert.capabilities:
                problems.append(f"expert '{expert.id}' declares no capabilities")

            problems.extend(expert.validate_against_ontology(entity_names))

            for cap in expert.capabilities:
                owner = seen_capabilities.get(cap.id)
                if owner is not None:
                    problems.append(
                        f"capability '{cap.id}' is declared by both '{owner}' and "
                        f"'{expert.id}'; capability ids must be unique so routing is "
                        f"unambiguous"
                    )
                seen_capabilities[cap.id] = expert.id
                if not cap.description and not cap.examples:
                    problems.append(
                        f"capability '{cap.id}' has neither a description nor examples; "
                        f"the router has no signal to match it"
                    )

            if expert.model:
                known = expert.model in declared_models or (
                    registry is not None and registry.exists(expert.model)
                )
                if not known:
                    problems.append(
                        f"expert '{expert.id}' is bound to model '{expert.model}', "
                        f"which is neither declared in this pack nor registered"
                    )

            for domain in expert.retrieval.domains:
                if domain not in domains:
                    problems.append(
                        f"expert '{expert.id}' retrieves from domain '{domain}', "
                        f"which this pack does not define"
                    )

        for role, model_id in (
            ("expert_model", self.defaults.expert_model),
            ("router_model", self.defaults.router_model),
            ("orchestrator_model", self.defaults.orchestrator_model),
        ):
            if model_id and model_id not in declared_models:
                if registry is None or not registry.exists(model_id):
                    problems.append(
                        f"defaults.{role} points at unknown model '{model_id}'"
                    )

        return problems

    # -- installation ------------------------------------------------------

    def install(
        self,
        *,
        graph: ContextGraph,
        registry: ModelRegistry,
        store: ChunkStore | None = None,
        embedder: Embedder | None = None,
        index_corpus: bool = True,
    ) -> dict[str, Any]:
        """Write the pack into the graph, the registry and the corpus store."""
        problems = self.validate(registry=registry)
        if problems:
            raise PackError(
                f"pack '{self.name}' has {len(problems)} problem(s); "
                f"run `alm pack validate` for the list",
                problems=problems,
                pack=self.name,
            )

        report: dict[str, Any] = {
            "pack": self.name,
            "version": self.version,
            "models": 0,
            "domains": [],
            "entities": 0,
            "experts": [],
            "capabilities": 0,
            "documents": 0,
            "chunks": 0,
            "policies": len(self.policies),
        }

        for model in self.models:
            registry.register(model)
            report["models"] += 1

        for ontology in self.ontologies:
            applied = ontology.apply(graph)
            report["domains"].append(ontology.domain)
            report["entities"] += applied["entities"]

        for expert in self.experts:
            if not expert.enabled:
                continue
            graph.register_expert(
                expert.id,
                domain=expert.domain,
                description=expert.description or expert.label,
                model_id=expert.model,
                capabilities=expert.declarations(),
                authority=expert.authority,
                # The full spec rides along on the node so the runtime can
                # rebuild the federation from the database alone, with no pack
                # directory present.
                attributes={
                    "spec": expert.model_dump(mode="json"),
                    "pack": self.name,
                    "tier": str(expert.tier),
                },
            )
            report["experts"].append(expert.id)
            report["capabilities"] += len(expert.capabilities)

        if index_corpus and store is not None and self.corpus_dir is not None:
            annotator = EntityAnnotator()
            for ontology in self.ontologies:
                annotator.load(ontology)
            ingestor = CorpusIngestor(store, embedder, annotator=annotator)
            for entry in ingestor.ingest_directory(self.corpus_dir):
                report["documents"] += 0 if entry.get("skipped") else 1
                report["chunks"] += entry.get("chunks", 0)

        logger.info(
            "Installed pack %s v%s: %d expert(s), %d capability(ies), %d chunk(s)",
            self.name,
            self.version,
            len(report["experts"]),
            report["capabilities"],
            report["chunks"],
        )
        return report

    # -- helpers -----------------------------------------------------------

    def ontology_for(self, domain: str) -> Ontology | None:
        for ontology in self.ontologies:
            if ontology.domain == domain:
                return ontology
        return None

    def expert(self, expert_id: str) -> ExpertSpec | None:
        for expert in self.experts:
            if expert.id == expert_id:
                return expert
        return None

    def domains(self) -> list[str]:
        return [o.domain for o in self.ontologies]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "root": str(self.root),
            "domains": self.domains(),
            "experts": [e.id for e in self.experts],
            "models": [m.model_id for m in self.models],
            "policies": len(self.policies),
            "eval_files": [str(p) for p in self.eval_files],
            "corpus": str(self.corpus_dir) if self.corpus_dir else None,
        }


def load_pack(path: str | Path) -> DomainPack:
    """Load a domain pack from a directory or ``pack.yaml`` path."""
    return DomainPack.load(path)


def discover_packs(directory: str | Path) -> list[Path]:
    """Return every pack directory under ``directory``."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob(f"*/{PACK_FILE}"))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _resolve(root: Path, entry: str | Path) -> Path:
    path = Path(entry)
    return path if path.is_absolute() else root / path
