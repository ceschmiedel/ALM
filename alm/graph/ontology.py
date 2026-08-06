"""Domain ontology — the backbone that ties a domain model to the federation.

The ontology does three jobs at once, which is why it is a single artifact
rather than three:

1. it defines the vocabulary and relations the domain model must master;
2. it structures training data (every distilled example references its
   entities), and
3. it is the schema CMRAG retrieves against.

Modelling the ontology and training the model are the same task seen from two
angles.  Concretely, an ontology is a YAML file inside a domain pack that is
loaded here and projected onto the Context Graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from alm.core.errors import OntologyError
from alm.graph.models import Node, NodeKind, Relation
from alm.graph.store import ContextGraph


class EntityType(BaseModel):
    """One class of the domain ontology."""

    name: str
    label: str = ""
    description: str = ""
    parent: str = ""
    attributes: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    sensitivity: str = Field(
        default="internal",
        description="Feeds IBAC: public | internal | confidential | restricted.",
    )

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity type name must not be empty")
        return value.strip()


class RelationDef(BaseModel):
    """A named relation between two entity types."""

    name: str
    source: str
    target: str
    description: str = ""
    cardinality: str = "many_to_many"


class Ontology(BaseModel):
    """A domain's vocabulary, relations and sensitivity classification."""

    domain: str
    label: str = ""
    description: str = ""
    version: str = "1"
    entities: list[EntityType] = Field(default_factory=list)
    relations: list[RelationDef] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ontology:
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Ontology:
        p = Path(path)
        if not p.exists():
            raise OntologyError(f"ontology file not found: {p}", path=str(p))
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise OntologyError(f"invalid YAML in {p}: {exc}", path=str(p)) from exc
        if not isinstance(data, dict):
            raise OntologyError(f"ontology file must contain a mapping: {p}", path=str(p))
        return cls.from_dict(data)

    # -- consistency -------------------------------------------------------

    def entity(self, name: str) -> EntityType | None:
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def entity_names(self) -> list[str]:
        return [e.name for e in self.entities]

    def validate_consistency(self) -> list[str]:
        """Return human-readable problems; an empty list means consistent."""
        problems: list[str] = []
        names = set(self.entity_names())

        seen: set[str] = set()
        for entity in self.entities:
            if entity.name in seen:
                problems.append(f"duplicate entity type '{entity.name}'")
            seen.add(entity.name)
            if entity.parent and entity.parent not in names:
                problems.append(
                    f"entity '{entity.name}' declares unknown parent '{entity.parent}'"
                )
            if entity.parent == entity.name:
                problems.append(f"entity '{entity.name}' is its own parent")

        for relation in self.relations:
            if relation.source not in names:
                problems.append(
                    f"relation '{relation.name}' has unknown source '{relation.source}'"
                )
            if relation.target not in names:
                problems.append(
                    f"relation '{relation.name}' has unknown target '{relation.target}'"
                )

        # A subclass cycle would make graph traversal non-terminating.
        for entity in self.entities:
            visited: set[str] = set()
            current = entity
            while current.parent:
                if current.parent in visited:
                    problems.append(
                        f"subclass cycle involving '{entity.name}' via '{current.parent}'"
                    )
                    break
                visited.add(current.parent)
                nxt = self.entity(current.parent)
                if nxt is None:
                    break
                current = nxt

        return problems

    # -- projection onto the graph ----------------------------------------

    def apply(self, graph: ContextGraph) -> dict[str, int]:
        """Project the ontology onto the Context Graph.

        Idempotent: applying the same ontology twice updates nodes in place
        rather than duplicating them.
        """
        problems = self.validate_consistency()
        if problems:
            raise OntologyError(
                f"ontology '{self.domain}' is inconsistent: {'; '.join(problems)}",
                domain=self.domain,
                problems=problems,
            )

        graph.register_domain(
            self.domain,
            label=self.label or self.domain,
            description=self.description,
            attributes={"ontology_version": self.version},
        )

        nodes = [
            Node(
                kind=NodeKind.ENTITY_TYPE,
                key=entity.name,
                label=entity.label or entity.name,
                description=entity.description,
                domain=self.domain,
                attributes={
                    "attributes": entity.attributes,
                    "keywords": entity.keywords,
                    "examples": entity.examples,
                    "sensitivity": entity.sensitivity,
                    "parent": entity.parent,
                },
            )
            for entity in self.entities
        ]
        graph.upsert_nodes(nodes)

        domain_node = graph.require_node(NodeKind.DOMAIN, self.domain)
        edge_count = 0
        for entity in self.entities:
            node = graph.require_node(NodeKind.ENTITY_TYPE, entity.name)
            graph.upsert_edge(node.node_id, domain_node.node_id, Relation.BELONGS_TO_DOMAIN)
            edge_count += 1
            if entity.parent:
                parent = graph.require_node(NodeKind.ENTITY_TYPE, entity.parent)
                graph.upsert_edge(node.node_id, parent.node_id, Relation.SUBCLASS_OF)
                edge_count += 1

        for relation in self.relations:
            source = graph.require_node(NodeKind.ENTITY_TYPE, relation.source)
            target = graph.require_node(NodeKind.ENTITY_TYPE, relation.target)
            graph.upsert_edge(
                source.node_id,
                target.node_id,
                Relation.RELATED_TO,
                attributes={
                    "name": relation.name,
                    "description": relation.description,
                    "cardinality": relation.cardinality,
                },
            )
            edge_count += 1

        return {"entities": len(self.entities), "edges": edge_count}

    def sensitivity_of(self, entity_name: str) -> str:
        entity = self.entity(entity_name)
        return entity.sensitivity if entity else "internal"

    def vocabulary(self) -> list[str]:
        """Flat vocabulary used to bias lexical matching and prompt priming."""
        terms: list[str] = []
        for entity in self.entities:
            terms.append(entity.name)
            terms.extend(entity.keywords)
            terms.extend(entity.attributes)
        for relation in self.relations:
            terms.append(relation.name)
        return sorted({t.lower() for t in terms if t})
