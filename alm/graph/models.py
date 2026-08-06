"""Context Graph value objects.

The graph is the federation's cortex: it holds the ontology, the capability
declarations of every Expert Agent, the authority weights arbitration uses,
and the intermediate state of a running plan.  Because routing reads these as
*data*, adding an expert is adding a node — the router is never rewritten.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from alm.core.ids import utc_now_iso


class NodeKind(StrEnum):
    """The node types the runtime understands."""

    DOMAIN = "domain"
    ENTITY_TYPE = "entity_type"
    EXPERT = "expert"
    CAPABILITY = "capability"
    MODEL = "model"
    DOCUMENT = "document"
    CONCEPT = "concept"
    POLICY = "policy"


class Relation(StrEnum):
    """Typed edges between nodes."""

    BELONGS_TO_DOMAIN = "belongs_to_domain"
    PROVIDES_CAPABILITY = "provides_capability"
    OPERATES_ON = "operates_on"
    HAS_AUTHORITY_OVER = "has_authority_over"
    BACKED_BY_MODEL = "backed_by_model"
    SUBCLASS_OF = "subclass_of"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    GOVERNED_BY = "governed_by"
    PRODUCES = "produces"


class Node(BaseModel):
    """A Context Graph node."""

    node_id: str = ""
    tenant_id: str = "default"
    kind: NodeKind
    key: str
    label: str = ""
    description: str = ""
    domain: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def text_for_embedding(self) -> str:
        """The text used to place this node in semantic space."""
        parts = [self.label or self.key, self.description]
        keywords = self.attributes.get("keywords")
        if isinstance(keywords, list):
            parts.extend(str(k) for k in keywords)
        examples = self.attributes.get("examples")
        if isinstance(examples, list):
            parts.extend(str(e) for e in examples)
        return " · ".join(p for p in parts if p)


class Edge(BaseModel):
    """A Context Graph edge."""

    edge_id: str = ""
    tenant_id: str = "default"
    source_id: str
    target_id: str
    relation: Relation | str
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class CapabilityDeclaration(BaseModel):
    """What an Expert Agent declares it can do, and over which entities.

    The router matches a subtask against these declarations instead of against
    hard-coded rules.  ``examples`` matter more than they look: they are the
    strongest signal available to the semantic matcher when the wording of a
    request does not reuse the vocabulary of the description.
    """

    capability_id: str
    expert_id: str = ""
    domain: str = ""
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    operates_on: list[str] = Field(
        default_factory=list, description="Ontology entity types this capability reads or writes."
    )
    produces: list[str] = Field(default_factory=list)
    required_scopes: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    tier_hint: str = ""

    def text_for_embedding(self) -> str:
        parts = [self.capability_id.replace("_", " "), self.description]
        parts.extend(self.keywords)
        parts.extend(self.examples)
        parts.extend(self.operates_on)
        return " · ".join(p for p in parts if p)


class CapabilityMatch(BaseModel):
    """A scored candidate produced by the capability matcher."""

    capability_id: str
    expert_id: str
    domain: str
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    entity_score: float = 0.0
    authority: float = 1.0
    reason: str = ""


class GraphStats(BaseModel):
    """Counts used by the CLI and the API health endpoint."""

    nodes: int = 0
    edges: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)
    experts: list[str] = Field(default_factory=list)
