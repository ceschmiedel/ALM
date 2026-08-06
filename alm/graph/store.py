"""The Context Graph — L2's routing substrate and the federation's memory.

Two things the architecture depends on live here:

**Declarative routing.** Capabilities are nodes with edges to the ontology
entity types they operate on. The router queries this graph; it does not
consult a table of ``if`` statements. Registering a new Expert Agent therefore
adds nodes and edges, and the routing behaviour changes without a code change.
That property is what makes the routing layer an evolving asset rather than a
tangle of conditionals.

**Explicit authority.** ``has_authority_over`` edges carry the weight that
arbitration uses when two domains overlap. Because the weights are graph data,
they are inspectable, auditable and tunable per tenant.

Node placement in semantic space is optional: with no embedder configured the
matcher degrades to lexical scoring, which is weaker but never wrong-by-
crashing. That degradation is reported in the trace so it never passes silently.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from sqlalchemy import select

from alm.core.errors import GraphError
from alm.core.ids import new_id
from alm.graph.models import (
    CapabilityDeclaration,
    Edge,
    GraphStats,
    Node,
    NodeKind,
    Relation,
)
from alm.persistence.database import session_scope
from alm.persistence.models import GraphEdgeRow, GraphNodeRow

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, safe on zero vectors and mismatched lengths."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _row_to_node(row: GraphNodeRow) -> Node:
    return Node(
        node_id=row.node_id,
        tenant_id=row.tenant_id,
        kind=NodeKind(row.kind),
        key=row.key,
        label=row.label,
        description=row.description,
        domain=row.domain,
        attributes=dict(row.attributes or {}),
        embedding=list(row.embedding) if row.embedding else None,
        created_at=row.created_at.isoformat() if row.created_at else "",
        updated_at=row.updated_at.isoformat() if row.updated_at else "",
    )


def _row_to_edge(row: GraphEdgeRow) -> Edge:
    return Edge(
        edge_id=row.edge_id,
        tenant_id=row.tenant_id,
        source_id=row.source_id,
        target_id=row.target_id,
        relation=row.relation,
        weight=row.weight,
        attributes=dict(row.attributes or {}),
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


class ContextGraph:
    """Persistent, tenant-scoped property graph.

    Parameters
    ----------
    tenant_id:
        All reads and writes are confined to this tenant.  Two tenants can hold
        the same domain with different authority weights and different experts.
    embedder:
        Optional callable mapping texts to vectors.  When present, nodes are
        embedded on write and semantic search becomes available.
    """

    def __init__(self, tenant_id: str = "default", embedder: EmbedFn | None = None) -> None:
        self.tenant_id = tenant_id
        self._embedder = embedder

    # -- embedder ----------------------------------------------------------

    @property
    def has_embedder(self) -> bool:
        return self._embedder is not None

    def set_embedder(self, embedder: EmbedFn | None) -> None:
        self._embedder = embedder

    def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if self._embedder is None or not texts:
            return None
        try:
            return self._embedder(texts)
        except Exception:
            logger.warning("Embedding failed; falling back to lexical matching", exc_info=True)
            return None

    # -- nodes -------------------------------------------------------------

    def upsert_node(self, node: Node, *, embed: bool = True) -> Node:
        """Insert or update a node, keyed by ``(tenant, kind, key)``."""
        node.tenant_id = self.tenant_id
        embedding = node.embedding
        if embed and embedding is None:
            vectors = self._embed([node.text_for_embedding()])
            embedding = vectors[0] if vectors else None

        with session_scope() as session:
            row = session.execute(
                select(GraphNodeRow).where(
                    GraphNodeRow.tenant_id == self.tenant_id,
                    GraphNodeRow.kind == str(node.kind),
                    GraphNodeRow.key == node.key,
                )
            ).scalar_one_or_none()

            if row is None:
                row = GraphNodeRow(
                    node_id=node.node_id or new_id("nod"),
                    tenant_id=self.tenant_id,
                    kind=str(node.kind),
                    key=node.key,
                )
                session.add(row)

            row.label = node.label or node.key
            row.description = node.description
            row.domain = node.domain
            row.attributes = node.attributes
            if embedding is not None:
                row.embedding = embedding
            session.flush()
            return _row_to_node(row)

    def upsert_nodes(self, nodes: Iterable[Node]) -> list[Node]:
        """Batch upsert; embeds all texts in one call when an embedder is set."""
        items = list(nodes)
        if not items:
            return []
        vectors = self._embed([n.text_for_embedding() for n in items])
        if vectors:
            for node, vector in zip(items, vectors, strict=False):
                if node.embedding is None:
                    node.embedding = vector
        return [self.upsert_node(n, embed=False) for n in items]

    def get_node(self, kind: NodeKind | str, key: str) -> Node | None:
        with session_scope() as session:
            row = session.execute(
                select(GraphNodeRow).where(
                    GraphNodeRow.tenant_id == self.tenant_id,
                    GraphNodeRow.kind == str(kind),
                    GraphNodeRow.key == key,
                )
            ).scalar_one_or_none()
            return _row_to_node(row) if row else None

    def get_node_by_id(self, node_id: str) -> Node | None:
        with session_scope() as session:
            row = session.get(GraphNodeRow, node_id)
            return _row_to_node(row) if row else None

    def require_node(self, kind: NodeKind | str, key: str) -> Node:
        node = self.get_node(kind, key)
        if node is None:
            raise GraphError(f"node not found: {kind}:{key}", kind=str(kind), key=key)
        return node

    def find_nodes(
        self,
        kind: NodeKind | str | None = None,
        *,
        domain: str | None = None,
    ) -> list[Node]:
        with session_scope() as session:
            stmt = select(GraphNodeRow).where(GraphNodeRow.tenant_id == self.tenant_id)
            if kind is not None:
                stmt = stmt.where(GraphNodeRow.kind == str(kind))
            if domain is not None:
                stmt = stmt.where(GraphNodeRow.domain == domain)
            rows = session.execute(stmt.order_by(GraphNodeRow.key)).scalars().all()
            return [_row_to_node(r) for r in rows]

    def delete_node(self, kind: NodeKind | str, key: str) -> bool:
        """Delete a node and every edge touching it."""
        with session_scope() as session:
            row = session.execute(
                select(GraphNodeRow).where(
                    GraphNodeRow.tenant_id == self.tenant_id,
                    GraphNodeRow.kind == str(kind),
                    GraphNodeRow.key == key,
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            edges = session.execute(
                select(GraphEdgeRow).where(
                    (GraphEdgeRow.source_id == row.node_id)
                    | (GraphEdgeRow.target_id == row.node_id)
                )
            ).scalars().all()
            for edge in edges:
                session.delete(edge)
            session.delete(row)
            return True

    # -- edges -------------------------------------------------------------

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: Relation | str,
        *,
        weight: float = 1.0,
        attributes: dict[str, Any] | None = None,
    ) -> Edge:
        with session_scope() as session:
            row = session.execute(
                select(GraphEdgeRow).where(
                    GraphEdgeRow.tenant_id == self.tenant_id,
                    GraphEdgeRow.source_id == source_id,
                    GraphEdgeRow.target_id == target_id,
                    GraphEdgeRow.relation == str(relation),
                )
            ).scalar_one_or_none()
            if row is None:
                row = GraphEdgeRow(
                    edge_id=new_id("edg"),
                    tenant_id=self.tenant_id,
                    source_id=source_id,
                    target_id=target_id,
                    relation=str(relation),
                )
                session.add(row)
            row.weight = weight
            row.attributes = attributes or {}
            session.flush()
            return _row_to_edge(row)

    def edges_from(self, node_id: str, relation: Relation | str | None = None) -> list[Edge]:
        with session_scope() as session:
            stmt = select(GraphEdgeRow).where(
                GraphEdgeRow.tenant_id == self.tenant_id,
                GraphEdgeRow.source_id == node_id,
            )
            if relation is not None:
                stmt = stmt.where(GraphEdgeRow.relation == str(relation))
            return [_row_to_edge(r) for r in session.execute(stmt).scalars().all()]

    def edges_to(self, node_id: str, relation: Relation | str | None = None) -> list[Edge]:
        with session_scope() as session:
            stmt = select(GraphEdgeRow).where(
                GraphEdgeRow.tenant_id == self.tenant_id,
                GraphEdgeRow.target_id == node_id,
            )
            if relation is not None:
                stmt = stmt.where(GraphEdgeRow.relation == str(relation))
            return [_row_to_edge(r) for r in session.execute(stmt).scalars().all()]

    def neighbors(
        self,
        node_id: str,
        relation: Relation | str | None = None,
        *,
        direction: str = "out",
    ) -> list[Node]:
        """Return adjacent nodes in the requested direction (``out``/``in``/``both``)."""
        edges: list[Edge] = []
        if direction in {"out", "both"}:
            edges.extend(self.edges_from(node_id, relation))
        if direction in {"in", "both"}:
            edges.extend(self.edges_to(node_id, relation))
        ids = [e.target_id if e.source_id == node_id else e.source_id for e in edges]
        out: list[Node] = []
        for nid in dict.fromkeys(ids):
            node = self.get_node_by_id(nid)
            if node is not None:
                out.append(node)
        return out

    # -- domain-level helpers ---------------------------------------------

    def domains(self) -> list[str]:
        return [n.key for n in self.find_nodes(NodeKind.DOMAIN)]

    def entity_types(self, domain: str | None = None) -> list[Node]:
        return self.find_nodes(NodeKind.ENTITY_TYPE, domain=domain)

    def experts(self, domain: str | None = None) -> list[Node]:
        return self.find_nodes(NodeKind.EXPERT, domain=domain)

    def capability_nodes(self, domain: str | None = None) -> list[Node]:
        return self.find_nodes(NodeKind.CAPABILITY, domain=domain)

    def capabilities(self, domain: str | None = None) -> list[CapabilityDeclaration]:
        """Return every declared capability as a typed value object."""
        out: list[CapabilityDeclaration] = []
        for node in self.capability_nodes(domain):
            attrs = node.attributes
            out.append(
                CapabilityDeclaration(
                    capability_id=node.key,
                    expert_id=str(attrs.get("expert_id", "")),
                    domain=node.domain,
                    description=node.description,
                    keywords=list(attrs.get("keywords", [])),
                    examples=list(attrs.get("examples", [])),
                    operates_on=list(attrs.get("operates_on", [])),
                    produces=list(attrs.get("produces", [])),
                    required_scopes=list(attrs.get("required_scopes", [])),
                    output_schema=dict(attrs.get("output_schema", {})),
                    estimated_cost=float(attrs.get("estimated_cost", 0.0)),
                    estimated_latency=float(attrs.get("estimated_latency", 0.0)),
                    tier_hint=str(attrs.get("tier_hint", "")),
                )
            )
        return out

    def capability_embedding(self, capability_id: str) -> list[float] | None:
        node = self.get_node(NodeKind.CAPABILITY, capability_id)
        return node.embedding if node else None

    def authority_weight(self, expert_id: str, domain: str) -> float:
        """Weight of ``expert_id`` on questions primarily about ``domain``.

        Defaults to 1.0 for the expert's own domain and 0.5 elsewhere, so a
        federation that never configures authority still arbitrates sensibly.
        """
        expert = self.get_node(NodeKind.EXPERT, expert_id)
        if expert is None:
            return 0.0
        domain_node = self.get_node(NodeKind.DOMAIN, domain)
        if domain_node is not None:
            for edge in self.edges_from(expert.node_id, Relation.HAS_AUTHORITY_OVER):
                if edge.target_id == domain_node.node_id:
                    return edge.weight
        return 1.0 if expert.domain == domain else 0.5

    def set_authority(self, expert_id: str, domain: str, weight: float) -> Edge:
        expert = self.require_node(NodeKind.EXPERT, expert_id)
        domain_node = self.require_node(NodeKind.DOMAIN, domain)
        return self.upsert_edge(
            expert.node_id, domain_node.node_id, Relation.HAS_AUTHORITY_OVER, weight=weight
        )

    # -- registration ------------------------------------------------------

    def register_domain(
        self,
        key: str,
        *,
        label: str = "",
        description: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        return self.upsert_node(
            Node(
                kind=NodeKind.DOMAIN,
                key=key,
                label=label or key,
                description=description,
                domain=key,
                attributes=attributes or {},
            )
        )

    def register_expert(
        self,
        expert_id: str,
        *,
        domain: str,
        description: str = "",
        model_id: str = "",
        capabilities: Sequence[CapabilityDeclaration] = (),
        authority: dict[str, float] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        """Declare an Expert Agent and everything the router needs to reach it.

        Creates the expert node, one capability node per declaration, the edges
        binding capabilities to the entity types they operate on, and the
        authority weights consulted during arbitration.
        """
        domain_node = self.get_node(NodeKind.DOMAIN, domain) or self.register_domain(domain)

        expert_node = self.upsert_node(
            Node(
                kind=NodeKind.EXPERT,
                key=expert_id,
                label=expert_id,
                description=description,
                domain=domain,
                attributes={**(attributes or {}), "model_id": model_id},
            )
        )
        self.upsert_edge(expert_node.node_id, domain_node.node_id, Relation.BELONGS_TO_DOMAIN)

        if model_id:
            model_node = self.upsert_node(
                Node(kind=NodeKind.MODEL, key=model_id, label=model_id, domain=domain)
            )
            self.upsert_edge(expert_node.node_id, model_node.node_id, Relation.BACKED_BY_MODEL)

        for cap in capabilities:
            cap_node = self.upsert_node(
                Node(
                    kind=NodeKind.CAPABILITY,
                    key=cap.capability_id,
                    label=cap.capability_id,
                    description=cap.description,
                    domain=cap.domain or domain,
                    attributes={
                        "expert_id": expert_id,
                        "keywords": cap.keywords,
                        "examples": cap.examples,
                        "operates_on": cap.operates_on,
                        "produces": cap.produces,
                        "required_scopes": cap.required_scopes,
                        "output_schema": cap.output_schema,
                        "estimated_cost": cap.estimated_cost,
                        "estimated_latency": cap.estimated_latency,
                        "tier_hint": cap.tier_hint,
                    },
                )
            )
            self.upsert_edge(
                expert_node.node_id, cap_node.node_id, Relation.PROVIDES_CAPABILITY
            )
            self.upsert_edge(
                cap_node.node_id, domain_node.node_id, Relation.BELONGS_TO_DOMAIN
            )
            for entity_type in cap.operates_on:
                entity_node = self.get_node(NodeKind.ENTITY_TYPE, entity_type)
                if entity_node is None:
                    entity_node = self.upsert_node(
                        Node(
                            kind=NodeKind.ENTITY_TYPE,
                            key=entity_type,
                            label=entity_type,
                            domain=cap.domain or domain,
                        )
                    )
                self.upsert_edge(cap_node.node_id, entity_node.node_id, Relation.OPERATES_ON)

        for target_domain, weight in (authority or {}).items():
            target_node = self.get_node(NodeKind.DOMAIN, target_domain) or self.register_domain(
                target_domain
            )
            self.upsert_edge(
                expert_node.node_id,
                target_node.node_id,
                Relation.HAS_AUTHORITY_OVER,
                weight=weight,
            )

        return expert_node

    # -- semantic search ---------------------------------------------------

    def semantic_search(
        self,
        text: str,
        *,
        kinds: Sequence[NodeKind | str] | None = None,
        domain: str | None = None,
        top_k: int = 10,
    ) -> list[tuple[Node, float]]:
        """Rank nodes by cosine similarity to ``text``.

        Returns an empty list when no embedder is configured — callers must
        treat that as "semantic signal unavailable", not as "no match".
        """
        vectors = self._embed([text])
        if not vectors:
            return []
        query = vectors[0]

        candidates: list[Node] = []
        if kinds:
            for kind in kinds:
                candidates.extend(self.find_nodes(kind, domain=domain))
        else:
            candidates = self.find_nodes(domain=domain)

        scored = [
            (node, cosine(query, node.embedding))
            for node in candidates
            if node.embedding
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def reembed_all(self) -> int:
        """Recompute embeddings for every node.  Returns the number updated."""
        if self._embedder is None:
            return 0
        nodes = self.find_nodes()
        if not nodes:
            return 0
        vectors = self._embed([n.text_for_embedding() for n in nodes])
        if not vectors:
            return 0
        for node, vector in zip(nodes, vectors, strict=False):
            node.embedding = vector
            self.upsert_node(node, embed=False)
        return len(nodes)

    # -- introspection -----------------------------------------------------

    def stats(self) -> GraphStats:
        nodes = self.find_nodes()
        by_kind: dict[str, int] = {}
        for node in nodes:
            by_kind[str(node.kind)] = by_kind.get(str(node.kind), 0) + 1
        with session_scope() as session:
            edge_count = len(
                session.execute(
                    select(GraphEdgeRow.edge_id).where(
                        GraphEdgeRow.tenant_id == self.tenant_id
                    )
                ).scalars().all()
            )
        return GraphStats(
            nodes=len(nodes),
            edges=edge_count,
            by_kind=by_kind,
            domains=[n.key for n in nodes if n.kind == NodeKind.DOMAIN],
            experts=[n.key for n in nodes if n.kind == NodeKind.EXPERT],
        )

    def clear(self) -> None:
        """Remove every node and edge for this tenant."""
        with session_scope() as session:
            for row in session.execute(
                select(GraphEdgeRow).where(GraphEdgeRow.tenant_id == self.tenant_id)
            ).scalars().all():
                session.delete(row)
            for row in session.execute(
                select(GraphNodeRow).where(GraphNodeRow.tenant_id == self.tenant_id)
            ).scalars().all():
                session.delete(row)
