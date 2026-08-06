"""CX · Context Graph, ontology projection and the session blackboard."""

from __future__ import annotations

import pytest

from alm.core.errors import GraphError, OntologyError
from alm.graph.blackboard import SessionBlackboard
from alm.graph.models import CapabilityDeclaration, Node, NodeKind, Relation
from alm.graph.ontology import Ontology
from alm.graph.store import cosine


def _ontology() -> Ontology:
    return Ontology(
        domain="legal",
        label="Legal",
        entities=[
            {"name": "Contract", "keywords": ["agreement", "msa"], "sensitivity": "confidential"},
            {"name": "Clause", "parent": "Contract", "keywords": ["provision"]},
        ],
        relations=[{"name": "contains", "source": "Contract", "target": "Clause"}],
    )


def test_upsert_is_idempotent_on_identity(graph):
    graph.upsert_node(Node(kind=NodeKind.DOMAIN, key="legal", label="Legal"))
    graph.upsert_node(Node(kind=NodeKind.DOMAIN, key="legal", label="Legal & Contracts"))

    nodes = graph.find_nodes(NodeKind.DOMAIN)
    assert len(nodes) == 1
    assert nodes[0].label == "Legal & Contracts"


def test_require_node_raises_for_unknown(graph):
    with pytest.raises(GraphError):
        graph.require_node(NodeKind.DOMAIN, "nope")


def test_ontology_projects_entities_and_edges(graph):
    report = _ontology().apply(graph)
    assert report["entities"] == 2

    clause = graph.require_node(NodeKind.ENTITY_TYPE, "Clause")
    parents = graph.neighbors(clause.node_id, Relation.SUBCLASS_OF)
    assert [p.key for p in parents] == ["Contract"]
    assert graph.get_node(NodeKind.ENTITY_TYPE, "Contract").attributes["sensitivity"] == (
        "confidential"
    )


def test_ontology_rejects_unknown_parent(graph):
    ontology = Ontology(
        domain="legal", entities=[{"name": "Clause", "parent": "Ghost"}]
    )
    with pytest.raises(OntologyError, match="unknown parent"):
        ontology.apply(graph)


def test_ontology_rejects_subclass_cycles():
    ontology = Ontology(
        domain="legal",
        entities=[
            {"name": "A", "parent": "B"},
            {"name": "B", "parent": "A"},
        ],
    )
    problems = ontology.validate_consistency()
    assert any("cycle" in p for p in problems)


def test_ontology_rejects_relation_to_unknown_entity():
    ontology = Ontology(
        domain="legal",
        entities=[{"name": "Contract"}],
        relations=[{"name": "x", "source": "Contract", "target": "Ghost"}],
    )
    problems = ontology.validate_consistency()
    assert any("unknown target" in p for p in problems)


def test_register_expert_wires_capabilities_and_authority(graph):
    _ontology().apply(graph)
    graph.register_expert(
        "contracts-expert",
        domain="legal",
        model_id="legal-slm",
        capabilities=[
            CapabilityDeclaration(
                capability_id="analyze_clause",
                domain="legal",
                description="Analyse a clause",
                operates_on=["Clause"],
                produces=["ClauseAnalysis"],
            )
        ],
        authority={"legal": 1.0, "finance": 0.35},
    )

    capabilities = graph.capabilities("legal")
    assert [c.capability_id for c in capabilities] == ["analyze_clause"]
    assert capabilities[0].expert_id == "contracts-expert"

    assert graph.authority_weight("contracts-expert", "legal") == 1.0
    assert graph.authority_weight("contracts-expert", "finance") == 0.35
    # Unknown expert has no voice at all.
    assert graph.authority_weight("ghost", "legal") == 0.0


def test_authority_defaults_without_explicit_edges(graph):
    graph.register_domain("legal")
    graph.register_domain("finance")
    graph.register_expert("solo", domain="legal", authority={})
    assert graph.authority_weight("solo", "legal") == 1.0
    assert graph.authority_weight("solo", "finance") == 0.5


def test_adding_an_expert_changes_routing_without_code(graph):
    """The declarative-routing property the architecture depends on."""
    _ontology().apply(graph)
    assert graph.capabilities("legal") == []

    graph.register_expert(
        "new-expert",
        domain="legal",
        capabilities=[
            CapabilityDeclaration(capability_id="new_cap", domain="legal", description="d")
        ],
    )
    assert [c.capability_id for c in graph.capabilities("legal")] == ["new_cap"]


def test_semantic_search_ranks_by_similarity(graph):
    graph.upsert_nodes(
        [
            Node(
                kind=NodeKind.CAPABILITY,
                key="analyze_clause",
                description="analyse contractual clauses and liability caps",
                domain="legal",
            ),
            Node(
                kind=NodeKind.CAPABILITY,
                key="assess_risk",
                description="assess operational and regulatory risk severity",
                domain="risk",
            ),
        ]
    )
    ranked = graph.semantic_search(
        "liability cap in a contractual clause", kinds=[NodeKind.CAPABILITY], top_k=2
    )
    assert ranked, "an embedder is configured, so semantic search must return results"
    assert ranked[0][0].key == "analyze_clause"


def test_semantic_search_is_empty_without_embedder(graph):
    graph.set_embedder(None)
    assert graph.semantic_search("anything") == []


def test_delete_node_removes_touching_edges(graph):
    _ontology().apply(graph)
    contract = graph.require_node(NodeKind.ENTITY_TYPE, "Contract")
    assert graph.delete_node(NodeKind.ENTITY_TYPE, "Contract") is True
    assert graph.get_node_by_id(contract.node_id) is None
    assert graph.edges_from(contract.node_id) == []


def test_tenants_are_isolated(graph):
    from alm.graph.store import ContextGraph

    graph.register_domain("legal")
    other = ContextGraph("other-tenant")
    assert other.domains() == []
    other.register_domain("finance")
    assert graph.domains() == ["legal"]


def test_blackboard_round_trips_with_provenance():
    board = SessionBlackboard("ses-1")
    board.put("answer:1", {"content": "hello"}, produced_by="legal", subtask_id="s1")
    board.put("scalar", 42)

    assert board.get("answer:1")["content"] == "hello"
    assert board.get("scalar") == 42
    assert board.get("missing", "default") == "default"

    snapshot = board.snapshot()
    assert snapshot["answer:1"]["produced_by"] == "legal"
    assert board.clear() == 2
    assert board.keys() == []


def test_cosine_handles_degenerate_vectors():
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
