"""SQLAlchemy models backing the Context Graph, CMRAG, MLOps and audit stores.

Everything the federation needs to be *reconstructible after the fact* lives
here.  A session's plan, the answers each expert gave, how arbitration decided,
and which governance rules were evaluated are all persisted — not because the
runtime needs them to answer, but because in a regulated vertical an answer
that cannot be re-examined is not adoptable.

Storage is deliberately portable: SQLite for a laptop, PostgreSQL for
production, with embeddings kept as JSON arrays so no vector extension is
required to get started (see :mod:`alm.cmrag.store` for the pgvector path).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase

from alm.core.ids import new_id, utc_now


class Base(DeclarativeBase):
    """Declarative base for every ALM table."""


def _now() -> datetime:
    return utc_now()


# ---------------------------------------------------------------------------
# Context Graph
# ---------------------------------------------------------------------------


class GraphNodeRow(Base):
    """A node of the Context Graph.

    One table holds every node kind (``domain``, ``entity_type``, ``expert``,
    ``capability``, ``document``, ``concept``…).  Keeping capabilities as graph
    nodes rather than as code is what lets a new Expert Agent join the
    federation by *adding a node*, without touching the router.
    """

    __tablename__ = "graph_nodes"

    node_id = Column(String(64), primary_key=True, default=lambda: new_id("nod"))
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    kind = Column(String(64), nullable=False, index=True)
    key = Column(String(256), nullable=False)
    label = Column(String(512), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    domain = Column(String(128), nullable=False, default="", index=True)
    attributes = Column(JSON, nullable=False, default=dict)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("tenant_id", "kind", "key", name="uq_graph_node_identity"),
        Index("ix_graph_nodes_kind_domain", "kind", "domain"),
    )


class GraphEdgeRow(Base):
    """A typed, weighted relation between two Context Graph nodes.

    ``weight`` carries domain-authority semantics for ``has_authority_over``
    edges: on the boundary between two domains, arbitration favours the expert
    whose domain is primary for the question, and that primacy is data in the
    graph rather than a branch in code.
    """

    __tablename__ = "graph_edges"

    edge_id = Column(String(64), primary_key=True, default=lambda: new_id("edg"))
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    source_id = Column(String(64), ForeignKey("graph_nodes.node_id"), nullable=False, index=True)
    target_id = Column(String(64), ForeignKey("graph_nodes.node_id"), nullable=False, index=True)
    relation = Column(String(128), nullable=False, index=True)
    weight = Column(Float, nullable=False, default=1.0)
    attributes = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "source_id", "target_id", "relation", name="uq_graph_edge_identity"
        ),
    )


class BlackboardRow(Base):
    """Intermediate execution state for a running plan.

    During a DAG run the shared state lives in the graph, not hidden in a
    process's memory: any step in the plan can read what has already been
    produced, and arbitration sees the whole chain of reasoning.
    """

    __tablename__ = "blackboard"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    tenant_id = Column(String(128), nullable=False, default="default")
    key = Column(String(256), nullable=False)
    value = Column(JSON, nullable=False, default=dict)
    produced_by = Column(String(128), nullable=False, default="")
    subtask_id = Column(String(64), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (
        UniqueConstraint("session_id", "key", name="uq_blackboard_key"),
    )


# ---------------------------------------------------------------------------
# CMRAG corpus
# ---------------------------------------------------------------------------


class DocumentRow(Base):
    """A source document owned by exactly one domain."""

    __tablename__ = "documents"

    document_id = Column(String(64), primary_key=True, default=lambda: new_id("doc"))
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    domain = Column(String(128), nullable=False, default="", index=True)
    title = Column(String(512), nullable=False, default="")
    uri = Column(String(1024), nullable=False, default="")
    content_hash = Column(String(64), nullable=False, default="", index=True)
    doc_metadata = Column("metadata", JSON, nullable=False, default=dict)
    sensitivity = Column(String(64), nullable=False, default="internal")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class ChunkRow(Base):
    """A retrievable fragment, annotated with the ontology entities it mentions.

    The entity annotation is what makes retrieval *ontology-aware* rather than
    merely semantic: an expert asking about a ``Clause`` gets fragments the
    graph associates with clauses, not everything that happens to embed nearby.
    """

    __tablename__ = "chunks"

    chunk_id = Column(String(64), primary_key=True, default=lambda: new_id("chk"))
    document_id = Column(String(64), ForeignKey("documents.document_id"), nullable=False, index=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    domain = Column(String(128), nullable=False, default="", index=True)
    ordinal = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)
    entities = Column(JSON, nullable=False, default=list)
    chunk_metadata = Column("metadata", JSON, nullable=False, default=dict)
    sensitivity = Column(String(64), nullable=False, default="internal")
    token_count = Column(Integer, nullable=False, default=0)
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(256), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    __table_args__ = (Index("ix_chunks_domain_tenant", "domain", "tenant_id"),)


# ---------------------------------------------------------------------------
# Model registry and MLOps
# ---------------------------------------------------------------------------


class ModelRow(Base):
    """A model the federation can serve.

    ``base_model`` + ``adapter`` is the economically important case: dozens of
    experts share one loaded base and bring only their own LoRA adapter, so GPU
    cost stops scaling with the number of agents.
    """

    __tablename__ = "models"

    model_id = Column(String(128), primary_key=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    tier = Column(String(32), nullable=False, default="slm", index=True)
    backend = Column(String(64), nullable=False, default="mock")
    model_name = Column(String(256), nullable=False, default="")
    base_model = Column(String(256), nullable=False, default="")
    adapter = Column(String(256), nullable=False, default="")
    adapter_uri = Column(String(1024), nullable=False, default="")
    endpoint = Column(String(1024), nullable=False, default="")
    params = Column(JSON, nullable=False, default=dict)
    context_window = Column(Integer, nullable=False, default=8192)
    cost_per_1k_input = Column(Float, nullable=False, default=0.0)
    cost_per_1k_output = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="active")
    active_version = Column(String(64), nullable=False, default="1")
    description = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class ModelVersionRow(Base):
    """An immutable version of a model or adapter, with its evaluation record.

    Promotion is gated on evaluation; rollback is just re-activating an earlier
    row.  A new version of one expert must never silently degrade the others.
    """

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(128), ForeignKey("models.model_id"), nullable=False, index=True)
    version = Column(String(64), nullable=False)
    artifact_uri = Column(String(1024), nullable=False, default="")
    status = Column(String(32), nullable=False, default="candidate")
    metrics = Column(JSON, nullable=False, default=dict)
    eval_run_id = Column(String(64), nullable=False, default="")
    notes = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_model_version"),
    )


class CalibrationRow(Base):
    """Fitted confidence calibration for one expert.

    Arbitration by confidence is only meaningful once the numbers are
    calibrated, so the parameters fitted from evaluation runs are persisted
    per expert and applied to every raw score at answer time.
    """

    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    expert_id = Column(String(128), nullable=False, index=True)
    tenant_id = Column(String(128), nullable=False, default="default")
    temperature = Column(Float, nullable=False, default=1.0)
    bias = Column(Float, nullable=False, default=0.0)
    sample_count = Column(Integer, nullable=False, default=0)
    # Explicit, not derived from sample_count: a fit that was attempted and
    # *rejected* still has samples, and must not be mistaken for a validated one.
    fitted = Column(Boolean, nullable=False, default=False)
    brier_before = Column(Float, nullable=False, default=0.0)
    brier_after = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("tenant_id", "expert_id", name="uq_calibration_expert"),
    )


class MetricSnapshotRow(Base):
    """A periodic health reading for one expert, used for drift detection."""

    __tablename__ = "metric_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(128), nullable=False, default="default")
    expert_id = Column(String(128), nullable=False, index=True)
    window_start = Column(DateTime(timezone=True), nullable=False, default=_now)
    window_end = Column(DateTime(timezone=True), nullable=False, default=_now)
    accuracy = Column(Float, nullable=False, default=0.0)
    fallback_rate = Column(Float, nullable=False, default=0.0)
    p95_latency_ms = Column(Float, nullable=False, default=0.0)
    mean_confidence = Column(Float, nullable=False, default=0.0)
    sample_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Sessions, governance and feedback
# ---------------------------------------------------------------------------


class SessionRow(Base):
    """A persisted federation run, complete enough to be replayed and audited."""

    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    principal = Column(String(256), nullable=False, default="")
    intent_text = Column(Text, nullable=False, default="")
    status = Column(String(32), nullable=False, default="running", index=True)
    strategy = Column(String(64), nullable=False, default="")
    router_confidence = Column(Float, nullable=False, default=0.0)
    used_fallback = Column(Boolean, nullable=False, default=False)
    answer = Column(Text, nullable=False, default="")
    plan = Column(JSON, nullable=False, default=dict)
    answers = Column(JSON, nullable=False, default=list)
    arbitration = Column(JSON, nullable=False, default=dict)
    trace = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    error = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class AuditRow(Base):
    """An append-only IBAC decision record."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    session_id = Column(String(64), nullable=False, default="", index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)
    evaluation_point = Column(String(64), nullable=False, default="")
    principal = Column(String(256), nullable=False, default="")
    expert_id = Column(String(128), nullable=False, default="", index=True)
    resource = Column(String(512), nullable=False, default="")
    action = Column(String(128), nullable=False, default="")
    decision = Column(String(16), nullable=False, default="allow", index=True)
    reason = Column(Text, nullable=False, default="")
    matched_policies = Column(JSON, nullable=False, default=list)
    detail = Column(JSON, nullable=False, default=dict)


class FeedbackRow(Base):
    """A production case worth learning from.

    Failures and escalations captured here are exactly the seed material for
    the next distillation cycle — the loop that makes the federation improve
    with use rather than decay.
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    session_id = Column(String(64), nullable=False, default="", index=True)
    expert_id = Column(String(128), nullable=False, default="", index=True)
    domain = Column(String(128), nullable=False, default="", index=True)
    kind = Column(String(32), nullable=False, default="failure", index=True)
    intent_text = Column(Text, nullable=False, default="")
    expected = Column(Text, nullable=False, default="")
    actual = Column(Text, nullable=False, default="")
    detail = Column(JSON, nullable=False, default=dict)
    consumed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvalRunRow(Base):
    """One arm of a controlled experiment (federation or monolithic baseline)."""

    __tablename__ = "eval_runs"

    run_id = Column(String(64), primary_key=True, default=lambda: new_id("run"))
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    name = Column(String(256), nullable=False, default="")
    arm = Column(String(32), nullable=False, default="federation", index=True)
    pack = Column(String(256), nullable=False, default="")
    dataset = Column(String(512), nullable=False, default="")
    domain = Column(String(128), nullable=False, default="", index=True)
    metrics = Column(JSON, nullable=False, default=dict)
    config = Column(JSON, nullable=False, default=dict)
    case_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)


class EvalCaseRow(Base):
    """The per-case record behind an :class:`EvalRunRow` aggregate."""

    __tablename__ = "eval_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("eval_runs.run_id"), nullable=False, index=True)
    case_id = Column(String(128), nullable=False, default="")
    domain = Column(String(128), nullable=False, default="")
    score = Column(Float, nullable=False, default=0.0)
    correct = Column(Boolean, nullable=False, default=False)
    latency_ms = Column(Float, nullable=False, default=0.0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    used_fallback = Column(Boolean, nullable=False, default=False)
    trace_complete = Column(Boolean, nullable=False, default=False)
    answer = Column(Text, nullable=False, default="")
    expected = Column(Text, nullable=False, default="")
    detail = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


class TrainingExampleRow(Base):
    """A distilled training example, before or after validation."""

    __tablename__ = "training_examples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset = Column(String(256), nullable=False, default="", index=True)
    tenant_id = Column(String(128), nullable=False, default="default")
    domain = Column(String(128), nullable=False, default="", index=True)
    expert_id = Column(String(128), nullable=False, default="", index=True)
    prompt = Column(Text, nullable=False, default="")
    completion = Column(Text, nullable=False, default="")
    reasoning = Column(Text, nullable=False, default="")
    entities = Column(JSON, nullable=False, default=list)
    source = Column(String(64), nullable=False, default="teacher")
    source_document_id = Column(String(64), nullable=False, default="")
    split = Column(String(16), nullable=False, default="train", index=True)
    status = Column(String(32), nullable=False, default="pending", index=True)
    reject_reason = Column(Text, nullable=False, default="")
    quality_score = Column(Float, nullable=False, default=0.0)
    content_hash = Column(String(64), nullable=False, default="", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
