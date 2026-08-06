"""Vector store over the CMRAG corpus.

Vectors are kept as JSON arrays next to the chunk and scored with NumPy, which
needs no extension and is fast enough well past the size of a typical domain
corpus (a full scan of 100k × 384-dim vectors is a few milliseconds).  Two
things keep that honest:

* the scan is always **pre-filtered by domain and tenant**, so an expert
  searches its own corpus rather than the whole federation's — which is the
  point of *context-managed* retrieval, not merely an optimisation;
* :class:`ChunkStore.needs_reindex` detects a change of embedding provider, so
  vectors from two different spaces are never compared.

For corpora beyond that, point ``ALM_DATABASE_URL`` at PostgreSQL and install
the ``postgres`` extra; see ``docs/deployment.md`` for the pgvector index.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import delete, select

from alm.core.ids import new_id
from alm.persistence.database import session_scope
from alm.persistence.models import ChunkRow, DocumentRow

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class RetrievedChunk:
    """A scored chunk with everything the citation needs."""

    __slots__ = (
        "chunk_id",
        "document_id",
        "title",
        "text",
        "domain",
        "entities",
        "metadata",
        "sensitivity",
        "score",
        "semantic_score",
        "lexical_score",
        "entity_score",
        "ordinal",
    )

    def __init__(
        self,
        *,
        chunk_id: str,
        document_id: str,
        title: str,
        text: str,
        domain: str,
        entities: list[str],
        metadata: dict[str, Any],
        sensitivity: str,
        score: float = 0.0,
        semantic_score: float = 0.0,
        lexical_score: float = 0.0,
        entity_score: float = 0.0,
        ordinal: int = 0,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.title = title
        self.text = text
        self.domain = domain
        self.entities = entities
        self.metadata = metadata
        self.sensitivity = sensitivity
        self.score = score
        self.semantic_score = semantic_score
        self.lexical_score = lexical_score
        self.entity_score = entity_score
        self.ordinal = ordinal

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "domain": self.domain,
            "entities": self.entities,
            "sensitivity": self.sensitivity,
            "score": round(self.score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "entity_score": round(self.entity_score, 6),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RetrievedChunk {self.chunk_id} score={self.score:.3f}>"


class ChunkStore:
    """Persistence and similarity search for CMRAG chunks."""

    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id

    # -- documents ---------------------------------------------------------

    def upsert_document(
        self,
        *,
        title: str,
        domain: str,
        uri: str = "",
        text_hash: str = "",
        metadata: dict[str, Any] | None = None,
        sensitivity: str = "internal",
    ) -> str:
        """Insert a document, or return the existing id when unchanged.

        Re-indexing an unchanged corpus is a no-op, so `alm cmrag index` is
        safe to run on every deploy. Identity is ``(tenant, domain, uri)`` when
        a URI is given (the common case — a corpus file's path is stable
        across reinstalls) and falls back to ``(tenant, domain, title)``
        otherwise. That identity lookup happens *before* the content-hash
        check: an edited file keeps its document id and has its old chunks
        replaced, rather than being inserted as a second, orphaned document
        that a pack reinstall would otherwise leave live in retrieval
        alongside the new version.
        """
        with session_scope() as session:
            identity_stmt = select(DocumentRow).where(
                DocumentRow.tenant_id == self.tenant_id,
                DocumentRow.domain == domain,
            )
            identity_stmt = identity_stmt.where(
                DocumentRow.uri == uri if uri else DocumentRow.title == title
            )
            existing = session.execute(identity_stmt).scalar_one_or_none()

            if existing is not None:
                if text_hash and existing.content_hash == text_hash:
                    return existing.document_id
                # Same document, changed content: drop its old chunks so a
                # reinstall never leaves a stale version live in retrieval.
                session.execute(
                    delete(ChunkRow).where(ChunkRow.document_id == existing.document_id)
                )
                existing.title = title
                existing.uri = uri
                existing.content_hash = text_hash
                existing.doc_metadata = metadata or {}
                existing.sensitivity = sensitivity
                session.flush()
                return existing.document_id

            row = DocumentRow(
                document_id=new_id("doc"),
                tenant_id=self.tenant_id,
                domain=domain,
                title=title,
                uri=uri,
                content_hash=text_hash,
                doc_metadata=metadata or {},
                sensitivity=sensitivity,
            )
            session.add(row)
            session.flush()
            return row.document_id

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with session_scope() as session:
            row = session.get(DocumentRow, document_id)
            if row is None or row.tenant_id != self.tenant_id:
                return None
            return {
                "document_id": row.document_id,
                "title": row.title,
                "domain": row.domain,
                "uri": row.uri,
                "metadata": dict(row.doc_metadata or {}),
                "sensitivity": row.sensitivity,
            }

    def documents(self, domain: str | None = None) -> list[dict[str, Any]]:
        with session_scope() as session:
            stmt = select(DocumentRow).where(DocumentRow.tenant_id == self.tenant_id)
            if domain:
                stmt = stmt.where(DocumentRow.domain == domain)
            rows = session.execute(stmt.order_by(DocumentRow.title)).scalars().all()
            return [
                {
                    "document_id": r.document_id,
                    "title": r.title,
                    "domain": r.domain,
                    "uri": r.uri,
                    "sensitivity": r.sensitivity,
                }
                for r in rows
            ]

    def delete_document(self, document_id: str) -> int:
        with session_scope() as session:
            result = session.execute(
                delete(ChunkRow).where(ChunkRow.document_id == document_id)
            )
            row = session.get(DocumentRow, document_id)
            if row is not None:
                session.delete(row)
            return int(result.rowcount or 0)

    # -- chunks ------------------------------------------------------------

    def add_chunks(
        self,
        document_id: str,
        chunks: Sequence[dict[str, Any]],
        *,
        domain: str,
        embeddings: Sequence[Sequence[float]] | None = None,
        embedding_model: str = "",
        sensitivity: str = "internal",
    ) -> list[str]:
        """Store chunks, optionally with their vectors."""
        ids: list[str] = []
        with session_scope() as session:
            for index, chunk in enumerate(chunks):
                chunk_id = new_id("chk")
                text = str(chunk.get("text", ""))
                row = ChunkRow(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    tenant_id=self.tenant_id,
                    domain=domain,
                    ordinal=int(chunk.get("ordinal", index)),
                    text=text,
                    entities=list(chunk.get("entities", [])),
                    chunk_metadata=dict(chunk.get("metadata", {})),
                    sensitivity=str(chunk.get("sensitivity", sensitivity)),
                    token_count=max(1, len(text.split())),
                    embedding=(
                        list(map(float, embeddings[index]))
                        if embeddings is not None and index < len(embeddings)
                        else None
                    ),
                    embedding_model=embedding_model,
                )
                session.add(row)
                ids.append(chunk_id)
        return ids

    def chunks_for_domain(self, domain: str | None = None) -> list[ChunkRow]:
        with session_scope() as session:
            stmt = select(ChunkRow).where(ChunkRow.tenant_id == self.tenant_id)
            if domain:
                stmt = stmt.where(ChunkRow.domain == domain)
            return list(session.execute(stmt).scalars().all())

    def needs_reindex(self, embedding_signature: str, domain: str | None = None) -> bool:
        """True when stored vectors came from a different embedding space."""
        with session_scope() as session:
            stmt = select(ChunkRow.embedding_model).where(
                ChunkRow.tenant_id == self.tenant_id,
                ChunkRow.embedding.is_not(None),
            )
            if domain:
                stmt = stmt.where(ChunkRow.domain == domain)
            models = {m for m in session.execute(stmt.limit(200)).scalars().all() if m}
            return bool(models) and embedding_signature not in models

    # -- search ------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float] | None,
        *,
        domains: Sequence[str] | None = None,
        entities: Sequence[str] | None = None,
        query_text: str = "",
        top_k: int = 8,
        candidate_limit: int = 5000,
    ) -> list[RetrievedChunk]:
        """Hybrid search: semantic cosine, lexical overlap and entity match.

        The three signals are complementary and the mix is deliberate.  Semantic
        similarity finds paraphrase; lexical overlap catches the exact clause
        numbers and identifiers that domain questions hinge on and that
        embeddings routinely smear; the entity signal enforces the ontology.
        """
        with session_scope() as session:
            stmt = select(ChunkRow).where(ChunkRow.tenant_id == self.tenant_id)
            if domains:
                stmt = stmt.where(ChunkRow.domain.in_(list(domains)))
            rows = list(session.execute(stmt.limit(candidate_limit)).scalars().all())

        if not rows:
            return []

        semantic = self._semantic_scores(rows, query_vector)
        lexical = self._lexical_scores(rows, query_text)
        entity = self._entity_scores(rows, entities)

        titles = self._titles({r.document_id for r in rows})

        results: list[RetrievedChunk] = []
        for index, row in enumerate(rows):
            score = (
                0.55 * semantic[index] + 0.30 * lexical[index] + 0.15 * entity[index]
            )
            if score <= 0.0:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    title=titles.get(row.document_id, ""),
                    text=row.text,
                    domain=row.domain,
                    entities=list(row.entities or []),
                    metadata=dict(row.chunk_metadata or {}),
                    sensitivity=row.sensitivity,
                    score=float(score),
                    semantic_score=float(semantic[index]),
                    lexical_score=float(lexical[index]),
                    entity_score=float(entity[index]),
                    ordinal=row.ordinal,
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]

    # -- scoring helpers ---------------------------------------------------

    @staticmethod
    def _semantic_scores(
        rows: Sequence[ChunkRow], query_vector: Sequence[float] | None
    ) -> np.ndarray:
        if query_vector is None:
            return np.zeros(len(rows))
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return np.zeros(len(rows))
        query = query / norm

        scores = np.zeros(len(rows), dtype=np.float32)
        vectors: list[np.ndarray] = []
        indices: list[int] = []
        for index, row in enumerate(rows):
            if not row.embedding:
                continue
            vector = np.asarray(row.embedding, dtype=np.float32)
            if vector.shape[0] != query.shape[0]:
                continue  # different embedding space — see needs_reindex()
            vectors.append(vector)
            indices.append(index)

        if not vectors:
            return scores

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        cosines = (matrix @ query) / norms
        # Clamp rather than rescale: mapping [-1,1] onto [0,1] would give every
        # unrelated chunk a 0.5 baseline and flatten the ranking.
        scores[indices] = np.clip(cosines, 0.0, 1.0)
        return scores

    @staticmethod
    def _lexical_scores(rows: Sequence[ChunkRow], query_text: str) -> np.ndarray:
        tokens = {t for t in _tokenise(query_text) if len(t) > 2}
        if not tokens:
            return np.zeros(len(rows))
        scores = np.zeros(len(rows), dtype=np.float32)
        for index, row in enumerate(rows):
            chunk_tokens = set(_tokenise(row.text))
            if not chunk_tokens:
                continue
            scores[index] = len(tokens & chunk_tokens) / len(tokens)
        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def _entity_scores(rows: Sequence[ChunkRow], entities: Sequence[str] | None) -> np.ndarray:
        wanted = {e.lower() for e in (entities or []) if e}
        if not wanted:
            return np.zeros(len(rows))
        scores = np.zeros(len(rows), dtype=np.float32)
        for index, row in enumerate(rows):
            present = {str(e).lower() for e in (row.entities or [])}
            if present:
                scores[index] = len(wanted & present) / len(wanted)
        return scores

    def _titles(self, document_ids: set[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        with session_scope() as session:
            rows = session.execute(
                select(DocumentRow.document_id, DocumentRow.title).where(
                    DocumentRow.document_id.in_(list(document_ids))
                )
            ).all()
            return {row[0]: row[1] for row in rows}

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with session_scope() as session:
            chunks = session.execute(
                select(ChunkRow.domain, ChunkRow.embedding_model).where(
                    ChunkRow.tenant_id == self.tenant_id
                )
            ).all()
            documents = session.execute(
                select(DocumentRow.domain).where(DocumentRow.tenant_id == self.tenant_id)
            ).scalars().all()

        by_domain: dict[str, int] = {}
        for domain, _ in chunks:
            by_domain[domain] = by_domain.get(domain, 0) + 1
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "chunks_by_domain": by_domain,
            "embedding_models": sorted({m for _, m in chunks if m}),
        }

    def clear(self, domain: str | None = None) -> int:
        with session_scope() as session:
            chunk_stmt = select(ChunkRow).where(ChunkRow.tenant_id == self.tenant_id)
            doc_stmt = select(DocumentRow).where(DocumentRow.tenant_id == self.tenant_id)
            if domain:
                chunk_stmt = chunk_stmt.where(ChunkRow.domain == domain)
                doc_stmt = doc_stmt.where(DocumentRow.domain == domain)
            chunks = list(session.execute(chunk_stmt).scalars().all())
            for chunk in chunks:
                session.delete(chunk)
            for document in session.execute(doc_stmt).scalars().all():
                session.delete(document)
            return len(chunks)


_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]
