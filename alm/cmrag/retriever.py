"""CMRAG — Context-Managed Retrieval-Augmented Generation.

The "managed" is the whole point.  Plain RAG asks "what is similar to this
question?"; CMRAG asks "what may *this expert* see, within *its* domain, about
*these ontology entities*?"  Three constraints are applied before similarity is
even considered:

1. **Domain scope** — an expert searches its own corpus. Less noise, fewer
   hallucinations, and a smaller blast radius when something goes wrong.
2. **Ontology scope** — the entity types the capability declares it operates on
   bias retrieval toward the right kind of fragment.
3. **Governance** — every candidate passes through IBAC before it reaches a
   prompt. No context access happens outside that check, and each denial is
   recorded.

Retrieval that is denied is *reported*, never silently dropped: an expert
answering with less evidence than it asked for must be visible in the trace.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from alm.cmrag.embeddings import Embedder
from alm.cmrag.store import ChunkStore, RetrievedChunk
from alm.protocol.answer import Citation

logger = logging.getLogger(__name__)

#: Returns ``True`` when a chunk may be shown to the requesting expert.
AccessFilter = Callable[[RetrievedChunk], bool]


class RetrievalResult:
    """What a retrieval call produced, including what it was denied."""

    __slots__ = ("chunks", "denied", "domains", "entities", "query", "degraded")

    def __init__(
        self,
        *,
        chunks: list[RetrievedChunk],
        denied: list[dict[str, Any]],
        domains: list[str],
        entities: list[str],
        query: str,
        degraded: bool = False,
    ) -> None:
        self.chunks = chunks
        self.denied = denied
        self.domains = domains
        self.entities = entities
        self.query = query
        #: True when semantic scoring was unavailable and results are lexical only.
        self.degraded = degraded

    def __len__(self) -> int:
        return len(self.chunks)

    def __bool__(self) -> bool:
        return bool(self.chunks)

    def citations(self) -> list[Citation]:
        return [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                snippet=c.text[:400],
                score=round(c.score, 6),
                domain=c.domain,
                entities=c.entities,
                # Sensitivity travels with the citation so the emission gate can
                # classify the answer from its evidence rather than guessing.
                metadata={**c.metadata, "sensitivity": c.sensitivity},
            )
            for c in self.chunks
        ]

    def as_context_blocks(self, max_chars: int = 6000) -> list[dict[str, Any]]:
        """Chunks shaped for prompt construction, within a character budget."""
        blocks: list[dict[str, Any]] = []
        used = 0
        for index, chunk in enumerate(self.chunks, start=1):
            text = chunk.text.strip()
            if used + len(text) > max_chars and blocks:
                break
            used += len(text)
            blocks.append(
                {
                    "index": index,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section": chunk.metadata.get("section", ""),
                    "domain": chunk.domain,
                    "entities": chunk.entities,
                    "text": text,
                    "score": round(chunk.score, 4),
                }
            )
        return blocks

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "domains": self.domains,
            "entities": self.entities,
            "chunks": [c.to_dict() for c in self.chunks],
            "denied": self.denied,
            "degraded": self.degraded,
        }


class CMRAGRetriever:
    """Domain-scoped, ontology-aware, governed retrieval."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | None = None,
        *,
        access_filter: AccessFilter | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.access_filter = access_filter

    def retrieve(
        self,
        query: str,
        *,
        domains: Sequence[str] | None = None,
        entities: Sequence[str] | None = None,
        top_k: int = 6,
        oversample: int = 3,
    ) -> RetrievalResult:
        """Retrieve for one expert, in its scope, under governance.

        ``oversample`` fetches more candidates than requested so that governance
        filtering does not silently shrink the evidence set below ``top_k`` when
        some fragments are out of bounds for this expert.
        """
        vector = None
        degraded = True
        if self.embedder is not None:
            try:
                vector = self.embedder.embed_one(query)
                degraded = False
            except Exception:
                logger.warning(
                    "Embedding the query failed; retrieval degraded to lexical matching",
                    exc_info=True,
                )

        candidates = self.store.search(
            vector,
            domains=list(domains) if domains else None,
            entities=list(entities) if entities else None,
            query_text=query,
            top_k=max(top_k * oversample, top_k),
        )

        allowed: list[RetrievedChunk] = []
        denied: list[dict[str, Any]] = []
        for chunk in candidates:
            if self.access_filter is not None and not self.access_filter(chunk):
                denied.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "domain": chunk.domain,
                        "sensitivity": chunk.sensitivity,
                        "entities": chunk.entities,
                    }
                )
                continue
            allowed.append(chunk)
            if len(allowed) >= top_k:
                break

        if denied:
            logger.info(
                "Retrieval for %r: %d fragment(s) withheld by governance",
                query[:60],
                len(denied),
            )

        return RetrievalResult(
            chunks=allowed,
            denied=denied,
            domains=list(domains or []),
            entities=list(entities or []),
            query=query,
            degraded=degraded,
        )

    def with_access_filter(self, access_filter: AccessFilter | None) -> CMRAGRetriever:
        """Return a view of this retriever bound to a different governance filter.

        Each expert gets its own filter for the same underlying corpus, which is
        exactly the per-agent access control IBAC calls for.
        """
        return CMRAGRetriever(self.store, self.embedder, access_filter=access_filter)

    def stats(self) -> dict[str, Any]:
        stats = self.store.stats()
        stats["embedder"] = self.embedder.signature if self.embedder else None
        if self.embedder is not None:
            stats["needs_reindex"] = self.store.needs_reindex(self.embedder.signature)
        return stats
