"""Corpus ingestion — chunking and ontology annotation.

Chunking respects structure before size: Markdown headings and blank lines are
boundaries, and a chunk carries its heading path so a retrieved fragment still
says which clause or section it came from.

Every chunk is then annotated with the ontology entity types it mentions. That
annotation is what makes retrieval *context-managed* rather than merely
semantic — an expert asking about a `Clause` gets fragments the graph knows are
about clauses, not everything that happens to sit nearby in vector space.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from alm.cmrag.embeddings import Embedder
from alm.cmrag.store import ChunkStore, content_hash
from alm.graph.ontology import Ontology

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_WORD_RE = re.compile(r"[\wÀ-ÿ]+")

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".text"}


class Chunker:
    """Structure-aware splitter with overlap."""

    def __init__(
        self,
        *,
        target_words: int = 220,
        max_words: int = 400,
        overlap_words: int = 40,
    ) -> None:
        self.target_words = target_words
        self.max_words = max_words
        self.overlap_words = overlap_words

    def split(self, text: str) -> list[dict[str, Any]]:
        """Split text into chunks carrying their heading path."""
        blocks = self._blocks(text)
        chunks: list[dict[str, Any]] = []
        buffer: list[str] = []
        buffer_words = 0
        heading_path: list[str] = []

        def flush() -> None:
            nonlocal buffer, buffer_words
            if not buffer:
                return
            body = "\n\n".join(buffer).strip()
            if body:
                chunks.append(
                    {
                        "text": body,
                        "ordinal": len(chunks),
                        "metadata": {"section": " › ".join(heading_path)} if heading_path else {},
                    }
                )
            if self.overlap_words > 0 and body:
                words = body.split()
                tail = words[-self.overlap_words :]
                buffer = [" ".join(tail)] if tail else []
                buffer_words = len(tail)
            else:
                buffer = []
                buffer_words = 0

        for kind, level, content in blocks:
            if kind == "heading":
                flush()
                heading_path = heading_path[: max(0, level - 1)]
                heading_path.append(content)
                buffer = []
                buffer_words = 0
                continue

            words = len(content.split())
            if words > self.max_words:
                # A single oversized paragraph is split on sentence boundaries.
                flush()
                for piece in self._split_long(content):
                    chunks.append(
                        {
                            "text": piece,
                            "ordinal": len(chunks),
                            "metadata": (
                                {"section": " › ".join(heading_path)} if heading_path else {}
                            ),
                        }
                    )
                continue

            if buffer_words + words > self.max_words:
                flush()
            buffer.append(content)
            buffer_words += words
            if buffer_words >= self.target_words:
                flush()

        flush()
        return [c for c in chunks if c["text"].strip()]

    def _blocks(self, text: str) -> list[tuple[str, int, str]]:
        blocks: list[tuple[str, int, str]] = []
        paragraph: list[str] = []
        for line in (text or "").splitlines():
            heading = _HEADING_RE.match(line.strip())
            if heading:
                if paragraph:
                    blocks.append(("text", 0, "\n".join(paragraph).strip()))
                    paragraph = []
                blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
                continue
            if not line.strip():
                if paragraph:
                    blocks.append(("text", 0, "\n".join(paragraph).strip()))
                    paragraph = []
                continue
            paragraph.append(line)
        if paragraph:
            blocks.append(("text", 0, "\n".join(paragraph).strip()))
        return [b for b in blocks if b[2]]

    def _split_long(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces: list[str] = []
        current: list[str] = []
        count = 0
        for sentence in sentences:
            words = len(sentence.split())
            if count + words > self.max_words and current:
                pieces.append(" ".join(current))
                current = []
                count = 0
            current.append(sentence)
            count += words
        if current:
            pieces.append(" ".join(current))
        return pieces


class EntityAnnotator:
    """Tags chunks with the ontology entity types they mention.

    Matching is over the entity name plus its declared keywords, on word
    boundaries and case-insensitively.  This is intentionally simple and
    inspectable: an operator can predict what will be tagged by reading the
    ontology, which matters more here than recall, because these tags gate
    retrieval and IBAC sensitivity.
    """

    def __init__(self, ontology: Ontology | None = None) -> None:
        self.patterns: dict[str, re.Pattern[str]] = {}
        self.sensitivity: dict[str, str] = {}
        self.entity_domain: dict[str, str] = {}
        if ontology is not None:
            self.load(ontology)

    def load(self, ontology: Ontology) -> None:
        for entity in ontology.entities:
            terms = {entity.name, *entity.keywords}
            escaped = sorted(
                (re.escape(t) for t in terms if t and len(t) > 2), key=len, reverse=True
            )
            if not escaped:
                continue
            self.patterns[entity.name] = re.compile(
                r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
            )
            self.sensitivity[entity.name] = entity.sensitivity
            self.entity_domain[entity.name] = ontology.domain

    def annotate(self, text: str) -> list[str]:
        """Tag with every ontology entity mentioned, across all domains.

        Cross-domain tags are kept because they genuinely help retrieval — a
        contract that discusses a regulatory obligation should be findable from
        the risk domain. Classification is a separate question; see
        :meth:`max_sensitivity`.
        """
        return [name for name, pattern in self.patterns.items() if pattern.search(text or "")]

    def max_sensitivity(
        self,
        entities: Sequence[str],
        default: str = "internal",
        *,
        domain: str = "",
    ) -> str:
        """Strictest sensitivity among the mentioned entities of ``domain``.

        Classification is governed by the ontology that **owns** the document.
        Letting any domain's entity raise it is a latent disaster: the word
        "breach" appears in every contract, it is also a keyword of the risk
        domain's ``Incident`` entity, and one keyword collision would silently
        reclassify an entire contract as restricted — locking the legal expert
        out of its own corpus and inverting the governance intent.

        Cross-domain mentions still influence retrieval; they just do not
        reclassify the document.
        """
        order = ["public", "internal", "confidential", "restricted"]
        best = default
        for entity in entities:
            if domain and self.entity_domain.get(entity, domain) != domain:
                continue
            level = self.sensitivity.get(entity)
            if level and order.index(level) > order.index(best):
                best = level
        return best


class CorpusIngestor:
    """Reads documents, chunks, annotates, embeds and stores them."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | None = None,
        chunker: Chunker | None = None,
        annotator: EntityAnnotator | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or Chunker()
        self.annotator = annotator or EntityAnnotator()

    def ingest_text(
        self,
        text: str,
        *,
        domain: str,
        title: str,
        uri: str = "",
        metadata: dict[str, Any] | None = None,
        sensitivity: str = "internal",
    ) -> dict[str, Any]:
        """Ingest one document.  Returns a small report."""
        return self.ingest_chunks(
            self.chunker.split(text),
            text=text,
            domain=domain,
            title=title,
            uri=uri,
            metadata=metadata,
            sensitivity=sensitivity,
        )

    def ingest_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        text: str,
        domain: str,
        title: str,
        uri: str = "",
        metadata: dict[str, Any] | None = None,
        sensitivity: str = "internal",
    ) -> dict[str, Any]:
        """Ingest a document whose chunking was decided by the caller.

        Prose is split by :class:`Chunker`, but a table is not prose: its
        chunk boundaries are row groups, and each one has to carry its header
        to stay retrievable (see :mod:`alm.cmrag.tabular`).  Both paths share
        the dedupe, annotation, embedding and storage below, so a table is
        governed and cited exactly like a document.

        ``text`` is the full rendering the content hash is taken over, so an
        unchanged re-upload is correctly a no-op.
        """
        digest = content_hash(text)
        existing = self.store.upsert_document(
            title=title,
            domain=domain,
            uri=uri,
            text_hash=digest,
            metadata=metadata,
            sensitivity=sensitivity,
        )
        # upsert_document returns the pre-existing id when the content is
        # unchanged; in that case there is nothing to re-chunk.
        already_indexed = any(
            c.document_id == existing for c in self.store.chunks_for_domain(domain)
        )
        if already_indexed:
            return {
                "document_id": existing,
                "title": title,
                "chunks": 0,
                "skipped": True,
                "reason": "unchanged",
            }

        for chunk in chunks:
            entities = self.annotator.annotate(chunk["text"])
            chunk["entities"] = entities
            chunk["sensitivity"] = self.annotator.max_sensitivity(
                entities, sensitivity, domain=domain
            )

        embeddings = None
        signature = ""
        if self.embedder is not None and chunks:
            embeddings = self.embedder.embed([c["text"] for c in chunks])
            signature = self.embedder.signature

        self.store.add_chunks(
            existing,
            chunks,
            domain=domain,
            embeddings=embeddings,
            embedding_model=signature,
            sensitivity=sensitivity,
        )
        return {
            "document_id": existing,
            "title": title,
            "chunks": len(chunks),
            "skipped": False,
            "embedded": embeddings is not None,
        }

    def ingest_table(
        self,
        data: bytes,
        filename: str,
        *,
        domain: str,
        sensitivity: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest a CSV or XLSX upload — one document per sheet."""
        from alm.cmrag.tabular import read_table, sheet_to_document

        reports: list[dict[str, Any]] = []
        sheets = read_table(data, filename)
        for sheet in sheets:
            document = sheet_to_document(sheet, source=filename)
            # Multi-sheet workbooks would otherwise collide on title, and the
            # (domain, uri) identity is what makes a re-upload replace rather
            # than duplicate — so the sheet name has to be part of the uri.
            title = (
                f"{Path(filename).stem} · {document['sheet']}"
                if len(sheets) > 1
                else Path(filename).stem
            )
            report = self.ingest_chunks(
                document["chunks"],
                text=document["text"],
                domain=domain,
                title=title,
                uri=f"upload://{filename}#{document['sheet']}",
                metadata={
                    **(metadata or {}),
                    "filename": filename,
                    "sheet": document["sheet"],
                    "rows": document["rows"],
                    "columns": document["columns"],
                    "kind": "table",
                },
                sensitivity=sensitivity,
            )
            report["sheet"] = document["sheet"]
            report["rows"] = document["rows"]
            report["columns"] = document["columns"]
            reports.append(report)
        return reports

    def ingest_file(
        self,
        path: str | Path,
        *,
        domain: str,
        metadata: dict[str, Any] | None = None,
        sensitivity: str = "internal",
    ) -> dict[str, Any]:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(
            text,
            domain=domain,
            title=file_path.stem.replace("-", " ").replace("_", " ").title(),
            uri=str(file_path),
            metadata={**(metadata or {}), "filename": file_path.name},
            sensitivity=sensitivity,
        )

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        domain: str = "",
        sensitivity: str = "internal",
    ) -> list[dict[str, Any]]:
        """Ingest a corpus tree.

        When ``domain`` is empty, the first path segment under the root names
        the domain — the layout domain packs use (``corpus/legal/*.md``).
        """
        root = Path(directory)
        if not root.exists():
            return []
        reports: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            resolved_domain = domain or (
                relative.parts[0] if len(relative.parts) > 1 else ""
            )
            if not resolved_domain:
                logger.warning("Skipping %s: cannot infer a domain from its path", path)
                continue
            reports.append(
                self.ingest_file(
                    path, domain=resolved_domain, sensitivity=sensitivity
                )
            )
        return reports

    def reindex(self, domain: str | None = None) -> int:
        """Recompute embeddings for stored chunks after an embedder change."""
        if self.embedder is None:
            return 0
        rows = self.store.chunks_for_domain(domain)
        if not rows:
            return 0
        vectors = self.embedder.embed([r.text for r in rows])
        signature = self.embedder.signature

        from sqlalchemy import select

        from alm.persistence.database import session_scope
        from alm.persistence.models import ChunkRow

        with session_scope() as session:
            for row, vector in zip(rows, vectors, strict=False):
                live = session.execute(
                    select(ChunkRow).where(ChunkRow.chunk_id == row.chunk_id)
                ).scalar_one_or_none()
                if live is None:
                    continue
                live.embedding = list(map(float, vector))
                live.embedding_model = signature
        return len(rows)


def keywords_of(text: str, limit: int = 12) -> list[str]:
    """Cheap keyword extraction, used to seed capability declarations."""
    counts: dict[str, int] = {}
    for word in _WORD_RE.findall((text or "").lower()):
        if len(word) < 4:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked: Iterable[tuple[str, int]] = sorted(
        counts.items(), key=lambda kv: kv[1], reverse=True
    )
    return [word for word, _ in list(ranked)[:limit]]
