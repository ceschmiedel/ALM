"""CX · embeddings, chunking, ontology annotation and governed retrieval."""

from __future__ import annotations

import pytest

from alm.cmrag.embeddings import HashEmbedder, available_embedders, get_embedder
from alm.cmrag.ingest import Chunker, CorpusIngestor, EntityAnnotator, keywords_of
from alm.cmrag.retriever import CMRAGRetriever
from alm.cmrag.store import ChunkStore, content_hash
from alm.graph.ontology import Ontology

_DOCUMENT = """# Acme MSA

## 7. Liability

Clause 7.2 caps the aggregate liability of each party at the total fees paid in
the twelve months preceding the claim. This cap excludes gross negligence.

## 4. Payment

Payment terms are net 30 days from the invoice date. Late payment accrues
interest at one percent per month on the outstanding balance.
"""


def _legal_ontology() -> Ontology:
    return Ontology(
        domain="legal",
        entities=[
            {"name": "Clause", "keywords": ["clause", "provision"], "sensitivity": "confidential"},
            {"name": "Contract", "keywords": ["agreement", "msa"], "sensitivity": "internal"},
        ],
    )


def _risk_ontology() -> Ontology:
    return Ontology(
        domain="risk",
        entities=[
            {"name": "Incident", "keywords": ["incident", "outage"], "sensitivity": "restricted"},
        ],
    )


# -- embeddings --------------------------------------------------------------


def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder(dim=64)
    first = embedder.embed_one("liability cap")
    second = embedder.embed_one("liability cap")
    assert first == second
    assert len(first) == 64
    assert sum(v * v for v in first) == pytest.approx(1.0, abs=1e-6)


def test_hash_embedder_separates_unrelated_text():
    embedder = HashEmbedder(dim=256)
    from alm.graph.store import cosine

    related = cosine(
        embedder.embed_one("liability cap in the contract"),
        embedder.embed_one("the contract liability cap"),
    )
    unrelated = cosine(
        embedder.embed_one("liability cap in the contract"),
        embedder.embed_one("penguins migrate across antarctica"),
    )
    assert related > unrelated


def test_embedder_signature_identifies_the_vector_space():
    assert HashEmbedder(dim=128).signature != HashEmbedder(dim=256).signature


def test_get_embedder_falls_back_when_provider_unavailable(monkeypatch):
    monkeypatch.setenv("ALM_EMBEDDING_PROVIDER", "sentence")
    from alm.config import reset_settings_cache

    reset_settings_cache()
    # sentence-transformers is not installed in the base environment; the
    # runtime must degrade rather than fail to start.
    embedder = get_embedder()
    assert embedder.name in {"hash", "sentence"}


def test_available_embedders_lists_providers():
    assert {"hash", "ollama", "openai"} <= set(available_embedders())


# -- chunking ----------------------------------------------------------------


def test_chunker_preserves_heading_path():
    chunks = Chunker(target_words=25, max_words=60, overlap_words=0).split(_DOCUMENT)
    assert chunks
    sections = {c["metadata"].get("section", "") for c in chunks}
    assert any("Liability" in s for s in sections)


def test_chunker_splits_oversized_paragraphs():
    long_text = " ".join(f"sentence number {i} about contracts." for i in range(200))
    chunks = Chunker(target_words=50, max_words=80).split(long_text)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 120 for c in chunks)


def test_keywords_of_ranks_frequent_terms():
    keywords = keywords_of("liability liability liability payment payment invoice")
    assert keywords[0] == "liability"


# -- annotation --------------------------------------------------------------


def test_annotator_tags_entities_by_keyword():
    annotator = EntityAnnotator(_legal_ontology())
    tags = annotator.annotate("This provision of the agreement is binding.")
    assert set(tags) == {"Clause", "Contract"}


def test_sensitivity_is_governed_by_the_owning_domain():
    """A keyword collision in another domain must not reclassify a document.

    This is the regression for a real bug: the risk ontology's `Incident`
    entity matched ordinary contract prose, promoting legal chunks to
    `restricted` and locking the legal expert out of its own corpus.
    """
    annotator = EntityAnnotator(_legal_ontology())
    annotator.load(_risk_ontology())

    entities = ["Clause", "Incident"]
    assert annotator.max_sensitivity(entities, "internal", domain="legal") == "confidential"
    assert annotator.max_sensitivity(entities, "internal", domain="risk") == "restricted"
    # Without a domain, the strictest wins — the old, unsafe behaviour.
    assert annotator.max_sensitivity(entities, "internal") == "restricted"


# -- store and retrieval -----------------------------------------------------


def test_ingest_then_retrieve(embedder):
    store = ChunkStore()
    ingestor = CorpusIngestor(store, embedder, annotator=EntityAnnotator(_legal_ontology()))
    report = ingestor.ingest_text(_DOCUMENT, domain="legal", title="Acme MSA")
    assert report["chunks"] > 0
    assert report["embedded"] is True

    retriever = CMRAGRetriever(store, embedder)
    result = retriever.retrieve("What is the liability cap?", domains=["legal"], top_k=3)
    assert result.chunks
    assert "liability" in result.chunks[0].text.lower()
    assert result.degraded is False


def test_reingesting_unchanged_content_is_a_noop(embedder):
    store = ChunkStore()
    ingestor = CorpusIngestor(store, embedder)
    first = ingestor.ingest_text(_DOCUMENT, domain="legal", title="Acme MSA")
    second = ingestor.ingest_text(_DOCUMENT, domain="legal", title="Acme MSA")
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["reason"] == "unchanged"


def test_retrieval_is_scoped_to_the_requested_domain(embedder):
    store = ChunkStore()
    ingestor = CorpusIngestor(store, embedder)
    ingestor.ingest_text(_DOCUMENT, domain="legal", title="MSA")
    ingestor.ingest_text(
        "Budget allocation for the platform is one million euros this year.",
        domain="finance",
        title="Budget",
    )

    retriever = CMRAGRetriever(store, embedder)
    legal_only = retriever.retrieve("budget allocation", domains=["legal"], top_k=5)
    assert all(c.domain == "legal" for c in legal_only.chunks)


def test_access_filter_withholds_and_reports(embedder):
    store = ChunkStore()
    ingestor = CorpusIngestor(store, embedder, annotator=EntityAnnotator(_legal_ontology()))
    ingestor.ingest_text(_DOCUMENT, domain="legal", title="MSA")

    retriever = CMRAGRetriever(store, embedder)
    unfiltered = retriever.retrieve("liability", domains=["legal"], top_k=10)
    assert unfiltered.chunks

    guarded = retriever.with_access_filter(lambda chunk: chunk.sensitivity != "confidential")
    filtered = guarded.retrieve("liability", domains=["legal"], top_k=10)
    assert len(filtered.chunks) < len(unfiltered.chunks)
    # A withheld fragment is reported, never silently dropped.
    assert filtered.denied


def test_retrieval_degrades_without_an_embedder(embedder):
    store = ChunkStore()
    CorpusIngestor(store, embedder).ingest_text(_DOCUMENT, domain="legal", title="MSA")

    lexical_only = CMRAGRetriever(store, embedder=None)
    result = lexical_only.retrieve("liability cap", domains=["legal"], top_k=3)
    assert result.degraded is True
    assert result.chunks, "lexical matching must still return results"


def test_needs_reindex_detects_a_changed_vector_space(embedder):
    store = ChunkStore()
    CorpusIngestor(store, embedder).ingest_text(_DOCUMENT, domain="legal", title="MSA")
    assert store.needs_reindex(embedder.signature) is False
    assert store.needs_reindex(HashEmbedder(dim=512).signature) is True


def test_context_blocks_respect_the_character_budget(embedder):
    store = ChunkStore()
    CorpusIngestor(store, embedder).ingest_text(_DOCUMENT, domain="legal", title="MSA")
    result = CMRAGRetriever(store, embedder).retrieve("liability", domains=["legal"], top_k=10)
    blocks = result.as_context_blocks(max_chars=120)
    assert blocks
    assert sum(len(b["text"]) for b in blocks) <= 400


def test_citations_carry_sensitivity_for_the_emission_gate(embedder):
    store = ChunkStore()
    CorpusIngestor(
        store, embedder, annotator=EntityAnnotator(_legal_ontology())
    ).ingest_text(_DOCUMENT, domain="legal", title="MSA")
    result = CMRAGRetriever(store, embedder).retrieve("liability", domains=["legal"], top_k=2)
    assert all("sensitivity" in c.metadata for c in result.citations())


def test_content_hash_is_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
