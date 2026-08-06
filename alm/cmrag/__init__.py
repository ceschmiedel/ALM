"""CX · CMRAG — context-managed retrieval, scoped by domain, ontology and IBAC."""

from alm.cmrag.embeddings import (
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    available_embedders,
    get_embedder,
)
from alm.cmrag.ingest import Chunker, CorpusIngestor, EntityAnnotator, keywords_of
from alm.cmrag.retriever import AccessFilter, CMRAGRetriever, RetrievalResult
from alm.cmrag.store import ChunkStore, RetrievedChunk, content_hash

__all__ = [
    "AccessFilter",
    "CMRAGRetriever",
    "ChunkStore",
    "Chunker",
    "CorpusIngestor",
    "Embedder",
    "EntityAnnotator",
    "HashEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "RetrievalResult",
    "RetrievedChunk",
    "SentenceTransformerEmbedder",
    "available_embedders",
    "content_hash",
    "get_embedder",
    "keywords_of",
]
