"""Voyage AI embedding helper shared by all memory writers/searchers."""

from __future__ import annotations

from functools import lru_cache

from langchain_voyageai import VoyageAIEmbeddings

# Must match the dimensionality declared in bootstrap.create_vector_indexes.
EMBED_MODEL = "voyage-4-large"
EMBED_DIMS = 1024


@lru_cache(maxsize=1)
def _client() -> VoyageAIEmbeddings:
    return VoyageAIEmbeddings(model=EMBED_MODEL)


def embed_document(text: str) -> list[float]:
    """Embed a document for storage."""
    return _client().embed_documents([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a query for vector search."""
    return _client().embed_query(text)
