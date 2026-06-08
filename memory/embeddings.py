"""Voyage AI embedding helper shared by all memory writers/searchers.

Uses the native ``voyageai`` SDK directly (no langchain-voyageai wrapper)
so there is no conflict with langchain-core>=1.0.  The client reads
``VOYAGE_API_KEY`` from the environment automatically.
"""

from __future__ import annotations

from functools import lru_cache

import voyageai

# Must match the dimensionality declared in bootstrap.create_vector_indexes.
EMBED_MODEL = "voyage-4-large"
EMBED_DIMS = 1024


@lru_cache(maxsize=1)
def _client() -> voyageai.Client:
    return voyageai.Client()


def embed_document(text: str) -> list[float]:
    """Embed a document for storage."""
    return _client().embed([text], model=EMBED_MODEL, input_type="document").embeddings[0]


def embed_query(text: str) -> list[float]:
    """Embed a query for vector search."""
    return _client().embed([text], model=EMBED_MODEL, input_type="query").embeddings[0]
