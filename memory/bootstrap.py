"""Idempotent Atlas index bootstrap for memory collections.

Creates regular pymongo indexes (TTL, lookup) and Atlas $vectorSearch
indexes. Re-running is safe: existing indexes are left in place.
"""

from __future__ import annotations

import logging
import time

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from memory.db import (
    EPISODIC_COLLECTION,
    PROCEDURAL_COLLECTION,
    SEMANTIC_COLLECTION,
    SHARED_COLLECTION,
    WORKING_COLLECTION,
    get_collection,
    get_db,
)
from memory.embeddings import EMBED_DIMS

logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "vector_index"
VECTOR_COLLECTIONS = (
    EPISODIC_COLLECTION,
    SEMANTIC_COLLECTION,
    PROCEDURAL_COLLECTION,
)


def _ensure_index(coll, keys, name, **opts) -> None:
    try:
        coll.create_index(keys, name=name, **opts)
    except OperationFailure as exc:
        if "already exists" in str(exc).lower():
            return
        raise


def _drop_index_if_exists(coll, name: str) -> None:
    try:
        coll.drop_index(name)
    except OperationFailure:
        pass


def _ensure_vector_index(coll) -> None:
    """Create the vector_index on `embedding` if it doesn't exist."""
    existing = {idx["name"] for idx in coll.list_search_indexes()}
    if VECTOR_INDEX_NAME in existing:
        return

    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBED_DIMS,
                    "similarity": "cosine",
                }
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )
    coll.create_search_index(model=model)
    logger.info("Created vector index on %s", coll.name)


def wait_for_vector_indexes(timeout: int = 180) -> None:
    """Poll until all vector indexes are queryable, or raise."""
    deadline = time.time() + timeout
    pending = set(VECTOR_COLLECTIONS)
    while pending and time.time() < deadline:
        for name in list(pending):
            coll = get_collection(name)
            for idx in coll.list_search_indexes():
                if idx["name"] == VECTOR_INDEX_NAME and idx.get("queryable"):
                    pending.discard(name)
                    break
        if pending:
            time.sleep(3)
    if pending:
        raise TimeoutError(
            f"Vector indexes not ready after {timeout}s: {sorted(pending)}"
        )


def bootstrap_indexes(wait_for_vectors: bool = False) -> None:
    """Create every index this app relies on. Safe to re-run."""
    db = get_db()

    # Working memory: one doc per session, TTL 24h.
    wm = db[WORKING_COLLECTION]
    _ensure_index(wm, [("session_id", 1)], "session_unique", unique=True)
    _ensure_index(wm, [("user_id", 1)], "user_lookup")
    _ensure_index(wm, [("updated_at", 1)], "ttl", expireAfterSeconds=60 * 60 * 24)

    # Episodic memory: timeline lookups.
    em = db[EPISODIC_COLLECTION]
    _ensure_index(em, [("user_id", 1), ("timestamp", -1)], "user_timeline")
    _ensure_index(em, [("event_type", 1)], "by_type")

    # Semantic memory: kind/key lookups.
    sm = db[SEMANTIC_COLLECTION]
    _ensure_index(sm, [("user_id", 1), ("kind", 1)], "user_kind")
    _ensure_index(
        sm, [("user_id", 1), ("kind", 1), ("key", 1)], "user_kind_key", unique=True
    )

    # Procedural memory: name lookups.
    pm = db[PROCEDURAL_COLLECTION]
    _ensure_index(pm, [("user_id", 1), ("name", 1)], "user_name", unique=True)

    # Shared memory: per-session slots (1h TTL) plus optional project-scoped
    # slots (no TTL) for long-lived state like strategic goals.
    sh = db[SHARED_COLLECTION]
    _ensure_index(sh, [("session_id", 1), ("slot", 1)], "session_slot")
    # One-time migration: the original unconditional TTL would also expire
    # project-scoped docs. Replace it with a partial TTL that only sweeps
    # session-scoped entries.
    _drop_index_if_exists(sh, "ttl")
    _ensure_index(
        sh,
        [("created_at", 1)],
        "ttl_session",
        expireAfterSeconds=60 * 60,
        partialFilterExpression={"scope": "session"},
    )
    _ensure_index(
        sh,
        [("project_key", 1), ("slot", 1), ("created_at", -1)],
        "project_slot",
        partialFilterExpression={"project_key": {"$type": "string"}},
    )

    # Vector search indexes.
    for name in VECTOR_COLLECTIONS:
        _ensure_vector_index(db[name])

    if wait_for_vectors:
        wait_for_vector_indexes()

    logger.info("Memory indexes bootstrapped on database %s", db.name)
