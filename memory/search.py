"""Atlas $vectorSearch helper used by MemoryManager."""

from __future__ import annotations

from typing import Any, Optional

from memory.db import get_collection

VECTOR_INDEX_NAME = "vector_index"


def vector_search(
    collection_name: str,
    query_embedding: list[float],
    *,
    limit: int = 5,
    num_candidates: Optional[int] = None,
    extra_filter: Optional[dict[str, Any]] = None,
    projection: Optional[dict[str, int]] = None,
) -> list[dict[str, Any]]:
    """Run an Atlas vector search and return enriched docs.

    `extra_filter` is applied as a $match after $vectorSearch (cheap
    post-filter; for large corpora prefer the `filter` field inside
    $vectorSearch with an indexed field).
    """
    coll = get_collection(collection_name)

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates or max(50, limit * 10),
                "limit": limit * 5,
            }
        },
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        {"$unset": "embedding"},
    ]
    if extra_filter:
        pipeline.append({"$match": extra_filter})
    if projection:
        # Inclusion projection from the caller; preserve `score` and `_id`.
        pipeline.append(
            {"$project": {**projection, "score": 1, "_id": 1}}
        )
    pipeline.append({"$limit": limit})

    try:
        results = list(coll.aggregate(pipeline))
    except Exception:
        # Vector index may not be ready yet — fall back to recent docs.
        cursor = (
            coll.find(extra_filter or {}, {"embedding": 0})
            .sort("_id", -1)
            .limit(limit)
        )
        results = list(cursor)

    for r in results:
        if "_id" in r:
            r["_id"] = str(r["_id"])
    return results
