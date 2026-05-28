"""High-level façade over the five memory collections.

Every read/write goes through this class so the Chainlit UI can attach
a single subscriber and trace every memory op.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from pymongo import ReturnDocument

from memory.db import (
    EPISODIC_COLLECTION,
    PROCEDURAL_COLLECTION,
    SEMANTIC_COLLECTION,
    SHARED_COLLECTION,
    WORKING_COLLECTION,
    get_collection,
)
from memory.embeddings import embed_document, embed_query
from memory.events import MemoryEvent, MemoryTrace
from memory.schemas import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    SharedMemory,
    WorkingMemory,
)
from memory.search import vector_search


class MemoryManager:
    """All memory reads and writes funnel through here."""

    def __init__(self, trace: Optional[MemoryTrace] = None) -> None:
        self.trace = trace or MemoryTrace()

    # ── tracing helper ──────────────────────────────────────────────────

    def _emit(self, **kwargs: Any) -> None:
        self.trace.emit(MemoryEvent(**kwargs))

    # ── working memory ─────────────────────────────────────────────────

    def read_session_context(
        self, session_id: str, *, reason: Optional[str] = None
    ) -> dict[str, Any]:
        t0 = time.time()
        doc = get_collection(WORKING_COLLECTION).find_one(
            {"session_id": session_id}, {"_id": 0}
        ) or {}
        self._emit(
            op="read",
            memory_type="working",
            description=reason or f"session={session_id}",
            latency_ms=int((time.time() - t0) * 1000),
            result_count=1 if doc else 0,
            session_id=session_id,
            data=doc or None,
        )
        return doc

    def update_session_context(
        self,
        session_id: str,
        user_id: str,
        updates: dict[str, Any],
        *,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        updates = {k: v for k, v in updates.items() if v is not None}
        updates["updated_at"] = datetime.utcnow()
        doc = get_collection(WORKING_COLLECTION).find_one_and_update(
            {"session_id": session_id},
            {"$set": {**updates, "user_id": user_id}, "$setOnInsert": {
                "session_id": session_id
            }},
            upsert=True,
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
        self._emit(
            op="write",
            memory_type="working",
            description=reason or f"updated {sorted(updates)}",
            latency_ms=int((time.time() - t0) * 1000),
            payload=updates,
            session_id=session_id,
            data=doc,
        )
        return doc

    def clear_session_context(
        self, session_id: str, *, reason: Optional[str] = None
    ) -> None:
        get_collection(WORKING_COLLECTION).delete_one({"session_id": session_id})
        self._emit(
            op="delete",
            memory_type="working",
            description=reason or f"cleared session={session_id}",
            session_id=session_id,
        )

    # ── episodic memory ────────────────────────────────────────────────

    def record_episode(
        self, episode: EpisodicMemory, *, reason: Optional[str] = None
    ) -> str:
        t0 = time.time()
        clean_doc = episode.model_dump()
        doc = {**clean_doc, "embedding": embed_document(
            f"{episode.event_type}: {episode.summary}"
        )}
        result = get_collection(EPISODIC_COLLECTION).insert_one(doc)
        clean_doc["_id"] = str(result.inserted_id)
        self._emit(
            op="write",
            memory_type="episodic",
            description=reason or f"{episode.event_type}: {episode.summary[:60]}",
            latency_ms=int((time.time() - t0) * 1000),
            payload={"entities": episode.entities},
            data=clean_doc,
        )
        return str(result.inserted_id)

    def search_episodes(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        *,
        reason: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        results = vector_search(
            EPISODIC_COLLECTION,
            embed_query(query),
            limit=limit,
            extra_filter={"user_id": user_id},
            projection={
                "summary": 1,
                "event_type": 1,
                "entities": 1,
                "timestamp": 1,
            },
        )
        self._emit(
            op="read",
            memory_type="episodic",
            description=reason or f"query={query!r}",
            latency_ms=int((time.time() - t0) * 1000),
            result_count=len(results),
            data=results,
        )
        return results

    def recent_episodes(
        self,
        user_id: str,
        limit: int = 10,
        event_type: Optional[str] = None,
        *,
        reason: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        q: dict[str, Any] = {"user_id": user_id}
        if event_type:
            q["event_type"] = event_type
        cursor = (
            get_collection(EPISODIC_COLLECTION)
            .find(q, {"embedding": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        results = list(cursor)
        for r in results:
            r["_id"] = str(r["_id"])
        self._emit(
            op="read",
            memory_type="episodic",
            description=reason or f"recent (type={event_type or 'any'})",
            latency_ms=int((time.time() - t0) * 1000),
            result_count=len(results),
            data=results,
        )
        return results

    # ── semantic memory ────────────────────────────────────────────────

    def upsert_semantic(
        self, fact: SemanticMemory, *, reason: Optional[str] = None
    ) -> str:
        t0 = time.time()
        clean_doc = fact.model_dump()
        doc = {**clean_doc, "embedding": embed_document(fact.content)}
        result = get_collection(SEMANTIC_COLLECTION).find_one_and_update(
            {"user_id": fact.user_id, "kind": fact.kind, "key": fact.key},
            {"$set": doc},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        clean_doc["_id"] = str(result["_id"])
        self._emit(
            op="write",
            memory_type="semantic",
            description=reason or f"{fact.kind}:{fact.key}",
            latency_ms=int((time.time() - t0) * 1000),
            payload={"value": fact.value},
            data=clean_doc,
        )
        return str(result["_id"])

    def search_semantic(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        kind: Optional[str] = None,
        *,
        reason: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        flt: dict[str, Any] = {"user_id": user_id}
        if kind:
            flt["kind"] = kind
        results = vector_search(
            SEMANTIC_COLLECTION,
            embed_query(query),
            limit=limit,
            extra_filter=flt,
            projection={"kind": 1, "key": 1, "value": 1, "content": 1, "metadata": 1},
        )
        self._emit(
            op="read",
            memory_type="semantic",
            description=reason or f"query={query!r} kind={kind or 'any'}",
            latency_ms=int((time.time() - t0) * 1000),
            result_count=len(results),
            data=results,
        )
        return results

    def list_semantic(
        self, user_id: str, kind: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        q: dict[str, Any] = {"user_id": user_id}
        if kind:
            q["kind"] = kind
        cursor = get_collection(SEMANTIC_COLLECTION).find(q, {"embedding": 0}).limit(limit)
        out = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out

    # ── procedural memory ──────────────────────────────────────────────

    def upsert_procedure(
        self, proc: ProceduralMemory, *, reason: Optional[str] = None
    ) -> str:
        t0 = time.time()
        clean_doc = proc.model_dump()
        doc = {**clean_doc, "embedding": embed_document(
            f"{proc.name}\n{proc.description}\n" + " ".join(proc.trigger_examples)
        )}
        result = get_collection(PROCEDURAL_COLLECTION).find_one_and_update(
            {"user_id": proc.user_id, "name": proc.name},
            {"$set": doc},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        clean_doc["_id"] = str(result["_id"])
        self._emit(
            op="write",
            memory_type="procedural",
            description=reason or f"workflow:{proc.name}",
            latency_ms=int((time.time() - t0) * 1000),
            data=clean_doc,
        )
        return str(result["_id"])

    def search_procedures(
        self,
        user_id: str,
        query: str,
        limit: int = 3,
        *,
        reason: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        results = vector_search(
            PROCEDURAL_COLLECTION,
            embed_query(query),
            limit=limit,
            extra_filter={"user_id": user_id},
            projection={
                "name": 1,
                "description": 1,
                "steps": 1,
                "trigger_examples": 1,
                "tags": 1,
            },
        )
        self._emit(
            op="read",
            memory_type="procedural",
            description=reason or f"query={query!r}",
            latency_ms=int((time.time() - t0) * 1000),
            result_count=len(results),
            data=results,
        )
        return results

    def list_procedures(self, user_id: str) -> list[dict[str, Any]]:
        cursor = get_collection(PROCEDURAL_COLLECTION).find(
            {"user_id": user_id}, {"embedding": 0}
        )
        out = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out

    # ── shared memory ──────────────────────────────────────────────────

    # Slots whose semantics are an append-only log unless the caller asks
    # otherwise. Everything else replaces in place (the historical default).
    _APPEND_DEFAULT_SLOTS = frozenset({"findings"})

    def write_shared(
        self,
        item: SharedMemory,
        mode: Optional[str] = None,
        *,
        reason: Optional[str] = None,
    ) -> str:
        """Write a shared-memory doc.

        `mode='replace'` upserts on the slot's identity keys (the legacy
        behavior — one doc per slot). `mode='append'` always inserts a
        new doc, building a history. When `mode` is None the slot's
        default is used (`findings` defaults to append).
        """
        if mode is None:
            mode = "append" if item.slot in self._APPEND_DEFAULT_SLOTS else "replace"
        if mode not in ("replace", "append"):
            raise ValueError(f"mode must be 'replace' or 'append', got {mode!r}")

        t0 = time.time()
        doc = item.model_dump()
        coll = get_collection(SHARED_COLLECTION)
        if mode == "append":
            new_id = str(coll.insert_one(doc).inserted_id)
        else:
            flt: dict[str, Any] = {"slot": item.slot, "scope": item.scope}
            if item.scope == "project":
                flt["project_key"] = item.project_key
            else:
                flt["session_id"] = item.session_id
            result = coll.find_one_and_update(
                flt,
                {"$set": doc},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            new_id = str(result["_id"])

        scope_tag = item.scope if item.scope != "session" else ""
        doc["_id"] = new_id
        self._emit(
            op="write",
            memory_type="shared",
            description=reason or (
                f"slot={item.slot} {item.from_agent}→{item.to_agent}"
                f" mode={mode}" + (f" {scope_tag}" if scope_tag else "")
            ),
            latency_ms=int((time.time() - t0) * 1000),
            session_id=item.session_id,
            agent=item.from_agent,
            data=doc,
        )
        return new_id

    def read_shared(
        self,
        session_id: Optional[str],
        slot: Optional[str] = None,
        *,
        project_key: Optional[str] = None,
        limit: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Read shared-memory slots, newest first.

        Always returns session-scoped docs for `session_id`. If
        `project_key` is provided, also includes any project-scoped docs
        (e.g. strategic goals) for that project. `limit` caps the result
        count and is useful when the caller only wants the latest entry
        per slot.
        """
        t0 = time.time()
        or_clauses: list[dict[str, Any]] = []
        if session_id:
            or_clauses.append({"scope": "session", "session_id": session_id})
        if project_key:
            or_clauses.append({"scope": "project", "project_key": project_key})
        if not or_clauses:
            return []
        q: dict[str, Any] = {"$or": or_clauses} if len(or_clauses) > 1 else or_clauses[0]
        if slot:
            q["slot"] = slot
        cursor = get_collection(SHARED_COLLECTION).find(q).sort("created_at", -1)
        if limit:
            cursor = cursor.limit(limit)
        results = list(cursor)
        for r in results:
            r["_id"] = str(r["_id"])
        self._emit(
            op="read",
            memory_type="shared",
            description=reason or (
                f"slot={slot or 'any'}"
                + (f" project={project_key}" if project_key else "")
            ),
            latency_ms=int((time.time() - t0) * 1000),
            result_count=len(results),
            session_id=session_id,
            data=results,
        )
        return results

    def clear_shared(
        self,
        session_id: Optional[str] = None,
        *,
        project_key: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        flt: dict[str, Any] = {}
        if session_id:
            flt["session_id"] = session_id
        if project_key:
            flt["project_key"] = project_key
        if not flt:
            return
        get_collection(SHARED_COLLECTION).delete_many(flt)
        self._emit(
            op="delete",
            memory_type="shared",
            description=reason or (
                f"cleared session={session_id}"
                + (f" project={project_key}" if project_key else "")
            ),
            session_id=session_id,
        )

    # ── inspector ──────────────────────────────────────────────────────

    def counts(self) -> dict[str, int]:
        return {
            "working": get_collection(WORKING_COLLECTION).count_documents({}),
            "episodic": get_collection(EPISODIC_COLLECTION).count_documents({}),
            "semantic": get_collection(SEMANTIC_COLLECTION).count_documents({}),
            "procedural": get_collection(PROCEDURAL_COLLECTION).count_documents({}),
            "shared": get_collection(SHARED_COLLECTION).count_documents({}),
        }
