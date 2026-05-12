"""
MongoDB-backed Chainlit data layer.

Persists Chainlit users, threads, steps, feedbacks, and elements in MongoDB
so the same connection used for agent memory powers the UI's thread history.

Collections (prefixed cl_ to avoid colliding with agent state):
    cl_users, cl_threads, cl_steps, cl_elements, cl_feedbacks
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    FeedbackDict,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().isoformat() + "Z"


class MongoDBDataLayer(BaseDataLayer):
    """Chainlit data layer backed by MongoDB."""

    def __init__(
        self,
        mongodb_uri: str,
        db_name: str = "agent_memory_simple",
        collection_prefix: str = "cl_",
        user_thread_limit: int = 1000,
    ):
        self._client = AsyncMongoClient(mongodb_uri)
        db = self._client[db_name]
        self.users = db[f"{collection_prefix}users"]
        self.threads = db[f"{collection_prefix}threads"]
        self.steps = db[f"{collection_prefix}steps"]
        self.elements = db[f"{collection_prefix}elements"]
        self.feedbacks = db[f"{collection_prefix}feedbacks"]
        self.user_thread_limit = user_thread_limit
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        if self._indexes_created:
            return
        await self.users.create_index("identifier", unique=True)
        await self.threads.create_index("userId")
        await self.threads.create_index([("updatedAt", DESCENDING)])
        await self.steps.create_index(
            [("threadId", ASCENDING), ("createdAt", ASCENDING)]
        )
        await self.steps.create_index("parentId")
        await self.elements.create_index("threadId")
        await self.elements.create_index("forId")
        await self.feedbacks.create_index("forId")
        await self.feedbacks.create_index("threadId")
        self._indexes_created = True

    # ---------- Users ----------
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        await self._ensure_indexes()
        doc = await self.users.find_one({"identifier": identifier})
        if not doc:
            return None
        return PersistedUser(
            id=doc["_id"],
            identifier=doc["identifier"],
            createdAt=doc["createdAt"],
            metadata=doc.get("metadata") or {},
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        existing = await self.get_user(user.identifier)
        if existing:
            await self.users.update_one(
                {"identifier": user.identifier},
                {"$set": {"metadata": user.metadata or {}}},
            )
        else:
            await self.users.insert_one(
                {
                    "_id": str(uuid.uuid4()),
                    "identifier": user.identifier,
                    "createdAt": _now(),
                    "metadata": user.metadata or {},
                }
            )
        return await self.get_user(user.identifier)

    async def _get_user_identifier_by_id(self, user_id: str) -> Optional[str]:
        doc = await self.users.find_one({"_id": user_id}, {"identifier": 1})
        return doc["identifier"] if doc else None

    # ---------- Threads ----------
    async def get_thread_author(self, thread_id: str) -> str:
        doc = await self.threads.find_one(
            {"_id": thread_id}, {"userIdentifier": 1}
        )
        if doc and doc.get("userIdentifier"):
            return doc["userIdentifier"]
        raise ValueError(f"Author not found for thread_id {thread_id}")

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        threads = await self._build_thread_dicts(thread_id=thread_id)
        return threads[0] if threads else None

    async def delete_thread(self, thread_id: str) -> None:
        await self.feedbacks.delete_many({"threadId": thread_id})
        await self.elements.delete_many({"threadId": thread_id})
        await self.steps.delete_many({"threadId": thread_id})
        await self.threads.delete_one({"_id": thread_id})

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        await self._ensure_indexes()
        user_identifier = None
        if user_id:
            user_identifier = await self._get_user_identifier_by_id(user_id)

        has_updates = (
            metadata is not None
            or name is not None
            or user_id is not None
            or tags is not None
        )

        existing = await self.threads.find_one({"_id": thread_id}, {"metadata": 1})
        if existing and not has_updates:
            return

        incoming_meta = metadata or {}
        base_meta = (existing or {}).get("metadata") or {}
        to_delete = {k for k, v in incoming_meta.items() if v is None}
        merged_meta = {
            **{k: v for k, v in base_meta.items() if k not in to_delete},
            **{k: v for k, v in incoming_meta.items() if v is not None},
        }

        name_value = name if name is not None else merged_meta.get("name")

        set_fields: Dict[str, Any] = {"updatedAt": _now()}
        if name_value is not None:
            set_fields["name"] = name_value
        if user_id is not None:
            set_fields["userId"] = user_id
        if user_identifier is not None:
            set_fields["userIdentifier"] = user_identifier
        if tags is not None:
            set_fields["tags"] = tags
        if metadata is not None or merged_meta:
            set_fields["metadata"] = merged_meta

        update_doc: Dict[str, Any] = {"$set": set_fields}
        if not existing:
            update_doc["$setOnInsert"] = {"createdAt": _now()}

        await self.threads.update_one({"_id": thread_id}, update_doc, upsert=True)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse:
        if not filters.userId:
            raise ValueError("userId is required")
        all_threads = await self._build_thread_dicts(user_id=filters.userId) or []

        search_keyword = filters.search.lower() if filters.search else None
        feedback_value = int(filters.feedback) if filters.feedback is not None else None

        filtered: List[ThreadDict] = []
        for thread in all_threads:
            keyword_match = True
            feedback_match = True
            if search_keyword:
                keyword_match = any(
                    search_keyword in (step.get("output") or "").lower()
                    for step in thread["steps"]
                )
            if feedback_value is not None:
                feedback_match = any(
                    (step.get("feedback") or {}).get("value") == feedback_value
                    for step in thread["steps"]
                )
            if keyword_match and feedback_match:
                filtered.append(thread)

        start = 0
        if pagination.cursor:
            for i, thread in enumerate(filtered):
                if thread["id"] == pagination.cursor:
                    start = i + 1
                    break
        end = start + pagination.first
        page = filtered[start:end] or []

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=len(filtered) > end,
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=page,
        )

    # ---------- Steps ----------
    @queue_until_user_message()
    async def create_step(self, step_dict: StepDict) -> None:
        await self._ensure_indexes()
        thread_id = step_dict.get("threadId")
        if thread_id:
            await self.update_thread(thread_id)

        show_input = step_dict.get("showInput")
        normalized_show_input = (
            str(show_input).lower() if show_input is not None else None
        )

        doc: Dict[str, Any] = {
            k: v
            for k, v in step_dict.items()
            if v is not None and not (isinstance(v, dict) and not v) and k != "id"
        }
        doc["metadata"] = step_dict.get("metadata") or {}
        doc["generation"] = step_dict.get("generation") or {}
        if normalized_show_input is not None:
            doc["showInput"] = normalized_show_input

        step_id = step_dict.get("id")
        if not step_id:
            return
        await self.steps.update_one(
            {"_id": step_id},
            {"$set": doc, "$setOnInsert": {"_id": step_id}},
            upsert=True,
        )

    @queue_until_user_message()
    async def update_step(self, step_dict: StepDict) -> None:
        await self.create_step(step_dict)

    @queue_until_user_message()
    async def delete_step(self, step_id: str) -> None:
        await self.feedbacks.delete_many({"forId": step_id})
        await self.elements.delete_many({"forId": step_id})
        await self.steps.delete_one({"_id": step_id})

    # ---------- Feedback ----------
    async def upsert_feedback(self, feedback: Feedback) -> str:
        await self._ensure_indexes()
        feedback.id = feedback.id or str(uuid.uuid4())
        doc = {
            "forId": feedback.forId,
            "threadId": feedback.threadId,
            "value": feedback.value,
            "comment": feedback.comment,
        }
        await self.feedbacks.update_one(
            {"_id": feedback.id},
            {"$set": doc, "$setOnInsert": {"_id": feedback.id}},
            upsert=True,
        )
        return feedback.id

    async def delete_feedback(self, feedback_id: str) -> bool:
        await self.feedbacks.delete_one({"_id": feedback_id})
        return True

    # ---------- Elements ----------
    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[ElementDict]:
        doc = await self.elements.find_one(
            {"_id": element_id, "threadId": thread_id}
        )
        return self._element_doc_to_dict(doc) if doc else None

    @queue_until_user_message()
    async def create_element(self, element: Element) -> None:
        await self._ensure_indexes()
        if not element.for_id:
            return
        element_dict: ElementDict = element.to_dict()
        doc = {k: v for k, v in element_dict.items() if v is not None and k != "id"}
        await self.elements.update_one(
            {"_id": element.id},
            {"$set": doc, "$setOnInsert": {"_id": element.id}},
            upsert=True,
        )

    @queue_until_user_message()
    async def delete_element(
        self, element_id: str, thread_id: Optional[str] = None
    ) -> None:
        query: Dict[str, Any] = {"_id": element_id}
        if thread_id:
            query["threadId"] = thread_id
        await self.elements.delete_one(query)

    @staticmethod
    def _element_doc_to_dict(doc: Dict[str, Any]) -> ElementDict:
        return ElementDict(
            id=doc["_id"],
            threadId=doc.get("threadId"),
            type=doc.get("type"),
            chainlitKey=doc.get("chainlitKey"),
            url=doc.get("url"),
            objectKey=doc.get("objectKey"),
            name=doc.get("name"),
            display=doc.get("display"),
            size=doc.get("size"),
            language=doc.get("language"),
            page=doc.get("page"),
            props=doc.get("props"),
            autoPlay=doc.get("autoPlay"),
            playerConfig=doc.get("playerConfig"),
            forId=doc.get("forId"),
            mime=doc.get("mime"),
        )

    @staticmethod
    def _step_doc_to_dict(
        doc: Dict[str, Any], feedback: Optional[FeedbackDict] = None
    ) -> StepDict:
        show_input = doc.get("showInput")
        return StepDict(
            id=doc["_id"],
            name=doc.get("name", ""),
            type=doc.get("type"),
            threadId=doc.get("threadId"),
            parentId=doc.get("parentId"),
            streaming=doc.get("streaming", False),
            waitForAnswer=doc.get("waitForAnswer"),
            isError=doc.get("isError"),
            metadata=doc.get("metadata") or {},
            tags=doc.get("tags"),
            input=(
                doc.get("input", "")
                if show_input not in [None, "false"]
                else ""
            ),
            output=doc.get("output", ""),
            createdAt=doc.get("createdAt"),
            start=doc.get("start"),
            end=doc.get("end"),
            generation=doc.get("generation"),
            showInput=show_input,
            language=doc.get("language"),
            feedback=feedback,
        )

    # ---------- Aggregations ----------
    async def _build_thread_dicts(
        self,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[ThreadDict]:
        await self._ensure_indexes()
        thread_query: Dict[str, Any] = {}
        if thread_id:
            thread_query["_id"] = thread_id
        if user_id:
            thread_query["userId"] = user_id

        cursor = self.threads.find(thread_query).sort(
            [("updatedAt", DESCENDING), ("createdAt", DESCENDING)]
        )
        if user_id and not thread_id:
            cursor = cursor.limit(self.user_thread_limit)

        thread_docs = await cursor.to_list(length=None)
        if not thread_docs:
            return []

        thread_ids = [t["_id"] for t in thread_docs]

        step_docs = await self.steps.find(
            {"threadId": {"$in": thread_ids}}
        ).sort("createdAt", ASCENDING).to_list(length=None)

        step_ids = [s["_id"] for s in step_docs]
        feedback_docs = await self.feedbacks.find(
            {"forId": {"$in": step_ids}}
        ).to_list(length=None) if step_ids else []
        feedbacks_by_step = {
            f["forId"]: FeedbackDict(
                forId=f["forId"],
                id=f["_id"],
                value=f.get("value"),
                comment=f.get("comment"),
            )
            for f in feedback_docs
        }

        element_docs = await self.elements.find(
            {"threadId": {"$in": thread_ids}}
        ).to_list(length=None)

        thread_dicts: Dict[str, ThreadDict] = {}
        for t in thread_docs:
            thread_dicts[t["_id"]] = ThreadDict(
                id=t["_id"],
                createdAt=t.get("createdAt"),
                name=t.get("name"),
                userId=t.get("userId"),
                userIdentifier=t.get("userIdentifier"),
                tags=t.get("tags"),
                metadata=t.get("metadata") or {},
                steps=[],
                elements=[],
            )

        for s in step_docs:
            tid = s.get("threadId")
            if tid in thread_dicts:
                thread_dicts[tid]["steps"].append(
                    self._step_doc_to_dict(s, feedbacks_by_step.get(s["_id"]))
                )

        for e in element_docs:
            tid = e.get("threadId")
            if tid in thread_dicts:
                thread_dicts[tid]["elements"].append(self._element_doc_to_dict(e))

        return list(thread_dicts.values())

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        await self._ensure_indexes()
        thread_docs = await self.threads.find(
            {"userId": user_id}, {"_id": 1}
        ).to_list(length=None)
        thread_ids = [t["_id"] for t in thread_docs]
        if not thread_ids:
            return []
        step_docs = await self.steps.find(
            {"threadId": {"$in": thread_ids}, "metadata.favorite": True}
        ).sort("createdAt", DESCENDING).to_list(length=None)
        return [self._step_doc_to_dict(s) for s in step_docs]

    # ---------- Misc ----------
    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        await self._client.close()

