"""Single source of truth for collection names + Mongo client wiring."""

from __future__ import annotations

import os
from functools import lru_cache

from pymongo import MongoClient

DB_NAME = os.getenv("PM_DB_NAME", "pm_agent_memory")

# Memory collections
WORKING_COLLECTION = "working_memory"
EPISODIC_COLLECTION = "episodic_memory"
SEMANTIC_COLLECTION = "semantic_memory"
PROCEDURAL_COLLECTION = "procedural_memory"
SHARED_COLLECTION = "shared_memory"

# Mock business collections
JIRA_COLLECTION = "mock_jira_tickets"
CALENDAR_COLLECTION = "mock_calendar_events"
MILESTONES_COLLECTION = "mock_milestones"
PROJECTS_COLLECTION = "projects"
TASKS_COLLECTION = "tasks"

# LangGraph checkpointer collections (created automatically by MongoDBSaver,
# but we list them so /reset can clear them too).
CHECKPOINT_COLLECTIONS = ("checkpoints_aio", "checkpoint_writes_aio")

ALL_MEMORY_COLLECTIONS = (
    WORKING_COLLECTION,
    EPISODIC_COLLECTION,
    SEMANTIC_COLLECTION,
    PROCEDURAL_COLLECTION,
    SHARED_COLLECTION,
)

ALL_BUSINESS_COLLECTIONS = (
    JIRA_COLLECTION,
    CALENDAR_COLLECTION,
    MILESTONES_COLLECTION,
    PROJECTS_COLLECTION,
    TASKS_COLLECTION,
)


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Point it at your Atlas cluster "
            "(mongodb+srv://...) in the .env file."
        )
    return MongoClient(uri, appname="pm-memory-demo")


def get_db():
    return get_client()[DB_NAME]


def get_collection(name: str):
    return get_db()[name]
