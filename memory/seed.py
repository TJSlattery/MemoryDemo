"""Load the Northwind Robotics demo dataset into MongoDB Atlas.

Idempotent: every record is upserted on a stable key, so re-running
overwrites in place rather than duplicating. Existing non-seed records
(`is_seed=False`) are left untouched.

Usage:
    python -m memory.seed                # seed only
    python -m memory.seed --bootstrap    # bootstrap indexes first
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Optional

from memory.bootstrap import bootstrap_indexes
from memory.db import (
    CALENDAR_COLLECTION,
    JIRA_COLLECTION,
    MILESTONES_COLLECTION,
    PROJECTS_COLLECTION,
    get_collection,
)
from memory.manager import MemoryManager
from memory.schemas import (
    EpisodicMemory,
    ProceduralMemory,
    ProceduralStep,
    SemanticMemory,
    SharedMemory,
)
from memory.seed_data import (
    CALENDAR_EVENTS,
    DECISIONS,
    EPICS,
    EPISODES,
    FEATURES,
    GLOSSARY,
    JIRA_TICKETS,
    MILESTONES,
    PEOPLE,
    PERSONA,
    PREFERENCES,
    PROCEDURES,
    PROJECT_SHARED_SEED,
    PROJECTS,
    RISKS,
    SHARED_SEED,
    STAKEHOLDERS,
    STORIES,
    WORKING_SEED,
)

logger = logging.getLogger(__name__)


def _semantic(kind: str, key: str, value: str, content: str) -> SemanticMemory:
    return SemanticMemory(
        user_id=PERSONA["user_id"],
        kind=kind,
        key=key,
        value=value,
        content=content,
        source="seed",
        is_seed=True,
    )


def _seed_semantic(mm: MemoryManager) -> int:
    n = 0
    for key, name, blurb in PEOPLE:
        mm.upsert_semantic(_semantic("person", key, name, f"{name}. {blurb}"))
        n += 1
    for key, name, blurb in STAKEHOLDERS:
        mm.upsert_semantic(_semantic("stakeholder", key, name, f"{name}. {blurb}"))
        n += 1
    for key, name, blurb in PROJECTS:
        mm.upsert_semantic(_semantic("project", key, name, f"{name} ({key}). {blurb}"))
        n += 1
    for key, value, blurb in GLOSSARY:
        mm.upsert_semantic(_semantic("glossary", key, value, f"{value}: {blurb}"))
        n += 1
    for key, value, blurb in DECISIONS:
        mm.upsert_semantic(_semantic("decision", key, value, blurb))
        n += 1
    for key, value, blurb in RISKS:
        mm.upsert_semantic(_semantic("risk", key, value, blurb))
        n += 1
    for key, value, blurb in PREFERENCES:
        mm.upsert_semantic(_semantic("preference", key, value, blurb))
        n += 1
    for key, value, project, blurb in EPICS:
        fact = _semantic("epic", key, value, f"{value} ({project}). {blurb}")
        fact.metadata = {"project": project}
        mm.upsert_semantic(fact)
        n += 1
    for key, value, project, epic, blurb in FEATURES:
        fact = _semantic("feature", key, value, f"{value} ({project}). {blurb}")
        fact.metadata = {"project": project, "epic": epic}
        mm.upsert_semantic(fact)
        n += 1
    for key, value, project, feature, criteria in STORIES:
        body = value + " Acceptance criteria: " + " | ".join(criteria)
        fact = _semantic("story", key, value, f"{body} ({project})")
        fact.metadata = {"project": project, "feature": feature, "acceptance_criteria": criteria}
        mm.upsert_semantic(fact)
        n += 1
    return n


def _seed_procedural(mm: MemoryManager) -> int:
    for proc in PROCEDURES:
        mm.upsert_procedure(
            ProceduralMemory(
                user_id=PERSONA["user_id"],
                name=proc["name"],
                description=proc["description"],
                steps=[
                    ProceduralStep(step=s, action=a) for s, a in proc["steps"]
                ],
                trigger_examples=proc["trigger_examples"],
                tags=proc["tags"],
                is_seed=True,
            )
        )
    return len(PROCEDURES)


def _seed_episodic(mm: MemoryManager) -> int:
    coll = get_collection("episodic_memory")
    coll.delete_many({"user_id": PERSONA["user_id"], "is_seed": True})
    for ts, etype, summary, entities in EPISODES:
        ep = EpisodicMemory(
            user_id=PERSONA["user_id"],
            event_type=etype,
            summary=summary,
            entities=entities,
            timestamp=ts,
            source="seed",
            is_seed=True,
        )
        mm.record_episode(ep)
    return len(EPISODES)


def _seed_business() -> tuple[int, int, int, int]:
    proj = get_collection(PROJECTS_COLLECTION)
    proj.delete_many({"is_seed": True})
    proj.insert_many(
        [{"key": k, "name": n, "summary": s, "is_seed": True} for k, n, s in PROJECTS]
    )

    jira = get_collection(JIRA_COLLECTION)
    jira.delete_many({"is_seed": True})
    jira.insert_many([{**t, "is_seed": True} for t in JIRA_TICKETS])

    cal = get_collection(CALENDAR_COLLECTION)
    cal.delete_many({"is_seed": True})
    cal.insert_many(
        [{**e, "created_at": datetime.utcnow(), "is_seed": True} for e in CALENDAR_EVENTS]
    )

    ms = get_collection(MILESTONES_COLLECTION)
    ms.delete_many({"is_seed": True})
    ms.insert_many([{**m, "is_seed": True} for m in MILESTONES])
    return len(PROJECTS), len(JIRA_TICKETS), len(CALENDAR_EVENTS), len(MILESTONES)


def seed_session(
    session_id: str,
    user_id: Optional[str] = None,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Pre-populate working + shared memory for a single chat session.

    Both stores are session-scoped, so this is called from the Chainlit
    UI on chat start (once the `thread_id` is known). Idempotent: skips
    each store if it already has data for the session unless `force=True`.
    """
    user_id = user_id or PERSONA["user_id"]
    mm = MemoryManager()
    counts = {"working": 0, "shared": 0}

    if force or not mm.read_session_context(session_id):
        mm.update_session_context(session_id, user_id, dict(WORKING_SEED))
        counts["working"] = 1

    if force or not mm.read_shared(session_id):
        for entry in SHARED_SEED:
            mm.write_shared(SharedMemory(session_id=session_id, **entry))
        counts["shared"] = len(SHARED_SEED)

    return counts


def _seed_project_shared(mm: MemoryManager) -> int:
    """Seed long-lived project-scoped shared-memory entries (e.g. goals).

    Idempotent: project-scoped writes upsert on (scope, project_key, slot).
    """
    for entry in PROJECT_SHARED_SEED:
        mm.write_shared(SharedMemory(**entry), mode="replace")
    return len(PROJECT_SHARED_SEED)


def seed(do_bootstrap: bool = False) -> dict[str, int]:
    if do_bootstrap:
        bootstrap_indexes(wait_for_vectors=False)
    mm = MemoryManager()
    counts = {
        "semantic": _seed_semantic(mm),
        "procedural": _seed_procedural(mm),
        "episodic": _seed_episodic(mm),
        "project_shared": _seed_project_shared(mm),
    }
    (
        counts["projects"],
        counts["jira"],
        counts["calendar"],
        counts["milestones"],
    ) = _seed_business()
    return counts


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Seed the PM agent demo data.")
    parser.add_argument("--bootstrap", action="store_true", help="Run index bootstrap first.")
    args = parser.parse_args()
    result = seed(do_bootstrap=args.bootstrap)
    print("Seeded:")
    for k, v in result.items():
        print(f"  {k:<11} {v}")
