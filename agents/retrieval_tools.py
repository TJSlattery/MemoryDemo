"""Read-only tools exposed to the Retrieval Agent."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from agents.runtime import get_memory_manager, get_session_id, get_user_id
from memory.db import (
    CALENDAR_COLLECTION,
    JIRA_COLLECTION,
    PROJECTS_COLLECTION,
    get_collection,
)
from memory.schemas import SharedMemory


@tool
def recall_session_state() -> dict:
    """Read the working-memory snapshot for the current session: what the PM
    is currently focused on, the active project/task, and any scratchpad notes."""
    return get_memory_manager().read_session_context(get_session_id())


@tool
def read_handoff(
    slot: Optional[str] = None,
    project_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Read the inter-agent shared-memory slots for the current session
    (payloads passed between Coordinator / Retrieval / Writer), newest first.

    Optional `slot` filters to one of: last_search_results, handoff_payload,
    disambiguation, scratch, plan, findings, goal. Pass `project_key` to also
    pick up any long-lived project-scoped slots (e.g. strategic goals). Pass
    `limit` to cap how many entries are returned."""
    return get_memory_manager().read_shared(
        get_session_id(), slot=slot, project_key=project_key, limit=limit,
    )


@tool
def share_with(
    slot: str,
    to_agent: str,
    payload: dict[str, Any],
    mode: Optional[str] = None,
) -> str:
    """Post a structured payload to the inter-agent shared-memory slot for
    this session so the next agent can consume it verbatim (no prose loss).

    Use `slot='findings'` (defaults to append mode) when handing structured
    rows — tickets, risks, owners — to the Writer so it can act on the exact
    IDs you found. Use `slot='disambiguation'` when you found multiple
    candidates and need the Coordinator to pick one. Other slots:
    last_search_results, handoff_payload, scratch, plan, goal.
    `mode` is 'replace' (overwrite the slot) or 'append' (push onto history)."""
    item = SharedMemory(
        session_id=get_session_id(),
        slot=slot,  # type: ignore[arg-type]
        from_agent="retrieval",
        to_agent=to_agent,
        payload=payload,
    )
    return get_memory_manager().write_shared(item, mode=mode)


@tool
def search_facts(query: str, kind: Optional[str] = None, limit: int = 5) -> list[dict]:
    """Vector-search semantic memory (long-term facts: people, projects,
    glossary, decisions, risks, preferences, epics, features, stories).
    Pass `kind` to scope to one of: person, team, project, stakeholder,
    glossary, decision, risk, preference, epic, feature, story."""
    return get_memory_manager().search_semantic(
        get_user_id(), query, limit=limit, kind=kind
    )


@tool
def search_history(query: str, limit: int = 5) -> list[dict]:
    """Vector-search episodic memory (the project timeline: tickets created,
    decisions logged, standups, status reports, sprints planned, risks)."""
    return get_memory_manager().search_episodes(get_user_id(), query, limit=limit)


@tool
def recent_history(event_type: Optional[str] = None, limit: int = 10) -> list[dict]:
    """List the most recent episodic events. Optional `event_type` filters to
    one of: ticket_created, calendar_invite, story_created, feature_created,
    epic_created, task_status_changed, decision_logged, risk_logged, standup,
    status_report, sprint_planned, conversation_summary, note."""
    return get_memory_manager().recent_episodes(
        get_user_id(), limit=limit, event_type=event_type
    )


@tool
def find_workflow(query: str, limit: int = 3) -> list[dict]:
    """Vector-search procedural memory for a workflow / template that
    matches the user's request (e.g. 'how do I run the standup?')."""
    return get_memory_manager().search_procedures(get_user_id(), query, limit=limit)


@tool
def list_workflows() -> list[dict]:
    """List every procedural workflow stored for this user (name + description)."""
    procs = get_memory_manager().list_procedures(get_user_id())
    return [
        {"name": p["name"], "description": p["description"], "tags": p.get("tags", [])}
        for p in procs
    ]


@tool
def search_all(query: str, limit_per_type: int = 3) -> dict:
    """Fan-out vector search across episodic + semantic + procedural memory.
    Use this only for genuinely open-ended 'what do you know about X' questions."""
    mm = get_memory_manager()
    user = get_user_id()
    return {
        "semantic": mm.search_semantic(user, query, limit=limit_per_type),
        "episodic": mm.search_episodes(user, query, limit=limit_per_type),
        "procedural": mm.search_procedures(user, query, limit=limit_per_type),
    }


@tool
def list_jira_tickets(
    status: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 25,
) -> list[dict]:
    """List mock Jira tickets. Optional filters: `status` (e.g. 'In Progress',
    'In Review', 'Done', 'To Do') and `project` (e.g. 'PROJ-ATLAS')."""
    q: dict = {}
    if status:
        q["status"] = status
    if project:
        q["project"] = project
    cursor = (
        get_collection(JIRA_COLLECTION)
        .find(q, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )
    return list(cursor)


@tool
def list_calendar_events(limit: int = 10) -> list[dict]:
    """List upcoming mock calendar events (meetings, dry-runs, kickoffs)."""
    cursor = (
        get_collection(CALENDAR_COLLECTION)
        .find({}, {"_id": 0})
        .sort("start", 1)
        .limit(limit)
    )
    return list(cursor)


@tool
def list_projects() -> list[dict]:
    """List all known projects (name, key, summary)."""
    cursor = get_collection(PROJECTS_COLLECTION).find({}, {"_id": 0})
    return list(cursor)


RETRIEVAL_TOOLS = [
    recall_session_state,
    read_handoff,
    share_with,
    search_facts,
    search_history,
    recent_history,
    find_workflow,
    list_workflows,
    search_all,
    list_jira_tickets,
    list_calendar_events,
    list_projects,
]
