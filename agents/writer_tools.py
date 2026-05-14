"""Write-only tools exposed to the Writer Agent.

Every business write also records a matching episodic event so the
timeline reflects what happened.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import tool

from agents.runtime import get_memory_manager, get_session_id, get_user_id
from memory.db import CALENDAR_COLLECTION, JIRA_COLLECTION, get_collection
from memory.schemas import (
    EpisodicMemory,
    ProceduralMemory,
    ProceduralStep,
    SemanticMemory,
    SharedMemory,
)


# ── memory writes ─────────────────────────────────────────────────────────


@tool
def update_focus(
    current_project: Optional[str] = None,
    current_task: Optional[str] = None,
    focus: Optional[str] = None,
    last_action: Optional[str] = None,
) -> dict:
    """Update working memory: what the PM is focused on right now."""
    return get_memory_manager().update_session_context(
        get_session_id(),
        get_user_id(),
        {
            "current_project": current_project,
            "current_task": current_task,
            "focus": focus,
            "last_action": last_action,
        },
    )


@tool
def note_to_self(text: str) -> str:
    """Append a short note to the working-memory scratchpad for this session."""
    mm = get_memory_manager()
    sid = get_session_id()
    current = mm.read_session_context(sid).get("scratchpad", []) or []
    current.append(text)
    mm.update_session_context(sid, get_user_id(), {"scratchpad": current})
    return f"Noted ({len(current)} item(s) in scratchpad)."


@tool
def log_event(
    event_type: str,
    summary: str,
    entities: Optional[dict[str, Any]] = None,
) -> str:
    """Record a discrete event in episodic memory. `event_type` must be one of:
    ticket_created, calendar_invite, story_created, feature_created,
    epic_created, task_status_changed, decision_logged, risk_logged,
    standup, status_report, sprint_planned, conversation_summary, note."""
    ep = EpisodicMemory(
        user_id=get_user_id(),
        event_type=event_type,  # type: ignore[arg-type]
        summary=summary,
        entities=entities or {},
    )
    return get_memory_manager().record_episode(ep)


@tool
def remember_fact(kind: str, key: str, value: str, content: str) -> str:
    """Upsert a long-term fact in semantic memory. `kind` must be one of:
    person, team, project, stakeholder, glossary, decision, risk, preference.
    `key` is a stable identifier (e.g. 'jane.doe'); `value` is the display
    form; `content` is the full prose used for vector search."""
    fact = SemanticMemory(
        user_id=get_user_id(),
        kind=kind,  # type: ignore[arg-type]
        key=key,
        value=value,
        content=content,
    )
    return get_memory_manager().upsert_semantic(fact)


@tool
def save_workflow(
    name: str,
    description: str,
    steps: list[str],
    trigger_examples: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Save (or overwrite) a procedural workflow. `steps` is an ordered list
    of action strings; each becomes one numbered step."""
    proc = ProceduralMemory(
        user_id=get_user_id(),
        name=name,
        description=description,
        steps=[ProceduralStep(step=i + 1, action=s) for i, s in enumerate(steps)],
        trigger_examples=trigger_examples or [],
        tags=tags or [],
    )
    return get_memory_manager().upsert_procedure(proc)


@tool
def share_with(
    slot: str,
    to_agent: str,
    payload: dict[str, Any],
    mode: Optional[str] = None,
    scope: str = "session",
    project_key: Optional[str] = None,
) -> str:
    """Drop a payload into the inter-agent shared-memory slot for this session.

    `slot` must be one of: last_search_results, handoff_payload, disambiguation,
    scratch, plan, findings, goal.
    `mode` is 'replace' (overwrite the slot in place) or 'append' (push a new
    doc onto the slot's history). Defaults to append for `findings`, replace
    for everything else.
    Pass `scope='project'` together with `project_key` to write a long-lived
    (no-TTL) doc visible to every session for that project — use this for
    strategic goals and other project-lifetime state.
    """
    item = SharedMemory(
        session_id=None if scope == "project" else get_session_id(),
        slot=slot,  # type: ignore[arg-type]
        from_agent="writer",
        to_agent=to_agent,
        payload=payload,
        scope=scope,  # type: ignore[arg-type]
        project_key=project_key,
    )
    return get_memory_manager().write_shared(item, mode=mode)


# ── business writes (each also logs an episodic event) ────────────────────

_PRJ_RE = re.compile(r"^PRJ-(\d+)$")


def _next_prj_key() -> str:
    """Allocate the next sequential PRJ-NNN key."""
    coll = get_collection(JIRA_COLLECTION)
    keys = [d["key"] for d in coll.find({}, {"key": 1})]
    nums = [int(m.group(1)) for k in keys if (m := _PRJ_RE.match(k))]
    return f"PRJ-{(max(nums) + 1) if nums else 700}"


@tool
def create_jira_ticket(
    title: str,
    project: str,
    priority: str = "Medium",
    assignee: Optional[str] = None,
) -> dict:
    """Create a mock Jira ticket. `project` must be one of the project keys
    (PROJ-ATLAS, PROJ-MOBILE, PROJ-REPORT). Returns the new ticket dict."""
    key = _next_prj_key()
    doc = {
        "key": key,
        "title": title,
        "project": project,
        "status": "To Do",
        "priority": priority,
        "assignee": assignee,
        "created_at": datetime.utcnow(),
        "is_seed": False,
    }
    get_collection(JIRA_COLLECTION).insert_one(doc)
    get_memory_manager().record_episode(
        EpisodicMemory(
            user_id=get_user_id(),
            event_type="ticket_created",
            summary=f"Created {key}: {title}",
            entities={"ticket": key, "project": project, "assignee": assignee},
        )
    )
    doc.pop("_id", None)
    return doc


@tool
def update_jira_status(key: str, status: str) -> dict:
    """Update a mock Jira ticket's status (e.g. 'In Progress', 'Done').
    Logs a `task_status_changed` episodic event."""
    coll = get_collection(JIRA_COLLECTION)
    before = coll.find_one({"key": key}, {"status": 1, "title": 1})
    if not before:
        return {"error": f"Ticket {key} not found."}
    coll.update_one({"key": key}, {"$set": {"status": status}})
    get_memory_manager().record_episode(
        EpisodicMemory(
            user_id=get_user_id(),
            event_type="task_status_changed",
            summary=f"{key} moved {before.get('status')} → {status}",
            entities={"ticket": key, "from": before.get("status"), "to": status},
        )
    )
    return {"key": key, "status": status, "previous": before.get("status")}


@tool
def create_calendar_event(
    title: str,
    start_iso: str,
    duration_minutes: int = 60,
    attendees: Optional[list[str]] = None,
    project: Optional[str] = None,
) -> dict:
    """Create a mock calendar event. `start_iso` is an ISO-8601 timestamp."""
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    doc = {
        "title": title,
        "start": start,
        "duration_minutes": duration_minutes,
        "attendees": attendees or [],
        "project": project,
        "created_at": datetime.utcnow(),
        "is_seed": False,
    }
    get_collection(CALENDAR_COLLECTION).insert_one(doc)
    get_memory_manager().record_episode(
        EpisodicMemory(
            user_id=get_user_id(),
            event_type="calendar_invite",
            summary=f"Scheduled '{title}' for {start.isoformat()} ({duration_minutes}m)",
            entities={"title": title, "project": project, "attendees": attendees or []},
        )
    )
    doc.pop("_id", None)
    doc["start"] = start.isoformat()
    return doc


@tool
def log_decision(title: str, rationale: str, project: Optional[str] = None) -> str:
    """Log a decision: writes a semantic fact (kind=decision) AND an episodic
    `decision_logged` event in one call."""
    mm = get_memory_manager()
    user = get_user_id()
    key = "dec_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    mm.upsert_semantic(SemanticMemory(
        user_id=user, kind="decision", key=key, value=title,
        content=f"{title}. {rationale}" + (f" Project: {project}." if project else ""),
        metadata={"project": project} if project else {},
    ))
    mm.record_episode(EpisodicMemory(
        user_id=user, event_type="decision_logged",
        summary=f"Decision: {title}",
        entities={"decision": key, "project": project},
    ))
    return f"Logged decision '{key}'."


@tool
def log_risk(
    title: str,
    description: str,
    project: Optional[str] = None,
    owner: Optional[str] = None,
) -> str:
    """Log a risk: writes a semantic fact (kind=risk) AND an episodic
    `risk_logged` event in one call."""
    mm = get_memory_manager()
    user = get_user_id()
    key = "risk_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    mm.upsert_semantic(SemanticMemory(
        user_id=user, kind="risk", key=key, value=title,
        content=f"{title}. {description}" + (f" Owner: {owner}." if owner else ""),
        metadata={"project": project, "owner": owner},
    ))
    mm.record_episode(EpisodicMemory(
        user_id=user, event_type="risk_logged",
        summary=f"Risk: {title}",
        entities={"risk": key, "project": project, "owner": owner},
    ))
    return f"Logged risk '{key}'."


def _slug(prefix: str, title: str) -> str:
    return prefix + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]


@tool
def create_epic(name: str, description: str, project: str) -> str:
    """Create an epic: a large, multi-feature initiative scoped to a project.
    Writes a semantic fact (kind=epic) AND an episodic `epic_created` event."""
    mm = get_memory_manager()
    user = get_user_id()
    key = _slug("epic_", name)
    mm.upsert_semantic(SemanticMemory(
        user_id=user, kind="epic", key=key, value=name,
        content=f"{name} ({project}). {description}",
        metadata={"project": project},
    ))
    mm.record_episode(EpisodicMemory(
        user_id=user, event_type="epic_created",
        summary=f"Epic created: {name}",
        entities={"epic": key, "project": project},
    ))
    return f"Created epic '{key}'."


@tool
def create_feature(
    name: str,
    description: str,
    project: str,
    parent_epic: Optional[str] = None,
) -> str:
    """Create a feature: a shippable capability under an epic.
    Writes a semantic fact (kind=feature) AND an episodic `feature_created` event.
    `parent_epic` is the key of the owning epic (e.g. 'epic_atlas_data_parity')."""
    mm = get_memory_manager()
    user = get_user_id()
    key = _slug("feat_", name)
    mm.upsert_semantic(SemanticMemory(
        user_id=user, kind="feature", key=key, value=name,
        content=f"{name} ({project}). {description}",
        metadata={"project": project, "epic": parent_epic},
    ))
    mm.record_episode(EpisodicMemory(
        user_id=user, event_type="feature_created",
        summary=f"Feature created: {name}",
        entities={"feature": key, "project": project, "epic": parent_epic},
    ))
    return f"Created feature '{key}'."


@tool
def create_user_story(
    persona: str,
    capability: str,
    benefit: str,
    project: str,
    acceptance_criteria: list[str],
    parent_feature: Optional[str] = None,
) -> str:
    """Create a user story in standard form: 'As a <persona>, I want <capability>,
    so that <benefit>'. `acceptance_criteria` is a list of Given/When/Then
    statements. Writes a semantic fact (kind=story) AND an episodic
    `story_created` event."""
    mm = get_memory_manager()
    user = get_user_id()
    title = f"As a {persona}, I want {capability}, so that {benefit}."
    key = _slug("story_", f"{persona} {capability}")
    body = title + " Acceptance criteria: " + " | ".join(acceptance_criteria)
    mm.upsert_semantic(SemanticMemory(
        user_id=user, kind="story", key=key, value=title,
        content=f"{body} ({project})",
        metadata={
            "project": project,
            "feature": parent_feature,
            "acceptance_criteria": acceptance_criteria,
        },
    ))
    mm.record_episode(EpisodicMemory(
        user_id=user, event_type="story_created",
        summary=f"Story created: {persona} \u2192 {capability}",
        entities={"story": key, "project": project, "feature": parent_feature},
    ))
    return f"Created story '{key}'."


WRITER_TOOLS = [
    update_focus,
    note_to_self,
    log_event,
    remember_fact,
    save_workflow,
    share_with,
    create_jira_ticket,
    update_jira_status,
    create_calendar_event,
    log_decision,
    log_risk,
    create_epic,
    create_feature,
    create_user_story,
]
