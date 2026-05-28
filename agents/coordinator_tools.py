"""Coordinator-level tools: handoffs to sub-agents + a small set of
direct convenience tools for headline questions.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool
from langgraph.utils.config import get_config

from agents.chart_tools import render_gantt
from agents.runtime import get_memory_manager, get_session_id, get_user_id
from memory.db import CALENDAR_COLLECTION, JIRA_COLLECTION, get_collection
from memory.schemas import SharedMemory

_retrieval_agent = None
_writer_agent = None


def _get_retrieval_agent():
    global _retrieval_agent
    if _retrieval_agent is None:
        from agents.retrieval import create_retrieval_agent
        _retrieval_agent = create_retrieval_agent()
    return _retrieval_agent


def _get_writer_agent():
    global _writer_agent
    if _writer_agent is None:
        from agents.writer import create_writer_agent
        _writer_agent = create_writer_agent()
    return _writer_agent


def _final_text(message: Any) -> str:
    """Pull a plain string out of a LangChain message (handles both
    string content and Anthropic content-block lists)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _invoke_subagent(agent, request: str) -> str:
    cfg = get_config() or {}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=cfg,
    )
    return _final_text(result["messages"][-1])


# ── handoff tools ──────────────────────────────────────────────────────────


@tool
def ask_retrieval(question: str) -> str:
    """Delegate a read-only question to the Retrieval Agent.

    Use this for ANY lookup: current focus, recent events, facts about
    people/projects/risks, applicable workflows, in-flight Jira tickets,
    upcoming calendar events. Pass a complete natural-language question
    — the sub-agent will pick the right memory tools.
    """
    return _invoke_subagent(_get_retrieval_agent(), question)


@tool
def ask_writer(instruction: str) -> str:
    """Delegate a write/action to the Writer Agent.

    Use this whenever something needs to be recorded or created: update
    focus, log an event, remember a fact, save a workflow, create a Jira
    ticket, schedule a meeting, log a decision/risk. Pass a complete
    natural-language instruction with all relevant details (titles,
    projects, assignees, timestamps).
    """
    return _invoke_subagent(_get_writer_agent(), instruction)


# ── shared-memory tools (coordinator-side coordination surface) ──────────


@tool
def post_plan(
    reason: str,
    to_agent: str,
    payload: dict[str, Any],
    slot: str = "plan",
    scope: str = "session",
    project_key: Optional[str] = None,
    mode: Optional[str] = None,
) -> str:
    """Post a plan, goal, or instruction to shared memory before delegating.

    `reason` is a one-sentence first-person rationale (≤20 words) shown in
    the memory-trace UI — explain *why* you're posting this, e.g. "I'm
    laying out the three steps so Retrieval and Writer share one plan".

    Use this to give sub-agents structured context they should consult before
    acting (e.g. the multi-step plan you've broken a request into, or a list
    of constraints). Default `slot='plan'` (session-scoped, replaces in place).
    Use `slot='goal'` with `scope='project'` and a `project_key` to record a
    long-lived strategic goal for the project that survives across sessions.
    `to_agent` is the intended consumer ('retrieval', 'writer', or 'any')."""
    item = SharedMemory(
        session_id=None if scope == "project" else get_session_id(),
        slot=slot,  # type: ignore[arg-type]
        from_agent="coordinator",
        to_agent=to_agent,
        payload=payload,
        scope=scope,  # type: ignore[arg-type]
        project_key=project_key,
    )
    return get_memory_manager().write_shared(item, mode=mode, reason=reason)


@tool
def read_shared_memory(
    reason: str,
    slot: Optional[str] = None,
    project_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Read what sub-agents have posted to shared memory in this session,
    newest first. Call this AFTER `ask_retrieval` / `ask_writer` to pick up
    the structured findings/handoffs they posted (rather than re-parsing
    their prose answer).

    `reason` is a one-sentence first-person rationale (≤20 words) shown in
    the memory-trace UI — explain *why* you're reading this slot, e.g.
    "I need Retrieval's findings to hand the exact ticket IDs to Writer".

    Optional `slot` filters to one of: last_search_results, handoff_payload,
    disambiguation, scratch, plan, findings, goal. Pass `project_key` to
    also pick up long-lived project-scoped slots (e.g. strategic goals).
    Pass `limit` to cap how many entries are returned."""
    return get_memory_manager().read_shared(
        get_session_id(),
        slot=slot,
        project_key=project_key,
        limit=limit,
        reason=reason,
    )


# ── direct convenience tools ──────────────────────────────────────────────


@tool
def list_in_flight_tasks(reason: str, project: Optional[str] = None) -> dict:
    """Return the headline 'what's in flight right now' snapshot:
    open Jira tickets (In Progress + In Review + To Do), upcoming
    calendar events in the next 14 days, and the 5 most recent episodic
    events. Optional `project` filters Jira to one project key
    (e.g. 'PROJ-ATLAS').

    `reason` is a one-sentence first-person rationale (≤20 words) shown in
    the memory-trace UI — explain *why* you're pulling this snapshot.
    """
    from datetime import datetime, timedelta

    jira_q: dict = {"status": {"$in": ["In Progress", "In Review", "To Do"]}}
    if project:
        jira_q["project"] = project
    tickets = list(
        get_collection(JIRA_COLLECTION)
        .find(jira_q, {"_id": 0})
        .sort([("status", 1), ("priority", 1)])
        .limit(50)
    )

    now = datetime.utcnow()
    horizon = now + timedelta(days=14)
    upcoming = list(
        get_collection(CALENDAR_COLLECTION)
        .find({"start": {"$gte": now, "$lte": horizon}}, {"_id": 0})
        .sort("start", 1)
        .limit(20)
    )

    recent = get_memory_manager().recent_episodes(
        get_user_id(), limit=5, reason=reason
    )

    return {
        "open_tickets": tickets,
        "upcoming_events": upcoming,
        "recent_activity": [
            {
                "timestamp": e["timestamp"].isoformat(),
                "event_type": e["event_type"],
                "summary": e["summary"],
            }
            for e in recent
        ],
    }


COORDINATOR_TOOLS = [
    ask_retrieval,
    ask_writer,
    post_plan,
    read_shared_memory,
    list_in_flight_tasks,
    render_gantt,
]
