"""Slash-command handlers for the PM agent demo Chainlit UI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import chainlit as cl

from agents.charting import build_gantt_figure, collect_gantt_rows
from agents.runtime import get_memory_manager
from memory.db import (
    ALL_MEMORY_COLLECTIONS,
    CALENDAR_COLLECTION,
    JIRA_COLLECTION,
    PROJECTS_COLLECTION,
    get_collection,
)

HELP_TEXT = """\
**Commands**

| Command | What it does |
|---|---|
| `/help` | Show this help |
| `/welcome` | Re-render the welcome screen (projects, team, stakeholders) |
| `/demo` | Show the 10 demo beats as click-to-run buttons |
| `/roadmap` | Snapshot of open tickets, upcoming meetings, recent activity |
| `/gantt [project] [days]` | Render an interactive Plotly Gantt of work in flight |
| `/memory <type>` | Inspect a memory store. Type ∈ working / episodic / semantic / procedural / shared / jira / calendar / counts / all |
| `/reset` | Wipe **everything** and re-seed the Northwind dataset (asks first) |
"""


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _block(title: str, payload: Any) -> str:
    body = json.dumps(_to_jsonable(payload), indent=2, default=str)
    return f"### {title}\n```json\n{body}\n```"


async def _send_block(title: str, payload: Any) -> None:
    await cl.Message(content=_block(title, payload)).send()


# ── /memory ────────────────────────────────────────────────────────────────


def _dump(name: str, limit: int = 50) -> list[dict]:
    return list(get_collection(name).find({}, {"_id": 0, "embedding": 0}).limit(limit))


async def cmd_memory(arg: str) -> None:
    arg = (arg or "all").strip().lower()
    mm = get_memory_manager()

    if arg in ("counts", "stats"):
        await _send_block("Memory counts", mm.counts())
        return
    if arg == "working":
        await _send_block("Working memory", _dump("working_memory"))
        return
    if arg == "episodic":
        await _send_block("Episodic memory (last 25)", _dump("episodic_memory", 25))
        return
    if arg == "semantic":
        await _send_block("Semantic memory", _dump("semantic_memory"))
        return
    if arg == "procedural":
        await _send_block("Procedural memory", _dump("procedural_memory"))
        return
    if arg == "shared":
        await _send_block("Shared memory", _dump("shared_memory"))
        return
    if arg == "jira":
        await _send_block("Jira tickets", _dump(JIRA_COLLECTION))
        return
    if arg == "calendar":
        await _send_block("Calendar events", _dump(CALENDAR_COLLECTION))
        return
    if arg in ("all", ""):
        await _send_block("Memory counts", mm.counts())
        for c in ALL_MEMORY_COLLECTIONS:
            await _send_block(c, _dump(c, 10))
        return
    await cl.Message(
        content=f"Unknown memory type `{arg}`. Try working, episodic, semantic, procedural, shared, jira, calendar, counts, or all."
    ).send()


# ── /roadmap ───────────────────────────────────────────────────────────────


async def cmd_roadmap() -> None:
    from datetime import timedelta

    now = datetime.utcnow()
    horizon = now + timedelta(days=14)
    tickets = list(
        get_collection(JIRA_COLLECTION)
        .find({"status": {"$in": ["In Progress", "In Review", "To Do"]}}, {"_id": 0})
        .sort([("status", 1), ("priority", 1)])
    )
    events = list(
        get_collection(CALENDAR_COLLECTION)
        .find({"start": {"$gte": now, "$lte": horizon}}, {"_id": 0})
        .sort("start", 1)
    )
    projects = list(get_collection(PROJECTS_COLLECTION).find({}, {"_id": 0}))
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "admin"
    recent = get_memory_manager().recent_episodes(user_id, limit=5)

    await _send_block(
        "🛫 What's in flight",
        {
            "projects": projects,
            "open_tickets": tickets,
            "upcoming_events": events,
            "recent_activity": [
                {"timestamp": e["timestamp"], "event_type": e["event_type"], "summary": e["summary"]}
                for e in recent
            ],
        },
    )


# ── /gantt ─────────────────────────────────────────────────────────────────


async def cmd_gantt(arg: str) -> None:
    """`/gantt [PROJECT] [DAYS]` — render a Plotly Gantt of work in flight.

    Bypasses the LLM: pulls tickets/events/milestones straight from Mongo
    and attaches the figure to a cl.Plotly element. Useful for the demo
    when you want a guaranteed clean render.
    """
    parts = (arg or "").split()
    project: str | None = None
    horizon_days = 90
    for token in parts:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            horizon_days = int(token)
        elif token.upper().startswith("PROJ-") or token.lower() in {"atlas", "mobile", "report"}:
            t = token.upper()
            project = t if t.startswith("PROJ-") else f"PROJ-{t}"

    now = datetime.utcnow()
    window_end = now + timedelta(days=max(horizon_days, 7))
    rows, milestones, skipped = collect_gantt_rows(
        project=project,
        window_start=now - timedelta(days=30),
        window_end=window_end,
    )
    if not rows and not milestones:
        await cl.Message(
            content=(
                f"No items in scope for project={project!r}, horizon={horizon_days}d. "
                "Try a wider horizon or omit the project filter."
            )
        ).send()
        return

    title = (
        f"{project} roadmap · next {horizon_days}d"
        if project
        else f"All projects · next {horizon_days}d"
    )
    fig = build_gantt_figure(
        rows=rows, milestones=milestones, title=title,
        group_by="project", now=now, window_end=window_end,
    )
    caption = (
        f"📊 **{title}** — {len(rows)} bars, {len(milestones)} milestones"
        + (f" · {skipped} skipped (no dates)" if skipped else "")
    )
    await cl.Message(
        content=caption,
        elements=[cl.Plotly(name=f"gantt_{project or 'all'}", figure=fig, display="inline")],
    ).send()


# ── /reset (with confirmation) ─────────────────────────────────────────────


async def cmd_reset() -> None:
    """Ask for confirmation; the actual wipe runs in `on_action`."""
    actions = [
        cl.Action(name="reset_confirm", label="Yes, wipe & re-seed", payload={"go": True}),
        cl.Action(name="reset_cancel", label="Cancel", payload={"go": False}),
    ]
    await cl.Message(
        content="⚠️ This will **delete all memory + mock business data + LangGraph checkpoints** and re-seed the Northwind dataset. Confirm?",
        actions=actions,
    ).send()
