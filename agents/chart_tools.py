"""Visualisation tools for the Coordinator agent.

Currently exposes `render_gantt`, which produces an interactive Plotly
timeline of tickets, calendar events and milestones and pushes it onto
the artifact bus for the Chainlit UI to render.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.tools import tool

from agents.runtime import get_artifact_trace
from agents.artifacts import ChartArtifact, make_artifact_name
from agents.charting import build_gantt_figure, collect_gantt_rows


@tool
def render_gantt(
    project: Optional[str] = None,
    horizon_days: int = 90,
    group_by: str = "project",
    include_tickets: bool = True,
    include_events: bool = True,
    include_milestones: bool = True,
) -> dict[str, Any]:
    """Render an interactive Gantt timeline of work in flight.

    Call this ONLY when the user explicitly asks for a Gantt chart, timeline,
    roadmap visualisation, or schedule diagram. The chart is rendered directly
    in the Chainlit UI; your reply should give a 1–2 sentence read of what's
    on it (counts, key dates, conflicts) — do NOT redescribe each row in
    prose.

    Args:
        project: Optional project key (e.g. "PROJ-ATLAS") to filter by.
        horizon_days: Days from today to include in the chart (default 90).
        group_by: One of "project" | "status" | "assignee" — drives the
            colour grouping. Default "project".
        include_tickets: Include open Jira tickets as bars.
        include_events: Include calendar meetings as bars.
        include_milestones: Overlay project milestones as diamond markers.
    """
    now = datetime.utcnow()
    window_end = now + timedelta(days=max(horizon_days, 7))

    rows, milestones, skipped = collect_gantt_rows(
        project=project,
        window_start=now - timedelta(days=30),
        window_end=window_end,
        include_tickets=include_tickets,
        include_events=include_events,
        include_milestones=include_milestones,
    )

    if not rows and not milestones:
        return {
            "rendered": False,
            "items": 0,
            "skipped_no_dates": skipped,
            "summary": (
                f"No items in scope for project={project!r}, "
                f"horizon={horizon_days}d. Try a wider horizon or omit the "
                "project filter."
            ),
        }

    title = "Roadmap"
    if project:
        title = f"{project} roadmap · next {horizon_days}d"
    else:
        title = f"All projects · next {horizon_days}d"

    fig = build_gantt_figure(
        rows=rows,
        milestones=milestones,
        title=title,
        group_by=group_by,
        now=now,
        window_end=window_end,
    )

    name = make_artifact_name("gantt")
    summary = _summarise(rows, milestones, now, project)

    get_artifact_trace().emit(
        ChartArtifact(
            name=name,
            figure=fig,
            summary=summary,
            kind="plotly",
            payload={
                "project": project,
                "horizon_days": horizon_days,
                "group_by": group_by,
                "row_count": len(rows),
                "milestone_count": len(milestones),
            },
        )
    )

    return {
        "rendered": True,
        "chart_id": name,
        "items": len(rows) + len(milestones),
        "skipped_no_dates": skipped,
        "horizon": f"{now:%Y-%m-%d} → {window_end:%Y-%m-%d}",
        "summary": summary,
    }


def _summarise(rows, milestones, now: datetime, project: Optional[str]) -> str:
    """Tight one-paragraph read of the chart for the LLM to paraphrase."""
    by_status: dict[str, int] = {}
    next_due = None
    next_due_key = None
    for r in rows:
        if r["kind"] == "ticket":
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            if r["status"] != "Done" and r["finish"] >= now:
                if next_due is None or r["finish"] < next_due:
                    next_due = r["finish"]
                    next_due_key = r["key"]
    next_ms = min(milestones, key=lambda m: m["date"]) if milestones else None

    parts = []
    scope = project or "all projects"
    parts.append(
        f"{len(rows)} bars across {scope}"
        + (f" ({', '.join(f'{k}: {v}' for k, v in by_status.items())})" if by_status else "")
    )
    if next_ms:
        days = (next_ms["date"] - now).days
        parts.append(f"next milestone: {next_ms['title']} in {days}d")
    if next_due_key and next_due:
        days = (next_due - now).days
        parts.append(f"earliest open due date: {next_due_key} in {days}d")
    return ". ".join(parts) + "."


__all__ = ["render_gantt"]
