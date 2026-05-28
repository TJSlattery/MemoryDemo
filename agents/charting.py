"""Pure data + figure helpers for the Gantt tool.

Kept separate from `chart_tools.py` so the LLM-facing tool stays small,
and so the `/gantt` slash command can reuse the same code path without
going through the agent loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import plotly.graph_objects as go

from memory.db import (
    CALENDAR_COLLECTION,
    JIRA_COLLECTION,
    MILESTONES_COLLECTION,
    get_collection,
)

# ── palette ───────────────────────────────────────────────────────────────
# MongoDB LeafyGreen design tokens (mongodb.design) so the Gantt picks up
# the same first-party look as the rest of the UI.
STATUS_COLOR = {
    "To Do":       "#889397",  # MDB Gray Base
    "In Progress": "#FFC010",  # MDB Yellow Base
    "In Review":   "#016BF8",  # MDB Blue Base
    "Done":        "#00684A",  # MDB Forest Green
    "Blocked":     "#DB3030",  # MDB Red Base
}
EVENT_COLOR     = "#5E0C90"   # MDB Purple Dark (calendar meetings)
MILESTONE_COLOR = "#970606"   # MDB Red Dark (diamond markers)
TODAY_COLOR     = "#DB3030"   # MDB Red Base
RISK_BORDER     = "#DB3030"
PROJECT_COLOR = {
    "PROJ-ATLAS":  "#00684A",  # MDB Forest Green
    "PROJ-MOBILE": "#00ED64",  # MDB Mint
    "PROJ-REPORT": "#1254B7",  # MDB Blue Dark
}
ASSIGNEE_FALLBACK = "#3D4F58"  # MDB Gray Dark

# Heuristic ticket durations when no due_date is recorded yet.
PRIORITY_DAYS = {"High": 14, "Medium": 21, "Low": 30}
DEFAULT_DAYS = 21


# ── data collection ──────────────────────────────────────────────────────


def collect_gantt_rows(
    project: Optional[str],
    window_start: datetime,
    window_end: datetime,
    *,
    include_tickets: bool = True,
    include_events: bool = True,
    include_milestones: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Pull tickets, calendar events and milestones from Mongo and shape them
    into uniform row dicts. Returns (rows, milestones, skipped_no_dates)."""

    rows: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    skipped = 0

    project_filter: dict = {}
    if project:
        project_filter["project"] = project

    if include_tickets:
        for t in get_collection(JIRA_COLLECTION).find(project_filter, {"_id": 0}):
            row = _ticket_row(t)
            if row is None:
                skipped += 1
                continue
            if row["finish"] < window_start or row["start"] > window_end:
                continue
            rows.append(row)

    if include_events:
        for e in get_collection(CALENDAR_COLLECTION).find(project_filter, {"_id": 0}):
            row = _event_row(e)
            if row is None:
                continue
            if row["finish"] < window_start or row["start"] > window_end:
                continue
            rows.append(row)

    if include_milestones:
        for m in get_collection(MILESTONES_COLLECTION).find(project_filter, {"_id": 0}):
            d = m.get("date")
            if not isinstance(d, datetime):
                continue
            if d < window_start or d > window_end:
                continue
            milestones.append({**m, "date": d})

    rows.sort(key=lambda r: (r["project"], r["start"]))
    milestones.sort(key=lambda m: m["date"])
    return rows, milestones, skipped


def _ticket_row(t: dict) -> Optional[dict]:
    start = t.get("start_date") or t.get("created_at")
    finish = t.get("due_date")
    estimated = False
    if not isinstance(start, datetime):
        return None
    if not isinstance(finish, datetime):
        days = PRIORITY_DAYS.get(t.get("priority", ""), DEFAULT_DAYS)
        finish = start + timedelta(days=days)
        estimated = True
    if finish <= start:
        finish = start + timedelta(days=1)
    title = t.get("title", "(untitled)")
    label = f"{t.get('key', '?')} · {_truncate(title, 42)}"
    return {
        "kind": "ticket",
        "key": t.get("key", "?"),
        "title": title,
        "label": label,
        "project": t.get("project", "?"),
        "status": t.get("status", "To Do"),
        "assignee": t.get("assignee", "—"),
        "priority": t.get("priority", "—"),
        "start": start,
        "finish": finish,
        "estimated": estimated,
        "linked_risks": list(t.get("linked_risks") or []),
    }


def _event_row(e: dict) -> Optional[dict]:
    start = e.get("start")
    if not isinstance(start, datetime):
        return None
    duration = int(e.get("duration_minutes", 60))
    finish = start + timedelta(minutes=max(duration, 30))
    title = e.get("title", "(meeting)")
    return {
        "kind": "event",
        "key": "📅",
        "title": title,
        "label": f"📅 {_truncate(title, 42)}",
        "project": e.get("project", "—"),
        "status": "Meeting",
        "assignee": ", ".join(e.get("attendees", []) or []) or "—",
        "priority": "—",
        "start": start,
        "finish": finish,
        "estimated": False,
        "linked_risks": [],
    }


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ── figure builder ────────────────────────────────────────────────────────


def build_gantt_figure(
    rows: list[dict[str, Any]],
    milestones: list[dict[str, Any]],
    *,
    title: str,
    group_by: str,
    now: datetime,
    window_end: datetime,
) -> go.Figure:
    """Build a polished horizontal-bar Gantt with milestones + Today line."""

    # Y-axis order: top → bottom in the order rows arrive (already sorted by
    # project then start). Reverse here so add_trace() puts the first row at
    # the top of the chart.
    rows = list(reversed(rows))
    y_labels = [r["label"] for r in rows]

    fig = go.Figure()

    # One go.Bar trace per legend group so the legend is meaningful and so
    # related rows share color naturally.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(_group_key(r, group_by), []).append(r)

    for group_label, group_rows in grouped.items():
        fills, borders = _colors_for(group_label, group_by, group_rows)
        line_colors = [
            RISK_BORDER if r["linked_risks"] else borders[i]
            for i, r in enumerate(group_rows)
        ]
        line_widths = [
            2 if r["linked_risks"] else (1 if borders[i] != "rgba(0,0,0,0)" else 0)
            for i, r in enumerate(group_rows)
        ]
        fig.add_trace(
            go.Bar(
                name=group_label,
                y=[r["label"] for r in group_rows],
                x=[(r["finish"] - r["start"]).total_seconds() * 1000 for r in group_rows],
                base=[r["start"] for r in group_rows],
                orientation="h",
                marker=dict(
                    color=fills,
                    line=dict(color=line_colors, width=line_widths),
                ),
                opacity=0.92,
                customdata=[
                    [
                        r["title"],
                        r["key"],
                        r["status"],
                        r["assignee"],
                        r["finish"],
                        r["project"],
                        ("⚠ linked risk" if r["linked_risks"] else ("~ estimated" if r["estimated"] else "")),
                    ]
                    for r in group_rows
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]} · %{customdata[2]} · %{customdata[5]}<br>"
                    "👤 %{customdata[3]}<br>"
                    "🗓 %{base|%a %b %-d} → %{customdata[4]|%a %b %-d}"
                    "<br>%{customdata[6]}"
                    "<extra></extra>"
                ),
            )
        )

    # Milestones overlay: scatter on the same y-axis as the bars, anchored to
    # the project's first row label so they line up visually. If there are no
    # bars, synthesise a single y-row per project so the diamonds still show.
    if milestones:
        synth_anchor = None
        if not y_labels:
            synth_anchor = "Milestones"
            y_labels = [synth_anchor]
        ms_y, ms_x, ms_text, ms_hover = [], [], [], []
        for m in milestones:
            anchor = _milestone_anchor(m, rows) or synth_anchor or y_labels[0]
            ms_y.append(anchor)
            ms_x.append(m["date"])
            ms_text.append(m["title"])
            ms_hover.append(
                f"<b>◆ {m['title']}</b><br>{m['project']}<br>🗓 "
                f"{m['date']:%a %b %-d}"
            )
        fig.add_trace(
            go.Scatter(
                name="Milestone",
                x=ms_x,
                y=ms_y,
                mode="markers+text",
                marker=dict(symbol="diamond", size=14, color=MILESTONE_COLOR,
                            line=dict(width=1.5, color="white")),
                text=ms_text,
                textposition="top center",
                textfont=dict(size=10, color="#3D4F58"),
                hovertext=ms_hover,
                hoverinfo="text",
            )
        )

    # Today marker. `add_vline` chokes on datetime x when computing the
    # annotation midpoint, so we draw the line + label as separate shapes.
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=now, x1=now, y0=0, y1=1,
        line=dict(color=TODAY_COLOR, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=now, y=1.0, xref="x", yref="paper",
        text="Today", showarrow=False,
        yanchor="bottom", yshift=4,
        font=dict(size=11, color=TODAY_COLOR),
    )

    n_rows = max(len(rows), 1)
    height = min(720, max(280, 28 * n_rows + 120))

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left",
                   font=dict(size=16, color="#001E2B")),
        font=dict(family="Euclid Circular A, Inter, -apple-system, system-ui, sans-serif",
                  size=12, color="#21313C"),
        margin=dict(l=180, r=40, t=70, b=40),
        height=height,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor="#E8EDEB",
                        font=dict(family="Euclid Circular A, Inter", size=12, color="#001E2B")),
        xaxis=dict(
            type="date",
            tickformat="%b %-d",
            showgrid=True,
            gridcolor="rgba(136,147,151,0.20)",
            ticks="outside",
            tickfont=dict(size=11),
            range=[min(r["start"] for r in rows) if rows else now, window_end],
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            showgrid=False,
            automargin=True,
        ),
        bargap=0.35,
    )
    return fig


def _group_key(row: dict, group_by: str) -> str:
    if group_by == "status":
        return row["status"]
    if group_by == "assignee":
        return row["assignee"]
    return row["project"]


def _colors_for(
    label: str, group_by: str, group_rows: list[dict]
) -> tuple[list[str], list[str]]:
    """Returns aligned (fills, borders) lists, one entry per row in the
    group. When grouping by project we colour fills by project and overlay
    status as the bar border so both dimensions read at a glance."""
    fills: list[str] = []
    borders: list[str] = []
    if group_by == "status":
        fill = STATUS_COLOR.get(label, ASSIGNEE_FALLBACK)
        for _ in group_rows:
            fills.append(fill)
            borders.append("rgba(0,0,0,0)")
        return fills, borders
    if group_by == "assignee":
        for _ in group_rows:
            fills.append(ASSIGNEE_FALLBACK)
            borders.append("rgba(0,0,0,0)")
        return fills, borders
    base = PROJECT_COLOR.get(label, ASSIGNEE_FALLBACK)
    for r in group_rows:
        fills.append(EVENT_COLOR if r["kind"] == "event" else base)
        borders.append(STATUS_COLOR.get(r["status"], "rgba(0,0,0,0)"))
    return fills, borders


def _milestone_anchor(m: dict, rows: list[dict]) -> Optional[str]:
    """Anchor a milestone to the y-label of the first row in its project."""
    for r in rows:
        if r["project"] == m.get("project"):
            return r["label"]
    return None
