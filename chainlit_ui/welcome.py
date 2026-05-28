"""Welcome screen: live cards for projects, team, and stakeholders.

Pulled fresh from MongoDB on every render so post-`/reset` state and any
Writer-driven additions show up immediately.
"""

from __future__ import annotations

import chainlit as cl

from memory.db import (
    PROJECTS_COLLECTION,
    SEMANTIC_COLLECTION,
    get_collection,
)

# Per-project accent (left border) so the three projects are visually distinct.
# Colours pulled from the MongoDB LeafyGreen palette (mongodb.design).
_PROJECT_ACCENT = {
    "PROJ-ATLAS": "#00684A",   # Forest Green (MDB primary)
    "PROJ-MOBILE": "#00ED64",  # Mint
    "PROJ-REPORT": "#016BF8",  # Blue
}
_DEFAULT_ACCENT = "#00684A"

_GRID_OPEN = (
    '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 16px 0;">'
)
_GRID_CLOSE = "</div>"


def _card(
    *,
    title: str,
    subtitle: str,
    body: str,
    accent: str = _DEFAULT_ACCENT,
    min_width: int = 260,
) -> str:
    return (
        f'<div style="flex:1 1 {min_width}px;border:1px solid rgba(127,127,127,0.25);'
        f'border-left:4px solid {accent};border-radius:12px;padding:12px 16px;'
        f'background:rgba(127,127,127,0.06);">'
        f'<div style="font-weight:600;font-size:14px;margin-bottom:2px;">{title}</div>'
        f'<div style="opacity:0.65;font-size:12px;margin-bottom:8px;">{subtitle}</div>'
        f'<div style="font-size:13px;line-height:1.45;">{body}</div>'
        f"</div>"
    )


def _strip_name_prefix(name: str, content: str) -> str:
    """Seeded `content` is `"{name}. {blurb}"` — drop the redundant name."""
    prefix = f"{name}. "
    return content[len(prefix):] if content.startswith(prefix) else content


def _projects_section() -> str:
    docs = list(get_collection(PROJECTS_COLLECTION).find({}, {"_id": 0}).sort("key", 1))
    if not docs:
        return ""
    cards = [
        _card(
            title=f"📦 {d['name']}",
            subtitle=d["key"],
            body=d.get("summary", ""),
            accent=_PROJECT_ACCENT.get(d["key"], _DEFAULT_ACCENT),
            min_width=280,
        )
        for d in docs
    ]
    return f"### Projects in flight\n{_GRID_OPEN}{''.join(cards)}{_GRID_CLOSE}"


def _people_section(kind: str, heading: str, icon: str, accent: str) -> str:
    docs = list(
        get_collection(SEMANTIC_COLLECTION)
        .find({"kind": kind}, {"_id": 0, "embedding": 0})
        .sort("value", 1)
    )
    if not docs:
        return ""
    cards = [
        _card(
            title=f"{icon} {d['value']}",
            subtitle=d["key"],
            body=_strip_name_prefix(d["value"], d.get("content", "")),
            accent=accent,
        )
        for d in docs
    ]
    return f"### {heading}\n{_GRID_OPEN}{''.join(cards)}{_GRID_CLOSE}"


# Quick-action prompts pinned to the welcome message so they stay clickable
# after Chainlit hides the initial @set_starters chips. (label, prompt)
WELCOME_PROMPTS: list[tuple[str, str]] = [
    ("🎬 Run live demo", "/demo"),
    ("🛫 What's in flight on Atlas?", "What's in flight on Atlas?"),
    ("🧠 Inspect all memory", "/memory all"),
]


def build_welcome_html() -> str:
    intro = (
        "👋 I'm **Mr. Anderson**, the PM assistant for **Tom** at **Leafy Technologies**. "
        "Three agents (Coordinator + Retrieval + Writer) over five MongoDB Atlas-backed memory types. "
        "Click a quick-action below, or type `/help` for slash commands."
    )
    sections = [
        intro,
        _projects_section(),
        _people_section("person", "Team", "👤", "#13AA52"),       # MDB green-base
        _people_section("stakeholder", "Stakeholders", "⭐", "#1254B7"),  # MDB blue-dark
        "_Tip: click **New Chat** in the sidebar (or run `/welcome`) to bring this view back._",
    ]
    return "\n\n".join(s for s in sections if s)


def _welcome_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="run_demo_prompt",
            label=label,
            payload={"prompt": prompt},
            tooltip=prompt,
        )
        for label, prompt in WELCOME_PROMPTS
    ]


async def send_welcome() -> None:
    await cl.Message(content=build_welcome_html(), actions=_welcome_actions()).send()


async def cmd_welcome() -> None:
    """Slash-command entry point — re-render the welcome in the current session."""
    await send_welcome()
