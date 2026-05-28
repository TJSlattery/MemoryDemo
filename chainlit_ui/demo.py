"""Live-demo helper: the 8 stage prompts as click-to-run cards."""

from __future__ import annotations

import chainlit as cl

from welcome import _card

# (number, heading, what-it-shows, prompt-text)
DEMO_PROMPTS: list[tuple[str, str, str, str]] = [
    (
        "1️⃣",
        "Recall",
        "Cross-collection semantic search by person + risk.",
        "Who owns the Atlas Migration and what's the latest risk?",
    ),
    (
        "2️⃣",
        "What's in flight",
        "Direct tool call (`list_in_flight_tasks`), no LLM round-trip for the lookup.",
        "What's in flight on Atlas?",
    ),
    (
        "3️⃣",
        "Working memory",
        "Populates `current_project`, `current_task`, `focus`, and `scratchpad` in one turn.",
        "I'm prepping the Atlas dry-run rehearsal today — main goal is to hit Q2 GA without surprises. Also jot down: confirm Priya is on the bridge call.",
    ),
    (
        "4️⃣",
        "Action + episodic",
        "Dual-write: Jira ticket + episodic event in one call.",
        "Create a Jira ticket on PROJ-ATLAS to rehearse the runbook in staging, high, assign Priya.",
    ),
    (
        "5️⃣",
        "Decision",
        "Semantic fact + episodic event in one call.",
        "Log a decision: freeze billing-service deploys 24h before cutover.",
    ),
    (
        "6️⃣",
        "Procedural ritual",
        "Workflow fetched from procedural memory and executed.",
        "Run the daily standup.",
    ),
    (
        "7️⃣",
        "Cross-turn recall",
        "Checkpointer-backed conversation memory + episodic recall.",
        "What did I just do?",
    ),
    (
        "8️⃣",
        "Inspect",
        "Dump every memory store with counts + samples.",
        "/memory all",
    ),
    (
        "9️⃣",
        "Pre-seeded focus",
        "Reads working memory pre-populated at chat start (current project, task, focus, scratchpad).",
        "What am I focused on right now, and what's on my scratchpad?",
    ),
    (
        "🔟",
        "Inter-agent handoff",
        "Surfaces shared-memory slots left by the Retrieval Agent for the Coordinator and Writer.",
        "Show me the most recent handoff payload and last search results between the agents.",
    ),
]

_DEMO_ACCENT = "#00684A"  # MongoDB Forest Green


def _demo_card(num: str, heading: str, blurb: str, prompt: str) -> str:
    body = (
        f"{blurb}<br/>"
        f'<code style="display:inline-block;margin-top:6px;padding:4px 8px;'
        f'border-radius:6px;background:rgba(127,127,127,0.12);font-size:12px;">'
        f"{prompt}</code>"
    )
    return _card(
        title=f"{num} {heading}",
        subtitle=f"Click ▶ {num} below to run",
        body=body,
        accent=_DEMO_ACCENT,
        min_width=320,
    )


def build_demo_html() -> str:
    cards = "".join(_demo_card(n, h, b, p) for n, h, b, p in DEMO_PROMPTS)
    return (
        "### 🎬 Live demo — 10 beats\n"
        '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 16px 0;">'
        f"{cards}"
        "</div>"
        "_Click a ▶ button below to send that beat. They run in any order — "
        "but **3 → 7** form a story (set focus, then ask 'what did I just do?'). "
        "**9 & 10** showcase working + shared memory pre-seeded at chat start._"
    )


async def cmd_demo() -> None:
    actions = [
        cl.Action(
            name="run_demo_prompt",
            label=f"▶ {num}",
            payload={"prompt": prompt},
            tooltip=heading,
        )
        for num, heading, _, prompt in DEMO_PROMPTS
    ]
    await cl.Message(content=build_demo_html(), actions=actions).send()
