"""Chainlit UI for the multi-agent PM memory demo.

Run from the project root with:
    chainlit run chainlit_ui/app.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Make the project root importable so `agents`, `memory`, `llm`
# resolve when Chainlit launches this file from inside chainlit_ui/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

import chainlit as cl
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from agents.artifacts import ChartArtifact
from agents.coordinator import create_coordinator_agent
from agents.runtime import get_artifact_trace, get_memory_manager
from data_layer import MongoDBDataLayer
from memory.db import DB_NAME
from memory.events import MemoryEvent
from memory.reset import reset as reset_demo
from memory.seed import seed_session

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
# Chainlit threads/users live alongside everything else by default so /reset
# only has to point at one database.
CHAINLIT_DB_NAME = os.getenv("CHAINLIT_DB_NAME", DB_NAME)

import commands  # noqa: E402  (after sys.path tweak above)
import demo  # noqa: E402
import welcome  # noqa: E402


@cl.data_layer
def get_data_layer() -> MongoDBDataLayer:
    return MongoDBDataLayer(mongodb_uri=MONGODB_URI, db_name=CHAINLIT_DB_NAME)


@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    expected_user = os.getenv("CHAINLIT_USERNAME", "admin")
    expected_pass = os.getenv("CHAINLIT_PASSWORD", "admin")
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username)
    return None


_agent = None
_checkpointer_client: Optional[MongoClient] = None


def get_agent():
    global _agent, _checkpointer_client
    if _agent is None:
        _checkpointer_client = MongoClient(MONGODB_URI, appname="pm-memory-demo-cl")
        checkpointer = MongoDBSaver(_checkpointer_client, db_name=DB_NAME)
        _agent = create_coordinator_agent(checkpointer=checkpointer)
    return _agent


# ── memory trace plumbing ──────────────────────────────────────────────────

_MEMORY_ICON = {
    "working": "🟢",
    "episodic": "📜",
    "semantic": "🧠",
    "procedural": "📘",
    "shared": "🔗",
}


def _format_event(e: MemoryEvent) -> str:
    icon = _MEMORY_ICON.get(e.memory_type, "•")
    op = e.op.upper()
    latency = f"{e.latency_ms}ms" if e.latency_ms else ""
    if e.result_count is None:
        count = ""
    elif e.result_count == 0:
        count = " — _no results_"
    elif e.result_count == 1:
        count = " — _1 result_"
    else:
        count = f" — _{e.result_count} results_"
    return f"{icon} `{e.memory_type:<10}` **{op}**  {e.description}{count}  _{latency}_"


def _strip_embeddings(obj: Any) -> Any:
    """Recursively drop `embedding` fields so the inspector stays readable."""
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, list):
        return [_strip_embeddings(v) for v in obj]
    return obj


def _to_json(data: Any) -> str:
    return json.dumps(_strip_embeddings(data), indent=2, default=str)


def _event_step_name(e: MemoryEvent) -> str:
    """Compact plain-text label for a per-event collapsible step."""
    icon = _MEMORY_ICON.get(e.memory_type, "•")
    if e.result_count is None:
        count = ""
    elif e.result_count == 0:
        count = " — no results"
    elif e.result_count == 1:
        count = " — 1 result"
    else:
        count = f" — {e.result_count} results"
    latency = f"  ({e.latency_ms}ms)" if e.latency_ms else ""
    return f"{icon} {e.memory_type:<10} — {e.description}{count}{latency}"


async def _render_event_steps(events: list[MemoryEvent]) -> None:
    for e in events:
        async with cl.Step(name=_event_step_name(e), type="tool") as sub:
            sub.output = f"```json\n{_to_json(e.data)}\n```"


async def _render_trace(events: list[MemoryEvent]) -> None:
    if not events:
        return
    reads = [e for e in events if e.op == "read" and e.data is not None]
    writes = [e for e in events if e.op == "write" and e.data is not None]
    if reads:
        async with cl.Step(
            name=f"🧠 Memories Read ({len(reads)})", type="tool"
        ):
            await _render_event_steps(reads)
    if writes:
        async with cl.Step(
            name=f"🧠 Memories Written ({len(writes)})", type="tool"
        ):
            await _render_event_steps(writes)


# ── streaming helpers ──────────────────────────────────────────────────────


def _iter_text(chunk) -> list[str]:
    content = getattr(chunk, "content", None)
    if not content:
        return []
    if isinstance(content, str):
        return [content]
    pieces: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                pieces.append(text)
    return pieces


_TOOL_DISPLAY = {
    "ask_retrieval": ("🔎 Retrieval Agent", "search"),
    "ask_writer": ("✍️  Writer Agent", "edit-3"),
    "list_in_flight_tasks": ("🛫 In-flight snapshot", "rocket"),
    "render_gantt": ("📊 Render Gantt", "bar-chart-2"),
    "post_plan": ("🧭 Post plan", "compass"),
    "read_shared_memory": ("🔗 Read shared memory", "link-2"),
}


async def _drain_artifacts(artifacts: list[ChartArtifact]) -> None:
    """Render and clear any artifacts the agent emitted since the last drain."""
    if not artifacts:
        return
    drained, artifacts[:] = artifacts[:], []
    for art in drained:
        elements = [cl.Plotly(name=art.name, figure=art.figure, display="inline")]
        caption = f"📊 **{art.summary}**" if art.summary else ""
        await cl.Message(content=caption, elements=elements).send()


async def _send_tool_step(tool_msg, reason: Optional[str] = None) -> None:
    name = getattr(tool_msg, "name", "tool")
    label, icon = _TOOL_DISPLAY.get(name, (name, "wrench"))
    body = str(getattr(tool_msg, "content", ""))
    if reason:
        body = f"> 💭 _{reason}_\n\n{body}"
    async with cl.Step(name=label, type="tool", icon=icon) as step:
        step.output = body


def _harvest_tool_args(
    chunk,
    args_by_id: dict[str, dict[str, Any]],
    chunk_buf: dict[int, dict[str, Any]],
) -> None:
    """Accumulate streamed tool-call args by `tool_call_id` so we can look up
    the `reason` argument when the matching ToolMessage arrives.

    Handles both the aggregated `tool_calls` field (parsed dict per call) and
    Anthropic's incremental `tool_call_chunks` (string deltas keyed by index).
    """
    for tc in getattr(chunk, "tool_calls", None) or []:
        tc_id = tc.get("id")
        args = tc.get("args")
        if tc_id and isinstance(args, dict):
            args_by_id[tc_id] = args

    for c in getattr(chunk, "tool_call_chunks", None) or []:
        idx = c.get("index", 0)
        entry = chunk_buf.setdefault(idx, {"id": None, "args_str": ""})
        if c.get("id"):
            entry["id"] = c["id"]
        if c.get("args"):
            entry["args_str"] += c["args"]
        if entry["id"] and entry["args_str"]:
            try:
                args_by_id[entry["id"]] = json.loads(entry["args_str"])
            except json.JSONDecodeError:
                pass  # partial JSON — wait for more chunks


def _thinking_html(label: str) -> str:
    """Single inline widget: text label + three softly-pulsing SVG dots.

    SMIL animation (not CSS keyframes) so the markup survives Chainlit's
    HTML sanitizer without needing a `<style>` block.
    """
    dots = "".join(
        f'<circle cx="{5 + 12 * i}" cy="5" r="3" fill="currentColor" opacity="0.3">'
        f'<animate attributeName="opacity" values="0.3;1;0.3" dur="1.4s" '
        f'repeatCount="indefinite" begin="{i * 0.23:.2f}s"/></circle>'
        for i in range(3)
    )
    return (
        '<div style="display:inline-flex;align-items:center;gap:10px;'
        'color:rgba(127,127,127,0.85);font-size:13px;line-height:1;'
        'padding:4px 0;">'
        f'<span>{label}</span>'
        f'<svg width="34" height="10" xmlns="http://www.w3.org/2000/svg">'
        f'{dots}</svg>'
        '</div>'
    )


class _Thinking:
    """Toggleable inline 'thinking' indicator backed by a single cl.Message
    with custom HTML (avoids Chainlit's double-decoration on cl.Step).
    Idempotent show/hide so it can be reopened in the gaps between tool
    returns and the next LLM chunk."""

    def __init__(self, label: str = "💭 Thinking") -> None:
        self._label = label
        self._msg: Optional[cl.Message] = None

    async def show(self, label: Optional[str] = None) -> None:
        if label is not None:
            self._label = label
        html = _thinking_html(self._label)
        if self._msg is None:
            self._msg = cl.Message(content=html)
            await self._msg.send()
        else:
            self._msg.content = html
            await self._msg.update()

    async def hide(self) -> None:
        if self._msg is not None:
            msg, self._msg = self._msg, None
            await msg.remove()


def _agent_config() -> dict:
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "admin"
    thread_id = cl.context.session.thread_id
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="🎬 Run live demo (10 beats)",
            message="/demo",
        ),
        cl.Starter(
            label="👥 Show team & projects",
            message="/welcome",
        ),
        cl.Starter(
            label="🛫 What's in flight on Atlas?",
            message="What's in flight on Atlas?",
        ),
        cl.Starter(
            label="🧠 Inspect all memory",
            message="/memory all",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    get_agent()
    cfg = _agent_config()["configurable"]
    seed_session(cfg["thread_id"], cfg["user_id"])
    await welcome.send_welcome()


@cl.on_chat_resume
async def on_chat_resume(thread) -> None:
    get_agent()
    cfg = _agent_config()["configurable"]
    seed_session(cfg["thread_id"], cfg["user_id"])


# ── slash command dispatch ────────────────────────────────────────────────


async def _handle_command(text: str) -> bool:
    """Return True if the message was a slash command (and was handled)."""
    if not text.startswith("/"):
        return False
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    if cmd == "/help":
        await cl.Message(content=commands.HELP_TEXT).send()
        return True
    if cmd == "/welcome":
        await welcome.cmd_welcome()
        return True
    if cmd == "/demo":
        await demo.cmd_demo()
        return True
    if cmd == "/roadmap":
        await commands.cmd_roadmap()
        return True
    if cmd == "/gantt":
        await commands.cmd_gantt(arg)
        return True
    if cmd == "/memory":
        await commands.cmd_memory(arg)
        return True
    if cmd == "/reset":
        await commands.cmd_reset()
        return True
    await cl.Message(content=f"Unknown command `{cmd}`. Try `/help`.").send()
    return True


@cl.action_callback("reset_confirm")
async def on_reset_confirm(action: cl.Action) -> None:
    await action.remove()
    deleted = reset_demo(seed_after=True, keep_checkpoints=False)
    total = sum(deleted.values())
    summary = "\n".join(f"- `{k}` → {v}" for k, v in deleted.items() if v)
    await cl.Message(
        content=(
            f"✅ Reset complete. Deleted **{total}** documents and re-seeded the "
            f"Leafy Technologies dataset.\n\n{summary or '_(nothing to delete)_'}\n\n"
            "Start a new chat to clear conversation context."
        )
    ).send()


@cl.action_callback("reset_cancel")
async def on_reset_cancel(action: cl.Action) -> None:
    await action.remove()
    await cl.Message(content="Reset cancelled.").send()


@cl.action_callback("run_demo_prompt")
async def on_run_demo_prompt(action: cl.Action) -> None:
    prompt = action.payload.get("prompt", "")
    if not prompt:
        return
    # Echo the prompt as a user message so the transcript reads correctly,
    # then drive it through the same path real input takes.
    await cl.Message(content=prompt, type="user_message").send()
    await _handle_user_input(prompt)


# ── main message loop ─────────────────────────────────────────────────────


async def _handle_user_input(text: str) -> None:
    if await _handle_command(text):
        return

    agent = get_agent()
    config = _agent_config()
    mm = get_memory_manager()

    # Capture every memory op that happens during this turn so we can render
    # them as one collapsible "🧠 Memory ops" step after the response.
    events: list[MemoryEvent] = []
    unsubscribe = mm.trace.subscribe(events.append)

    # Capture chart artifacts (Plotly figures) emitted by tools and render
    # them inline immediately after the tool call that produced them.
    artifacts: list[ChartArtifact] = []
    unsubscribe_art = get_artifact_trace().subscribe(artifacts.append)

    # Track tool-call args streamed on AI chunks so we can surface the
    # `reason` argument on the matching tool-result step.
    tool_args_by_id: dict[str, dict[str, Any]] = {}
    tool_chunk_buf: dict[int, dict[str, Any]] = {}

    response: Optional[cl.Message] = None
    thinking = _Thinking()
    await thinking.show()
    try:
        async for chunk, _ in agent.astream(
            {"messages": [{"role": "user", "content": text}]},
            config=config,
            stream_mode="messages",
        ):
            if getattr(chunk, "type", None) == "tool":
                if response is not None:
                    await response.update()
                    response = None
                await thinking.hide()
                tc_id = getattr(chunk, "tool_call_id", "") or ""
                reason = tool_args_by_id.get(tc_id, {}).get("reason")
                await _send_tool_step(chunk, reason=reason)
                await _drain_artifacts(artifacts)
                # Tool returned; LLM is about to think again until the next chunk.
                await thinking.show()
                continue

            _harvest_tool_args(chunk, tool_args_by_id, tool_chunk_buf)

            for piece in _iter_text(chunk):
                await thinking.hide()
                if response is None:
                    response = cl.Message(content="")
                    await response.send()
                await response.stream_token(piece)

        if response is not None:
            await response.update()
        await _drain_artifacts(artifacts)
    finally:
        await thinking.hide()
        unsubscribe()
        unsubscribe_art()

    await _render_trace(events)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _handle_user_input(message.content)

