"""
Chainlit UI for the MongoDB-backed memory agent.

Run from the project root with:
    chainlit run chainlit_ui/app.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Make the project root importable so `agent` and `grove_llm` resolve when
# Chainlit launches this file from inside the chainlit_ui/ folder.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

import chainlit as cl

from agent import create_memory_agent
from data_layer import MongoDBDataLayer

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
CHAINLIT_DB_NAME = os.getenv("CHAINLIT_DB_NAME", "agent_memory_simple")


@cl.data_layer
def get_data_layer() -> MongoDBDataLayer:
    """Persist threads/messages to MongoDB so the left-side history panel works."""
    return MongoDBDataLayer(mongodb_uri=MONGODB_URI, db_name=CHAINLIT_DB_NAME)


@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    """Single-user auth backed by env vars; required for thread persistence."""
    expected_user = os.getenv("CHAINLIT_USERNAME", "admin")
    expected_pass = os.getenv("CHAINLIT_PASSWORD", "admin")
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username)
    return None


_agent = None


def get_agent():
    """Build the agent once and reuse it across sessions."""
    global _agent
    if _agent is None:
        _agent = create_memory_agent(MONGODB_URI)
    return _agent


def _iter_text(chunk) -> list[str]:
    """Pull printable text out of a streamed message chunk (string or block list)."""
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
    "save_memory": ("Saved memory", "brain"),
    "retrieve_memories": ("Memory search", "brain"),
}


async def _send_tool_step(tool_msg) -> None:
    """Render a tool result as a collapsible chip instead of inline text."""
    name = getattr(tool_msg, "name", "tool")
    label, icon = _TOOL_DISPLAY.get(name, (name, "wrench"))
    async with cl.Step(name=label, type="tool", icon=icon) as step:
        step.output = str(getattr(tool_msg, "content", ""))


def _agent_config() -> dict:
    """Build the LangGraph config from the current Chainlit session."""
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "default_user"
    thread_id = cl.context.session.thread_id
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


@cl.on_chat_start
async def on_chat_start() -> None:
    get_agent()


@cl.on_chat_resume
async def on_chat_resume(thread) -> None:
    get_agent()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent = get_agent()
    config = _agent_config()

    response: Optional[cl.Message] = None

    async for chunk, _ in agent.astream(
        {"messages": [{"role": "user", "content": message.content}]},
        config=config,
        stream_mode="messages",
    ):
        if getattr(chunk, "type", None) == "tool":
            if response is not None:
                await response.update()
                response = None
            await _send_tool_step(chunk)
            continue

        for piece in _iter_text(chunk):
            if response is None:
                response = cl.Message(content="")
                await response.send()
            await response.stream_token(piece)

    if response is not None:
        await response.update()
