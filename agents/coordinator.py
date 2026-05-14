"""Coordinator Agent — Sonnet-class supervisor that delegates to the
Retrieval and Writer sub-agents.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agents.coordinator_tools import COORDINATOR_TOOLS
from agents.prompts import COORDINATOR_SYSTEM_PROMPT
from llm import create_chat_model
from memory.seed_data import PERSONA

DEFAULT_COORDINATOR_MODEL = os.getenv("COORDINATOR_MODEL", "claude-sonnet-4-5")


def _format_system_prompt() -> str:
    return COORDINATOR_SYSTEM_PROMPT.format(
        persona_name=PERSONA["display_name"],
        persona_title=PERSONA["title"],
        persona_company=PERSONA["company"],
    )


def create_coordinator_agent(
    model: Optional[BaseChatModel | str] = None,
    *,
    checkpointer=None,
):
    """Build the coordinator (supervisor) agent.

    `model` defaults to Claude Sonnet 4.5 (configurable via the
    COORDINATOR_MODEL env var). Pass a `checkpointer` to persist multi-turn
    conversation state across requests (the Chainlit UI does this).
    """
    if model is None:
        model = DEFAULT_COORDINATOR_MODEL
    if isinstance(model, str):
        model = create_chat_model(model)

    return create_agent(
        model,
        system_prompt=_format_system_prompt(),
        tools=COORDINATOR_TOOLS,
        checkpointer=checkpointer,
    )
