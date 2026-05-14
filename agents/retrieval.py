"""Retrieval Agent — read-only ReAct agent over memory + mock business state.

Built with `langchain.agents.create_agent` so it slots into the same
runtime as the existing `agent.py` factory (and so the coordinator can
call it the same way).
"""

from __future__ import annotations

import os
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agents.prompts import RETRIEVAL_SYSTEM_PROMPT
from agents.retrieval_tools import RETRIEVAL_TOOLS
from llm import create_chat_model

DEFAULT_RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "claude-haiku-4-5")


def create_retrieval_agent(model: Optional[BaseChatModel | str] = None):
    """Build the retrieval ReAct agent.

    `model` may be a chat model instance or an Anthropic model identifier
    (routed through the configured LLM provider). Defaults to Claude
    Haiku 4.5.
    """
    if model is None:
        model = DEFAULT_RETRIEVAL_MODEL
    if isinstance(model, str):
        model = create_chat_model(model)

    return create_agent(
        model,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        tools=RETRIEVAL_TOOLS,
    )
