"""Writer Agent — write-only ReAct agent over memory + mock business state."""

from __future__ import annotations

import os
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agents.prompts import WRITER_SYSTEM_PROMPT
from agents.writer_tools import WRITER_TOOLS
from llm import create_chat_model

DEFAULT_WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-haiku-4-5")


def create_writer_agent(model: Optional[BaseChatModel | str] = None):
    """Build the writer ReAct agent.

    `model` may be a chat model instance or an Anthropic model identifier
    (routed through the configured LLM provider). Defaults to Claude
    Haiku 4.5.
    """
    if model is None:
        model = DEFAULT_WRITER_MODEL
    if isinstance(model, str):
        model = create_chat_model(model)

    return create_agent(
        model,
        system_prompt=WRITER_SYSTEM_PROMPT,
        tools=WRITER_TOOLS,
    )
