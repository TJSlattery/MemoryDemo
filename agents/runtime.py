"""Shared runtime wiring for the agent tools.

A single process-wide `MemoryManager` so every agent (and the UI) sees
the same `MemoryTrace`. Tools pull `user_id` / `session_id` out of the
LangGraph configurable just like the original `agent.py` does.
"""

from __future__ import annotations

from typing import Optional

from langgraph.utils.config import get_config

from agents.artifacts import ArtifactTrace
from memory import MemoryManager

_manager: Optional[MemoryManager] = None
_artifact_trace: Optional[ArtifactTrace] = None


def get_memory_manager() -> MemoryManager:
    """Return the process-wide MemoryManager (lazily constructed)."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def set_memory_manager(manager: MemoryManager) -> None:
    """Override the singleton (used by tests / smoke scripts)."""
    global _manager
    _manager = manager


def get_artifact_trace() -> ArtifactTrace:
    """Return the process-wide ArtifactTrace (lazily constructed).

    Tools push `ChartArtifact`s here; the Chainlit UI subscribes per turn.
    """
    global _artifact_trace
    if _artifact_trace is None:
        _artifact_trace = ArtifactTrace()
    return _artifact_trace


def get_user_id(default: str = "admin") -> str:
    cfg = get_config() or {}
    return cfg.get("configurable", {}).get("user_id", default)


def get_session_id(default: str = "default") -> str:
    cfg = get_config() or {}
    configurable = cfg.get("configurable", {})
    # Chainlit's thread_id is a stable per-conversation key.
    return configurable.get("thread_id") or configurable.get("session_id", default)
