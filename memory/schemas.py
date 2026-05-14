"""Pydantic schemas for memory documents.

These are validation/projection helpers. Documents are stored as plain
BSON dicts so MongoDB can index them directly.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Working memory (short-term, per session)
# ─────────────────────────────────────────────────────────────────────────────


class WorkingMemory(BaseModel):
    session_id: str
    user_id: str
    current_project: Optional[str] = None
    current_task: Optional[str] = None
    focus: Optional[str] = None
    last_action: Optional[str] = None
    scratchpad: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Episodic memory (long-term timeline)
# ─────────────────────────────────────────────────────────────────────────────

EpisodicEventType = Literal[
    "ticket_created",
    "calendar_invite",
    "story_created",
    "feature_created",
    "epic_created",
    "task_status_changed",
    "decision_logged",
    "risk_logged",
    "standup",
    "status_report",
    "sprint_planned",
    "conversation_summary",
    "note",
]


class EpisodicMemory(BaseModel):
    user_id: str
    event_type: EpisodicEventType
    summary: str
    entities: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Literal["agent", "user", "tool", "seed"] = "agent"
    is_seed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Semantic memory (long-term facts)
# ─────────────────────────────────────────────────────────────────────────────

SemanticKind = Literal[
    "person",
    "team",
    "project",
    "stakeholder",
    "glossary",
    "decision",
    "risk",
    "preference",
    "epic",
    "feature",
    "story",
]


class SemanticMemory(BaseModel):
    user_id: str
    kind: SemanticKind
    key: str
    value: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.9
    source: Literal["explicit", "inferred", "seed"] = "explicit"
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_seed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Procedural memory (long-term workflows / templates)
# ─────────────────────────────────────────────────────────────────────────────


class ProceduralStep(BaseModel):
    step: int
    action: str
    description: str = ""


class ProceduralMemory(BaseModel):
    user_id: str
    name: str
    description: str
    steps: list[ProceduralStep] = Field(default_factory=list)
    trigger_examples: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_seed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared memory (short-term inter-agent handoff)
# ─────────────────────────────────────────────────────────────────────────────

SharedSlot = Literal[
    "last_search_results",
    "handoff_payload",
    "disambiguation",
    "scratch",
    "plan",
    "findings",
    "goal",
]

SharedScope = Literal["session", "project", "user"]


class SharedMemory(BaseModel):
    session_id: Optional[str] = None
    slot: SharedSlot
    from_agent: str
    to_agent: str
    payload: dict[str, Any] = Field(default_factory=dict)
    scope: SharedScope = "session"
    project_key: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Chat history (conversational episodic sub-form)
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    session_id: str
    user_id: str
    role: Literal["user", "assistant", "tool", "system"]
    content: Any
    agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
