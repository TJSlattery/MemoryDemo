"""In-process event bus for memory operations.

Lets the Chainlit UI subscribe to every read/write the MemoryManager
performs and render it as a chip in the memory-trace panel without the
agents knowing the UI exists.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional

MemoryType = Literal[
    "working", "episodic", "semantic", "procedural", "shared"
]
OpKind = Literal["read", "write", "delete"]


@dataclass
class MemoryEvent:
    """One memory operation, emitted to subscribers."""

    op: OpKind
    memory_type: MemoryType
    description: str
    latency_ms: int = 0
    result_count: Optional[int] = None
    payload: dict[str, Any] = field(default_factory=dict)
    # Full doc(s) read or written this op, captured for the UI's inspector
    # panel. None for deletes and for callers that don't supply it.
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    agent: Optional[str] = None


Subscriber = Callable[[MemoryEvent], None]


class MemoryTrace:
    """Thread-safe pub/sub for memory events."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def _unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return _unsubscribe

    def emit(self, event: MemoryEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                # Subscribers must never break the agent loop.
                pass
