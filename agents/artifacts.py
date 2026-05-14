"""In-process bus for visual artifacts (charts, etc.) emitted by tools.

Mirrors `memory.events.MemoryTrace`: tools push, the Chainlit UI subscribes
and renders. Keeping this in `agents/` (not `chainlit_ui/`) preserves the
rule that tool modules never import the UI.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional

ArtifactKind = Literal["plotly"]


@dataclass
class ChartArtifact:
    """One chart produced during a turn, emitted to subscribers."""

    name: str
    figure: Any  # plotly.graph_objects.Figure (kept untyped to avoid import cost)
    summary: str = ""
    kind: ArtifactKind = "plotly"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


Subscriber = Callable[[ChartArtifact], None]


class ArtifactTrace:
    """Thread-safe pub/sub for chart artifacts."""

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

    def emit(self, artifact: ChartArtifact) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(artifact)
            except Exception:
                # Subscribers must never break the agent loop.
                pass


def make_artifact_name(prefix: str) -> str:
    """Stable per-turn unique name so the UI never deduplicates a re-render."""
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


__all__ = ["ChartArtifact", "ArtifactTrace", "make_artifact_name"]


# Optional getter for callers that want to import from this module directly.
def _make_default_trace() -> ArtifactTrace:  # pragma: no cover - trivial
    return ArtifactTrace()
