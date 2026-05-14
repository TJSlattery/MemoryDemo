"""MongoDB-backed memory subsystem for the PM agent.

Five collections, one per memory type, all sharing the same Atlas
cluster and Voyage embedding model:

    working_memory     short-term, session-scoped scratchpad (TTL)
    episodic_memory    long-term timeline of events
    semantic_memory    long-term facts about people/projects/glossary
    procedural_memory  long-term workflows and templates
    shared_memory      short-term inter-agent handoff slots (TTL)

Multi-turn conversation history is handled by the LangGraph
`MongoDBSaver` checkpointer (collections `checkpoints_aio` and
`checkpoint_writes_aio`), not this subsystem.
"""

from memory.events import MemoryEvent, MemoryTrace
from memory.manager import MemoryManager

__all__ = ["MemoryManager", "MemoryEvent", "MemoryTrace"]
