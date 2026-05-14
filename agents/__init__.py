"""Multi-agent layer for the PM assistant.

Three agents:
  * retrieval — read-only ReAct agent over memory + mock business state
  * writer    — write-only ReAct agent that mutates memory + mock business state
  * coordinator — supervisor agent (Sonnet) that routes to the two above
"""

from agents.runtime import get_memory_manager, set_memory_manager

__all__ = ["get_memory_manager", "set_memory_manager"]
