"""Coordinator Agent that routes requests to appropriate sub-agents."""

import json
import uuid
import asyncio
import threading
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from bson import ObjectId

from shared.llm import llm_service
from shared.logger import get_logger
from shared.config import settings
from agents.worklog import worklog_agent
from agents.retrieval import retrieval_agent
from agents.mcp_agent import MCPAgent
from utils.context_engineering import compress_tool_result
from config.prompts import get_system_prompt, get_prompt_stats
from memory import MemoryManager
from memory.workflow_executor import WorkflowExecutor

logger = get_logger("coordinator")


def convert_objectids_to_str(obj):
    """Recursively convert ObjectId and datetime instances to strings for JSON serialization."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_objectids_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids_to_str(item) for item in obj]
    else:
        return obj

# Tool definitions for Claude
COORDINATOR_TOOLS = [
    {
        "name": "get_tasks",
        "description": "Get all tasks, optionally filtered by status or project. Use this when user asks 'what are my tasks' or 'show me all tasks'. If user preferences show 'Focus on [Project]', include project_name parameter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "done"],
                    "description": "Filter by task status"
                },
                "project_name": {
                    "type": "string",
                    "description": "Filter by project name"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Filter by task priority"
                }
            }
        }
    },
    {
        "name": "search_tasks",
        "description": "Search for tasks using hybrid search (vector + text). Use for semantic searches combined with filters. Example: 'high-priority memory tasks that are in-progress' → query='memory', priority='high', status='in_progress'. For assignee-specific queries like 'Mike's tasks', use assignee='Mike Chen'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The semantic search query (e.g., 'memory', 'debugging', 'video')"
                },
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "done"],
                    "description": "Optional: Filter results by status"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Optional: Filter results by priority"
                },
                "assignee": {
                    "type": "string",
                    "description": "Optional: Filter results by assignee (partial match, case-insensitive). 'Mike' matches 'Mike Chen', 'sarah' matches 'Sarah Thompson'"
                },
                "project_name": {
                    "type": "string",
                    "description": "Optional: Filter results by project name"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "complete_task",
        "description": "Mark a task as done/completed. Requires task_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "start_task",
        "description": "Mark a task as in_progress (currently working on it). Requires task_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "add_note_to_task",
        "description": "Add a note/context to a task. Requires task_id and note text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                },
                "note": {
                    "type": "string",
                    "description": "The note text to add"
                }
            },
            "required": ["task_id", "note"]
        }
    },
    {
        "name": "get_projects",
        "description": "List all projects with their status and task counts.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "search_projects",
        "description": "Search for projects by name using hybrid search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The project name or search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_tasks_by_time",
        "description": "Get tasks based on when activity happened. Use for temporal questions like 'what did I do today', 'tasks from this week', 'what did I complete yesterday'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timeframe": {
                    "type": "string",
                    "enum": ["today", "yesterday", "this_week", "last_week", "this_month"],
                    "description": "Relative time period to query"
                },
                "activity_type": {
                    "type": "string",
                    "enum": ["created", "started", "completed", "note_added", "updated"],
                    "description": "Type of activity to filter by (optional)"
                },
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "done"],
                    "description": "Current status filter (optional)"
                }
            },
            "required": ["timeframe"]
        }
    },
    {
        "name": "create_task",
        "description": "Create a new task in a project. IMPORTANT: Use assignee, due_date, and blockers parameters when mentioned - do NOT put them in context. Returns task_id in result which can be used with start_task, complete_task, or add_note_to_task for multi-step operations (e.g., create then start).",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Task title"
                },
                "project_name": {
                    "type": "string",
                    "description": "Project to add task to (required - look up by name using search_projects if needed)"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Task priority (default: medium)"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context/description for the task. Do NOT use this for assignee, due_date, or blockers."
                },
                "assignee": {
                    "type": "string",
                    "description": "Person or team responsible. REQUIRED when user says 'assign to X'. Use full name (e.g., 'Sarah Thompson'). Do NOT put in context."
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date when user says 'due X'. Pass natural language as-is (e.g., 'in 5 days', 'tomorrow'). Do NOT put in context."
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of blockers preventing progress. Do NOT put in context."
                }
            },
            "required": ["title", "project_name"]
        }
    },
    {
        "name": "update_task",
        "description": "Update an existing task's fields including assignee, due_date, and blockers. Do NOT put these in context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                },
                "title": {
                    "type": "string",
                    "description": "New task title"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "New priority"
                },
                "context": {
                    "type": "string",
                    "description": "New or additional context. Do NOT use for assignee, due_date, or blockers."
                },
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "done"],
                    "description": "New status"
                },
                "project_name": {
                    "type": "string",
                    "description": "Move task to this project (will look up by name)"
                },
                "assignee": {
                    "type": "string",
                    "description": "New assignee. Use when changing who is responsible. Do NOT put in context."
                },
                "due_date": {
                    "type": "string",
                    "description": "New due date in natural language (e.g., 'in 5 days', 'tomorrow'). Do NOT put in context."
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated list of blockers (replaces existing). Do NOT put in context."
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "stop_task",
        "description": "Stop working on a task and mark it as todo. Use when user says 'I stopped working on X' or 'pause X'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "create_project",
        "description": "Create a new project. Use when user says 'create a project', 'make a new project', 'start a project called X'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Project name"
                },
                "description": {
                    "type": "string",
                    "description": "Project description"
                },
                "context": {
                    "type": "string",
                    "description": "Rich context about the project. Include relevant research from recent conversation, semantic knowledge, or web searches. This helps provide background and rationale for the project."
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "update_project",
        "description": "Update an existing project's name, description, context, or status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                },
                "name": {
                    "type": "string",
                    "description": "New project name"
                },
                "description": {
                    "type": "string",
                    "description": "New description"
                },
                "context": {
                    "type": "string",
                    "description": "New or additional context"
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "archived"],
                    "description": "Project status"
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "add_note_to_project",
        "description": "Add a note to a project. Use for general notes, updates, or observations about the project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                },
                "note": {
                    "type": "string",
                    "description": "The note text to add"
                }
            },
            "required": ["project_id", "note"]
        }
    },
    {
        "name": "add_context_to_task",
        "description": "Add or update rich context for a task. Context is different from notes - it's the main descriptive content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                },
                "context": {
                    "type": "string",
                    "description": "Context to add or update"
                }
            },
            "required": ["task_id", "context"]
        }
    },
    {
        "name": "add_context_to_project",
        "description": "Add or update rich context for a project. Context is different from notes - it's the main descriptive content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                },
                "context": {
                    "type": "string",
                    "description": "Context to add or update"
                }
            },
            "required": ["project_id", "context"]
        }
    },
    {
        "name": "add_decision_to_project",
        "description": "Record an important decision made about the project. Use for architectural decisions, technology choices, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                },
                "decision": {
                    "type": "string",
                    "description": "The decision to record"
                }
            },
            "required": ["project_id", "decision"]
        }
    },
    {
        "name": "add_method_to_project",
        "description": "Add a technology, method, or approach being used in the project. Use for tracking tech stack, methodologies, frameworks, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                },
                "method": {
                    "type": "string",
                    "description": "Technology, method, or approach to add"
                }
            },
            "required": ["project_id", "method"]
        }
    },
    {
        "name": "get_task",
        "description": "Get details of a single task by ID. Use when you have the task_id and need full details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the task"
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "get_project",
        "description": "Get details of a single project by ID. Use when you have the project_id and need full details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "MongoDB ObjectId of the project"
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "get_project_by_name",
        "description": "Get a project by exact name match (case-insensitive). Faster and more precise than search_projects when you know the exact project name. Example: 'Alpha' matches 'Project Alpha'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Exact name of the project (case-insensitive)"
                }
            },
            "required": ["project_name"]
        }
    },
    {
        "name": "get_action_history",
        "description": """Get history of past actions from long-term memory. Supports both filter-based and semantic search.

FILTER MODE (when no semantic_query):
- "What did I do today/yesterday/this week?"
- "What have I completed?"
- "What tasks did I start?"
- "Summarize my activity"

SEMANTIC MODE (when semantic_query provided):
- "Find tasks related to debugging"
- "Show me memory-related work"
- "What have I done on voice agents?"
- "Search for API integration tasks"

Only available when long-term memory is enabled.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "semantic_query": {
                    "type": "string",
                    "description": "Natural language search query for semantic/vector search over action history. Use when user wants to find actions related to specific topics (e.g., 'debugging', 'memory', 'API'). Mutually exclusive with time_range filters - semantic mode takes precedence."
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range to query (filter mode only, ignored if semantic_query provided)",
                    "enum": ["today", "yesterday", "this_week", "last_week", "this_month", "all"]
                },
                "action_type": {
                    "type": "string",
                    "description": "Filter by action type (works in both modes)",
                    "enum": ["complete", "start", "stop", "create", "update", "add_note", "search", "all"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10
                },
                "summarize": {
                    "type": "boolean",
                    "description": "Return activity summary instead of detailed list. Use for 'summarize my activity' type queries. Returns both pre-formatted narrative and raw stats for flexibility.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "resolve_disambiguation",
        "description": """Resolve a pending disambiguation by selecting one of the numbered options shown in context.
Use this when:
- User says "the first one", "the second one", "number 3", etc.
- User refers to a result by its position in the last search
- Multiple results were shown and user is picking one

The selection number is 1-based (1 = first result, 2 = second result, etc.)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "integer",
                    "description": "1-based selection number (1 = first, 2 = second, 3 = third, etc.)",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["selection"]
        }
    },
    {
        "name": "search_knowledge",
        "description": """Search cached knowledge from previous research, web searches, and learning. Use this to check 'what we know about X' BEFORE doing new research or calling external tools.

Examples:
- "What do we know about gaming use cases?"
- "What have we learned about NPC memory systems?"
- "Show me our research on voice agents"

CRITICAL CITATION REQUIREMENTS:
1. Present ALL knowledge results returned (typically 5-10 entries), not just the top match
2. Each result MUST have source citation: "📚 From [topic] (source: [tavily/knowledge_base], score: X.XX):"
3. Examples:
   - "📚 From warehouse_robotics (source: tavily, score: 1.00, cached: Jan 15): [content]"
   - "📚 From autonomous_agents (source: knowledge_base, score: 0.64): [content]"
   - "📚 From multi_agent_systems (source: knowledge_base, score: 0.65): [content]"
4. NEVER present cached info without citation - proves it came from memory, not your training

For research queries, be comprehensive:
- Present ALL knowledge entries (even tangential ones with score >0.60)
- Also search projects: call search_projects with the query
- Also search tasks: call search_tasks with the query
- Format: "Related Projects: [name]" and "Related Tasks: [title]"

This shows the full context available and proves you're using stored memory.

Returns cached knowledge with source, confidence, similarity scores, and timestamps.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in cached knowledge"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "analyze_tool_discoveries",
        "description": """Analyze patterns in tool discoveries to suggest new features, Atlas optimizations, and templates. Shows what should be built next based on user behavior and MCP tool usage patterns.

Examples:
- "What patterns do you see in recent tool discoveries?"
- "What should we build next based on my usage?"
- "Analyze my tool discovery history"

Returns:
- Suggested new tools (repeated requests that could become built-in)
- Atlas optimization opportunities (indexes, collections)
- Template candidates (repeated workflows)
- Feature gaps (failed or missing functionality)

Only available when long-term memory is enabled.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze (default 7)",
                    "default": 7
                }
            },
            "required": []
        }
    },
    {
        "name": "list_templates",
        "description": "List all available project templates with their phases and task counts. Use when user asks 'what templates do I have', 'show me templates', 'list templates', or 'what project templates are available'.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "remember_about_user",
        "description": """Persist a fact about the user (name, preferred name, timezone, location, role, working hours, email, pronouns, etc.) so it can be recalled in future sessions.

Use this whenever the user reveals a stable personal detail, either explicitly ("My name is Tom", "Remember that I'm in London") or implicitly when it's clearly self-referential and unambiguous ("I'm an engineer at Acme", "Call me Tomek"). Do NOT use it for transient state (current task, current project) — those go in working memory automatically.

Standard keys (prefer these when applicable): name, preferred_name, pronouns, timezone, location, role, company, email, working_hours. Arbitrary keys are allowed for other stable facts.

After storing, briefly acknowledge in your reply (e.g. "Got it, I'll remember that.") only the first time you learn a fact. Don't re-acknowledge on every mention.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Snake_case identifier for the fact (e.g. 'name', 'timezone', 'role')."
                },
                "value": {
                    "type": "string",
                    "description": "The fact value as a short string (e.g. 'Tom', 'Europe/London', 'Senior engineer')."
                },
                "source": {
                    "type": "string",
                    "enum": ["explicit", "inferred"],
                    "description": "'explicit' if the user stated it directly, 'inferred' if you deduced it from context.",
                    "default": "explicit"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "forget_about_user",
        "description": "Delete a previously stored profile fact about the user. Use when the user says 'forget my X', 'I'm no longer X', or corrects you about a stale fact. Provide the same key that was used with remember_about_user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key of the profile fact to delete (e.g. 'email')."
                }
            },
            "required": ["key"]
        }
    }
]

# System prompts are now defined in config/prompts.py
# Use get_system_prompt(streamlined=True/False) to retrieve them


class CoordinatorAgent:
    """Coordinator agent that routes user requests to specialized agents using tool use."""

    def __init__(self, memory_manager=None, db=None):
        self.llm = llm_service
        self.worklog_agent = worklog_agent
        self.retrieval_agent = retrieval_agent
        self.memory = memory_manager  # Memory manager for long-term action history
        self.last_debug_info = []  # Store debug info from last request (deprecated - use current_turn)
        self.current_turn = None  # Current conversation turn with all tool calls

        # Memory configuration
        self.memory_config = {
            "short_term": True,
            "long_term": True,
            "shared": True,
            "context_injection": True
        }

        # Session tracking
        self.session_id = None
        self.user_id = "default_user"
        self.current_chain_id = None
        self.memory_ops = {}  # Track memory operations for debug panel

        # MCP Agent (lazy initialized)
        self.db = db
        self.mcp_agent: Optional[MCPAgent] = None
        self.mcp_mode_enabled = settings.mcp_mode_enabled  # Can be toggled via UI

        # Persistent event loop for async MCP operations
        self._async_loop = None
        self._async_thread = None
        self._loop_ready = threading.Event()

    def set_session(self, session_id: str, user_id: str = None):
        """Set session context for memory isolation.

        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
        """
        self.session_id = session_id
        if user_id:
            self.user_id = user_id

    def _ensure_event_loop(self):
        """Ensure there's a running event loop for async MCP operations."""
        if self._async_loop is None or not self._async_loop.is_running():
            def run_loop():
                """Run event loop in background thread."""
                self._async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._async_loop)
                self._loop_ready.set()
                self._async_loop.run_forever()

            self._async_thread = threading.Thread(target=run_loop, daemon=True)
            self._async_thread.start()
            self._loop_ready.wait()  # Wait for loop to be ready
            logger.debug("Started persistent event loop for async operations")

    def _run_async(self, coro):
        """Run an async coroutine on the persistent event loop.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        self._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()

    async def enable_mcp_mode(self):
        """Initialize and enable MCP mode"""
        if self.mcp_agent is None or not self.mcp_agent._initialized:
            if self.db is None:
                return {
                    "success": False,
                    "error": "Database connection not available for MCP Agent"
                }

            from shared.embeddings import embed_query
            self.mcp_agent = MCPAgent(
                db=self.db,
                memory_manager=self.memory,
                embedding_fn=embed_query
            )
            try:
                await self.mcp_agent.initialize()
            except Exception as e:
                # Reset agent on failed initialization
                self.mcp_agent = None
                raise

        self.mcp_mode_enabled = True
        status = self.mcp_agent.get_status()
        logger.info(f"🔬 MCP Mode enabled: {status}")
        return {"success": True, **status}

    def disable_mcp_mode(self):
        """Disable MCP mode (keep connections for reuse)"""
        self.mcp_mode_enabled = False
        logger.info("🔬 MCP Mode disabled (connections preserved)")
        return {"success": True, "mcp_mode_enabled": False}

    def _build_context_injection(self) -> str:
        """
        Build context injection section for system prompt.

        Combines:
        - Working Memory (short-term): current project/task/action
        - Semantic Memory (long-term): preferences
        - Procedural Memory (long-term): rules
        - Disambiguation (short-term): pending selections
        """

        if not self.memory_config.get("context_injection"):
            return ""

        if not self.session_id:
            return ""

        parts = []

        # ═══════════════════════════════════════════════════════════════
        # WORKING MEMORY (Short-term session context)
        # ═══════════════════════════════════════════════════════════════

        session_context = self.memory.read_session_context(self.session_id)
        if session_context:
            if session_context.get("current_project"):
                parts.append(f"Current project: {session_context['current_project']}")

            if session_context.get("current_task"):
                parts.append(f"Current task: {session_context['current_task']}")

            if session_context.get("last_action"):
                parts.append(f"Last action: {session_context['last_action']}")

        # ═══════════════════════════════════════════════════════════════
        # SEMANTIC MEMORY (Long-term user profile - identity facts)
        # ═══════════════════════════════════════════════════════════════

        profile_facts = self.memory.get_profile(self.user_id, min_confidence=0.5)
        if profile_facts:
            parts.append("")  # Blank line separator
            parts.append("User profile (Semantic Memory - persistent facts):")
            for fact in profile_facts[:10]:
                parts.append(f"  • {fact['key']}: {fact['value']}")

        # ═══════════════════════════════════════════════════════════════
        # SEMANTIC MEMORY (Long-term preferences)
        # ═══════════════════════════════════════════════════════════════

        preferences = self.memory.get_preferences(self.user_id, min_confidence=0.5)
        if preferences:
            parts.append("")  # Blank line separator
            parts.append("User preferences (Semantic Memory):")
            for pref in preferences[:5]:  # Limit to top 5
                parts.append(f"  • {pref['key']}: {pref['value']}")

        # ═══════════════════════════════════════════════════════════════
        # PROCEDURAL MEMORY (Long-term rules)
        # ═══════════════════════════════════════════════════════════════

        rules = self.memory.get_rules(self.user_id, min_confidence=0.5)
        if rules:
            parts.append("")  # Blank line separator
            parts.append("User rules (Procedural Memory):")
            for rule in rules[:5]:  # Limit to top 5
                action_descriptions = {
                    "complete_current_task": "complete the current task",
                    "start_next_task": "start the next task",
                    "stop_current_task": "stop/pause the current task",
                    "skip_current_task": "skip the current task"
                }
                action_desc = action_descriptions.get(rule['action_type'], rule['action_type'])
                parts.append(f"  • When user says \"{rule['trigger_pattern']}\" → {action_desc}")

        # ═══════════════════════════════════════════════════════════════
        # WORKFLOWS (Procedural Memory - Multi-step patterns)
        # ═══════════════════════════════════════════════════════════════

        workflows = self.memory.get_workflows(self.user_id)
        if workflows:
            parts.append("")  # Blank line separator
            parts.append("Available Workflows (Procedural Memory):")
            for workflow in workflows[:3]:  # Limit to top 3 most relevant
                parts.append(f"  • {workflow['name']}: {workflow['description']}")
                steps = workflow.get('workflow', {}).get('steps', [])
                if steps:
                    parts.append(f"    Steps:")
                    for step in steps[:3]:  # Show first 3 steps
                        action = step.get('action', 'unknown')
                        desc = step.get('description', '')
                        if desc:
                            parts.append(f"      {step['step']}. {action} - {desc}")
                        else:
                            parts.append(f"      {step['step']}. {action}")
                examples = workflow.get('examples', [])
                if examples:
                    parts.append(f"    Examples: \"{examples[0]}\"")
                parts.append("")  # Blank line after each workflow

        # ═══════════════════════════════════════════════════════════════
        # DISAMBIGUATION (Short-term pending selections)
        # ═══════════════════════════════════════════════════════════════

        disambiguation = self.memory.get_pending_disambiguation(self.session_id)
        if disambiguation:
            parts.append("")  # Blank line separator
            parts.append(f"Pending selection from search \"{disambiguation.get('query', '')}\":")
            for r in disambiguation.get("results", []):
                idx = r.get('index', 0) + 1
                title = r.get('title', 'Unknown')
                project = r.get('project', 'Unknown')
                parts.append(f"  {idx}. {title} ({project})")
            parts.append("User may refer to these by number (e.g., 'the first one', 'number 2').")

        if not parts:
            return ""

        self.memory_ops["context_injected"] = True

        # Build the injection string
        context_str = "\n".join(parts)

        return f"""

<memory_context>
{context_str}
</memory_context>

Use this memory context to:
- Filter queries to the current project when relevant
- Apply user preferences (e.g., focus_project filters task queries)
- Execute user rules when trigger words are detected
- Resolve references like "it", "that task", "the first one", "number 2"
- Do NOT mention the memory system to the user unless asked
"""

    def _check_rule_triggers(self, user_message: str) -> dict:
        """
        Check if user message matches any stored rule triggers.

        Checks long-term procedural memory for matching rules.

        Args:
            user_message: User's message to check

        Returns:
            Dict with matched_rule if found, None otherwise
        """
        if not self.memory or not self.user_id:
            return None

        msg_lower = user_message.lower()

        # Get rules from long-term procedural memory
        rules = self.memory.get_rules(self.user_id, min_confidence=0.5)
        if not rules:
            return None

        # Check each rule's trigger
        for rule in rules:
            trigger = rule.get("trigger_pattern", "").lower()
            if not trigger:
                continue

            # Check if trigger appears in message
            # Support both exact phrase and word matching
            if trigger in msg_lower:
                return {
                    "matched": True,
                    "trigger": trigger,
                    "action": rule.get("action_type"),
                    "original_rule": rule
                }

        return None

    def _extract_context_from_turn(self, user_message: str,
                                    tool_calls: list,
                                    tool_results: list) -> dict:
        """
        Extract context updates from completed turn.

        Working Memory (short-term): current project/task/action
        Semantic Memory (long-term): preferences
        Procedural Memory (long-term): rules

        Args:
            user_message: User's message
            tool_calls: List of tool calls made
            tool_results: List of tool results

        Returns:
            Dict of context updates to apply (working memory only)
        """
        updates = {}  # For short-term working memory only

        # ═══════════════════════════════════════════════════════════════
        # WORKING MEMORY: Extract from tool calls (short-term)
        # ═══════════════════════════════════════════════════════════════

        for i, call in enumerate(tool_calls or []):
            tool_name = call.get("name", "")
            tool_input = call.get("input", {})
            result = tool_results[i] if i < len(tool_results or []) else {}

            # Project context
            if tool_name in ["get_tasks", "get_project", "search_tasks", "search_projects"]:
                if tool_input.get("project_name"):
                    updates["current_project"] = tool_input["project_name"]
                elif isinstance(result, dict) and result.get("project_name"):
                    updates["current_project"] = result["project_name"]

            # Task context from operations
            if tool_name in ["start_task", "complete_task", "stop_task", "get_task"]:
                if isinstance(result, dict):
                    if result.get("task_title"):
                        updates["current_task"] = result["task_title"]
                        updates["current_task_id"] = result.get("task_id")
                    if result.get("project_name"):
                        updates["current_project"] = result["project_name"]

                # Track action
                action_map = {
                    "start_task": "start",
                    "complete_task": "complete",
                    "stop_task": "stop"
                }
                if tool_name in action_map:
                    updates["last_action"] = action_map[tool_name]

            # Store search results for disambiguation (CR-MH)
            if tool_name in ["search_tasks", "search_projects"]:
                if isinstance(result, dict):
                    results_list = result.get("results", [])
                    if len(results_list) > 1:
                        # Store for "the first one" type references
                        self.memory.store_disambiguation(
                            session_id=self.session_id,
                            query=tool_input.get("query", ""),
                            results=[
                                {
                                    "task_id": str(r.get("_id", r.get("task_id", ""))),
                                    "title": r.get("title", ""),
                                    "project": r.get("project_name", "")
                                }
                                for r in results_list[:5]
                            ],
                            source_agent="coordinator"
                        )

        # ═══════════════════════════════════════════════════════════════
        # SEMANTIC MEMORY: Extract preferences (long-term)
        # ═══════════════════════════════════════════════════════════════

        msg_lower = user_message.lower()
        import re

        # Focus/context setting patterns (expanded for natural language)
        focus_patterns = [
            ("i'm focusing on", "explicit"),
            ("i am focusing on", "explicit"),
            ("focusing on", "explicit"),
            ("focus on", "explicit"),
            ("i'm working on", "explicit"),
            ("i am working on", "explicit"),
            ("working on", "inferred"),
            ("let's work on", "explicit"),
            ("i want to work on", "explicit"),
            ("switch to", "explicit"),
            ("i'm building", "explicit"),
            ("i am building", "explicit"),
            ("building", "inferred"),
            ("set focus to", "explicit"),
            ("set context to", "explicit"),
            ("my focus is", "explicit"),
            ("my context is", "explicit")
        ]

        for pattern, source in focus_patterns:
            if pattern in msg_lower:
                # Extract focus/context (better extraction with regex)
                rest_of_message = msg_lower.split(pattern, 1)[1].strip()

                # Try to extract quoted text first (e.g., "gaming demo")
                quote_match = re.search(r'["\']([^"\']+)["\']', rest_of_message)
                if quote_match:
                    focus_text = quote_match.group(1)
                else:
                    # Extract until punctuation or common stop words
                    words = rest_of_message.split()
                    stop_words = {'for', 'with', 'using', 'and', 'or', 'but', 'because'}
                    focus_words = []
                    for word in words[:5]:  # Limit to 5 words
                        cleaned = word.strip(".,!?;:")
                        if cleaned in stop_words:
                            break
                        focus_words.append(cleaned)
                    focus_text = " ".join(focus_words)

                if focus_text and len(focus_text) > 1:
                    # Update WORKING MEMORY (session context) immediately
                    updates["focus"] = focus_text
                    updates["context"] = focus_text

                    # Also try to detect if it's a project name
                    # (contains words like "project", "demo", "app", etc.)
                    if any(word in focus_text for word in ["project", "demo", "app", "system"]):
                        updates["current_project"] = focus_text

                    # Save to long-term semantic memory
                    if self.memory and self.user_id:
                        self.memory.record_preference(
                            user_id=self.user_id,
                            key="focus_context",
                            value=focus_text,
                            source=source,
                            confidence=0.9 if source == "explicit" else 0.7
                        )
                        self.memory_ops["preference_recorded"] = True

                        logger.info(
                            f"📌 Context set via natural language: '{focus_text}' "
                            f"(pattern: '{pattern}')"
                        )
                break

        # Priority preference
        if "high priority" in msg_lower or "only important" in msg_lower:
            if self.memory and self.user_id:
                self.memory.record_preference(
                    user_id=self.user_id,
                    key="priority_filter",
                    value="high",
                    source="explicit",
                    confidence=0.85
                )
                self.memory_ops["preference_recorded"] = True
        elif "all priorities" in msg_lower or "any priority" in msg_lower:
            if self.memory and self.user_id:
                self.memory.record_preference(
                    user_id=self.user_id,
                    key="priority_filter",
                    value="all",
                    source="explicit",
                    confidence=0.85
                )
                self.memory_ops["preference_recorded"] = True

        # ═══════════════════════════════════════════════════════════════
        # PROCEDURAL MEMORY: Extract rules (long-term)
        # ═══════════════════════════════════════════════════════════════

        rule_patterns = [
            (r"when (?:i say |i type |i write )?['\"]?([^'\",]+)['\"]?,?\s+(?:do |then )?(.+)", "when"),
            (r"whenever (?:i say |i type )?['\"]?([^'\",]+)['\"]?,?\s+(?:do |then )?(.+)", "whenever"),
            (r"if i say ['\"]?([^'\",]+)['\"]?,?\s+(?:do |then )?(.+)", "if"),
            (r"always (.+) when (.+)", "always")
        ]

        import re
        for pattern, pattern_type in rule_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                if pattern_type == "always":
                    # Reversed order: "always ACTION when TRIGGER"
                    action_text = match.group(1).strip()
                    trigger = match.group(2).strip()
                else:
                    # Normal order: "when TRIGGER, do ACTION"
                    trigger = match.group(1).strip()
                    action_text = match.group(2).strip()

                # Clean up action text (remove trailing punctuation)
                action_text = action_text.rstrip(".,;")
                trigger = trigger.rstrip(".,;")

                # Map action text to action type
                action_map = {
                    "complete": "complete_current_task",
                    "finish": "complete_current_task",
                    "done": "complete_current_task",
                    "start": "start_next_task",
                    "next": "start_next_task",
                    "skip": "skip_current_task",
                    "stop": "stop_current_task",
                    "pause": "stop_current_task"
                }

                action_type = action_text  # Default to full text
                for key, mapped_action in action_map.items():
                    if key in action_text:
                        action_type = mapped_action
                        break

                # Save to long-term procedural memory
                if self.memory and self.user_id:
                    self.memory.record_rule(
                        user_id=self.user_id,
                        trigger=trigger,
                        action=action_type,
                        source="explicit",
                        confidence=0.85
                    )
                    self.memory_ops["rule_recorded"] = True
                break  # Only extract first rule per message

        # ═══════════════════════════════════════════════════════════════
        # CONTEXT SWITCH DETECTION (clears working memory)
        # ═══════════════════════════════════════════════════════════════

        switch_patterns = ["switch to", "actually,", "no wait", "change to", "never mind"]
        for pattern in switch_patterns:
            if pattern in msg_lower:
                # Clear task-level context on project switch
                if updates.get("current_project"):
                    updates["current_task"] = None
                    updates["current_task_id"] = None
                    updates["last_action"] = None
                break

        return updates

    def _summarize_output(self, result: Any) -> str:
        """
        Create a brief summary of tool output for debug display.

        Args:
            result: The tool result to summarize

        Returns:
            Brief string summary
        """
        if result is None:
            return "None"
        elif isinstance(result, dict):
            if "tasks" in result:
                return f"{result.get('count', 0)} tasks returned"
            elif "projects" in result:
                return f"{result.get('count', 0)} projects returned"
            elif "success" in result:
                if result["success"]:
                    return "Success"
                else:
                    return f"Error: {result.get('error', 'Unknown')}"
            else:
                # Truncate dict representation
                return str(result)[:200]
        elif isinstance(result, list):
            return f"{len(result)} items returned"
        else:
            return str(result)[:200]

    def _record_action(
        self,
        action_type: str,
        entity_type: str,
        entity: dict,
        metadata: dict = None,
        handoff_id: str = None
    ):
        """Record action to long-term memory using new API.

        Args:
            action_type: Action type (e.g., "complete", "start", "create", "add_note")
            entity_type: Entity type ("task", "project", "search")
            entity: Entity data dict
            metadata: Additional metadata
            handoff_id: Optional handoff ID for agent chains
        """
        if not self.memory or not self.memory_config.get("long_term"):
            return

        if not self.session_id:
            return

        # Record using new Memory Manager API
        action_id = self.memory.record_action(
            user_id=self.user_id,
            session_id=self.session_id,
            action_type=action_type,
            entity_type=entity_type,
            entity=entity,
            metadata=metadata or {},
            source_agent="coordinator",  # Future: will vary by agent
            triggered_by="agent_handoff" if handoff_id else "user",
            handoff_id=handoff_id,
            generate_embedding=True
        )

        # Track for debug
        self.memory_ops["action_recorded"] = True
        self.memory_ops["recorded_action_type"] = action_type

        logger.debug(f"Recorded action: {action_type} on {entity_type} (id={action_id})")

    def _can_static_tools_handle(self, intent: str, user_message: str) -> bool:
        """
        Returns True if our static tools can handle this request.
        Returns False if we should try MCP Agent.

        Args:
            intent: Classified intent
            user_message: User's message

        Returns:
            True if static tools can handle, False if MCP Agent should try
        """
        # Intents that static tools handle well
        static_intents = [
            "create_task", "update_task", "complete_task", "delete_task",
            "start_task", "stop_task",
            "create_project", "update_project",
            "list_tasks", "list_projects", "search_tasks", "search_projects",
            "get_history", "add_note", "add_context",
            "get_task", "get_project", "get_tasks_by_time"
        ]

        if intent in static_intents:
            return True

        # Intents that need MCP (external tool discovery - Tier 4)
        # NOTE: "unknown" is NOT in this list - unknown intents should be handled
        # by the LLM with built-in tools (Tier 3), not require MCP mode (Tier 4)
        mcp_intents = [
            "research", "web_search", "find_information",
            "complex_query", "aggregation", "data_extraction",
            "advanced_mongodb_query"  # Complex aggregation pipelines, analytics via MongoDB MCP
        ]

        if intent in mcp_intents:
            return False

        # Default to static tools (including LLM with built-in tools for "unknown")
        return True

    def _classify_intent(self, user_message: str) -> str:
        """
        Classify user intent from message.

        Args:
            user_message: User's message

        Returns:
            Intent string
        """
        msg_lower = user_message.lower()

        # Task operations
        if any(word in msg_lower for word in ["create task", "add task", "new task", "make task"]):
            return "create_task"
        if any(word in msg_lower for word in ["complete", "done", "finish"]):
            return "complete_task"
        if any(word in msg_lower for word in ["start", "begin", "working on"]):
            return "start_task"
        if any(word in msg_lower for word in ["stop", "pause"]):
            return "stop_task"

        # Project operations
        if any(word in msg_lower for word in ["create project", "new project"]):
            return "create_project"

        # Queries
        if any(word in msg_lower for word in ["show me", "list", "get tasks", "what are my tasks"]):
            return "list_tasks"
        if any(word in msg_lower for word in ["search for", "find"]):
            return "search_tasks"

        # Research/web search
        if any(word in msg_lower for word in ["search the web", "look up", "research", "find information about", "what's the latest"]):
            return "web_search"

        # Advanced MongoDB query generation (aggregation pipelines, complex analytics)
        if any(word in msg_lower for word in ["aggregation pipeline", "generate query", "build query", "create query", "mongodb query", "analyze completion rates", "time-series analysis"]):
            return "advanced_mongodb_query"

        # Default to unknown (will use Tier 3 built-in LLM agents)
        return "unknown"

    def _build_tool_registry(self) -> Dict[str, Any]:
        """
        Build a tool registry mapping action names to callable functions.

        Returns:
            Dict mapping action name to callable function
        """
        from bson import ObjectId

        return {
            # Task operations
            "create_task": lambda **kwargs: self.worklog_agent._create_task(**kwargs),
            "update_task": lambda **kwargs: self.worklog_agent._update_task(**kwargs),
            "start_task": lambda task_id: self.worklog_agent._update_task(task_id, status="in_progress"),
            "complete_task": lambda task_id: self.worklog_agent._complete_task(task_id),
            "stop_task": lambda task_id: self.worklog_agent._update_task(task_id, status="todo"),
            "get_task": lambda task_id: self.worklog_agent._get_task(task_id),
            "add_note_to_task": lambda task_id, note: self.worklog_agent._add_note("task", task_id, note),

            # Project operations
            "create_project": lambda **kwargs: self.worklog_agent._create_project(**kwargs),
            "update_project": lambda **kwargs: self.worklog_agent._update_project(**kwargs),
            "get_project": lambda project_id: self.worklog_agent._get_project(project_id),

            # Search operations - use retrieval agent for search
            "search_tasks": lambda query, limit=10, **kwargs: self.retrieval_agent.hybrid_search_tasks(query, limit, **kwargs),
            "search_projects": lambda query, limit=5: self.retrieval_agent.hybrid_search_projects(query, limit),
            "get_tasks": lambda **kwargs: self.worklog_agent._list_tasks(**kwargs),
            "get_projects": lambda **kwargs: self.worklog_agent._list_projects(**kwargs),
        }

    def _classify_multi_step_intent(self, user_message: str) -> dict:
        """
        Detect if message contains multiple intents that need sequential execution.

        This enables complex requests like:
        - "Research the gaming market and create a GTM project with tasks"
        - "Look up MongoDB features then make a project for it"
        - "Find information about AI trends and then create tasks"

        Args:
            user_message: User's message

        Returns:
            {
                "is_multi_step": bool,
                "steps": [
                    {"intent": "research", "description": "Research gaming market"},
                    {"intent": "create_project", "description": "Create GTM project"},
                    {"intent": "generate_tasks", "description": "Generate tasks from template"}
                ]
            }
        """
        msg_lower = user_message.lower()

        # Keywords that indicate multi-step requests
        multi_step_indicators = [
            " and ",
            " then ",
            ", then ",
            " and then ",
            " followed by ",
            " after that ",
        ]

        # Check for sequential indicators
        has_sequential_indicator = any(indicator in msg_lower for indicator in multi_step_indicators)

        # Common multi-step patterns
        research_create_pattern = any([
            ("research" in msg_lower or "look up" in msg_lower or "find out" in msg_lower or "find information" in msg_lower)
            and ("create" in msg_lower or "make" in msg_lower or "add" in msg_lower),
        ])

        # If no multi-step indicators, return early
        if not (has_sequential_indicator and research_create_pattern):
            return {"is_multi_step": False, "steps": []}

        # Use LLM to parse the steps
        logger.info(f"Detected potential multi-step request: {user_message}")

        prompt = f"""Parse this user request into sequential steps.

User Request: "{user_message}"

Analyze the request and break it into sequential steps. Each step should have:
- intent: One of [research, create_project, generate_tasks, create_task, update_task, other]
- description: Clear description of what to do in this step

Respond with ONLY valid JSON in this exact format:
{{
    "steps": [
        {{"intent": "research", "description": "Research gaming market trends and opportunities"}},
        {{"intent": "create_project", "description": "Create GTM project for gaming market"}},
        {{"intent": "generate_tasks", "description": "Generate tasks from GTM template"}}
    ]
}}

Example inputs and outputs:

Input: "Research gaming market and create a GTM project with tasks"
Output:
{{
    "steps": [
        {{"intent": "research", "description": "Research gaming market trends and opportunities"}},
        {{"intent": "create_project", "description": "Create GTM project for gaming vertical"}},
        {{"intent": "generate_tasks", "description": "Generate tasks from GTM Roadmap Template"}}
    ]
}}

Input: "Look up MongoDB features then make a project for it"
Output:
{{
    "steps": [
        {{"intent": "research", "description": "Research MongoDB features and capabilities"}},
        {{"intent": "create_project", "description": "Create project for MongoDB integration"}}
    ]
}}

Now parse the actual user request above. Respond with ONLY the JSON, no other text."""

        try:
            # Use LLM to parse steps with low temperature for consistency
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic parsing
                max_tokens=1000
            )

            # Clean the response - remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Remove ```json or ``` markers
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1] if len(lines) > 2 else lines)

            # Parse JSON
            parsed = json.loads(response_clean)

            if "steps" in parsed and len(parsed["steps"]) > 1:
                logger.info(f"Parsed {len(parsed['steps'])} steps from multi-step request")
                return {
                    "is_multi_step": True,
                    "steps": parsed["steps"]
                }
            else:
                logger.info("LLM did not identify multiple steps")
                return {"is_multi_step": False, "steps": []}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse multi-step response as JSON: {e}")
            logger.error(f"Response was: {response}")
            return {"is_multi_step": False, "steps": []}
        except Exception as e:
            logger.error(f"Error parsing multi-step intent: {e}")
            return {"is_multi_step": False, "steps": []}

    async def _execute_multi_step(
        self,
        steps: List[dict],
        original_request: str,
        user_id: str
    ) -> dict:
        """
        Execute multiple steps sequentially, passing context between them.

        This enables complex workflows like:
        1. Research (via MCP)
        2. Create project
        3. Generate tasks from procedural memory template

        Args:
            steps: List of step dicts with "intent" and "description"
            original_request: Original user request
            user_id: User identifier

        Returns:
            Combined results from all steps with success status
        """
        results = []
        context = {
            "original_request": original_request,
            "user_id": user_id
        }

        logger.info(f"Executing {len(steps)} steps for multi-step workflow")

        # Check if request mentions using previous research
        if "research we just did" in original_request.lower() or "recent research" in original_request.lower():
            logger.info("Request mentions previous research - checking semantic memory for recent results")
            # Search for recent Tavily research in semantic memory
            if self.memory and user_id:
                recent_research = self.memory.get_recent_knowledge(user_id, limit=1)
                if recent_research:
                    context["research_results"] = recent_research[0].get("result") or recent_research[0].get("value", "")
                    context["research_source"] = recent_research[0].get("source", "semantic_cache")
                    context["research_query"] = recent_research[0].get("query") or recent_research[0].get("key", "")
                    logger.info(f"✓ Found recent research: {context.get('research_query', 'unknown')[:50]}...")

        for i, step in enumerate(steps):
            step_num = i + 1
            logger.info(f"Step {step_num}/{len(steps)}: {step['intent']} - {step['description']}")

            try:
                if step["intent"] == "research":
                    # Route to MCP Agent for web search
                    if self.mcp_mode_enabled and self.mcp_agent:
                        logger.info(f"Routing research to MCP Agent: {step['description']}")
                        result = await self.mcp_agent.handle_request(
                            user_request=step["description"],
                            intent="research",
                            context=context,
                            user_id=user_id
                        )

                        if result.get("success"):
                            context["research_results"] = result.get("result")
                            context["research_source"] = result.get("source")

                            results.append({
                                "step": step_num,
                                "type": "research",
                                "success": True,
                                "source": result.get("source"),
                                "preview": self._truncate(str(result.get("result")), 200)
                            })
                            logger.info(f"✓ Research completed via {result.get('source')}")
                        else:
                            results.append({
                                "step": step_num,
                                "type": "research",
                                "success": False,
                                "error": result.get("error", "Research failed")
                            })
                            logger.error(f"✗ Research failed: {result.get('error')}")
                    else:
                        error_msg = "MCP mode not enabled - cannot perform research"
                        results.append({
                            "step": step_num,
                            "type": "research",
                            "success": False,
                            "error": error_msg
                        })
                        logger.warning(f"⚠️  {error_msg}")

                elif step["intent"] == "create_project":
                    # Check if this is a GTM project or if next step will generate tasks
                    is_gtm = any(word in step["description"].lower()
                                for word in ["gtm", "go-to-market", "go to market"])

                    # Also check if any subsequent step will generate tasks (needs template)
                    will_generate_tasks = any(s.get("intent") == "generate_tasks" for s in steps[i+1:])

                    # Load template from procedural memory
                    template = None
                    if is_gtm or will_generate_tasks:
                        logger.info("Loading template from procedural memory (GTM project or task generation requested)")

                        # Query for GTM template by name (more reliable than trigger matching)
                        template_doc = self.db.memory_procedural.find_one({
                            "user_id": user_id,
                            "rule_type": "template",
                            "name": "GTM Roadmap Template"
                        })

                        if template_doc:
                            context["template"] = template_doc
                            logger.info(f"✓ Found template: {template_doc.get('name')}")
                        else:
                            logger.warning("⚠️  GTM template not found in procedural memory")

                    # Extract project name from description or context
                    project_name = self._extract_project_name(step["description"], context)

                    # Create project via worklog agent
                    logger.info(f"Creating project: {project_name}")
                    project_result = self.worklog_agent._create_project(
                        name=project_name,
                        description=step["description"],
                        context=context.get("research_results", "")[:500] if context.get("research_results") else ""
                    )

                    if project_result.get("success"):
                        context["project"] = project_result.get("project")
                        context["project_id"] = project_result.get("project_id")

                        results.append({
                            "step": step_num,
                            "type": "create_project",
                            "success": True,
                            "project_name": project_name,
                            "project_id": project_result.get("project_id"),
                            "template_loaded": template is not None
                        })
                        logger.info(f"✓ Project created: {project_name}")
                    else:
                        results.append({
                            "step": step_num,
                            "type": "create_project",
                            "success": False,
                            "error": "Failed to create project"
                        })
                        logger.error("✗ Project creation failed")

                elif step["intent"] == "generate_tasks":
                    # Use template + research to generate tasks
                    template = context.get("template")
                    project = context.get("project")
                    project_id = context.get("project_id")

                    if not template:
                        error_msg = "No template available from previous steps"
                        results.append({
                            "step": step_num,
                            "type": "generate_tasks",
                            "success": False,
                            "error": error_msg
                        })
                        logger.warning(f"⚠️  {error_msg}")
                        continue

                    if not project or not project_id:
                        error_msg = "No project available from previous steps"
                        results.append({
                            "step": step_num,
                            "type": "generate_tasks",
                            "success": False,
                            "error": error_msg
                        })
                        logger.warning(f"⚠️  {error_msg}")
                        continue

                    # Generate tasks from template phases
                    logger.info(f"Generating tasks from template: {template.get('name')}")
                    tasks_created = []
                    phases_data = template.get("template", {}).get("phases", [])

                    # Build task context from template + research
                    task_context_base = f"Generated from {template.get('name')}"
                    research_results = context.get("research_results")
                    if research_results:
                        # Truncate research to keep task context concise
                        research_preview = str(research_results)[:200] if isinstance(research_results, str) else str(research_results)[:200]
                        task_context_base += f"\n\nProject context: {research_preview}..."

                    for phase in phases_data:
                        phase_name = phase.get("name", "")
                        logger.info(f"  Phase: {phase_name}")

                        for task_item in phase.get("tasks", []):
                            # Handle both formats: string (old) or dict with guiding_questions (new)
                            if isinstance(task_item, str):
                                task_title = task_item
                                task_context = task_context_base
                            else:
                                task_title = task_item.get("title", "Unknown task")
                                guiding_questions = task_item.get("guiding_questions", [])

                                # If we have research and guiding questions, tailor the context
                                if research_results and guiding_questions:
                                    questions_text = "\n".join(f"- {q}" for q in guiding_questions)

                                    # Use fast Haiku model for quick task-specific summaries
                                    try:
                                        from shared.llm import LLMService
                                        fast_llm = LLMService(model=settings.llm_model_fast)  # Fast, cheap model

                                        tailored_research = fast_llm.generate(
                                            messages=[{
                                                "role": "user",
                                                "content": f"Extract relevant insights from this research for the task '{task_title}'. Answer these questions in 2-3 clear sentences. Be direct - no preamble, no meta-commentary, just the facts:\n\n{questions_text}\n\nResearch:\n{str(research_results)[:1200]}"
                                            }],
                                            max_tokens=150,  # Reduced for faster generation
                                            temperature=0.2  # Lower temperature for concise, factual answers
                                        )
                                        # Clean up any remaining meta-commentary from Haiku
                                        clean_research = tailored_research.strip()
                                        # Remove common preambles if they snuck through
                                        for preamble in ["Based on the research, ", "Here are the insights: ", "Here's a concise response: "]:
                                            if clean_research.startswith(preamble):
                                                clean_research = clean_research[len(preamble):]

                                        task_context = f"{clean_research}"
                                        logger.debug(f"    Generated tailored research for: {task_title}")
                                    except Exception as e:
                                        logger.warning(f"    Failed to tailor research for {task_title}: {e}")
                                        task_context = task_context_base
                                else:
                                    task_context = task_context_base

                            # Add phase prefix to task title for context
                            full_title = f"[{phase_name}] {task_title}"

                            task_result = self.worklog_agent._create_task(
                                title=full_title,
                                project_id=project_id,
                                priority="medium",
                                context=task_context
                            )

                            if task_result.get("success"):
                                tasks_created.append(task_result["task"]["title"])
                                logger.debug(f"    ✓ Created: {full_title}")
                            else:
                                logger.warning(f"    ✗ Failed to create: {full_title}")

                    # Update template usage count
                    if tasks_created and template.get("_id"):
                        try:
                            from bson import ObjectId
                            self.memory.long_term.update_one(
                                {"_id": ObjectId(template["_id"])},
                                {
                                    "$inc": {"times_used": 1},
                                    "$set": {"last_used": datetime.utcnow()}
                                }
                            )
                            logger.debug(f"Incremented usage count for template: {template.get('name')}")
                        except Exception as e:
                            logger.warning(f"Failed to update template usage: {e}")

                    results.append({
                        "step": step_num,
                        "type": "generate_tasks",
                        "success": True,
                        "tasks_created": len(tasks_created),
                        "template_used": template.get("name"),
                        "phases": len(phases_data),
                        "tasks_preview": tasks_created[:5]  # Preview first 5
                    })
                    logger.info(f"✓ Generated {len(tasks_created)} tasks across {len(phases_data)} phases")

                else:
                    # Unknown step type
                    error_msg = f"Unknown step type: {step['intent']}"
                    results.append({
                        "step": step_num,
                        "type": step["intent"],
                        "success": False,
                        "error": error_msg
                    })
                    logger.warning(f"⚠️  {error_msg}")

            except Exception as e:
                logger.error(f"Error executing step {step_num}: {e}", exc_info=True)
                results.append({
                    "step": step_num,
                    "type": step.get("intent", "unknown"),
                    "success": False,
                    "error": str(e)
                })

        # Calculate overall success
        successful_steps = [r for r in results if r.get("success")]
        all_success = len(successful_steps) == len(steps)

        logger.info(f"Multi-step execution complete: {len(successful_steps)}/{len(steps)} steps successful")

        return {
            "success": all_success,
            "steps_completed": len(successful_steps),
            "total_steps": len(steps),
            "results": results,
            "context": {
                "project_created": context.get("project", {}).get("name") if context.get("project") else None,
                "project_id": context.get("project_id"),
                "research_cached": context.get("research_results") is not None,
                "research_source": context.get("research_source"),
                "template_used": context.get("template", {}).get("name") if context.get("template") else None,
                "tasks_generated": sum(r.get("tasks_created", 0) for r in results)
            }
        }

    def _extract_project_name(self, description: str, context: dict) -> str:
        """
        Extract or generate project name from description and context.

        Args:
            description: Step description
            context: Execution context

        Returns:
            Project name
        """
        import re

        # First, check for quoted project names (most reliable)
        # Handles: Create 'OpenFleet AI Framework Launch' project using...
        # Or: Create "Gaming Market" project with...
        quoted_match = re.search(r"['\"]([^'\"]+)['\"]", description)
        if quoted_match:
            return quoted_match.group(1).strip()

        # Common patterns to extract from description
        patterns = [
            "create project for ",
            "create ",
            "make project for ",
            "make ",
            "project for ",
        ]

        desc_lower = description.lower()
        for pattern in patterns:
            if pattern in desc_lower:
                # Extract text after pattern, up to common stop words
                idx = desc_lower.index(pattern) + len(pattern)
                remaining = description[idx:].strip()

                # Stop at these words to avoid including template/method info
                stop_words = [" using ", " with ", " from ", " based on "]
                for stop in stop_words:
                    if stop in remaining.lower():
                        remaining = remaining[:remaining.lower().index(stop)]
                        break

                name = remaining.strip()
                # Clean up
                name = name.replace(" project", "").replace(" vertical", "")
                if name:
                    return name

        # Fallback: Use keywords from original request
        original = context.get("original_request", "")
        keywords = []

        # Extract important words (capitalize nouns/proper nouns)
        for word in original.split():
            if len(word) > 3 and word[0].isupper():
                keywords.append(word)

        if keywords:
            return " ".join(keywords[:3]) + " Project"

        # Final fallback
        return "New GTM Project"

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def _format_multi_step_response(self, result: dict) -> dict:
        """
        Format multi-step execution results for user display.

        Args:
            result: Result dict from _execute_multi_step()

        Returns:
            Dict with "response" and "debug" keys
        """
        # Handle failures
        if not result["success"]:
            failed = [r for r in result["results"] if not r.get("success")]
            failed_step = failed[0] if failed else {"error": "Unknown"}

            response_text = (
                f"⚠️ Workflow partially completed ({result['steps_completed']}/{result['total_steps']} steps successful).\n\n"
                f"❌ Issue at step {failed_step.get('step', '?')}: {failed_step.get('error', 'Unknown error')}"
            )

            return {
                "response": response_text,
                "debug": result
            }

        # Build success message
        parts = []
        parts.append(f"✅ **Multi-step workflow completed** ({result['steps_completed']}/{result['total_steps']} steps)")
        parts.append("")  # Blank line

        # Research summary
        research = next((r for r in result["results"] if r["type"] == "research"), None)
        if research and research.get("success"):
            source = research.get("source", "web search")
            parts.append(f"**1. Research completed** (via {source})")
            if research.get("preview"):
                parts.append(f"   {research['preview'][:150]}...")
            parts.append("")

        # Project created
        project = next((r for r in result["results"] if r["type"] == "create_project"), None)
        if project and project.get("success"):
            project_name = project.get("project_name", "New Project")
            parts.append(f"**2. Project created**: {project_name}")

            if project.get("template_loaded"):
                parts.append(f"   📋 Template detected and loaded")
            parts.append("")

        # Tasks generated
        tasks = next((r for r in result["results"] if r["type"] == "generate_tasks"), None)
        if tasks and tasks.get("success"):
            task_count = tasks.get("tasks_created", 0)
            phases = tasks.get("phases", 0)
            parts.append(f"**3. Tasks generated**: {task_count} tasks across {phases} phases")

            # Show preview of tasks
            if tasks.get("tasks_preview"):
                parts.append(f"   Preview:")
                for task in tasks["tasks_preview"][:5]:
                    parts.append(f"   • {task}")
                if task_count > 5:
                    parts.append(f"   ... and {task_count - 5} more")
            parts.append("")

        # Show context summary
        context = result.get("context", {})

        if context.get("template_used"):
            parts.append(f"📚 **Template**: {context['template_used']}")

        if context.get("research_source"):
            parts.append(f"🔍 **Research source**: {context['research_source']}")

        # Add helpful next steps
        parts.append("")
        parts.append("---")
        parts.append("💡 **Next steps**:")
        parts.append(f"• View project tasks: Show me tasks for {context.get('project_created', 'the project')}")
        parts.append("• Start working: Start the first task")

        return {
            "response": "\n".join(parts),
            "debug": result
        }

    def _format_mcp_response(self, mcp_result: dict) -> dict:
        """Format MCP Agent result for the user"""
        if not mcp_result.get("success"):
            return {
                "response": f"I tried to figure this out but encountered an error: {mcp_result.get('error', 'Unknown error')}",
                "debug": mcp_result
            }

        source_messages = {
            "knowledge_cache": "📚 Found this in my knowledge cache:",
            "discovery_reuse": "🔄 I've solved this before:",
            "new_discovery": "🆕 Here's what I found:"
        }

        prefix = source_messages.get(mcp_result.get("source"), "")

        # Format result content
        # For cache hits, prefer summary if available
        result_content = mcp_result.get("result", "")

        # Check if this is from cache and has a summary
        if mcp_result.get("source") == "knowledge_cache" and mcp_result.get("summary"):
            formatted_result = mcp_result.get("summary")
            logger.debug(f"Using cached summary ({len(formatted_result)} chars)")
        else:
            # Use full result content
            if isinstance(result_content, list):
                formatted_result = "\n\n".join(str(item) for item in result_content)
            else:
                formatted_result = str(result_content)

        response_text = f"{prefix}\n\n{formatted_result}" if prefix else formatted_result

        return {
            "response": response_text,
            "debug": {
                "source": mcp_result.get("source"),
                "mcp_server": mcp_result.get("mcp_server"),
                "tool_used": mcp_result.get("tool_used"),
                "execution_time_ms": mcp_result.get("execution_time_ms"),
                "discovery_id": mcp_result.get("discovery_id")
            }
        }

    def _get_available_tools(self) -> List[Dict]:
        """Get tools available based on current settings.

        Returns:
            List of tool definitions
        """
        # Start with all base tools except get_action_history
        tools = [t for t in COORDINATOR_TOOLS if t["name"] != "get_action_history"]

        # Only include history tool if long-term memory is enabled
        if self.optimizations.get("memory_long_term") and self.memory:
            # Find the history tool definition
            history_tool = next((t for t in COORDINATOR_TOOLS if t["name"] == "get_action_history"), None)
            if history_tool:
                tools.append(history_tool)

        return tools

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a tool and return the result with debug info.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input parameters

        Returns:
            Tuple of (tool_result, debug_info)
        """
        import time

        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        start_time = time.time()
        error_msg = None
        result = None

        try:
            if tool_name == "get_tasks":
                # Get all tasks with optional filtering - DIRECT DATABASE CALL
                status = tool_input.get("status")
                project_name = tool_input.get("project_name")
                priority = tool_input.get("priority")

                # Convert project_name to project_id if provided
                project_id = None
                if project_name:
                    from shared.db import get_collection, PROJECTS_COLLECTION
                    projects_collection = get_collection(PROJECTS_COLLECTION)
                    project_doc = projects_collection.find_one(
                        {"name": {"$regex": f"^{project_name}$", "$options": "i"}}
                    )
                    if project_doc:
                        project_id = str(project_doc["_id"])

                # Call worklog agent's direct method (bypasses LLM)
                result = self.worklog_agent._list_tasks(
                    project_id=project_id,
                    status=status,
                    priority=priority,
                    limit=50
                )

            elif tool_name == "search_tasks":
                # Hybrid search for tasks with optional filters
                query = tool_input["query"]
                limit = tool_input.get("limit", 10)
                status = tool_input.get("status")
                priority = tool_input.get("priority")
                assignee = tool_input.get("assignee")
                project_name = tool_input.get("project_name")

                # Convert project_name to project_id if provided
                project_id = None
                if project_name:
                    from shared.db import get_collection, PROJECTS_COLLECTION
                    projects_collection = get_collection(PROJECTS_COLLECTION)
                    project_doc = projects_collection.find_one(
                        {"name": {"$regex": f"^{project_name}$", "$options": "i"}}
                    )
                    if project_doc:
                        project_id = str(project_doc["_id"])

                tasks = self.retrieval_agent.hybrid_search_tasks(
                    query,
                    limit,
                    status=status,
                    priority=priority,
                    project_id=project_id,
                    assignee=assignee
                )

                # Convert ObjectId to string for JSON serialization
                for task in tasks:
                    if "_id" in task:
                        task["_id"] = str(task["_id"])
                    if "project_id" in task and task["project_id"]:
                        task["project_id"] = str(task["project_id"])

                result = {
                    "success": True,
                    "tasks": tasks,
                    "count": len(tasks)
                }

            elif tool_name == "complete_task":
                # Mark task as done - DIRECT DATABASE CALL
                task_id = tool_input["task_id"]
                result = self.worklog_agent._complete_task(task_id)

            elif tool_name == "start_task":
                # Mark task as in_progress - DIRECT DATABASE CALL
                task_id = tool_input["task_id"]
                result = self.worklog_agent._update_task(task_id, status="in_progress")

            elif tool_name == "add_note_to_task":
                # Add note to task - DIRECT DATABASE CALL
                task_id = tool_input["task_id"]
                note = tool_input["note"]
                result = self.worklog_agent._add_note("task", task_id, note)

            elif tool_name == "get_projects":
                # Get all projects - DIRECT DATABASE CALL
                result = self.worklog_agent._list_projects(limit=50)

            elif tool_name == "search_projects":
                # Hybrid search for projects
                query = tool_input["query"]
                limit = tool_input.get("limit", 5)

                projects = self.retrieval_agent.hybrid_search_projects(query, limit)

                # Convert ObjectId to string for JSON serialization
                for project in projects:
                    if "_id" in project:
                        project["_id"] = str(project["_id"])

                result = {
                    "success": True,
                    "projects": projects,
                    "count": len(projects)
                }

            elif tool_name == "get_tasks_by_time":
                # Temporal query for tasks based on activity timestamps
                from datetime import timedelta

                timeframe = tool_input["timeframe"]
                activity_type = tool_input.get("activity_type")
                status = tool_input.get("status")

                # Convert timeframe to dates
                now = datetime.now()
                since = None
                until = None

                if timeframe == "today":
                    since = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif timeframe == "yesterday":
                    since = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    until = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif timeframe == "this_week":
                    # Start of week (Monday)
                    since = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                elif timeframe == "last_week":
                    # Start of last week
                    since = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
                    # End of last week
                    until = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                elif timeframe == "this_month":
                    since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

                # Query tasks by activity
                tasks = self.retrieval_agent.get_tasks_by_activity(
                    since=since,
                    until=until,
                    activity_type=activity_type,
                    status=status
                )

                # Convert ObjectId to string for JSON serialization
                for task in tasks:
                    if "_id" in task:
                        task["_id"] = str(task["_id"])
                    if "project_id" in task and task["project_id"]:
                        task["project_id"] = str(task["project_id"])

                result = {
                    "success": True,
                    "tasks": tasks,
                    "count": len(tasks),
                    "timeframe": timeframe,
                    "activity_type": activity_type
                }

            elif tool_name == "create_task":
                # Create a new task - look up project by name first
                title = tool_input["title"]
                project_name = tool_input["project_name"]
                priority = tool_input.get("priority", "medium")
                context = tool_input.get("context", "")
                assignee = tool_input.get("assignee")
                due_date = tool_input.get("due_date")
                blockers = tool_input.get("blockers")

                # Look up project by name
                projects = self.retrieval_agent.hybrid_search_projects(project_name, limit=1)

                if not projects:
                    result = {
                        "success": False,
                        "error": f"Project not found: '{project_name}'. Use search_projects or get_projects to see available projects."
                    }
                else:
                    project_id = projects[0]["_id"]

                    # Create task via worklog agent
                    result = self.worklog_agent._create_task(
                        title=title,
                        project_id=str(project_id),
                        priority=priority,
                        context=context,
                        assignee=assignee,
                        due_date=due_date,
                        blockers=blockers
                    )

            elif tool_name == "update_task":
                # Update task fields
                task_id = tool_input["task_id"]
                title = tool_input.get("title")
                priority = tool_input.get("priority")
                context = tool_input.get("context")
                status = tool_input.get("status")
                project_name = tool_input.get("project_name")
                assignee = tool_input.get("assignee")
                due_date = tool_input.get("due_date")
                blockers = tool_input.get("blockers")

                # Look up project if moving to different project
                project_id = None
                if project_name:
                    projects = self.retrieval_agent.hybrid_search_projects(project_name, limit=1)
                    if projects:
                        project_id = str(projects[0]["_id"])
                    else:
                        result = {
                            "success": False,
                            "error": f"Project not found: '{project_name}'"
                        }

                if project_name and not project_id:
                    pass  # Error already set above
                else:
                    result = self.worklog_agent._update_task(
                        task_id=task_id,
                        title=title,
                        priority=priority,
                        context=context,
                        status=status,
                        project_id=project_id,
                        assignee=assignee,
                        due_date=due_date,
                        blockers=blockers
                    )

            elif tool_name == "stop_task":
                # Mark task as todo (stopped)
                task_id = tool_input["task_id"]
                result = self.worklog_agent._update_task(task_id, status="todo")

            elif tool_name == "create_project":
                # Create a new project
                name = tool_input["name"]
                description = tool_input.get("description", "")
                context = tool_input.get("context", "")
                result = self.worklog_agent._create_project(
                    name=name,
                    description=description,
                    context=context
                )

            elif tool_name == "update_project":
                # Update project fields
                project_id = tool_input["project_id"]
                name = tool_input.get("name")
                description = tool_input.get("description")
                context = tool_input.get("context")
                status = tool_input.get("status")
                result = self.worklog_agent._update_project(
                    project_id=project_id,
                    name=name,
                    description=description,
                    context=context,
                    status=status
                )

            elif tool_name == "add_note_to_project":
                # Add note to project
                project_id = tool_input["project_id"]
                note = tool_input["note"]
                result = self.worklog_agent._add_note("project", project_id, note)

            elif tool_name == "add_context_to_task":
                # Add context to task
                task_id = tool_input["task_id"]
                context = tool_input["context"]
                result = self.worklog_agent._add_context("task", task_id, context)

            elif tool_name == "add_context_to_project":
                # Add context to project
                project_id = tool_input["project_id"]
                context = tool_input["context"]
                result = self.worklog_agent._add_context("project", project_id, context)

            elif tool_name == "add_decision_to_project":
                # Add decision to project
                project_id = tool_input["project_id"]
                decision = tool_input["decision"]
                result = self.worklog_agent._add_decision(project_id, decision)

            elif tool_name == "add_method_to_project":
                # Add method/technology to project
                project_id = tool_input["project_id"]
                method = tool_input["method"]
                result = self.worklog_agent._add_method(project_id, method)

            elif tool_name == "get_task":
                # Get single task by ID
                task_id = tool_input["task_id"]
                result = self.worklog_agent._get_task(task_id)

            elif tool_name == "get_project":
                # Get single project by ID
                project_id = tool_input["project_id"]
                result = self.worklog_agent._get_project(project_id)

            elif tool_name == "get_project_by_name":
                # Get project by exact name match
                from shared.db import get_project_by_name
                project_name = tool_input["project_name"]
                project = get_project_by_name(project_name)

                if project:
                    result = {
                        "success": True,
                        "project": project.model_dump(by_alias=True)
                    }
                else:
                    result = {
                        "success": False,
                        "error": f"Project '{project_name}' not found. Use get_projects or search_projects to see available projects."
                    }

            elif tool_name == "get_action_history":
                # Query long-term memory for action history using new API
                if not self.memory:
                    result = {
                        "success": False,
                        "error": "Long-term memory not enabled"
                    }
                else:
                    semantic_query = tool_input.get("semantic_query")
                    time_range = tool_input.get("time_range", "all")
                    action_type = tool_input.get("action_type", "all")
                    limit = tool_input.get("limit", 10)
                    summarize = tool_input.get("summarize", False)

                    # SEMANTIC SEARCH MODE (when semantic_query provided)
                    if semantic_query:
                        # Use vector search
                        search_results = self.memory.search_history(
                            user_id=self.user_id,
                            semantic_query=semantic_query,
                            time_range=time_range if time_range != "all" else None,
                            action_type=action_type if action_type != "all" else None,
                            limit=limit
                        )

                        # Check if error returned (dict with "error" key)
                        if isinstance(search_results, dict) and "error" in search_results:
                            result = search_results
                        else:
                            # Format semantic search results
                            formatted_actions = []
                            for action in search_results:
                                formatted_actions.append({
                                    "action": action["action_type"],
                                    "task": action.get("entity", {}).get("task_title"),
                                    "project": action.get("entity", {}).get("project_name"),
                                    "timestamp": action["timestamp"].strftime("%Y-%m-%d %H:%M") if action.get("timestamp") else None,
                                    "agent": action.get("source_agent", "coordinator"),
                                    "note": action.get("metadata", {}).get("note", "")[:100] if action.get("metadata", {}).get("note") else None,
                                    "similarity_score": round(action.get("score", 0), 3),
                                    "matched_text": action.get("embedding_text", "")[:200]
                                })

                            result = {
                                "success": True,
                                "type": "semantic_search",
                                "query": semantic_query,
                                "count": len(formatted_actions),
                                "actions": formatted_actions
                            }

                    # FILTER MODE (traditional time-based filtering)
                    elif summarize:
                        # Return activity summary
                        summary = self.memory.get_activity_summary(
                            user_id=self.user_id,
                            time_range=time_range if time_range != "all" else "this_week"
                        )

                        # Generate narrative from raw stats
                        narrative = self.memory.generate_narrative(summary)

                        result = {
                            "success": True,
                            "type": "summary",
                            "time_range": summary["time_range"],
                            "total": summary["total"],
                            "narrative": narrative,  # Pre-formatted narrative
                            "by_type": summary["by_type"],
                            "by_project": summary["by_project"],
                            "by_agent": summary["by_agent"],
                            "timeline": summary["timeline"]
                        }
                    else:
                        # Return detailed history
                        history = self.memory.get_action_history(
                            user_id=self.user_id,
                            time_range=time_range if time_range != "all" else None,
                            action_type=action_type if action_type != "all" else None,
                            limit=limit
                        )

                        # Format for LLM
                        formatted_actions = []
                        for action in history:
                            formatted_actions.append({
                                "action": action["action_type"],
                                "task": action.get("entity", {}).get("task_title"),
                                "project": action.get("entity", {}).get("project_name"),
                                "timestamp": action["timestamp"].strftime("%Y-%m-%d %H:%M") if action.get("timestamp") else None,
                                "agent": action.get("source_agent", "coordinator"),
                                "note": action.get("metadata", {}).get("note", "")[:100] if action.get("metadata", {}).get("note") else None
                            })

                        result = {
                            "success": True,
                            "type": "history",
                            "time_range": time_range,
                            "count": len(formatted_actions),
                            "actions": formatted_actions
                        }

            elif tool_name == "resolve_disambiguation":
                # Resolve pending disambiguation by selecting an option
                if not self.memory or not self.session_id:
                    result = {
                        "success": False,
                        "error": "Memory not available or session not set"
                    }
                else:
                    selection = tool_input["selection"]
                    index = selection - 1  # Convert 1-based to 0-based

                    selected = self.memory.resolve_disambiguation(self.session_id, index)

                    if selected:
                        result = {
                            "success": True,
                            "selected_index": selection,
                            "task_id": selected.get("task_id"),
                            "title": selected.get("title"),
                            "project": selected.get("project"),
                            "message": f"Selected option {selection}: {selected.get('title')}"
                        }
                    else:
                        # Check if there's a pending disambiguation
                        pending = self.memory.get_pending_disambiguation(self.session_id)
                        if not pending:
                            result = {
                                "success": False,
                                "error": "No pending disambiguation to resolve"
                            }
                        else:
                            result = {
                                "success": False,
                                "error": f"Invalid selection {selection}. Valid options: 1-{len(pending.get('results', []))}"
                            }

            elif tool_name == "list_templates":
                # List all available project templates
                if not self.memory or not self.user_id:
                    result = {
                        "success": False,
                        "error": "Memory not available or user not set"
                    }
                else:
                    # Get all templates from procedural memory
                    templates = list(self.db.memory_procedural.find({
                        "user_id": self.user_id,
                        "rule_type": "template"
                    }))

                    # Format templates for display
                    formatted_templates = []
                    for tmpl in templates:
                        phases = tmpl.get("phases", [])
                        total_tasks = sum(len(phase.get("tasks", [])) for phase in phases)

                        formatted_templates.append({
                            "name": tmpl.get("name", "Unnamed Template"),
                            "description": tmpl.get("description", ""),
                            "phases": len(phases),
                            "total_tasks": total_tasks,
                            "phase_names": [p.get("name", "") for p in phases],
                            "trigger": tmpl.get("trigger", ""),
                            "times_used": tmpl.get("times_used", 0)
                        })

                    result = {
                        "success": True,
                        "templates": formatted_templates,
                        "count": len(formatted_templates)
                    }

                    logger.info(f"📋 Listed {len(formatted_templates)} templates for user {self.user_id}")

            elif tool_name == "search_knowledge":
                # Search cached knowledge from semantic memory
                if not self.memory or not self.user_id:
                    result = {
                        "success": False,
                        "error": "Long-term memory not enabled"
                    }
                else:
                    query = tool_input.get("query", "")
                    limit = tool_input.get("limit", 5)

                    # Search semantic memory for cached knowledge
                    knowledge_results = self.memory.search_knowledge(
                        user_id=self.user_id,
                        query=query,
                        limit=limit
                    )

                    # Format results for display
                    formatted_results = []
                    for item in knowledge_results:
                        # Handle both formats: knowledge base (key/value) and Tavily cache (query/result)
                        topic = item.get("key") or item.get("query", "")
                        content = item.get("value") or item.get("result") or item.get("summary", "")
                        cached_at = item.get("created_at") or item.get("fetched_at", "")

                        formatted_results.append({
                            "topic": topic,
                            "content": content,
                            "tags": item.get("tags", []),
                            "source": item.get("source", "unknown"),
                            "cached_at": cached_at.isoformat() if hasattr(cached_at, 'isoformat') else str(cached_at),
                            "confidence": item.get("confidence", 0),
                            "score": item.get("score", 0),
                            "times_accessed": item.get("times_accessed", 0)
                        })

                    result = {
                        "success": True,
                        "knowledge": formatted_results,
                        "count": len(formatted_results),
                        "message": f"Found {len(formatted_results)} cached knowledge entries" if formatted_results else "No cached knowledge found for this query"
                    }

                    logger.info(f"🔍 Searched knowledge cache: '{query}' → {len(formatted_results)} results")

            elif tool_name == "analyze_tool_discoveries":
                # Analyze tool discovery patterns
                if not self.memory or not self.user_id:
                    result = {
                        "success": False,
                        "error": "Long-term memory not enabled"
                    }
                else:
                    days = tool_input.get("days", 7)

                    # Import analysis function
                    from memory.discovery_analysis import analyze_discoveries

                    # Run analysis
                    analysis = analyze_discoveries(
                        db=self.db,
                        user_id=self.user_id,
                        days=days
                    )

                    result = {
                        "success": True,
                        "analysis": analysis,
                        "period_days": days
                    }

                    logger.info(f"📊 Analyzed tool discoveries for {days} days")

            elif tool_name == "remember_about_user":
                if not self.memory or not self.user_id:
                    result = {"success": False, "error": "Long-term memory not enabled"}
                else:
                    key = str(tool_input.get("key", "")).strip().lower()
                    value = tool_input.get("value", "")
                    source = tool_input.get("source", "explicit")
                    if not key or value in (None, ""):
                        result = {"success": False, "error": "key and value are required"}
                    else:
                        previous = self.memory.get_profile_fact(self.user_id, key)
                        doc_id = self.memory.record_profile_fact(
                            user_id=self.user_id,
                            key=key,
                            value=value,
                            source=source,
                        )
                        result = {
                            "success": True,
                            "key": key,
                            "value": value,
                            "previous_value": previous.get("value") if previous else None,
                            "is_new": previous is None,
                            "id": doc_id,
                        }
                        logger.info(f"🧠 Profile fact stored: {key}={value!r} (new={previous is None})")

            elif tool_name == "forget_about_user":
                if not self.memory or not self.user_id:
                    result = {"success": False, "error": "Long-term memory not enabled"}
                else:
                    key = str(tool_input.get("key", "")).strip().lower()
                    if not key:
                        result = {"success": False, "error": "key is required"}
                    else:
                        deleted = self.memory.delete_profile_fact(self.user_id, key)
                        result = {"success": True, "key": key, "deleted": deleted}
                        logger.info(f"🧠 Profile fact forget {key!r}: deleted={deleted}")

            else:
                result = {"success": False, "error": f"Unknown tool: {tool_name}"}
                error_msg = f"Unknown tool: {tool_name}"

            # Record significant actions to long-term memory using new API
            if self.memory_config.get("long_term") and result and result.get("success"):
                # Record task completions
                if tool_name == "complete_task" and result.get("task"):
                    self._record_action(
                        action_type="complete",
                        entity_type="task",
                        entity={
                            "task_id": str(result["task"].get("_id")),
                            "task_title": result["task"].get("title"),
                            "project_id": str(result.get("project_id")) if result.get("project_id") else None,
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "previous_status": result.get("previous_status", "in_progress"),
                            "new_status": "done"
                        }
                    )

                # Record task starts
                elif tool_name == "start_task" and result.get("task"):
                    self._record_action(
                        action_type="start",
                        entity_type="task",
                        entity={
                            "task_id": str(result["task"].get("_id")),
                            "task_title": result["task"].get("title"),
                            "project_id": str(result.get("project_id")) if result.get("project_id") else None,
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "previous_status": result.get("previous_status", "todo"),
                            "new_status": "in_progress"
                        }
                    )

                # Record task stops
                elif tool_name == "stop_task" and result.get("task"):
                    self._record_action(
                        action_type="stop",
                        entity_type="task",
                        entity={
                            "task_id": str(result["task"].get("_id")),
                            "task_title": result["task"].get("title"),
                            "project_id": str(result.get("project_id")) if result.get("project_id") else None,
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "previous_status": result.get("previous_status", "in_progress"),
                            "new_status": "todo"
                        }
                    )

                # Record task creations
                elif tool_name == "create_task" and result.get("task"):
                    self._record_action(
                        action_type="create",
                        entity_type="task",
                        entity={
                            "task_id": str(result["task"].get("_id")),
                            "task_title": result["task"].get("title"),
                            "project_id": str(result.get("project_id")) if result.get("project_id") else None,
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "priority": result["task"].get("priority"),
                            "context": result["task"].get("context", "")[:200]
                        }
                    )

                # Record task updates
                elif tool_name == "update_task" and result.get("task"):
                    self._record_action(
                        action_type="update",
                        entity_type="task",
                        entity={
                            "task_id": str(result["task"].get("_id")),
                            "task_title": result["task"].get("title"),
                            "project_id": str(result.get("project_id")) if result.get("project_id") else None,
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "updates": tool_input
                        }
                    )

                # Record notes added
                elif tool_name == "add_note_to_task":
                    self._record_action(
                        action_type="add_note",
                        entity_type="task",
                        entity={
                            "task_id": tool_input.get("task_id"),
                            "task_title": result.get("task_title"),
                            "project_name": result.get("project_name")
                        },
                        metadata={
                            "note": tool_input.get("note", "")[:200]
                        }
                    )

                # Record project creations
                elif tool_name == "create_project" and result.get("project"):
                    self._record_action(
                        action_type="create",
                        entity_type="project",
                        entity={
                            "project_id": str(result["project"].get("_id")),
                            "project_name": result["project"].get("name")
                        },
                        metadata={
                            "description": result["project"].get("description", "")[:200]
                        }
                    )

                # Record searches
                elif tool_name in ["search_tasks", "search_projects"]:
                    self._record_action(
                        action_type="search",
                        entity_type="search",
                        entity={
                            "query": tool_input.get("query")
                        },
                        metadata={
                            "search_type": "tasks" if tool_name == "search_tasks" else "projects",
                            "results_count": len(result.get("tasks", result.get("projects", [])))
                        }
                    )

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            error_msg = str(e)
            result = {"success": False, "error": str(e)}

        finally:
            # Always calculate duration, even on error
            duration_ms = int((time.time() - start_time) * 1000)

            # Use new summarize method for consistent output summary
            output_summary = self._summarize_output(result)

            # Capture latency breakdown from agents
            breakdown = {}

            # Check if retrieval agent has timings
            if hasattr(self.retrieval_agent, 'last_query_timings') and self.retrieval_agent.last_query_timings:
                breakdown.update(self.retrieval_agent.last_query_timings)
                self.retrieval_agent.last_query_timings = {}  # Reset

            # Check if worklog agent has timings
            if hasattr(self.worklog_agent, 'last_query_timings') and self.worklog_agent.last_query_timings:
                breakdown.update(self.worklog_agent.last_query_timings)
                self.worklog_agent.last_query_timings = {}  # Reset

            # Calculate processing overhead (Python, serialization, etc.)
            if breakdown:
                known_time = sum(breakdown.values())
                processing_overhead = max(0, duration_ms - known_time)
                if processing_overhead > 0:
                    breakdown["processing"] = processing_overhead

            # Build debug info (don't include full result to avoid serialization issues)
            debug_info = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "duration_ms": duration_ms,
                "output_summary": output_summary,
                "success": result.get("success", False) if result else False,
                "error": error_msg
            }

            # Also append to current turn if we're tracking it
            if self.current_turn is not None:
                self.current_turn["tool_calls"].append({
                    "name": tool_name,
                    "input": tool_input,
                    "output": output_summary,
                    "duration_ms": duration_ms,
                    "breakdown": breakdown if breakdown else None,  # Include latency breakdown
                    "success": result.get("success", False) if result else False,
                    "error": error_msg
                })

            return result, debug_info

    # Tools whose output is mechanically formattable; skipping the second
    # ("format_response") LLM call removes ~2s from the most common turns.
    _SHORT_CIRCUIT_TOOLS = {"get_tasks", "get_projects"}

    def _project_name_map(self, project_ids: List[str]) -> Dict[str, str]:
        """Return {project_id_str: name} for the given ids in a single query."""
        if not project_ids:
            return {}
        try:
            from shared.db import get_collection, PROJECTS_COLLECTION
            unique_ids = []
            seen = set()
            for pid in project_ids:
                if pid and pid not in seen:
                    seen.add(pid)
                    unique_ids.append(ObjectId(pid))
            if not unique_ids:
                return {}
            projects = get_collection(PROJECTS_COLLECTION).find(
                {"_id": {"$in": unique_ids}}, {"name": 1}
            )
            return {str(p["_id"]): p.get("name", "-") for p in projects}
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Project name lookup failed: {exc}")
            return {}

    def _format_tasks_short_circuit(self, tasks: List[Dict[str, Any]]) -> str:
        """Render tasks grouped by status using the FORMAT block from the streamlined prompt."""
        if not tasks:
            return "No tasks found."

        name_map = self._project_name_map([t.get("project_id") for t in tasks])

        groups = {"in_progress": [], "todo": [], "done": []}
        for t in tasks:
            status = t.get("status") or "todo"
            if status in groups:
                groups[status].append(t)

        headings = [
            ("in_progress", "## ◐ In Progress"),
            ("todo", "## ○ To Do"),
            ("done", "## ✓ Done"),
        ]
        sections = []
        for key, heading in headings:
            items = groups[key]
            if not items:
                continue
            lines = [f"{heading} ({len(items)})"]
            for i, t in enumerate(items, 1):
                title = t.get("title", "Untitled")
                priority = t.get("priority") or "-"
                project = name_map.get(t.get("project_id") or "", "-")
                lines.append(f"{i}. **{title}** ({priority}) - {project}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "No tasks found."

    def _format_projects_short_circuit(self, projects: List[Dict[str, Any]]) -> str:
        """Render projects as a compact markdown list."""
        if not projects:
            return "No projects found."
        lines = [f"## Projects ({len(projects)})"]
        for i, p in enumerate(projects, 1):
            name = p.get("name", "Untitled")
            status = p.get("status") or "-"
            desc = (p.get("description") or "").strip()
            row = f"{i}. **{name}** ({status})"
            if desc:
                row += f" — {desc[:80]}"
            lines.append(row)
        return "\n".join(lines)

    def process(self, user_message: str, conversation_history: Optional[List[Dict[str, Any]]] = None, input_type: str = "text", turn_number: int = 1, optimizations: Optional[Dict[str, bool]] = None, return_debug: bool = False, session_id: Optional[str] = None, on_text_delta: Optional[Any] = None, on_thinking_delta: Optional[Any] = None, on_tool_start: Optional[Any] = None, on_tool_end: Optional[Any] = None) -> Union[str, Dict[str, Any]]:
        """
        Process a user message using Claude's native tool use.

        Voice and text inputs are processed identically - Claude decides which tools to call.

        Args:
            user_message: User's message (voice transcript or text)
            conversation_history: Optional conversation history
            input_type: Type of input ("text" or "voice") - for logging only
            turn_number: The turn number for this request (for debug tracking)
            optimizations: Optional dict of optimization toggles (compress_results, streamlined_prompt, prompt_caching)
            return_debug: If True, return dict with response and debug info instead of just response string
            session_id: Optional session ID for memory isolation

        Returns:
            If return_debug=False: Agent's response string (default, backwards compatible)
            If return_debug=True: Dict with {"response": str, "debug": {...}}
        """
        # Store optimizations for use throughout the process
        self.optimizations = optimizations or {}

        # Initialize memory operations tracking
        self.memory_ops = {
            "context_injected": False,
            "context_updated": False,
            "action_recorded": False,
            "preference_recorded": False,
            "rule_recorded": False,
            "handoffs_created": 0,
            "handoffs_consumed": 0,
            "recorded_action_type": None,
            "memory_read_ms": 0,
            "memory_write_ms": 0
        }

        # Set session if provided
        if session_id:
            self.session_id = session_id

        # Start new chain for this request
        self.current_chain_id = str(uuid.uuid4())

        # Set session on agents if session_id provided and shared memory enabled
        if session_id and self.optimizations.get("memory_shared"):
            retrieval_agent.set_session(session_id)
            worklog_agent.set_session(session_id)

        # Get system prompt based on streamlined toggle
        streamlined = self.optimizations.get("streamlined_prompt", True)
        system_prompt = get_system_prompt(streamlined)
        prompt_stats = get_prompt_stats(streamlined)

        # BUILD CONTEXT-ENHANCED PROMPT
        import time
        read_start = time.time()

        if self.memory_config.get("context_injection") and self.memory:
            context_injection = self._build_context_injection()
            if context_injection:
                system_prompt += context_injection
                logger.info(f"📊 Context injected into system prompt")

        self.memory_ops["memory_read_ms"] = (time.time() - read_start) * 1000

        # CHECK FOR RULE TRIGGERS (TTL-R)
        rule_match = None
        if self.memory_config.get("context_injection") and self.memory:
            rule_match = self._check_rule_triggers(user_message)
            if rule_match:
                # Inject rule execution directive into system prompt
                rule_directive = f"""

<rule_triggered>
User's message matched rule: "{rule_match['trigger']}" → {rule_match['action']}
Execute the rule action: {rule_match['action']}
</rule_triggered>
"""
                system_prompt += rule_directive
                self.memory_ops["rule_triggered"] = rule_match['trigger']
                logger.info(f"🔔 Rule triggered: '{rule_match['trigger']}' → {rule_match['action']}")

        # ═══════════════════════════════════════════════════════════════
        # WORKFLOW MATCHING: Check for multi-step procedural patterns
        # ═══════════════════════════════════════════════════════════════

        workflow_match = None
        if self.memory_config.get("long_term") and self.memory:
            # Try semantic workflow search (regex + vector similarity)
            workflow_match = self.memory.search_workflows_semantic(
                user_id=self.user_id,
                user_message=user_message,
                min_score=0.7  # Minimum similarity score for semantic matches
            )

            if workflow_match:
                match_type = workflow_match.get("match_type", "unknown")
                match_score = workflow_match.get("match_score", 0)
                workflow_name = workflow_match.get("name", "Unknown Workflow")
                logger.info(f"🔄 Workflow matched ({match_type}, score={match_score:.2f}): '{workflow_name}'")

                # Execute workflow programmatically using WorkflowExecutor
                tool_registry = self._build_tool_registry()
                executor = WorkflowExecutor(tool_registry=tool_registry)

                try:
                    workflow_result = executor.execute_workflow(
                        workflow=workflow_match,
                        user_message=user_message,
                        context={}
                    )

                    self.memory_ops["workflow_executed"] = workflow_name
                    self.memory_ops["workflow_success"] = workflow_result.get("success", False)

                    # Format response based on workflow result
                    if workflow_result.get("success"):
                        steps_completed = workflow_result.get("steps_completed", 0)
                        total_steps = workflow_result.get("total_steps", 0)
                        response_text = f"✅ **{workflow_name}** completed successfully ({steps_completed}/{total_steps} steps)"

                        # Add details from workflow results
                        results = workflow_result.get("results", [])
                        if results:
                            response_text += "\n\n**Steps executed:**"
                            for result in results:
                                action = result.get("action", "unknown")
                                response_text += f"\n• {action}"

                        return {
                            "response": response_text,
                            "confidence": "high",
                            "workflow_executed": True,
                            "workflow_result": workflow_result
                        }
                    else:
                        error_msg = workflow_result.get("error", "Unknown error")
                        response_text = f"⚠️ **{workflow_name}** failed: {error_msg}"
                        return {
                            "response": response_text,
                            "confidence": "high",
                            "workflow_executed": True,
                            "workflow_result": workflow_result
                        }

                except Exception as e:
                    logger.exception(f"Workflow execution failed: {e}")
                    # Fall through to normal LLM processing
                    workflow_match = None

        # ═══════════════════════════════════════════════════════════════
        # MULTI-STEP ROUTING: Check if request contains multiple steps
        # ═══════════════════════════════════════════════════════════════

        multi_step = self._classify_multi_step_intent(user_message)

        if multi_step["is_multi_step"]:
            logger.info(f"🔄 Detected multi-step request with {len(multi_step['steps'])} steps")

            # Execute multi-step workflow
            import asyncio
            workflow_start = time.time()
            result = asyncio.run(self._execute_multi_step(
                steps=multi_step["steps"],
                original_request=user_message,
                user_id=self.user_id
            ))
            workflow_time = int((time.time() - workflow_start) * 1000)

            # Add workflow steps to debug panel
            if self.current_turn:
                for step_result in result.get("results", []):
                    self.current_turn["tool_calls"].append({
                        "name": f"workflow_step_{step_result.get('step', '?')}_{step_result.get('type', 'unknown')}",
                        "input": {"description": step_result.get("description", ""), "step": step_result.get("step")},
                        "output": f"Success: {step_result.get('success')}" + (f" - {step_result.get('error')}" if not step_result.get('success') else ""),
                        "duration_ms": workflow_time // len(result.get("results", [1])),  # Approximate per-step time
                        "success": step_result.get("success", False)
                    })

                # Add workflow summary
                self.current_turn["total_duration_ms"] = workflow_time
                self.current_turn["tools_called"].extend([f"step_{r.get('type')}" for r in result.get("results", [])])

            # Format response for user
            formatted = self._format_multi_step_response(result)

            if return_debug:
                return {
                    "response": formatted.get("response"),
                    "debug": {
                        **formatted.get("debug", {}),
                        "workflow": True,
                        "steps": len(multi_step["steps"]),
                        "workflow_time_ms": workflow_time,
                        "tool_calls": self.current_turn.get("tool_calls", []) if self.current_turn else []
                    }
                }
            else:
                return formatted["response"]

        # ═══════════════════════════════════════════════════════════════
        # TIER 4: MCP ROUTING - External Tool Discovery (Requires Toggle)
        # ═══════════════════════════════════════════════════════════════
        # This section handles requests that require EXTERNAL tools via MCP:
        # - Web search and research (via Tavily)
        # - Advanced MongoDB query generation (via MongoDB MCP server)
        # - Other external tool discovery
        #
        # Requests that don't match Tier 1 (patterns), Tier 2 (slash commands),
        # or Tier 4 (MCP external tools) will fall through to Tier 3 (LLM with
        # built-in agents) below.
        # ═══════════════════════════════════════════════════════════════

        intent = self._classify_intent(user_message)
        logger.info(f"📊 Classified intent: {intent}")

        # Initialize current turn for tracking (needed for both MCP and normal paths)
        self.current_turn = {
            "turn": turn_number,
            "user_input": user_message[:100] + "..." if len(user_message) > 100 else user_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "tool_calls": [],
            "llm_calls": [],
            "total_duration_ms": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cache_hit": False,
            "tools_called": []
        }

        # Check if this request needs EXTERNAL MCP tools (Tier 4)
        if not self._can_static_tools_handle(intent, user_message):
            logger.info(f"📊 Intent '{intent}' requires external MCP tools, checking if MCP mode enabled...")

            # MCP mode check - only required for EXTERNAL tool discovery (Tier 4)
            if not self.mcp_mode_enabled:
                logger.warning(f"📊 MCP mode disabled, falling back to local search for intent: {intent}")

                # Fallback: Search local data instead of failing
                if intent in ["web_search", "research", "find_information"]:
                    logger.info("🔍 Searching local tasks, projects, and knowledge as fallback")

                    # Use LLM with built-in tools to search local data
                    # This will search tasks, projects, and cached knowledge
                    # Intent changes to "unknown" so it uses Tier 3 (LLM + built-in tools)
                    intent = "search_tasks"  # Fallback to local search
                    logger.info(f"📊 Fallback: Changed intent from web_search to search_tasks")

                    # Let it proceed to LLM with built-in tools below
                    # The LLM will call search_tasks, search_projects, and search_knowledge
                    # Skip MCP routing and go to normal LLM flow
                    pass
                elif intent == "advanced_mongodb_query":
                    response_text = "I don't have access to the MongoDB MCP server for advanced query generation. Enable Experimental MCP Mode to use MongoDB's query generation capabilities for complex aggregations and analytics."

                    if return_debug:
                        return {
                            "response": response_text,
                            "debug": {
                                "could_use_mcp": True,
                                "intent": intent,
                                "mcp_available": settings.mcp_available
                            }
                        }
                    else:
                        return response_text
                else:
                    # Other MCP intents without fallback - return error
                    response_text = "I don't have access to external research tools for this request. Enable Experimental MCP Mode to let me search the web and discover new tools."
                    if return_debug:
                        return {
                            "response": response_text,
                            "debug": {
                                "could_use_mcp": True,
                                "intent": intent,
                                "mcp_available": settings.mcp_available
                            }
                        }
                    else:
                        return response_text

        # Only route to MCP if mode is enabled AND intent requires it
        if not self._can_static_tools_handle(intent, user_message) and self.mcp_mode_enabled:
            # Route to MCP Agent
            logger.info(f"🔬 Routing to MCP Agent (intent: {intent})")

            try:
                # Ensure MCP agent is initialized
                if self.mcp_agent is None:
                    init_result = self._run_async(self.enable_mcp_mode())
                    if not init_result.get("success"):
                        error_response = f"Failed to initialize MCP mode: {init_result.get('error')}"
                        if return_debug:
                            return {"response": error_response, "debug": {"error": init_result.get('error')}}
                        else:
                            return error_response

                # Get current context for MCP agent
                context = {}
                if self.session_id and self.memory:
                    session_context = self.memory.read_session_context(self.session_id)
                    if session_context:
                        context = session_context

                # Call MCP agent using persistent event loop
                mcp_result = self._run_async(self.mcp_agent.handle_request(
                    user_request=user_message,
                    intent=intent,
                    context=context,
                    user_id=self.user_id
                ))

                # Format MCP result for user
                formatted = self._format_mcp_response(mcp_result)

                logger.info(f"🔬 MCP Agent result: success={mcp_result.get('success')}, source={mcp_result.get('source')}")

                # Track MCP tool call in current_turn for debug panel
                if self.current_turn:
                    source = mcp_result.get("source", "unknown")

                    # Track cache check operation if it occurred
                    cache_check_info = mcp_result.get("cache_check_info")
                    if cache_check_info and cache_check_info.get("checked"):
                        cache_hit = cache_check_info.get("hit", False)
                        cache_score = cache_check_info.get("score", 0)
                        cache_threshold = cache_check_info.get("threshold", 0.65)
                        cache_time = cache_check_info.get("time_ms", 0)

                        # Add cache check as a separate tool call
                        self.current_turn["tool_calls"].append({
                            "name": "search_knowledge",
                            "input": {"query": user_message[:80], "threshold": cache_threshold},
                            "output": f"{'HIT' if cache_hit else 'MISS'} (score: {cache_score:.2f})",
                            "duration_ms": cache_time,
                            "success": cache_hit,
                            "source": "cache_check"
                        })

                    # Use special display for different sources
                    if source == "knowledge_cache":
                        mcp_tool_name = "cache/knowledge"
                        cache_score = mcp_result.get("cache_score", 0)
                        input_display = {"query": user_message[:80], "score": f"{cache_score:.2f}"}
                    elif source == "discovery_reuse":
                        mcp_server = mcp_result.get('mcp_server', 'unknown')
                        tool_used = mcp_result.get('tool_used', 'unknown')
                        mcp_tool_name = f"{mcp_server}/{tool_used}"
                        input_display = mcp_result.get("arguments", {})
                    else:
                        mcp_server = mcp_result.get('mcp_server', 'unknown')
                        tool_used = mcp_result.get('tool_used', 'unknown')
                        mcp_tool_name = f"{mcp_server}/{tool_used}"
                        input_display = mcp_result.get("arguments", {})

                    self.current_turn["tool_calls"].append({
                        "name": mcp_tool_name,
                        "input": input_display,
                        "output": str(mcp_result.get("result", ""))[:200] + "..." if len(str(mcp_result.get("result", ""))) > 200 else str(mcp_result.get("result", "")),
                        "duration_ms": mcp_result.get("execution_time_ms", 0),
                        "success": mcp_result.get("success", False),
                        "source": source
                    })
                    self.current_turn["tools_called"].append(mcp_tool_name)
                    self.current_turn["total_duration_ms"] = mcp_result.get("execution_time_ms", 0)

                if return_debug:
                    return {
                        "response": formatted["response"],
                        "debug": {
                            **formatted.get("debug", {}),
                            "routed_to_mcp": True,
                            "intent": intent
                        }
                    }
                else:
                    return formatted["response"]

            except Exception as e:
                logger.error(f"🔬 MCP Agent error: {e}", exc_info=True)
                error_response = f"MCP Agent encountered an error: {str(e)}"

                if return_debug:
                    return {
                        "response": error_response,
                        "debug": {
                            "mcp_error": str(e),
                            "intent": intent
                        }
                    }
                else:
                    return error_response

        # ═══════════════════════════════════════════════════════════════
        # TIER 3: LLM AGENT WITH BUILT-IN TOOLS (Always Available)
        # ═══════════════════════════════════════════════════════════════
        # This section handles ALL requests using Claude with built-in tools:
        # - worklog_agent (tasks, projects, notes)
        # - retrieval_agent (search, memory)
        # - memory system (context, rules, preferences)
        #
        # This tier is ALWAYS available and does NOT require MCP toggle.
        # ═══════════════════════════════════════════════════════════════

        # Get prompt caching setting
        cache_prompts = self.optimizations.get("prompt_caching", True)

        logger.info("=" * 80)
        logger.info("=== NEW REQUEST ===")
        logger.info(f"Input type: {input_type}")
        logger.info(f"User message: {user_message[:200]}...")
        logger.info(f"History length: {len(conversation_history) if conversation_history else 0}")
        logger.info(f"📊 Prompt: {prompt_stats['type']} ({prompt_stats['word_count']} words, ~{int(prompt_stats['estimated_tokens'])} tokens)")
        logger.info(f"📊 Caching: {'enabled' if cache_prompts else 'disabled'}")

        # Reset old debug info (kept for backwards compatibility)
        self.last_debug_info = []

        # Build messages - SAME PATH FOR VOICE AND TEXT
        # CRITICAL: Serialize conversation history to ensure datetime/ObjectId objects are converted to strings
        # Otherwise the Anthropic API will fail with "Object of type datetime is not JSON serializable"
        # Truncate history to last MAX_HISTORY_MESSAGES turns to keep prompt-cache stable and tokens low
        MAX_HISTORY_MESSAGES = 20
        history_src = conversation_history or []
        if len(history_src) > MAX_HISTORY_MESSAGES:
            history_src = history_src[-MAX_HISTORY_MESSAGES:]
        # Drop any history entries whose content is empty/whitespace — the
        # Anthropic API rejects messages with empty string content
        # ("messages.N: user/assistant messages must have non-empty content").
        # An empty turn can sneak in when a previous request errored out or
        # when a session is resumed from steps that lacked input/output.
        def _has_content(msg: Dict[str, Any]) -> bool:
            c = msg.get("content")
            if isinstance(c, str):
                return bool(c.strip())
            if isinstance(c, list):
                return len(c) > 0
            return c is not None
        history_src = [m for m in history_src if _has_content(m)]
        messages = convert_objectids_to_str(list(history_src))
        if not (user_message and user_message.strip()):
            raise ValueError("user_message is empty; refusing to call the LLM with an empty turn")
        messages.append({"role": "user", "content": user_message})

        # Get available tools based on settings
        available_tools = self._get_available_tools()

        # Use Claude's native tool use - TIME THIS CALL
        import time
        logger.info("=" * 80)
        logger.info("=== INITIAL LLM CALL (DECIDE ACTION) ===")
        logger.info(f"📊 Turn {turn_number}")
        logger.info(f"📊 Sending {len(available_tools)} tools to LLM")
        logger.info(f"📊 Tool names: {', '.join([t['name'] for t in available_tools])}")
        logger.info(f"📊 Messages count: {len(messages)}")
        logger.info(f"📊 Last message roles: {[m.get('role') for m in messages[-5:]]}")
        logger.info(f"📊 Last message preview: {str(messages[-1].get('content', ''))[:150]}...")
        logger.info(f"📊 System prompt length: {len(system_prompt)} chars")
        logger.info("=" * 80)

        # Detect if this is an action request that requires tool use
        user_message = messages[-1].get('content', '') if messages else ''
        action_keywords = ['create', 'update', 'complete', 'start', 'add', 'mark', 'assign', 'change', 'delete', 'remove', 'show', 'list', 'what', 'get', 'tell', 'know']
        requires_tool = any(keyword in user_message.lower() for keyword in action_keywords)

        # Self-referential profile questions ("what's my name", "who am I",
        # "do you know my email") should be answered from the injected
        # <user_profile> context, not by forcing an unrelated tool call.
        msg_lower = user_message.lower()
        self_reference_patterns = [
            "my name", "my email", "my timezone", "my location", "my role",
            "my company", "my pronouns", "about me", "who am i", "what am i",
            "remember me", "remember about me",
        ]
        is_self_query = any(p in msg_lower for p in self_reference_patterns)

        # Extended thinking exposes the model's reasoning to the UI. It's
        # incompatible with `tool_choice` forcing, so when it's enabled we
        # let the model pick freely (the prompt already biases towards tools
        # for action requests).
        thinking_enabled = bool(self.optimizations.get("extended_thinking", False))
        thinking_kwarg: Dict[str, Any] = {}
        if thinking_enabled:
            thinking_kwarg["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(self.optimizations.get("thinking_budget_tokens", 1024)),
            }

        llm_kwargs: Dict[str, Any] = {}
        if requires_tool and not is_self_query and not thinking_enabled:
            llm_kwargs['tool_choice'] = {"type": "any"}  # Force tool use
            logger.info("🔧 Forcing tool use for action/query request")
        elif thinking_enabled and requires_tool and not is_self_query:
            logger.info("💭 Extended thinking on — skipping tool_choice force")
        elif is_self_query:
            logger.info("👤 Self-referential profile query — letting model answer from context")

        llm_start = time.time()
        # When thinking is enabled, switch decide_action to streaming so the
        # caller can render thinking deltas live. Text deltas (rare on this
        # call — typically tool_use) and thinking deltas are forwarded via
        # the registered callbacks; the final assembled message is used as
        # the response just like the non-streaming path.
        if thinking_enabled and (on_thinking_delta is not None or on_text_delta is not None):
            response = None
            decide_max_tokens = max(
                1024 + int(self.optimizations.get("thinking_budget_tokens", 1024)),
                2048,
            )
            for kind, payload in self.llm.generate_with_tools_stream(
                messages=messages,
                tools=available_tools,
                system=system_prompt,
                max_tokens=decide_max_tokens,
                temperature=0.3,
                cache_prompts=cache_prompts,
                **llm_kwargs,
                **thinking_kwarg,
            ):
                if kind == "thinking" and on_thinking_delta is not None:
                    try:
                        on_thinking_delta(payload)
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug(f"on_thinking_delta callback failed: {cb_exc}")
                elif kind == "text" and on_text_delta is not None:
                    try:
                        on_text_delta(payload)
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug(f"on_text_delta callback failed: {cb_exc}")
                elif kind == "done":
                    response = payload
        else:
            response = self.llm.generate_with_tools(
                messages=messages,
                tools=available_tools,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.3,
                cache_prompts=cache_prompts,
                **llm_kwargs,
                **thinking_kwarg,
            )
        llm_duration = int((time.time() - llm_start) * 1000)

        # Capture token usage
        if hasattr(response, 'usage'):
            self.current_turn["tokens_in"] += getattr(response.usage, 'input_tokens', 0)
            self.current_turn["tokens_out"] += getattr(response.usage, 'output_tokens', 0)
            cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
            if cache_read > 0:
                self.current_turn["cache_hit"] = True

        # DEBUG: Check response
        logger.info("=" * 80)
        logger.info("=== INITIAL LLM RESPONSE ===")
        logger.info(f"📊 Stop reason: {response.stop_reason}")
        logger.info(f"📊 Content blocks: {len(response.content)}")
        logger.info(f"📊 Content types: {[b.type for b in response.content]}")

        tool_use_blocks = [b for b in response.content if hasattr(b, 'type') and b.type == 'tool_use']
        logger.info(f"📊 Tool use blocks: {len(tool_use_blocks)}")

        if tool_use_blocks:
            logger.info(f"📊 ✅ Tools called: {[b.name for b in tool_use_blocks]}")
        else:
            # Log why no tool was called - check the text response
            text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
            if text_blocks:
                logger.warning(f"📊 ❌ NO TOOLS CALLED - LLM responded with text only")
                logger.warning(f"📊 Text response: {text_blocks[0][:300]}...")
            logger.warning(f"📊 ⚠️  CRITICAL: LLM should call tools but didn't!")
        logger.info("=" * 80)

        # Track the initial LLM call
        self.current_turn["llm_calls"].append({
            "purpose": "decide_action",
            "duration_ms": llm_duration
        })
        logger.info(f"LLM decide_action took {llm_duration}ms")

        # Short-circuit: when the model picked exactly one read-only tool whose
        # output we can format mechanically, skip the second LLM call entirely.
        # This typically removes ~2s from the most common turns (list/show).
        skip_format = self.optimizations.get("skip_format_for_lists", True)
        if (
            skip_format
            and response.stop_reason == "tool_use"
            and len(tool_use_blocks) == 1
            and tool_use_blocks[0].name in self._SHORT_CIRCUIT_TOOLS
        ):
            sc_block = tool_use_blocks[0]
            logger.info(f"⚡ Short-circuiting format_response for {sc_block.name}")

            # Fire tool_start so UI can render the step before execution
            if on_tool_start is not None:
                try:
                    on_tool_start({"id": sc_block.id, "name": sc_block.name, "input": sc_block.input})
                except Exception as cb_exc:  # noqa: BLE001
                    logger.debug(f"on_tool_start callback failed: {cb_exc}")
            _sc_t0 = time.time()
            sc_result, sc_debug = self._execute_tool(sc_block.name, sc_block.input)
            if on_tool_end is not None:
                try:
                    on_tool_end({
                        "id": sc_block.id,
                        "name": sc_block.name,
                        "result": sc_result,
                        "duration_ms": int((time.time() - _sc_t0) * 1000),
                    })
                except Exception as cb_exc:  # noqa: BLE001
                    logger.debug(f"on_tool_end callback failed: {cb_exc}")
            sc_debug["iteration"] = 1
            sc_debug["index"] = len(self.last_debug_info) + 1
            self.last_debug_info.append(sc_debug)

            if sc_block.name == "get_tasks":
                final_text = self._format_tasks_short_circuit(sc_result.get("tasks", []))
            else:  # get_projects
                final_text = self._format_projects_short_circuit(sc_result.get("projects", []))

            # Mirror the timing/aggregation block that the normal path runs
            if self.current_turn is not None:
                tool_time = sum(tc["duration_ms"] for tc in self.current_turn["tool_calls"])
                llm_time = sum(lc["duration_ms"] for lc in self.current_turn["llm_calls"])
                self.current_turn["llm_time_ms"] = llm_time
                self.current_turn["total_duration_ms"] = tool_time + llm_time
                logger.info(
                    f"Turn complete (short-circuit): LLM={llm_time}ms, "
                    f"Tools={tool_time}ms, Total={self.current_turn['total_duration_ms']}ms"
                )

            if return_debug:
                embedding_time = mongodb_time = processing_time = 0
                for tc in self.current_turn.get("tool_calls", []):
                    bd = tc.get("breakdown") or {}
                    embedding_time += bd.get("embedding_generation", 0)
                    mongodb_time += bd.get("mongodb_query", 0)
                    processing_time += bd.get("processing", 0)
                return {
                    "response": final_text.strip(),
                    "debug": {
                        "tokens_in": self.current_turn.get("tokens_in", 0),
                        "tokens_out": self.current_turn.get("tokens_out", 0),
                        "llm_time_ms": self.current_turn.get("llm_time_ms", 0),
                        "tool_time_ms": sum(tc["duration_ms"] for tc in self.current_turn.get("tool_calls", [])),
                        "cache_hit": self.current_turn.get("cache_hit", False),
                        "tools_called": [tc["name"] for tc in self.current_turn.get("tool_calls", [])],
                        "embedding_time_ms": embedding_time,
                        "mongodb_time_ms": mongodb_time,
                        "processing_time_ms": processing_time,
                        "memory_ops": self.memory_ops,
                        "short_circuited": True,
                    },
                }
            return final_text.strip()

        # Handle tool calls in a loop until Claude is done
        max_iterations = 10
        iteration = 0

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool use iteration {iteration}")

            # Collect tool_use blocks for this iteration in stable order
            tu_blocks = [b for b in response.content if b.type == "tool_use"]

            # Wrap _execute_tool with start/end callbacks so the UI can
            # render each tool step the moment it begins and update it as
            # soon as the result arrives. Callbacks may run from worker
            # threads when tools execute in parallel — the chainlit-side
            # bridge uses a thread-safe queue to absorb that.
            def _run_tool(b):
                if on_tool_start is not None:
                    try:
                        on_tool_start({"id": b.id, "name": b.name, "input": b.input})
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug(f"on_tool_start callback failed: {cb_exc}")
                _t0 = time.time()
                _res, _dbg = self._execute_tool(b.name, b.input)
                if on_tool_end is not None:
                    try:
                        on_tool_end({
                            "id": b.id,
                            "name": b.name,
                            "result": _res,
                            "duration_ms": int((time.time() - _t0) * 1000),
                        })
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug(f"on_tool_end callback failed: {cb_exc}")
                return _res, _dbg

            # Execute tools concurrently when there are multiple in the same
            # iteration. Most tools are I/O-bound (Mongo + embeddings) so a
            # ThreadPoolExecutor gives near-linear speedup. Single-tool
            # iterations skip the executor to avoid thread overhead.
            if len(tu_blocks) > 1:
                from concurrent.futures import ThreadPoolExecutor
                logger.info(f"⚡ Executing {len(tu_blocks)} tools in parallel")
                with ThreadPoolExecutor(max_workers=min(len(tu_blocks), 8)) as ex:
                    exec_results = list(ex.map(_run_tool, tu_blocks))
            else:
                exec_results = [_run_tool(b) for b in tu_blocks]

            tool_results = []
            for content_block, (result, debug_info) in zip(tu_blocks, exec_results):
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id

                logger.info(f"Tool call: {tool_name}")
                logger.debug(f"Tool input: {tool_input}")

                # Add iteration number to debug info
                debug_info["iteration"] = iteration
                debug_info["index"] = len(self.last_debug_info) + 1

                # Store debug info
                self.last_debug_info.append(debug_info)

                # Apply compression if enabled (before serialization)
                compress = self.optimizations.get("compress_results", True)
                compressed_result = compress_tool_result(tool_name, result, compress=compress)

                # Add tool result (convert ObjectIds to strings first)
                serializable_result = convert_objectids_to_str(compressed_result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(serializable_result)
                })

            # Add assistant message with tool use
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Add tool results
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get next response from Claude - TIME THIS CALL
            logger.info("=" * 80)
            logger.info(f"=== FOLLOW-UP LLM CALL (ITERATION {iteration}) ===")
            logger.info(f"📊 Sending tool results back to LLM")
            logger.info(f"📊 Sending {len(available_tools)} tools to LLM")
            logger.info(f"📊 Messages count: {len(messages)}")
            logger.info(f"📊 Last message roles: {[m.get('role') for m in messages[-5:]]}")
            logger.info(f"📊 Tool results count: {len(tool_results)}")
            logger.info("=" * 80)

            llm_start = time.time()
            # When the caller registers a text-delta callback we stream the
            # follow-up response token-by-token so the UI can render text as
            # it arrives. Decide_action above stays non-streaming because its
            # output is tool_use blocks, not user-visible text.
            if on_text_delta is not None:
                logger.info("🌊 Streaming follow-up LLM response")
                response = None
                followup_max_tokens = 4096
                if thinking_enabled:
                    followup_max_tokens = max(
                        4096 + int(self.optimizations.get("thinking_budget_tokens", 1024)),
                        4096,
                    )
                for kind, payload in self.llm.generate_with_tools_stream(
                    messages=messages,
                    tools=available_tools,
                    system=system_prompt,
                    max_tokens=followup_max_tokens,
                    temperature=0.3,
                    cache_prompts=cache_prompts,
                    **thinking_kwarg,
                ):
                    if kind == "text":
                        try:
                            on_text_delta(payload)
                        except Exception as cb_exc:  # noqa: BLE001
                            logger.debug(f"on_text_delta callback failed: {cb_exc}")
                    elif kind == "thinking" and on_thinking_delta is not None:
                        try:
                            on_thinking_delta(payload)
                        except Exception as cb_exc:  # noqa: BLE001
                            logger.debug(f"on_thinking_delta callback failed: {cb_exc}")
                    elif kind == "done":
                        response = payload
            else:
                response = self.llm.generate_with_tools(
                    messages=messages,
                    tools=available_tools,
                    system=system_prompt,
                    max_tokens=4096,
                    temperature=0.3,
                    cache_prompts=cache_prompts,
                    **thinking_kwarg,
                )
            llm_duration = int((time.time() - llm_start) * 1000)

            # Capture token usage
            if hasattr(response, 'usage'):
                self.current_turn["tokens_in"] += getattr(response.usage, 'input_tokens', 0)
                self.current_turn["tokens_out"] += getattr(response.usage, 'output_tokens', 0)
                cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
                if cache_read > 0:
                    self.current_turn["cache_hit"] = True

            logger.info("=" * 80)
            logger.info(f"=== FOLLOW-UP LLM RESPONSE (ITERATION {iteration}) ===")
            logger.info(f"📊 Stop reason: {response.stop_reason}")
            logger.info(f"📊 Content blocks: {len(response.content)}")
            logger.info(f"📊 Content types: {[b.type for b in response.content]}")
            logger.info("=" * 80)

            # Track this LLM call
            self.current_turn["llm_calls"].append({
                "purpose": "format_response",
                "duration_ms": llm_duration
            })
            logger.info(f"LLM format_response took {llm_duration}ms")

        # Extract final text response
        final_text = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                final_text += content_block.text

        # Calculate total duration for this turn (LLM + tools)
        if self.current_turn is not None:
            # Sum tool time
            tool_time = sum(tc["duration_ms"] for tc in self.current_turn["tool_calls"])

            # Sum LLM time
            llm_time = sum(lc["duration_ms"] for lc in self.current_turn["llm_calls"])

            # Total = LLM + tools
            self.current_turn["llm_time_ms"] = llm_time
            self.current_turn["total_duration_ms"] = tool_time + llm_time

            logger.info(f"Turn complete: LLM={llm_time}ms, Tools={tool_time}ms, Total={self.current_turn['total_duration_ms']}ms")

        # UPDATE SESSION CONTEXT FROM TURN
        write_start = time.time()

        if self.memory_config.get("short_term") and self.session_id and self.memory:
            # Extract context updates from this turn
            context_updates = self._extract_context_from_turn(
                user_message,
                self.current_turn.get("tool_calls", []),
                [tc.get("result") for tc in self.current_turn.get("tool_calls", [])]
            )

            if context_updates:
                self.memory.update_session_context(
                    self.session_id,
                    context_updates,
                    self.user_id
                )
                self.memory_ops["context_updated"] = True
                logger.info(f"📊 Session context updated with {len(context_updates)} fields")

        self.memory_ops["memory_write_ms"] = (time.time() - write_start) * 1000

        logger.info("Request processing complete")
        logger.info("=" * 80)

        # Return debug info if requested (for evals), otherwise just the response text
        if return_debug:
            # Aggregate timing breakdown across all tool calls
            embedding_time = 0
            mongodb_time = 0
            processing_time = 0

            for tool_call in self.current_turn.get("tool_calls", []):
                breakdown = tool_call.get("breakdown")
                if breakdown:
                    embedding_time += breakdown.get("embedding_generation", 0)
                    mongodb_time += breakdown.get("mongodb_query", 0)
                    processing_time += breakdown.get("processing", 0)

            return {
                "response": final_text.strip(),
                "debug": {
                    "tokens_in": self.current_turn.get("tokens_in", 0),
                    "tokens_out": self.current_turn.get("tokens_out", 0),
                    "llm_time_ms": self.current_turn.get("llm_time_ms", 0),
                    "tool_time_ms": sum(tc["duration_ms"] for tc in self.current_turn.get("tool_calls", [])),
                    "cache_hit": self.current_turn.get("cache_hit", False),
                    "tools_called": [tc["name"] for tc in self.current_turn.get("tool_calls", [])],
                    "embedding_time_ms": embedding_time,
                    "mongodb_time_ms": mongodb_time,
                    "processing_time_ms": processing_time,
                    "memory_ops": self.memory_ops
                }
            }
        else:
            return final_text.strip()

    def process_stream(self, user_message: str, conversation_history: Optional[List[Dict[str, Any]]] = None, input_type: str = "text", turn_number: int = 1, optimizations: Optional[Dict[str, bool]] = None, session_id: Optional[str] = None):
        """
        Process a user message and stream the response token-by-token.

        Internally this delegates to ``process(..., on_text_delta=...,
        on_thinking_delta=...)``, bridging the (sync) callbacks into a
        generator via a thread-safe queue so the caller can iterate
        deltas as they arrive from the model.

        Yields:
            Tuples of (kind, payload, debug_info) where:
              - kind == "text":       payload is a text delta (str)
              - kind == "thinking":   payload is an extended-thinking delta (str)
              - kind == "tool_start": payload is {"id", "name", "input"} for a
                                      tool that is about to execute
              - kind == "tool_end":   payload is {"id", "name", "result",
                                      "duration_ms"} once the tool returns
              - kind == "done":       payload is "" and debug_info holds the
                                      final aggregated debug dict
            ``debug_info`` is only populated on the final ``done`` chunk.
        """
        import queue
        import threading

        q: "queue.Queue[Any]" = queue.Queue()
        SENTINEL = object()

        def _on_text(text: str) -> None:
            q.put(("text", text))

        def _on_thinking(text: str) -> None:
            q.put(("thinking", text))

        def _on_tool_start(payload: Dict[str, Any]) -> None:
            q.put(("tool_start", payload))

        def _on_tool_end(payload: Dict[str, Any]) -> None:
            q.put(("tool_end", payload))

        result_holder: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                result_holder["result"] = self.process(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    input_type=input_type,
                    turn_number=turn_number,
                    optimizations=optimizations,
                    return_debug=True,
                    session_id=session_id,
                    on_text_delta=_on_text,
                    on_thinking_delta=_on_thinking,
                    on_tool_start=_on_tool_start,
                    on_tool_end=_on_tool_end,
                )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc
            finally:
                q.put(SENTINEL)

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()

        streamed_chunks: List[str] = []
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            kind, payload = item
            if kind == "text":
                streamed_chunks.append(payload)
            yield (kind, payload, None)

        worker.join()

        if "error" in result_holder:
            raise result_holder["error"]

        result = result_holder.get("result", "")
        if isinstance(result, dict):
            response_text = result.get("response", "")
            debug_info = result.get("debug", {})
        else:
            response_text = result
            debug_info = {}

        # If the streaming path didn't fire (e.g. short-circuit, or model
        # produced no text deltas) emit the full response now so the caller
        # still receives the text. Otherwise emit only the trailing
        # difference (usually empty after stripping).
        streamed = "".join(streamed_chunks)
        remainder = response_text[len(streamed):] if response_text.startswith(streamed) else response_text
        if remainder or not streamed:
            yield ("text", remainder, None)
        yield ("done", "", debug_info)


# Global coordinator instance with memory manager
try:
    from memory import MemoryManager
    from shared.db import MongoDB
    from shared.embeddings import embed_query

    # Create memory manager with embedding function
    mongodb = MongoDB()
    db = mongodb.get_database()

    # Use embed_query function for embeddings
    memory_manager = MemoryManager(
        db=db,
        embedding_fn=embed_query
    )

    # Share memory manager with all agents
    retrieval_agent.memory = memory_manager
    worklog_agent.memory = memory_manager

    coordinator = CoordinatorAgent(memory_manager=memory_manager, db=db)
    logger.info("✅ Coordinator initialized with memory manager and database")
    logger.info("✅ Retrieval and Worklog agents have shared memory access")
    logger.info(f"✅ MCP mode available: {settings.mcp_available}")
except Exception as e:
    logger.warning(f"⚠️  Failed to initialize memory manager: {e}")
    logger.warning("Coordinator running without memory support")
    coordinator = CoordinatorAgent(memory_manager=None, db=None)