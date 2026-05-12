"""Worklog Agent for task and project management operations."""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from bson import ObjectId

from shared.llm import llm_service
from shared.logger import get_logger
from shared.embeddings import embed_document
from shared.db import (
    create_task as db_create_task,
    update_task as db_update_task,
    add_task_note,
    get_task as db_get_task,
    create_project as db_create_project,
    update_project as db_update_project,
    add_project_note,
    add_project_method,
    add_project_decision,
    get_project as db_get_project,
    get_collection,
    TASKS_COLLECTION,
    PROJECTS_COLLECTION,
)
from shared.models import Task, Project

logger = get_logger("worklog")


class WorklogAgent:
    """Agent for handling task and project management operations using Claude."""

    def __init__(self, memory_manager=None):
        self.llm = llm_service
        self.tools = self._define_tools()
        self.last_query_timings = {}  # Track latency breakdown for debug panel
        self.memory = memory_manager  # Shared memory for agent handoffs
        self.session_id = None  # Current session ID for handoffs

    def set_session(self, session_id: str):
        """Set the current session ID for shared memory operations.

        Args:
            session_id: Session identifier
        """
        self.session_id = session_id

    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define all available tools for the agent."""
        return [
            {
                "name": "create_task",
                "description": "Create a new task. IMPORTANT: Use dedicated parameters for assignee, due_date, and blockers - do NOT put these in the context field.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the task"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project ID (MongoDB ObjectId as string)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Rich context about the task"
                        },
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional array of notes"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Task priority"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done"],
                            "description": "Task status (defaults to 'todo')"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Person or team responsible for this task. REQUIRED when user says 'assign to X', 'for X', etc. Use full name (e.g. 'Sarah Thompson' not 'Sarah'). DO NOT put in context field."
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of blockers preventing progress on this task. DO NOT put in context field."
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date when user says 'due X'. Pass natural language as-is (e.g., 'in 5 days', 'tomorrow', 'next Friday'). DO NOT put in context field."
                        }
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "update_task",
                "description": "Update an existing task's fields including assignee, blockers, and due date",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (MongoDB ObjectId as string)"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done"],
                            "description": "New status"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "New priority"
                        },
                        "context": {
                            "type": "string",
                            "description": "New or additional context"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "New project ID"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "New assignee. Use this parameter when changing who is responsible. DO NOT put in context field."
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Updated list of blockers (replaces existing blockers). DO NOT put in context field."
                        },
                        "due_date": {
                            "type": "string",
                            "description": "New due date in natural language (e.g., 'in 5 days', 'tomorrow') or ISO format. DO NOT put in context field."
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "complete_task",
                "description": "Mark a task as completed",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (MongoDB ObjectId as string)"
                        },
                        "completion_note": {
                            "type": "string",
                            "description": "Optional note about task completion"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "create_project",
                "description": "Create a new project with optional description, context, and stakeholders",
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
                            "description": "Rich context about the project"
                        },
                        "stakeholders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stakeholders (people or teams involved in the project)"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "update_project",
                "description": "Update an existing project's fields including stakeholders",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        },
                        "name": {
                            "type": "string",
                            "description": "New name"
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
                        },
                        "stakeholders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Updated list of stakeholders (replaces existing)"
                        }
                    },
                    "required": ["project_id"]
                }
            },
            {
                "name": "add_note",
                "description": "Add a note to a task or project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_type": {
                            "type": "string",
                            "enum": ["task", "project"],
                            "description": "Type of target (task or project)"
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target ID (MongoDB ObjectId as string)"
                        },
                        "note": {
                            "type": "string",
                            "description": "Note content to add"
                        }
                    },
                    "required": ["target_type", "target_id", "note"]
                }
            },
            {
                "name": "add_context",
                "description": "Add or update context for a task or project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_type": {
                            "type": "string",
                            "enum": ["task", "project"],
                            "description": "Type of target (task or project)"
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target ID (MongoDB ObjectId as string)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context to add or update"
                        }
                    },
                    "required": ["target_type", "target_id", "context"]
                }
            },
            {
                "name": "add_decision",
                "description": "Add a decision to a project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        },
                        "decision": {
                            "type": "string",
                            "description": "Decision to record"
                        }
                    },
                    "required": ["project_id", "decision"]
                }
            },
            {
                "name": "add_method",
                "description": "Add a technology/method/approach to a project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
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
                "name": "add_blocker",
                "description": "Add a blocker to a task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (MongoDB ObjectId as string)"
                        },
                        "blocker": {
                            "type": "string",
                            "description": "Description of what is blocking progress"
                        }
                    },
                    "required": ["task_id", "blocker"]
                }
            },
            {
                "name": "remove_blocker",
                "description": "Remove a blocker from a task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (MongoDB ObjectId as string)"
                        },
                        "blocker": {
                            "type": "string",
                            "description": "Blocker text to remove (must match exactly)"
                        }
                    },
                    "required": ["task_id", "blocker"]
                }
            },
            {
                "name": "add_stakeholder",
                "description": "Add a stakeholder to a project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        },
                        "stakeholder": {
                            "type": "string",
                            "description": "Name of stakeholder (person or team)"
                        }
                    },
                    "required": ["project_id", "stakeholder"]
                }
            },
            {
                "name": "add_project_update",
                "description": "Add a status update to a project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        },
                        "update_content": {
                            "type": "string",
                            "description": "Status update content describing recent progress or changes"
                        }
                    },
                    "required": ["project_id", "update_content"]
                }
            },
            {
                "name": "list_tasks",
                "description": "List tasks with optional filters",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Filter by project ID"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done"],
                            "description": "Filter by status"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return (default: 20)",
                            "default": 20
                        }
                    }
                }
            },
            {
                "name": "list_projects",
                "description": "List projects with optional filters",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "archived"],
                            "description": "Filter by status"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of projects to return (default: 20)",
                            "default": 20
                        }
                    }
                }
            },
            {
                "name": "get_task",
                "description": "Get a single task by ID",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID (MongoDB ObjectId as string)"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "get_project",
                "description": "Get a single project by ID",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        }
                    },
                    "required": ["project_id"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Dictionary with the tool execution result
        """
        try:
            if tool_name == "create_task":
                return self._create_task(**tool_input)
            elif tool_name == "update_task":
                return self._update_task(**tool_input)
            elif tool_name == "complete_task":
                return self._complete_task(**tool_input)
            elif tool_name == "create_project":
                return self._create_project(**tool_input)
            elif tool_name == "update_project":
                return self._update_project(**tool_input)
            elif tool_name == "add_note":
                return self._add_note(**tool_input)
            elif tool_name == "add_context":
                return self._add_context(**tool_input)
            elif tool_name == "add_decision":
                return self._add_decision(**tool_input)
            elif tool_name == "add_method":
                return self._add_method(**tool_input)
            elif tool_name == "add_blocker":
                return self._add_blocker(**tool_input)
            elif tool_name == "remove_blocker":
                return self._remove_blocker(**tool_input)
            elif tool_name == "add_stakeholder":
                return self._add_stakeholder(**tool_input)
            elif tool_name == "add_project_update":
                return self._add_project_update(**tool_input)
            elif tool_name == "list_tasks":
                return self._list_tasks(**tool_input)
            elif tool_name == "list_projects":
                return self._list_projects(**tool_input)
            elif tool_name == "get_task":
                return self._get_task(**tool_input)
            elif tool_name == "get_project":
                return self._get_project(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    def _create_task(
        self,
        title: str,
        project_id: Optional[str] = None,
        context: str = "",
        notes: Optional[List[str]] = None,
        priority: Optional[Literal["low", "medium", "high"]] = None,
        status: Literal["todo", "in_progress", "done"] = "todo",
        assignee: Optional[str] = None,
        blockers: Optional[List[str]] = None,
        due_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new task."""
        from shared.embeddings import build_task_embedding_text

        # Parse due_date if provided
        parsed_due_date = None
        if due_date:
            parsed_due_date = self._parse_due_date(due_date)

        # Create task model
        task = Task(
            title=title,
            project_id=ObjectId(project_id) if project_id else None,
            context=context,
            notes=notes or [],
            priority=priority,
            status=status,
            assignee=assignee,
            blockers=blockers or [],
            due_date=parsed_due_date
        )

        # Generate embedding from comprehensive task data
        task_doc = task.model_dump(by_alias=True)
        embedding_text = build_task_embedding_text(task_doc)
        task.embedding = embed_document(embedding_text)

        # Create in database
        task_id = db_create_task(task, action_note=f"Task created: {title}")

        # Return created task
        created_task = db_get_task(task_id)
        return {
            "success": True,
            "task_id": str(task_id),
            "task": self._task_to_dict(created_task)
        }

    def _update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        status: Optional[Literal["todo", "in_progress", "done"]] = None,
        priority: Optional[Literal["low", "medium", "high"]] = None,
        context: Optional[str] = None,
        project_id: Optional[str] = None,
        assignee: Optional[str] = None,
        blockers: Optional[List[str]] = None,
        due_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing task."""
        import time

        # Track timings for debug panel
        timings = {}

        task_oid = ObjectId(task_id)
        current_task = db_get_task(task_oid)

        if not current_task:
            return {"success": False, "error": "Task not found"}

        updates = {}
        changes = []

        if title is not None:
            updates["title"] = title
            changes.append(f"title changed to '{title}'")

        if status is not None:
            updates["status"] = status
            changes.append(f"status changed to '{status}'")

        if priority is not None:
            updates["priority"] = priority
            changes.append(f"priority changed to '{priority}'")

        if context is not None:
            updates["context"] = context
            changes.append("context updated")

        if project_id is not None:
            updates["project_id"] = ObjectId(project_id)
            changes.append(f"moved to project {project_id}")

        if assignee is not None:
            updates["assignee"] = assignee
            changes.append(f"assignee changed to '{assignee}'")

        if blockers is not None:
            updates["blockers"] = blockers
            changes.append(f"blockers updated ({len(blockers)} blocker(s))")

        if due_date is not None:
            parsed_due_date = self._parse_due_date(due_date)
            updates["due_date"] = parsed_due_date
            changes.append(f"due date set to {parsed_due_date.strftime('%Y-%m-%d') if parsed_due_date else 'none'}")

        # Re-generate embedding if any content fields changed
        if any(field is not None for field in [title, context, assignee, blockers, due_date]):
            from shared.embeddings import build_task_embedding_text
            start = time.time()

            # Build updated task document for embedding
            updated_task_doc = current_task.model_dump(by_alias=True)
            updated_task_doc.update(updates)

            embedding_text = build_task_embedding_text(updated_task_doc)
            updates["embedding"] = embed_document(embedding_text)
            timings["embedding_generation"] = int((time.time() - start) * 1000)

        # Time MongoDB update operation
        action_note = "; ".join(changes) if changes else "Task updated"
        start = time.time()
        success = db_update_task(task_oid, updates, "updated", action_note)
        updated_task = db_get_task(task_oid)
        timings["mongodb_query"] = int((time.time() - start) * 1000)

        # Store timings for coordinator to access
        self.last_query_timings = timings

        return {
            "success": success,
            "task": self._task_to_dict(updated_task)
        }

    def _complete_task(
        self,
        task_id: str,
        completion_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark a task as completed."""
        import time

        # Track timings for debug panel
        timings = {}

        task_oid = ObjectId(task_id)

        updates = {
            "status": "done",
            "completed_at": datetime.utcnow()
        }

        action_note = completion_note if completion_note else "Task completed"

        # Time MongoDB update operation
        start = time.time()
        success = db_update_task(task_oid, updates, "completed", action_note)
        updated_task = db_get_task(task_oid)
        timings["mongodb_query"] = int((time.time() - start) * 1000)

        # Store timings for coordinator to access
        self.last_query_timings = timings

        return {
            "success": success,
            "task": self._task_to_dict(updated_task)
        }

    def _create_project(
        self,
        name: str,
        description: str = "",
        context: str = "",
        stakeholders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new project."""
        from shared.embeddings import build_project_embedding_text

        # Create project model
        project = Project(
            name=name,
            description=description,
            context=context,
            stakeholders=stakeholders or []
        )

        # Generate embedding from comprehensive project data
        project_doc = project.model_dump(by_alias=True)
        embedding_text = build_project_embedding_text(project_doc)
        project.embedding = embed_document(embedding_text)

        # Create in database
        project_id = db_create_project(project, action_note=f"Project created: {name}")

        # Return created project
        created_project = db_get_project(project_id)
        return {
            "success": True,
            "project_id": str(project_id),
            "project": self._project_to_dict(created_project)
        }

    def _update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        context: Optional[str] = None,
        status: Optional[Literal["active", "archived"]] = None,
        stakeholders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update an existing project."""
        project_oid = ObjectId(project_id)
        current_project = db_get_project(project_oid)

        if not current_project:
            return {"success": False, "error": "Project not found"}

        updates = {}
        changes = []

        if name is not None:
            updates["name"] = name
            changes.append(f"name changed to '{name}'")

        if description is not None:
            updates["description"] = description
            changes.append("description updated")

        if context is not None:
            updates["context"] = context
            changes.append("context updated")

        if stakeholders is not None:
            updates["stakeholders"] = stakeholders
            changes.append(f"stakeholders updated ({len(stakeholders)} stakeholder(s))")

        if status is not None:
            updates["status"] = status
            changes.append(f"status changed to '{status}'")

        # Re-generate embedding if any content fields changed
        if any(field is not None for field in [name, description, context, stakeholders]):
            from shared.embeddings import build_project_embedding_text

            # Build updated project document for embedding
            updated_project_doc = current_project.model_dump(by_alias=True)
            updated_project_doc.update(updates)

            embedding_text = build_project_embedding_text(updated_project_doc)
            updates["embedding"] = embed_document(embedding_text)

        # Update in database
        action_note = "; ".join(changes) if changes else "Project updated"
        success = db_update_project(project_oid, updates, "updated", action_note)

        # Return updated project
        updated_project = db_get_project(project_oid)
        return {
            "success": success,
            "project": self._project_to_dict(updated_project)
        }

    def _add_note(
        self,
        target_type: Literal["task", "project"],
        target_id: str,
        note: str
    ) -> Dict[str, Any]:
        """Add a note to a task or project."""
        target_oid = ObjectId(target_id)

        if target_type == "task":
            success = add_task_note(target_oid, note)
        else:  # project
            success = add_project_note(target_oid, note)

        return {
            "success": success,
            "message": f"Note added to {target_type}"
        }

    def _add_context(
        self,
        target_type: Literal["task", "project"],
        target_id: str,
        context: str
    ) -> Dict[str, Any]:
        """Add or update context for a task or project."""
        target_oid = ObjectId(target_id)

        if target_type == "task":
            # Get current task to regenerate embedding
            current_task = db_get_task(target_oid)
            if not current_task:
                return {"success": False, "error": "Task not found"}

            embedding_text = f"{current_task.title}\n{context}".strip()
            embedding = embed_document(embedding_text)

            success = db_update_task(
                target_oid,
                {"context": context, "embedding": embedding},
                "context_added",
                "Context updated"
            )
        else:  # project
            # Get current project to regenerate embedding
            current_project = db_get_project(target_oid)
            if not current_project:
                return {"success": False, "error": "Project not found"}

            embedding_text = f"{current_project.name}\n{current_project.description}".strip()
            embedding = embed_document(embedding_text)

            success = db_update_project(
                target_oid,
                {"context": context, "embedding": embedding},
                "context_added",
                "Context updated"
            )

        return {
            "success": success,
            "message": f"Context updated for {target_type}"
        }

    def _add_decision(self, project_id: str, decision: str) -> Dict[str, Any]:
        """Add a decision to a project."""
        project_oid = ObjectId(project_id)
        success = add_project_decision(project_oid, decision)

        return {
            "success": success,
            "message": "Decision added to project"
        }

    def _add_method(self, project_id: str, method: str) -> Dict[str, Any]:
        """Add a method/technology to a project."""
        project_oid = ObjectId(project_id)
        success = add_project_method(project_oid, method)

        return {
            "success": success,
            "message": "Method added to project"
        }

    def _add_blocker(self, task_id: str, blocker: str) -> Dict[str, Any]:
        """Add a blocker to a task."""
        task_oid = ObjectId(task_id)
        task = db_get_task(task_oid)

        if not task:
            return {"success": False, "error": "Task not found"}

        # Add blocker to list
        current_blockers = task.blockers or []
        if blocker not in current_blockers:
            current_blockers.append(blocker)

            success = db_update_task(
                task_oid,
                {"blockers": current_blockers},
                "blocker_added",
                f"Blocker added: {blocker}"
            )

            return {
                "success": success,
                "message": f"Blocker added to task",
                "blocker": blocker
            }
        else:
            return {
                "success": False,
                "message": "Blocker already exists"
            }

    def _remove_blocker(self, task_id: str, blocker: str) -> Dict[str, Any]:
        """Remove a blocker from a task."""
        task_oid = ObjectId(task_id)
        task = db_get_task(task_oid)

        if not task:
            return {"success": False, "error": "Task not found"}

        # Remove blocker from list
        current_blockers = task.blockers or []
        if blocker in current_blockers:
            current_blockers.remove(blocker)

            success = db_update_task(
                task_oid,
                {"blockers": current_blockers},
                "blocker_removed",
                f"Blocker removed: {blocker}"
            )

            return {
                "success": success,
                "message": f"Blocker removed from task",
                "blocker": blocker
            }
        else:
            return {
                "success": False,
                "message": "Blocker not found"
            }

    def _add_stakeholder(self, project_id: str, stakeholder: str) -> Dict[str, Any]:
        """Add a stakeholder to a project."""
        project_oid = ObjectId(project_id)
        project = db_get_project(project_oid)

        if not project:
            return {"success": False, "error": "Project not found"}

        # Add stakeholder to list
        current_stakeholders = project.stakeholders or []
        if stakeholder not in current_stakeholders:
            current_stakeholders.append(stakeholder)

            success = db_update_project(
                project_oid,
                {"stakeholders": current_stakeholders},
                "stakeholder_added",
                f"Stakeholder added: {stakeholder}"
            )

            return {
                "success": success,
                "message": f"Stakeholder added to project",
                "stakeholder": stakeholder
            }
        else:
            return {
                "success": False,
                "message": "Stakeholder already exists"
            }

    def _add_project_update(self, project_id: str, update_content: str) -> Dict[str, Any]:
        """Add a status update to a project."""
        project_oid = ObjectId(project_id)
        project = db_get_project(project_oid)

        if not project:
            return {"success": False, "error": "Project not found"}

        # Create new update
        new_update = {
            "date": datetime.utcnow(),
            "content": update_content
        }

        # Add to updates list
        current_updates = project.updates or []
        current_updates.append(new_update)

        success = db_update_project(
            project_oid,
            {"updates": current_updates},
            "update_added",
            f"Status update added"
        )

        return {
            "success": success,
            "message": "Status update added to project",
            "update": new_update
        }

    def _parse_due_date(self, due_date_str: str) -> Optional[datetime]:
        """Parse a due date string into a datetime object.

        Supports:
        - ISO format: "2024-12-31"
        - Relative: "in 3 days", "next Friday", "tomorrow"

        Args:
            due_date_str: Date string to parse

        Returns:
            Parsed datetime or None if parsing fails
        """
        from dateutil import parser
        from dateutil.relativedelta import relativedelta
        import re

        due_date_str = due_date_str.strip().lower()

        try:
            # Try ISO format first
            if re.match(r'\d{4}-\d{2}-\d{2}', due_date_str):
                return datetime.fromisoformat(due_date_str)

            # Handle relative dates
            now = datetime.utcnow()

            if due_date_str == "today":
                return now.replace(hour=23, minute=59, second=59)
            elif due_date_str == "tomorrow":
                return (now + relativedelta(days=1)).replace(hour=23, minute=59, second=59)
            elif due_date_str.startswith("in ") and "day" in due_date_str:
                # "in 3 days"
                match = re.search(r'in (\d+) day', due_date_str)
                if match:
                    days = int(match.group(1))
                    return (now + relativedelta(days=days)).replace(hour=23, minute=59, second=59)
            elif due_date_str.startswith("in ") and "week" in due_date_str:
                # "in 2 weeks"
                match = re.search(r'in (\d+) week', due_date_str)
                if match:
                    weeks = int(match.group(1))
                    return (now + relativedelta(weeks=weeks)).replace(hour=23, minute=59, second=59)
            elif "next" in due_date_str:
                # "next friday", "next week"
                # Use dateutil parser for natural language
                return parser.parse(due_date_str, fuzzy=True)

            # Fall back to dateutil parser
            return parser.parse(due_date_str, fuzzy=True)

        except Exception as e:
            logger.warning(f"Failed to parse due date '{due_date_str}': {e}")
            return None

    def _list_tasks(
        self,
        project_id: Optional[str] = None,
        status: Optional[Literal["todo", "in_progress", "done"]] = None,
        priority: Optional[Literal["low", "medium", "high"]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """List tasks with optional filters."""
        import time

        # Track timings for debug panel
        timings = {}

        collection = get_collection(TASKS_COLLECTION)

        # Build query
        query = {}
        if project_id:
            query["project_id"] = ObjectId(project_id)
        if status:
            query["status"] = status
        if priority:
            query["priority"] = priority

        # Time MongoDB query execution
        start = time.time()
        cursor = collection.find(query, {"embedding": 0}).sort("created_at", -1).limit(limit)
        tasks = [Task(**doc) for doc in cursor]
        timings["mongodb_query"] = int((time.time() - start) * 1000)

        # Store timings for coordinator to access
        self.last_query_timings = timings

        return {
            "success": True,
            "count": len(tasks),
            "tasks": [self._task_to_dict(task) for task in tasks]
        }

    def _list_projects(
        self,
        status: Optional[Literal["active", "archived"]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """List projects with optional filters."""
        collection = get_collection(PROJECTS_COLLECTION)

        # Build query
        query = {}
        if status:
            query["status"] = status

        # Execute query - exclude embedding field for performance
        cursor = collection.find(query, {"embedding": 0}).sort("created_at", -1).limit(limit)
        projects = [Project(**doc) for doc in cursor]

        return {
            "success": True,
            "count": len(projects),
            "projects": [self._project_to_dict(project) for project in projects]
        }

    def _get_task(self, task_id: str) -> Dict[str, Any]:
        """Get a single task by ID."""
        task_oid = ObjectId(task_id)
        task = db_get_task(task_oid)

        if not task:
            return {"success": False, "error": "Task not found"}

        return {
            "success": True,
            "task": self._task_to_dict(task)
        }

    def _get_project(self, project_id: str) -> Dict[str, Any]:
        """Get a single project by ID."""
        project_oid = ObjectId(project_id)
        project = db_get_project(project_oid)

        if not project:
            return {"success": False, "error": "Project not found"}

        return {
            "success": True,
            "project": self._project_to_dict(project)
        }

    def apply_voice_update(
        self,
        task_id: Optional[str] = None,
        project_id: Optional[str] = None,
        updates: Optional[Dict[str, Any]] = None,
        voice_log_entry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply batch updates from parsed voice input to a task or project.

        Args:
            task_id: Optional task ID to update
            project_id: Optional project ID to update
            updates: dict with fields to update:
                - status: new status
                - notes_to_add: list of notes to add
                - context: context to add/update
            voice_log_entry: dict with:
                - summary: cleaned summary
                - raw_transcript: original speech
                - extracted: parsed voice structure

        Returns:
            dict with success status and updated entity
        """
        logger.info(f"apply_voice_update: task_id={task_id}, project_id={project_id}")
        logger.debug(f"Updates: {updates}")
        logger.debug(f"Voice log summary: {voice_log_entry.get('summary', 'N/A')}")

        if not task_id and not project_id:
            logger.warning("apply_voice_update called without task_id or project_id")
            return {"success": False, "error": "Must provide either task_id or project_id"}

        updates = updates or {}
        voice_log_entry = voice_log_entry or {}

        # Process task update
        if task_id:
            logger.info(f"Applying voice update to task {task_id}")
            task_oid = ObjectId(task_id)
            current_task = db_get_task(task_oid)

            if not current_task:
                logger.error(f"Task {task_id} not found")
                return {"success": False, "error": "Task not found"}

            # Build update dict
            update_fields = {}
            changes = []

            # Update status if provided
            if "status" in updates:
                logger.info(f"Updating task status: {current_task.status} → {updates['status']}")
                update_fields["status"] = updates["status"]
                changes.append(f"status changed to '{updates['status']}'")
                if updates["status"] == "done":
                    update_fields["completed_at"] = datetime.utcnow()

            # Add context if provided
            if "context" in updates:
                # Append to existing context
                new_context = current_task.context
                if new_context:
                    new_context += f"\n\n{updates['context']}"
                else:
                    new_context = updates["context"]
                update_fields["context"] = new_context
                changes.append("context updated")

                # Regenerate embedding
                embedding_text = f"{current_task.title}\n{new_context}".strip()
                update_fields["embedding"] = embed_document(embedding_text)

            # Create voice activity log entry
            from shared.models import ActivityLogEntry
            log_entry = ActivityLogEntry(
                action="voice_update",
                note=voice_log_entry.get("summary", "Voice update"),
                summary=voice_log_entry.get("summary"),
                raw_transcript=voice_log_entry.get("raw_transcript"),
                extracted=voice_log_entry.get("extracted")
            )

            # Add to activity log
            if "activity_log" not in update_fields:
                update_fields["activity_log"] = current_task.activity_log
            update_fields["activity_log"].append(log_entry)

            # Apply updates
            action_note = "; ".join(changes) if changes else "Voice update applied"
            db_update_task(task_oid, update_fields, "voice_update", action_note)

            # Add notes separately if provided
            if "notes_to_add" in updates:
                for note in updates["notes_to_add"]:
                    add_task_note(task_oid, note)

            # Return updated task
            updated_task = db_get_task(task_oid)
            return {
                "success": True,
                "task": self._task_to_dict(updated_task)
            }

        # Process project update
        else:  # project_id
            project_oid = ObjectId(project_id)
            current_project = db_get_project(project_oid)

            if not current_project:
                return {"success": False, "error": "Project not found"}

            # Build update dict
            update_fields = {}
            changes = []

            # Update status if provided
            if "status" in updates:
                update_fields["status"] = updates["status"]
                changes.append(f"status changed to '{updates['status']}'")

            # Add context if provided
            if "context" in updates:
                # Append to existing context
                new_context = current_project.context
                if new_context:
                    new_context += f"\n\n{updates['context']}"
                else:
                    new_context = updates["context"]
                update_fields["context"] = new_context
                changes.append("context updated")

                # Regenerate embedding
                embedding_text = f"{current_project.name}\n{current_project.description}".strip()
                update_fields["embedding"] = embed_document(embedding_text)

            # Create voice activity log entry
            from shared.models import ActivityLogEntry
            log_entry = ActivityLogEntry(
                action="voice_update",
                note=voice_log_entry.get("summary", "Voice update"),
                summary=voice_log_entry.get("summary"),
                raw_transcript=voice_log_entry.get("raw_transcript"),
                extracted=voice_log_entry.get("extracted")
            )

            # Add to activity log
            if "activity_log" not in update_fields:
                update_fields["activity_log"] = current_project.activity_log
            update_fields["activity_log"].append(log_entry)

            # Apply updates
            action_note = "; ".join(changes) if changes else "Voice update applied"
            db_update_project(project_oid, update_fields, "voice_update", action_note)

            # Add notes separately if provided
            if "notes_to_add" in updates:
                for note in updates["notes_to_add"]:
                    add_project_note(project_oid, note)

            # Return updated project
            updated_project = db_get_project(project_oid)
            return {
                "success": True,
                "project": self._project_to_dict(updated_project)
            }

    def create_task_from_voice(
        self,
        title: str,
        project_id: Optional[str] = None,
        source: Optional[str] = None,
        context: str = "",
        from_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a task that was mentioned in voice input.

        Args:
            title: Task title
            project_id: Optional project ID
            source: Source attribution (e.g., "mentioned by Sarah in standup")
            context: Task context
            from_transcript: Original transcript snippet

        Returns:
            dict with success status and created task
        """
        # Build context with source attribution
        full_context = context
        if source:
            attribution = f"Source: {source}"
            full_context = f"{attribution}\n\n{context}" if context else attribution

        # Create task model
        task = Task(
            title=title,
            project_id=ObjectId(project_id) if project_id else None,
            context=full_context,
            status="todo"
        )

        # Generate embedding
        embedding_text = f"{title}\n{full_context}".strip()
        task.embedding = embed_document(embedding_text)

        # Add voice creation log entry
        from shared.models import ActivityLogEntry
        log_entry = ActivityLogEntry(
            action="created_from_voice",
            note=f"Task created from voice input{': ' + source if source else ''}",
            summary=f"Created: {title}",
            raw_transcript=from_transcript
        )
        task.activity_log.append(log_entry)

        # Create in database
        action_note = f"Task created from voice: {title}"
        if source:
            action_note += f" (source: {source})"
        task_id = db_create_task(task, action_note=action_note)

        # Return created task
        created_task = db_get_task(task_id)
        return {
            "success": True,
            "task_id": str(task_id),
            "task": self._task_to_dict(created_task)
        }

    def _task_to_dict(self, task: Optional[Task]) -> Optional[Dict[str, Any]]:
        """Convert Task model to dictionary for JSON serialization."""
        if not task:
            return None

        task_dict = task.model_dump(exclude={"embedding"})  # Exclude large embedding
        task_dict["_id"] = str(task.id) if task.id else None
        task_dict["project_id"] = str(task.project_id) if task.project_id else None
        return task_dict

    def _project_to_dict(self, project: Optional[Project]) -> Optional[Dict[str, Any]]:
        """Convert Project model to dictionary for JSON serialization."""
        if not project:
            return None

        project_dict = project.model_dump(exclude={"embedding"})  # Exclude large embedding
        project_dict["_id"] = str(project.id) if project.id else None
        return project_dict

    def process(self, user_message: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Process a user message and execute appropriate tools.

        Args:
            user_message: User's natural language request
            conversation_history: Optional conversation history

        Returns:
            Agent's response
        """
        # Build messages
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": user_message})

        # System prompt for the agent
        system_prompt = """You are a helpful task and project management assistant.
You have access to tools for creating, updating, and managing tasks and projects.

⚠️ CRITICAL: You MUST use tools to perform ALL data operations. NEVER respond with text-only
confirmations like "✓ Created task X". If you don't call a tool, nothing happens in the database.
Always call the actual tool (create_task, update_task, etc.) and report the tool's response.

IMPORTANT - Task and Project Fields:
Tasks support these enrichment fields:
- assignee: Person or team responsible (use when user mentions "assign to X", "for X", "X's task")
- due_date: Deadline in ISO format or natural language ("tomorrow", "next Friday", "in 3 days")
- blockers: List of blockers preventing progress
- priority: low, medium, or high
- status: todo, in_progress, or done

Projects support these enrichment fields:
- stakeholders: List of people/teams involved
- updates: Status updates with timestamps
- context: Rich background information

When users ask about their tasks or projects:
- ALWAYS use the list_tasks or list_projects tools to check what exists
- Don't assume the database is empty - always check first
- Report the actual results from the tools

When creating or updating tasks/projects:
- Extract ALL relevant information from the user's message (assignee, due date, priority, etc.)
- Use the enrichment fields directly in create_task and update_task tools
- Parse natural language dates into ISO format or relative expressions
- Ask for clarification ONLY if critical information is missing (e.g., which project for a new task)
- Provide clear confirmation of actions taken with all field values

CRITICAL: When a user says "assign to Mike Chen, due tomorrow", you MUST use the assignee and due_date
parameters in the create_task tool. DO NOT say these fields don't exist or add them to context/description
as a workaround.

Example - CORRECT tool usage:
User: "Create a high priority task for API documentation, assign to Sarah, due in 5 days"
Tool call: create_task(
    title="API documentation",
    project_id="<project_id>",
    priority="high",
    assignee="Sarah Thompson",
    due_date="in 5 days"
)

Example - WRONG (do NOT do this):
User: "Create a task, assign to Mike, due tomorrow"
Tool call: create_task(
    title="task",
    context="Assigned to: Mike, Due: tomorrow"  ❌ WRONG - use assignee and due_date parameters!
)

Always use the tools to perform actual operations - don't just describe what you would do."""

        # Tool use loop
        max_iterations = 5
        iteration = 0

        # Detect if this is an action request that requires tool use
        action_keywords = ['create', 'update', 'complete', 'start', 'add', 'mark', 'assign', 'change', 'delete', 'remove']
        requires_tool = any(keyword in user_message.lower() for keyword in action_keywords)

        while iteration < max_iterations:
            iteration += 1

            # Force tool use on first iteration for action requests
            llm_kwargs = {}
            if iteration == 1 and requires_tool:
                llm_kwargs['tool_choice'] = {"type": "any"}  # Force the model to use a tool
                logger.info(f"🔧 Forcing tool use for action request: {user_message[:50]}")

            # Call Claude with tools
            response = self.llm.generate_with_tools(
                messages=messages,
                tools=self.tools,
                system=system_prompt,
                max_tokens=4096,
                **llm_kwargs
            )

            logger.info(f"📊 Response stop_reason: {response.stop_reason}")

            # Check if we're done (no tool use)
            if response.stop_reason == "end_turn":
                # Extract text response
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                return text_content

            # Process tool uses
            if response.stop_reason == "tool_use":
                # Add assistant's response to messages
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Execute tools and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id

                        # Execute tool
                        result = self.execute_tool(tool_name, tool_input)

                        # Debug logging
                        print(f"[DEBUG] Tool: {tool_name}, Input: {tool_input}")
                        print(f"[DEBUG] Result count: {result.get('count', 'N/A')}, Success: {result.get('success', 'N/A')}")

                        # Add result
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result)
                        })

                # Add tool results to messages
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

        return "Maximum iterations reached. Please try again with a simpler request."


# Global agent instance - memory manager will be set by coordinator
worklog_agent = WorklogAgent(memory_manager=None)
