"""Retrieval Agent for semantic and temporal search operations."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal
from bson import ObjectId
from rapidfuzz import fuzz

from shared.llm import llm_service
from shared.logger import get_logger
from shared.embeddings import embed_query as embedding_embed_query
from shared.db import (
    get_collection,
    get_project as db_get_project,
    TASKS_COLLECTION,
    PROJECTS_COLLECTION,
)
from shared.models import Task, Project

logger = get_logger("retrieval")


class RetrievalAgent:
    """Agent for handling search and retrieval operations using Claude."""

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
                "name": "embed_query",
                "description": "Generate embedding vector for a query text using Voyage AI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Query text to embed"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "search_semantic",
                "description": "Perform semantic vector search across tasks and/or projects",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "collections": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["tasks", "projects"]
                            },
                            "description": "Collections to search (default: both tasks and projects)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_by_date",
                "description": "Search for activity on a specific date",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to search: 'yesterday', 'today', or 'YYYY-MM-DD' format"
                        },
                        "include_incomplete": {
                            "type": "boolean",
                            "description": "Include tasks that were worked on but not completed (default: false)",
                            "default": False
                        }
                    },
                    "required": ["date"]
                }
            },
            {
                "name": "search_incomplete",
                "description": "Find tasks with activity since a date but status not done",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "since_date": {
                            "type": "string",
                            "description": "Start date: 'yesterday', 'today', or 'YYYY-MM-DD' format"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project ID to filter by"
                        }
                    },
                    "required": ["since_date"]
                }
            },
            {
                "name": "search_progress",
                "description": "Get progress summary for a project including tasks and activity",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Project ID (MongoDB ObjectId as string)"
                        },
                        "since_date": {
                            "type": "string",
                            "description": "Optional: Show activity since this date ('yesterday', 'today', or 'YYYY-MM-DD')"
                        }
                    },
                    "required": ["project_id"]
                }
            },
            {
                "name": "search_by_assignee",
                "description": "Find tasks assigned to a specific person or team",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "assignee": {
                            "type": "string",
                            "description": "Name of the person or team"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done"],
                            "description": "Optional: Filter by task status"
                        }
                    },
                    "required": ["assignee"]
                }
            },
            {
                "name": "search_blocked_tasks",
                "description": "Find all tasks that have blockers (are blocked)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "Optional: Filter by project ID"
                        }
                    }
                }
            },
            {
                "name": "search_by_due_date",
                "description": "Find tasks by due date (overdue, due soon, due this week, etc.)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "enum": ["overdue", "due_today", "due_this_week", "due_next_week", "all"],
                            "description": "Due date filter to apply"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Optional: Filter by assignee"
                        }
                    },
                    "required": ["filter"]
                }
            },
            {
                "name": "search_by_stakeholder",
                "description": "Find projects where a specific person or team is a stakeholder",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stakeholder": {
                            "type": "string",
                            "description": "Name of the stakeholder (person or team)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "planned", "completed", "archived"],
                            "description": "Optional: Filter by project status"
                        }
                    },
                    "required": ["stakeholder"]
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
            if tool_name == "embed_query":
                return self._embed_query(**tool_input)
            elif tool_name == "search_semantic":
                return self._search_semantic(**tool_input)
            elif tool_name == "search_by_date":
                return self._search_by_date(**tool_input)
            elif tool_name == "search_incomplete":
                return self._search_incomplete(**tool_input)
            elif tool_name == "search_progress":
                return self._search_progress(**tool_input)
            elif tool_name == "search_by_assignee":
                return self._search_by_assignee(**tool_input)
            elif tool_name == "search_blocked_tasks":
                return self._search_blocked_tasks(**tool_input)
            elif tool_name == "search_by_due_date":
                return self._search_by_due_date(**tool_input)
            elif tool_name == "search_by_stakeholder":
                return self._search_by_stakeholder(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse a date string into a datetime object.

        Args:
            date_str: Date string ('yesterday', 'today', or 'YYYY-MM-DD')

        Returns:
            datetime object
        """
        date_str_lower = date_str.lower()
        now = datetime.utcnow()

        if date_str_lower == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str_lower == "yesterday":
            yesterday = now - timedelta(days=1)
            return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Parse YYYY-MM-DD format
            try:
                parsed = datetime.strptime(date_str, "%Y-%m-%d")
                return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                raise ValueError(f"Invalid date format: {date_str}. Use 'yesterday', 'today', or 'YYYY-MM-DD'")

    def _embed_query(self, text: str) -> Dict[str, Any]:
        """Generate embedding vector for a query text."""
        embedding = embedding_embed_query(text)

        return {
            "success": True,
            "text": text,
            "embedding_dimensions": len(embedding),
            "embedding": embedding
        }

    def _search_semantic(
        self,
        query: str,
        collections: Optional[List[Literal["tasks", "projects"]]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Perform semantic vector search across collections.

        Args:
            query: Search query text
            collections: Collections to search (default: ["tasks", "projects"])
            limit: Maximum results per collection

        Returns:
            Search results with similarity scores
        """
        if collections is None:
            collections = ["tasks", "projects"]

        # Generate query embedding
        query_embedding = embedding_embed_query(query)

        results = {
            "success": True,
            "query": query,
            "results": []
        }

        # Search tasks
        if "tasks" in collections:
            tasks_collection = get_collection(TASKS_COLLECTION)

            # MongoDB Atlas Vector Search aggregation
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Atlas search index name
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,  # Scan more candidates for better results
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "title": 1,
                        "status": 1,
                        "priority": 1,
                        "context": 1,
                        "project_id": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            try:
                task_results = list(tasks_collection.aggregate(pipeline))
                for task in task_results:
                    task["_id"] = str(task["_id"])
                    task["project_id"] = str(task["project_id"]) if task.get("project_id") else None
                    task["type"] = "task"
                results["results"].extend(task_results)
            except Exception as e:
                results["tasks_error"] = f"Vector search failed: {str(e)}. Ensure vector search index is created."

        # Search projects
        if "projects" in collections:
            projects_collection = get_collection(PROJECTS_COLLECTION)

            # MongoDB Atlas Vector Search aggregation
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Atlas search index name
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "name": 1,
                        "description": 1,
                        "status": 1,
                        "context": 1,
                        "created_at": 1,
                        "last_activity": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            try:
                project_results = list(projects_collection.aggregate(pipeline))
                for project in project_results:
                    project["_id"] = str(project["_id"])
                    project["type"] = "project"
                results["results"].extend(project_results)
            except Exception as e:
                results["projects_error"] = f"Vector search failed: {str(e)}. Ensure vector search index is created."

        # Sort all results by score (descending)
        if "score" in results["results"][0] if results["results"] else {}:
            results["results"].sort(key=lambda x: x.get("score", 0), reverse=True)

        results["count"] = len(results["results"])
        return results

    def _search_by_date(
        self,
        date: str,
        include_incomplete: bool = False
    ) -> Dict[str, Any]:
        """
        Search for activity on a specific date.

        Args:
            date: Date string ('yesterday', 'today', or 'YYYY-MM-DD')
            include_incomplete: Include tasks worked on but not completed

        Returns:
            Chronological activity for the date
        """
        # Parse date
        target_date = self._parse_date(date)
        next_day = target_date + timedelta(days=1)

        results = {
            "success": True,
            "date": date,
            "parsed_date": target_date.isoformat(),
            "tasks": [],
            "projects": []
        }

        # Search tasks with activity on this date
        tasks_collection = get_collection(TASKS_COLLECTION)

        # Build match condition
        match_condition = {
            "activity_log.timestamp": {
                "$gte": target_date,
                "$lt": next_day
            }
        }

        if not include_incomplete:
            # Only include tasks that were completed on this date
            match_condition["completed_at"] = {
                "$gte": target_date,
                "$lt": next_day
            }

        pipeline = [
            {"$match": match_condition},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "status": 1,
                    "priority": 1,
                    "project_id": 1,
                    "created_at": 1,
                    "completed_at": 1,
                    "activity_log": {
                        "$filter": {
                            "input": "$activity_log",
                            "as": "log",
                            "cond": {
                                "$and": [
                                    {"$gte": ["$$log.timestamp", target_date]},
                                    {"$lt": ["$$log.timestamp", next_day]}
                                ]
                            }
                        }
                    }
                }
            },
            {"$sort": {"activity_log.timestamp": 1}}
        ]

        task_results = list(tasks_collection.aggregate(pipeline))
        for task in task_results:
            task["_id"] = str(task["_id"])
            task["project_id"] = str(task["project_id"]) if task.get("project_id") else None
        results["tasks"] = task_results

        # Search projects with activity on this date
        projects_collection = get_collection(PROJECTS_COLLECTION)

        pipeline = [
            {
                "$match": {
                    "activity_log.timestamp": {
                        "$gte": target_date,
                        "$lt": next_day
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "description": 1,
                    "status": 1,
                    "activity_log": {
                        "$filter": {
                            "input": "$activity_log",
                            "as": "log",
                            "cond": {
                                "$and": [
                                    {"$gte": ["$$log.timestamp", target_date]},
                                    {"$lt": ["$$log.timestamp", next_day]}
                                ]
                            }
                        }
                    }
                }
            },
            {"$sort": {"activity_log.timestamp": 1}}
        ]

        project_results = list(projects_collection.aggregate(pipeline))
        for project in project_results:
            project["_id"] = str(project["_id"])
        results["projects"] = project_results

        results["task_count"] = len(task_results)
        results["project_count"] = len(project_results)

        return results

    def _search_incomplete(
        self,
        since_date: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find tasks with activity since a date but status not done.

        Args:
            since_date: Start date string
            project_id: Optional project ID filter

        Returns:
            Incomplete tasks with recent activity
        """
        # Parse date
        start_date = self._parse_date(since_date)

        tasks_collection = get_collection(TASKS_COLLECTION)

        # Build match condition
        match_condition = {
            "status": {"$ne": "done"},
            "activity_log.timestamp": {"$gte": start_date}
        }

        if project_id:
            match_condition["project_id"] = ObjectId(project_id)

        pipeline = [
            {"$match": match_condition},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "status": 1,
                    "priority": 1,
                    "context": 1,
                    "project_id": 1,
                    "created_at": 1,
                    "last_worked_on": 1,
                    "recent_activity": {
                        "$filter": {
                            "input": "$activity_log",
                            "as": "log",
                            "cond": {"$gte": ["$$log.timestamp", start_date]}
                        }
                    }
                }
            },
            {"$sort": {"last_worked_on": -1}}
        ]

        task_results = list(tasks_collection.aggregate(pipeline))
        for task in task_results:
            task["_id"] = str(task["_id"])
            task["project_id"] = str(task["project_id"]) if task.get("project_id") else None

        return {
            "success": True,
            "since_date": since_date,
            "parsed_date": start_date.isoformat(),
            "project_id": project_id,
            "count": len(task_results),
            "tasks": task_results
        }

    def _search_progress(
        self,
        project_id: str,
        since_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get progress summary for a project.

        Args:
            project_id: Project ID
            since_date: Optional date to filter activity

        Returns:
            Progress summary with stats and activity
        """
        project_oid = ObjectId(project_id)

        # Get project
        project = db_get_project(project_oid)
        if not project:
            return {"success": False, "error": "Project not found"}

        # Get all tasks for project
        tasks_collection = get_collection(TASKS_COLLECTION)

        # Calculate task statistics
        pipeline = [
            {"$match": {"project_id": project_oid}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]

        status_counts = list(tasks_collection.aggregate(pipeline))
        stats = {
            "total": 0,
            "todo": 0,
            "in_progress": 0,
            "done": 0
        }

        for item in status_counts:
            status = item["_id"]
            count = item["count"]
            stats[status] = count
            stats["total"] += count

        # Calculate completion percentage
        stats["completion_percentage"] = (
            (stats["done"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )

        # Get recent activity
        activity_filter = {}
        if since_date:
            start_date = self._parse_date(since_date)
            activity_filter = {"activity_log.timestamp": {"$gte": start_date}}

        # Get tasks with recent activity
        match_condition = {"project_id": project_oid}
        if since_date:
            match_condition.update(activity_filter)

        recent_tasks_pipeline = [
            {"$match": match_condition},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "status": 1,
                    "last_worked_on": 1,
                    "activity_log": 1 if not since_date else {
                        "$filter": {
                            "input": "$activity_log",
                            "as": "log",
                            "cond": {"$gte": ["$$log.timestamp", start_date]}
                        }
                    }
                }
            },
            {"$sort": {"last_worked_on": -1}},
            {"$limit": 20}
        ]

        recent_tasks = list(tasks_collection.aggregate(recent_tasks_pipeline))
        for task in recent_tasks:
            task["_id"] = str(task["_id"])

        # Build result
        result = {
            "success": True,
            "project_id": project_id,
            "project": {
                "_id": str(project.id),
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "context": project.context,
                "methods": project.methods,
                "decisions": project.decisions
            },
            "statistics": stats,
            "recent_tasks": recent_tasks,
            "recent_activity_count": sum(len(task.get("activity_log", [])) for task in recent_tasks)
        }

        if since_date:
            result["since_date"] = since_date

        return result

    def _search_by_assignee(
        self,
        assignee: str,
        status: Optional[Literal["todo", "in_progress", "done"]] = None
    ) -> Dict[str, Any]:
        """
        Find tasks assigned to a specific person or team.

        Args:
            assignee: Name of the person or team
            status: Optional status filter

        Returns:
            List of tasks assigned to the person/team
        """
        tasks_collection = get_collection(TASKS_COLLECTION)

        # Build query
        query = {"assignee": assignee}
        if status:
            query["status"] = status

        # Find matching tasks
        tasks = list(tasks_collection.find(query).sort("due_date", 1).limit(50))

        # Convert ObjectId to string
        for task in tasks:
            task["_id"] = str(task["_id"])
            if task.get("project_id"):
                task["project_id"] = str(task["project_id"])

        return {
            "success": True,
            "assignee": assignee,
            "status_filter": status,
            "count": len(tasks),
            "tasks": tasks
        }

    def _search_blocked_tasks(
        self,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find all tasks that have blockers.

        Args:
            project_id: Optional project ID filter

        Returns:
            List of blocked tasks
        """
        tasks_collection = get_collection(TASKS_COLLECTION)

        # Build query
        query = {"blockers": {"$exists": True, "$ne": []}}
        if project_id:
            query["project_id"] = ObjectId(project_id)

        # Find matching tasks
        tasks = list(tasks_collection.find(query).sort("priority", -1).limit(50))

        # Convert ObjectId to string
        for task in tasks:
            task["_id"] = str(task["_id"])
            if task.get("project_id"):
                task["project_id"] = str(task["project_id"])

        return {
            "success": True,
            "project_id_filter": project_id,
            "count": len(tasks),
            "tasks": tasks
        }

    def _search_by_due_date(
        self,
        filter: Literal["overdue", "due_today", "due_this_week", "due_next_week", "all"],
        assignee: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find tasks by due date.

        Args:
            filter: Due date filter to apply
            assignee: Optional assignee filter

        Returns:
            List of tasks matching the due date filter
        """
        from datetime import datetime, timedelta

        tasks_collection = get_collection(TASKS_COLLECTION)
        now = datetime.utcnow()

        # Build query
        query = {"due_date": {"$exists": True, "$ne": None}}

        if filter == "overdue":
            query["due_date"] = {"$lt": now}
            query["status"] = {"$ne": "done"}
        elif filter == "due_today":
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            query["due_date"] = {"$gte": start_of_day, "$lte": end_of_day}
        elif filter == "due_this_week":
            end_of_week = now + timedelta(days=(7 - now.weekday()))
            query["due_date"] = {"$gte": now, "$lte": end_of_week}
        elif filter == "due_next_week":
            start_of_next_week = now + timedelta(days=(7 - now.weekday() + 1))
            end_of_next_week = start_of_next_week + timedelta(days=7)
            query["due_date"] = {"$gte": start_of_next_week, "$lte": end_of_next_week}
        # "all" - no date filter, just has due_date

        if assignee:
            query["assignee"] = assignee

        # Find matching tasks
        tasks = list(tasks_collection.find(query).sort("due_date", 1).limit(50))

        # Convert ObjectId to string
        for task in tasks:
            task["_id"] = str(task["_id"])
            if task.get("project_id"):
                task["project_id"] = str(task["project_id"])

        return {
            "success": True,
            "filter": filter,
            "assignee_filter": assignee,
            "count": len(tasks),
            "tasks": tasks
        }

    def _search_by_stakeholder(
        self,
        stakeholder: str,
        status: Optional[Literal["active", "planned", "completed", "archived"]] = None
    ) -> Dict[str, Any]:
        """
        Find projects where a specific person or team is a stakeholder.

        Args:
            stakeholder: Name of the stakeholder
            status: Optional status filter

        Returns:
            List of projects with the stakeholder
        """
        projects_collection = get_collection(PROJECTS_COLLECTION)

        # Build query
        query = {"stakeholders": stakeholder}
        if status:
            query["status"] = status

        # Find matching projects
        projects = list(projects_collection.find(query).sort("last_activity", -1).limit(50))

        # Convert ObjectId to string
        for project in projects:
            project["_id"] = str(project["_id"])

        return {
            "success": True,
            "stakeholder": stakeholder,
            "status_filter": status,
            "count": len(projects),
            "projects": projects
        }

    def fuzzy_match_task(
        self,
        reference: str,
        project_hint: Optional[str] = None,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Fuzzy match a task reference using both vector and text similarity.

        Args:
            reference: Informal task mention (e.g., "the login thing")
            project_hint: Optional project name/ID hint to narrow search
            threshold: Minimum confidence score (0.0-1.0)

        Returns:
            dict with:
                - match: Best matching task (if confidence > threshold)
                - confidence: Score 0.0-1.0
                - alternatives: List of other candidates if ambiguous
        """
        logger.info(f"fuzzy_match_task: reference='{reference}', project_hint={project_hint}, threshold={threshold}")
        tasks_collection = get_collection(TASKS_COLLECTION)

        # Get query embedding for semantic search
        logger.debug(f"Generating embedding for: '{reference}'")
        query_embedding = embedding_embed_query(reference)
        logger.debug(f"Embedding generated: {len(query_embedding)} dimensions")

        # Build base match condition
        match_condition = {"is_test": {"$ne": True}}  # Always exclude test data
        if project_hint:
            # Try to match project by name or ID
            project_match = self.fuzzy_match_project(project_hint, threshold=0.6)
            if project_match.get("match"):
                project_id = project_match["match"]["_id"]
                match_condition["project_id"] = ObjectId(project_id)

        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": 10
                }
            },
            # Filter test data and optionally project
            {"$match": match_condition},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "status": 1,
                    "priority": 1,
                    "context": 1,
                    "project_id": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        try:
            logger.debug("Executing vector search pipeline...")
            candidates = list(tasks_collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(candidates)} candidate(s)")
            for i, c in enumerate(candidates[:3]):
                logger.debug(f"  Candidate {i+1}: '{c.get('title', 'N/A')}' (vector_score={c.get('vector_score', 0.0):.3f})")
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return {
                "match": None,
                "confidence": 0.0,
                "alternatives": [],
                "error": f"Vector search failed: {str(e)}"
            }

        if not candidates:
            logger.warning(f"No candidates found for reference '{reference}'")
            return {
                "match": None,
                "confidence": 0.0,
                "alternatives": []
            }

        # Calculate combined scores (vector + text similarity)
        logger.debug("Calculating combined scores (60% vector + 40% text)...")
        scored_candidates = []
        for task in candidates:
            # Normalize vector score (typically 0.0-1.0 but can vary)
            vector_score = min(task.get("vector_score", 0.0), 1.0)

            # Text similarity using fuzzy matching
            text_score = fuzz.ratio(reference.lower(), task["title"].lower()) / 100.0

            # Combined score (weighted average: 60% vector, 40% text)
            combined_score = (0.6 * vector_score) + (0.4 * text_score)

            logger.debug(f"  '{task['title'][:40]}...' → vector={vector_score:.3f}, text={text_score:.3f}, combined={combined_score:.3f}")

            task["_id"] = str(task["_id"])
            task["project_id"] = str(task["project_id"]) if task.get("project_id") else None
            task["confidence"] = combined_score
            task.pop("vector_score", None)

            scored_candidates.append(task)

        # Sort by combined score
        scored_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # Get best match
        best_match = scored_candidates[0]
        best_confidence = best_match["confidence"]

        logger.info(f"Best match: '{best_match['title']}' (confidence={best_confidence:.3f}, threshold={threshold})")

        # Check if best match meets threshold
        if best_confidence >= threshold:
            logger.info(f"✓ Match found above threshold!")
            # Check if there are close alternatives (within 0.1 of best)
            alternatives = [
                task for task in scored_candidates[1:4]
                if task["confidence"] >= threshold and (best_confidence - task["confidence"]) <= 0.1
            ]
            if alternatives:
                logger.debug(f"Found {len(alternatives)} close alternative(s)")

            return {
                "match": best_match,
                "confidence": best_confidence,
                "alternatives": alternatives
            }
        else:
            logger.warning(f"✗ No match above threshold (best={best_confidence:.3f} < {threshold})")
            # No match above threshold, return top candidates
            return {
                "match": None,
                "confidence": best_confidence,
                "alternatives": scored_candidates[:3]
            }

    def fuzzy_match_project(
        self,
        reference: str,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Fuzzy match a project reference using both vector and text similarity.

        Args:
            reference: Informal project mention (e.g., "the mobile app")
            threshold: Minimum confidence score (0.0-1.0)

        Returns:
            dict with:
                - match: Best matching project (if confidence > threshold)
                - confidence: Score 0.0-1.0
                - alternatives: List of other candidates if ambiguous
        """
        logger.info(f"fuzzy_match_project: reference='{reference}', threshold={threshold}")
        projects_collection = get_collection(PROJECTS_COLLECTION)

        # Get query embedding for semantic search
        logger.debug(f"Generating embedding for project reference: '{reference}'")
        query_embedding = embedding_embed_query(reference)

        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": 10
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "description": 1,
                    "status": 1,
                    "context": 1,
                    "created_at": 1,
                    "last_activity": 1,
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        try:
            logger.debug("Executing project vector search...")
            candidates = list(projects_collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(candidates)} project candidate(s)")
        except Exception as e:
            logger.error(f"Project vector search failed: {e}", exc_info=True)
            return {
                "match": None,
                "confidence": 0.0,
                "alternatives": [],
                "error": f"Vector search failed: {str(e)}"
            }

        if not candidates:
            logger.warning(f"No project candidates found for reference '{reference}'")
            return {
                "match": None,
                "confidence": 0.0,
                "alternatives": []
            }

        # Calculate combined scores (vector + text similarity)
        logger.debug("Calculating project combined scores...")
        scored_candidates = []
        for project in candidates:
            # Normalize vector score
            vector_score = min(project.get("vector_score", 0.0), 1.0)

            # Text similarity using fuzzy matching on name
            text_score = fuzz.ratio(reference.lower(), project["name"].lower()) / 100.0

            # Combined score (weighted average: 60% vector, 40% text)
            combined_score = (0.6 * vector_score) + (0.4 * text_score)

            logger.debug(f"  '{project['name'][:40]}...' → vector={vector_score:.3f}, text={text_score:.3f}, combined={combined_score:.3f}")

            project["_id"] = str(project["_id"])
            project["confidence"] = combined_score
            project.pop("vector_score", None)

            scored_candidates.append(project)

        # Sort by combined score
        scored_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # Get best match
        best_match = scored_candidates[0]
        best_confidence = best_match["confidence"]

        logger.info(f"Best project match: '{best_match['name']}' (confidence={best_confidence:.3f})")

        # Check if best match meets threshold
        if best_confidence >= threshold:
            logger.info(f"✓ Project match found above threshold!")
            # Check if there are close alternatives (within 0.1 of best)
            alternatives = [
                proj for proj in scored_candidates[1:4]
                if proj["confidence"] >= threshold and (best_confidence - proj["confidence"]) <= 0.1
            ]

            return {
                "match": best_match,
                "confidence": best_confidence,
                "alternatives": alternatives
            }
        else:
            logger.warning(f"✗ No project match above threshold (best={best_confidence:.3f} < {threshold})")
            # No match above threshold, return top candidates
            return {
                "match": None,
                "confidence": best_confidence,
                "alternatives": scored_candidates[:3]
            }

    def hybrid_search_tasks(self, query: str, limit: int = 5,
                           status: str = None, priority: str = None,
                           project_id: str = None, assignee: str = None) -> list:
        """
        Hybrid search combining vector + full-text with optional filters.

        Examples:
            "the debugging doc" → "Create debugging methodologies doc"
            "memory", priority="high", status="in_progress" → High-priority in-progress memory tasks
            "tasks", assignee="Mike Chen", status="in_progress" → Mike Chen's in-progress tasks

        Args:
            query: Semantic search query
            limit: Maximum number of results to return
            status: Optional filter by status (todo, in_progress, done)
            priority: Optional filter by priority (low, medium, high)
            project_id: Optional filter by project_id
            assignee: Optional filter by assignee (partial match, case-insensitive). "Mike" matches "Mike Chen"

        Returns:
            List of task dicts with _id, title, context, status, priority, assignee, project_id, project_name, score
        """
        import time

        logger.info(f"hybrid_search_tasks: query='{query}', limit={limit}, status={status}, priority={priority}, project_id={project_id}, assignee={assignee}")

        # Track timings for debug panel
        timings = {}

        # Time embedding generation
        start = time.time()
        query_embedding = embedding_embed_query(query)
        timings["embedding_generation"] = int((time.time() - start) * 1000)
        logger.debug(f"Query embedding generated: {len(query_embedding)} dimensions ({timings['embedding_generation']}ms)")

        # Build hybrid search pipeline
        # Use higher multiplier (10x) to get more results before post-filtering
        search_limit = limit * 10

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorSearch": [
                                {
                                    "$vectorSearch": {
                                        "index": "vector_index",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": max(100, search_limit),
                                        "limit": search_limit
                                    }
                                }
                            ],
                            "textSearch": [
                                {
                                    "$search": {
                                        "index": "tasks_text_index",
                                        "text": {
                                            "query": query,
                                            "path": ["title", "context", "notes"],
                                            "fuzzy": {"maxEdits": 1}
                                        }
                                    }
                                },
                                {"$limit": search_limit}
                            ]
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorSearch": 0.6,
                            "textSearch": 0.4
                        }
                    }
                }
            },
            # Filter out test data and apply optional filters AFTER ranking
            {
                "$match": {
                    "is_test": {"$ne": True}
                }
            },
            # Lookup project name from project_id
            {
                "$lookup": {
                    "from": "projects",
                    "localField": "project_id",
                    "foreignField": "_id",
                    "as": "project_doc"
                }
            },
            {
                "$addFields": {
                    "project_name": {"$arrayElemAt": ["$project_doc.name", 0]}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "context": 1,
                    "status": 1,
                    "priority": 1,
                    "assignee": 1,
                    "project_id": 1,
                    "project_name": 1,
                    "score": {"$meta": "score"}
                }
            },
            {"$limit": limit}
        ]

        # Build filter conditions for post-filtering (after ranking)
        filter_conditions = {"is_test": {"$ne": True}}
        if status:
            filter_conditions["status"] = status
        if priority:
            filter_conditions["priority"] = priority
        if project_id:
            from bson import ObjectId
            filter_conditions["project_id"] = ObjectId(project_id)
        if assignee:
            # Use case-insensitive partial match for assignee
            # "Mike" matches "Mike Chen", "mike" matches "Mike Chen", etc.
            filter_conditions["assignee"] = {"$regex": assignee, "$options": "i"}

        # Update the match stage with all filters
        pipeline[1]["$match"] = filter_conditions

        # Execute search
        try:
            # Time MongoDB query execution
            start = time.time()
            tasks_collection = get_collection(TASKS_COLLECTION)
            results = list(tasks_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            # Calculate processing overhead
            timings["processing"] = max(0, timings.get("total", 0) - timings["embedding_generation"] - timings["mongodb_query"])

            # Store timings for coordinator to access
            self.last_query_timings = timings

            logger.info(f"Hybrid search returned {len(results)} task(s) (embed: {timings['embedding_generation']}ms, db: {timings['mongodb_query']}ms)")

            # Log top results
            for i, task in enumerate(results[:3], 1):
                logger.debug(f"  {i}. '{task['title'][:50]}...' (score={task.get('score', 0):.3f})")

            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            self.last_query_timings = timings  # Store even on error
            return []

    def hybrid_search_projects(self, query: str, limit: int = 5) -> list:
        """
        Hybrid search combining vector + full-text for matching informal project references.

        Args:
            query: Informal project reference from voice input
            limit: Maximum number of results to return

        Returns:
            List of project dicts with _id, name, description, context, status, stakeholders, score
        """
        import time

        logger.info(f"hybrid_search_projects: query='{query}', limit={limit}")

        # Track timings for debug panel
        timings = {}

        # Time embedding generation
        start = time.time()
        query_embedding = embedding_embed_query(query)
        timings["embedding_generation"] = int((time.time() - start) * 1000)
        logger.debug(f"Query embedding generated: {len(query_embedding)} dimensions ({timings['embedding_generation']}ms)")

        # Build hybrid search pipeline
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorSearch": [
                                {
                                    "$vectorSearch": {
                                        "index": "vector_index",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": max(50, limit * 2),
                                        "limit": limit * 2
                                    }
                                }
                            ],
                            "textSearch": [
                                {
                                    "$search": {
                                        "index": "projects_text_index",
                                        "text": {
                                            "query": query,
                                            "path": ["name", "description", "context"],
                                            "fuzzy": {"maxEdits": 1}
                                        }
                                    }
                                },
                                {"$limit": limit * 2}
                            ]
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorSearch": 0.6,
                            "textSearch": 0.4
                        }
                    }
                }
            },
            # Filter out test data
            {
                "$match": {
                    "is_test": {"$ne": True}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "description": 1,
                    "context": 1,
                    "status": 1,
                    "stakeholders": 1,
                    "score": {"$meta": "score"}
                }
            },
            {"$limit": limit}
        ]

        # Execute search
        try:
            # Time MongoDB query execution
            start = time.time()
            projects_collection = get_collection(PROJECTS_COLLECTION)
            results = list(projects_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            # Store timings for coordinator to access
            self.last_query_timings = timings

            logger.info(f"Hybrid search returned {len(results)} project(s) (embed: {timings['embedding_generation']}ms, db: {timings['mongodb_query']}ms)")

            # Log top results
            for i, proj in enumerate(results[:3], 1):
                logger.debug(f"  {i}. '{proj['name'][:50]}...' (score={proj.get('score', 0):.3f})")

            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            self.last_query_timings = timings  # Store even on error
            return []

    def vector_search_tasks(self, query: str, limit: int = 5) -> list:
        """
        Vector-only semantic search for tasks using Voyage embeddings.
        Faster than hybrid but may miss exact keyword matches.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of task dicts with _id, title, context, status, project_id, score
        """
        import time

        logger.info(f"vector_search_tasks: query='{query}', limit={limit}")

        # Track timings
        timings = {}

        # Time embedding generation
        start = time.time()
        query_embedding = embedding_embed_query(query)
        timings["embedding_generation"] = int((time.time() - start) * 1000)

        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            # Lookup project name from project_id
            {
                "$lookup": {
                    "from": "projects",
                    "localField": "project_id",
                    "foreignField": "_id",
                    "as": "project_doc"
                }
            },
            {
                "$addFields": {
                    "project_name": {"$arrayElemAt": ["$project_doc.name", 0]}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "context": 1,
                    "status": 1,
                    "project_id": 1,
                    "project_name": 1,
                    "priority": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Execute search
        try:
            start = time.time()
            tasks_collection = get_collection(TASKS_COLLECTION)
            results = list(tasks_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            self.last_query_timings = timings

            logger.info(f"Vector search returned {len(results)} task(s) (embed: {timings['embedding_generation']}ms, db: {timings['mongodb_query']}ms)")

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            self.last_query_timings = timings
            return []

    def text_search_tasks(self, query: str, limit: int = 5) -> list:
        """
        Text-only keyword search for tasks using MongoDB text index.
        Fastest search method but purely keyword-based (no semantic understanding).

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of task dicts with _id, title, context, status, project_id, score
        """
        import time

        logger.info(f"text_search_tasks: query='{query}', limit={limit}")

        # Track timings (no embedding for text search)
        timings = {"embedding_generation": 0}

        # Build text search pipeline
        pipeline = [
            {
                "$search": {
                    "index": "tasks_text_index",
                    "text": {
                        "query": query,
                        "path": ["title", "context", "notes.content"],
                        "fuzzy": {"maxEdits": 1}
                    }
                }
            },
            # Lookup project name from project_id
            {
                "$lookup": {
                    "from": "projects",
                    "localField": "project_id",
                    "foreignField": "_id",
                    "as": "project_doc"
                }
            },
            {
                "$addFields": {
                    "project_name": {"$arrayElemAt": ["$project_doc.name", 0]}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "context": 1,
                    "status": 1,
                    "project_id": 1,
                    "project_name": 1,
                    "priority": 1,
                    "score": {"$meta": "searchScore"}
                }
            },
            {"$limit": limit}
        ]

        # Execute search
        try:
            start = time.time()
            tasks_collection = get_collection(TASKS_COLLECTION)
            results = list(tasks_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            self.last_query_timings = timings

            logger.info(f"Text search returned {len(results)} task(s) (db: {timings['mongodb_query']}ms)")

            return results
        except Exception as e:
            logger.error(f"Text search failed: {e}", exc_info=True)
            self.last_query_timings = timings
            return []

    def vector_search_projects(self, query: str, limit: int = 5) -> list:
        """
        Vector-only semantic search for projects using Voyage embeddings.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of project dicts with _id, name, description, context, status, score
        """
        import time

        logger.info(f"vector_search_projects: query='{query}', limit={limit}")

        # Track timings
        timings = {}

        # Time embedding generation
        start = time.time()
        query_embedding = embedding_embed_query(query)
        timings["embedding_generation"] = int((time.time() - start) * 1000)

        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "description": 1,
                    "context": 1,
                    "status": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Execute search
        try:
            start = time.time()
            projects_collection = get_collection(PROJECTS_COLLECTION)
            results = list(projects_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            self.last_query_timings = timings

            logger.info(f"Vector search returned {len(results)} project(s) (embed: {timings['embedding_generation']}ms, db: {timings['mongodb_query']}ms)")

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            self.last_query_timings = timings
            return []

    def text_search_projects(self, query: str, limit: int = 5) -> list:
        """
        Text-only keyword search for projects using MongoDB text index.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of project dicts with _id, name, description, context, status, score
        """
        import time

        logger.info(f"text_search_projects: query='{query}', limit={limit}")

        # Track timings (no embedding for text search)
        timings = {"embedding_generation": 0}

        # Build text search pipeline
        pipeline = [
            {
                "$search": {
                    "index": "projects_text_index",
                    "text": {
                        "query": query,
                        "path": ["name", "description", "context"],
                        "fuzzy": {"maxEdits": 1}
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "description": 1,
                    "context": 1,
                    "status": 1,
                    "score": {"$meta": "searchScore"}
                }
            },
            {"$limit": limit}
        ]

        # Execute search
        try:
            start = time.time()
            projects_collection = get_collection(PROJECTS_COLLECTION)
            results = list(projects_collection.aggregate(pipeline))
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            self.last_query_timings = timings

            logger.info(f"Text search returned {len(results)} project(s) (db: {timings['mongodb_query']}ms)")

            return results
        except Exception as e:
            logger.error(f"Text search failed: {e}", exc_info=True)
            self.last_query_timings = timings
            return []

    def get_tasks_by_activity(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        activity_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> list:
        """
        Query tasks based on activity_log timestamps.

        Args:
            since: Start date for activity (inclusive)
            until: End date for activity (exclusive)
            activity_type: Type of activity (created, started, completed, note_added, updated)
            status: Current status filter (todo, in_progress, done)
            limit: Maximum results

        Returns:
            List of task dicts matching the activity criteria
        """
        import time

        logger.info(f"get_tasks_by_activity: since={since}, until={until}, activity_type={activity_type}, status={status}")

        # Track timings for debug panel
        timings = {}

        # Build query
        query = {}

        if status:
            query["status"] = status

        # Build activity_log query with $elemMatch
        if since or until or activity_type:
            elem_match = {}

            if activity_type:
                elem_match["action"] = activity_type

            if since or until:
                timestamp_query = {}
                if since:
                    timestamp_query["$gte"] = since
                if until:
                    timestamp_query["$lt"] = until
                elem_match["timestamp"] = timestamp_query

            query["activity_log"] = {"$elemMatch": elem_match}

        logger.debug(f"Activity query: {query}")

        try:
            # Time MongoDB query execution
            start = time.time()
            tasks_collection = get_collection(TASKS_COLLECTION)
            results = list(
                tasks_collection.find(query, {"embedding": 0})
                .sort("activity_log.timestamp", -1)
                .limit(limit)
            )
            timings["mongodb_query"] = int((time.time() - start) * 1000)

            # Store timings for coordinator to access
            self.last_query_timings = timings

            logger.info(f"Activity query returned {len(results)} task(s) (db: {timings['mongodb_query']}ms)")

            return results
        except Exception as e:
            logger.error(f"Activity query failed: {e}", exc_info=True)
            self.last_query_timings = timings  # Store even on error
            return []

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
        system_prompt = """You are a helpful search and retrieval assistant.
You have access to tools for semantic search, date-based search, and progress tracking.
Use the appropriate tools to help users find tasks, projects, and track progress.

When searching:
- Use semantic search for finding tasks/projects by meaning or content
- Use date-based search for activity on specific dates
- Use incomplete search for finding stalled or unfinished work
- Use progress search for project status and statistics

Always use the tools to perform actual searches - don't just describe what you would do.
Provide clear, organized summaries of search results."""

        # Tool use loop
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call Claude with tools
            response = self.llm.generate_with_tools(
                messages=messages,
                tools=self.tools,
                system=system_prompt,
                max_tokens=4096
            )

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
retrieval_agent = RetrievalAgent(memory_manager=None)
