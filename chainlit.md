# Mr. Anderson — PM Memory Demo

A multi-agent project-management assistant for **Tom**, Senior PM at **Leafy Technologies**.
Three agents (Coordinator + Retrieval + Writer) over five MongoDB Atlas-backed memory types.

## Memory at a glance

| Type | Lifetime | Stored as | Used for |
|---|---|---|---|
| 🟢 Working | session (24h TTL) | `working_memory` | what Tom is focused on right now |
| 📜 Episodic | long | `episodic_memory` (vector) | timeline of every event |
| 🧠 Semantic | long | `semantic_memory` (vector) | facts: people, projects, decisions, risks, epics, features, stories |
| 📘 Procedural | long | `procedural_memory` (vector) | reusable workflows (standup, status, sprint) |
| 🔗 Shared | 1h TTL | `shared_memory` | inter-agent handoff slots |

Every memory op fires into a live trace — watch the **🧠 Memory ops** chip after each turn.

## Demo script (8 beats)

1. **Recall** — _"Who owns the Atlas Migration and what's the latest risk?"_  → semantic search by person + risk.
2. **What's in flight** — _"What's in flight on Atlas?"_  → direct `list_in_flight_tasks` tool, no LLM round-trips for the lookup.
3. **Working memory** — _"I'm prepping the Atlas dry-run rehearsal today — main goal is to hit Q2 GA without surprises. Also jot down: confirm Priya is on the bridge call."_  → populates `current_project`, `current_task`, `focus`, **and** appends to the `scratchpad` in one turn (`update_focus` + `note_to_self`). Verify with `/memory working`.
4. **Action + episodic** — _"Create a Jira ticket on PROJ-ATLAS to rehearse the runbook in staging, high, assign Priya."_  → ticket + episodic event in one shot.
5. **Decision** — _"Log a decision: freeze billing-service deploys 24h before cutover."_  → semantic + episodic in one call.
6. **Procedural ritual** — _"Run the daily standup."_  → fetches the workflow from procedural memory and executes the steps.
7. **Cross-turn recall** — _"What did I just do?"_  → checkpointer-backed conversation memory plus episodic recall.
8. **Inspect & reset** — `/memory all` to peek at every store, then `/reset` to wipe and re-seed for the next run.

## Slash commands

- `/help` — list commands
- `/roadmap` — projects, open tickets, upcoming meetings, recent activity
- `/memory <type>` — inspect working / episodic / semantic / procedural / shared / jira / calendar / counts / all
- `/reset` — wipe everything and re-seed the Leafy Technologies dataset (asks first)
