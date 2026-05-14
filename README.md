# Mr. Anderson — PM Memory Demo

A multi-agent project-management assistant for **Tom**, Senior PM at the
fictional **Northwind Robotics**. Built to show off five distinct memory
patterns in a single LangGraph + MongoDB Atlas stack with a Chainlit UI.

## Architecture

```
┌──────────────┐      ┌────────────────┐      ┌──────────────┐
│  Chainlit UI │────▶ │  Coordinator   │────▶ │  Retrieval   │  (read-only)
│ (Mr.Anderson)│      │  Sonnet 4.5    │      └──────────────┘
└──────────────┘      │                │      ┌──────────────┐
        ▲             │                │────▶ │   Writer     │  (writes)
        │             └────────┬───────┘      └──────────────┘
        │                      │                      │
        │                      ▼                      ▼
        │             ┌──────────────────────────────────────┐
        └──────────── │  MemoryManager  + MemoryTrace        │
                      └──────────────────────────────────────┘
                                       │
                            ┌──────────┴──────────┐
                            ▼                     ▼
                     MongoDB Atlas          Voyage AI
                     (5 memory cols +       (voyage-4-large
                      Jira/calendar         embeddings)
                      mocks + LangGraph
                      checkpoints)
```

| Memory | Lifetime | Collection | Used for |
|---|---|---|---|
| 🟢 Working | 24h TTL | `working_memory` | what Tom is focused on right now |
| 📜 Episodic | long, vector | `episodic_memory` | timeline of every event |
| 🧠 Semantic | long, vector | `semantic_memory` | facts: people, projects, decisions, risks, epics, features, stories |
| 📘 Procedural | long, vector | `procedural_memory` | reusable workflows (standup, status, sprint planning) |
| 🔗 Shared | session: 1h TTL · project: no TTL | `shared_memory` | inter-agent handoff slots + long-lived project goals |

Every read/write goes through `MemoryManager`, which emits a `MemoryEvent` to
`MemoryTrace`. The Chainlit UI subscribes to that bus per turn and renders
each op as a line under a collapsible **🧠 Memory ops** step.

### Shared memory: how the agents coordinate

All three agents (Coordinator, Retrieval, Writer) read and write the
`shared_memory` collection through `MemoryManager`. Each doc is keyed by
`(scope, slot)` plus either `session_id` (short-term) or `project_key`
(long-term). Slots are typed:

| Slot | Default mode | Typical producer → consumer |
|---|---|---|
| `plan` | replace | Coordinator → sub-agents (the multi-step plan for this turn) |
| `findings` | append | Retrieval → Writer (structured rows: ticket keys, owners, IDs) |
| `disambiguation` | replace | Retrieval → Coordinator (multiple candidates, pick one) |
| `handoff_payload` | replace | Writer → Coordinator (summary of writes for the user) |
| `last_search_results` / `scratch` | replace | free-form |
| `goal` | replace | Coordinator (project-scoped, no TTL) |

The point is to avoid paraphrase loss between agents: Retrieval posts the
exact rows it found into `findings`, the Writer reads them verbatim, and
the Coordinator stops having to re-state ticket IDs and assignees in
prose between sub-agent calls. Project-scoped `goal` docs persist across
sessions so every later turn can consult them via
`read_shared_memory(project_key=...)`.

## Quickstart

### 1. Prerequisites

- Python 3.13
- A MongoDB Atlas cluster (free tier works) — note the connection string
- A Voyage AI API key (`voyage-4-large`)
- A Grove proxy API key for Anthropic models (or swap `grove_llm.py` for plain
  `langchain-anthropic` if you have direct API access)

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# MongoDB Atlas
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
PM_DB_NAME=pm_agent_memory          # optional, default shown

# Voyage AI (embeddings)
VOYAGE_API_KEY=...

# Anthropic (default provider; streams tokens to the UI)
ANTHROPIC_API_KEY=sk-ant-...
# Optional: route through the Grove proxy instead (no streaming).
# LLM_PROVIDER=grove
# GROVE_API_KEY=...
COORDINATOR_MODEL=claude-sonnet-4-5  # optional
RETRIEVAL_MODEL=claude-haiku-4-5     # optional
WRITER_MODEL=claude-haiku-4-5        # optional

# Chainlit auth (single-user)
CHAINLIT_USERNAME=admin
CHAINLIT_PASSWORD=admin
CHAINLIT_AUTH_SECRET=$(chainlit create-secret)
```

### 4. Seed the demo data

```bash
python -m memory.reset    # wipes + re-seeds Northwind Robotics
```

This bootstraps Atlas vector indexes, then loads:

- One persona (Tom)
- Three projects (Atlas Migration, Mobile App v2, Reporting Revamp) with
  tickets, epics, features, and user stories
- Three procedural workflows (daily standup, status report, sprint planning)
- A handful of seed decisions, risks, and calendar events

### 5. Launch the UI

```bash
chainlit run chainlit_ui/app.py
```

Open <http://localhost:8000>, log in with the credentials from `.env`, and
follow the demo script in the welcome panel.

## The 8-step demo script

1. **Recall** — _"Who owns the Atlas Migration and what's the latest risk?"_
2. **What's in flight** — _"What's in flight on Atlas?"_
3. **Working memory** — _"I'm prepping the Atlas dry-run rehearsal today — main goal is to hit Q2 GA without surprises. Also jot down: confirm Priya is on the bridge call."_
4. **Action + episodic** — _"Create a Jira ticket on PROJ-ATLAS to rehearse the runbook in staging, high, assign Priya."_
5. **Decision** — _"Log a decision: freeze billing-service deploys 24h before cutover."_
6. **Procedural ritual** — _"Run the daily standup."_
7. **Cross-turn recall** — _"What did I just do?"_
8. **Inspect & reset** — `/memory all`, then `/reset`

## Slash commands

| Command | What it does |
|---|---|
| `/help` | List commands |
| `/roadmap` | Projects, open tickets, upcoming meetings, recent activity |
| `/gantt [project] [days]` | Render an interactive Plotly Gantt of tickets, meetings and milestones (e.g. `/gantt PROJ-ATLAS 60`) |
| `/memory <type>` | Inspect: `working` / `episodic` / `semantic` / `procedural` / `shared` / `jira` / `calendar` / `counts` / `all` |
| `/reset` | Wipe everything (memory + business data + LangGraph checkpoints) and re-seed (asks for confirmation first) |

## Project layout

```
agents/
  coordinator.py        Sonnet supervisor + handoff tools
  retrieval.py          Haiku read-only sub-agent
  writer.py             Haiku write/action sub-agent
  prompts.py            All system prompts in one place
  *_tools.py            Per-agent tool inventories
  runtime.py            Process-wide MemoryManager singleton
memory/
  schemas.py            Pydantic models for the five memory types
  manager.py            All reads/writes funnel through here, emit trace events
  events.py             MemoryEvent + MemoryTrace pub/sub
  db.py                 Collection names + Mongo client wiring
  embeddings.py         Voyage AI wrappers
  search.py             $vectorSearch helper
  bootstrap.py          Index creation (vector + TTL + standard)
  seed_data.py          Northwind dataset (persona, projects, workflows, ...)
  seed.py               Loader
  reset.py              Wipe + re-seed entry point (also called by /reset)
chainlit_ui/
  app.py                Main Chainlit app
  commands.py           Slash command handlers
  data_layer.py         MongoDB-backed thread/message persistence
chainlit.md             Welcome screen + demo script (lives at project root —
                        Chainlit reads it from the current working directory)
```

## Resetting between dry-runs

Either click the `/reset` confirmation in the UI or run from the shell:

```bash
python -m memory.reset                  # wipe + re-seed
python -m memory.reset --no-seed        # wipe only
python -m memory.reset --keep-checkpoints
```
