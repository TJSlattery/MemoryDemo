"""Leafy Technologies demo dataset.

Pure data, no imports from `memory.*`. Loaded by `memory.seed`.

Persona: Tom, Senior PM at Leafy Technologies, owns three
in-flight initiatives (Atlas Migration, Mobile App v2, Reporting Revamp).
"""

from __future__ import annotations

from datetime import datetime, timedelta

NOW = datetime.utcnow()

PERSONA = {
    "user_id": "admin",
    "display_name": "Tom",
    "title": "Senior Project Manager",
    "company": "Leafy Technologies",
}

# ── Semantic facts ────────────────────────────────────────────────────────
PEOPLE = [
    ("jane.doe", "Jane Doe", "Engineering Manager, Platform team. Owns the Atlas Migration project. Prefers async standups."),
    ("marcus.chen", "Marcus Chen", "Tech Lead, Mobile team. Drives Mobile App v2. Strong opinions on Flutter."),
    ("priya.patel", "Priya Patel", "Senior Engineer on the Atlas Migration. Subject-matter expert on billing service internals."),
    ("sam.okafor", "Sam Okafor", "Engineer on the Reporting Revamp. New hire, joined six weeks ago."),
    ("lisa.wong", "Lisa Wong", "Product Designer supporting Mobile App v2. On PTO June 10–17."),
    ("diego.ramirez", "Diego Ramirez", "QA Lead across all three projects. Owns release sign-off."),
]

STAKEHOLDERS = [
    ("karen.liu", "Karen Liu", "VP Product. Primary sponsor for the Reporting Revamp. Wants weekly status by Friday EOD."),
    ("will.bridges", "Will Bridges", "CTO. Sponsors the Atlas Migration. Reads decision logs but skips standups."),
]

PROJECTS = [
    ("PROJ-ATLAS", "Atlas Migration", "Move the billing service from self-hosted MongoDB to MongoDB Atlas. Target GA: end of Q2. Owner: Jane Doe. Status: in progress."),
    ("PROJ-MOBILE", "Mobile App v2", "Rewrite the customer-facing mobile app on Flutter. Target beta: end of Q3. Owner: Marcus Chen. Status: in progress."),
    ("PROJ-REPORT", "Reporting Revamp", "Replace the legacy nightly batch reports with a real-time analytics dashboard. Target GA: Q4. Owner: Sam Okafor. Status: planning."),
]

GLOSSARY = [
    ("leafy_technologies", "Leafy Technologies", "Mid-sized technology company. ~400 employees. Tom's employer."),
    ("prj_prefix", "PRJ-", "Jira ticket prefix used across all Leafy engineering projects."),
    ("billing_service", "billing service", "Internal Python service that issues invoices and tracks subscription usage. Currently runs on a self-hosted MongoDB replica set."),
    ("flutter_decision", "Flutter framework", "Cross-platform UI framework adopted for Mobile App v2 in place of React Native."),
]

DECISIONS = [
    ("dec_atlas_over_selfhosted", "Chose Atlas over self-managed MongoDB", "After a 3-week evaluation, the platform team chose MongoDB Atlas over continuing to self-host. Driver: ops cost and on-call burden. Approved by Will Bridges on April 12."),
    ("dec_flutter_over_rn", "Mobile App v2 will use Flutter, not React Native", "Marcus's team prototyped both. Flutter won on tooling, hot-reload speed, and shared rendering across iOS/Android. Decision logged April 28."),
]

RISKS = [
    ("risk_stripe_webhook", "Atlas Migration blocked on Stripe webhook update", "Stripe webhook signing keys must be rotated when the billing service moves clusters. Stripe support ticket #84321 open. Owner: Priya Patel."),
    ("risk_lisa_pto", "Designer PTO during Mobile beta cutover", "Lisa Wong is out June 10–17, overlapping the planned beta cutover for Mobile App v2. Mitigation: pre-approve all v2 screens by June 9."),
]

PREFERENCES = [
    ("user_name", "Tom", "The PM goes by Tom. Address him by first name."),
    ("standup_format", "async via Slack thread", "Tom prefers async standups in Slack #pm-standup. Synchronous only on Mondays."),
    ("status_report_day", "Friday", "Weekly status reports go out Fridays before EOD to Karen Liu."),
]

# Epics group features; features group stories. All tied to a project key.
EPICS = [
    ("epic_atlas_data_parity", "Atlas data parity", "PROJ-ATLAS", "Achieve full read/write parity between the legacy MongoDB cluster and Atlas before cutover. Owner: Jane Doe."),
    ("epic_mobile_v2_beta", "Mobile App v2 beta launch", "PROJ-MOBILE", "Ship Mobile App v2 to a closed beta of 500 users by end of Q3. Owner: Marcus Chen."),
    ("epic_realtime_reporting", "Real-time reporting pipeline", "PROJ-REPORT", "Replace nightly batch with sub-minute analytics for the leadership dashboard. Owner: Sam Okafor."),
]

FEATURES = [
    ("feat_dual_write", "Dual-write billing to Atlas + legacy", "PROJ-ATLAS", "epic_atlas_data_parity", "Every billing write hits both clusters; reads stay on legacy until parity tests pass."),
    ("feat_read_shadowing", "Read shadowing & parity tests", "PROJ-ATLAS", "epic_atlas_data_parity", "Mirror production reads to Atlas and report any divergence in a daily report."),
    ("feat_v2_auth", "Mobile App v2 auth flow", "PROJ-MOBILE", "epic_mobile_v2_beta", "Replace legacy session cookies with OAuth + biometric unlock on iOS/Android."),
    ("feat_v2_nav_shell", "Mobile v2 navigation shell", "PROJ-MOBILE", "epic_mobile_v2_beta", "Flutter-based bottom-nav shell with deep-link support and offline cache."),
    ("feat_streaming_ingest", "Streaming ingest from Kafka", "PROJ-REPORT", "epic_realtime_reporting", "Replace nightly S3 dump with Kafka topic consumed into the analytics warehouse."),
]

STORIES = [
    ("story_billing_dual_write", "As a billing engineer, I want every invoice write to go to both clusters so that we can detect drift before cutover.", "PROJ-ATLAS", "feat_dual_write", ["Given a new invoice, when it's written, then both the legacy and Atlas clusters reflect it within 1s.", "Given an Atlas write fails, when the legacy write succeeds, then the failure is logged and surfaced in the daily parity report."]),
    ("story_v2_biometric_unlock", "As a returning customer, I want to unlock the app with Face ID or fingerprint so that I don't have to re-enter my password.", "PROJ-MOBILE", "feat_v2_auth", ["Given a customer has previously logged in, when they reopen the app, then they are prompted for biometric unlock.", "Given biometric unlock fails 3 times, then the app falls back to password entry."]),
    ("story_realtime_dashboard", "As Karen (VP Product), I want yesterday's revenue numbers visible by 7am so that I can review them before the exec sync.", "PROJ-REPORT", "feat_streaming_ingest", ["Given last night's transactions, when I open the dashboard at 7am, then the revenue total reflects all transactions through 11:59pm.", "Given the streaming ingest is delayed, then the dashboard shows a 'data lag' banner with the staleness in minutes."]),
]

# ── Procedural workflows ──────────────────────────────────────────────────
PROCEDURES = [
    {
        "name": "daily_standup",
        "description": "Run a 15-minute async standup across the three active projects.",
        "trigger_examples": ["run standup", "do the standup", "kick off standup"],
        "tags": ["ritual", "daily"],
        "steps": [
            (1, "Post the standup template to #pm-standup"),
            (2, "Tag each project lead (Jane, Marcus, Sam) with the three prompts: yesterday / today / blockers"),
            (3, "Wait 90 minutes, then summarise blockers in a thread reply"),
            (4, "Log a `standup` episodic memory with the blocker summary"),
        ],
    },
    {
        "name": "weekly_status_report",
        "description": "Compose the Friday status report for Karen Liu (VP Product).",
        "trigger_examples": ["weekly status", "Friday status report", "send the status update"],
        "tags": ["ritual", "weekly"],
        "steps": [
            (1, "Pull the last 7 days of episodic events for each active project"),
            (2, "Summarise: shipped, in progress, blocked, decisions, risks"),
            (3, "Format as the standard 5-section markdown template"),
            (4, "Send to Karen via email + cross-post to #leadership"),
            (5, "Log a `status_report` episodic memory"),
        ],
    },
    {
        "name": "sprint_planning",
        "description": "Two-week sprint planning ritual for any project.",
        "trigger_examples": ["plan the next sprint", "sprint planning", "kick off the sprint"],
        "tags": ["ritual", "sprint"],
        "steps": [
            (1, "Confirm the sprint goal with the project lead"),
            (2, "Pull the top of the backlog and size with the team"),
            (3, "Commit to a sprint scope; create Jira tickets for new work"),
            (4, "Schedule the mid-sprint check-in calendar invite"),
            (5, "Log a `sprint_planned` episodic memory with goal + scope"),
        ],
    },
    {
        "name": "create_user_story",
        "description": "Standard template for a well-formed user story.",
        "trigger_examples": ["write a user story", "draft a story", "story for"],
        "tags": ["template", "story"],
        "steps": [
            (1, "Title in the form: As a <persona>, I want <capability>, so that <benefit>"),
            (2, "List 3–5 acceptance criteria in Given/When/Then format"),
            (3, "Note dependencies and out-of-scope items"),
            (4, "Add a rough size (S/M/L) and target sprint"),
        ],
    },
    {
        "name": "risk_assessment",
        "description": "Lightweight risk log entry for any newly surfaced risk.",
        "trigger_examples": ["log a risk", "assess this risk", "new risk"],
        "tags": ["risk"],
        "steps": [
            (1, "Capture: description, project, likelihood (L/M/H), impact (L/M/H)"),
            (2, "Identify owner and mitigation plan"),
            (3, "Set a review date (default: 2 weeks out)"),
            (4, "Log a `risk_logged` episodic memory and persist as semantic risk"),
        ],
    },
]

# ── Episodic history ──────────────────────────────────────────────────────
EPISODES = [
    (NOW - timedelta(days=21), "ticket_created", "Created PRJ-401: stand up Atlas cluster in us-east-1", {"ticket": "PRJ-401", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=20), "ticket_created", "Created PRJ-402: dual-write billing service to old + Atlas", {"ticket": "PRJ-402", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=18), "decision_logged", "Decision: chose Atlas over self-managed MongoDB for billing", {"decision": "dec_atlas_over_selfhosted"}),
    (NOW - timedelta(days=16), "sprint_planned", "Sprint 12 planned for Atlas Migration: dual-write + read shadowing", {"project": "PROJ-ATLAS", "sprint": 12}),
    (NOW - timedelta(days=14), "ticket_created", "Created PRJ-501: Flutter spike for Mobile App v2 navigation shell", {"ticket": "PRJ-501", "project": "PROJ-MOBILE"}),
    (NOW - timedelta(days=12), "decision_logged", "Decision: Mobile App v2 will use Flutter, not React Native", {"decision": "dec_flutter_over_rn"}),
    (NOW - timedelta(days=11), "risk_logged", "Risk: Stripe webhook signing keys must rotate during Atlas cutover", {"risk": "risk_stripe_webhook"}),
    (NOW - timedelta(days=10), "status_report", "Sent weekly status to Karen: Atlas dual-write live, Flutter spike green, Reporting still scoping", {}),
    (NOW - timedelta(days=8), "ticket_created", "Created PRJ-601: scope discovery for real-time reporting pipeline", {"ticket": "PRJ-601", "project": "PROJ-REPORT"}),
    (NOW - timedelta(days=7), "standup", "Standup blockers: Priya waiting on Stripe support, Marcus on iOS provisioning", {}),
    (NOW - timedelta(days=6), "sprint_planned", "Sprint 6 planned for Mobile App v2: navigation shell + auth flow", {"project": "PROJ-MOBILE", "sprint": 6}),
    (NOW - timedelta(days=5), "ticket_created", "Created PRJ-403: cutover runbook for billing service to Atlas", {"ticket": "PRJ-403", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=4), "risk_logged", "Risk: Lisa Wong PTO June 10–17 overlaps Mobile beta cutover", {"risk": "risk_lisa_pto"}),
    (NOW - timedelta(days=3), "status_report", "Sent weekly status to Karen: Atlas on track, Mobile on track, Reporting kickoff scheduled", {}),
    (NOW - timedelta(days=2), "ticket_created", "Created PRJ-602: hire frontend contractor for reporting dashboard", {"ticket": "PRJ-602", "project": "PROJ-REPORT"}),
    (NOW - timedelta(days=19), "epic_created", "Epic created: Atlas data parity", {"epic": "epic_atlas_data_parity", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=15), "feature_created", "Feature created: Dual-write billing to Atlas + legacy", {"feature": "feat_dual_write", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=13), "story_created", "Story created: dual-write parity for billing engineer", {"story": "story_billing_dual_write", "project": "PROJ-ATLAS"}),
    (NOW - timedelta(days=9), "feature_created", "Feature created: Mobile App v2 auth flow", {"feature": "feat_v2_auth", "project": "PROJ-MOBILE"}),
    (NOW - timedelta(days=9), "story_created", "Story created: biometric unlock for returning customer", {"story": "story_v2_biometric_unlock", "project": "PROJ-MOBILE"}),
]

# ── Mock business state (Jira / calendar / milestones) ───────────────────
# Each ticket carries `start_date` and `due_date` so the Gantt tool can plot
# them without falling back to heuristics. Dates fan out around `NOW` to give
# a chart with both completed, in-flight, and upcoming work.
JIRA_TICKETS = [
    # PROJ-ATLAS — Q2 cutover effort
    {"key": "PRJ-402", "title": "Dual-write billing service to old + Atlas", "project": "PROJ-ATLAS", "status": "In Progress", "assignee": "priya.patel", "priority": "High", "created_at": NOW - timedelta(days=20), "start_date": NOW - timedelta(days=20), "due_date": NOW + timedelta(days=17), "linked_risks": ["risk_stripe_webhook"]},
    {"key": "PRJ-403", "title": "Cutover runbook for billing service to Atlas", "project": "PROJ-ATLAS", "status": "In Review", "assignee": "jane.doe", "priority": "High", "created_at": NOW - timedelta(days=5), "start_date": NOW - timedelta(days=5), "due_date": NOW + timedelta(days=14)},
    {"key": "PRJ-404", "title": "Rotate Stripe webhook signing keys", "project": "PROJ-ATLAS", "status": "To Do", "assignee": "priya.patel", "priority": "High", "created_at": NOW - timedelta(days=3), "start_date": NOW + timedelta(days=5), "due_date": NOW + timedelta(days=18), "linked_risks": ["risk_stripe_webhook"]},
    {"key": "PRJ-405", "title": "Parity test harness for billing reads", "project": "PROJ-ATLAS", "status": "In Progress", "assignee": "diego.ramirez", "priority": "Medium", "created_at": NOW - timedelta(days=10), "start_date": NOW - timedelta(days=10), "due_date": NOW + timedelta(days=12)},
    {"key": "PRJ-406", "title": "Cutover comms plan for billing customers", "project": "PROJ-ATLAS", "status": "To Do", "assignee": "tom", "priority": "Medium", "created_at": NOW - timedelta(days=1), "start_date": NOW + timedelta(days=7), "due_date": NOW + timedelta(days=20)},
    {"key": "PRJ-407", "title": "Decommission legacy billing replica set", "project": "PROJ-ATLAS", "status": "To Do", "assignee": "jane.doe", "priority": "Low", "created_at": NOW - timedelta(days=1), "start_date": NOW + timedelta(days=25), "due_date": NOW + timedelta(days=60)},
    # PROJ-MOBILE — v2 beta
    {"key": "PRJ-501", "title": "Flutter spike: navigation shell", "project": "PROJ-MOBILE", "status": "Done", "assignee": "marcus.chen", "priority": "Medium", "created_at": NOW - timedelta(days=14), "start_date": NOW - timedelta(days=14), "due_date": NOW - timedelta(days=8)},
    {"key": "PRJ-502", "title": "Mobile App v2 auth flow", "project": "PROJ-MOBILE", "status": "In Progress", "assignee": "marcus.chen", "priority": "High", "created_at": NOW - timedelta(days=6), "start_date": NOW - timedelta(days=6), "due_date": NOW + timedelta(days=30)},
    {"key": "PRJ-503", "title": "Biometric unlock (Face ID + fingerprint)", "project": "PROJ-MOBILE", "status": "In Progress", "assignee": "marcus.chen", "priority": "High", "created_at": NOW - timedelta(days=2), "start_date": NOW - timedelta(days=2), "due_date": NOW + timedelta(days=35), "linked_risks": ["risk_lisa_pto"]},
    {"key": "PRJ-504", "title": "Beta TestFlight + Play Internal setup", "project": "PROJ-MOBILE", "status": "To Do", "assignee": "marcus.chen", "priority": "Medium", "created_at": NOW - timedelta(days=1), "start_date": NOW + timedelta(days=28), "due_date": NOW + timedelta(days=50)},
    # PROJ-REPORT — real-time reporting
    {"key": "PRJ-601", "title": "Scope discovery for real-time reporting pipeline", "project": "PROJ-REPORT", "status": "In Progress", "assignee": "sam.okafor", "priority": "Medium", "created_at": NOW - timedelta(days=8), "start_date": NOW - timedelta(days=8), "due_date": NOW + timedelta(days=14)},
    {"key": "PRJ-602", "title": "Hire frontend contractor for reporting dashboard", "project": "PROJ-REPORT", "status": "To Do", "assignee": "tom", "priority": "Medium", "created_at": NOW - timedelta(days=2), "start_date": NOW - timedelta(days=2), "due_date": NOW + timedelta(days=21)},
    {"key": "PRJ-603", "title": "Streaming ingest spike (Kafka → warehouse)", "project": "PROJ-REPORT", "status": "To Do", "assignee": "sam.okafor", "priority": "Medium", "created_at": NOW - timedelta(days=1), "start_date": NOW + timedelta(days=14), "due_date": NOW + timedelta(days=45)},
]

CALENDAR_EVENTS = [
    {"title": "Atlas Migration cutover dry-run", "attendees": ["jane.doe", "priya.patel", "diego.ramirez", "tom"], "start": NOW + timedelta(days=2, hours=10), "duration_minutes": 60, "project": "PROJ-ATLAS"},
    {"title": "Reporting Revamp kickoff with Karen", "attendees": ["karen.liu", "sam.okafor", "tom"], "start": NOW + timedelta(days=4, hours=14), "duration_minutes": 45, "project": "PROJ-REPORT"},
    {"title": "Mobile v2 design review with Lisa", "attendees": ["marcus.chen", "lisa.wong", "tom"], "start": NOW + timedelta(days=10, hours=15), "duration_minutes": 45, "project": "PROJ-MOBILE"},
    {"title": "Atlas pre-GA architecture review", "attendees": ["jane.doe", "priya.patel", "will.bridges", "tom"], "start": NOW + timedelta(days=14, hours=11), "duration_minutes": 60, "project": "PROJ-ATLAS"},
    {"title": "Reporting architecture review with Karen", "attendees": ["karen.liu", "sam.okafor", "tom"], "start": NOW + timedelta(days=20, hours=14), "duration_minutes": 45, "project": "PROJ-REPORT"},
    {"title": "Mobile v2 TestFlight kickoff", "attendees": ["marcus.chen", "diego.ramirez", "tom"], "start": NOW + timedelta(days=35, hours=10), "duration_minutes": 30, "project": "PROJ-MOBILE"},
]

# Single-point-in-time milestones overlaid on the Gantt as diamond markers.
MILESTONES = [
    {"key": "ms_atlas_freeze", "title": "Atlas: deploy freeze",   "project": "PROJ-ATLAS",  "date": NOW + timedelta(days=18)},
    {"key": "ms_atlas_ga",     "title": "Atlas: GA cutover",      "project": "PROJ-ATLAS",  "date": NOW + timedelta(days=22)},
    {"key": "ms_mobile_beta",  "title": "Mobile v2: beta launch", "project": "PROJ-MOBILE", "date": NOW + timedelta(days=55)},
    {"key": "ms_report_ga",    "title": "Reporting: GA",          "project": "PROJ-REPORT", "date": NOW + timedelta(days=82)},
]

# ── Per-session seeds (working + shared memory) ───────────────────────────
# Both stores are session-scoped, so these are written into the active
# Chainlit thread on chat start (see `memory.seed.seed_session`).
WORKING_SEED = {
    "current_project": "PROJ-ATLAS",
    "current_task": "Atlas dry-run rehearsal prep",
    "focus": "Confirm cutover runbook coverage and stakeholder readiness for Q2 GA.",
    "last_action": "Reviewed PRJ-403 cutover runbook draft with Jane.",
    "scratchpad": [
        "Ping Priya re: Stripe webhook ticket #84321 status.",
        "Block Lisa Wong's PTO conflict on the cutover calendar.",
    ],
}

SHARED_SEED = [
    {
        "slot": "last_search_results",
        "from_agent": "retrieval",
        "to_agent": "coordinator",
        "payload": {
            "query": "Atlas cutover blockers",
            "hits": [
                {"key": "risk_stripe_webhook", "type": "risk", "summary": "Stripe webhook signing keys must rotate during Atlas cutover."},
                {"key": "PRJ-403", "type": "ticket", "summary": "Cutover runbook for billing service to Atlas (In Review)."},
            ],
        },
    },
    {
        "slot": "handoff_payload",
        "from_agent": "retrieval",
        "to_agent": "writer",
        "payload": {
            "instruction": "Schedule a 30-min runbook walkthrough with Priya and Diego before the dry-run.",
            "context": {"project": "PROJ-ATLAS", "tickets": ["PRJ-403"]},
        },
    },
]

# Project-scoped shared memory: long-lived strategic goals visible to every
# session for the project (no TTL). Seeded once via `seed()`, not per-session.
PROJECT_SHARED_SEED = [
    {
        "slot": "goal",
        "scope": "project",
        "project_key": "PROJ-ATLAS",
        "from_agent": "coordinator",
        "to_agent": "any",
        "payload": {
            "title": "Atlas cutover with zero customer-visible downtime",
            "deadline": "end of Q2",
            "non_negotiables": [
                "No more than 5 minutes of write-path downtime during cutover.",
                "Stripe webhook signing keys must be rotated before cutover, not during.",
            ],
            "owner": "jane.doe",
        },
    },
]
