"""Reset the demo to a clean state.

Wipes:
  * every memory collection (working / episodic / semantic / procedural / shared)
  * every mock business collection (Jira / calendar / projects / tasks)
  * LangGraph checkpointer collections (so the agent forgets prior turns)

Then optionally re-seeds the Leafy Technologies dataset.

Usage:
    python -m memory.reset                # wipe + re-seed (default)
    python -m memory.reset --no-seed      # wipe only
    python -m memory.reset --keep-checkpoints
"""

from __future__ import annotations

import argparse
import logging

from memory.db import (
    ALL_BUSINESS_COLLECTIONS,
    ALL_MEMORY_COLLECTIONS,
    CHECKPOINT_COLLECTIONS,
    get_db,
)

logger = logging.getLogger(__name__)


def reset(
    seed_after: bool = True,
    keep_checkpoints: bool = False,
) -> dict[str, int]:
    """Wipe all demo collections. Returns per-collection deleted counts."""
    db = get_db()
    deleted: dict[str, int] = {}

    targets = list(ALL_MEMORY_COLLECTIONS) + list(ALL_BUSINESS_COLLECTIONS)
    if not keep_checkpoints:
        targets += list(CHECKPOINT_COLLECTIONS)

    existing = set(db.list_collection_names())
    for name in targets:
        if name not in existing:
            deleted[name] = 0
            continue
        result = db[name].delete_many({})
        deleted[name] = result.deleted_count

    if seed_after:
        # Imported lazily so `--no-seed` callers don't pay the import cost.
        from memory.seed import seed

        # Bootstrap is idempotent; running it on every reset means a
        # first-time user gets working vector indexes without an extra step.
        seed(do_bootstrap=True)

    return deleted


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Reset the PM agent demo.")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Wipe collections but skip re-seeding.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Leave LangGraph checkpointer collections intact.",
    )
    args = parser.parse_args()
    deleted = reset(seed_after=not args.no_seed, keep_checkpoints=args.keep_checkpoints)

    print("Cleared:")
    for k, v in deleted.items():
        print(f"  {k:<28} {v}")
    if not args.no_seed:
        print("Re-seeded Leafy Technologies dataset.")
