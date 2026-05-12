"""
Interactive REPL for the MongoDB-backed memory agent.

Usage:
    python main.py
    python main.py --user alice --thread morning-chat
"""

import argparse
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from agent import create_memory_agent


def _iter_text(chunk) -> list[str]:
    """Extract text fragments from a streamed message chunk.

    Anthropic returns content as a list of typed blocks; other providers may
    return a plain string. Tool-call blocks are skipped.
    """
    content = getattr(chunk, "content", None)
    if not content:
        return []
    if isinstance(content, str):
        return [content]
    pieces: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                pieces.append(text)
    return pieces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the memory agent.")
    parser.add_argument(
        "--mongodb-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB connection URI (env: MONGODB_URI).",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("USER_ID", "default_user"),
        help="User identifier for memory namespacing (env: USER_ID).",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Conversation thread id. Defaults to a fresh uuid.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier passed to create_memory_agent (overrides env `model`).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thread_id = args.thread or f"thread-{uuid.uuid4().hex[:8]}"

    print(f"Starting agent (user={args.user}, thread={thread_id}).")
    print("Type 'exit' or 'quit' to end the session.\n")

    agent = create_memory_agent(args.mongodb_uri, model=args.model)
    config = {"configurable": {"thread_id": thread_id, "user_id": args.user}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        print("Agent: ", end="", flush=True)
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            for piece in _iter_text(chunk):
                print(piece, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
