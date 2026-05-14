"""
Simple Agent with MongoDB-backed Memory (Custom Tools Approach)
Uses custom @tool functions for direct control over memory operations.
Simpler and more transparent than the langmem approach.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.tools import tool
from langchain_voyageai import VoyageAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from langgraph.utils.config import get_config
from pymongo import MongoClient

from llm import create_chat_model
from memory.db import DB_NAME

# Initialize embedding model
embedding_model = VoyageAIEmbeddings(
    model="voyage-4-large"
)

# Vector search index configuration for memory collection
index_config = create_vector_index_config(
    embed=embedding_model,
    dims=1024,
    relevance_score_fn="dotProduct",
    fields=["content"]
)


def create_memory_tools(store: MongoDBStore):
    """Create custom memory tools using MongoDBStore with vector search."""

    @tool
    def save_memory(content: str) -> str:
        """Save important information to memory for the current user."""
        config = get_config()
        user_id = config.get("configurable", {}).get("user_id", "default_user")

        store.put(
            namespace=("user", user_id, "memories"),
            key=f"memory_{hash(content)}",
            value={"content": content}
        )

        return f"Memory saved: {content}"

    @tool
    def retrieve_memories(query: str) -> str:
        """Retrieve relevant memories based on a query for the current user."""
        config = get_config()
        user_id = config.get("configurable", {}).get("user_id", "default_user")

        namespace = ("user", user_id, "memories")
        results = store.search(namespace, query=query, limit=5)

        if results:
            memories = [result.value["content"] for result in results]
            return "Retrieved memories:\n" + "\n".join(f"- {mem}" for mem in memories)
        return "No relevant memories found."

    return [save_memory, retrieve_memories]


def create_memory_agent(mongodb_uri: str, model=None):
    """Create an agent with MongoDB-backed memory using custom tools.

    `model` may be a chat model instance or a string model identifier.
    String identifiers are routed through the configured LLM provider
    (see ``llm.create_chat_model``).
    """

    if model is None:
        model = os.getenv("model", "claude-haiku-4-5")
    if isinstance(model, str):
        model = create_chat_model(model)

    system_prompt = """You are a helpful AI assistant with memory capabilities.

When a user sends you a message:
1. First, check your memory about them using retrieve_memories
2. Use what you find to personalize your response
3. If they share new information, save it using save_memory

Example:
- User: "I have a peanut allergy — I need to be careful when eating out."
  -> Call save_memory("User has a peanut allergy and needs peanut-safe food options")
  -> Respond: "Got it, I've saved that. I'll always keep your peanut allergy in mind when suggesting places to eat."

- User: "Can you recommend somewhere for dinner tonight?"
  -> Call retrieve_memories(query="dietary restrictions or food preferences")
  -> If found: Recommend peanut-safe options based on saved preferences
  -> If not found: Ask about any preferences or dietary needs first

Your memory persists across conversations, so you can remember users over time. Be conversational and helpful!"""

    client = MongoClient(mongodb_uri)
    db = client[DB_NAME]
    collection = db["memories"]

    store = MongoDBStore(
        collection=collection,
        index_config=index_config,
        auto_index_timeout=120,
    )

    memory_tools = create_memory_tools(store)
    checkpointer = MongoDBSaver(client, db_name=DB_NAME)

    agent = create_agent(
        model,
        system_prompt=system_prompt,
        tools=memory_tools,
        checkpointer=checkpointer,
    )

    return agent