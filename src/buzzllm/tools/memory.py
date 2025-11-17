from pathlib import Path
from typing import Optional

import os
from agentic_memory.memory_system import AgenticMemorySystem


_memory_system: Optional[AgenticMemorySystem] = None


def _get_memory_system() -> AgenticMemorySystem:
  global _memory_system
  if _memory_system is None:
    os.environ.setdefault("CHROMA_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    _memory_system = AgenticMemorySystem(
      model_name="text-embedding-3-small",
      llm_backend="openai",
      llm_model="gpt-5.1",
      evo_threshold=100,
      persist_directory=str(Path.home() /".buzzllm" / "agentic_memory_db"),
      collection_name="chat_memories",
    )
  return _memory_system


def add_memory(content: str, tags: list[str] | None = None, category: str = "Chat"):
  """
  Store a concise long-term memory about the user or ongoing work.

  Call this only when the user explicitly asks you to remember or store something,
  e.g.:
    - "Remember this for later:"
    - "Store this in memory"
    - "This is my style, and this is how we'll code from now on"
    - "Please remember that my favorite language is Python"

  Guidelines:
    - Make `content` a short, self-contained fact that will be useful later.
    - Do not store secrets (passwords, API keys, highly sensitive personal data).
    - Avoid transient chatter or information that is clearly short-lived.

  Args:
    content:
      Concise statement to remember, e.g.
      "User's favorite programming language is Python."
    tags:
      Optional labels for organization, e.g. ["preference", "python"] or
      ["project", "workflow"].
    category:
      Broad bucket for the memory, e.g. "Chat", "Project", "Research".

  Returns:
    dict with:
      memory_id: Unique ID of the stored memory.
      status: "stored" if successfully stored.

  Examples:
    add_memory(
      content="User's favorite programming language is Python.",
      tags=["preference", "programming", "python"],
      category="Chat",
    )

    add_memory(
      content="User wants all code examples in TypeScript with strict types.",
      tags=["preference", "coding-style", "typescript"],
      category="Chat",
    )
  """
  mem = _get_memory_system()
  mem_id = mem.add_note(content=content, tags=tags or [], category=category)
  return {"memory_id": mem_id, "status": "stored"}


def search_memory(query: str, k: int = 5):
  """
  Search previously stored long-term memories relevant to the current request.

  Use this when:
    - The user refers to things they asked you to remember before.
      e.g. "Use the style I told you earlier", "What did I say my preferences were?"
    - You need to recall facts the user explicitly requested to store.

  Guidelines:
    - Use a concise paraphrase of what you are looking for as `query`.
    - Call at most once per user turn.
    - If no results are found, just answer based on the current context.

  Args:
    query:
      Natural language search string, e.g.
        "user coding style preferences"
        "user favorite programming language"
        "project notes about personal finance dashboard"
    k:
      Maximum number of results to return. 1 <= k <= 20. Default: 5.

  Returns:
    List of memory objects with:
      id: Memory ID.
      content: Stored memory text.
      tags: List of tags.
      context: Optional short context string.
      category: Category label.

  Example:
    results = search_memory("user favorite programming language", k=3)
    # results might look like:
    # [
    #   {
    #     "id": "c0b9a9e4-6c0e-47d5-8e3d-1a9ac2a1b837",
    #     "content": "User's favorite programming language is Python.",
    #     "tags": ["preference", "programming", "python"],
    #     "context": "User programming preferences",
    #     "category": "Chat",
    #   }
    # ]
  """
  mem = _get_memory_system()
  results = mem.search_agentic(query, k=k)
  return [
    {
      "id": r["id"],
      "content": r["content"],
      "tags": r.get("tags", []),
      "context": r.get("context", ""),
      "category": r.get("category", ""),
    }
    for r in results
  ]
