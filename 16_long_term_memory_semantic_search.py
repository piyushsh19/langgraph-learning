"""
Basic LangGraph long-term memory semantic search using InMemoryStore.
This version uses a LOCAL Hugging Face embedding model:
    sentence-transformers/all-MiniLM-L6-v2

It does not require OpenAI embeddings or an OpenAI API key.
The store is in memory, so memories disappear when the process exits.
"""

from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------------------------------------------------
# 1. Embedding model
# -------------------------------------------------------------------
# The model must already exist in the Hugging Face cache.
# local_files_only=True prevents an automatic download.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)


# -------------------------------------------------------------------
# 2. Create a semantic-searchable in-memory store
# -------------------------------------------------------------------
store = InMemoryStore(
    index={
        "embed": embedding_model,
        "dims": 384,  # all-MiniLM-L6-v2 produces 384-dimensional vectors
    }
)

# Namespace groups memories by user. Use a stable user ID in an app.
namespace = ("users", "u1")


# -------------------------------------------------------------------
# 3. Save long-term memories
# -------------------------------------------------------------------
# Each memory needs a unique key within the namespace.
memories = {
    "1": {"data": "User prefers concise answers over long explanations"},
    "2": {"data": "User likes examples in Python"},
    "3": {"data": "User usually works late at night"},
    "4": {"data": "User prefers dark mode in applications"},
    "5": {"data": "User is learning machine learning"},
    "6": {"data": "User dislikes overly theoretical explanations"},
    "7": {"data": "User prefers step-by-step reasoning"},
    "8": {"data": "User is based in India"},
    "9": {"data": "User likes real-world analogies"},
    "10": {"data": "User prefers bullet points over paragraphs"},
}

for memory_id, value in memories.items():
    store.put(namespace, memory_id, value)


# -------------------------------------------------------------------
# 4. Search memories by meaning (not just exact keywords)
# -------------------------------------------------------------------
def search_memories(query: str, limit: int = 3) -> None:
    """Print the closest semantic matches for a natural-language query."""
    results = store.search(namespace, query=query, limit=limit)

    print(f"\nQuery: {query!r}")
    if not results:
        print("No matching memories found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item.value} (key={item.key}, score={item.score})")


if __name__ == "__main__":
    search_memories("What is the user currently learning?", limit=1)
    search_memories("What are the user's response preferences?", limit=3)
    search_memories("Which examples should I use when explaining code?", limit=3)
