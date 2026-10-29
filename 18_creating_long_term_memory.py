"""
LangGraph example: extract stable facts from the latest user message and save
them to a long-term memory Store.

This graph CREATES memories. It does not retrieve memories to answer the user.
The node returns a fixed acknowledgement after saving any useful memories.

Requires OPENAI_API_KEY in your environment or .env file.
"""

from typing import List
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()


# -------------------------------------------------------------------
# 1. Define the structured decision returned by the memory extractor
# -------------------------------------------------------------------
class MemoryDecision(BaseModel):
    should_write: bool = Field(
        description="True only when the latest message contains useful long-term memory."
    )
    memories: List[str] = Field(
        default_factory=list,
        description="Short, atomic, stable facts about the user to remember.",
    )


# -------------------------------------------------------------------
# 2. Create an LLM that returns the structured MemoryDecision object
# -------------------------------------------------------------------
extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory_extractor = extractor_llm.with_structured_output(MemoryDecision)


# -------------------------------------------------------------------
# 3. Define the node: extract and save long-term memories
# -------------------------------------------------------------------
def remember_only_node(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    # This example assumes the latest message is the user's new message.
    last_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "Extract LONG-TERM memories from the user's latest message.\n"
                    "Only store stable, user-specific information, such as goals, "
                    "ongoing projects, background, or durable preferences.\n"
                    "Do NOT store temporary requests, one-time questions, or guesses.\n"
                    "If nothing is worth remembering, set should_write=false and "
                    "return an empty memories list.\n"
                    "Each memory must be one short, atomic sentence.\n"
                    "Do not infer sensitive personal attributes."
                )
            ),
            {"role": "user", "content": last_msg},
        ]
    )

    # Write to the long-term memory Store only when the extractor says to.
    saved_memories = []
    if decision.should_write:
        for memory in decision.memories:
            memory = memory.strip()
            if not memory:
                continue

            memory_id = str(uuid4())
            store.put(namespace, memory_id, {"data": memory})
            saved_memories.append(memory)

    # This is intentionally NOT a normal chatbot answer.
    # The node acknowledges the memory operation without using memories.
    if saved_memories:
        acknowledgement = f"Saved {len(saved_memories)} long-term memory item(s)."
    else:
        acknowledgement = "No long-term memory was saved."

    return {"messages": [{"role": "assistant", "content": acknowledgement}]}


# -------------------------------------------------------------------
# 4. Build and compile the graph with the store
# -------------------------------------------------------------------
store = InMemoryStore()

builder = StateGraph(MessagesState)
builder.add_node("remember", remember_only_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

graph = builder.compile(store=store)


# -------------------------------------------------------------------
# 5. Run the graph, then inspect the saved memories
# -------------------------------------------------------------------
if __name__ == "__main__":
    config = {"configurable": {"user_id": "u1"}}

    user_message = (
        "I am a full-stack developer learning LangGraph and AI engineering. "
        "I prefer practical Python examples with step-by-step explanations."
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )

    print("Assistant:", result["messages"][-1].content)

    # Inspect the user's saved long-term memories.
    namespace = ("user", "u1", "details")
    print("\nSaved memories:")
    for item in store.search(namespace):
        print("-", item.value.get("data", ""))
