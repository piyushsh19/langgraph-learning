"""
LangGraph example: read existing long-term memory from a Store and use it
to personalize a chatbot response.

This example uses:
- InMemoryStore: stores memory records for the lifetime of this process
- user_id in graph config: selects which user's memories to read
- store.search(namespace): retrieves records in that user's details namespace
- ChatOpenAI: generates the response using the retrieved memory

Requires OPENAI_API_KEY in your environment or .env file.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()


# -------------------------------------------------------------------
# 1. Create the long-term memory store and seed example memories
#    Do this before invoking the graph.
# -------------------------------------------------------------------
store = InMemoryStore()

user_id = "u1"
user_details_namespace = ("user", user_id, "details")

# In this teaching example, memories are stored as separate records.
store.put(
    user_details_namespace,
    "profile_1",
    {"data": "The user is learning AI engineering and LangGraph."},
)
store.put(
    user_details_namespace,
    "profile_2",
    {"data": "The user has full-stack development experience."},
)
store.put(
    user_details_namespace,
    "preference_1",
    {"data": "The user prefers step-by-step explanations."},
)
store.put(
    user_details_namespace,
    "preference_2",
    {"data": "The user likes practical Python examples and real-world analogies."},
)
store.put(
    user_details_namespace,
    "project_1",
    {"data": "The user is building a LangGraph learning repository."},
)


# -------------------------------------------------------------------
# 2. Define the system prompt template
# -------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.

Use the user's existing memory below when it is relevant to the current question.
Personalize your answer based on known preferences and context, but do not invent
facts or force irrelevant memories into the response.

If no memory is available, answer normally.

Existing user memory:
{user_details_content}
"""


# -------------------------------------------------------------------
# 3. Define the graph node
# -------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chat_node(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    # Read the user ID supplied at invocation time.
    current_user_id = config["configurable"]["user_id"]

    # Namespace is the address where this user's memories are stored.
    user_details = ("user", current_user_id, "details")

    # Read existing records. This is read-only: no memories are written here.
    items = store.search(user_details)

    # Convert the memory records into text for the system prompt.
    if items:
        user_details_content = "\n".join(
            f"- {item.value.get('data', '')}"
            for item in items
            if item.value.get("data")
        )
    else:
        user_details_content = "(No saved memories found.)"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_details_content=user_details_content
    )
    system_message = SystemMessage(content=system_prompt)

    response = llm.invoke([system_message, *state["messages"]])

    # MessagesState appends this response to the existing message list.
    return {"messages": [response]}


# -------------------------------------------------------------------
# 4. Build and compile the graph with the store
# -------------------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
