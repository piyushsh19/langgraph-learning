"""
15_stm_summarization.py

LangGraph short-term memory example: summarization + message trimming.
Uses local Qwen/Qwen2.5-3B-Instruct from the Hugging Face cache.
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Load the local model once. No hosted LLM or OpenAI API key is used.
pipeline_model = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    model_kwargs={"local_files_only": True},
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)
llm = ChatHuggingFace(llm=pipeline_model)


class ChatState(MessagesState):
    """Chat history plus a running summary of older conversation turns."""
    summary: str


def chat_node(state: ChatState) -> dict:
    """Generate a response using the summary and recent messages."""
    messages_for_model = []
    existing_summary = state.get("summary", "")

    if existing_summary:
        messages_for_model.append(
            SystemMessage(
                content=(
                    "Summary of earlier conversation. Use it as context, "
                    "but prioritize the user's latest message:\n"
                    f"{existing_summary}"
                )
            )
        )

    messages_for_model.extend(state["messages"])
    response = llm.invoke(messages_for_model)
    print("[chat] Response generated.")
    return {"messages": [response]}


def summarize_and_trim(state: ChatState) -> dict:
    """Summarize the conversation and retain only its latest two messages."""
    existing_summary = state.get("summary", "")

    if existing_summary:
        prompt = (
            "Update the existing conversation summary using the conversation "
            "messages. Preserve important facts, preferences, decisions, and "
            "unresolved questions. Avoid repetition. Return only the updated "
            "summary.\n\nExisting summary:\n"
            f"{existing_summary}"
        )
    else:
        prompt = (
            "Summarize the conversation messages, preserving important facts, "
            "preferences, decisions, and unresolved questions. Be concise. "
            "Return only the summary."
        )

    # The prompt is sent to the model but is not stored in the chat history.
    response = llm.invoke(
        [*state["messages"], HumanMessage(content=prompt)]
    )

    # Keep the last two messages verbatim and delete older messages.
    messages_to_delete = state["messages"][:-2]
    deletion_updates = [
        RemoveMessage(id=message.id)
        for message in messages_to_delete
        if message.id is not None
    ]

    print(
        "[summarize_and_trim] Summary updated; "
        f"removing {len(deletion_updates)} older message(s)."
    )
    return {
        "summary": str(response.content),
        "messages": deletion_updates,
    }


def should_summarize(
    state: ChatState,
) -> Literal["summarize", "__end__"]:
    """Trigger summarization when more than six messages are stored."""
    return "summarize" if len(state["messages"]) > 6 else END


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_and_trim)
builder.add_edge(START, "chat")
builder.add_conditional_edges(
    "chat",
    should_summarize,
    {"summarize": "summarize", END: END},
)
builder.add_edge("summarize", END)

# InMemorySaver stores checkpoints only for the lifetime of this process.
graph = builder.compile(checkpointer=InMemorySaver())


def show_state(config: dict) -> None:
    """Print the current checkpointed summary and message history."""
    snapshot = graph.get_state(config)
    values = snapshot.values