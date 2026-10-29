"""
LangGraph PostgreSQL checkpointing with a local Hugging Face LLM.

Workflow:
START -> call_model -> END

Concepts:
- MessagesState for chat history
- PostgreSQL-backed checkpoints using PostgresSaver
- thread_id to resume a conversation
- Local Qwen2.5-3B-Instruct model (no OpenAI API)

Prerequisites:
1. PostgreSQL must be running and the database in DB_URI must exist.
2. Install dependencies from requirements-postgres-local-llm.txt.
3. Configure DATABASE_URL in a .env file.
4. Qwen2.5-3B-Instruct must already be in the local Hugging Face cache.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph


# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)


# -------------------------------------------------------------------
# 2. Load the local model from the Hugging Face cache
# -------------------------------------------------------------------

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    model_kwargs={"local_files_only": True},
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)
llm = ChatHuggingFace(llm=llm_pipeline)


# -------------------------------------------------------------------
# 3. Define the graph
# -------------------------------------------------------------------

def call_model(state: MessagesState) -> dict:
    """Answer using all messages in this thread's current state."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)


# -------------------------------------------------------------------
# 4. Run with a PostgreSQL checkpointer
# -------------------------------------------------------------------

def ask(graph, question: str, thread_id: str) -> str:
    """Send one message to a named conversation thread."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return str(result["messages"][-1].content)


def main() -> None:
    print(f"Local model: {MODEL_ID}")
    print("PostgreSQL checkpointing is enabled.")
    print("Type 'exit', 'quit', or 'bye' to stop.")

    # The context manager opens and closes the PostgreSQL connection pool.
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Run setup to create/update the checkpoint tables. Usually needed
        # once per database/schema; safe to run during this learning example.
        checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)
        thread_id = input(
            "Conversation thread ID [thread-1]: "
        ).strip() or "thread-1"

        print(f"Using thread_id={thread_id!r}. Reuse it to resume this thread.")

        while True:
            user_text = input("\nYou: ").strip()

            if user_text.lower() in {"exit", "quit", "bye"}:
                print("Goodbye!")
                break
            if not user_text:
                continue

            answer = ask(graph, user_text, thread_id)
            print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
