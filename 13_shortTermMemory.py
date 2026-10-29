"""
LangGraph message-state persistence and threads using a local Hugging Face LLM.

Workflow: START -> call_model -> END
"""

from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Load only from the local Hugging Face cache.
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
model = ChatHuggingFace(llm=llm_pipeline)


def call_model(state: MessagesState) -> dict:
    """Pass the complete conversation history to the local model."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def main() -> None:
    print(f"Local model: {MODEL_ID}")
    print("Reusing thread_id='thread-1' keeps this conversation's history.")
    print("Type 'exit', 'quit', or 'bye' to stop.")

    config = {"configurable": {"thread_id": "thread-1"}}

    while True:
        user_text = input("\nYou: ").strip()

        if user_text.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        if not user_text:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )
        print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
