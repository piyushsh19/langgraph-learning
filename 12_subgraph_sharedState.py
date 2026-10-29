
"""
LangGraph shared-state subgraph example.

Local LLM: Qwen2.5-3B-Instruct

Parent graph:
START -> create_draft -> improvement_subgraph -> final_summary -> END

Subgraph:
START -> review_draft -> revise_draft -> END

Both graphs use the same SharedState schema.
"""

from typing import TypedDict

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END


load_dotenv()

# --------------------------------------------------
# 1. Load the local LLM once
# --------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    model_kwargs={
        "local_files_only": True,
    },
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

llm = ChatHuggingFace(llm=llm_pipeline)


# --------------------------------------------------
# 2. Shared state schema
# --------------------------------------------------

class SharedState(TypedDict, total=False):
    topic: str
    draft: str
    feedback: str
    revised_draft: str
    final_summary: str


# --------------------------------------------------
# 3. Parent graph node: create initial draft
# --------------------------------------------------

def create_draft(state: SharedState) -> dict:
    """Create an initial draft from the topic."""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful educational content writer. "
                "Write a clear, informative draft of approximately "
                "150 words. Use simple language and a logical structure."
            )
        ),
        HumanMessage(
            content=f"Write a draft about: {state['topic']}"
        ),
    ]

    response = llm.invoke(messages)

    print("\n[Parent] Initial draft created.")

    return {
        "draft": str(response.content)
    }


# --------------------------------------------------
# 4. Subgraph node: review the draft
# --------------------------------------------------

def review_draft(state: SharedState) -> dict:
    """
    Read the shared topic and draft fields.
    Write feedback into the shared state.
    """

    messages = [
        SystemMessage(
            content=(
                "You are an editor. Review the supplied draft for "
                "clarity, accuracy, organization, and completeness. "
                "Give three specific, actionable suggestions. "
                "Do not rewrite the entire draft."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Draft:\n{state['draft']}"
            )
        ),
    ]

    response = llm.invoke(messages)

    print("[Subgraph] Draft reviewed.")

    return {
        "feedback": str(response.content)
    }


# --------------------------------------------------
# 5. Subgraph node: revise the draft
# --------------------------------------------------

def revise_draft(state: SharedState) -> dict:
    """
    Read topic, draft, and feedback from the shared state.
    Write the revised draft into the same state schema.
    """

    messages = [
        SystemMessage(
            content=(
                "You are an expert editor. Improve the original draft "
                "using the supplied feedback. Keep the writing clear, "
                "well structured, and relevant to the topic. "
                "Return only the revised draft."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Original draft:\n{state['draft']}\n\n"
                f"Feedback:\n{state['feedback']}"
            )
        ),
    ]

    response = llm.invoke(messages)

    print("[Subgraph] Draft revised.")

    return {
        "revised_draft": str(response.content)
    }


# --------------------------------------------------
# 6. Build the subgraph using SharedState
# --------------------------------------------------

subgraph_builder = StateGraph(SharedState)

subgraph_builder.add_node("review_draft", review_draft)
subgraph_builder.add_node("revise_draft", revise_draft)

subgraph_builder.add_edge(START, "review_draft")
subgraph_builder.add_edge("review_draft", "revise_draft")
subgraph_builder.add_edge("revise_draft", END)

improvement_subgraph = subgraph_builder.compile()


# --------------------------------------------------
# 7. Parent graph node: create final summary
# --------------------------------------------------

def create_final_summary(state: SharedState) -> dict:
    """Summarize the revised draft."""

    messages = [
        SystemMessage(
            content=(
                "Summarize the supplied content in exactly three "
                "concise bullet points. Return only the bullet points."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Revised draft:\n{state['revised_draft']}"
            )
        ),
    ]

    response = llm.invoke(messages)

    print("[Parent] Final summary created.")

    return {
        "final_summary": str(response.content)
    }


# --------------------------------------------------
# 8. Build the parent graph
# --------------------------------------------------

parent_builder = StateGraph(SharedState)

parent_builder.add_node("create_draft", create_draft)

# Add the compiled subgraph directly as a parent node.
parent_builder.add_node(
    "improvement_subgraph",
    improvement_subgraph,
)

parent_builder.add_node(
    "create_final_summary",
    create_final_summary,
)

parent_builder.add_edge(START, "create_draft")

parent_builder.add_edge(
    "create_draft",
    "improvement_subgraph",
)

parent_builder.add_edge(
    "improvement_subgraph",
    "create_final_summary",
)

parent_builder.add_edge("create_final_summary", END)

graph = parent_builder.compile()


# --------------------------------------------------
# 9. Execute the workflow
# --------------------------------------------------

def main():
    topic = input("Enter a topic: ").strip()

    if not topic:
        print("Please enter a topic.")
        return

    result = graph.invoke(
        {
            "topic": topic,
        },
        config={
            "run_name": "shared_state_subgraph_demo",
            "tags": ["langgraph", "subgraph", "shared-state"],
            "metadata": {
                "model": MODEL_ID,
            },
        },
    )

    print("\n" + "=" * 50)
    print("INITIAL DRAFT")
    print("=" * 50)
    print(result.get("draft", ""))

    print("\n" + "=" * 50)
    print("EDITOR FEEDBACK")
    print("=" * 50)
    print(result.get("feedback", ""))

    print("\n" + "=" * 50)
    print("REVISED DRAFT")
    print("=" * 50)
    print(result.get("revised_draft", ""))

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(result.get("final_summary", ""))


if __name__ == "__main__":
    main()
