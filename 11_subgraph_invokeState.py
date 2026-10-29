
"""LangGraph subgraph example using a local Hugging Face LLM.

Workflow:
Question -> English answer -> Hindi translation subgraph
"""

from typing import TypedDict

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()

# --------------------------------------------------
# 1. Load the local LLM
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

# Reuse the same local model for both tasks
local_llm = ChatHuggingFace(llm=llm_pipeline)


# --------------------------------------------------
# 2. Translation subgraph
# --------------------------------------------------

class SubState(TypedDict):
    input_text: str
    translated_text: str


def translate_text(state: SubState) -> dict[str, str]:
    """Translate the English answer into Hindi."""

    messages = [
        SystemMessage(
            content=(
                "You are a professional Hindi translator. "
                "Translate the supplied text into natural, clear Hindi. "
                "Preserve the original meaning. "
                "Return only the Hindi translation."
            )
        ),
        HumanMessage(
            content=f"Translate this text into Hindi:\n\n{state['input_text']}"
        ),
    ]

    response = local_llm.invoke(messages)

    return {
        "translated_text": str(response.content)
    }


subgraph_builder = StateGraph(SubState)

subgraph_builder.add_node("translate_text", translate_text)

subgraph_builder.add_edge(START, "translate_text")
subgraph_builder.add_edge("translate_text", END)

subgraph = subgraph_builder.compile()


# --------------------------------------------------
# 3. Parent graph state
# --------------------------------------------------

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str


# --------------------------------------------------
# 4. Parent graph nodes
# --------------------------------------------------

def generate_answer(state: ParentState) -> dict[str, str]:
    """Generate an answer in English using the local LLM."""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Answer the user's question clearly and accurately "
                "in English. Keep the answer concise."
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    response = local_llm.invoke(messages)

    return {
        "answer_eng": str(response.content)
    }


def translate_answer(state: ParentState) -> dict[str, str]:
    """Invoke the translation subgraph."""

    result = subgraph.invoke({
        "input_text": state["answer_eng"]
    })

    return {
        "answer_hin": result["translated_text"]
    }


# --------------------------------------------------
# 5. Build the parent graph
# --------------------------------------------------

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", translate_answer)

parent_builder.add_edge(START, "answer")
parent_builder.add_edge("answer", "translate")
parent_builder.add_edge("translate", END)

graph = parent_builder.compile()


# --------------------------------------------------
# 6. Run the workflow
# --------------------------------------------------

def main() -> None:
    print(f"Using local model: {MODEL_ID}")

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = graph.invoke({
        "question": question,
        "answer_eng": "",
        "answer_hin": "",
    })

    print("\n--- Answer (English) ---")
    print(result["answer_eng"])

    print("\n--- उत्तर (Hindi) ---")
    print(result["answer_hin"])


if __name__ == "__main__":
    main()
