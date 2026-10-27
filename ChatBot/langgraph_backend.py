import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Local Hugging Face model


model_id = "Qwen/Qwen2.5-3B-Instruct"

pipeline_model = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    model_kwargs={},
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
)
llm_pipeline = HuggingFacePipeline(pipeline=pipeline_model)
llm = ChatHuggingFace(llm=llm_pipeline)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

conn = sqlite3.connect(
    "checkpoints.sqlite",
    check_same_thread=False,
)
checkpointer = SqliteSaver(conn)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)