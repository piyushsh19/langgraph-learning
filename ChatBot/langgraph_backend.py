import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

# Local Hugging Face model
load_dotenv()


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



# Tools
ddg = DuckDuckGoSearchRun(region="us-en")

@tool
def web_search(query: str) -> str:
    """Search the web for current information, latest news,
    today's gold prices, recent events, and up-to-date facts.
    Use this tool whenever information may have changed recently."""
    return ddg.invoke(query)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'apple', 'tesla') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


tools = [web_search, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """
You are a helpful assistant with access to these tools:

1. DuckDuckGo search:
   Use this for current information, news, today's prices,
   recent events, and facts that may have changed.

2. get_stock_price:
   Use this when the user asks for the latest stock quote.
   Pass the appropriate stock symbol.

3. calculator:
   Use this for arithmetic calculations.

IMPORTANT RULES:
- For current or recent information, you MUST use DuckDuckGo search.
- For questions about today's gold price, search the web.
- Do not guess current prices or claim to have searched if you did not.
- After receiving tool results, summarize them for the user.
- For simple arithmetic, use the calculator tool.
"""


def chat_node(state: ChatState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    # Debug: inspect whether the model actually requested a tool
    print("\n--- MODEL DEBUG ---")
    print("Tool calls:", response.tool_calls)
    print("AI response:", response.content)
    print("-------------------\n")

    return {"messages": [response]}


tool_node = ToolNode(tools)



graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

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