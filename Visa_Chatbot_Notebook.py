from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
tools=[search]
from langchain.chat_models.base import init_chat_model
llm = init_chat_model("ollama:llama3.2:3b")
llm_with_tool = llm.bind_tools(tools)
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

from langgraph.types import Command, interrupt

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages:Annotated[list,add_messages]

def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=tools)

def chatbot(state: State):
    message = llm_with_tool.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    
    return {"messages": [message]}

## Stategraph
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder= StateGraph(State)
builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
builder.add_node("tools", tool_node)
builder.add_edge("tools", "chatbot")
## Add Edges
builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge(START, "chatbot")

## compile the graph
graph=builder.compile(checkpointer=memory)



user_input = "What is green card visa fees I told you earlier."
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": user_input},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()