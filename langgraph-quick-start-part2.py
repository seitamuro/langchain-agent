#https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot
import json

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
  messages: Annotated[list, add_messages]

tool = DuckDuckGoSearchRun()
tools = [tool]

graph_builder = StateGraph(State)

llm = ChatBedrock(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
  model_kwargs={"max_tokens": 1500}
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
  return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
  def __init__(self, tools: list) -> None:
    self.tools_by_name = {tool.name: tool for tool in tools}

  def __call__(self, inputs: dict):
    if messages := inputs.get("messages", []):
      message = messages[-1]
    else:
      raise ValueError("No messages in inputs")
    outputs = []
    for tool_call in message.tool_calls:
      tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
      outputs.append(ToolMessage(
        content=json.dumps(tool_result),
        name = tool_call["name"],
        tool_call_id=tool_call["id"]
      ))
      return {"messages": outputs}

tool_node = BasicToolNode(tools)
graph_builder.add_node("tools", tool_node)

def route_tools(
  state: State
) -> Literal["tools", "__end__"]:
  if isinstance(state, list):
    ai_message = state[-1]
  elif messages := state.get("messages", []):
    ai_message = messages[-1]
  else:
    raise ValueError(f"No messages found in input state to tool_edge: {state}")
  if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
    return "tools"
  return "__end__"

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "__end__": "__end__"})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

graph.get_graph().print_ascii()

while True:
  user_input = input("User: ")
  print("User: " + user_input)
  if user_input.lower() in ["quit", "exit", "q"]:
    print("Goodbye!")
    break
  for event in graph.stream({"messages": [("user", user_input)]}):
    for value in event.values():
      if isinstance(value["messages"][-1], BaseMessage):
        print("Assistant: " + value["messages"][-1].content)  

"""
response = tool.invoke("日本の首都はどこですか?")
print(response)
"""