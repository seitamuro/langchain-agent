#https://qiita.com/minorun365/items/6ca84b62230519d1d0ef

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

@tool
def search(query: str) -> str:
  """call to surf the web."""
  if "東京" in query:
    return "東京は今日も最高気温35度超えの猛暑です。"
  return "日本は今日、全国的に晴れです。"

tools = [search]

tool_node = ToolNode(tools)

model = ChatBedrockConverse(
  model="anthropic.claude-3-sonnet-20240229-v1:0",
).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
  messages = state["messages"]
  last_message = messages[-1]
  if last_message.tool_calls:
    return "tools"
  return END

def call_model(state: MessagesState):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue)

workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

app.get_graph().print_ascii()

final_state = app.invoke(
  {"messages": [HumanMessage(content="東京の天気を教えて")]},
  config={"configurable": {"thread_id": 42}}
)
result = final_state["messages"][-1].content

print(result)