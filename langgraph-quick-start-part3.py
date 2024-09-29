## Checkpointを使うための準備
from langgraph.checkpoint.memory import MemorySaver

# 本番環境ではSqliteSaverやPostgresSaverを使うと良い
memory = MemorySaver()

## Part2で作成したエージェントを読み込む
from typing import Annotated

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
  messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = DuckDuckGoSearchResults(max_results=2)
tools = [tool]
llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs={"max_tokens": 1500})
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
  return { "messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node(tool_node)

graph_builder.add_conditional_edges(
  "chatbot", 
  tools_condition
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)
graph.get_graph().print_ascii()

# checkpointのidを指定
config = {"configurable": {"thread_id": "1"}}

user_input = "Hi there! You must not forget '1'."
events = graph.stream(
  {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
  event["messages"][-1].pretty_print()

user_input = "Remember a number?"
events = graph.stream(
  {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
  event["messages"][-1].pretty_print()

events = graph.stream(
  {"messages": [("user", "What is the number?")]}, {"configurable": {"thread_id": "2"}}, stream_mode="values"
)

snapshot = graph.get_state(config)
print("\nSnapshot:")
print(snapshot)