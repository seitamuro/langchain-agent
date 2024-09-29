#https://langchain-ai.github.io/langgraph/tutorials/introduction/

import getpass
import os

from typing import Annotated

from typing_extensions import TypedDict

from langchain_aws import ChatBedrock

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 必要な環境変数をエコーなしで入力させる関数
def _set_env(var: str):
  if not os.environ.get(var):
    os.environ[var] = getpass.getpass(f"{var} ")

class State(TypedDict):
  messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatBedrock(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
  model_kwargs={"max_tokens": 1500}
)

def chatbot(state: State):
  return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
graph.get_graph().print_ascii()

while True:
  user_input = input("User: ")
  print("User: " + user_input)
  if user_input.lower() in ["quit", "exit", "q"]:
    print("Goodbye!")
    break
  for event in graph.stream({"messages": ("user", user_input)}):
    for value in event.values():
      print("Assistant: " + value["messages"][-1].content)