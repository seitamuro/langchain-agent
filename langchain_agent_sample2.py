from langchain_aws import ChatBedrock
from langchain_aws import BedrockLLM
from langchain.agents import AgentExecutor, create_xml_agent, Tool, initialize_agent, AgentType
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

"""
llm = BedrockLLM(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)
"""

chat = ChatBedrock(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
  model_kwargs={"max_tokens": 1500}
)

search = DuckDuckGoSearchRun()
tools = [
  Tool(
    name="duckduckgo-search",
    func=search.run,
    description="このツールはユーザーから検索キーワードを受け取り、Web上の最新情報を検索します。"
  )
]

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("""
現在の20代の日本人男性の平均身長を教えて。
"""
)