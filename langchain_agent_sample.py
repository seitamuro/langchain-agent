from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_xml_agent, Tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

chat = ChatBedrock(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
  model_kwargs={"max_tokens": 1500}
)
tools = [
  Tool(
    name="duckduckgo-search",
    func=search.run,
    description="このツールはユーザーから検索キーワードを受け取り、Web上の最新情報を検索します。"
  )
]
agent = create_xml_agent(chat, tools, prompt=hub.pull("hwchase17/xml-agent-convo"))
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
result = agent_executor.invoke({"input": "What is the mission of DASH"})
print(result)
