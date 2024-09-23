# https://zenn.dev/umi_mori/books/prompt-engineer/viewer/youtube_langchain_chatgpt
import sys

import boto3

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_community.document_loaders import YoutubeLoader
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.indexes import VectorstoreIndexCreator

args = sys.argv

youtube_url = args[1]
search_query = args[2]

loader = YoutubeLoader.from_youtube_url(youtube_url, language="ja")
raw_document = loader.load()[0]

bedrock_embeddings = BedrockEmbeddings(
  model_id="amazon.titan-embed-text-v1",
)

index = VectorstoreIndexCreator(
  vectorstore_cls=Chroma,
  embedding=bedrock_embeddings,
).from_loaders([loader])

query = "LangChain Indexesってなに?"

answer = index.query(question=query, llm=BedrockLLM(model_id="anthropic.claude-3-sonnet-20240229-v1:0"))
print(answer)