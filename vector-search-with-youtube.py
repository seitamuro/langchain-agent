import sys

from langchain_community.document_loaders import YoutubeLoader
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

args = sys.argv

youtube_url = args[1]
search_query = args[2]

loader = YoutubeLoader.from_youtube_url(youtube_url, language="ja")
raw_documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, BedrockEmbeddings())
docs = db.similarity_search(search_query)
print("Found: ", len(docs))
print(docs[0].page_content)

"""
bedrock_embeddings = BedrockEmbeddings(
  model_id="anthropic.claude-3-sonnet-20240229-v1:0",
)

index = VectorstoreIndexCreator(
  vectorstore_cls=Chroma,
  embedding=bedrock_embeddings,
).from_loaders([loader])

query = "LangChain Indexesってなに?"

answer = index.query(query)
print(answer)
"""