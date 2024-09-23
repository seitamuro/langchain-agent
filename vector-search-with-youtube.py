import sys

from langchain_community.document_loaders import YoutubeLoader
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

args = sys.argv

youtube_url = args[1]
loader = YoutubeLoader.from_youtube_url(youtube_url)
raw_documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, BedrockEmbeddings())
query = "What is LangChain Indexes?"
docs = db.similarity_search(query)
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