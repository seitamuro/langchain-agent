import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

embedding_model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model_name)

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")
collection.add(
  documents=["This is document.", "My name is John."],
  metadatas=[{"source": "my_source"}, {"source": "my_source"}],
  ids=["id1", "id2"]
)

query_vector = model.encode(["Who"])

search_results = collection.query(
  query_embeddings=query_vector,
  n_results=1
)

print(search_results)

#default_ef = embedding_functions.DefaultEmbeddingFunction()
#vector = default_ef.__call__(["test"])
#print(vector)