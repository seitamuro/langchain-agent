import chromadb
import streamlit as st
import glob
import os

# デフォルト以外の埋め込みモデルを読み込む
@st.cache_resource
def create_embedding_functions():
  model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  ef = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
  return ef

# Chromaのコレクションを作成する
@st.cache_resource
def create_collection():
  chroma_client = chromadb.Client()
  collection = chroma_client.create_collection(name="my_collection")
  return collection

# Chromaのコレクションにドキュメントを追加する
def add_document(collection, ef, filename, document):
  embedding = ef([document])[0]

  collection.add(
    embeddings=[embedding],
    documents=[document],
    metadatas=[{"source": filename}],
    ids=[filename]
  )

  return embedding

def glob_files():
  return glob.glob("data/*.txt")

st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

collection = create_collection()

with col1:
  st.header("構築")

  ef = create_embedding_functions()
  
  filelist = glob_files()
  
  for filepath in filelist:
    with open(filepath, encoding="utf8") as f:
      content = f.read()
      filename = os.path.basename(filepath)
      embedding = add_document(collection, ef, filename, content)
      st.subheader(filename)
      st.code(content)
      st.write(embedding)
      st.divider()
      
@st.fragment
def search():
  ef = create_embedding_functions()
  
  query_text = st.text_input("検索文字列", value="エビ餃子が食べたい")
  embedding = ef([query_text])[0]
  
  st.subheader("入力内容")
  st.code(query_text)
  st.write(embedding)

  
  result = collection.query(
    query_embeddings=[embedding],
    n_results=2
  )

  st.subheader("検索結果")
  st.write(result)

with col2:
  st.header("検索")
  search()