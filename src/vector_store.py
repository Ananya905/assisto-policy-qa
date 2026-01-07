from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db
