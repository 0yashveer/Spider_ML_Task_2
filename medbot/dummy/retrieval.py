import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient("./chroma_db")

guidelines_collection = client.get_collection("ml_chunks")
qa_collection = client.get_collection("qa_chunks")

def query_embedding(query):

    embedding = embedding_model(query)

    return embedding.tolist()

def search_guidelines(query_emedding, n_results=3):
    return guidelines_collection.query(query_embeddings=[query_embedding],
                                    n_results=n_results)

def search_qa(query_embedding, n_results=5):
    return qa_collection.query(query_embeddings=[query_embedding],
                            n_results=n_results)