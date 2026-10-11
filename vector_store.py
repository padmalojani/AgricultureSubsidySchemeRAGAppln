import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import config

def get_embeddings():
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY)

def create_vector_store(documents):
    """Creates a new FAISS vector store from documents and saves it locally."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    save_vector_store(vector_store)
    return vector_store

def load_vector_store():
    """Loads the persistent FAISS index from local storage if available."""
    embeddings = get_embeddings()
    index_file = os.path.join(config.FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        try:
            return FAISS.load_local(
                config.FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    return None

def save_vector_store(vector_store):
    """Saves the FAISS vector store to trusted local storage."""
    if vector_store:
        vector_store.save_local(config.FAISS_INDEX_DIR)

def add_documents_to_store(vector_store, documents):
    """Adds new documents to an existing vector store and persists them."""
    if vector_store is None:
        return create_vector_store(documents)
    vector_store.add_documents(documents)
    save_vector_store(vector_store)
    return vector_store

def search_similar_documents(vector_store, query, k=config.DEFAULT_RETRIEVAL_K):
    """Searches for similar documents in the index."""
    if not vector_store:
        return []
    return vector_store.similarity_search(query, k=k)

def clear_vector_store():
    """Clears the local FAISS index files."""
    import shutil
    if os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)