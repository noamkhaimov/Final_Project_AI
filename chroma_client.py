import os
import time
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """Get consistent Ollama embedding function using nomic-embed-text for faster embeddings"""
    return OllamaEmbeddings(
        model="nomic-embed-text",  # Using faster embedding model
        base_url=os.getenv("OLLAMA_API_BASE", "http://ollama:11434")
    )

def get_llm_model_name():
    """Get the LLM model name for question answering (separate from embeddings)"""
    return os.getenv("OLLAMA_MODEL", "academic")

def init_chroma():
    """
    Initialize and return a ChromaDB client with retry logic
    """
    chroma_host = os.environ.get("CHROMA_HOST", "chromadb")
    max_retries = 5
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create persistent client
            client = chromadb.HttpClient(
                host=chroma_host,
                port=8000
            )
            return client
            
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Attempt {attempt + 1} failed to connect to ChromaDB. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("All attempts to connect to ChromaDB failed.")
                raise ValueError(f"Could not connect to ChromaDB after {max_retries} attempts: {str(e)}")

def get_collection(collection_name="academic_papers"):
    """Get or create a ChromaDB collection"""
    client = init_chroma()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def get_langchain_chroma(collection_name="academic_papers"):
    """
    Get a LangChain Chroma instance configured with Ollama embeddings
    """
    embedding_function = get_embedding_function()
    
    return Chroma(
        client=init_chroma(),
        collection_name=collection_name,
        embedding_function=embedding_function
    )

def add_documents(chunks, doc_id):
    """Add document chunks to the ChromaDB collection using LangChain Chroma for consistency"""
    # Use LangChain Chroma to ensure consistent embedding
    chroma_instance = get_langchain_chroma()
    
    # Extract text content and metadata from LangChain documents
    texts = []
    metadatas = []
    
    for chunk in chunks:
        if hasattr(chunk, 'page_content'):
            texts.append(chunk.page_content)
            # Preserve metadata from the chunk
            metadatas.append(chunk.metadata if hasattr(chunk, 'metadata') else {})
        else:
            texts.append(chunk)
            metadatas.append({})
    
    # Generate unique IDs for each chunk
    ids = [f"{doc_id}_{i}" for i in range(len(texts))]
    
    # Add documents using LangChain Chroma with metadata preserved
    chroma_instance.add_texts(texts=texts, metadatas=metadatas, ids=ids)

def similarity_search(query, k=5):
    """Perform similarity search using LangChain Chroma"""
    chroma = get_langchain_chroma()
    return chroma.similarity_search(query, k=k)
