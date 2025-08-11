# Standard library imports
import os
from time import time

# Third-party LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Local application imports
from ..database.chroma_client import get_langchain_chroma, add_documents, get_llm_model_name
from .prompt_templates import QUERY_PROMPT


def load_pdf(doc_path: str, verbose: bool = False):
    """Load a PDF file and return its content as documents.
    
    Args:
        doc_path: Path to the PDF file
        verbose: Whether to print debug information
    
    Returns:
        List of documents from the PDF
    """
    if not doc_path:
        raise ValueError("PDF path cannot be empty")

    loader = PyPDFLoader(file_path=doc_path)
    data = loader.load()
    
    if verbose:
        print("PDF loaded successfully")
        print(f"First 100 chars: {data[0].page_content[:100]}")
    
    return data


def split_documents(documents, chunk_size: int = 1200, chunk_overlap: int = 300, verbose: bool = False):
    """Split documents into chunks for processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        verbose: Whether to print debug information
    
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    if verbose:
        print(f"Split into {len(chunks)} chunks")
        print(f"Example chunk: {chunks[0].page_content[:200]}...")
    
    return chunks


def process_pdf(pdf_path: str, verbose: bool = False):
    """Process a PDF file and add it to the vector store.
    
    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to print debug information
    
    Returns:
        The ChromaDB client for querying
    """
    # Extract filename without extension for document ID
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    
    documents = load_pdf(pdf_path, verbose)
    chunks = split_documents(documents, verbose=verbose)
    
    # Convert LangChain documents to text for ChromaDB
    texts = [doc.page_content for doc in chunks]
    add_documents(texts, doc_id)
    
    return get_langchain_chroma()  # Get a fresh LangChain Chroma instance


def build_chain(model_name: str = None) -> RunnablePassthrough:
    """Build a RAG chain for querying documents.
    
    Args:
        model_name: Name of the Ollama model to use (defaults to configured LLM model)
    
    Returns:
        A runnable chain that can be used for querying
    """
    if model_name is None:
        model_name = get_llm_model_name()
    
    # Use academic model only for final answer generation
    llm = ChatOllama(model=model_name)

    # Get a fresh Chroma instance and create simple retriever (much faster)
    vector_store = get_langchain_chroma()
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Single most relevant document for maximum speed

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | QUERY_PROMPT
        | llm
        | StrOutputParser()
    )
