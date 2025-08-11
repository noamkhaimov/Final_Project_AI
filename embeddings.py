import os
from pathlib import Path
from ..database import chroma, add_documents
from .pdf_processor import ingest_pdf


def store_pdf(file_path: Path) -> str:
    """Store PDF chunks in ChromaDB and return the collection name"""
    chunks = ingest_pdf(file_path)
    add_documents(chunks)
    return chroma._collection.name  # Chroma returns the collection id internally
