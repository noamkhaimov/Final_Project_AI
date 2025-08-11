"""PDF processing utilities for handling PDF files."""
from pathlib import Path
from typing import List
import logging
from PyPDF2 import PdfReader
from langchain_community.docstore.document import Document
from ..rag import load_pdf, split_documents


def ingest_pdf(path: Path) -> List[Document]:
    """Load and split a PDF file into chunks."""
    documents = load_pdf(str(path))
    return split_documents(documents)


class PDFProcessor:
    """
    PDF processing class to handle PDF file operations
    Provides methods for reading and extracting content from PDF files
    """

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set = {'pdf'}) -> bool:
        """
        Check if uploaded file has allowed extension

        Args:
            filename: Name of the uploaded file
            allowed_extensions: Set of allowed file extensions

        Returns:
            bool: True if file extension is allowed
        """
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        """
        Extract text content from PDF file

        Args:
            file_stream: File stream object containing PDF data

        Returns:
            str: Extracted text content from PDF
        """
        try:
            # Create PDF reader object from file stream
            pdf_reader = PdfReader(file_stream)

            # Extract text from all pages
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Extract text from current page
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    # Log error but continue processing other pages
                    logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"

            return text_content.strip()

        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
            raise
