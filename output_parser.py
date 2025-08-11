"""Output parser for RAG responses."""
from typing import Dict, List, Tuple
import re
from ..models import QueryOutput


class RAGOutputParser:
    """Parses the output from the LLM into a structured format."""
    
    def parse_response(self, 
                      llm_output: str, 
                      context_docs: List[Dict],
                      model_name: str) -> QueryOutput:
        """
        Parse the LLM output and create a structured response.
        
        Args:
            llm_output: Raw output from the LLM
            context_docs: List of document chunks used as context
            model_name: Name of the model used (for logging only)
            
        Returns:
            QueryOutput object with structured answer
        """
        try:
            # Get the filename from the first document used in context
            # Since documents are returned in order of relevance, this is the most relevant source
            filename = context_docs[0].metadata.get('filename', 'unknown') if context_docs else 'unknown'
            
            return QueryOutput(
                success=True,
                filepdfname=filename,
                answer=llm_output.strip()
            )
            
        except Exception as e:
            # If there's any error in parsing, return an error response
            return QueryOutput(
                success=False,
                filepdfname="unknown",
                answer=f"Error processing response: {str(e)}"
            )
