from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime


class UploadResponse(BaseModel):
    ids: List[str] = Field(example=["doc_123", "doc_456"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What is the main finding in paper X?")


class QueryAnswer(BaseModel):
    answer: str
    sources: List[str]


class QueryInput(BaseModel):
    """Validates the input for the query endpoint."""
    question: str = Field(..., description="The question to ask about the uploaded papers")
    
    @validator('question')
    def question_not_empty(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Question must be ≥3 characters')
        return v.strip()


class DocumentSource(BaseModel):
    """Represents a source document used in the answer."""
    document_id: str = Field(..., description="The ID of the source document")
    filename: str = Field(..., description="The original filename of the source")
    snippet: str = Field(..., description="The relevant text snippet from the source")


class QueryOutput(BaseModel):
    """Response format for the query endpoint."""
    success: bool = Field(default=True, description="Whether the query was successful")
    filepdfname: str = Field(..., description="The filename from which the answer was derived")
    answer: str = Field(..., description="The AI-generated answer to the question")
