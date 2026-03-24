from typing import Dict, List, Any

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request model for asking a question about a document"""
    file_hash: str
    question: str
    limit: int = 10


class QuestionResponse(BaseModel):
    """Response model for question answering"""
    status: str
    message: str
    file_hash: str
    question: str
    answers: List[Dict[str, Any]]
    indexed: bool
