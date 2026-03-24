from typing import Dict, List, Any, Optional

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

class UploadedFileInfo(BaseModel):
    """Model for uploaded file information"""
    file_hash: str
    filename: str
    upload_date: Optional[str] = None
    file_size: Optional[int] = None
    status: Optional[str] = None


class UploadedFilesListResponse(BaseModel):
    """Response model for list of uploaded files"""
    status: str
    message: str
    files: List[UploadedFileInfo]
    total_count: int = 0