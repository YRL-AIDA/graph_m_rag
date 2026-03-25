from typing import Dict, List, Any, Optional

from pydantic import BaseModel


class CollectionCreateRequest(BaseModel):
    """Request model for creating a Qdrant collection"""
    collection_name: str
    vector_size: int = 2048
    distance: str = "COSINE"  # COSINE, DOT, EUCLID


class CollectionInfo(BaseModel):
    """Model for collection information"""
    name: str
    vectors_count: Optional[int] = None
    points_count: Optional[int] = None


class CollectionsListResponse(BaseModel):
    """Response model for list of collections"""
    status: str
    message: str
    collections: List[CollectionInfo]
    total_count: int = 0


class QuestionRequest(BaseModel):
    """Request model for asking a question about a document"""
    file_hash: str
    question: str
    limit: int = 10
    collection_name: Optional[str] = None  # Optional collection name
    use_llm: bool = False  # Option to generate answer using LLM

class QuestionResponse(BaseModel):
    """Response model for question answering"""
    status: str
    message: str
    file_hash: str
    question: str
    answers: List[Dict[str, Any]]
    indexed: bool
    collection_name: Optional[str] = None  # Collection name used
    llm_answer: Optional[str] = None  # LLM-generated answer if use_llm is True

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