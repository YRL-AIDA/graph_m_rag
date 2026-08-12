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
    use_reranker: bool = False  # Option to use API reranker for re-ranking results
    use_mmr_reranker: bool = False  # Option to use MMR (Maximal Marginal Relevance) diversity-based reranking
    mmr_lambda: float = 0.7  # MMR relevance-diversity tradeoff (1.0 = pure relevance)
    mmr_min_relevance: float = 0.0  # MMR minimum relevance threshold
    use_semantic_graph: bool = False  # Option to enrich context with Entity and Community nodes from semantic graph
    use_structured_graph: bool = False  # Option to enrich context with structural graph neighbours (ORDER walk + cross-graph bridge)
    use_structural_parent_only: bool = False  # When use_structured_graph=True: only walk PARENT edges (skip ORDER neighbours)
    use_iterative_search: bool = False  # Option to use iterative (feedback-driven) retrieval for multi-hop questions
    use_question_decomposition: bool = False  # Option to decompose complex questions into sub-questions
    answer_format: Optional[str] = None  # Expected answer format: 'Int', 'Float', 'List', 'Str', 'None'


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
    context_blocks: Optional[List[str]] = None  # All context blocks sent to LLM
    response_metadata: Optional[Dict[str, Any]] = None  # Server-side timing breakdown:
    #   search_ms          — Qdrant search time
    #   enrichment_ms      — Neo4j + semantic graph enrichment + context building
    #   llm_generation_ms  — LLM call time
    #   total_ms           — server-side total (excludes network serialisation)
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