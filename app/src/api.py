"""
Main API application for PDF processing.
Handles PDF upload to S3, processing with MinerU service,
and computing embeddings for each element in the result.
"""
import base64
import functools
import hashlib
import io
import json
import logging
import time
from datetime import datetime
from operator import itemgetter
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
import requests
from starlette.responses import HTMLResponse, FileResponse

from app.src.llm_client import ModelMessageDict, LLMClient
from app.src.qdrant_client_api import get_qdrant_client
from app.src.qwen3_emb_client import EmbeddingClient
from app.src.minio_client import MinioClient
from app.src.mineru_client import MinerUClient
from app.config.settings import settings
from app.src.reranker_client import RerankerClient
from app.src.schemas.reranker import Message
from app.src.utils.data_model import QuestionResponse, QuestionRequest, UploadedFileInfo, UploadedFilesListResponse, CollectionCreateRequest, CollectionInfo, CollectionsListResponse
from app.src.utils.mmr_reranker import mmr_rerank_with_threshold
from app.src.question_decomposer import decompose_question, merge_search_results
from app.src.iterative_search import iterative_retrieval
from app.src.utils.answer_formatter import format_answer
from semantic_graph.traversal import TraversalConfig, UnifiedGraphCrawler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Semantic graph enrichment availability flag
# The semantic_graph module provides advanced entity/community expansion
# via Manager methods accessible through the shared Neo4j connection.
try:
    from semantic_graph.manager import Manager as SemanticManager, ManagerConfig
    SEMANTIC_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Semantic graph module not available: {e}")
    SEMANTIC_GRAPH_AVAILABLE = False
    SemanticManager = None
    ManagerConfig = None

# Import document index service for Neo4j graph creation
try:
    from documet_index import DocumentIndexService, create_neo4j_graph
    NEO4J_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neo4j document index not available: {e}")
    NEO4J_AVAILABLE = False
    DocumentIndexService = None
    create_neo4j_graph = None

# Initialize clients
minio_client = MinioClient(logger)
mineru_client = MinerUClient(base_url=f"{settings.mineru.MINERU_HOST}:{settings.mineru.MINERU_PORT}")
emb_client = EmbeddingClient(base_url=settings.embedding.EMBEDDING_BASE_URL)
reranker_client = RerankerClient(base_url=settings.reranker.RERANKER_BASE_URL)


# C7: Module-level LRU cache for question embeddings.
# Must be at module scope so the cache persists across requests.
@functools.lru_cache(maxsize=256)
def _cached_question_embedding(text: str) -> list:
    """Return embedding for *text*, caching up to 256 most recent queries."""
    return emb_client.get_text_embedding(text)
qdrant_client = get_qdrant_client()
llm_client = LLMClient(base_url=settings.llm.LLM_BASE_URL)

# Create FastAPI application
app = FastAPI(
    title="PDF Processing API",
    description="API for uploading PDF documents to S3, processing with MinerU, and computing embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload"""
    status: str
    message: str
    file_hash: str
    s3_path: str
    mineru_result_path: str
    embeddings_computed: int
    processing_time: float
    neo4j_graph_created: bool = False


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    services: Dict[str, str]


async def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate MD5 hash of file content"""
    return hashlib.md5(file_bytes).hexdigest()


def convert_to_serializable(obj):
    """Convert object to JSON serializable format"""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def compute_embeddings_for_elements(elements: List[Dict], file_hash: str) -> int:
    """
    Compute embeddings for each element in the MinerU result

    Args:
        elements: List of elements from MinerU result in the format:
            [
              {
                "type": "text",
                "text": "Text content",
                "text_level": 1,
                "bbox": [x1, y1, x2, y2],
                "page_idx": 0
              },
              {
                "type": "image",
                "img_path": "path/to/image.jpg",
                "image_caption": [...],
                "image_footnote": [...],
                "bbox": [x1, y1, x2, y2],
                "page_idx": 0
              },
              {
                "type": "table",
                "img_path": "path/to/table.jpg",
                "table_caption": [...],
                "table_footnote": [...],
                "table_body": "<table>...</table>",
                "bbox": [x1, y1, x2, y2],
                "page_idx": 0
              },
              {
                "type": "discarded",
                ...
              }
            ]
        file_hash: Hash of the original PDF file

    Returns:
        Number of elements processed
    """
    processed_count = 0
    # Prepare lists for batch saving to Qdrant
    embeddings_list = []
    texts_list = []
    metadata_list = []

    # Use region_id counter similar to Neo4j's create_graph_from_mineru_result
    # to ensure matching IDs between Qdrant and Neo4j
    region_id = 0

    # Process each element according to its type
    for i, element in enumerate(elements):
        if not isinstance(element, dict):
            logger.warning(f"Element {i} is not a dictionary, skipping")
            continue

        element_type = element.get("type")

        # Skip discarded elements
        if element_type == "discarded":
            logger.info(f"Skipping discarded element {i}")
            continue

        # Prepare text content based on element type
        text_content = ""

        if element_type == "text":
            text = element.get("text", "")
            text_level = element.get("text_level")

            # Format text with level information if available
            if text == "":
                continue

            if text_level == 1:
                text_content = f"Title: {text}"
                element_type = "title"
            elif text_level is not None:
                text_content = f"Text (level {text_level}): {text}"
            else:
                text_content = f"Text: {text}"

        elif element_type == "image":
            # Combine image path and captions if available
            img_path = element.get("img_path", "")
            image_captions = element.get("image_caption", [])
            image_footnotes = element.get("image_footnote", [])

            # Download image from MinIO
            try:
                image_data = minio_client.get_object(
                    bucket_name=minio_client.bucket_name,
                    object_name=img_path
                )

                # Get image as bytes
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                caption_text = " ".join(image_captions) if image_captions else ""
                footnote_text = " ".join(image_footnotes) if image_footnotes else ""
                text = f'Figure | Image | Chart:'
                if caption_text:
                    text = f'{text} | {caption_text}'
                if footnote_text:
                    text = f'{text} | {footnote_text}'
                # Compute embedding for the image
                embedding = emb_client.get_image_text_embedding_base64(text, image_base64)

                # Prepare data for Qdrant with region_id matching Neo4j
                embeddings_list.append(embedding)
                texts_list.append(f"Image: {img_path}")  # Text representation for Qdrant

                metadata = {
                    "region_id": region_id,
                    "element_index": i,
                    "element_type": element_type,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat(),
                    "original_element": element,
                    "img_path": img_path,
                    "bbox": element.get("bbox", []),
                    "page_idx": element.get("page_idx", 0)
                }

                metadata_list.append(metadata)
                # Save embedding to S3 with a specific naming convention
                embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                embedding_data = {
                    "original_element": element,
                    "img_path": img_path,
                    "embedding": embedding,
                    "region_id": region_id,
                    "element_index": i,
                    "element_type": element_type,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat()
                }

                # Convert to JSON and upload to MinIO
                embedding_json = json.dumps(embedding_data, ensure_ascii=False)
                minio_client.put_object(
                    bucket_name=minio_client.bucket_name,
                    object_name=embedding_key,
                    data=embedding_json.encode('utf-8'),
                    content_type='application/json'
                )

                processed_count += 1
                region_id += 1
                logger.info(f"Computed embedding for image element {i} (region_id: {region_id-1}, type: {element_type}, path: {img_path})")

                # Create text embeddings from image captions if available
                caption_text = " ".join(image_captions) if image_captions else ""
                footnote_text = " ".join(image_footnotes) if image_footnotes else ""

                if caption_text:
                    caption_content = f"Image Caption: {caption_text}"

                    try:
                        caption_embedding = emb_client.get_text_embedding(caption_content)

                        # Prepare data for Qdrant with region_id matching Neo4j
                        embeddings_list.append(caption_embedding)
                        texts_list.append(caption_content)

                        caption_metadata = {
                            "region_id": region_id,
                            "element_index": i,
                            "element_type": f"{element_type}_caption",
                            "file_hash": file_hash,
                            "created_at": datetime.now().isoformat(),
                            "original_element": element,
                            "img_path": img_path,
                            "bbox": element.get("bbox", []),
                            "page_idx": element.get("page_idx", 0)
                        }
                        metadata_list.append(caption_metadata)

                        # Save caption embedding to S3
                        caption_embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                        caption_embedding_data = {
                            "original_element": element,
                            "img_path": img_path,
                            "text": caption_content,
                            "embedding": caption_embedding,
                            "region_id": region_id,
                            "element_index": i,
                            "element_type": f"{element_type}_caption",
                            "file_hash": file_hash,
                            "created_at": datetime.now().isoformat()
                        }

                        caption_json = json.dumps(caption_embedding_data, ensure_ascii=False)
                        minio_client.put_object(
                            bucket_name=minio_client.bucket_name,
                            object_name=caption_embedding_key,
                            data=caption_json.encode('utf-8'),
                            content_type='application/json'
                        )

                        processed_count += 1
                        region_id += 1
                        logger.info(f"Computed text embedding for image caption (region_id: {region_id-1}, type: {element_type}_caption)")

                    except Exception as e:
                        logger.error(f"Failed to compute text embedding for image caption element {i}: {e}")

                if footnote_text:
                    footnote_content = f"Image Footnote: {footnote_text}"

                    try:
                        footnote_embedding = emb_client.get_text_embedding(footnote_content)

                        # Prepare data for Qdrant with region_id matching Neo4j
                        embeddings_list.append(footnote_embedding)
                        texts_list.append(footnote_content)

                        footnote_metadata = {
                            "region_id": region_id,
                            "element_index": i,
                            "element_type": f"{element_type}_footnote",
                            "file_hash": file_hash,
                            "created_at": datetime.now().isoformat(),
                            "original_element": element,
                            "img_path": img_path,
                            "bbox": element.get("bbox", []),
                            "page_idx": element.get("page_idx", 0)
                        }
                        metadata_list.append(footnote_metadata)

                        # Save caption embedding to S3
                        footnote_embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                        footnote_embedding_data = {
                            "original_element": element,
                            "img_path": img_path,
                            "text": footnote_text,
                            "embedding": footnote_embedding,
                            "region_id": region_id,
                            "element_index": i,
                            "element_type": f"{element_type}_footnote",
                            "file_hash": file_hash,
                            "created_at": datetime.now().isoformat()
                        }

                        caption_json = json.dumps(footnote_embedding_data, ensure_ascii=False)
                        minio_client.put_object(
                            bucket_name=minio_client.bucket_name,
                            object_name=footnote_embedding_key,
                            data=caption_json.encode('utf-8'),
                            content_type='application/json'
                        )

                        processed_count += 1
                        region_id += 1
                        logger.info(f"Computed text embedding for image footnote (region_id: {region_id-1}, type: {element_type}_footnote)")

                    except Exception as e:
                        logger.error(f"Failed to compute text embedding for image footnote element {i}: {e}")

                continue

            except Exception as e:
                logger.error(f"Failed to download or process image {img_path} for element {i}: {e}")


        elif element_type == "table":
            # Combine table information
            img_path = element.get("img_path", "")
            table_captions = element.get("table_caption", [])
            table_footnotes = element.get("table_footnote", [])
            table_body = element.get("table_body", "")

            caption_text = " ".join(table_captions) if table_captions else ""
            footnote_text = " ".join(table_footnotes) if table_footnotes else ""

            # First, create a separate text embedding for table captions/footnotes if available
            if caption_text:
                caption_content = f"Table Caption: {caption_text}"

                try:
                    caption_embedding = emb_client.get_text_embedding(caption_content)

                    # Prepare data for Qdrant with region_id matching Neo4j
                    embeddings_list.append(caption_embedding)
                    texts_list.append(caption_content)

                    caption_metadata = {
                        "region_id": region_id,
                        "element_index": i,
                        "element_type": f"{element_type}_caption",
                        "file_hash": file_hash,
                        "created_at": datetime.now().isoformat(),
                        "original_element": element,
                        "img_path": img_path,
                        "bbox": element.get("bbox", []),
                        "page_idx": element.get("page_idx", 0)
                    }
                    metadata_list.append(caption_metadata)

                    # Save caption embedding to S3
                    caption_embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                    caption_embedding_data = {
                        "original_element": element,
                        "img_path": img_path,
                        "text": caption_content,
                        "embedding": caption_embedding,
                        "region_id": region_id,
                        "element_index": i,
                        "element_type": f"{element_type}_caption",
                        "file_hash": file_hash,
                        "created_at": datetime.now().isoformat()
                    }

                    caption_json = json.dumps(caption_embedding_data, ensure_ascii=False)
                    minio_client.put_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=caption_embedding_key,
                        data=caption_json.encode('utf-8'),
                        content_type='application/json'
                    )

                    processed_count += 1
                    region_id += 1
                    logger.info(f"Computed text embedding for table caption (region_id: {region_id-1}, type: {element_type}_caption)")

                except Exception as e:
                    logger.error(f"Failed to compute text embedding for table caption element {i}: {e}")
            if footnote_text:
                footnote_content = f"Table Footnote: {footnote_text}"

                try:
                    footnote_embedding = emb_client.get_text_embedding(footnote_content)

                    # Prepare data for Qdrant with region_id matching Neo4j
                    embeddings_list.append(footnote_embedding)
                    texts_list.append(footnote_content)

                    footnote_metadata = {
                        "region_id": region_id,
                        "element_index": i,
                        "element_type": f"{element_type}_footnote",
                        "file_hash": file_hash,
                        "created_at": datetime.now().isoformat(),
                        "original_element": element,
                        "img_path": img_path,
                        "bbox": element.get("bbox", []),
                        "page_idx": element.get("page_idx", 0)
                    }
                    metadata_list.append(footnote_metadata)

                    # Save caption embedding to S3
                    footnote_embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                    footnote_embedding_data = {
                        "original_element": element,
                        "img_path": img_path,
                        "text": footnote_content,
                        "embedding": footnote_embedding,
                        "region_id": region_id,
                        "element_index": i,
                        "element_type": f"{element_type}_footnote",
                        "file_hash": file_hash,
                        "created_at": datetime.now().isoformat()
                    }

                    footnote_json = json.dumps(footnote_embedding_data, ensure_ascii=False)
                    minio_client.put_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=footnote_embedding_key,
                        data=footnote_json.encode('utf-8'),
                        content_type='application/json'
                    )

                    processed_count += 1
                    region_id += 1
                    logger.info(f"Computed text embedding for table footnote (region_id: {region_id-1}, type: {element_type}_footnote)")

                except Exception as e:
                    logger.error(f"Failed to compute text embedding for table footnote element {i}: {e}")

            # Now process the full table content (body + captions + footnotes)
            text_content = "Table: "
            #if caption_text:
            #    text_content += f" | Caption: {caption_text}"
            #if footnote_text:
            #    text_content += f" | Footnote: {footnote_text}"
            if table_body:
                text_content += f" | Body: {table_body}"

        elif element_type == "equation":
            # Extract LaTeX equation content
            latex = element.get("latex", "")

            if latex == "":
                continue

            text_content = f"Equation: {latex}"
        else:
            # For unknown types, try to extract any available text content
            text_content = json.dumps(element, ensure_ascii=False)

        # Only process elements with non-empty text content
        if text_content.strip() or element_type != "image":
            try:
                # Generate embedding using the embedding client
                embedding = emb_client.get_text_embedding(text_content)

                # Prepare data for Qdrant with region_id matching Neo4j
                embeddings_list.append(embedding)
                texts_list.append(text_content)

                metadata = {
                    "region_id": region_id,
                    "element_index": i,
                    "element_type": element_type,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat(),
                    "page_idx": element.get("page_idx", 0),
                    "bbox": element.get("bbox", []),
                    "original_element": element
                }
                metadata_list.append(metadata)

                # Save embedding to S3 with a specific naming convention
                embedding_key = f"embeddings/{file_hash}/region_{region_id}.json"
                embedding_data = {
                    "original_element": element,
                    "text": text_content,
                    "embedding": embedding,
                    "region_id": region_id,
                    "element_index": i,
                    "element_type": element_type,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat()
                }

                # Convert to JSON and upload to MinIO
                embedding_json = json.dumps(embedding_data, ensure_ascii=False)
                minio_client.put_object(
                    bucket_name=minio_client.bucket_name,
                    object_name=embedding_key,
                    data=embedding_json.encode('utf-8'),
                    content_type='application/json'
                )

                processed_count += 1
                region_id += 1
                logger.info(f"Computed embedding for element {i} (region_id: {region_id-1}, type: {element_type})")

            except Exception as e:
                logger.error(f"Failed to compute embedding for element {i} (type: {element_type}): {e}")
                continue
    try:
        # Create collection if it doesn't exist (using the size of the first embedding)
        if embeddings_list and len(embeddings_list) > 0:
            qdrant_client.create_collection(vector_size=len(embeddings_list[0]))

            # Save embeddings to Qdrant
            success = qdrant_client.save_embeddings(
                embeddings=embeddings_list,
                texts=texts_list,
                metadata_list=metadata_list
            )

            if success:
                logger.info(f"Saved {len(embeddings_list)} embeddings to Qdrant collection")
            else:
                logger.error("Failed to save embeddings to Qdrant")
    except Exception as e:
        logger.error(f"Error saving embeddings to Qdrant: {e}")

    return processed_count


def process_with_mineru(file_path: str) -> Dict[str, Any]:
    """
    Process PDF file with MinerU service

    Args:
        file_path: Path to the temporary file

    Returns:
        MinerU processing result
    """
    try:
        # Process the file using the MinerU client
        result = mineru_client.process_document(
            file_path=file_path,
            backend="pipeline",
            method="auto",
            lang="en",
            formula_enable=True,
            table_enable=True
        )
        return result
    except Exception as e:
        logger.error(f"Error calling MinerU service: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing with MinerU service: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload-pdf": "Upload and process PDF file",
            "GET /health": "Health check",
            "POST /ask-document": "Ask a question about a document",
            "GET /ask-document": "Web interface for asking questions about documents",
            "GET /uploaded-files": "Get list of uploaded files with hashes",
            "GET /collections": "Get list of Qdrant collections",
            "POST /collections": "Create a new Qdrant collection",
            "DELETE /collections/{collection_name}": "Delete a Qdrant collection"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {}

    # Check MinIO connectivity
    try:
        minio_client.list_buckets()
        services_status["s3"] = "healthy"
    except Exception as e:
        logger.error(f"S3 health check failed: {e}")
        services_status["s3"] = f"unhealthy: {str(e)}"

    # Check embedding service
    try:
        # Test embedding generation
        test_embedding = emb_client.get_text_embedding("test")
        if test_embedding and len(test_embedding) > 0:
            services_status["embedding"] = "healthy"
        else:
            services_status["embedding"] = "unhealthy: invalid response"
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        services_status["embedding"] = f"unhealthy: {str(e)}"

    # Check MinerU service
    try:
        if mineru_client.health_check():
            services_status["mineru"] = "healthy"
        else:
            services_status["mineru"] = "unhealthy: service not responding"
    except Exception as e:
        logger.error(f"MinerU service health check failed: {e}")
        services_status["mineru"] = f"unhealthy: {str(e)}"

    all_healthy = all(status == "healthy" for status in services_status.values())

    return HealthCheckResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services_status
    )

@app.get("/uploaded-files", response_model=UploadedFilesListResponse)
def get_uploaded_files():
    """
    Get list of all uploaded PDF files with their hashes

    Returns:
        List of uploaded files with file names and hashes
    """
    try:
        # List all PDF objects in MinIO
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix="pdfs/"
        )

        files = []
        seen_hashes = set()

        for pdf_path in existing_pdfs:
            # Extract file_hash from path: pdfs/{file_hash}_{filename}/{filename}
            # or pdfs/{file_hash}_{filename}
            parts = pdf_path.split('/')
            if len(parts) >= 2:
                dir_name = parts[1]  # e.g., "a1b2c3d4_filename.pdf"
                file_name = parts[-1] if len(parts) > 2 else dir_name.split('_', 1)[-1] if '_' in dir_name else dir_name

                # Extract hash from directory name (format: hash_filename)
                if '_' in dir_name:
                    file_hash = dir_name.split('_', 1)[0]
                else:
                    # Fallback: try to extract from filename
                    file_hash = "unknown"

                # Skip duplicates (same hash)
                if file_hash in seen_hashes:
                    continue
                seen_hashes.add(file_hash)

                # Try to get upload date from object metadata
                try:
                    stat = minio_client.client.stat_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=pdf_path
                    )
                    upload_date = stat.last_modified.isoformat() if stat.last_modified else "unknown"
                except Exception:
                    upload_date = "unknown"

                files.append(UploadedFileInfo(
                    file_name=file_name,
                    file_hash=file_hash,
                    s3_path=pdf_path,
                    upload_date=upload_date
                ))

        return UploadedFilesListResponse(
            status="success",
            files=files
        )

    except Exception as e:
        logger.error(f"Error listing uploaded files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving file list: {str(e)}"
        )


@app.get("/collections", response_model=CollectionsListResponse)
def get_collections():
    """
    Get list of all Qdrant collections with their info

    Returns:
        List of collections with name and point counts
    """
    try:
        client = get_qdrant_client()
        collection_names = client.list_collections()

        collections_info = []
        for col_name in collection_names:
            try:
                # Create a client for this specific collection to get info
                col_client = get_qdrant_client(collection_name=col_name)
                col_info = col_client.client.get_collection(col_name)
                collections_info.append(CollectionInfo(
                    name=col_name,
                    vectors_count=col_info.vectors_count if hasattr(col_info, 'vectors_count') else None,
                    points_count=col_info.points_count if hasattr(col_info, 'points_count') else col_info.vectors_count
                ))
            except Exception as e:
                logger.warning(f"Could not get info for collection {col_name}: {e}")
                collections_info.append(CollectionInfo(name=col_name))

        return CollectionsListResponse(
            status="success",
            message=f"Found {len(collections_info)} collections",
            collections=collections_info,
            total_count=len(collections_info)
        )
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving collections list: {str(e)}"
        )


@app.post("/collections", response_model=Dict[str, Any])
def create_collection(request: CollectionCreateRequest):
    """
    Create a new Qdrant collection

    Args:
        request: CollectionCreateRequest with collection_name, vector_size, and distance

    Returns:
        Status of collection creation
    """
    try:
        client = get_qdrant_client(collection_name=request.collection_name)

        # Check if collection already exists
        if client.client.collection_exists(request.collection_name):
            return {
                "status": "already_exists",
                "message": f"Collection '{request.collection_name}' already exists",
                "collection_name": request.collection_name
            }

        # Map distance string to enum
        from qdrant_client.http.models import Distance as QdrantDistance
        distance_map = {
            "COSINE": QdrantDistance.COSINE,
            "DOT": QdrantDistance.DOT,
            "EUCLID": QdrantDistance.EUCLID
        }
        distance = distance_map.get(request.distance.upper(), QdrantDistance.COSINE)

        # Create collection
        success = client.create_collection(
            vector_size=request.vector_size,
            distance=distance
        )

        if success:
            return {
                "status": "success",
                "message": f"Collection '{request.collection_name}' created successfully",
                "collection_name": request.collection_name,
                "vector_size": request.vector_size,
                "distance": request.distance
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create collection '{request.collection_name}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating collection: {str(e)}"
        )


@app.delete("/collections/{collection_name}", response_model=Dict[str, Any])
def delete_collection(collection_name: str):
    """
    Delete a Qdrant collection

    Args:
        collection_name: Name of the collection to delete

    Returns:
        Status of collection deletion
    """
    try:
        client = get_qdrant_client(collection_name=collection_name)

        # Check if collection exists
        if not client.client.collection_exists(collection_name):
            return {
                "status": "not_found",
                "message": f"Collection '{collection_name}' does not exist",
                "collection_name": collection_name
            }

        # Delete collection
        success = client.delete_collection()

        if success:
            return {
                "status": "success",
                "message": f"Collection '{collection_name}' deleted successfully",
                "collection_name": collection_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete collection '{collection_name}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}"
        )


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF document to S3, process with MinerU, and compute embeddings

    Steps:
    1. Validate and read PDF file
    2. Check if file already exists by hash
    3. Upload PDF to S3
    4. Save to temporary file for processing
    5. Process with MinerU service
    6. Store MinerU results in S3
    7. Compute embeddings for each element in the result
    """
    start_time = time.time()
    temp_file_path = None

    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF with .pdf extension"
        )

    # Validate filename to prevent path traversal attacks
    safe_filename = Path(file.filename).name
    if safe_filename != file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    try:
        # Read file content into memory
        # For large files, consider using SpooledTemporaryFile or streaming
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )

        # Calculate file hash based on content
        file_hash = hashlib.md5(content).hexdigest()

        # Check if PDF with this hash already exists
        existing_pdf_prefix = f"pdfs/{file_hash}_"
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=existing_pdf_prefix
        )

        if existing_pdfs:
            logger.info(f"PDF with hash {file_hash} already exists in MinIO, skipping processing")
            processing_time = time.time() - start_time

            first_pdf_path = existing_pdfs[0]
            existing_unique_id = first_pdf_path.split('/')[1] if '/' in first_pdf_path else file_hash

            return PDFUploadResponse(
                status="already_processed",
                message="PDF file was already processed previously",
                file_hash=file_hash,
                s3_path=first_pdf_path,
                mineru_result_path=f"mineru_results/{existing_unique_id}/result.json",
                embeddings_computed=0,
                processing_time=processing_time
            )

        # Create unique identifier for this file
        file_unique_id = f"{file_hash}_{safe_filename}"
        pdf_s3_key = f"pdfs/{file_unique_id}/{safe_filename}"
        mineru_result_key = f"mineru_results/{file_unique_id}/result.json"

        # Upload original PDF to MinIO first
        logger.info(f"Uploading PDF to S3: {pdf_s3_key}")
        minio_client.upload(
            bucket_name=minio_client.bucket_name,
            object_name=pdf_s3_key,
            data=content,
            content_type="application/pdf"
        )
        logger.info(f"Successfully uploaded PDF to S3: {pdf_s3_key}")

        # Create temporary file for MinerU processing
        temp_dir = Path("/tmp/pdf_processing")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"{file_hash}_{safe_filename}"
        temp_file_path = str(temp_file)

        try:
            # Write content to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(content)

            # Verify file was written correctly
            if not temp_file.exists():
                raise IOError(f"Failed to create temporary file: {temp_file_path}")

            # Process with MinerU
            logger.info(f"Processing PDF {file_hash} with MinerU service")
            mineru_result = process_with_mineru(temp_file_path)

        finally:
            # Clean up temporary file immediately after processing
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Removed temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not remove temporary file {temp_file_path}: {cleanup_error}")

        # Store MinerU result in S3
        logger.info(f"Storing MinerU result to S3: {mineru_result_key}")
        mineru_result_serializable = convert_to_serializable(mineru_result)
        mineru_result_serializable["metadata"] = {
            "file_hash": file_hash,
            "original_filename": safe_filename,
            "processed_at": datetime.now().isoformat()
        }

        result_json = json.dumps(mineru_result_serializable, ensure_ascii=False, indent=2)

        minio_client.put_object(
            bucket_name=minio_client.bucket_name,
            object_name=mineru_result_key,
            data=result_json.encode('utf-8'),
            content_type="application/json"
        )

        logger.info(f"Successfully stored MinerU result to S3: {mineru_result_key}")

        # Save images from MinerU result
        images = mineru_result.get("results", {}).get("result", {}).get("results", {}).get("images_base64", {})

        if images:
            logger.info(f"Saving {len(images)} images from MinerU result")
            for img_key, image_base64 in images.items():
                try:
                    image_key = f"images/{img_key}"
                    image_data = base64.b64decode(image_base64)

                    minio_client.put_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=image_key,
                        data=image_data,
                        content_type="image/jpeg"
                    )
                except Exception as img_error:
                    logger.error(f"Failed to save image {img_key}: {img_error}")

        # Compute embeddings for each element in the result
        logger.info(f"Computing embeddings for MinerU result elements")

        # Extract elements from MinerU result
        elements = mineru_result.get("results", {}).get("result", {}).get("results", {}).get("content_list", [])

        if not elements:
            logger.warning(f"No content elements found in MinerU result for file {file_hash}")

        # Compute embeddings synchronously
        embeddings_count = compute_embeddings_for_elements(elements, file_hash)
        logger.info(f"Completed embedding computation: {embeddings_count} elements processed for file {file_hash}")

        # Create Neo4j graph from MinerU result
        neo4j_graph_created = False
        if NEO4J_AVAILABLE:
            try:
                logger.info(f"Creating Neo4j graph for document '{file_hash}'")
                neo4j_graph_created = create_neo4j_graph(mineru_result, file_hash)
                if neo4j_graph_created:
                    logger.info(f"Successfully created Neo4j graph for document '{file_hash}'")
                else:
                    logger.info(f"Document '{file_hash}' already exists in Neo4j, skipping graph creation")
            except Exception as neo4j_error:
                logger.error(f"Failed to create Neo4j graph for document '{file_hash}': {neo4j_error}")
                # Don't fail the entire process if Neo4j graph creation fails
                # The document is still available in MinIO and Qdrant
        else:
            logger.info("Neo4j document index not available, skipping graph creation")

        # Make semantic graph
        if NEO4J_AVAILABLE:
            try:
                url = "http://localhost:9595/process-document"
                data = {
                    "document_id": file_hash
                }

                response = requests.post(url, json=data)

                logger.info(f"Semantic graph construction initiated for document {file_hash}. Status: {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    logger.info(f"Semantic graph construction response: {response_data}")

                    # Connect structural and semantic graphs after semantic graph is built
                    try:
                        connect_structural_and_semantic_graphs(file_hash)
                    except Exception as e:
                        logger.error(f"Error connecting structural and semantic graphs: {e}")
                        
                    # Additionally, run the comprehensive graph connection process
                    try:
                        comprehensive_connect_graphs(file_hash)
                    except Exception as e:
                        logger.error(f"Error in comprehensive graph connection: {e}")

                else:
                    logger.error(f"Semantic graph construction failed with status {response.status_code}: {response.json()}")
            except Exception as e:
                logger.error(f"Error initiating semantic graph construction for document {file_hash}: {e}")
        else:
            logger.info("Neo4j document index not available, skipping semantic graph creation")

        processing_time = time.time() - start_time

        return PDFUploadResponse(
            status="success",
            message="PDF uploaded, processed with MinerU, embeddings computed, and Neo4j graph created",
            file_hash=file_hash,
            s3_path=pdf_s3_key,
            mineru_result_path=mineru_result_key,
            embeddings_computed=embeddings_count,
            processing_time=processing_time,
            neo4j_graph_created=neo4j_graph_created
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )
    finally:
        # Ensure temporary file is cleaned up even if an error occurs
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file in finally block: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary file {temp_file_path}: {cleanup_error}")


def check_document_indexed(file_hash: str, client=None) -> bool:
    """
    Check if a document is already indexed in Qdrant by file_hash

    Args:
        file_hash: Hash of the PDF file
        client: Optional QdrantClientWrapper instance. If not provided, uses default client.

    Returns:
        True if document is indexed, False otherwise
    """
    try:
        from qdrant_client.http import models

        # Use provided client or default
        q_client = client or qdrant_client

        # Search for any point with this file_hash
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash)
                )
            ]
        )

        results = q_client.search(
            query_vector=[0.0] * 2048,  # Dummy vector, we just need to check existence
            limit=1,
            filter_condition=filter_condition
        )

        return len(results.points) > 0
    except Exception as e:
        logger.error(f"Error checking if document is indexed: {e}")
        return False


def connect_structural_and_semantic_graphs(file_hash: str):
    """
    Connect structural graph (created from MinerU results) with semantic graph (created by semantic_graph module)

    Args:
        file_hash: Hash of the PDF file
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j not available, skipping graph connection")
        return

    try:
        # Connect to Neo4j to create relationships between structural and semantic graphs
        neo4j_service = DocumentIndexService()

        # Create a relationship between the document node (structural graph)
        # and any corresponding community nodes (semantic graph)
        with neo4j_service.driver.session() as session:
            # First, ensure the document node exists in the structural graph
            # Create or merge the document node if it doesn't exist
            doc_query = """
            MERGE (d:Document {file_hash: $file_hash})
            ON CREATE SET d.created_at = datetime()
            ON MATCH SET d.updated_at = datetime()
            RETURN d
            """
            session.run(doc_query, file_hash=file_hash)

            # Find all entities related to this document in the structural graph
            struct_query = """
            MATCH (d:Document {file_hash: $file_hash})<-[:PART_OF]-(e:Entity)
            RETURN e.title AS entity_title, e.label AS entity_label
            """

            struct_results = session.run(struct_query, file_hash=file_hash)
            document_entities = [(record["entity_title"], record["entity_label"]) for record in struct_results]

            # Connect document entities to their corresponding communities in the semantic graph
            connected_count = 0
            for entity_title, entity_label in document_entities:
                # Find the corresponding entity in the semantic graph and its community
                connect_query = """
                MATCH (d:Document {file_hash: $file_hash})
                MATCH (e:Entity {title: $entity_title, type: $entity_label})
                MATCH (e)-[:IN_COMMUNITY]->(c:Community)
                MERGE (d)-[:CONNECTS_TO {relationship_type: 'SEMANTIC_CONNECTION', created_at: datetime()}]->(c)
                RETURN COUNT(*) AS connection_count
                """

                result = session.run(
                    connect_query,
                    file_hash=file_hash,
                    entity_title=entity_title,
                    entity_label=entity_label
                )

                record = result.single()
                connected_count += record["connection_count"] if record else 0

            logger.info(f"Created {connected_count} connections between structural and semantic graphs for document {file_hash}")

        neo4j_service.close()

    except Exception as e:
        logger.error(f"Error connecting structural and semantic graphs for document {file_hash}: {e}")
        # Re-raise the exception to be caught by the caller
        raise


def comprehensive_connect_graphs(file_hash: str):
    """
    Perform a comprehensive connection between structural and semantic graphs.
    
    Args:
        file_hash: Hash of the PDF file
    """
    if not NEO4J_AVAILABLE:
        logger.warning("Neo4j not available, skipping comprehensive graph connection")
        return

    try:
        from connect_graphs import main as connect_graphs_main

        # Run the full graph connection pipeline (6 steps):
        #   1. connect_graphs_by_document        (LINKED_TO_TEXTUNIT, LINKED_TO_DOCUMENT,
        #                                          CONNECTED_TO_DOCUMENT, LINKED_TO_STRUCTURE,
        #                                          MENTIONS_ELEMENT, NEAR_REGION)
        #   2. connect_structural_and_semantic_nodes (CONNECTS_TO, CONNECTED_TO_REGION,
        #                                              DESCRIBES_STRUCTURE, SAME_CONTENT_AS)
        #   3. create_semantic_links             (semantic_link)
        #   4. compute_bridge_edge_weights       (weight on all bridge edges)
        #   5. create_aggregated_community_region_links (AGGREGATED_REGIONS)
        #   6. get_connection_statistics         (summary stats)
        connect_graphs_main(document_id=file_hash)

    except Exception as e:
        logger.error(f"Error in comprehensive graph connection for document {file_hash}: {e}")
        raise


def index_document_by_hash(file_hash: str, client=None) -> bool:
    """
    Index a document by its hash if it exists in MinIO

    Args:
        file_hash: Hash of the PDF file
        client: Optional QdrantClientWrapper instance. If not provided, uses default client.

    Returns:
        True if successfully indexed, False otherwise
    """
    try:
        global qdrant_client

        # Use provided client or default
        q_client = client or qdrant_client

        # Temporarily set the global client's collection_name for compute_embeddings_for_elements
        original_collection_name = qdrant_client.collection_name
        if client and client != qdrant_client:
            qdrant_client.collection_name = client.collection_name

        # Check if mineru result exists in MinIO
        mineru_result_key = f"mineru_results/{file_hash}"

        # Try to find the actual mineru result path
        existing_objects = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=f"mineru_results/"
        )

        # Find the matching mineru result for this file_hash
        matching_mineru_path = None
        for obj_path in existing_objects:
            if obj_path.startswith(f"mineru_results/{file_hash}_"):
                matching_mineru_path = obj_path
                break

        if not matching_mineru_path:
            logger.error(f"No MinerU result found for file_hash: {file_hash}")
            return False

        # Download mineru result
        mineru_result_json = minio_client.get_object(
            bucket_name=minio_client.bucket_name,
            object_name=matching_mineru_path
        )
        mineru_result = json.loads(mineru_result_json.decode('utf-8'))

        # Extract elements from MinerU result
        elements = []
        if "results" in mineru_result and "result" in mineru_result["results"]:
            results_data = mineru_result["results"]["result"]["results"]
            if "content_list" in results_data:
                elements.extend(results_data["content_list"])

        if not elements:
            logger.error(f"No elements found in MinerU result for file_hash: {file_hash}")
            return False

        # Compute embeddings for elements
        embeddings_count = compute_embeddings_for_elements(elements, file_hash)
        logger.info(f"Indexed {embeddings_count} elements for file_hash: {file_hash}")

        # Restore original collection name
        qdrant_client.collection_name = original_collection_name

        return embeddings_count > 0

    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return False


@app.get("/ask-document", response_class=HTMLResponse)
async def ask_document_page():
    """Serve the Ask Document web interface"""
    static_path = Path(__file__).parent.parent / "static"
    html_file = static_path / "ask-document.html"

    if html_file.exists():
        return FileResponse(html_file)
    else:
        raise HTTPException(
            status_code=404,
            detail="Web interface not found"
        )


@app.post("/ask-document", response_model=QuestionResponse)
async def ask_document(request: QuestionRequest,
                       use_semantic_graph: bool = True,
                       use_structured_graph: bool = True,
                       use_iterative_search: bool = False,
                       use_question_decomposition: bool = False):
    """
    Ask a question about a specific document by file_hash.
    If the document is not indexed, it will be indexed first.

    All retrieval/context/generation strategies are controllable via
    QuestionRequest fields or query parameters:

      Retrieval:
        use_reranker           — API reranker post-search reordering
        use_mmr_reranker       — MMR diversity reranking
        mmr_lambda             — relevance-diversity tradeoff (0-1)
        mmr_min_relevance      — MMR minimum relevance threshold
        use_question_decomposition — decompose complex questions

      Context enrichment:
        use_semantic_graph     — entity + community enrichment
        use_structured_graph   — structural ORDER walk + cross-graph bridge

      Generation:
        use_iterative_search   — 2-round feedback-driven retrieval
        use_llm                — generate final answer via LLM
    """
    from qdrant_client.http import models

    file_hash = request.file_hash
    question = request.question
    limit = request.limit
    collection_name = request.collection_name
    use_llm = request.use_llm
    use_reranker = request.use_reranker
    use_mmr_reranker = getattr(request, 'use_mmr_reranker', False)
    mmr_lambda = getattr(request, 'mmr_lambda', 0.7)
    mmr_min_relevance = getattr(request, 'mmr_min_relevance', 0.0)
    use_question_decomposition = (
        use_question_decomposition or
        getattr(request, 'use_question_decomposition', False)
    )
    answer_format = getattr(request, 'answer_format', None)

    # Use specified collection or default
    client = get_qdrant_client(collection_name=collection_name) if collection_name else qdrant_client
    actual_collection_name = collection_name or qdrant_client.collection_name

    # Check if document is already indexed
    is_indexed = check_document_indexed(file_hash, client)

    if not is_indexed:
        logger.info(f"Document {file_hash} is not indexed in collection '{actual_collection_name}', attempting to index it...")
        index_success = index_document_by_hash(file_hash, client)

        if not index_success:
            return QuestionResponse(
                status="error",
                message=f"Failed to index document with hash {file_hash}. Document may not exist in MinIO.",
                file_hash=file_hash,
                question=question,
                answers=[],
                indexed=False,
                collection_name=actual_collection_name
            )

        is_indexed = True
        logger.info(f"Document {file_hash} successfully indexed in collection '{actual_collection_name}'")

    try:
        server_start = time.time()

        # C6: Multi-hop question decomposition — break complex questions
        # into sub-questions and search each independently.
        # Controlled by request flag OR global settings flag.
        decomposition_enabled = (
            use_question_decomposition or
            getattr(settings.llm, "QUESTION_DECOMPOSITION_ENABLED", False)
        )
        sub_questions = None
        if decomposition_enabled and question:
            try:
                sub_questions = decompose_question(
                    question,
                    llm_client,
                    model_name=settings.llm.LLM_MODEL_NAME,
                )
                if len(sub_questions) > 1:
                    logger.info(
                        "Multi-hop: decomposed into %d sub-questions",
                        len(sub_questions),
                    )
                else:
                    sub_questions = None  # single-hop, use original flow
            except Exception:
                logger.debug(
                    "Question decomposition skipped due to error", exc_info=True
                )
                sub_questions = None

        # Create filter for file_hash
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash)
                )
            ]
        )

        # Search for relevant chunks (retrieve more candidates for reranking)
        # Phase 2: aggressive retrieval — 5x multiplier + floor of 50-60
        # to reduce false N/A caused by missing relevant chunks at the retrieval stage.
        # Reranker / MMR will later filter to the final top-k.
        rerank_top_n = settings.reranker.RERANKER_TOP_N if hasattr(settings.reranker, 'RERANKER_TOP_N') else limit
        if use_reranker:
            search_limit = max(limit * 5, rerank_top_n)
        else:
            search_limit = max(limit * 5, 50)

        # --- Run search (multi-query if decomposed, single otherwise) ---
        if sub_questions and len(sub_questions) > 1:
            # Multi-hop: search each sub-question independently
            all_results = []
            for sq in sub_questions:
                try:
                    sq_embedding = emb_client.get_text_embedding(sq)
                except Exception:
                    logger.debug(
                        "Failed to embed sub-question, skipping: %s", sq
                    )
                    continue
                sq_results = client.search(
                    query_vector=sq_embedding,
                    limit=search_limit,
                    filter_condition=filter_condition,
                )
                formatted = []
                for point in sq_results.points:
                    formatted.append({
                        "text": point.payload.get("text", ""),
                        "score": point.score,
                        "payload": point.payload,
                    })
                all_results.append(formatted)

            if all_results:
                merged = merge_search_results(all_results, search_limit)
                logger.info(
                    "Merged %d results from %d sub-queries into %d unique chunks",
                    sum(len(r) for r in all_results),
                    len(all_results),
                    len(merged),
                )
                # Convert merged dicts to Qdrant-style points for downstream
                from types import SimpleNamespace as _SN

                class _FakePoint:
                    def __init__(self, d):
                        self.payload = d.get("payload", {})
                        self.score = d.get("score", 0)

                search_results = type(
                    "SearchResults", (), {"points": [_FakePoint(m) for m in merged]}
                )()
            else:
                # Fallback to standard search (C7: LRU-cached embedding)
                question_embedding = _cached_question_embedding(question)
                search_results = client.search(
                    query_vector=question_embedding,
                    limit=search_limit,
                    filter_condition=filter_condition,
                )
        else:
            # Standard single-query search (C7: LRU-cached embedding)
            question_embedding = _cached_question_embedding(question)
            search_results = client.search(
                query_vector=question_embedding,
                limit=search_limit,
                filter_condition=filter_condition,
            )

        # Format results
        answers = []
        documents_to_rerank = []
        search_results_map = {}

        # Initialize Neo4j service for context enrichment (if available)
        neo4j_service = None
        if NEO4J_AVAILABLE:
            try:
                neo4j_service = DocumentIndexService()
                logger.info("Neo4j service initialized for context enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j service: {e}")
                neo4j_service = None

        # Initialize Semantic Graph Manager for entity/community enrichment
        semantic_manager = None
        if SEMANTIC_GRAPH_AVAILABLE and use_semantic_graph:
            try:
                sem_config = ManagerConfig(
                    uri=f"neo4j://{os.environ.get('URL', 'localhost:7687')}",
                    user=os.environ.get('USER_NEO4J', 'neo4j'),
                    password=os.environ.get('PASSWORD', ''),
                    name_db=os.environ.get('NAME_DB', 'neo4j')
                )
                semantic_manager = SemanticManager(sem_config)
                logger.info("Semantic Graph Manager initialized for context enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize Semantic Graph Manager: {e}")
                semantic_manager = None

        for idx, result in enumerate(search_results.points):
            payload = result.payload
            element_type = payload.get("element_type", "")
            original_element = payload.get("original_element", {})
            text = payload.get("text", "")
            message = Message()
            answer = {
                "text": text,
                "score": result.score,
                "element_type": element_type,
                "element_index": payload.get("element_index", 0),
                "page_idx": original_element.get("page_idx", 0) if original_element else 0,
                "img_path": original_element.get("img_path", None),  # Store img_path for images and tables
                "image_base64": None,  # Will be populated for image and table elements
                "bbox": original_element.get("bbox", None),  # Store bbox for visualization
                "neo4j_context": None,  # Will store related context from Neo4j
                "is_related_context": False  # Flag to indicate if this is a related context answer
            }

            # Enrich with Neo4j context for image/table related elements
            if neo4j_service and element_type in ("image", "table", "image_caption", "image_footnote", "table_caption",
                                                  "table_footnote"):
            #if neo4j_service and element_type in ("image_caption", "image_footnote"):
                try:
                    related_context = neo4j_service.get_related_context(file_hash, element_type, text)
                    if related_context and (related_context.get("parent_element") or related_context.get("sibling_captions") or related_context.get("sibling_footnotes")):
                        answer["neo4j_context"] = related_context
                        logger.debug(f"Added Neo4j context for {element_type}: {related_context}")

                        # Create separate answers from Neo4j context
                        # Add parent element as a separate answer if available
                        parent_element = related_context.get("parent_element")
                        if parent_element:
                            parent_answer = {
                                "text": parent_element.get("text", ""),
                                "score": result.score * 0.9,  # Slightly lower score as it's related context
                                "element_type": parent_element.get("type", "unknown"),
                                "element_index": payload.get("element_index", 0),
                                "page_idx": original_element.get("page_idx", 0) if original_element else 0,
                                "img_path": parent_element.get("image", None),
                                "image_base64": None,
                                "bbox": parent_element.get("bbox", None),
                                "neo4j_context": None,
                                "is_related_context": True,
                                "related_to_element_type": element_type
                            }

                            # Download image for parent element if it's an image/table
                            if parent_element.get("type") in ("image", "table") and parent_element.get("image"):
                                try:
                                    image_data = minio_client.get_object(
                                        bucket_name=minio_client.bucket_name,
                                        object_name=parent_element["image"]
                                    )
                                    parent_answer["image_base64"] = base64.b64encode(image_data).decode('utf-8')
                                except Exception as e:
                                    logger.warning(f"Failed to download parent image {parent_element.get('image')}: {e}")

                            answers.append(parent_answer)
                            logger.debug(f"Added parent element answer: {parent_element.get('type')}")

                        # Add sibling captions as separate answers
                        for caption in related_context.get("sibling_captions", []):
                            caption_text = caption.get("text", "")
                            if caption_text:
                                # Get image from parent element if available
                                img_path_for_caption = None
                                image_base64_for_caption = None

                                # Try to get image from the original element or parent element
                                if element_type in ("image_caption", "image_footnote"):
                                    # For caption/footnote, get image from parent_element
                                    if parent_element and parent_element.get("type") == "image":
                                        img_path_for_caption = parent_element.get("image")
                                elif element_type in ("table_caption", "table_footnote"):
                                    # For table caption/footnote, get image from parent_element
                                    if parent_element and parent_element.get("type") == "table":
                                        img_path_for_caption = parent_element.get("image")
                                elif element_type == "image":
                                    # For image element, use its own img_path
                                    img_path_for_caption = original_element.get("img_path")
                                elif element_type == "table":
                                    # For table element, use its own img_path
                                    img_path_for_caption = original_element.get("img_path")

                                # Download image if we have a path
                                if img_path_for_caption:
                                    try:
                                        image_data = minio_client.get_object(
                                            bucket_name=minio_client.bucket_name,
                                            object_name=img_path_for_caption
                                        )
                                        image_base64_for_caption = base64.b64encode(image_data).decode('utf-8')
                                    except Exception as e:
                                        logger.warning(f"Failed to download image {img_path_for_caption} for caption: {e}")

                                caption_answer = {
                                    "text": caption_text,
                                    "score": result.score * 0.85,
                                    "element_type": "image_caption" if element_type.startswith("image") else "table_caption",
                                    "element_index": payload.get("element_index", 0),
                                    "page_idx": original_element.get("page_idx", 0) if original_element else 0,
                                    "img_path": img_path_for_caption,
                                    "image_base64": image_base64_for_caption,
                                    "bbox": original_element.get("bbox", None),
                                    "neo4j_context": None,
                                    "is_related_context": True,
                                    "related_to_element_type": element_type
                                }
                                answers.append(caption_answer)
                                logger.debug(f"Added sibling caption answer: {caption_text[:50]}...")

                        # Add sibling footnotes as separate answers
                        for footnote in related_context.get("sibling_footnotes", []):
                            footnote_text = footnote.get("text", "")
                            if footnote_text:
                                # Get image from parent element if available
                                img_path_for_footnote = None
                                image_base64_for_footnote = None

                                # Try to get image from the original element or parent element
                                if element_type in ("image_caption", "image_footnote"):
                                    # For caption/footnote, get image from parent_element
                                    if parent_element and parent_element.get("type") == "image":
                                        img_path_for_footnote = parent_element.get("image")
                                elif element_type in ("table_caption", "table_footnote"):
                                    # For table caption/footnote, get image from parent_element
                                    if parent_element and parent_element.get("type") == "table":
                                        img_path_for_footnote = parent_element.get("image")
                                elif element_type == "image":
                                    # For image element, use its own img_path
                                    img_path_for_footnote = original_element.get("img_path")
                                elif element_type == "table":
                                    # For table element, use its own img_path
                                    img_path_for_footnote = original_element.get("img_path")

                                # Download image if we have a path
                                if img_path_for_footnote:
                                    try:
                                        image_data = minio_client.get_object(
                                            bucket_name=minio_client.bucket_name,
                                            object_name=img_path_for_footnote
                                        )
                                        image_base64_for_footnote = base64.b64encode(image_data).decode('utf-8')
                                    except Exception as e:
                                        logger.warning(f"Failed to download image {img_path_for_footnote} for footnote: {e}")

                                footnote_answer = {
                                    "text": footnote_text,
                                    "score": result.score * 0.85,
                                    "element_type": "image_footnote" if element_type.startswith("image") else "table_footnote",
                                    "element_index": payload.get("element_index", 0),
                                    "page_idx": original_element.get("page_idx", 0) if original_element else 0,
                                    "img_path": img_path_for_footnote,
                                    "image_base64": image_base64_for_footnote,
                                    "bbox": original_element.get("bbox", None),
                                    "neo4j_context": None,
                                    "is_related_context": True,
                                    "related_to_element_type": element_type
                                }
                                answers.append(footnote_answer)
                                logger.debug(f"Added sibling footnote answer: {footnote_text[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to get Neo4j context for {element_type}: {e}")

            # Download image data for image and table elements, or for caption/footnote with image reference
            if element_type in ("image") and answer["img_path"]:
                try:
                    image_data = minio_client.get_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=answer["img_path"]
                    )
                    image_captions = original_element.get("image_caption", [])
                    image_footnotes = original_element.get("image_footnote", [])
                    caption_text = " ".join(image_captions) if image_captions else ""
                    footnote_text = " ".join(image_footnotes) if image_footnotes else ""
                    text = f'Figure | Image:'
                    if caption_text:
                        text = f'{text} | {caption_text}'
                    if footnote_text:
                        text = f'{text} | {footnote_text}'
                    answer["image_base64"] = base64.b64encode(image_data).decode('utf-8')
                    message.add_img_content_base64(answer["image_base64"])
                    message.add_text_content(text)
                    message.set_type('image/text')

                except Exception as e:
                    logger.error(f"Failed to download image {answer['img_path']}: {e}")
            elif element_type in ("image_caption", "image_footnote", "table_caption", "table_footnote") and answer.get("img_path"):
                # For caption/footnote elements that have an associated image
                try:
                    image_data = minio_client.get_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=answer["img_path"]
                    )
                    answer["image_base64"] = base64.b64encode(image_data).decode('utf-8')
                    message.add_img_content_base64(answer["image_base64"])
                    message.add_text_content(text)
                    message.set_type('image/text')
                except Exception as e:
                    logger.warning(f"Failed to download image {answer['img_path']} for {element_type}: {e}")
                    message.add_text_content(text)
            else:
                message.add_text_content(text)

            answers.append(answer)
            documents_to_rerank.append(message)
            search_results_map[idx] = answer

        # Apply reranking if enabled and we have documents
        use_reranker = getattr(request, 'use_reranker', False)
        if use_reranker and documents_to_rerank:
            try:
                rerank_result = reranker_client.rerank(
                    query_text=question,
                    messages=documents_to_rerank
                )

                # Reorder answers based on reranker scores — take top `limit` results
                rerank_messages = sorted(
                    rerank_result.messages, reverse=True, key=lambda x: x.score
                )
                reranked_answers = []
                for res in rerank_messages[:limit]:
                    answer_copy = answers[res.message_id].copy()
                    answer_copy["reranker_score"] = res.score
                    answer_copy["original_score"] = answer_copy["score"]
                    answer_copy["score"] = res.score  # Use reranker score as primary
                    reranked_answers.append(answer_copy)

                answers = reranked_answers
                logger.info(f"Reranking applied: {len(answers)} results reordered")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original search results.")

        # Apply MMR (Maximal Marginal Relevance) diversity reranking if enabled
        if use_mmr_reranker and answers:
            try:
                # Collect text blocks and embeddings from already-loaded answers
                mmr_items: list[str] = []
                mmr_embeddings: list[list[float]] = []
                mmr_answer_indices: list[int] = []

                for idx, ans in enumerate(answers):
                    text = ans.get("text", "")
                    # Try to get embedding from payload or compute on-the-fly
                    emb = None
                    payload = ans.get("payload", {})
                    if isinstance(payload, dict):
                        original = payload.get("original_element", {})
                        if isinstance(original, dict):
                            emb = original.get("embedding")
                    if not emb and isinstance(ans.get("score"), (int, float)):
                        # Fallback: use question embedding as proxy (not ideal but safe)
                        pass

                    if text and emb and isinstance(emb, list) and len(emb) > 0:
                        mmr_items.append(text)
                        mmr_embeddings.append(emb)
                        mmr_answer_indices.append(idx)

                if len(mmr_items) >= 2:
                    question_emb = None
                    if question:
                        try:
                            question_emb = emb_client.get_text_embedding(question)
                        except Exception:
                            pass
                    if question_emb:
                        reranked_items, mmr_scores = mmr_rerank_with_threshold(
                            items=mmr_items,
                            item_embeddings=mmr_embeddings,
                            query_embedding=question_emb,
                            lambda_param=mmr_lambda,
                            top_k=min(len(mmr_items), limit),
                            min_relevance=mmr_min_relevance,
                            min_mmr=0.0,
                        )
                        # Rebuild answers in MMR order
                        mmr_map = {text: (score, idx) for text, score, idx
                                   in zip(reranked_items, mmr_scores, mmr_answer_indices)
                                   if text in set(reranked_items)}
                        reranked = []
                        seen = set()
                        for item_text in reranked_items:
                            if item_text in mmr_map and item_text not in seen:
                                score_val, orig_idx = mmr_map[item_text]
                                ans_copy = answers[orig_idx].copy()
                                ans_copy["mmr_score"] = score_val
                                ans_copy["mmr_lambda"] = mmr_lambda
                                ans_copy["original_score"] = ans_copy.get("original_score", ans_copy["score"])
                                ans_copy["score"] = score_val  # Use MMR score as primary
                                reranked.append(ans_copy)
                                seen.add(item_text)
                        if reranked:
                            answers = reranked
                            logger.info(
                                "MMR reranking applied: %d results (λ=%.2f, min_relevance=%.2f)",
                                len(answers), mmr_lambda, mmr_min_relevance
                            )
                else:
                    logger.debug("Not enough items with embeddings for MMR reranking")
            except Exception as e:
                logger.warning("MMR reranking failed: %s. Using original results.", e)

        # Server-side timing checkpoint — end of search/rerank phase
        search_end = time.time()

        # Generate LLM answer if requested
        llm_answer = None

        # Check if iterative retrieval was requested via body field or query param
        iterative_enabled = request.use_iterative_search or use_iterative_search
        context_parts = []
        if use_llm and answers:
            try:
                # Create messages for LLM with proper structure for Qwen3VL-32B
                # System message will be built dynamically later, once we know
                # which enrichment sources produced results — see below.

                # User message with context and question
                user_message = ModelMessageDict(role='user')

                # Build context with text, images, and Neo4j-enriched context
                context_parts = []

                # First, add all images at the beginning for better model attention
                for idx, ans in enumerate(answers):
                    element_type = ans.get("element_type", "")

                    # Add image if available (for image, table, and caption/footnote elements)
                    if element_type in ("image", "table", "image_caption", "image_footnote") and ans.get("image_base64"):
                        user_message.add_img_content_base64(ans["image_base64"])
                        # Add marker for image reference
                        img_ref = f"[ИЗОБРАЖЕНИЕ | CHART | FIGURE | IMAGE {idx+1}: тип={element_type}]"
                        if ans.get("text"):
                            img_ref += f" | {ans['text']}"
                        context_parts.append(img_ref)

                # Build list of region_ids from Qdrant search results
                region_ids = []
                for idx, ans in enumerate(answers):
                    element_index = ans.get("element_index")
                    if element_index is not None:
                        region_ids.append(f"{file_hash}|{element_index}")

                # Add separator before text context
                context_parts.append("--- DOCUMENT CONTEXT ---")

                # Now add all text content with structured formatting
                for idx, ans in enumerate(answers):
                    element_type = ans.get("element_type", "")

                    # Add text content from the main answer
                    if ans.get("text"):
                        text_marker = f"[BLOCK {idx+1}]"
                        if element_type:
                            text_marker += f" (тип: {element_type})"
                        context_parts.append(f"{text_marker}\n{ans['text']}")

                    # Add Neo4j context if available (for caption/footnote enrichment)
                    neo4j_context = ans.get("neo4j_context")
                    if neo4j_context:
                        # For caption/footnote elements, add parent image/table context
                        parent_element = neo4j_context.get("parent_element")
                        if parent_element:
                            parent_text = parent_element.get("text", "")
                            if parent_text:
                                context_parts.append(f"→ СВЯЗАННЫЙ ЭЛЕМЕНТ ({parent_element.get('type', 'element')}): {parent_text}")

                        # For image/table elements, add caption/footnote context
                        sibling_captions = neo4j_context.get("sibling_captions", [])
                        for cap in sibling_captions:
                            cap_text = cap.get("text", "")
                            if cap_text:
                                context_parts.append(f"→ ПОДПИСЬ: {cap_text}")

                        sibling_footnotes = neo4j_context.get("sibling_footnotes", [])
                        for fn in sibling_footnotes:
                            fn_text = fn.get("text", "")
                            if fn_text:
                                context_parts.append(f"→ СНОСКА: {fn_text}")

                # --- Semantic Graph: Entities and Communities ---
                if use_semantic_graph and semantic_manager and region_ids:
                    # Enrich with entities found via semantic graph
                    try:
                        entities = semantic_manager.get_entities_by_region_ids(
                            region_ids, limit=20
                        )
                        if entities:
                            context_parts.append(
                                "--- SEMANTIC GRAPH: ENTITIES ---"
                            )
                            for ent in entities:
                                title = ent.get("title", "")
                                ent_type = ent.get("type", "")
                                description = ent.get("description", "")
                                degree = ent.get("degree", 0)
                                line = f"• [{ent_type}] {title}"
                                if description:
                                    line += f" — {description[:300]}"
                                if degree:
                                    line += f" (связей: {degree})"
                                context_parts.append(line)
                        else:
                            logger.info("No entities found via semantic graph for region_ids")
                    except Exception as e:
                        logger.warning(
                            "Failed to enrich context with semantic graph entities: %s", e
                        )

                    # Enrich with communities found via semantic graph
                    try:
                        communities = semantic_manager.get_communities_by_region_ids(
                            region_ids, limit=10
                        )
                        if communities:
                            context_parts.append(
                                "--- SEMANTIC GRAPH: COMMUNITIES ---"
                            )
                            for comm in communities:
                                title = comm.get("title", "")
                                summary = comm.get("summary", "")
                                size = comm.get("size", 0)
                                rating = comm.get("rating")
                                rating_explanation = comm.get("rating_explanation", "")
                                findings = comm.get("findings", [])
                                full_content = comm.get("full_content", "")

                                line = f"• {title}"
                                if rating is not None:
                                    line += f" [рейтинг: {rating}/10]"
                                if summary:
                                    line += f" — {summary[:500]}"
                                if size:
                                    line += f" (размер: {size})"
                                context_parts.append(line)

                                # Append top findings as sub-items
                                if findings and isinstance(findings, list):
                                    for finding in findings[:3]:
                                        if isinstance(finding, dict):
                                            f_summary = finding.get("summary", "")
                                            f_explanation = finding.get("explanation", "")
                                            if f_summary:
                                                context_parts.append(
                                                    f"  ∟ вывод: {f_summary[:300]}"
                                                )
                                            if f_explanation:
                                                context_parts.append(
                                                    f"    пояснение: {f_explanation[:300]}"
                                                )
                                        elif isinstance(finding, str):
                                            context_parts.append(f"  ∟ {finding[:400]}")

                                # Append rating explanation
                                if rating_explanation:
                                    context_parts.append(
                                        f"  ∟ обоснование рейтинга: {rating_explanation[:300]}"
                                    )

                                # Append full_content if available
                                if full_content:
                                    context_parts.append(
                                        f"  ∟ полное содержание: {full_content[:800]}"
                                    )
                        else:
                            logger.info("No communities found via semantic graph for region_ids")
                    except Exception as e:
                        logger.warning(
                            "Failed to enrich context with semantic graph communities: %s", e
                        )

                # --- Strategy D: Cross-Graph Bridge ---
                # Use the already-connected semantic_manager directly to avoid
                # creating a redundant third Neo4j driver connection.
                if use_structured_graph and region_ids and SEMANTIC_GRAPH_AVAILABLE and semantic_manager:
                    try:
                        bridge_regions = semantic_manager.get_cross_graph_bridge(
                            region_ids, max_regions=15
                        )
                        if bridge_regions:
                            context_parts.append(
                                "--- SEMANTIC GRAPH: REGIONS VIA ENTITIES (CROSS-GRAPH) ---"
                            )
                            for br_data in bridge_regions:
                                br_text = br_data.get("text", "")
                                br_label = br_data.get("label", "Region")
                                br_entity = br_data.get("source_entity", "")
                                if br_text:
                                    src = f" [через: {br_entity}]" if br_entity else ""
                                    context_parts.append(
                                        f"• [{br_label.upper()}]{src} "
                                        f"{br_text[:500]}"
                                    )
                    except Exception as e:
                        logger.warning(
                            "Failed to enrich context via cross-graph bridge: %s", e
                        )

                # --- Strategy A: Structural walk ---
                if use_structured_graph and region_ids and NEO4J_AVAILABLE and neo4j_service:
                    try:
                        order_neighbors = neo4j_service.get_order_neighbors(
                            region_ids, window_size=3, include_parent=True,
                        )
                        if order_neighbors:
                            context_parts.append(
                                "--- STRUCTURAL GRAPH: NEIGHBOUR REGIONS BY READING ORDER ---"
                            )
                            for nb_data in order_neighbors:
                                nb_text = nb_data.get("text", "")
                                nb_label = nb_data.get("label", "")
                                nb_source = nb_data.get("source", "order")
                                if nb_text:
                                    prefix = {
                                        "order": f"[{nb_label.upper() if nb_label else 'REGION'}]",
                                        "parent": f"[PARENT: {nb_label.upper() if nb_label else 'REGION'}]",
                                    }.get(nb_source, f"[{nb_label.upper() if nb_label else 'REGION'}]")
                                    context_parts.append(
                                        f"• {prefix} {nb_text[:500]}"
                                    )
                    except Exception as e:
                        logger.warning(
                            "Failed to enrich context with structural walk: %s", e
                        )

                # --- Compute question embedding once for BFS + embedding search + MMR ---
                question_embedding = None
                if question:
                    try:
                        question_embedding = emb_client.get_text_embedding(question)
                    except Exception:
                        logger.debug("Failed to compute question embedding", exc_info=True)

                # --- Unified BFS Crawler (Strategy E) ---
                if use_semantic_graph and use_structured_graph and region_ids \
                        and semantic_manager and NEO4J_AVAILABLE and neo4j_service:
                    try:
                        seed_regions = []
                        for ans in answers:
                            elem_idx = ans.get("element_index")
                            if elem_idx is not None and ans.get("text"):
                                seed_regions.append({
                                    "region_id": f"{file_hash}|{elem_idx}",
                                    "text": ans.get("text", ""),
                                    "embedding": ans.get("embedding"),
                                })
                        # D3: Dynamic context window — estimate question complexity
                        # and allocate proportionate token budget (2000–8000).
                        def _estimate_query_complexity(q: str) -> float:
                            """Return a 0-1 complexity score for a question."""
                            if not q:
                                return 0.3
                            score = 0.0
                            words = q.split()
                            # Length factor
                            n = len(words)
                            if n < 5:
                                score += 0.1
                            elif n < 10:
                                score += 0.25
                            elif n < 20:
                                score += 0.4
                            else:
                                score += 0.6
                            # Multi-part indicator
                            if "?" in q:
                                score += 0.1 * q.count("?")
                            # Entity/keyword density
                            capitalized = sum(1 for w in words if w[0].isupper())
                            if capitalized > 0:
                                score += min(capitalized * 0.03, 0.2)
                            # Comparison/contrast indicators
                            compare_words = {"vs", "compare", "difference", "versus",
                                             "better", "between", "both", "than", "or"}
                            if any(cw in q.lower() for cw in compare_words):
                                score += 0.1
                            return min(score, 1.0)

                        complexity = _estimate_query_complexity(question or "")
                        # Map 0-1 complexity → 2000–8000 token budget
                        dyn_budget = int(2000 + complexity * 6000)
                        logger.info(
                            "Question complexity=%.2f → dynamic token budget=%d",
                            complexity, dyn_budget,
                        )

                        crawler_config = TraversalConfig()
                        crawler = UnifiedGraphCrawler(
                            manager=semantic_manager,
                            neo4j_service=neo4j_service,
                            config=crawler_config,
                        )
                        bfs_results = crawler.crawl(
                            seed_regions=seed_regions,
                            question_embedding=question_embedding,
                            token_budget=dyn_budget,
                            question_text=question,
                        )
                        if bfs_results:
                            context_parts.append(
                                "--- BFS GRAPH TRAVERSAL: EXPANDED CONTEXT ---"
                            )
                            for bfr in bfs_results:
                                src = bfr.get("source", "unknown")
                                bfr_text = bfr.get("text", "")
                                if bfr_text:
                                    label = {
                                        "order_neighbor": "ORDER-СОСЕД",
                                        "direct_entity": "СУЩНОСТЬ",
                                        "related_1hop": "СВЯЗАННАЯ СУЩНОСТЬ",
                                        "related_2hop": "СВЯЗАННАЯ 2-HOP",
                                        "community_sibling": "СУЩНОСТЬ СООБЩЕСТВА",
                                        "qdrant_match": "QDRANT РЕЗУЛЬТАТ",
                                    }.get(src, src.upper())
                                    context_parts.append(
                                        f"• [{label}] {bfr_text[:500]}"
                                    )
                        logger.info(
                            "BFS crawler: %d regions enriched from %d seeds",
                            len(bfs_results), len(seed_regions),
                        )
                    except Exception as e:
                        logger.warning("BFS crawler enrichment failed: %s", e)

                # --- Semantic Embedding Search: Entities & Communities ---
                if use_semantic_graph and question_embedding:
                    try:
                        import aiohttp
                        import uuid as _uuid
                        from semantic_graph.config import (
                            ENTITY_EMBEDDINGS_COLLECTION,
                            ENTITY_EMBEDDINGS_NAMESPACE,
                            COMMUNITY_EMBEDDINGS_COLLECTION,
                            COMMUNITY_EMBEDDINGS_NAMESPACE,
                            QDRANT_URL as _SG_QDRANT_URL,
                            QDRANT_API_KEY as _SG_QDRANT_API_KEY,
                        )

                        qdrant_headers = {}
                        if _SG_QDRANT_API_KEY:
                            qdrant_headers["api-key"] = _SG_QDRANT_API_KEY

                        async with aiohttp.ClientSession() as session:
                            # Search entity embeddings
                            try:
                                async with session.post(
                                    f"{_SG_QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}/points/search",
                                    json={
                                        "vector": question_embedding,
                                        "limit": 10,
                                        "with_payload": True,
                                        "with_vector": False,
                                    },
                                    headers=qdrant_headers,
                                ) as resp:
                                    if resp.status == 200:
                                        ent_results = (await resp.json()).get("result", [])
                                        if ent_results:
                                            context_parts.append(
                                                "--- SEMANTIC SEARCH: ENTITIES (EMBEDDINGS) ---"
                                            )
                                            for pt in ent_results:
                                                payload = pt.get("payload", {})
                                                title = payload.get("entity_title", "")
                                                etype = payload.get("entity_type", "")
                                                desc = payload.get("description", "")
                                                score = pt.get("score", 0)
                                                line = f"• [{etype}] {title} (score: {score:.3f})"
                                                if desc:
                                                    line += f" — {desc[:250]}"
                                                context_parts.append(line)
                            except Exception:
                                logger.debug("Entity embedding search skipped", exc_info=True)

                            # Search community embeddings
                            try:
                                async with session.post(
                                    f"{_SG_QDRANT_URL}/collections/{COMMUNITY_EMBEDDINGS_COLLECTION}/points/search",
                                    json={
                                        "vector": question_embedding,
                                        "limit": 5,
                                        "with_payload": True,
                                        "with_vector": False,
                                    },
                                    headers=qdrant_headers,
                                ) as resp:
                                    if resp.status == 200:
                                        comm_results = (await resp.json()).get("result", [])
                                        if comm_results:
                                            context_parts.append(
                                                "--- SEMANTIC SEARCH: COMMUNITIES (EMBEDDINGS) ---"
                                            )
                                            for pt in comm_results:
                                                payload = pt.get("payload", {})
                                                ctitle = payload.get("community_title", payload.get("title", ""))
                                                csummary = payload.get("summary", "")
                                                score = pt.get("score", 0)
                                                line = f"• {ctitle} (score: {score:.3f})"
                                                if csummary:
                                                    line += f" — {csummary[:400]}"
                                                context_parts.append(line)
                            except Exception:
                                logger.debug("Community embedding search skipped", exc_info=True)
                    except Exception as e:
                        logger.warning("Semantic embedding search failed: %s", e)

                # Add end marker
                context_parts.append("--- КОНЕЦ КОНТЕКСТА ---")

                # --- MMR Reranking: diversity-aware context selection ---
                if getattr(getattr(settings, 'mmr', None), 'USE_MMR_RERANKING', False) and question_embedding and len(context_parts) > 2:
                    try:
                        # Embed each context part for MMR
                        part_embeddings = []
                        for part in context_parts:
                            part_embeddings.append(emb_client.get_text_embedding(part[:2000]))
                        reranked_parts, _ = mmr_rerank_with_threshold(
                            items=context_parts,
                            item_embeddings=part_embeddings,
                            query_embedding=question_embedding,
                            lambda_param=settings.mmr.MMR_LAMBDA,
                            top_k=settings.mmr.MMR_TOP_K,
                            min_relevance=settings.mmr.MMR_MIN_RELEVANCE,
                        )
                        if reranked_parts:
                            logger.info(
                                "MMR reranking: %d -> %d context blocks (lambda=%.2f)",
                                len(context_parts), len(reranked_parts), settings.mmr.MMR_LAMBDA,
                            )
                            context_parts = reranked_parts
                            # Ensure end marker is present
                            if "--- КОНЕЦ КОНТЕКСТА ---" not in context_parts:
                                context_parts.append("--- КОНЕЦ КОНТЕКСТА ---")
                    except Exception as e:
                        logger.warning("MMR reranking failed, using original: %s", e)

                # --- Deduplicate context blocks & enforce per-source token budgets ---
                MAX_CONTEXT_CHARS = 12000

                # Map section header → source budget fraction (sum ≤ 1.0)
                _SECTION_SOURCE: dict[str, str] = {
                    "--- DOCUMENT CONTEXT ---": "qdrant",
                    "--- SEMANTIC GRAPH: ENTITIES ---": "entities",
                    "--- SEMANTIC GRAPH: COMMUNITIES ---": "communities",
                    "--- SEMANTIC GRAPH: REGIONS VIA ENTITIES (CROSS-GRAPH) ---": "cross_graph",
                    "--- STRUCTURAL GRAPH: NEIGHBOUR REGIONS BY READING ORDER ---": "order_neighbors",
                    "--- BFS GRAPH TRAVERSAL: EXPANDED CONTEXT ---": "bfs_crawler",
                    "--- SEMANTIC SEARCH: ENTITIES (EMBEDDINGS) ---": "embedding_search",
                    "--- SEMANTIC SEARCH: COMMUNITIES (EMBEDDINGS) ---": "embedding_search",
                }
                _SOURCE_BUDGET_FRAC: dict[str, float] = {
                    "qdrant": 0.30,
                    "entities": 0.10,
                    "communities": 0.15,
                    "cross_graph": 0.07,
                    "order_neighbors": 0.05,
                    "bfs_crawler": 0.15,
                    "embedding_search": 0.08,
                }

                seen_texts: set = set()
                deduped_parts = []
                total_chars = 0
                source_chars: dict[str, int] = {}
                current_source: str | None = None

                for part in context_parts:
                    # Detect section header to switch source
                    if part in _SECTION_SOURCE:
                        current_source = _SECTION_SOURCE[part]
                        if current_source not in source_chars:
                            source_chars[current_source] = 0
                        # Always include headers (they serve as section separators
                        # and are small); they don't count toward source budget.
                        deduped_parts.append(part)
                        total_chars += len(part)
                        continue

                    # Skip "end marker"
                    if part == "--- КОНЕЦ КОНТЕКСТА ---":
                        deduped_parts.append(part)
                        total_chars += len(part)
                        continue

                    # Dedup
                    key = part[:120].replace(" ", "").replace("\n", "").lower()
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)

                    # Global cap
                    if total_chars + len(part) > MAX_CONTEXT_CHARS:
                        remaining = MAX_CONTEXT_CHARS - total_chars
                        if remaining >= 60:
                            deduped_parts.append(
                                part[:remaining] + "... [TRUNCATED]"
                            )
                            total_chars = MAX_CONTEXT_CHARS
                        break

                    # Per-source cap (skip if source has exhausted its budget)
                    src = current_source or "qdrant"
                    budget = int(MAX_CONTEXT_CHARS * _SOURCE_BUDGET_FRAC.get(src, 0.08))
                    if source_chars.get(src, 0) + len(part) > budget:
                        # Source budget exhausted — skip this part
                        continue

                    deduped_parts.append(part)
                    total_chars += len(part)
                    source_chars[src] = source_chars.get(src, 0) + len(part)

                if len(deduped_parts) < len(context_parts):
                    logger.info(
                        "Context dedup/budget: %d -> %d parts, ~%d chars "
                        "(limit %d, source_usage=%s)",
                        len(context_parts), len(deduped_parts), total_chars,
                        MAX_CONTEXT_CHARS, source_chars,
                    )

                context_parts = deduped_parts

                # --- C5: Feedback-driven iterative retrieval ---
                # Runs AFTER graph enrichment so the first round already sees
                # structural + semantic + BFS + embedding results instead of
                # just raw Qdrant chunks.
                iterative_used = False
                if iterative_enabled and use_llm and context_parts:
                    try:
                        # Build refinement search function for round 2
                        async def _refinement_search(refinement_query: str) -> dict:
                            try:
                                ref_embedding = emb_client.get_text_embedding(refinement_query)
                            except Exception:
                                return {"context": []}
                            ref_results = client.search(
                                query_vector=ref_embedding,
                                limit=max(search_limit, limit),
                                filter_condition=filter_condition,
                            )
                            context = []
                            for point in ref_results.points:
                                text = point.payload.get("text", "")
                                if text:
                                    context.append(text)
                            return {"context": context}

                        # First-round context = graph-enriched context_parts
                        first_context = context_parts

                        iter_result = await iterative_retrieval(
                            question=question,
                            llm_client=llm_client,
                            search_fn=_refinement_search,
                            max_rounds=2,
                            model_name=settings.llm.LLM_MODEL_NAME,
                        )

                        llm_answer = iter_result.get("answer", "")
                        total_rounds = iter_result.get("total_rounds", 1)
                        # Sanitize iterative answer: normalize "Fail to answer" variants
                        _normalized = llm_answer.strip().lower()
                        _refusal_patterns = ("fail to answer", "unable to answer", "cannot answer",
                                             "cannot provide", "no answer", "i don't know")
                        if _normalized in _refusal_patterns or any(
                            _normalized.startswith(p) for p in _refusal_patterns
                        ):
                            llm_answer = "Not answerable"
                        logger.info(
                            "Iterative retrieval (with graph enrichment): "
                            "%d rounds, %d context blocks",
                            total_rounds, len(context_parts),
                        )
                        iterative_used = True
                    except Exception as e:
                        logger.warning(
                            "Iterative retrieval failed: %s. "
                            "Falling back to standard LLM.", e
                        )
                        iterative_used = False

                user_prompt = ""  # initialised for logging use when LLM path is skipped
                context = ""      # initialised for logging use when LLM path is skipped
                if iterative_used:
                    # Iterative retrieval already called the LLM and produced
                    # llm_answer — skip the standard LLM call below.
                    pass
                elif not context_parts:
                    # No context to send to LLM
                    pass
                else:
                    # Standard LLM path: compose user message from context_parts
                    context = "\n\n".join(context_parts)
                    user_message.add_text_content(context)

                    # Formulate clear question with instructions
                    user_prompt = f"""
--- USER QUESTION ---
{question}

--- INSTRUCTIONS ---
Analyze the images and text context provided above.
Answer the question using ONLY information from the context.
If images/tables in the context are relevant to the question, incorporate them in your answer.
Be concise: start directly with the answer value, then optionally add brief supporting evidence.

ANSWER:"""
                    user_message.add_text_content(user_prompt)

                # ===== LOG ALL CONTEXT BLOCKS =====
                logger.info("=" * 80)
                logger.info(f"LLM CONTEXT BLOCKS for question: {question}")
                logger.info(f"Number of answers/chunks: {len(answers)}")
                logger.info(f"Number of context_parts: {len(context_parts)}")
                logger.info("-" * 40)

                # --- Dynamic system prompt: describe which enrichment sources produced results ---
                _SECTION_DESCRIPTIONS: dict[str, str] = {
                    "--- DOCUMENT CONTEXT ---":
                        "document text blocks with element-type markup",
                    "--- SEMANTIC GRAPH: ENTITIES ---":
                        "entities from the semantic knowledge graph (people, organizations, "
                        "places, events) extracted from the document",
                    "--- SEMANTIC GRAPH: COMMUNITIES ---":
                        "communities from the semantic graph with findings and confidence ratings",
                    "--- SEMANTIC GRAPH: REGIONS VIA ENTITIES (CROSS-GRAPH) ---":
                        "document regions linked through entities (cross-graph bridge)",
                    "--- STRUCTURAL GRAPH: NEIGHBOUR REGIONS BY READING ORDER ---":
                        "structurally adjacent regions (previous/next document element)",
                    "--- BFS GRAPH TRAVERSAL: EXPANDED CONTEXT ---":
                        "expanded context via BFS graph traversal "
                        "(related entities, 1-hop/2-hop links, communities)",
                    "--- SEMANTIC SEARCH: ENTITIES (EMBEDDINGS) ---":
                        "entities found through embedding-based semantic similarity search",
                    "--- SEMANTIC SEARCH: COMMUNITIES (EMBEDDINGS) ---":
                        "communities found through embedding-based semantic similarity search",
                }
                context_sources: list[str] = []
                for part in context_parts:
                    desc = _SECTION_DESCRIPTIONS.get(part)
                    if desc and desc not in context_sources:
                        context_sources.append(desc)

                source_list = "\n".join(f"  - {s}" for s in context_sources) if context_sources \
                    else "  - document text blocks"

                # Build answer format hint based on request
                format_hint = ""
                if answer_format:
                    fmt_lower = answer_format.strip().lower()
                    if fmt_lower == 'int':
                        format_hint = "8a. The expected answer is a single INTEGER — output ONLY the number"
                    elif fmt_lower == 'float':
                        format_hint = "8a. The expected answer is a single FLOAT/DECIMAL number — output ONLY the value"
                    elif fmt_lower == 'list':
                        format_hint = (
                            "8a. The expected answer is a LIST of items separated by commas — "
                            "output each item on the first line, comma-separated"
                        )
                    elif fmt_lower == 'str':
                        format_hint = "8a. The expected answer is a short STRING — output ONLY the answer text"
                    elif fmt_lower == 'none':
                        format_hint = "8a. Only answer if clearly found; otherwise say 'Not answerable'"

                system_prompt = f"""You are a document analysis assistant. Answer user questions using ONLY the provided context.

CONTEXT SOURCES:
{source_list}

ANSWER RULES:
1. Use ONLY information from the provided context. Do not use external knowledge.
2. If the answer is NOT found in the context or you cannot give a precise answer, respond strictly: "Not answerable"
3. Any refusal variant (e.g. "Fail to answer", "Unable to answer", "I don't know") MUST be replaced with "Not answerable"
4. For images, tables, and charts, carefully analyze visual information together with captions
5. Be precise and concise — give the answer value first, then optional brief evidence
6. If context contains contradictory information, note it
7. Always answer in ENGLISH regardless of the question language
8. Pay attention to confidence ratings in communities — higher rating means more reliable findings
{format_hint}
9. Format: start your response with the tag [FINAL_ANSWER]: followed by the direct answer value on its own line. Then optionally add supporting reasoning on subsequent lines.

NUMERICAL ACCURACY RULES:
10. When extracting numbers from tables, ALWAYS double-check the row AND column labels. The number must correspond to the EXACT intersection of the question's row and column. For example, if asked "What was Revenue in FY2023?", find the row labeled "Revenue" (or its equivalent) AND the column labeled "FY2023" — then report the value at their intersection.
11. Report the EXACT number as it appears in the context. Do not round, approximate, or convert units unless the question explicitly asks for it. If the table says "17,564", report "17564" (for Int) or vice versa as appropriate.
12. Before finalizing a numerical answer, scan the surrounding context for another occurrence of the same metric — if two numbers differ significantly (e.g., more than 20% apart), state the discrepancy and report the one that best matches the question's scope.
13. For financial documents: pay close attention to whether a number is in millions, billions, or raw units. Check table headers and footnotes for units and multipliers.

FEW-SHOT EXAMPLES (numerical extraction from tables):

Example 1 — Financial table row/column identification:
Context: "Table: Revenue by Year (in millions). Row 'Total Revenue', Columns: 'FY2021'=18078, 'FY2022'=19500, 'FY2023'=21000"
Question: "What was the total revenue COSTCO FY2021?"
Correct answer: 18078
Why: Row='Total Revenue', Column='FY2021', cell value=18078. Not FY2022 or FY2023.

Example 2 — Avoiding row confusion in financial statements:
Context: "Balance Sheet Data. Row 'Common Equity', FY2021 column shows 18078. Row 'Long-term Debt', FY2021 column shows 10314."
Question: "Common equity COSTCO FY2021"
Correct answer: 18078
Why: Identified row 'Common Equity' (not 'Long-term Debt'), column 'FY2021', value=18078. The number 10314 belongs to a DIFFERENT row — do not report it.

Example 3 — Percentage and ratio values:
Context: "Financial Ratios table: 'Total debt to total assets' = 0.192 (19.2%). 'Current ratio' = 0.1264."
Question: "What is the total debt to total assets ratio?"
Correct answer: 0.192
Why: Row 'Total debt to total assets' has value 0.192, NOT 0.1264 (which is 'Current ratio' — a completely different metric)."""

                system_message = ModelMessageDict(role='system')
                system_message.add_text_content(system_prompt)

                logger.info("SYSTEM PROMPT:")
                logger.info(system_prompt[:500] + ("..." if len(system_prompt) > 500 else ""))
                logger.info("-" * 40)
                logger.info("USER MESSAGE CONTEXT BLOCKS:")
                for i, part in enumerate(context_parts):
                    if len(part) > 1000:
                        logger.info(f"  Block [{i}]: {part[:1000]}... [TRUNCATED, total len={len(part)}]")
                    else:
                        logger.info(f"  Block [{i}]: {part}")
                logger.info("-" * 40)
                logger.info("QUESTION + INSTRUCTION:")
                logger.info(user_prompt[:1000] + ("..." if len(user_prompt) > 1000 else ""))
                logger.debug("FULL RAW CONTEXT (debug level):\n%s", context if context_parts else "(empty)")
                logger.info("=" * 80)
                # ===== END LOG =====

                # Call LLM (skip if iterative retrieval already produced an answer)
                enrichment_end = generation_start = generation_end = time.time()
                if not iterative_used:
                    # Server-side timing — end of enrichment/context assembly
                    enrichment_end = time.time()
                    generation_start = time.time()

                    success, llm_responses = llm_client.send_message(
                        messages=[system_message, user_message],
                        max_tokens=4096,
                        temperature=0.1,
                        top_p=0.95
                    )
                    generation_end = time.time()

                    if success and llm_responses:
                        llm_answer = llm_responses[0]
                        # Sanitize: normalize "Fail to answer" and similar refusal variants
                        _normalized = llm_answer.strip().lower()
                        _refusal_patterns = ("fail to answer", "unable to answer", "cannot answer",
                                             "cannot provide", "no answer", "i don't know", "failed to answer")
                        if _normalized in _refusal_patterns or any(
                            _normalized.startswith(p) for p in _refusal_patterns
                        ):
                            llm_answer = "Not answerable"
                        # Post-process answer format for structured types (Int/Float/List)
                        if answer_format and answer_format.strip().lower() in ('int', 'float', 'list'):
                            formatted = format_answer(llm_answer, answer_format)
                            if formatted is not None:
                                llm_answer = str(formatted) if not isinstance(formatted, list) \
                                    else ", ".join(str(item) for item in formatted)
                        logger.info(f"LLM answer generated successfully for question: {question}")
                    else:
                        logger.warning(f"LLM failed to generate answer for question: {question}")
            except Exception as e:
                logger.error(f"Error generating LLM answer: {e}")
                llm_answer = f"Error generating LLM answer: {str(e)}"

        # Close Neo4j service connection (after all graph enrichment)
        if neo4j_service:
            try:
                neo4j_service.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j service: {e}")

        # Close Semantic Graph Manager connection (after all semantic enrichment)
        if semantic_manager:
            try:
                semantic_manager.close()
            except Exception as e:
                logger.warning(f"Error closing Semantic Graph Manager: {e}")

        # Unified post-processing: normalize any refusal variant to "Not answerable"
        if llm_answer and isinstance(llm_answer, str):
            _final_norm = llm_answer.strip().lower()
            _refusal = ("fail to answer", "unable to answer", "cannot answer",
                        "cannot provide", "no answer", "i don't know", "not answerable")
            if _final_norm in _refusal or any(_final_norm.startswith(p) for p in _refusal):
                llm_answer = "Not answerable"

        return QuestionResponse(
            status="success",
            message=f"Found {len(answers)} relevant chunks for the question in collection '{actual_collection_name}'",
            file_hash=file_hash,
            question=question,
            answers=answers,
            indexed=is_indexed,
            collection_name=actual_collection_name,
            llm_answer=llm_answer,
            context_blocks=context_parts if use_llm else None,
            response_metadata={
                "search_ms": round((search_end - server_start) * 1000),
                "enrichment_ms": round((enrichment_end - search_end) * 1000),
                "llm_generation_ms": round((generation_end - generation_start) * 1000),
                "total_ms": round((generation_end - server_start) * 1000),
            } if use_llm and answers else {
                "search_ms": round((search_end - server_start) * 1000),
                "total_ms": round((search_end - server_start) * 1000),
            }
        )

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/collections/{collection_name}/files", response_model=UploadedFilesListResponse)
def get_collection_files(collection_name: str):
    """
    Get list of uploaded files for a specific collection based on file_hash in Qdrant points

    Args:
        collection_name: Name of the Qdrant collection

    Returns:
        List of unique files (by file_hash) that have embeddings in this collection
    """
    try:
        from qdrant_client.http import models

        client = get_qdrant_client(collection_name=collection_name)

        # Check if collection exists
        if not client.client.collection_exists(collection_name):
            return UploadedFilesListResponse(
                status="success",
                message=f"Collection '{collection_name}' does not exist or is empty",
                files=[],
                total_count=0
            )

        # Use scroll to get all points and extract unique file_hash values
        seen_hashes = set()
        files = []

        try:
            # Scroll through all points to get unique file hashes
            offset = None
            limit = 100

            while True:
                records, offset = client.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not records:
                    break

                for record in records:
                    payload = record.payload if hasattr(record, 'payload') else {}
                    file_hash = payload.get('file_hash')

                    if file_hash and file_hash not in seen_hashes:
                        seen_hashes.add(file_hash)

                        # Try to get file info from MinIO
                        try:
                            existing_pdfs = minio_client.list_objects(
                                bucket_name=minio_client.bucket_name,
                                prefix=f"pdfs/{file_hash}_"
                            )

                            if existing_pdfs:
                                pdf_path = existing_pdfs[0]
                                parts = pdf_path.split('/')
                                dir_name = parts[1] if len(parts) >= 2 else pdf_path
                                file_name = parts[-1] if len(parts) > 2 else dir_name.split('_', 1)[-1] if '_' in dir_name else dir_name

                                # Get upload date
                                try:
                                    stat = minio_client.client.stat_object(
                                        bucket_name=minio_client.bucket_name,
                                        object_name=pdf_path
                                    )
                                    upload_date = stat.last_modified.isoformat() if stat.last_modified else "unknown"
                                except Exception:
                                    upload_date = "unknown"

                                files.append(UploadedFileInfo(
                                    file_hash=file_hash,
                                    filename=file_name,
                                    upload_date=upload_date
                                ))
                        except Exception as e:
                            logger.warning(f"Could not get file info for hash {file_hash}: {e}")
                            # Still add the hash even if we can't get file info
                            files.append(UploadedFileInfo(
                                file_hash=file_hash,
                                filename=f"unknown_{file_hash[:8]}",
                                upload_date="unknown"
                            ))

                if len(records) < limit:
                    break

            return UploadedFilesListResponse(
                status="success",
                message=f"Found {len(files)} unique files in collection '{collection_name}'",
                files=files,
                total_count=len(files)
            )

        except Exception as e:
            logger.error(f"Error scrolling collection: {e}")
            # Fallback: return empty list if scroll fails
            return UploadedFilesListResponse(
                status="success",
                message=f"Collection '{collection_name}' exists but could not retrieve files: {str(e)}",
                files=[],
                total_count=0
            )

    except Exception as e:
        logger.error(f"Error getting collection files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving collection files: {str(e)}"
        )


@app.get("/pdf/{file_hash}")
async def get_pdf_file(file_hash: str):
    """
    Get PDF file by file_hash for viewing in the browser

    Args:
        file_hash: Hash of the PDF file

    Returns:
        PDF file content with appropriate content type
    """
    try:
        # Find the PDF file in MinIO
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=f"pdfs/{file_hash}"
        )

        if not existing_pdfs:
            raise HTTPException(
                status_code=404,
                detail=f"PDF file with hash {file_hash} not found"
            )

        pdf_path = existing_pdfs[0]

        # Download the PDF file
        pdf_data = minio_client.get_object(
            bucket_name=minio_client.bucket_name,
            object_name=pdf_path
        )

        # Return the PDF file with appropriate headers
        from fastapi.responses import Response
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename=\"{file_hash}.pdf\""
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving PDF file {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving PDF file: {str(e)}"
        )


@app.get("/api/pdf/{file_hash}/info")
async def get_pdf_info(file_hash: str):
    """
    Get PDF file info including available pages and dimensions

    Args:
        file_hash: Hash of the PDF file

    Returns:
        PDF metadata
    """
    try:
        # Find the PDF file in MinIO
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=f"pdfs/{file_hash}"
        )

        if not existing_pdfs:
            raise HTTPException(
                status_code=404,
                detail=f"PDF file with hash {file_hash} not found"
            )

        pdf_path = existing_pdfs[0]

        # Get file stats
        stat = minio_client.client.stat_object(
            bucket_name=minio_client.bucket_name,
            object_name=pdf_path
        )

        return {
            "status": "success",
            "file_hash": file_hash,
            "file_name": pdf_path.split('/')[-1],
            "s3_path": pdf_path,
            "size": stat.size,
            "last_modified": stat.last_modified.isoformat() if stat.last_modified else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF info for {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting PDF info: {str(e)}"
        )

@app.get("/api/pdf/{file_hash}/page/{page_number}")
async def get_pdf_page_with_bbox(
    file_hash: str,
    page_number: int,
    bboxes: Optional[str] = None
):
    """
    Get PDF page as image with optional bbox highlights

    Args:
        file_hash: Hash of the PDF file
        page_number: Page number (0-indexed)
        bboxes: JSON string of bboxes to highlight: [{"bbox": [x1, y1, x2, y2], "color": "#FF0000", "label": "text"}, ...]

    Returns:
        PNG image of the PDF page with highlighted bboxes
    """
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw
    import io
    import json

    try:
        # Find the PDF file in MinIO
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=f"pdfs/{file_hash}"
        )

        if not existing_pdfs:
            raise HTTPException(
                status_code=404,
                detail=f"PDF file with hash {file_hash} not found"
            )

        pdf_path = existing_pdfs[0]

        # Download the PDF file
        pdf_data = minio_client.get_object(
            bucket_name=minio_client.bucket_name,
            object_name=pdf_path
        )

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_data, filetype="pdf")

        # Validate page number
        if page_number < 0 or page_number >= len(doc):
            doc.close()
            raise HTTPException(
                status_code=404,
                detail=f"Page {page_number} not found. PDF has {len(doc)} pages (0-{len(doc)-1})"
            )

        # Render page to image (higher resolution for better quality)
        page = doc[page_number]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Parse bboxes if provided
        bbox_list = []
        if bboxes:
            try:
                bbox_list = json.loads(bboxes)
            except json.JSONDecodeError:
                logger.warning(f"Invalid bboxes JSON: {bboxes}")

        # Draw bboxes on image if provided
        if bbox_list:
            draw = ImageDraw.Draw(img)

            # Get page dimensions for coordinate scaling
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            for bbox_item in bbox_list:
                if not isinstance(bbox_item, dict):
                    continue

                bbox = bbox_item.get("bbox", [])
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox

                # MinerU returns coordinates in normalized format (0-1000)
                # Convert to PDF page coordinates
                scale_x = page_width / 1000.0
                scale_y = page_height / 1000.0

                scaled_x1 = int(x1 * scale_x)
                scaled_y1 = int(y1 * scale_y)
                scaled_x2 = int(x2 * scale_x)
                scaled_y2 = int(y2 * scale_y)

                # Now scale to match the rendered image resolution (2x zoom)
                img_scale_x = pix.width / page_width
                img_scale_y = pix.height / page_height

                scaled_x1 = int(scaled_x1 * img_scale_x)
                scaled_y1 = int(scaled_y1 * img_scale_y)
                scaled_x2 = int(scaled_x2 * img_scale_x)
                scaled_y2 = int(scaled_y2 * img_scale_y)

                # Get color (default to red with semi-transparent fill)
                color = bbox_item.get("color", "#FF0000")
                label = bbox_item.get("label", "")

                # Draw rectangle outline
                draw.rectangle(
                    [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
                    outline=color,
                    width=3
                )

                # Draw semi-transparent fill
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                # Parse color to RGBA with transparency
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16) if len(color) >= 5 else 0
                    b = int(color[5:7], 16) if len(color) >= 7 else 0
                    fill_color = (r, g, b, 50)  # 50/255 transparency
                else:
                    fill_color = (255, 0, 0, 50)  # Default red with transparency

                overlay_draw.rectangle(
                    [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
                    fill=fill_color
                )

                # Composite overlay onto main image
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)

                # Draw label if provided
                if label:
                    # Draw text background
                    text_bbox = draw.textbbox((scaled_x1, scaled_y1 - 20), label)
                    draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
                    draw.text((scaled_x1, scaled_y1 - 20), label, fill=(255, 255, 255, 255))

        doc.close()

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        from fastapi.responses import Response
        return Response(
            content=img_bytes.read(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=\"{file_hash}_page_{page_number}.png\""
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering PDF page {page_number} for {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error rendering PDF page: {str(e)}"
        )


@app.get("/api/pdf/{file_hash}/mineru-bboxes")
async def get_mineru_bboxes(file_hash: str, page_idx: Optional[int] = None):
    """
    Get all bounding boxes from MinerU results for a PDF

    Args:
        file_hash: Hash of the PDF file
        page_idx: Optional page index to filter bboxes

    Returns:
        List of bboxes with metadata
    """
    try:
        # Find mineru result for this file_hash
        existing_objects = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=f"mineru_results/"
        )

        matching_mineru_path = None
        for obj_path in existing_objects:
            if obj_path.startswith(f"mineru_results/{file_hash}_"):
                matching_mineru_path = obj_path
                break

        if not matching_mineru_path:
            raise HTTPException(
                status_code=404,
                detail=f"No MinerU result found for file_hash: {file_hash}"
            )

        # Download mineru result
        mineru_result_json = minio_client.get_object(
            bucket_name=minio_client.bucket_name,
            object_name=matching_mineru_path
        )
        mineru_result = json.loads(mineru_result_json.decode('utf-8'))

        # Extract elements from MinerU result
        elements = []
        if "results" in mineru_result and "result" in mineru_result["results"]:
            results_data = mineru_result["results"]["result"]["results"]
            if "content_list" in results_data:
                elements.extend(results_data["content_list"])

        # Filter by page if specified
        if page_idx is not None:
            elements = [e for e in elements if e.get("page_idx") == page_idx]

        # Format bboxes
        bboxes = []
        for i, element in enumerate(elements):
            if not isinstance(element, dict):
                continue

            bbox = element.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            element_type = element.get("type", "unknown")

            # Handle text elements with text_level == 1 as titles
            if element_type == "text":
                text_level = element.get("text_level")
                if text_level == 1:
                    element_type = "title"

            # Assign colors based on element type
            color_map = {
                "title": "#9b59b6",        # Purple - headings (text_level == 1)
                "text": "#3498db",         # Blue - regular text and equations
                "image": "#27ae60",        # Green - visual image embeddings
                "image_caption": "#2ecc71", # Light Green - image captions
                "image_footnote": "#1abc9c", # Teal - image footnotes
                "table": "#e67e22",        # Orange - table contents
                "table_caption": "#f39c12", # Yellow-Orange - table captions
                "table_footnote": "#d35400", # Dark Orange - table footnotes
                "equation": "#e74c3c",     # Red - equations
                "discarded": "#95a5a6"     # Gray - discarded elements (not indexed)
            }
            color = color_map.get(element_type, "#333333")

            bbox_info = {
                "element_index": i,
                "element_type": element_type,
                "bbox": bbox,
                "page_idx": element.get("page_idx", 0),
                "color": color,
                "label": f"{element_type}_{i}"
            }

            # Add text preview for different element types
            if element_type == "text":
                text = element.get("text", "")
                text_level = element.get("text_level")
                bbox_info["text_preview"] = text[:100] + "..." if len(text) > 100 else text
                if text_level is not None:
                    bbox_info["text_level"] = text_level

            elif element_type == "title":
                text = element.get("text", "")
                bbox_info["text_preview"] = f"📑 Title: {text[:100]}..." if len(text) > 100 else f"📑 Title: {text}"
                bbox_info["is_title"] = True

            # Add image_caption/image_footnote as separate bbox entries for image elements
            elif element_type == "image":
                image_captions = element.get("image_caption", [])
                image_footnotes = element.get("image_footnote", [])

                img_bbox = element.get("bbox")

                # Create main image bbox entry
                preview_parts = []
                caption_text = " ".join(image_captions) if image_captions else ""
                footnote_text = " ".join(image_footnotes) if image_footnotes else ""

                if caption_text:
                    preview_parts.append(f"🖼️ Caption: {caption_text[:50]}")
                if footnote_text:
                    preview_parts.append(f"📝 Footnote: {footnote_text[:50]}")

                bbox_info["text_preview"] = " | ".join(preview_parts) if preview_parts else f"🖼️ Image #{i}"
                bboxes.append(bbox_info)

                # Create separate bbox entry for image_caption if exists
                if image_captions:
                    caption_text = " ".join(image_captions)
                    caption_bbox_info = {
                        "element_index": i,
                        "element_type": "image_caption",
                        "bbox": img_bbox,
                        "page_idx": element.get("page_idx", 0),
                        "color": color_map["image_caption"],
                        "label": caption_text[:100] + "..." if len(caption_text) > 100 else caption_text,
                        "text_preview": f"📷 Caption: {caption_text[:100]}..." if len(caption_text) > 100 else f"📷 Caption: {caption_text}"
                    }
                    bboxes.append(caption_bbox_info)

                # Create separate bbox entry for image_footnote if exists
                if image_footnotes:
                    footnote_text = " ".join(image_footnotes)
                    footnote_bbox_info = {
                        "element_index": i,
                        "element_type": "image_footnote",
                        "bbox": img_bbox,
                        "page_idx": element.get("page_idx", 0),
                        "color": color_map["image_footnote"],
                        "label": footnote_text[:100] + "..." if len(footnote_text) > 100 else footnote_text,
                        "text_preview": f"📝 Footnote: {footnote_text[:100]}..." if len(footnote_text) > 100 else f"📝 Footnote: {footnote_text}"
                    }
                    bboxes.append(footnote_bbox_info)

                continue

            # Add table_caption/table_footnote as separate bbox entries for table elements
            elif element_type == "table":
                table_captions = element.get("table_caption", [])
                table_footnotes = element.get("table_footnote", [])

                table_bbox = element.get("bbox")

                # Create main table bbox entry
                preview_parts = []
                caption_text = " ".join(table_captions) if table_captions else ""
                footnote_text = " ".join(table_footnotes) if table_footnotes else ""

                if caption_text:
                    preview_parts.append(f"📊 Caption: {caption_text[:50]}")
                if footnote_text:
                    preview_parts.append(f"📝 Footnote: {footnote_text[:50]}")

                bbox_info["text_preview"] = " | ".join(preview_parts) if preview_parts else f"📊 Table #{i}"
                bboxes.append(bbox_info)

                # Create separate bbox entry for table_caption if exists
                if table_captions:
                    caption_text = " ".join(table_captions)
                    caption_bbox_info = {
                        "element_index": i,
                        "element_type": "table_caption",
                        "bbox": table_bbox,
                        "page_idx": element.get("page_idx", 0),
                        "color": color_map["table_caption"],
                        "label": caption_text[:100] + "..." if len(caption_text) > 100 else caption_text,
                        "text_preview": f"📋 Caption: {caption_text[:100]}..." if len(caption_text) > 100 else f"📋 Caption: {caption_text}"
                    }
                    bboxes.append(caption_bbox_info)

                # Create separate bbox entry for table_footnote if exists
                if table_footnotes:
                    footnote_text = " ".join(table_footnotes)
                    footnote_bbox_info = {
                        "element_index": i,
                        "element_type": "table_footnote",
                        "bbox": table_bbox,
                        "page_idx": element.get("page_idx", 0),
                        "color": color_map["table_footnote"],
                        "label": footnote_text[:100] + "..." if len(footnote_text) > 100 else footnote_text,
                        "text_preview": f"📝 Footnote: {footnote_text[:100]}..." if len(footnote_text) > 100 else f"📝 Footnote: {footnote_text}"
                    }
                    bboxes.append(footnote_bbox_info)

                continue

            elif element_type == "equation":
                text = element.get("text", "")
                bbox_info["text_preview"] = f"∫ Equation: {text[:100]}..." if len(text) > 100 else f"∫ Equation: {text}"

            elif element_type == "discarded":
                text = element.get("text", "")
                bbox_info["text_preview"] = f"🗑️ Discarded: {text[:100]}..." if len(text) > 100 else f"🗑️ Discarded: {text}"

            bboxes.append(bbox_info)

        return {
            "status": "success",
            "file_hash": file_hash,
            "page_idx": page_idx,
            "total_elements": len(bboxes),
            "bboxes": bboxes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MinerU bboxes for {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting bboxes: {str(e)}"
        )

@app.delete("/documents/{file_hash}", response_model=Dict[str, Any])
def delete_document(file_hash: str):
    """
    Delete a document and all its related data from Qdrant, Neo4j, and MinIO

    This endpoint removes:
    - All points from Qdrant with matching file_hash
    - Document graph from Neo4j
    - All related files from MinIO (PDF, MinerU results, embeddings)

    Args:
        file_hash: Hash of the PDF file to delete

    Returns:
        Status of deletion operations
    """
    try:
        results = {
            "file_hash": file_hash,
            "qdrant_deleted": False,
            "neo4j_deleted": False,
            "minio_deleted": False,
            "minio_files_removed": 0
        }

        # Delete from Qdrant
        try:
            qdrant_success = qdrant_client.delete_points_by_file_hash(file_hash)
            results["qdrant_deleted"] = qdrant_success
            logger.info(f"Qdrant deletion for {file_hash}: {'success' if qdrant_success else 'failed'}")
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {e}")
            results["qdrant_deleted"] = False

        # Delete from Neo4j
        if NEO4J_AVAILABLE:
            try:
                neo4j_service = DocumentIndexService()
                neo4j_success = neo4j_service.delete_graph(file_hash)
                neo4j_service.close()
                results["neo4j_deleted"] = neo4j_success
                logger.info(f"Neo4j deletion for {file_hash}: {'success' if neo4j_success else 'failed'}")
            except Exception as e:
                logger.error(f"Error deleting from Neo4j: {e}")
                results["neo4j_deleted"] = False
        else:
            logger.warning("Neo4j not available, skipping Neo4j deletion")
            results["neo4j_deleted"] = None

        # Delete from MinIO
        try:
            # Remove PDF file
            pdf_prefix = f"pdfs/{file_hash}"
            minio_files_removed = 0

            # Try to remove PDF
            existing_pdf_objects = minio_client.list_objects(
                bucket_name=minio_client.bucket_name,
                prefix=pdf_prefix
            )
            for obj_path in existing_pdf_objects:
                try:
                    minio_client.remove_object(bucket_name=minio_client.bucket_name, object_name=obj_path)
                    minio_files_removed += 1
                    logger.info(f"Removed PDF object: {obj_path}")
                except Exception as e:
                    logger.warning(f"Could not remove PDF object {obj_path}: {e}")

            # Remove MinerU results
            mineru_prefix = f"mineru_results/{file_hash}"
            existing_mineru_objects = minio_client.list_objects(
                bucket_name=minio_client.bucket_name,
                prefix=mineru_prefix
            )
            for obj_path in existing_mineru_objects:
                try:
                    minio_client.remove_object(bucket_name=minio_client.bucket_name, object_name=obj_path)
                    minio_files_removed += 1
                    logger.info(f"Removed MinerU result object: {obj_path}")
                except Exception as e:
                    logger.warning(f"Could not remove MinerU object {obj_path}: {e}")

            # Remove embeddings
            embeddings_prefix = f"embeddings/{file_hash}"
            existing_embeddings_objects = minio_client.list_objects(
                bucket_name=minio_client.bucket_name,
                prefix=embeddings_prefix
            )
            for obj_path in existing_embeddings_objects:
                try:
                    minio_client.remove_object(bucket_name=minio_client.bucket_name, object_name=obj_path)
                    minio_files_removed += 1
                    logger.info(f"Removed embedding object: {obj_path}")
                except Exception as e:
                    logger.warning(f"Could not remove embedding object {obj_path}: {e}")

            results["minio_deleted"] = minio_files_removed > 0
            results["minio_files_removed"] = minio_files_removed
            logger.info(f"MinIO deletion for {file_hash}: removed {minio_files_removed} files")
        except Exception as e:
            logger.error(f"Error deleting from MinIO: {e}")
            results["minio_deleted"] = False

        # Check overall success
        all_success = (
            results["qdrant_deleted"] and
            (results["neo4j_deleted"] is None or results["neo4j_deleted"]) and
            results["minio_deleted"]
        )

        results["status"] = "success" if all_success else "partial"

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@app.delete("/documents/all", response_model=Dict[str, Any])
def delete_all_documents():
    """
    Delete ALL documents and their related data from Qdrant, Neo4j, and MinIO

    WARNING: This is a destructive operation that will remove all indexed documents!

    This endpoint removes:
    - All points from Qdrant collection
    - All document graphs from Neo4j
    - All files from MinIO (PDFs, MinerU results, embeddings)

    Returns:
        Status of deletion operations
    """
    try:
        results = {
            "qdrant_deleted": False,
            "neo4j_deleted": False,
            "minio_deleted": False,
            "minio_files_removed": 0,
            "warning": "This operation deleted ALL documents from the system"
        }

        # Delete all from Qdrant
        try:
            qdrant_success = qdrant_client.delete_all_points()
            results["qdrant_deleted"] = qdrant_success
            logger.info(f"Qdrant delete all: {'success' if qdrant_success else 'failed'}")
        except Exception as e:
            logger.error(f"Error deleting all from Qdrant: {e}")
            results["qdrant_deleted"] = False

        # Delete all from Neo4j
        if NEO4J_AVAILABLE:
            try:
                neo4j_service = DocumentIndexService()
                neo4j_success = neo4j_service.delete_all_graphs()
                neo4j_service.close()
                results["neo4j_deleted"] = neo4j_success
                logger.info(f"Neo4j delete all: {'success' if neo4j_success else 'failed'}")
            except Exception as e:
                logger.error(f"Error deleting all from Neo4j: {e}")
                results["neo4j_deleted"] = False
        else:
            logger.warning("Neo4j not available, skipping Neo4j deletion")
            results["neo4j_deleted"] = None

        # Delete all from MinIO - remove all objects
        try:
            all_objects = minio_client.list_objects(bucket_name=minio_client.bucket_name)
            minio_files_removed = 0

            for obj_path in all_objects:
                try:
                    # Only remove objects in our managed prefixes
                    if obj_path.startswith("pdfs/") or obj_path.startswith("mineru_results/") or obj_path.startswith("embeddings/"):
                        minio_client.remove_object(bucket_name=minio_client.bucket_name, object_name=obj_path)
                        minio_files_removed += 1
                        logger.info(f"Removed object: {obj_path}")
                except Exception as e:
                    logger.warning(f"Could not remove object {obj_path}: {e}")

            results["minio_deleted"] = minio_files_removed > 0
            results["minio_files_removed"] = minio_files_removed
            logger.info(f"MinIO delete all: removed {minio_files_removed} files")
        except Exception as e:
            logger.error(f"Error deleting all from MinIO: {e}")
            results["minio_deleted"] = False

        # Check overall success
        all_success = (
            results["qdrant_deleted"] and
            (results["neo4j_deleted"] is None or results["neo4j_deleted"]) and
            results["minio_deleted"]
        )

        results["status"] = "success" if all_success else "partial"

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting all documents: {str(e)}"
        )


def run_api(
        host: str = "0.0.0.0",
        port: int = 9191,
        reload: bool = False,
        log_level: str = "info") -> None:
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    run_api()

