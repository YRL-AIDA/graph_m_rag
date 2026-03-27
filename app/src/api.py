"""
Main API application for PDF processing.
Handles PDF upload to S3, processing with MinerU service,
and computing embeddings for each element in the result.
"""
import base64
import hashlib
import io
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path

from starlette.responses import HTMLResponse, FileResponse

from app.src.llm_client import ModelMessageDict, send_messasge
from app.src.qdrant_client_api import get_qdrant_client
from app.src.qwen3_emb_client import EmbeddingClient
from app.src.minio_client import MinioClient
from app.src.mineru_client import MinerUClient
from app.config.settings import settings
from app.src.utils.data_model import QuestionResponse, QuestionRequest, UploadedFileInfo, UploadedFilesListResponse, CollectionCreateRequest, CollectionInfo, CollectionsListResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
minio_client = MinioClient(logger)
mineru_client = MinerUClient(base_url=f"{settings.mineru.MINERU_HOST}:{settings.mineru.MINERU_PORT}")
emb_client = EmbeddingClient(base_url=settings.embedding.EMBEDDING_BASE_URL)
qdrant_client = get_qdrant_client()

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

                # Compute embedding for the image
                embedding = emb_client.get_image_embedding_base64(image_base64)

                # Prepare data for Qdrant
                embeddings_list.append(embedding.embedding)
                texts_list.append(f"Image: {img_path}")  # Text representation for Qdrant

                metadata = {
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
                embedding_key = f"embeddings/{file_hash}/element_{i}.json"
                embedding_data = {
                    "original_element": element,
                    "img_path": img_path,
                    "embedding": embedding.embedding,
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
                logger.info(f"Computed embedding for image element {i} (type: {element_type}, path: {img_path})")

                # Create text embeddings from image captions if available
                caption_text = " ".join(image_captions) if image_captions else ""
                footnote_text = " ".join(image_footnotes) if image_footnotes else ""

                if caption_text:
                    caption_content = f"Image Caption: {caption_text}"

                    try:
                        caption_embedding = emb_client.get_text_embedding(caption_content)

                        # Prepare data for Qdrant
                        embeddings_list.append(caption_embedding.embedding)
                        texts_list.append(caption_content)

                        caption_metadata = {
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
                        caption_embedding_key = f"embeddings/{file_hash}/element_{i}_caption.json"
                        caption_embedding_data = {
                            "original_element": element,
                            "img_path": img_path,
                            "text": caption_content,
                            "embedding": caption_embedding.embedding,
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
                        logger.info(f"Computed text embedding for image caption element {i} (type: {element_type}_caption)")

                    except Exception as e:
                        logger.error(f"Failed to compute text embedding for image caption element {i}: {e}")

                if footnote_text:
                    footnote_content = f"Image Footnote: {footnote_text}"

                    try:
                        footnote_embedding = emb_client.get_text_embedding(footnote_content)

                        # Prepare data for Qdrant
                        embeddings_list.append(footnote_embedding.embedding)
                        texts_list.append(footnote_content)

                        footnote_metadata = {
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
                        footnote_embedding_key = f"embeddings/{file_hash}/element_{i}_footnote.json"
                        footnote_embedding_data = {
                            "original_element": element,
                            "img_path": img_path,
                            "text": footnote_text,
                            "embedding": footnote_embedding.embedding,
                            "element_index": i,
                            "element_type": f"{element_type}_caption",
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
                        logger.info(f"Computed text embedding for image caption element {i} (type: {element_type}_footnote)")

                    except Exception as e:
                        logger.error(f"Failed to compute text embedding for image caption element {i}: {e}")

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

                    # Prepare data for Qdrant
                    embeddings_list.append(caption_embedding.embedding)
                    texts_list.append(caption_content)

                    caption_metadata = {
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
                    caption_embedding_key = f"embeddings/{file_hash}/element_{i}_caption.json"
                    caption_embedding_data = {
                        "original_element": element,
                        "img_path": img_path,
                        "text": caption_content,
                        "embedding": caption_embedding.embedding,
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
                    logger.info(f"Computed text embedding for table caption element {i} (type: {element_type}_caption)")

                except Exception as e:
                    logger.error(f"Failed to compute text embedding for table caption element {i}: {e}")
            if footnote_text:
                footnote_content = f"Table Footnote: {footnote_text}"

                try:
                    footnote_embedding = emb_client.get_text_embedding(footnote_content)

                    # Prepare data for Qdrant
                    embeddings_list.append(footnote_embedding.embedding)
                    texts_list.append(footnote_content)

                    footnote_metadata = {
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
                    footnote_embedding_key = f"embeddings/{file_hash}/element_{i}_caption.json"
                    footnote_embedding_data = {
                        "original_element": element,
                        "img_path": img_path,
                        "text": footnote_content,
                        "embedding": footnote_embedding.embedding,
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
                    logger.info(f"Computed text embedding for table caption element {i} (type: {element_type}_caption)")

                except Exception as e:
                    logger.error(f"Failed to compute text embedding for table caption element {i}: {e}")

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

                # Prepare data for Qdrant
                embeddings_list.append(embedding.embedding)
                texts_list.append(text_content)

                metadata = {
                    "element_index": i,
                    "element_type": element_type,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat(),
                    "original_element": element
                }
                metadata_list.append(metadata)

                # Save embedding to S3 with a specific naming convention
                embedding_key = f"embeddings/{file_hash}/element_{i}.json"
                embedding_data = {
                    "original_element": element,
                    "text": text_content,
                    "embedding": embedding.embedding,
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
                logger.info(f"Computed embedding for element {i} (type: {element_type})")

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
        if test_embedding and len(test_embedding.embedding) > 0:
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

        processing_time = time.time() - start_time

        return PDFUploadResponse(
            status="success",
            message="PDF uploaded, processed with MinerU, and embeddings computed",
            file_hash=file_hash,
            s3_path=pdf_s3_key,
            mineru_result_path=mineru_result_key,
            embeddings_computed=embeddings_count,
            processing_time=processing_time
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
def ask_document(request: QuestionRequest):
    """
    Ask a question about a specific document by file_hash.
    If the document is not indexed, it will be indexed first.

    Steps:
    1. Check if document is indexed in Qdrant by file_hash
    2. If not indexed, index the document from MinIO
    3. Generate embedding for the question
    4. Search for relevant chunks in Qdrant filtered by file_hash
    5. Return the results
    6. Optionally generate answer using LLM

    Args:
        request: QuestionRequest with file_hash, question, limit, collection_name, and use_llm

    """
    from qdrant_client.http import models

    file_hash = request.file_hash
    question = request.question
    limit = request.limit
    collection_name = request.collection_name
    use_llm = request.use_llm

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
        # Generate embedding for the question
        question_embedding = emb_client.get_text_embedding(question)

        # Create filter for file_hash
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash)
                )
            ]
        )

        # Search for relevant chunks
        search_results = client.search(
            query_vector=question_embedding.embedding,
            limit=limit,
            filter_condition=filter_condition
        )

        # Format results
        answers = []
        for result in search_results.points:
            payload = result.payload
            element_type = payload.get("element_type", "")
            original_element = payload.get("original_element", {})

            answer = {
                "text": payload.get("text", ""),
                "score": result.score,
                "element_type": element_type,
                "element_index": payload.get("element_index", 0),
                "page_idx": original_element.get("page_idx", 0) if original_element else 0,
                "img_path": original_element.get("img_path", None),  # Store img_path for images and tables
                "image_base64": None,  # Will be populated for image and table elements
                "bbox": original_element.get("bbox", None)  # Store bbox for visualization
            }

            # Download image data for image and table elements
            if element_type in ("image", "table") and answer["img_path"]:
                try:
                    image_data = minio_client.get_object(
                        bucket_name=minio_client.bucket_name,
                        object_name=answer["img_path"]
                    )
                    answer["image_base64"] = base64.b64encode(image_data).decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to download image {answer['img_path']}: {e}")

            answers.append(answer)
        # Generate LLM answer if requested
        llm_answer = None
        if use_llm and answers:
            try:
                # Create messages for LLM with system prompt
                message = ModelMessageDict(role='user')
                system_prompt = "Вы помощник, который отвечает на вопросы на основе предоставленного контекста. Если ответ не найден в контексте, скажите об этом."
                message.add_text_content(system_prompt)

                # Build context with text and images
                context_parts = []
                for ans in answers:
                    element_type = ans.get("element_type", "")

                    # Add image if available (for both image and table elements)
                    if element_type in ("image", "table") and ans.get("image_base64"):
                        message.add_img_content_base64(ans["image_base64"])

                    # Add text content
                    if ans.get("text"):
                        context_parts.append(ans["text"])

                # Combine all text context
                if context_parts:
                    context = "\n\n".join(context_parts)
                    message.add_text_content(context)
                    user_prompt = f"Вопрос: {question} Ответьте на вопрос, используя только информацию из контекста (текст и изображения)."""
                    message.add_text_content(user_prompt)

                # Call LLM
                success, llm_responses = send_messasge(
                    messages=message
                )

                if success and llm_responses:
                    llm_answer = llm_responses[0]
                    logger.info(f"LLM answer generated successfully for question: {question}")
                else:
                    logger.warning(f"LLM failed to generate answer for question: {question}")
            except Exception as e:
                logger.error(f"Error generating LLM answer: {e}")
                llm_answer = f"Error generating LLM answer: {str(e)}"

        return QuestionResponse(
            status="success",
            message=f"Found {len(answers)} relevant chunks for the question in collection '{actual_collection_name}'",
            file_hash=file_hash,
            question=question,
            answers=answers,
            indexed=is_indexed,
            collection_name=actual_collection_name,
            llm_answer=llm_answer
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
                        "label": f"image_caption_{i}",
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
                        "label": f"image_footnote_{i}",
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
                        "label": f"table_caption_{i}",
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
                        "label": f"table_footnote_{i}",
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