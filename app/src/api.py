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

from app.src.qdrant_client_api import get_qdrant_client
from app.src.qwen3_emb_client import EmbeddingClient
from app.src.minio_client import MinioClient
from app.src.mineru_client import MinerUClient
from app.config.settings import settings
from app.src.utils.data_model import QuestionResponse, QuestionRequest

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
            if text_level is not None:
                text_content = f"Text (level {text_level}): {text}"
            else:
                text_content = f"Text: {text}"

        elif element_type == "image":
            # Combine image path and captions if available
            img_path = element.get("img_path", "")
            # Download image from MinIO
            try:
                image_data = minio_client.get_object(
                    bucket_name=minio_client.bucket_name,
                    object_name=img_path
                )

                # Get image as bytes
                image_base64 = base64.b64encode(image_data)

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
                    "img_path": img_path
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

                # Skip the rest of the processing since we've already handled the image
                continue
            except Exception as e:
                logger.error(f"Failed to download or process image {img_path} for element {i}: {e}")
                # If image processing fails, fall back to text-based approach
                image_captions = element.get("image_caption", [])
                image_footnotes = element.get("image_footnote", [])

                caption_text = " ".join(image_captions) if image_captions else ""
                footnote_text = " ".join(image_footnotes) if image_footnotes else ""

                text_content = f"Image: {img_path}"
                if caption_text:
                    text_content += f" | Caption: {caption_text}"
                if footnote_text:
                    text_content += f" | Footnote: {footnote_text}"

        elif element_type == "table":
            # Combine table information
            img_path = element.get("img_path", "")
            table_captions = element.get("table_caption", [])
            table_footnotes = element.get("table_footnote", [])
            table_body = element.get("table_body", "")

            caption_text = " ".join(table_captions) if table_captions else ""
            footnote_text = " ".join(table_footnotes) if table_footnotes else ""

            text_content = f"Table: {img_path}"
            if caption_text:
                text_content += f" | Caption: {caption_text}"
            if footnote_text:
                text_content += f" | Footnote: {footnote_text}"
            if table_body:
                text_content += f" | Body: {table_body}"

        else:
            # For unknown types, try to extract any available text content
            text_content = json.dumps(element, ensure_ascii=False)

        # Only process elements with non-empty text content
        if text_content.strip():
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
            lang="ru",
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
            "GET /uploaded-files": "Get list of uploaded files with hashes"
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

@app.post("/upload-pdf", response_model=PDFUploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF document to S3, process with MinerU, and compute embeddings

    Steps:
    1. Upload PDF to S3
    2. Process with MinerU service
    3. Store MinerU results in S3
    4. Compute embeddings for each element in the result
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be in PDF format"
        )

    temp_file_path = None

    try:
        # Read file content
        content = file.file.read()

        # Calculate file hash based on content only (not filename)
        file_hash = hashlib.md5(content).hexdigest()

        # Check if PDF with this hash already exists (regardless of filename)
        # Use hash-based prefix to detect duplicates by content
        existing_pdf_prefix = f"pdfs/{file_hash}_"
        existing_pdfs = minio_client.list_objects(
            bucket_name=minio_client.bucket_name,
            prefix=existing_pdf_prefix
        )

        if existing_pdfs:
            logger.info(f"PDF with hash {file_hash} already exists in MinIO, skipping processing")
            processing_time = time.time() - start_time

            # Get the first existing PDF path
            first_pdf_path = existing_pdfs[0] if existing_pdfs else f"pdfs/{file_hash}_{file.filename}/{file.filename}"
            # Extract file_unique_id from the existing path
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

        # Create unique directory for this file using both hash and filename
        # to avoid conflicts when same content has different filenames
        file_unique_id = f"{file_hash}_{file.filename}"
        pdf_s3_key = f"pdfs/{file_unique_id}/{file.filename}"

        # Create temporary file
        temp_dir = Path("/tmp/pdf_processing")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = str(temp_dir / f"{file_hash}_{file.filename}")

        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(content)

        # Process with MinerU
        logger.info(f"Processing PDF {file_hash} with MinerU service")
        mineru_result = process_with_mineru(temp_file_path)

        # Store MinerU result in S3 with hash in the filename
        mineru_result_key = f"mineru_results/{file_unique_id}/result.json"
        # Add file_hash to the result JSON
        mineru_result_serializable = convert_to_serializable(mineru_result)
        # Add metadata with file hash and original filename
        mineru_result_serializable["metadata"] = {
            "file_hash": file_hash,
            "original_filename": file.filename,
            "processed_at": datetime.now().isoformat()
        }
        result_json = json.dumps(mineru_result_serializable, ensure_ascii=False, indent=2)

        minio_client.put_object(
            bucket_name=minio_client.bucket_name,
            object_name=mineru_result_key,
            data=result_json.encode('utf-8'),
            content_type="application/json"
        )

        images = mineru_result["results"]["result"]["results"]["images_base64"]

        for key, image_base64 in images.items():
            image_key = f"images/{key}"
            image_data = base64.b64decode(image_base64)
            image_buffer = io.BytesIO(image_data)

            minio_client.put_object(
                bucket_name=minio_client.bucket_name,
                object_name=image_key,
                data=image_buffer,
                content_type="image/jpeg"
            )

        logger.info(f"Stored MinerU result to S3: {mineru_result_key}")

        # Compute embeddings for each element in the result
        logger.info(f"Computing embeddings for MinerU result elements")

        # Extract elements from MinerU result - structure may vary depending on MinerU output
        elements = []
        elements.extend(mineru_result["results"]["result"]["results"]["content_list"])
        # Compute embeddings synchronously
        embeddings_count = compute_embeddings_for_elements(elements, file_hash)
        logger.info(f"Completed synchronous embedding computation: {embeddings_count} elements processed")

        # Upload original PDF to MinIO
        minio_client.upload(
            bucket_name=minio_client.bucket_name,
            object_name=pdf_s3_key,
            data=content,
            content_type="application/pdf"
        )

        logger.info(f"Uploaded PDF to S3: {pdf_s3_key}")

        processing_time = time.time() - start_time

        return PDFUploadResponse(
            status="success",
            message="PDF uploaded, processed with MinerU, and embeddings computed",
            file_hash=file_hash,
            s3_path=pdf_s3_key,
            mineru_result_path=mineru_result_key,
            embeddings_computed=embeddings_count,  # Actual count computed
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file_path}: {e}")


def check_document_indexed(file_hash: str) -> bool:
    """
    Check if a document is already indexed in Qdrant by file_hash

    Args:
        file_hash: Hash of the PDF file

    Returns:
        True if document is indexed, False otherwise
    """
    try:
        from qdrant_client.http import models

        # Search for any point with this file_hash
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash)
                )
            ]
        )

        results = qdrant_client.search(
            query_vector=[0.0] * 2048,  # Dummy vector, we just need to check existence
            limit=1,
            filter_condition=filter_condition
        )

        return len(results.points) > 0
    except Exception as e:
        logger.error(f"Error checking if document is indexed: {e}")
        return False


def index_document_by_hash(file_hash: str) -> bool:
    """
    Index a document by its hash if it exists in MinIO

    Args:
        file_hash: Hash of the PDF file

    Returns:
        True if successfully indexed, False otherwise
    """
    try:
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
    """
    from qdrant_client.http import models

    file_hash = request.file_hash
    question = request.question
    limit = request.limit

    # Check if document is already indexed
    is_indexed = check_document_indexed(file_hash)

    if not is_indexed:
        logger.info(f"Document {file_hash} is not indexed, attempting to index it...")
        index_success = index_document_by_hash(file_hash)

        if not index_success:
            return QuestionResponse(
                status="error",
                message=f"Failed to index document with hash {file_hash}. Document may not exist in MinIO.",
                file_hash=file_hash,
                question=question,
                answers=[],
                indexed=False
            )

        is_indexed = True
        logger.info(f"Document {file_hash} successfully indexed")

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
        search_results = qdrant_client.search(
            query_vector=question_embedding.embedding,
            limit=limit,
            filter_condition=filter_condition
        )

        # Format results
        answers = []
        for result in search_results.points:
            answer = {
                "text": result.payload.get("text", ""),
                "score": result.score,
                "element_type": result.payload.get("element_type", ""),
                "element_index": result.payload.get("element_index", 0),
                "page_idx": result.payload.get("original_element", {}).get("page_idx", 0) if result.payload.get("original_element") else 0
            }
            answers.append(answer)

        return QuestionResponse(
            status="success",
            message=f"Found {len(answers)} relevant chunks for the question",
            file_hash=file_hash,
            question=question,
            answers=answers,
            indexed=is_indexed
        )

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
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