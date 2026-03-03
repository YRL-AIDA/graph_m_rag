"""
Main API application for PDF processing.
Handles PDF upload to S3, processing with MinerU service,
and computing embeddings for each element in the result.
"""
import hashlib
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

from app.src.qwen3_emb_client import EmbeddingClient
from app.src.s3_client import S3Client
from app.src.mineru_client import MinerUClient
from app.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
s3_client = S3Client(logger)
mineru_client = MinerUClient()
emb_client = EmbeddingClient(base_url="http://192.168.19.127:10114/embedding")

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
        elements: List of elements from MinerU result
        file_hash: Hash of the original PDF file

    Returns:
        Number of elements processed
    """
    processed_count = 0

    # Process elements recursively to extract text content
    def extract_texts_from_element(element):
        texts = []
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    # Add key as context for the value
                    texts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    texts.extend(extract_texts_from_element(value))
        elif isinstance(element, list):
            for item in element:
                texts.extend(extract_texts_from_element(item))
        return texts

    # Collect all text elements from the MinerU result
    all_texts = []

    if isinstance(elements, list):
        for element in elements:
            all_texts.extend(extract_texts_from_element(element))
    elif isinstance(elements, dict):
        all_texts.extend(extract_texts_from_element(elements))

    # Compute embeddings for each text
    for i, text in enumerate(all_texts):
        if text.strip():  # Only process non-empty texts
            try:
                # Generate embedding using the embedding client
                embedding = emb_client.get_text_embedding(text)

                # Save embedding to S3 with a specific naming convention
                embedding_key = f"embeddings/{file_hash}/element_{i}.json"
                embedding_data = {
                    "text": text,
                    "embedding": embedding.embedding,
                    "element_index": i,
                    "file_hash": file_hash,
                    "created_at": datetime.now().isoformat()
                }

                # Convert to JSON and upload to S3
                embedding_json = json.dumps(embedding_data, ensure_ascii=False)
                s3_client.put_object(
                    bucket_name=s3_client.bucket_name,
                    object_name=embedding_key,
                    data=embedding_json.encode('utf-8'),
                    content_type='application/json'
                )

                processed_count += 1
                logger.info(f"Computed embedding for element {i} of {len(all_texts)}")

            except Exception as e:
                logger.error(f"Failed to compute embedding for element {i}: {e}")
                continue

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
        result = mineru_client.analyze_pdf(
            file_path=file_path,
            include_original=True,
            include_previews=True
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
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {}

    # Check S3 connectivity
    try:
        s3_client.list_buckets()
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

        # Calculate file hash
        file_hash = hashlib.md5(content).hexdigest()

        # Create temporary file
        temp_dir = Path("/tmp/pdf_processing")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = str(temp_dir / f"{file_hash}_{file.filename}")

        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(content)

        # Upload original PDF to S3
        pdf_s3_key = f"pdfs/{file_hash}/{file.filename}"
        s3_client.upload(
            bucket_name=s3_client.bucket_name,
            object_name=pdf_s3_key,
            data=content,
            content_type="application/pdf"
        )

        logger.info(f"Uploaded PDF to S3: {pdf_s3_key}")

        # Process with MinerU
        logger.info(f"Processing PDF {file_hash} with MinerU service")
        mineru_result = process_with_mineru(temp_file_path)

        # Store MinerU result in S3
        mineru_result_key = f"mineru_results/{file_hash}/result.json"
        mineru_result_serializable = convert_to_serializable(mineru_result)
        result_json = json.dumps(mineru_result_serializable, ensure_ascii=False, indent=2)

        s3_client.put_object(
            bucket_name=s3_client.bucket_name,
            object_name=mineru_result_key,
            data=result_json.encode('utf-8'),
            content_type="application/json"
        )

        logger.info(f"Stored MinerU result to S3: {mineru_result_key}")

        # Compute embeddings for each element in the result
        logger.info(f"Computing embeddings for MinerU result elements")

        # Extract elements from MinerU result - structure may vary depending on MinerU output
        elements = []
        if "structured_data" in mineru_result:
            elements.append(mineru_result["structured_data"])
        elif "pages" in mineru_result:
            elements.extend(mineru_result["pages"])
        elif "blocks" in mineru_result:
            elements.extend(mineru_result["blocks"])
        elif "elements" in mineru_result:
            elements.extend(mineru_result["elements"])
        else:
            # If no specific structure found, treat the entire result as elements
            elements.append(mineru_result)

        # Compute embeddings synchronously
        embeddings_count = compute_embeddings_for_elements(elements, file_hash)
        logger.info(f"Completed synchronous embedding computation: {embeddings_count} elements processed")

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