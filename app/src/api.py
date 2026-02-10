# app/api.py
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles
import os
from pathlib import Path

from app.src.services.minio_service import MinioService
from app.src.services.mineru_service import MinerUService
from app.src.services.rag_service import RAGService
from app.src.services.pdf_processor import PDFProcessorService
from app.config.settings import settings

# Настройка логгера
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="RAG PDF Processing API",
    description="API для обработки PDF файлов с использованием MinerU, MinIO и RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервисов
minio_service = MinioService()
mineru_service = MinerUService()
rag_service = RAGService()
pdf_processor = PDFProcessorService()


# Модели Pydantic для запросов и ответов
class PDFUploadRequest(BaseModel):
    """Модель для загрузки PDF"""
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Дополнительные метаданные для PDF"
    )
    force_reprocess: bool = Field(
        default=False,
        description="Принудительная повторная обработка, даже если файл уже существует"
    )


class PDFProcessingResponse(BaseModel):
    """Модель ответа обработки PDF"""
    status: str
    message: str
    file_hash: str
    file_exists: bool
    processing_time: float
    results: Dict[str, Any]


class PDFStatusResponse(BaseModel):
    """Модель ответа статуса PDF"""
    file_hash: str
    exists_in_minio: bool
    buckets: Dict[str, Any]
    mineru_analysis: Optional[Dict[str, Any]] = None
    rag_indexed: bool = False


class BatchProcessingResponse(BaseModel):
    """Модель ответа пакетной обработки"""
    total_files: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]


# Вспомогательные функции
def calculate_file_hash(file_bytes: bytes) -> str:
    """Вычисление хэша файла"""
    return hashlib.md5(file_bytes).hexdigest()


def get_minio_bucket_for_hash(file_hash: str, bucket_type: str = "pdf-analysis") -> str:
    """Получение bucket для хэша файла"""
    # Используем первые 2 символа хэша как поддиректорию для равномерного распределения
    prefix = file_hash[:2]
    return f"{bucket_type}/{prefix}/{file_hash}"


async def save_file_temporarily(file: UploadFile) -> str:
    """Сохранение файла во временное хранилище"""
    temp_dir = Path("/tmp/rag_uploads")
    temp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    temp_path = temp_dir / f"{timestamp}_{file.filename}"

    async with aiofiles.open(temp_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    return str(temp_path)


def cleanup_temp_file(file_path: str):
    """Удаление временного файла"""
    try:
        os.remove(file_path)
        logger.info(f"Удален временный файл: {file_path}")
    except Exception as e:
        logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")


# Эндпоинты API
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "RAG PDF Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /process-pdf": "Обработка PDF файла",
            "GET /status/{file_hash}": "Проверка статуса обработки",
            "POST /batch-process": "Пакетная обработка PDF",
            "GET /retrieve/{file_hash}": "Получение результатов обработки",
            "DELETE /delete/{file_hash}": "Удаление обработанного PDF"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья всех сервисов"""
    services_status = {}

    # Проверка MinIO
    try:
        minio_service.client.list_buckets()
        services_status["minio"] = "healthy"
    except Exception as e:
        services_status["minio"] = f"unhealthy: {str(e)}"

    # Проверка MinerU
    try:
        # Попробуем простой запрос к MinerU
        import requests
        response = requests.get(f"{settings.mineru.url}/health", timeout=5)
        services_status[
            "mineru"] = "healthy" if response.status_code == 200 else f"unhealthy: HTTP {response.status_code}"
    except Exception as e:
        services_status["mineru"] = f"unreachable: {str(e)}"

    # Проверка Qdrant
    try:
        rag_service.qdrant.client.get_collections()
        services_status["qdrant"] = "healthy"
    except Exception as e:
        services_status["qdrant"] = f"unhealthy: {str(e)}"

    # Проверка Neo4j
    try:
        rag_service.neo4j.driver.verify_connectivity()
        services_status["neo4j"] = "healthy"
    except Exception as e:
        services_status["neo4j"] = f"unhealthy: {str(e)}"

    # Проверка Ollama
    try:
        import requests
        response = requests.get(f"{settings.ollama.url}/api/tags", timeout=5)
        services_status[
            "ollama"] = "healthy" if response.status_code == 200 else f"unhealthy: HTTP {response.status_code}"
    except Exception as e:
        services_status["ollama"] = f"unreachable: {str(e)}"

    all_healthy = all(status == "healthy" for status in services_status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services_status
    }


@app.post("/process-pdf", response_model=PDFProcessingResponse)
async def process_pdf(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        metadata: Optional[str] = Query(None, description="JSON строка с метаданными"),
        force_reprocess: bool = Query(False, description="Принудительная повторная обработка")
):
    """
    Обработка PDF файла

    Этапы:
    1. Вычисление хэша файла
    2. Проверка существования в MinIO
    3. Если файл новый или force_reprocess=True, отправка в MinerU
    4. Сохранение результатов в MinIO
    5. Индексация в RAG (в фоне)
    """

    start_time = time.time()

    # Проверяем тип файла
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть в формате PDF"
        )

    try:
        # Читаем содержимое файла
        content = await file.read()

        # Вычисляем хэш файла
        file_hash = calculate_file_hash(content)

        # Парсим метаданные
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Не удалось распарсить метаданные: {metadata}")

        # Добавляем информацию о загрузке в метаданные
        meta_dict.update({
            "original_filename": file.filename,
            "upload_timestamp": datetime.now().isoformat(),
            "file_size": len(content)
        })

        # Проверяем существование файла в MinIO
        pdf_exists = False
        analysis_exists = False

        # Проверяем в разных bucket'ах
        buckets_to_check = ["pdf-originals", "pdf-analysis"]
        for bucket in buckets_to_check:
            try:
                # Ищем файлы с префиксом хэша
                objects = list(minio_service.client.list_objects(
                    bucket,
                    prefix=file_hash,
                    recursive=True
                ))
                if objects:
                    if bucket == "pdf-originals":
                        pdf_exists = True
                    elif bucket == "pdf-analysis":
                        analysis_exists = True
            except Exception as e:
                logger.warning(f"Ошибка при проверке bucket {bucket}: {e}")

        # Если файл уже существует и не требуется переобработка
        if pdf_exists and analysis_exists and not force_reprocess:
            logger.info(f"PDF {file_hash} уже обработан, возвращаем существующий результат")

            # Получаем существующий анализ
            analysis_data = None
            try:
                # Ищем файл анализа
                objects = list(minio_service.client.list_objects(
                    "pdf-analysis",
                    prefix=file_hash,
                    recursive=True
                ))

                if objects:
                    analysis_obj = minio_service.client.get_object(
                        "pdf-analysis",
                        objects[0].object_name
                    )
                    analysis_data = json.loads(analysis_obj.read())
                    analysis_obj.close()
                    analysis_obj.release_conn()
            except Exception as e:
                logger.error(f"Ошибка при получении существующего анализа: {e}")
                analysis_data = {"error": "Не удалось получить существующий анализ"}

            processing_time = time.time() - start_time

            return PDFProcessingResponse(
                status="success",
                message="PDF уже обработан ранее",
                file_hash=file_hash,
                file_exists=True,
                processing_time=processing_time,
                results={
                    "mineru_analysis": analysis_data,
                    "already_processed": True
                }
            )

        # Сохраняем файл во временное хранилище
        temp_file_path = await save_file_temporarily(file)

        try:
            # Обрабатываем PDF через MinerU
            logger.info(f"Обработка PDF {file_hash} через MinerU")

            # Отправляем в MinerU
            mineru_result = mineru_service.analyze_document_with_base64(
                temp_file_path,
                include_original=True,
                include_previews=True
            )

            if mineru_result.get("status") != "success":
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка обработки MinerU: {mineru_result.get('error', 'Unknown error')}"
                )

            # Сохраняем оригинальный PDF в MinIO
            pdf_metadata = {
                **meta_dict,
                "file_hash": file_hash,
                "processed_by": "mineru",
                "processing_timestamp": datetime.now().isoformat()
            }

            pdf_upload_result = minio_service.upload_file(
                bucket_name="pdf-originals",
                file_name=f"{file_hash}.pdf",
                file_data=content,
                content_type="application/pdf",
                metadata=pdf_metadata
            )

            # Сохраняем результат анализа MinerU в MinIO
            analysis_filename = f"{file_hash}_analysis.json"
            analysis_json = json.dumps(mineru_result, ensure_ascii=False, indent=2).encode('utf-8')

            analysis_metadata = {
                "original_pdf_hash": file_hash,
                "original_pdf_path": pdf_upload_result.get("minio_path", ""),
                "processing_timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0"
            }

            analysis_upload_result = minio_service.upload_file(
                bucket_name="pdf-analysis",
                file_name=analysis_filename,
                file_data=analysis_json,
                content_type="application/json",
                metadata=analysis_metadata
            )

            # Извлекаем текст из анализа для индексации в RAG
            extracted_text = ""
            if mineru_result.get("text"):
                extracted_text = mineru_result["text"]
            elif mineru_result.get("structured_data"):
                # Извлекаем текст из структурированных данных
                structured = mineru_result["structured_data"]
                text_parts = []

                for block in structured.get("text_blocks", []):
                    if "text" in block:
                        text_parts.append(block["text"])

                extracted_text = "\n\n".join(text_parts)

            # Если есть извлеченный текст, индексируем в RAG в фоне
            if extracted_text and background_tasks is not None:
                background_tasks.add_task(
                    index_pdf_in_rag,
                    file_hash,
                    extracted_text,
                    mineru_result,
                    meta_dict
                )

            # Формируем полный результат
            final_result = {
                "file_info": {
                    "hash": file_hash,
                    "filename": file.filename,
                    "size": len(content),
                    "original_path": pdf_upload_result.get("minio_path", ""),
                    "analysis_path": analysis_upload_result.get("minio_path", "")
                },
                "mineru_analysis": mineru_result,
                "metadata": meta_dict,
                "processing_info": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": time.time() - start_time,
                    "indexed_in_rag": extracted_text != ""
                }
            }

            processing_time = time.time() - start_time

            return PDFProcessingResponse(
                status="success",
                message="PDF успешно обработан",
                file_hash=file_hash,
                file_exists=pdf_exists,
                processing_time=processing_time,
                results=final_result
            )

        finally:
            # Удаляем временный файл
            cleanup_temp_file(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


async def index_pdf_in_rag(
        file_hash: str,
        extracted_text: str,
        mineru_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
):
    """Фоновая задача для индексации PDF в RAG"""
    try:
        logger.info(f"Начало индексации PDF {file_hash} в RAG")

        # Подготавливаем метаданные для RAG
        rag_metadata = {
            "file_hash": file_hash,
            "document_type": "pdf",
            "source": "mineru_analysis",
            "original_filename": metadata.get("original_filename", ""),
            "processing_timestamp": datetime.now().isoformat(),
            "mineru_analysis_available": True
        }

        # Добавляем информацию из анализа MinerU
        if mineru_analysis.get("structured_data"):
            structured = mineru_analysis["structured_data"]
            rag_metadata.update({
                "tables_count": len(structured.get("tables", [])),
                "figures_count": len(structured.get("figures", [])),
                "images_count": len(structured.get("images", []))
            })

        # Индексируем в RAG
        rag_result = rag_service.index_document(
            file_name=f"{file_hash}.pdf",
            content=extracted_text,
            metadata=rag_metadata
        )

        logger.info(f"PDF {file_hash} успешно проиндексирован в RAG")
        return rag_result

    except Exception as e:
        logger.error(f"Ошибка индексации PDF {file_hash} в RAG: {e}")
        return None


@app.get("/status/{file_hash}", response_model=PDFStatusResponse)
async def get_pdf_status(
        file_hash: str,
        include_analysis: bool = Query(False, description="Включать ли анализ MinerU в ответ")
):
    """Получение статуса обработки PDF по хэшу"""
    try:
        # Проверяем существование в разных bucket'ах MinIO
        buckets_to_check = ["pdf-originals", "pdf-analysis"]
        buckets_status = {}

        for bucket in buckets_to_check:
            try:
                objects = list(minio_service.client.list_objects(
                    bucket,
                    prefix=file_hash,
                    recursive=True
                ))
                buckets_status[bucket] = {
                    "exists": len(objects) > 0,
                    "objects_count": len(objects),
                    "object_names": [obj.object_name for obj in objects]
                }
            except Exception as e:
                buckets_status[bucket] = {
                    "exists": False,
                    "error": str(e)
                }

        exists_in_minio = any(
            buckets_status[bucket].get("exists", False)
            for bucket in buckets_to_check
        )

        # Проверяем индексацию в RAG
        rag_indexed = False
        try:
            # Ищем документ в Qdrant по file_hash в payload
            from qdrant_client.http import models as qdrant_models

            # Создаем фильтр для поиска по file_hash
            filter_condition = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.file_hash",
                        match=qdrant_models.MatchValue(value=file_hash)
                    )
                ]
            )

            # Получаем коллекцию RAG
            collections = rag_service.qdrant.client.get_collections()
            if settings.qdrant.collection_name in [c.name for c in collections.collections]:
                count_result = rag_service.qdrant.client.count(
                    collection_name=settings.qdrant.collection_name,
                    count_request=qdrant_models.CountRequest(filter=filter_condition)
                )
                rag_indexed = count_result.count > 0
        except Exception as e:
            logger.warning(f"Не удалось проверить индексацию в RAG: {e}")

        # Получаем анализ MinerU если требуется
        mineru_analysis = None
        if include_analysis and buckets_status.get("pdf-analysis", {}).get("exists"):
            try:
                # Берем первый найденный объект анализа
                objects = list(minio_service.client.list_objects(
                    "pdf-analysis",
                    prefix=file_hash,
                    recursive=True
                ))

                if objects:
                    analysis_obj = minio_service.client.get_object(
                        "pdf-analysis",
                        objects[0].object_name
                    )
                    analysis_data = json.loads(analysis_obj.read())
                    mineru_analysis = analysis_data
                    analysis_obj.close()
                    analysis_obj.release_conn()
            except Exception as e:
                logger.error(f"Ошибка получения анализа MinerU: {e}")

        return PDFStatusResponse(
            file_hash=file_hash,
            exists_in_minio=exists_in_minio,
            buckets=buckets_status,
            mineru_analysis=mineru_analysis,
            rag_indexed=rag_indexed
        )

    except Exception as e:
        logger.error(f"Ошибка получения статуса PDF {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статуса: {str(e)}"
        )


@app.post("/batch-process", response_model=BatchProcessingResponse)
async def batch_process_pdfs(
        files: List[UploadFile] = File(...),
        background_tasks: BackgroundTasks = None,
        metadata: Optional[str] = Query(None, description="Общие метаданные для всех файлов")
):
    """Пакетная обработка нескольких PDF файлов"""
    try:
        # Парсим общие метаданные
        common_metadata = {}
        if metadata:
            try:
                common_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Не удалось распарсить общие метаданные: {metadata}")

        results = []
        successful = 0
        failed = 0

        for file in files:
            try:
                # Проверяем тип файла
                if not file.filename.lower().endswith('.pdf'):
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": "Файл не является PDF"
                    })
                    failed += 1
                    continue

                # Читаем содержимое файла
                content = await file.read()

                # Вычисляем хэш
                file_hash = calculate_file_hash(content)

                # Создаем метаданные для этого файла
                file_metadata = common_metadata.copy()
                file_metadata.update({
                    "original_filename": file.filename,
                    "batch_processing": True,
                    "batch_timestamp": datetime.now().isoformat()
                })

                # Сохраняем файл во временное хранилище
                temp_file_path = await save_file_temporarily(file)

                try:
                    # Проверяем, существует ли уже файл
                    pdf_exists = False
                    try:
                        objects = list(minio_service.client.list_objects(
                            "pdf-originals",
                            prefix=file_hash,
                            recursive=True
                        ))
                        pdf_exists = len(objects) > 0
                    except:
                        pass

                    if pdf_exists:
                        results.append({
                            "filename": file.filename,
                            "status": "skipped",
                            "file_hash": file_hash,
                            "message": "Файл уже обработан ранее"
                        })
                        successful += 1
                        continue

                    # Обрабатываем через MinerU
                    mineru_result = mineru_service.analyze_document_with_base64(
                        temp_file_path,
                        include_original=True,
                        include_previews=True
                    )

                    if mineru_result.get("status") != "success":
                        raise Exception(f"MinerU error: {mineru_result.get('error')}")

                    # Сохраняем оригинальный PDF
                    pdf_upload_result = minio_service.upload_file(
                        bucket_name="pdf-originals",
                        file_name=f"{file_hash}.pdf",
                        file_data=content,
                        content_type="application/pdf",
                        metadata={
                            **file_metadata,
                            "file_hash": file_hash
                        }
                    )

                    # Сохраняем анализ
                    analysis_json = json.dumps(mineru_result, ensure_ascii=False, indent=2).encode('utf-8')

                    analysis_upload_result = minio_service.upload_file(
                        bucket_name="pdf-analysis",
                        file_name=f"{file_hash}_analysis.json",
                        file_data=analysis_json,
                        content_type="application/json"
                    )

                    # Извлекаем текст для RAG
                    extracted_text = ""
                    if mineru_result.get("text"):
                        extracted_text = mineru_result["text"]

                    # Добавляем задачу индексации в RAG
                    if extracted_text and background_tasks is not None:
                        background_tasks.add_task(
                            index_pdf_in_rag,
                            file_hash,
                            extracted_text,
                            mineru_result,
                            file_metadata
                        )

                    results.append({
                        "filename": file.filename,
                        "status": "success",
                        "file_hash": file_hash,
                        "pdf_path": pdf_upload_result.get("minio_path", ""),
                        "analysis_path": analysis_upload_result.get("minio_path", ""),
                        "indexed_in_rag": extracted_text != ""
                    })
                    successful += 1

                finally:
                    cleanup_temp_file(temp_file_path)

            except Exception as e:
                logger.error(f"Ошибка обработки файла {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1

        return BatchProcessingResponse(
            total_files=len(files),
            successful=successful,
            failed=failed,
            results=results
        )

    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка пакетной обработки: {str(e)}"
        )


@app.get("/retrieve/{file_hash}")
async def retrieve_pdf_results(
        file_hash: str,
        include_images: bool = Query(False, description="Включать ли изображения в base64"),
        include_rag: bool = Query(False, description="Включать ли результаты RAG поиска")
):
    """Получение всех результатов обработки PDF по хэшу"""
    try:
        result = {
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat()
        }

        # 1. Получаем оригинальный PDF
        try:
            objects = list(minio_service.client.list_objects(
                "pdf-originals",
                prefix=file_hash,
                recursive=True
            ))

            if objects:
                pdf_obj = minio_service.client.get_object(
                    "pdf-originals",
                    objects[0].object_name
                )
                pdf_info = minio_service.get_file_info("pdf-originals", objects[0].object_name)

                result["pdf_info"] = {
                    "exists": True,
                    "object_name": objects[0].object_name,
                    "metadata": pdf_info.get("metadata", {}),
                    "size": pdf_info.get("size", 0)
                }
            else:
                result["pdf_info"] = {"exists": False}
        except Exception as e:
            result["pdf_info"] = {"exists": False, "error": str(e)}

        # 2. Получаем анализ MinerU
        try:
            objects = list(minio_service.client.list_objects(
                "pdf-analysis",
                prefix=file_hash,
                recursive=True
            ))

            if objects:
                analysis_obj = minio_service.client.get_object(
                    "pdf-analysis",
                    objects[0].object_name
                )
                analysis_data = json.loads(analysis_obj.read())
                analysis_obj.close()
                analysis_obj.release_conn()

                result["mineru_analysis"] = analysis_data

                # Если нужно, извлекаем изображения из анализа
                if include_images and analysis_data.get("images"):
                    result["images"] = analysis_data["images"]
            else:
                result["mineru_analysis"] = {"exists": False}
        except Exception as e:
            result["mineru_analysis"] = {"exists": False, "error": str(e)}

        # 3. Получаем результаты RAG
        if include_rag:
            try:
                # Ищем документ в RAG
                from qdrant_client.http import models as qdrant_models

                filter_condition = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_hash",
                            match=qdrant_models.MatchValue(value=file_hash)
                        )
                    ]
                )

                # Получаем точки из Qdrant
                scroll_result = rag_service.qdrant.client.scroll(
                    collection_name=settings.qdrant.collection_name,
                    scroll_filter=filter_condition,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )

                rag_points = []
                for point in scroll_result[0]:
                    rag_points.append({
                        "id": point.id,
                        "payload": point.payload,
                        "score": getattr(point, 'score', 1.0)
                    })

                result["rag_results"] = {
                    "exists": len(rag_points) > 0,
                    "points_count": len(rag_points),
                    "points": rag_points
                }

            except Exception as e:
                result["rag_results"] = {"exists": False, "error": str(e)}

        return result

    except Exception as e:
        logger.error(f"Ошибка получения результатов PDF {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения результатов: {str(e)}"
        )


@app.delete("/delete/{file_hash}")
async def delete_pdf_data(
        file_hash: str,
        delete_all: bool = Query(True, description="Удалять все связанные данные"),
        include_rag: bool = Query(True, description="Удалять также из RAG")
):
    """Удаление обработанного PDF и всех связанных данных"""
    try:
        deleted_items = []

        # Buckets для удаления
        buckets_to_delete = ["pdf-originals", "pdf-analysis", "extracted-images", "processed-docs"]

        for bucket in buckets_to_delete:
            try:
                objects = list(minio_service.client.list_objects(
                    bucket,
                    prefix=file_hash,
                    recursive=True
                ))

                for obj in objects:
                    minio_service.client.remove_object(bucket, obj.object_name)
                    deleted_items.append(f"{bucket}/{obj.object_name}")

            except Exception as e:
                logger.warning(f"Ошибка удаления из {bucket}: {e}")

        # Удаление из RAG если требуется
        if include_rag:
            try:
                from qdrant_client.http import models as qdrant_models

                # Находим точки с данным file_hash
                filter_condition = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_hash",
                            match=qdrant_models.MatchValue(value=file_hash)
                        )
                    ]
                )

                # Получаем ID точек для удаления
                scroll_result = rag_service.qdrant.client.scroll(
                    collection_name=settings.qdrant.collection_name,
                    scroll_filter=filter_condition,
                    limit=1000,
                    with_payload=False
                )

                point_ids = [point.id for point in scroll_result[0]]

                if point_ids:
                    rag_service.qdrant.client.delete(
                        collection_name=settings.qdrant.collection_name,
                        points_selector=qdrant_models.PointIdsList(
                            points=point_ids
                        )
                    )
                    deleted_items.append(f"rag/{len(point_ids)}_points")

            except Exception as e:
                logger.warning(f"Ошибка удаления из RAG: {e}")

        return {
            "status": "success",
            "message": f"Удалено {len(deleted_items)} объектов",
            "file_hash": file_hash,
            "deleted_items": deleted_items
        }

    except Exception as e:
        logger.error(f"Ошибка удаления PDF {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка удаления: {str(e)}"
        )


@app.post("/reprocess/{file_hash}")
async def reprocess_pdf(
        file_hash: str,
        background_tasks: BackgroundTasks = None,
        force: bool = Query(True, description="Принудительная переобработка")
):
    """Повторная обработка PDF файла"""
    try:
        # Находим оригинальный PDF
        objects = list(minio_service.client.list_objects(
            "pdf-originals",
            prefix=file_hash,
            recursive=True
        ))

        if not objects:
            raise HTTPException(
                status_code=404,
                detail=f"PDF с хэшем {file_hash} не найден"
            )

        # Загружаем оригинальный PDF
        pdf_obj = minio_service.client.get_object(
            "pdf-originals",
            objects[0].object_name
        )
        pdf_data = pdf_obj.read()
        pdf_obj.close()
        pdf_obj.release_conn()

        # Получаем метаданные из оригинального файла
        pdf_info = minio_service.get_file_info("pdf-originals", objects[0].object_name)
        original_metadata = pdf_info.get("metadata", {})

        # Сохраняем во временный файл
        temp_dir = Path("/tmp/rag_reprocess")
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"reprocess_{file_hash}.pdf"
        with open(temp_path, 'wb') as f:
            f.write(pdf_data)

        try:
            # Обрабатываем через MinerU
            mineru_result = mineru_service.analyze_document_with_base64(
                str(temp_path),
                include_original=True,
                include_previews=True
            )

            if mineru_result.get("status") != "success":
                raise Exception(f"MinerU error: {mineru_result.get('error')}")

            # Сохраняем новый анализ (перезаписываем старый)
            analysis_json = json.dumps(mineru_result, ensure_ascii=False, indent=2).encode('utf-8')

            analysis_upload_result = minio_service.upload_file(
                bucket_name="pdf-analysis",
                file_name=f"{file_hash}_analysis_reprocessed.json",
                file_data=analysis_json,
                content_type="application/json",
                metadata={
                    "original_pdf_hash": file_hash,
                    "reprocessed": True,
                    "reprocess_timestamp": datetime.now().isoformat(),
                    "force": force
                }
            )

            # Извлекаем текст для RAG
            extracted_text = ""
            if mineru_result.get("text"):
                extracted_text = mineru_result["text"]

            # Удаляем старые данные из RAG
            try:
                from qdrant_client.http import models as qdrant_models

                filter_condition = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_hash",
                            match=qdrant_models.MatchValue(value=file_hash)
                        )
                    ]
                )

                rag_service.qdrant.client.delete(
                    collection_name=settings.qdrant.collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=filter_condition
                    )
                )
            except Exception as e:
                logger.warning(f"Ошибка удаления старых данных из RAG: {e}")

            # Индексируем в RAG заново
            if extracted_text and background_tasks is not None:
                background_tasks.add_task(
                    index_pdf_in_rag,
                    file_hash,
                    extracted_text,
                    mineru_result,
                    original_metadata
                )

            return {
                "status": "success",
                "message": "PDF успешно переобработан",
                "file_hash": file_hash,
                "analysis_path": analysis_upload_result.get("minio_path", ""),
                "reprocessed": True,
                "indexed_in_rag": extracted_text != ""
            }

        finally:
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка переобработки PDF {file_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка переобработки: {str(e)}"
        )


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware для логирования всех запросов"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.2f}s"
    )

    return response


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers
    )