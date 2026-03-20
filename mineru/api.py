#!/usr/bin/env python3
"""
FastAPI сервер для MinerU API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
import uuid
from pathlib import Path
import json
import uvicorn

from manager import MinerUManager, ProcessingConfig

manager = MinerUManager()

app = FastAPI(
    title="MinerU API",
    description="API для обработки PDF и изображений с использованием MinerU",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


tasks = {}


@app.get("/")
def root():
    """Корневой эндпоинт."""
    return {
        "service": "MinerU API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process",
            "status": "/status/{task_id}",
            "download": "/download/{file_path}",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "MinerU API"}


@app.post("/process", response_model=ProcessResponse)
def process_document(
        file: UploadFile = File(...),
        backend: str = Query("vlm", description="Бэкенд обработки: pipeline или vlm"),
        method: str = Query("auto", description="Метод обработки: auto, txt, ocr"),
        lang: str = Query("ru", description="Язык документа"),
        formula_enable: bool = Query(True, description="Включить обработку формул"),
        table_enable: bool = Query(True, description="Включить обработку таблиц"),
        start_page: int = Query(0, description="Начальная страница (0-indexed)"),
        end_page: Optional[int] = Query(None, description="Конечная страница")
):
    task_id = str(uuid.uuid4())

    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    temp_dir = tempfile.mkdtemp(prefix=f"mineru_{task_id}_")
    file_path = Path(temp_dir) / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {str(e)}")

    config = ProcessingConfig(
        backend=backend,
        method=method,
        lang=lang,
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page_id=start_page,
        end_page_id=end_page
    )

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")

    try:
        if file_ext == '.pdf':
            result = manager.process_pdf(file_bytes, config, temp_dir)
        else:
            result = manager.process_image(file_bytes)

        if result["success"]:
            status = "completed"
            results = result
            error = None
        else:
            status = "failed"
            results = result
            error = result.get("error", "Unknown error")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")

    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    return ProcessResponse(
        status=status,
        message="Документ обработан" if status == "completed" else "Ошибка при обработке документа",
        results={"result": results} if results else None
    )

def run_api(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info") -> None:
    import asyncio
    # Устанавливаем политику цикла для Windows при необходимости
    if os.name == "nt":  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )