#!/usr/bin/env python3
"""
FastAPI сервер для MinerU API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
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

os.environ['MODELSCOPE_CACHE'] = '/home/sunveil/Documents/projects/laba/graph-m-rag/src/mineru/models'
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
    """Модель ответа на обработку документа."""
    task_id: str
    status: str
    message: str
    download_links: Optional[Dict[str, str]] = None
    results: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


tasks = {}


@app.get("/")
async def root():
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
async def health_check():
    return {"status": "healthy", "service": "MinerU API"}


@app.post("/process", response_model=ProcessResponse)
async def process_document(
        file: UploadFile = File(...),
        backend: str = Query("pipeline", description="Бэкенд обработки: pipeline или vlm"),
        method: str = Query("auto", description="Метод обработки: auto, txt, ocr"),
        lang: str = Query("ru", description="Язык документа"),
        formula_enable: bool = Query(True, description="Включить обработку формул"),
        table_enable: bool = Query(True, description="Включить обработку таблиц"),
        start_page: int = Query(0, description="Начальная страница (0-indexed)"),
        end_page: Optional[int] = Query(None, description="Конечная страница"),
        background_tasks: BackgroundTasks = BackgroundTasks()
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

    tasks[task_id] = {
        "status": "processing",
        "temp_dir": temp_dir,
        "results": None,
        "error": None
    }

    background_tasks.add_task(
        process_task,
        task_id,
        file_bytes,
        file_ext,
        config,
        temp_dir
    )

    base_url = os.getenv("BASE_URL", "http://localhost:8001")
    download_links = {
        "status": f"{base_url}/status/{task_id}",
        "api_docs": f"{base_url}/docs"
    }

    return ProcessResponse(
        task_id=task_id,
        status="processing",
        message="Документ принят в обработку",
        download_links=download_links
    )


async def process_task(
        task_id: str,
        file_bytes: bytes,
        file_ext: str,
        config: ProcessingConfig,
        temp_dir: str
):
    try:
        if file_ext == '.pdf':
            result = manager.process_pdf(file_bytes, config, temp_dir)
        else:
            result = manager.process_image(file_bytes)

        if result["success"]:
            tasks[task_id].update({
                "status": "completed",
                "results": result,
                "error": None
            })
        else:
            tasks[task_id].update({
                "status": "failed",
                "results": result,
                "error": result.get("error", "Unknown error")
            })

    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "results": None,
            "error": str(e)
        })

        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    task = tasks[task_id]

    if task["status"] == "completed":
        results = task["results"]
        download_links = {}

        if results.get("format") in ["pipeline", "vlm"] and "results" in results:
            base_url = os.getenv("BASE_URL", "http://localhost:8001")
            files = results["results"]["files"]

            for key, file_path in files.items():
                if key == "images_dir":
                    download_links[key] = f"{base_url}/download/{task_id}/images.zip"
                elif Path(file_path).exists():
                    rel_path = Path(file_path).relative_to(results["results"]["output_dir"])
                    download_links[key] = f"{base_url}/download/{task_id}/{rel_path}"

        # Add images_base64 to the response if available
        response_results = {
            "download_links": download_links,
            "format": results.get("format"),
            "summary": results.get("results", {})  # .get("pdf_info", {})
        }

        if "images_base64" in results.get("results", {}):
            response_results["images_base64"] = results["results"]["images_base64"]

        if results.get("format") == "vlm" and "image_base64" in results.get("results", {}):
            response_results["image_base64"] = results["results"]["image_base64"]

        return StatusResponse(
            task_id=task_id,
            status="completed",
            results=response_results
        )

    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        error=task.get("error")
    )


@app.get("/download/{task_id}/{file_path:path}")
async def download_file(task_id: str, file_path: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Задача еще не завершена")

    results = task["results"]
    if not results or "results" not in results:
        raise HTTPException(status_code=404, detail="Результаты не найдены")

    output_dir = results["results"]["output_dir"]
    full_path = Path(output_dir) / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        full_path.resolve().relative_to(Path(output_dir).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type="application/octet-stream"
    )


@app.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    task = tasks[task_id]
    temp_dir = task.get("temp_dir")

    if temp_dir and Path(temp_dir).exists():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    del tasks[task_id]

    return {"status": "cleaned", "task_id": task_id}

def run_api(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info") -> None:
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )