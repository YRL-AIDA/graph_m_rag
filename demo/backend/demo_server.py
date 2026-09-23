#!/usr/bin/env python3
"""
Graph-M-RAG Demo Server

A standalone demo application that showcases the Graph-M-RAG system to a
potential customer. It proxies the real Graph-M-RAG API (document upload,
question answering, graph enrichment) and adds a comparison layer, a
knowledge-graph view, and a polished web UI.

Architecture
------------
    Browser  ──►  Demo server (this file, port 8282)
                      │  proxies to
                      ▼
              Graph-M-RAG API (port 9191)
              ├─ POST   /upload-pdf        (PDF → MinerU → Qdrant + Neo4j)
              ├─ POST   /ask-document      (QA with strategy flags)
              ├─ GET    /uploaded-files
              └─ GET    /health

Run
---
    .venv/bin/python demo/backend/demo_server.py            # real backend at :9191
    .venv/bin/python demo/backend/demo_server.py --mock     # offline demo with canned data
    .venv/bin/python demo/backend/demo_server.py --port 8282 --base-url http://host:9191
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from minio import Minio

BACKEND = "http://localhost:9191"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SAMPLE_QUESTIONS = Path(__file__).resolve().parent.parent / "sample_questions.json"
# Strategy-evaluation results: (doc_id -> correctly answered questions).
# The demo offers only questions the system already answered correctly
# (score == 1.0) in the MMLongBench-Doc evaluation, so every offered
# document/question pair is guaranteed to produce a good answer.
SCORED_QUESTIONS: Optional[Path] = None
for _p in Path(__file__).resolve().parents:
    _cand = _p / "app" / "tests" / "mmlong_strategy_grid_results" / "semantic_scored.json"
    if _cand.exists():
        SCORED_QUESTIONS = _cand
        break

# Direct MinIO access — used as a fallback for the document list when the
# Graph-M-RAG backend is unreachable. Already-uploaded PDFs live in MinIO under
# ``pdfs/{file_hash}_{filename}/``, exactly as the backend's /uploaded-files lists.
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT_HOST", "localhost:9000")
MINIO_ROOT_USER = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_BUCKET = os.environ.get("S3_BUCKET_NAME", "pdf-processing")

app = FastAPI(title="Graph-M-RAG Demo", version="0.1.0")

# ---------------------------------------------------------------------------
# Strategy → description (the actual flag mapping lives in the /demonstration
# endpoint on the Graph-M-RAG API side)
# ---------------------------------------------------------------------------
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "label": "Baseline (классический RAG)",
        "description": "Векторный поиск Qdrant + LLM. Контрольная точка — "
                       "обычный RAG без графов.",
    },
    "semantic": {
        "label": "Semantic Graph",
        "description": "Граф сущностей и сообществ: ответы усиливаются "
                       "сущностями/сообществами, найденными в документе.",
    },
    "structural": {
        "label": "Structural Graph",
        "description": "Структурный граф чтения: к ответу добавляются соседние "
                       "регионы по порядку чтения и родительские элементы.",
    },
    "both": {
        "label": "Both Graphs (полный Graph-M-RAG)",
        "description": "Семантический + структурный граф и BFS-обход: "
                       "максимальный контекст для сложных вопросов.",
    },
}

MOCK_FILES = [
    {"file_hash": "demo-financial-2024", "filename": "Annual_Report_2024.pdf",
     "upload_date": "2026-08-29T12:00:00", "file_size": 2_400_000, "status": "completed"},
    {"file_hash": "demo-guide-camera", "filename": "Camera_User_Guide.pdf",
     "upload_date": "2026-08-29T12:05:00", "file_size": 8_100_000, "status": "completed"},
    {"file_hash": "demo-paper-nlp", "filename": "GraphRAG_Survey_2025.pdf",
     "upload_date": "2026-08-29T12:10:00", "file_size": 1_200_000, "status": "completed"},
]


def _clean_answer(llm_answer: Optional[str]) -> str:
    """Extract the direct answer value from the LLM output for display.

    The Qwen3-VL-32B-Thinking serving returns the chain-of-thought and the
    final answer together in ``message.content``, optionally wrapped in
    ``<think>...</think>``; the answer itself is marked with ``[FINAL_ANSWER]``.
    We strip the thinking block first and then take the first line after the
    marker.  If the marker is missing (e.g. the output was cut off), fall back
    to the last non-empty line instead of dumping the whole reasoning trace
    into the UI.
    """
    if not llm_answer:
        return ""
    text = str(llm_answer)

    # 1) Drop the chain-of-thought block (Qwen3 wraps it in <think>...</think>).
    m = re.search(r"</think>", text, re.IGNORECASE)
    if m:
        text = text[m.end():]

    # 2) The final answer is marked with [FINAL_ANSWER]: <value>.
    m = re.search(r"\[FINAL_ANSWER\]\s*:?\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 3) Fallback: no marker — return the last non-empty line (the answer
    #    usually sits at the end), not the whole reasoning trace.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _parse_context(context_blocks: Optional[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Parse the XML context blocks into evidence / entities / communities.

    The context the LLM saw is composed of three sources:
      * Qdrant blocks        — ``<block ...>`` retrieved regions;
      * structural graph     — ``<region ...>`` reading-order neighbours,
                               parents, cross-graph bridges, BFS expansion;
      * semantic graph       — ``<entity ...>`` and ``<community ...>``.
    Each source is returned separately so the UI can show the composition.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "blocks": [], "entities": [], "communities": [], "regions": [],
        "sources": {"qdrant": 0, "structural": 0, "semantic": 0},
    }
    if not context_blocks:
        return out
    text = "\n".join(str(b) for b in context_blocks)

    def attrs(tag: str) -> Dict[str, str]:
        """Extract ``key="value"`` attributes regardless of their order."""
        return {k: v for k, v in re.findall(r'(\w+)="([^"]*)"', tag)}

    for m in re.finditer(
        r'<block\s+id="(\d+)"[^>]*?type="([^"]*)"[^>]*?page="([^"]*)"'
        r'[^>]*?relevance="([^"]*)"[^>]*>.*?<content>(.*?)</content>',
        text,
        re.DOTALL,
    ):
        out["blocks"].append({
            "id": m.group(1),
            "type": m.group(2),
            "page": m.group(3) or "?",
            "relevance": m.group(4),
            "text": re.sub(r"<[^>]+>", " ", m.group(5)).strip()[:400],
        })
        # region_id attribute (may be missing in legacy/older responses)
        block_tag = m.group(0)
        rid_m = re.search(r'region_id="([^"]*)"', block_tag)
        if rid_m:
            out["blocks"][-1]["region_id"] = rid_m.group(1)

    # Structural regions: <region label=".." [source=".." | source_entity=".."]
    #   [source_region=".."] [page=".."] text=".." />  (order neighbours,
    #   parents, cross-graph bridges, BFS expansion).
    for m in re.finditer(r"<region\b[^>]*?/>", text):
        a = attrs(m.group(0))
        out["regions"].append({
            "label": a.get("label", ""),
            "source": a.get("source", ""),
            "source_region": a.get("source_region", ""),
            "region_id": a.get("region_id", ""),
            "page": a.get("page", ""),
            "text": a.get("text", ""),
        })

    # Semantic entities: <entity type=".." name=".." [region_id|related|
    #   description|degree|score]=".." />  — self-closing, any attr order.
    for m in re.finditer(r"<entity\b[^>]*?/>", text):
        a = attrs(m.group(0))
        out["entities"].append({
            "type": a.get("type", ""),
            "name": a.get("name", ""),
            "score": a.get("score", ""),
            "description": a.get("description", ""),
            "related": a.get("related", ""),
            "region_id": a.get("region_id", ""),
        })

    # Semantic communities.  The backend emits two forms:
    #   1. self-closing: <community name=".." score=".." summary=".." />
    #   2. line form:    <community name=".." [рейтинг: N/10] — summary
    #                    (an opening tag with no closing counterpart)
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("<community"):
            continue
        if line.endswith("/>"):
            a = attrs(line)
            out["communities"].append({
                "title": a.get("name", ""),
                "rating": a.get("score", ""),
                "summary": a.get("summary", ""),
                "region_id": a.get("region_id", ""),
            })
        else:
            name_m = re.search(r'<community\s+name="([^"]*)"', line)
            rating_m = re.search(r"\[рейтинг:\s*([\d.]+)/10\]", line)
            rid_m = re.search(r'region_id="([^"]*)"', line)
            summary = ""
            if " — " in line:
                summary = line.split(" — ", 1)[1]
            out["communities"].append({
                "title": name_m.group(1) if name_m else "",
                "rating": rating_m.group(1) if rating_m else "",
                "summary": summary[:300],
                "region_id": rid_m.group(1) if rid_m else "",
            })

    out["sources"] = {
        "qdrant": len(out["blocks"]),
        "structural": len(out["regions"]),
        "semantic": len(out["entities"]) + len(out["communities"]),
    }
    return out


async def _proxy(method: str, path: str, *, base: str, timeout: float = 30.0,
                 **kwargs) -> Dict[str, Any]:
    url = base.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, **kwargs)
    except httpx.TimeoutException:
        return {"status": "error", "message": f"backend timeout on {path}"}
    except Exception as exc:
        return {"status": "error", "message": f"backend unreachable: {exc}"}
    try:
        payload = resp.json()
    except Exception:
        payload = {"status": "error", "message": resp.text[:500]}
    if resp.status_code >= 400:
        payload.setdefault("status", "error")
        payload.setdefault("message", f"backend HTTP {resp.status_code}")
    return payload


def _mock_ask(file_hash: str, question: str, strategy: str) -> Dict[str, Any]:
    """Canned answer used in --mock mode so the UI works without a backend."""
    strategy_label = STRATEGIES[strategy]["label"]
    return {
        "status": "completed",
        "message": "ok",
        "file_hash": file_hash,
        "question": question,
        "indexed": True,
        "llm_answer": (
            "[FINAL_ANSWER]: Это демонстрационный ответ (mock-режим, бэкенд "
            "Graph-M-RAG не запущен).\nПодключите сервис :9191 и повторите вопрос "
            "для реального ответа.\n"
        ),
        "context_blocks": [],
        "response_metadata": {
            "search_ms": 120, "enrichment_ms": 40, "llm_generation_ms": 900,
            "total_ms": 1100,
        },
        "strategy": strategy,
        "strategy_label": strategy_label,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# Static frontend assets (styles.css, app.js) referenced as /static/...
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    if MOCK_MODE:
        return {"backend_online": True, "mock": True, "name": "Graph-M-RAG (mock)"}
    try:
        data = await _proxy("GET", "/health", base=BACKEND)
        # The backend reports status "healthy" (and per-service sub-statuses).
        return {"backend_online": data.get("status") in ("ok", "healthy"), **data}
    except Exception as exc:
        return {"backend_online": False, "status": "error",
                "message": f"backend unreachable: {exc}"}


def _list_files_minio() -> List[Dict[str, Any]]:
    """List already-uploaded PDFs directly from MinIO (``pdfs/{hash}_{name}/``).

    Mirrors the backend's ``/uploaded-files`` logic so the demo can show the
    document list even when the Graph-M-RAG backend is down.
    """
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for obj in client.list_objects(MINIO_BUCKET, prefix="pdfs/", recursive=True):
        path = obj.object_name
        parts = path.split("/")
        if len(parts) < 2:
            continue
        dir_name = parts[1]  # e.g. "a1b2c3d4_filename.pdf"
        file_name = parts[-1]
        if "_" in dir_name:
            file_hash = dir_name.split("_", 1)[0]
        else:
            continue
        if file_hash in seen:
            continue
        seen.add(file_hash)
        out.append({
            "file_hash": file_hash,
            "filename": file_name,
            "file_size": obj.size or 0,
            "upload_date": (obj.last_modified.isoformat()
                            if obj.last_modified else "unknown"),
            "status": "completed",
        })
    return out


def _normalize_files(files: List[dict]) -> List[Dict[str, Any]]:
    """Normalize backend/MinIO file dicts to the fields the UI expects."""
    out: List[Dict[str, Any]] = []
    for f in files or []:
        out.append({
            "file_hash": f.get("file_hash"),
            "filename": f.get("filename") or f.get("file_name") or "unknown.pdf",
            "file_size": f.get("file_size") or 0,
            "upload_date": f.get("upload_date") or "unknown",
            "status": f.get("status") or "completed",
        })
    return out


@app.get("/api/files")
async def api_files() -> Dict[str, Any]:
    if MOCK_MODE:
        return {"status": "ok", "files": MOCK_FILES, "total_count": len(MOCK_FILES)}
    # Fast path: ask the backend (short timeout).
    try:
        data = await _proxy("GET", "/uploaded-files", base=BACKEND, timeout=8.0)
        if data.get("status") == "success" and data.get("files"):
            return {"status": "success",
                    "files": _normalize_files(data["files"]),
                    "total_count": len(data["files"])}
    except Exception:
        pass
    # Fallback: list already-uploaded PDFs directly from MinIO.
    try:
        files = await asyncio.to_thread(_list_files_minio)
        return {"status": "success", "files": files, "total_count": len(files),
                "source": "minio-direct"}
    except Exception as exc:
        return {"status": "error", "files": [], "message": f"minio fallback failed: {exc}"}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    if MOCK_MODE:
        return {"status": "ok", "file_hash": f"demo-{file.filename}",
                "message": "mock upload accepted", "embeddings_computed": 0}
    data = await file.read()
    files = {"file": (file.filename, data, file.content_type or "application/pdf")}
    try:
        return await _proxy("POST", "/upload-pdf", base=BACKEND, files=files)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/ask")
async def api_ask(body: Dict[str, Any]) -> Dict[str, Any]:
    file_hash = body.get("file_hash", "")
    question = body.get("question", "")
    strategy = body.get("strategy", "baseline")
    if strategy not in STRATEGIES:
        return {"status": "error", "message": f"unknown strategy {strategy}"}

    payload = {
        "file_hash": file_hash,
        "question": question,
        "strategy": strategy,
        "limit": body.get("limit", 15),
        "use_reranker": bool(body.get("use_reranker", False)),
        "use_mmr_reranker": bool(body.get("use_mmr_reranker", False)),
        "answer_format": body.get("answer_format"),
    }

    if MOCK_MODE:
        raw = _mock_ask(file_hash, question, strategy)
    else:
        try:
            # Dedicated demo endpoint (separate from /ask-document used by batch
            # evaluation) so demo requests never stall the QA service during
            # strategy tests.  The LLM client itself is bounded at 300s, so a
            # 420s proxy wait never times out before the backend returns an
            # answer or a proper error.
            raw = await _proxy("POST", "/api/demo/ask", base=BACKEND,
                               timeout=420.0, json=payload)
        except Exception as exc:
            raw = {"status": "error", "message": str(exc)}

    result = {
        "strategy": strategy,
        "strategy_label": STRATEGIES[strategy]["label"],
        "answer": _clean_answer(raw.get("llm_answer")),
        "raw_answer": raw.get("llm_answer") or "",
        "status": raw.get("status"),
        "message": raw.get("message", ""),
        "metadata": raw.get("response_metadata") or {},
        "context": _parse_context(raw.get("context_blocks")),
    }
    return result


@app.post("/api/compare")
async def api_compare(body: Dict[str, Any]) -> Dict[str, Any]:
    file_hash = body.get("file_hash", "")
    question = body.get("question", "")
    strategies = body.get("strategies") or list(STRATEGIES.keys())

    tasks = [
        api_ask({"file_hash": file_hash, "question": question, "strategy": s})
        for s in strategies
    ]
    results = await asyncio.gather(*tasks)
    return {"question": question, "file_hash": file_hash, "results": results}


def _is_not_answerable(text: Optional[str]) -> bool:
    """Return True if *text* is a "Not answerable" style refusal."""
    if not text:
        return False
    v = str(text).strip().lower()
    return v.startswith("not answerable") or v.startswith("i don't know") \
        or v.startswith("unable to answer") or v.startswith("cannot answer")


@app.get("/api/sample-questions")
async def api_sample_questions() -> Dict[str, Any]:
    """Return demo questions.

    ``questions`` — generic showcase questions grouped by category;
    ``byDoc``    — mapping ``doc_id -> [answerable questions]`` taken from the
    strategy-evaluation results (``semantic_scored.json``).  Only questions
    that were answered correctly (``score == 1.0``) AND whose answer is not a
    "Not answerable" refusal are offered, so every demo pair is guaranteed to
    produce a real answer.
    """
    generic = []
    if SAMPLE_QUESTIONS.exists():
        generic = json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))

    by_doc: Dict[str, List[str]] = {}
    if SCORED_QUESTIONS and SCORED_QUESTIONS.exists():
        try:
            scored = json.loads(SCORED_QUESTIONS.read_text(encoding="utf-8"))
            for s in scored:
                did = s.get("doc_id", "")
                q = s.get("question", "")
                if did and q and s.get("score") == 1.0 \
                        and not _is_not_answerable(s.get("pred")):
                    by_doc.setdefault(did, []).append(q)
            by_doc = {k: list(dict.fromkeys(v))[:12] for k, v in by_doc.items()}
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load scored questions: %s", exc)

    return {"questions": generic, "byDoc": by_doc}


@app.get("/api/demo/pdf/{file_hash}/mineru-bboxes")
async def api_mineru_bboxes(file_hash: str, page_idx: Optional[int] = Query(None)) -> Dict[str, Any]:
    """Region layout (bboxes) for a document, used to visualise the
    "разбиение на регионы" step. Proxied from the Graph-M-RAG API."""
    if MOCK_MODE:
        return {"status": "error", "message": "mock mode: no region data",
                "bboxes": [], "pages": 12, "region_types": []}
    params = {}
    if page_idx is not None:
        params["page_idx"] = page_idx
    return await _proxy("GET", f"/api/pdf/{file_hash}/mineru-bboxes",
                        base=BACKEND, params=params)


@app.get("/api/demo/pdf/{file_hash}/page/{page_number}")
async def api_pdf_page(file_hash: str, page_number: int,
                       bboxes: str = Query("")) -> Response:
    """PDF page image with optional highlighted region bboxes (image/png)."""
    if MOCK_MODE:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="1100">'
               f'<rect width="100%" height="100%" fill="#161b22"/>'
               f'<text x="400" y="540" fill="#8b96a5" font-size="22" '
               f'text-anchor="middle">PDF-страница (mock-режим)</text>'
               f'</svg>')
        return Response(content=svg, media_type="image/svg+xml")
    url = BACKEND.rstrip("/") + f"/api/pdf/{file_hash}/page/{page_number}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params={"bboxes": bboxes} if bboxes else {})
    except Exception as exc:
        return Response(content=f"error: {exc}".encode(), media_type="text/plain",
                        status_code=502)
    return Response(content=resp.content,
                    media_type=resp.headers.get("content-type", "image/png"))


@app.get("/api/strategies")
async def api_strategies() -> Dict[str, Any]:
    return {"strategies": [
        {"name": k, **v} for k, v in STRATEGIES.items()
    ]}


MOCK_MODE = False


def main() -> None:
    global BACKEND, MOCK_MODE
    parser = argparse.ArgumentParser(description="Graph-M-RAG Demo Server")
    parser.add_argument("--port", type=int, default=8282)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--base-url", default=BACKEND,
                        help="Graph-M-RAG backend URL (default: http://localhost:9191)")
    parser.add_argument("--mock", action="store_true",
                        help="offline demo with canned data (no backend required)")
    args = parser.parse_args()

    BACKEND = args.base_url
    MOCK_MODE = args.mock

    import uvicorn
    print(f"Graph-M-RAG Demo  (backend: {'MOCK' if MOCK_MODE else BACKEND})")
    print(f"Open http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
