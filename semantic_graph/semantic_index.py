import logging
import os
import sys
import time
import asyncio
import uuid

from pathlib import Path
from typing import Any, Dict

import aiohttp
from aiohttp import (
    ClientConnectorError,
    ClientResponseError,
    ServerTimeoutError,
)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root for app import

from clasterization import create_communities
from config import (
    API_HOST,
    API_PORT,
    CLUSTERIZATION_SEED,
    COMMUNITY_REPORT_PROMPT,
    EMBEDDING_BASE_URL,
    EMBEDDING_MAX_CONCURRENCY,
    EMBEDDING_TIMEOUT,
    ENTITY_EMBEDDINGS_BATCH_SIZE,
    ENTITY_EMBEDDINGS_COLLECTION,
    ENTITY_EMBEDDINGS_NAMESPACE,
    ENTITY_TYPES,
    MAX_CLUSTER_SIZE,
    MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    USE_LCC,
)
from create_community_report import run_community_reports_pipeline_async
from dtype import (
    DocumentRequest,
    EntitiesRequest,
    EntitiesResponse,
    EntityCreate,
    RelationshipCreate,
)
from graphrag import run_extraction_pipeline_async
from manager import Manager, ManagerConfig
from Qdrant_extractor.dataframe_builder import build_chunks_dataframe
from Qdrant_extractor.qdrant_adapter import QdrantStreamAdapter
from app.src.qwen3_emb_client import EmbeddingClient

load_dotenv()

config = ManagerConfig(
    uri='neo4j://' + os.environ['URL'],
    user=os.environ['USER_NEO4J'],
    password=os.environ['PASSWORD'],
    name_db=os.environ['NAME_DB']
)

doc_manager = Manager(config)

# --- Embedding client ---
emb_client = EmbeddingClient(base_url=EMBEDDING_BASE_URL, timeout=EMBEDDING_TIMEOUT)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Chunk Processing Service")

# --- Вспомогательные функции ---
def _build_response(doc_id: str, total_chunks: int, start_time: float, status: str, extra_stats: dict = None) -> Dict[str, Any]:
    """Формирует стандартизированный ответ API."""
    stats = {
        "total_chunks": total_chunks,
        "processing_time_ms": round((time.time() - start_time) * 1000),
        "status": status
    }
    if extra_stats:
        stats.update(extra_stats)

    return {
        "document_id": doc_id,
        "statistics": stats
    }

# --- Эндпоинт ---
@app.post("/process-document")
async def process_document(request: DocumentRequest) -> Dict[str, Any]:
    start_time = time.time()
    doc_id = request.document_id
    logger.info(f"Starting processing for document: {doc_id}")

    # 1. Загрузка данных
    # ВАЖНО: Если build_chunks_dataframe - это блокирующая I/O операция,
    # ее тоже нужно запускать в отдельном потоке, чтобы не блокировать цикл событий.
    try:
        adapter = QdrantStreamAdapter(base_url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Запускаем синхронный код в потоке, чтобы не блокировать event loop
        df = build_chunks_dataframe(adapter, doc_id=doc_id)
    except Exception as e:
        logger.error(f"Failed to load data from Qdrant: {e}")
        raise HTTPException(status_code=500, detail="Qdrant connection error")

    if df.empty:
        raise HTTPException(status_code=404, detail="Document not found or empty")

    input_df = df[['chunk_id', 'text']].rename(columns={'chunk_id': 'id'})
    total_chunks = input_df.shape[0]

    # 2. Извлечение и суммаризация графа знаний (ИЗМЕНЕНИЕ)
    try:
        # Просто используем await, так как мы в async-функции
        entities, relationships = await run_extraction_pipeline_async(
            text_units=input_df,
            extraction_model=MODEL_NAME,
            summarization_model=MODEL_NAME,
            entity_types=ENTITY_TYPES,
            max_gleanings=0,
            max_summary_length=8000,
            max_input_tokens=8000
        )
    except ValueError as e:
        logger.warning(f"Graph extraction yielded no results: {e}")
        return _build_response(doc_id, total_chunks, start_time, status='completed_without_entities')
    except Exception as e:
        logger.error(f"Error during graph extraction: {e}")
        raise HTTPException(status_code=500, detail="Graph extraction failed")

    # 3. Сохранение в графовую БД
    # Аналогично пункту 1, если doc_manager.add_entities_batch - блокирующая операция
    try:
        save_result = doc_manager.add_entities_batch(
            EntitiesRequest(entities=[EntityCreate(**ent) for ent in entities.to_dict(orient='records')],
                            relationships=[RelationshipCreate(**rel) for rel in
                                           relationships.to_dict(orient='records')]))
        neo4j_status = save_result if isinstance(save_result, EntitiesResponse) else {}
    except Exception as e:
        logger.error(f"Failed to save to Neo4j: {e}")
        raise HTTPException(status_code=500, detail="Failed to save data to Graph DB")


    # 4. Формирование успешного ответа
    return _build_response(
        doc_id=doc_id,
        total_chunks=total_chunks,
        start_time=start_time,
        status='completed',
        extra_stats=neo4j_status
    )

@app.get("/clastrize_graph")
async def clastrize_graph() -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting clusterization for graph")
    relations_df = doc_manager.get_entity_relationships()
    examples = await create_communities(relations_df, max_cluster_size=MAX_CLUSTER_SIZE, use_lcc=USE_LCC, seed=CLUSTERIZATION_SEED)
    print(examples)
    save_result = doc_manager.insert_communities_to_neo4j(examples)
    neo4j_status = save_result if isinstance(save_result, EntitiesResponse) else {}
    return _build_response(
        doc_id='None',
        total_chunks=0,
        start_time=start_time,
        status='completed',
        extra_stats=neo4j_status
    )

@app.get("/create_community_report")
async def create_community_report() -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting create community report")
    community_reports = await run_community_reports_pipeline_async(
        relationships=doc_manager.get_entity_relationships(),
        entities=doc_manager.get_entities(),
        communities=doc_manager.get_community(),
        model=MODEL_NAME,
        prompt=COMMUNITY_REPORT_PROMPT,
        max_input_length=8000,
        max_report_length=2000,
        max_concurrent=4
    )
    save_result = doc_manager.update_community_reports(community_reports)
    neo4j_status = save_result if isinstance(save_result, EntitiesResponse) else {}
    return _build_response(
        doc_id='None',
        total_chunks=0,
        start_time=start_time,
        status='completed',
        extra_stats=neo4j_status
    )

@app.get("/compute_entity_embeddings")
async def compute_entity_embeddings() -> Dict[str, Any]:
    """Compute 2048-dim embeddings for Entity nodes with changed descriptions.

    Flow: Neo4j (read candidates) → EmbeddingClient (sync, via asyncio.to_thread) →
          Qdrant (upsert) → Neo4j (write embedding_updated_at).
    """
    start_time = time.time()
    logger.info("Starting compute_entity_embeddings")

    stats = {
        "embeddings_added": 0,
        "embeddings_updated": 0,
        "embeddings_skipped": 0,
        "embeddings_failed": 0,
    }

    async with aiohttp.ClientSession() as session:
        # 1. Check / create Qdrant collection
        collection_url = f"{QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}"
        try:
            async with session.get(collection_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    vector_config = data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                    if vector_config.get("size") != 2048:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection dimension mismatch: expected 2048, got {vector_config.get('size')}"
                        )
                    if vector_config.get("distance") != "Cosine":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection distance metric mismatch: expected Cosine, got {vector_config.get('distance')}"
                        )
                elif resp.status == 404:
                    create_body = {"vectors": {"size": 2048, "distance": "Cosine"}}
                    async with session.put(collection_url, json=create_body) as create_resp:
                        if create_resp.status not in (200, 201):
                            error_text = await create_resp.text()
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to create Qdrant collection: {error_text}"
                            )
                    logger.info("Created Qdrant collection '%s'", ENTITY_EMBEDDINGS_COLLECTION)
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Qdrant unavailable: {error_text}"
                    )
        except HTTPException:
            raise
        except (ClientConnectorError, ServerTimeoutError) as e:
            raise HTTPException(status_code=500, detail=f"Qdrant unavailable: {e}")

        # 2. Get candidates from Neo4j
        try:
            candidates = doc_manager.get_entities_needing_embedding()
        except Exception as e:
            logger.error("Failed to read candidates from Neo4j: %s", e)
            raise HTTPException(status_code=500, detail="Neo4j read error")

        if not candidates:
            logger.info("No entities needing embedding computation")
            return _build_response("None", 0, start_time, "completed", stats)

        # 3-4. Compute embeddings via EmbeddingClient with Semaphore for concurrency
        semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)

        async def fetch_embedding(entity: dict) -> dict:
            async with semaphore:
                try:
                    embedding = await asyncio.to_thread(
                        emb_client.get_text_embedding, entity["description"]
                    )
                    if len(embedding) != 2048:
                        logger.warning(
                            "Embedding dimension mismatch for entity '%s' (%s): expected 2048, got %d",
                            entity["title"], entity["type"], len(embedding)
                        )
                        return None
                    entity["embedding_vector"] = embedding
                    return entity
                except ValueError as e:
                    logger.warning(
                        "Invalid embedding response for entity '%s' (%s): %s",
                        entity["title"], entity["type"], e
                    )
                    return None
                except Exception as e:
                    logger.warning(
                        "Embedding service error for entity '%s' (%s): %s",
                        entity["title"], entity["type"], e
                    )
                    return None

        # Classify and run in parallel
        for entity in candidates:
            if entity["embedding_updated_at"] is None:
                entity["is_new"] = True
            else:
                entity["is_new"] = False

        tasks = [fetch_embedding(e) for e in candidates]
        results = await asyncio.gather(*tasks)

        # Separate successful from failed
        successful = [r for r in results if r is not None]
        stats["embeddings_failed"] = len(candidates) - len(successful)

        if not successful:
            logger.warning("All embedding requests failed")
            return _build_response("None", 0, start_time, "completed", stats)

        # 5. Batch upsert to Qdrant
        points_url = f"{QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}/points?wait=true"

        x_qdrant_api_key = QDRANT_API_KEY
        headers = {}
        if x_qdrant_api_key:
            headers["api-key"] = x_qdrant_api_key

        for i in range(0, len(successful), ENTITY_EMBEDDINGS_BATCH_SIZE):
            batch = successful[i:i + ENTITY_EMBEDDINGS_BATCH_SIZE]
            points = []
            for entity in batch:
                entity_id = f"{entity['title']}|{entity['type']}"
                points.append({
                    "id": str(uuid.uuid5(ENTITY_EMBEDDINGS_NAMESPACE, entity_id)),
                    "vector": entity["embedding_vector"],
                    "payload": {
                        "entity_title": entity["title"],
                        "entity_type": entity["type"],
                        "entity_id": entity_id,
                        "description": entity["description"],
                    }
                })

            try:
                async with session.put(
                    points_url,
                    json={"points": points},
                    headers=headers
                ) as resp:
                    resp.raise_for_status()
            except (ClientResponseError, ServerTimeoutError, ClientConnectorError) as e:
                logger.error("Qdrant upsert batch %d-%d failed: %s", i, i + len(batch), e)
                stats["embeddings_failed"] += len(batch)
                # Mark all entities in this batch as failed
                for entity in batch:
                    entity["qdrant_failed"] = True
                continue

            # 6. Update Neo4j embedding_updated_at for this batch
            neo4j_batch = [{"title": e["title"], "type": e["type"]} for e in batch if not e.get("qdrant_failed")]
            if not neo4j_batch:
                continue

            try:
                updated_count = doc_manager.set_entity_embedding_updated_at(neo4j_batch)
            except Exception as e:
                logger.error("Neo4j write error for batch %d-%d: %s", i, i + len(batch), e)
                raise HTTPException(status_code=500, detail=f"Neo4j write error: {e}")

            # Update stats
            for entity in batch:
                if not entity.get("qdrant_failed"):
                    if entity.get("is_new"):
                        stats["embeddings_added"] += 1
                    else:
                        stats["embeddings_updated"] += 1

    return _build_response("None", 0, start_time, "completed", stats)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
