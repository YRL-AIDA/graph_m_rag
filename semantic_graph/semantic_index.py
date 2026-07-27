import logging
import os
import sys
import time
import asyncio
import math
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
    COMMUNITY_EMBEDDINGS_BATCH_SIZE,
    COMMUNITY_EMBEDDINGS_COLLECTION,
    COMMUNITY_EMBEDDINGS_NAMESPACE,
    COMMUNITY_REPORT_PROMPT,
    DOCUMENTS_COLLECTION,
    EMBEDDING_BASE_URL,
    EMBEDDING_MAX_CONCURRENCY,
    EMBEDDING_TIMEOUT,
    ENTITY_EMBEDDINGS_BATCH_SIZE,
    ENTITY_EMBEDDINGS_COLLECTION,
    ENTITY_EMBEDDINGS_NAMESPACE,
    ENTITY_TYPES,
    GRAPH_EXTRACTION_PROMPT,
    LLM_API_KEY,
    LLM_URL,
    MAX_CLUSTER_SIZE,
    MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    QUERY_EXTRACTION_API_KEY,
    QUERY_EXTRACTION_LLM_URL,
    QUERY_EXTRACTION_MODEL_NAME,
    QUERY_EXTRACTION_TOKENIZER_URL,
    TOKENIZER_URL,
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
from dtype.search import (
    SearchRequest,
    SearchResponse,
    SearchStatistics,
    TokensBreakdown,
)
from graphrag import AsyncGraphExtractor, AsyncLLMClient, run_extraction_pipeline_async
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



@app.get("/compute_community_embeddings")
async def compute_community_embeddings() -> Dict[str, Any]:
    """Compute 2048-dim embeddings for Community nodes with changed summaries.

    Flow: Neo4j (read candidates) → EmbeddingClient (sync, via asyncio.to_thread) →
          Qdrant (upsert) → Neo4j (write embedding_updated_at).
    """
    start_time = time.time()
    logger.info("Starting compute_community_embeddings")

    stats = {
        "embeddings_added": 0,
        "embeddings_updated": 0,
        "embeddings_skipped": 0,
        "embeddings_failed": 0,
    }

    async with aiohttp.ClientSession() as session:
        # 1. Check / create Qdrant collection
        collection_url = f"{QDRANT_URL}/collections/{COMMUNITY_EMBEDDINGS_COLLECTION}"
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
                    logger.info("Created Qdrant collection '%s'", COMMUNITY_EMBEDDINGS_COLLECTION)
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
            candidates = doc_manager.get_communities_needing_embedding()
        except Exception as e:
            logger.error("Failed to read candidates from Neo4j: %s", e)
            raise HTTPException(status_code=500, detail="Neo4j read error")

        if not candidates:
            logger.info("No communities needing embedding computation")
            return _build_response("None", 0, start_time, "completed", stats)

        # 3-4. Compute embeddings via EmbeddingClient with Semaphore for concurrency
        semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)

        async def fetch_embedding(community: dict) -> dict:
            async with semaphore:
                try:
                    embedding = await asyncio.to_thread(
                        emb_client.get_text_embedding, community["summary"]
                    )
                    if len(embedding) != 2048:
                        logger.warning(
                            "Embedding dimension mismatch for community '%s' (id=%s): expected 2048, got %d",
                            community.get("title"), community.get("id"), len(embedding)
                        )
                        return None
                    community["embedding_vector"] = embedding
                    return community
                except ValueError as e:
                    logger.warning(
                        "Invalid embedding response for community '%s' (id=%s): %s",
                        community.get("title"), community.get("id"), e
                    )
                    return None
                except Exception as e:
                    logger.warning(
                        "Embedding service error for community '%s' (id=%s): %s",
                        community.get("title"), community.get("id"), e
                    )
                    return None

        # Classify and run in parallel
        for community in candidates:
            if community["embedding_updated_at"] is None:
                community["is_new"] = True
            else:
                community["is_new"] = False

        tasks = [fetch_embedding(c) for c in candidates]
        results = await asyncio.gather(*tasks)

        # Separate successful from failed
        successful = [r for r in results if r is not None]
        stats["embeddings_failed"] = len(candidates) - len(successful)

        if not successful:
            logger.warning("All embedding requests failed")
            return _build_response("None", 0, start_time, "completed", stats)

        # 5. Batch upsert to Qdrant
        points_url = f"{QDRANT_URL}/collections/{COMMUNITY_EMBEDDINGS_COLLECTION}/points?wait=true"

        x_qdrant_api_key = QDRANT_API_KEY
        headers = {}
        if x_qdrant_api_key:
            headers["api-key"] = x_qdrant_api_key

        for i in range(0, len(successful), COMMUNITY_EMBEDDINGS_BATCH_SIZE):
            batch = successful[i:i + COMMUNITY_EMBEDDINGS_BATCH_SIZE]
            points = []
            for community in batch:
                community_id_str = str(community["id"])
                points.append({
                    "id": str(uuid.uuid5(COMMUNITY_EMBEDDINGS_NAMESPACE, community_id_str)),
                    "vector": community["embedding_vector"],
                    "payload": {
                        "community_id": community_id_str,
                        "title": community["title"],
                        "level": community["level"],
                        "summary": community["summary"],
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
                for community in batch:
                    community["qdrant_failed"] = True
                continue

            # 6. Update Neo4j embedding_updated_at for this batch
            neo4j_batch = [{"id": c["id"]} for c in batch if not c.get("qdrant_failed")]
            if not neo4j_batch:
                continue

            try:
                updated_count = doc_manager.set_community_embedding_updated_at(neo4j_batch)
            except Exception as e:
                logger.error("Neo4j write error for batch %d-%d: %s", i, i + len(batch), e)
                raise HTTPException(status_code=500, detail=f"Neo4j write error: {e}")

            # Update stats
            for community in batch:
                if not community.get("qdrant_failed"):
                    if community.get("is_new"):
                        stats["embeddings_added"] += 1
                    else:
                        stats["embeddings_updated"] += 1

    return _build_response("None", 0, start_time, "completed", stats)



# --- Вспомогательные функции для /search ---

def _cosine_similarity(a: list, b: list) -> float:
    """Вычисление косинусного сходства между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _run_cypher(cypher: str, **params) -> list:
    """Выполнить параметризованный Cypher-запрос через Neo4j driver."""
    with doc_manager.conn.graph.session(database=doc_manager.name_db) as session:
        result = session.run(cypher, params)
        records = list(result)
        return [dict(r) for r in records]


# --- Эндпоинт POST /search ---

@app.post("/search", response_model=SearchResponse)
async def search_graph(request: SearchRequest) -> SearchResponse:
    """Семантический поиск по графу знаний.

    Возвращает релевантные текстовые блоки, сущности и сообщества
    для заданного вопроса пользователя.
    """
    start_time = time.time()
    question_lower = request.question.lower()
    proportions = request.proportions

    # --- Шаг 0: Инициализация и расчёт бюджетов токенов ---
    text_budget = int(request.max_tokens * proportions.text_units)
    entity_budget = int(request.max_tokens * proportions.entities)
    community_budget = int(request.max_tokens * proportions.communities)

    entity_counters = {"search_miss_count": 0}
    fallback_used = False

    # --- Шаг 1: Предобработка вопроса и извлечение сущностей через LLM ---
    entities_from_question: list = []
    async with AsyncLLMClient(
        base_url=QUERY_EXTRACTION_LLM_URL,
        tokenizer_url=QUERY_EXTRACTION_TOKENIZER_URL,
        api_key=QUERY_EXTRACTION_API_KEY,
    ) as llm:
        extractor = AsyncGraphExtractor(
            llm_client=llm,
            model=QUERY_EXTRACTION_MODEL_NAME,
            max_gleanings=0,
        )
        try:
            entities_df, _relationships_df = await extractor.extract(
                text=question_lower,
                entity_types=ENTITY_TYPES,
                source_id="search_query",
            )
            if not entities_df.empty:
                entities_from_question = entities_df[["title", "type", "description"]].to_dict(orient="records")
        except Exception as e:
            logger.warning("LLM extraction failed for search query: %s", e)
            entities_from_question = []

    entities_extracted_count = len(entities_from_question)

    # --- Шаг 2: Вычисление эмбеддингов ---

    # 2.1 Эмбеддинг вопроса
    try:
        question_embedding = await asyncio.to_thread(
            emb_client.get_text_embedding, question_lower
        )
    except Exception as e:
        logger.error("Failed to compute embedding for question: %s", e)
        raise HTTPException(status_code=500, detail="Failed to compute embedding for question")

    # 2.2 Эмбеддинги сущностей (параллельно)
    async def _compute_entity_embedding(entity: dict) -> dict | None:
        entity_text = f"{entity['title']} ({entity['type']})"
        try:
            embedding = await asyncio.to_thread(emb_client.get_text_embedding, entity_text)
            return {"title": entity["title"], "type": entity["type"], "embedding": embedding}
        except Exception as e:
            logger.warning("Embedding failed for entity '%s' (%s): %s", entity["title"], entity["type"], e)
            return None

    if entities_from_question:
        entity_embedding_results = await asyncio.gather(
            *[_compute_entity_embedding(e) for e in entities_from_question]
        )
        valid_entities = [r for r in entity_embedding_results if r is not None]
    else:
        valid_entities = []

    # --- Шаг 3: Распределение долей токенов между сущностями ---
    N = len(valid_entities)
    if N == 0:
        community_budget += entity_budget
        entity_budget = 0
        fallback_used = True
        per_entity_entity_budget_base = 0
        per_entity_entity_remainder = 0
        per_entity_community_budget_base = 0
        per_entity_community_remainder = 0
    else:
        per_entity_entity_budget_base = entity_budget // N
        per_entity_entity_remainder = entity_budget % N
        per_entity_community_budget_base = community_budget // N
        per_entity_community_remainder = community_budget % N

    # --- Хелпер для Qdrant API-запросов ---
    async with aiohttp.ClientSession() as session:
        qdrant_headers = {}
        if QDRANT_API_KEY:
            qdrant_headers["api-key"] = QDRANT_API_KEY

        # --- Шаг 4: Поиск текстовых блоков из коллекции documents ---
        text_pool: list = []
        remaining_text_budget = text_budget

        search_body: dict = {
            "vector": question_embedding,
            "limit": 100,
            "with_payload": True,
            "with_vector": False,
        }
        if request.documents_filter == "text_only":
            search_body["filter"] = {
                "must": [{"key": "element_type", "match": {"value": "text"}}]
            }

        try:
            async with session.post(
                f"{QDRANT_URL}/collections/{DOCUMENTS_COLLECTION}/points/search",
                json=search_body,
                headers=qdrant_headers,
            ) as resp:
                if resp.status >= 400:
                    raise HTTPException(
                        status_code=500,
                        detail="Qdrant documents collection unavailable",
                    )
                qdrant_result = await resp.json()
        except (ClientConnectorError, ServerTimeoutError) as e:
            raise HTTPException(status_code=500, detail="Qdrant documents collection unavailable") from e

        for point in (qdrant_result.get("result") or []):
            if remaining_text_budget <= 0:
                break
            payload = point.get("payload") or {}
            original_element = payload.get("original_element") or {}
            text = original_element.get("text")
            if not text:
                continue
            token_count = await llm.count_tokens(text, MODEL_NAME)
            if token_count <= remaining_text_budget:
                text_pool.append(text)
                remaining_text_budget -= token_count
            else:
                break

        text_tokens_used = text_budget - remaining_text_budget

        # --- Шаг 5: Поиск по графу для каждой валидной сущности ---
        entity_pools: list = []
        community_pools: list = []

        async def _process_entity(i: int, entity: dict) -> tuple[list, list]:
            """Обработать одну сущность: шаги 5.1–5.5."""
            per_entity_e_budget = per_entity_entity_budget_base + (1 if i < per_entity_entity_remainder else 0)
            per_entity_c_budget = per_entity_community_budget_base + (1 if i < per_entity_community_remainder else 0)
            entity_pool_i: list = []
            community_pool_i: list = []
            ee_budget_rem = per_entity_e_budget
            ec_budget_rem = per_entity_c_budget
            entity_title = entity["title"]
            entity_type = entity["type"]
            

            # 5.1 Qdrant entity_embeddings поиск точки входа
            entry = None
            try:
                async with session.post(
                    f"{QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}/points/search",
                    json={
                        "vector": entity["embedding"],
                        "limit": 1,
                        "with_payload": True,
                        "with_vector": False,
                    },
                    headers=qdrant_headers,
                ) as resp:
                    if resp.status >= 400:
                        raise HTTPException(
                            status_code=500,
                            detail="Qdrant entity_embeddings collection unavailable",
                        )
                    qdrant_res = await resp.json()
                    results = qdrant_res.get("result") or []
                    if results:
                        payload_res = results[0].get("payload") or {}
                        entry = {
                            "title": payload_res.get("entity_title"),
                            "type": payload_res.get("entity_type"),
                        }
            except (ClientConnectorError, ServerTimeoutError) as e:
                raise HTTPException(
                    status_code=500,
                    detail="Qdrant entity_embeddings collection unavailable",
                ) from e

            if entry is None:
                entity_counters["search_miss_count"] += 1
                ec_budget_rem += ee_budget_rem
                ee_budget_rem = 0
                # per-entity fallback (шаг 6)
                fallback_pool = await _fallback_search(
                    session=session,
                    qdrant_headers=qdrant_headers,
                    question_embedding=question_embedding,
                    budget=ec_budget_rem,
                )
                return ([], fallback_pool)

            # 5.2 Neo4j получение ноды входа
            try:
                neo4j_results = await _run_cypher(
                    "MATCH (e:Entity {title: $title, type: $type}) "
                    "RETURN e.title AS title, e.type AS type, e.description AS description",
                    title=entry["title"],
                    type=entry["type"],
                )
            except Exception as e:
                logger.error("Neo4j connection error: %s", e)
                raise HTTPException(status_code=500, detail="Neo4j connection error") from e

            if not neo4j_results:
                # Entity не найдена в Neo4j
                ec_budget_rem += ee_budget_rem
                ee_budget_rem = 0
                # per-entity fallback
                fallback_pool = await _fallback_search(
                    session=session,
                    qdrant_headers=qdrant_headers,
                    question_embedding=question_embedding,
                    budget=ec_budget_rem,
                )
                return ([], fallback_pool)

            entry_node = neo4j_results[0]

            # 5.3 Добавление ноды входа в entity-пул
            entry_desc = entry_node.get("description") or "No description"
            entry_text = f"[{entry_node['title']}] ({entry_node['type']}): {entry_desc}"
            token_count = await llm.count_tokens(entry_text, MODEL_NAME)
            entity_set_i = {(entity_title, entity_type)}
            if token_count <= ee_budget_rem:
                entity_pool_i.append(entry_text)
                ee_budget_rem -= token_count
            entity_set_i.add((entry_node["title"], entry_node["type"]))

            # 5.4 Поиск связанных сущностей
            try:
                related_results = await _run_cypher(
                    "MATCH (entry:Entity {title: $title, type: $type})-[r:RELATED]-(related:Entity) "
                    "RETURN related.title AS title, related.type AS type, "
                    "related.description AS description, r.description AS rel_description",
                    title=entry["title"],
                    type=entry["type"],
                )
            except Exception as e:
                logger.error("Neo4j connection error (RELATED): %s", e)
                raise HTTPException(status_code=500, detail="Neo4j connection error") from e

            if related_results and ee_budget_rem > 0:
                # Эмбеддинги связей (параллельно)
                async def _compute_rel_embedding(rel: dict) -> dict:
                    rel_text = rel.get("rel_description") or ""
                    try:
                        emb = await asyncio.to_thread(emb_client.get_text_embedding, rel_text)
                        rel["rel_embedding"] = emb
                        return rel
                    except Exception as e:
                        logger.warning("Embedding failed for relation: %s", e)
                        rel["rel_embedding"] = None
                        return rel

                related_with_embs = await asyncio.gather(
                    *[_compute_rel_embedding(r) for r in related_results]
                )

                # Cosine similarity сортировка
                for rel in related_with_embs:
                    if rel.get("rel_embedding") is not None:
                        rel["score"] = _cosine_similarity(rel["rel_embedding"], question_embedding)
                    else:
                        rel["score"] = 0.0

                related_with_embs.sort(key=lambda r: r["score"], reverse=True)

                for rel in related_with_embs:
                    if ee_budget_rem <= 0:
                        break
                    desc = rel.get("description") or "No description"
                    rel_desc = rel.get("rel_description") or ""
                    rel_text = f"[{rel['title']}] ({rel['type']}): {desc} | Relation: {rel_desc}"
                    token_count = await llm.count_tokens(rel_text, MODEL_NAME)
                    if token_count <= ee_budget_rem:
                        entity_pool_i.append(rel_text)
                        ee_budget_rem -= token_count
                    entity_set_i.add((rel["title"], rel["type"]))

            # 5.5 Поиск leaf Community
            if ec_budget_rem > 0 and entity_set_i:
                titles_list = [t for t, _ in entity_set_i]
                types_list = [tp for _, tp in entity_set_i]
                try:
                    # Поиск leaf сообществ (имеет IS_CHILD_OF, не имеет IS_PARENT_OF)
                    communities_raw = await _run_cypher(
                        "MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity) "
                        "WHERE e.title IN $titles AND e.type IN $types "
                        "OPTIONAL MATCH (c)-[:IS_PARENT_OF]->(child:Community) "
                        "OPTIONAL MATCH (c)-[:IS_CHILD_OF]->(parent:Community) "
                        "WITH c, collect(DISTINCT child) AS children, collect(DISTINCT parent) AS parents "
                        "WHERE size(children) = 0 AND size(parents) > 0 "
                        "RETURN DISTINCT c.id AS id, c.title AS title, c.summary AS summary, c.level AS level",
                        titles=titles_list,
                        types=types_list,
                    )
                except Exception as e:
                    logger.error("Neo4j connection error (Community): %s", e)
                    raise HTTPException(status_code=500, detail="Neo4j connection error") from e

                if communities_raw:
                    # Подсчёт count_ent для каждого community
                    comm_with_counts = []
                    for comm in communities_raw:
                        try:
                            count_res = await _run_cypher(
                                "MATCH (c:Community {id: $community_id})-[:CONSISTS_OF]->(e:Entity) "
                                "WHERE e.title IN $titles AND e.type IN $types "
                                "RETURN count(e) AS count_ent",
                                community_id=comm["id"],
                                titles=titles_list,
                                types=types_list,
                            )
                            count_ent = count_res[0].get("count_ent", 0) if count_res else 0
                        except Exception:
                            count_ent = 0
                        comm["count_ent"] = count_ent
                        comm_with_counts.append(comm)

                    comm_with_counts.sort(key=lambda c: c["count_ent"], reverse=True)

                    for comm in comm_with_counts:
                        if ec_budget_rem <= 0:
                            break
                        summary = comm.get("summary") or "No summary"
                        comm_text = f"[{comm['title']}]: {summary}"
                        token_count = await llm.count_tokens(comm_text, MODEL_NAME)
                        if token_count <= ec_budget_rem:
                            community_pool_i.append(comm_text)
                            ec_budget_rem -= token_count

            return (entity_pool_i, community_pool_i)

        async def _fallback_search(
            *,
            session: aiohttp.ClientSession,
            qdrant_headers: dict,
            question_embedding: list,
            budget: int,
        ) -> list:
            """Шаг 6: Fallback-путь поиска community."""
            fallback_pool: list = []

            # 6.1 Qdrant поиск корневого community (level=0)
            try:
                async with session.post(
                    f"{QDRANT_URL}/collections/{COMMUNITY_EMBEDDINGS_COLLECTION}/points/search",
                    json={
                        "vector": question_embedding,
                        "limit": 1,
                        "with_payload": True,
                        "with_vector": False,
                        "filter": {
                            "must": [{"key": "level", "match": {"value": 0}}]
                        },
                    },
                    headers=qdrant_headers,
                ) as resp:
                    if resp.status >= 400:
                        raise HTTPException(
                            status_code=500,
                            detail="Qdrant community_embeddings collection unavailable",
                        )
                    qdrant_res = await resp.json()
                    results = qdrant_res.get("result") or []
                    community_id = results[0]["payload"]["community_id"] if results else None
            except (ClientConnectorError, ServerTimeoutError) as e:
                raise HTTPException(
                    status_code=500,
                    detail="Qdrant community_embeddings collection unavailable",
                ) from e

            if community_id is None:
                return []

            # 6.2 Neo4j получение Community
            try:
                neo4j_res = await _run_cypher(
                    "MATCH (c:Community {id: $community_id}) "
                    "RETURN c.title AS title, c.summary AS summary",
                    community_id=community_id,
                )
            except Exception as e:
                logger.error("Neo4j connection error (fallback): %s", e)
                raise HTTPException(status_code=500, detail="Neo4j connection error") from e

            if not neo4j_res:
                return []

            root = neo4j_res[0]

            # 6.3 Добавление корневого community в пул
            summary = root.get("summary") or "No summary"
            root_text = f"[{root['title']}]: {summary}"
            token_count = await llm.count_tokens(root_text, MODEL_NAME)
            if token_count <= budget:
                fallback_pool.append(root_text)
                budget -= token_count

            # 6.4 Поиск дочерних community
            try:
                children_raw = await _run_cypher(
                    "MATCH (root:Community {id: $community_id})-[:IS_PARENT_OF]->(child:Community) "
                    "RETURN child.id AS id, child.title AS title, child.summary AS summary, child.level AS level",
                    community_id=community_id,
                )
            except Exception as e:
                logger.error("Neo4j connection error (children): %s", e)
                raise HTTPException(status_code=500, detail="Neo4j connection error") from e

            if children_raw and budget > 0:
                # 6.5 Получение эмбеддингов дочерних community из Qdrant (параллельно)

                async def _fetch_child_embedding(child: dict) -> dict:
                    point_id = str(uuid.uuid5(COMMUNITY_EMBEDDINGS_NAMESPACE, str(child["id"])))
                    try:
                        async with session.get(
                            f"{QDRANT_URL}/collections/{COMMUNITY_EMBEDDINGS_COLLECTION}/points/{point_id}",
                            params={"with_vector": "true", "with_payload": "false"},
                            headers=qdrant_headers,
                        ) as resp:
                            if resp.status == 404:
                                child["score"] = 0.0
                                return child
                            resp.raise_for_status()
                            data = await resp.json()
                            vector = data.get("result", {}).get("vector")
                            if vector:
                                child["score"] = _cosine_similarity(vector, question_embedding)
                            else:
                                child["score"] = 0.0
                            return child
                    except Exception:
                        child["score"] = 0.0
                        return child

                children_with_scores = await asyncio.gather(
                    *[_fetch_child_embedding(c) for c in children_raw]
                )
                children_with_scores.sort(key=lambda c: c["score"], reverse=True)

                # 6.6 Добавление дочерних community в пул
                for child in children_with_scores:
                    if budget <= 0:
                        break
                    child_summary = child.get("summary") or "No summary"
                    child_text = f"[{child['title']}]: {child_summary}"
                    token_count = await llm.count_tokens(child_text, MODEL_NAME)
                    if token_count <= budget:
                        fallback_pool.append(child_text)
                        budget -= token_count

            return fallback_pool

        # Шаг 5 (продолжение): параллельная обработка всех сущностей

        async def _count_tokens_list(items: list) -> int:
            total = 0
            for item in items:
                total += await llm.count_tokens(item, MODEL_NAME)
            return total

        if not fallback_used and valid_entities:
            process_tasks = [
                _process_entity(i, entity) for i, entity in enumerate(valid_entities)
            ]
            entity_community_results = await asyncio.gather(*process_tasks)
            entity_pools = [r[0] for r in entity_community_results]
            community_pools = [r[1] for r in entity_community_results]
        else:
            entity_pools = []
            community_pools = []

        # --- Шаг 7: Объединение пулов (round-robin) ---
        if not fallback_used and valid_entities:
            # 7.1 Entity round-robin
            merged_entity_pool: list = []
            max_entity_len = max((len(pool) for pool in entity_pools), default=0)
            for round_idx in range(max_entity_len):
                for pool in entity_pools:
                    if round_idx < len(pool):
                        merged_entity_pool.append(pool[round_idx])

            # 7.2 Community round-robin
            merged_community_pool: list = []
            max_comm_len = max((len(pool) for pool in community_pools), default=0)
            for round_idx in range(max_comm_len):
                for pool in community_pools:
                    if round_idx < len(pool):
                        merged_community_pool.append(pool[round_idx])

            # Подсчёт использованных токенов
            total_entity_tokens = await _count_tokens_list(merged_entity_pool)
            total_community_tokens = await _count_tokens_list(merged_community_pool)
            total_text_tokens = text_tokens_used

        else:
            # Глобальный fallback
            merged_entity_pool = []
            fallback_community_pool = await _fallback_search(
                session=session,
                qdrant_headers=qdrant_headers,
                question_embedding=question_embedding,
                budget=community_budget,
            )

            # Подсчёт токенов fallback community pool
            merged_community_pool = fallback_community_pool
            total_entity_tokens = 0
            total_community_tokens = await _count_tokens_list(merged_community_pool)
            total_text_tokens = text_tokens_used

    # --- Шаг 8: Формирование ответа ---
    processing_time_ms = int((time.time() - start_time) * 1000)

    entities_matched = len(valid_entities) - entity_counters["search_miss_count"] if not fallback_used else 0

    return SearchResponse(
        text_units=text_pool,
        entities=merged_entity_pool,
        communities=merged_community_pool,
        statistics=SearchStatistics(
            processing_time_ms=processing_time_ms,
            tokens_used=TokensBreakdown(
                text_units=total_text_tokens,
                entities=total_entity_tokens,
                communities=total_community_tokens,
            ),
            tokens_remaining=TokensBreakdown(
                text_units=text_budget - total_text_tokens,
                entities=entity_budget - total_entity_tokens,
                communities=community_budget - total_community_tokens,
            ),
            entities_extracted_from_question=entities_extracted_count,
            entities_matched_in_graph=entities_matched,
            entities_search_misses=entity_counters["search_miss_count"],
            fallback_used=fallback_used,
            total_items=TokensBreakdown(
                text_units=len(text_pool),
                entities=len(merged_entity_pool),
                communities=len(merged_community_pool),
            ),
        ),
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
