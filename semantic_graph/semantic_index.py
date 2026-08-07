import functools
import logging
import os
import re
import sys
import time
import asyncio
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

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

from clasterization import clastrize_graph as _clastrize_graph_impl, create_communities
from config import (
    ADAPTIVE_BUDGET_ALLOCATION,
    API_HOST,
    API_PORT,
    CLUSTERIZATION_SEED,
    COMMUNITY_EMBEDDINGS_BATCH_SIZE,
    COMMUNITY_EMBEDDINGS_COLLECTION,
    COMMUNITY_EMBEDDINGS_NAMESPACE,
    COMMUNITY_REPORT_PROMPT,
    DOCUMENTS_COLLECTION,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSION,
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
    ENTITY_QUERY_EXPANSION_ENABLED,
    ENTITY_QUERY_EXPANSION_TOP_K,
    ENTITY_QUERY_EXPANSION_SIM_THRESHOLD,
    HYBRID_SEARCH_ENABLED,
    HYBRID_SEARCH_ALPHA,
)
# Resolve expected embedding dimension: explicit config or default 2048
_EMBEDDING_DIM = EMBEDDING_DIMENSION if EMBEDDING_DIMENSION > 0 else 2048
from create_community_report import run_community_reports_pipeline_async
from embeddings import cosine_similarity
from metrics import compute_graph_metrics
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
    uri='bolt://' + os.environ['URL'],
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

    # 2.5. Compute per-run extraction stats
    extraction_stats: Dict[str, Any] = {}
    try:
        from metrics import compute_extraction_stats, evaluate_entity_quality
        extraction_stats = compute_extraction_stats(input_df, entities, relationships)
        quality_stats = evaluate_entity_quality(entities)
        extraction_stats.update(quality_stats)
    except Exception as e:
        logger.warning("Extraction stats computation failed: %s", e)

    # 3. Сохранение в графовую БД
    # Аналогично пункту 1, если doc_manager.add_entities_batch - блокирующая операция
    try:
        save_result = doc_manager.add_entities_batch(
            EntitiesRequest(entities=[EntityCreate(**ent) for ent in entities.to_dict(orient='records')],
                            relationships=[RelationshipCreate(**rel) for rel in
                                           relationships.to_dict(orient='records')]))
        neo4j_status = save_result.model_dump() if isinstance(save_result, EntitiesResponse) else {}
    except Exception as e:
        logger.error(f"Failed to save to Neo4j: {e}")
        raise HTTPException(status_code=500, detail="Failed to save data to Graph DB")

    # 3.5. Document-scoped clustering
    cluster_stats = {}
    try:
        result = await _clastrize_graph_impl(
            manager=doc_manager,
            max_cluster_size=MAX_CLUSTER_SIZE,
            document_id=doc_id,
        )
        if result is not None:
            communities, inter_edges = result
            cluster_stats = doc_manager.insert_communities_to_neo4j(
                communities, document_id=doc_id
            )
            if inter_edges:
                inter_stats = doc_manager.save_inter_community_links(
                    inter_edges, document_id=doc_id
                )
                cluster_stats["inter_community_links"] = inter_stats.get(
                    "links_created", 0
                )
    except Exception as e:
        logger.warning(
            "Document-scoped clustering failed for '%s': %s",
            doc_id,
            e,
        )
        # Non-fatal: clustering failure should not prevent the document
        # from being processed successfully.

    # 4. Compute graph quality metrics
    metric_stats = {}
    try:
        metric_stats = await compute_graph_metrics(doc_manager)
    except Exception as e:
        logger.warning("Graph metrics computation failed for '%s': %s", doc_id, e)

    # 4.5. Auto-update bridge connections between structural and semantic graphs
    bridge_stats = {}
    try:
        import asyncio as _asyncio
        from connect_graphs import main as connect_graphs_main
        await _asyncio.to_thread(connect_graphs_main, document_id=doc_id)
        bridge_stats = {"bridge_connections": "updated"}
    except Exception as e:
        logger.warning(
            "Bridge connection update failed for '%s': %s", doc_id, e
        )

    # 5. Формирование успешного ответа
    if extraction_stats:
        neo4j_status["extraction_stats"] = extraction_stats
    if cluster_stats:
        neo4j_status["cluster_stats"] = cluster_stats
    response = _build_response(
        doc_id=doc_id,
        total_chunks=total_chunks,
        start_time=start_time,
        status='completed',
        extra_stats=neo4j_status
    )
    if metric_stats:
        response["metrics"] = metric_stats
    if bridge_stats:
        response["bridge_stats"] = bridge_stats
    return response

@app.get("/clastrize_graph")
async def clastrize_graph(document_id: Optional[str] = None) -> Dict[str, Any]:
    start_time = time.time()
    if document_id is not None:
        logger.info(
            "Starting document-scoped clusterization for document '%s'",
            document_id,
        )
    else:
        logger.info("Starting global clusterization for entire graph")

    result = await _clastrize_graph_impl(
        manager=doc_manager,
        max_cluster_size=MAX_CLUSTER_SIZE,
        document_id=document_id,
    )
    if result is None:
        return _build_response(
            doc_id=document_id or "None",
            total_chunks=0,
            start_time=start_time,
            status="completed_without_communities",
        )
    examples, inter_edges = result
    logger.debug("Community examples: %s", examples)
    save_result = doc_manager.insert_communities_to_neo4j(
        examples, document_id=document_id
    )
    neo4j_status = save_result if isinstance(save_result, EntitiesResponse) else {}

    # Save inter-community links
    if inter_edges:
        inter_stats = doc_manager.save_inter_community_links(
            inter_edges, document_id=document_id
        )
        neo4j_status["inter_community_links"] = inter_stats.get("links_created", 0)

    # Compute post-clustering metrics
    metric_stats = {}
    try:
        metric_stats = await compute_graph_metrics(doc_manager)
    except Exception as e:
        logger.warning("Graph metrics computation failed after clustering: %s", e)

    response = _build_response(
        doc_id=document_id or "None",
        total_chunks=0,
        start_time=start_time,
        status="completed",
        extra_stats=neo4j_status,
    )
    if metric_stats:
        response["metrics"] = metric_stats
    return response

@app.get("/create_community_report")
async def create_community_report() -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting create community report")
    existing_report_hashes = doc_manager.get_community_report_hashes()
    community_reports = await run_community_reports_pipeline_async(
        relationships=doc_manager.get_entity_relationships(),
        entities=doc_manager.get_entities(),
        communities=doc_manager.get_community(),
        model=MODEL_NAME,
        prompt=COMMUNITY_REPORT_PROMPT,
        max_input_length=8000,
        max_report_length=2000,
        max_concurrent=4,
        existing_report_hashes=existing_report_hashes,
    )
    save_result = doc_manager.update_community_reports(community_reports)
    neo4j_status = save_result if isinstance(save_result, EntitiesResponse) else {}
    neo4j_status["incremental_hash_count"] = len(existing_report_hashes)
    return _build_response(
        doc_id='None',
        total_chunks=0,
        start_time=start_time,
        status='completed',
        extra_stats=neo4j_status
    )


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Compute observability metrics for the knowledge graph.

    Returns graph-level statistics from Neo4j including entity/relationship
    counts, community sizes, orphan entities, and type distributions.
    """
    start_time = time.time()
    logger.info("Computing graph metrics")
    try:
        metrics = await compute_graph_metrics(doc_manager)
    except Exception as e:
        logger.error("Failed to compute graph metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {e}")
    processing_time_ms = round((time.time() - start_time) * 1000)
    return {
        "metrics": metrics,
        "processing_time_ms": processing_time_ms,
    }




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
                    if vector_config.get("size") != _EMBEDDING_DIM:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection dimension mismatch: expected {_EMBEDDING_DIM}, got {vector_config.get('size')}"
                        )
                    if vector_config.get("distance") != "Cosine":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection distance metric mismatch: expected Cosine, got {vector_config.get('distance')}"
                        )
                elif resp.status == 404:
                    create_body = {"vectors": {"size": _EMBEDDING_DIM, "distance": "Cosine"}}
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
                    if len(embedding) != _EMBEDDING_DIM:
                        logger.warning(
                            "Embedding dimension mismatch for entity '%s' (%s): expected %d, got %d",
                            entity["title"], entity["type"], _EMBEDDING_DIM, len(embedding)
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
                    if vector_config.get("size") != _EMBEDDING_DIM:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection dimension mismatch: expected {_EMBEDDING_DIM}, got {vector_config.get('size')}"
                        )
                    if vector_config.get("distance") != "Cosine":
                        raise HTTPException(
                            status_code=500,
                            detail=f"Qdrant collection distance metric mismatch: expected Cosine, got {vector_config.get('distance')}"
                        )
                elif resp.status == 404:
                    create_body = {"vectors": {"size": _EMBEDDING_DIM, "distance": "Cosine"}}
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
                    if len(embedding) != _EMBEDDING_DIM:
                        logger.warning(
                            "Embedding dimension mismatch for community '%s' (id=%s): expected %d, got %d",
                            community.get("title"), community.get("id"), _EMBEDDING_DIM, len(embedding)
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


# ---------------------------------------------------------------------------
# C3: Hybrid dense + keyword scoring helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lowercase alphanumeric token set for keyword matching."""
    return set(re.findall(r"\w+", text.lower()))


def _keyword_score(query_tokens: set, doc_text: str, avgdl: float = 200.0,
                   k1: float = 1.2, b: float = 0.75) -> float:
    """BM25-alike keyword relevance score.

    Parameters
    ----------
    query_tokens:
        Lowercase token set of the question.
    doc_text:
        The text block to score.
    avgdl:
        Estimated average document length in words (tunable).
    k1, b:
        BM25 hyper-parameters.

    Returns
    -------
    float
        Normalised score in [0, 1].
    """
    if not query_tokens or not doc_text:
        return 0.0

    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0

    dl = len(doc_tokens)
    # IDF-like: how many query tokens are present
    matches = query_tokens & doc_tokens
    if not matches:
        return 0.0

    idf_sum = sum(
        math.log(1.0 + (1.0 / max(1, doc_tokens.count(t))))  # rough IDF
        for t in matches
    )
    # BM25 term scoring with smooth saturation
    score = 0.0
    for t in matches:
        tf = doc_tokens.count(t)
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * dl / max(1.0, avgdl))
        score += idf_sum * numerator / max(1.0, denominator)
    # Normalise to [0, 1]
    return min(1.0, score / max(1.0, len(query_tokens)))


def _hybrid_score(dense_score: float, keyword_score: float, alpha: float) -> float:
    """Combine dense and sparse scores.

    ``alpha=1.0`` → dense-only, ``alpha=0.0`` → keyword-only.
    """
    return alpha * dense_score + (1.0 - alpha) * keyword_score


@app.post("/search", response_model=SearchResponse)
async def search_graph(request: SearchRequest) -> SearchResponse:
    """Semantic search over the knowledge graph.

    Returns relevant text blocks, entities, and communities
    for a given user question.
    """
    start_time = time.time()
    question_lower = request.question.lower()
    proportions = request.proportions

    # --- Step 0: Initialise and calculate token budgets ---
    text_budget = int(request.max_tokens * proportions.text_units)
    entity_budget = int(request.max_tokens * proportions.entities)
    community_budget = int(request.max_tokens * proportions.communities)

    if text_budget <= 0:
        logger.warning(f"Text budget is {text_budget}; search request may return empty results. "
                       f"Consider increasing the budget or reducing context.")

    entity_counters = {"search_miss_count": 0}
    fallback_used = False

    # --- Step 1: Preprocess question and extract entities via LLM ---
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

        # --- Step 2: Compute embeddings ---

        # 2.1 Question embedding (C7: LRU-cached to avoid recomputation)
        @functools.lru_cache(maxsize=256)
        def _cached_question_embedding(text: str) -> list:
            return emb_client.get_text_embedding(text)

        try:
            question_embedding = await asyncio.to_thread(
                _cached_question_embedding, question_lower
            )
        except Exception as e:
            logger.error("Failed to compute embedding for question: %s", e)
            raise HTTPException(status_code=500, detail="Failed to compute embedding for question")

        # 2.2 Entity embeddings (parallel)
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

        # --- Step 2.3: Entity-aware query expansion via Qdrant ---
        expanded_entity_count = 0
        if ENTITY_QUERY_EXPANSION_ENABLED and valid_entities:
            seen_titles = {f"{e['title']}|{e['type']}" for e in valid_entities}
            try:
                async with aiohttp.ClientSession() as session:
                    search_url = f"{QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}/points/search"
                    qdrant_headers = {}
                    if QDRANT_API_KEY:
                        qdrant_headers["api-key"] = QDRANT_API_KEY
                    for entity in valid_entities:
                        try:
                            async with session.post(
                                search_url,
                                json={
                                    "vector": entity["embedding"],
                                    "limit": ENTITY_QUERY_EXPANSION_TOP_K + 1,  # +1 to filter self
                                    "with_payload": True,
                                },
                                headers=qdrant_headers,
                            ) as resp:
                                if resp.status != 200:
                                    continue
                                result = await resp.json()
                                for hit in result.get("result", []):
                                    score = hit.get("score", 0)
                                    if score < ENTITY_QUERY_EXPANSION_SIM_THRESHOLD:
                                        continue
                                    payload = hit.get("payload", {})
                                    key = f"{payload.get('entity_title','')}|{payload.get('entity_type','')}"
                                    if key in seen_titles:
                                        continue
                                    seen_titles.add(key)
                                    # Embed the expansion entity
                                    exp_emb = await asyncio.to_thread(
                                        emb_client.get_text_embedding, key
                                    )
                                    valid_entities.append({
                                        "title": payload.get("entity_title", ""),
                                        "type": payload.get("entity_type", ""),
                                        "embedding": exp_emb,
                                        "is_expanded": True,
                                        "expansion_score": score,
                                    })
                                    expanded_entity_count += 1
                        except Exception:
                            continue
                if expanded_entity_count:
                    logger.info(
                        "Query expansion added %d entities from Qdrant (top_k=%d, threshold=%.2f)",
                        expanded_entity_count,
                        ENTITY_QUERY_EXPANSION_TOP_K,
                        ENTITY_QUERY_EXPANSION_SIM_THRESHOLD,
                    )
            except Exception as e:
                logger.warning("Query expansion failed: %s", e)

        # --- Step 3: Distribute token shares across entities ---
        N = len(valid_entities)
        if N == 0:
            community_budget += entity_budget
            entity_budget = 0
            fallback_used = True
            per_entity_entity_budgets: list = []
            per_entity_community_budgets: list = []
        elif ADAPTIVE_BUDGET_ALLOCATION:
            # Proportional allocation based on cosine similarity to question
            scores = [
                # score in [0, 1]: clamp negative values to 0
                max(0.0, cosine_similarity(e["embedding"], question_embedding))
                for e in valid_entities
            ]
            total_score = sum(scores) + 1e-8
            # Allocate entity budget proportionally, minimum floor of 1 token
            per_entity_entity_budgets = [
                max(1, int(entity_budget * s / total_score)) for s in scores
            ]
            # Normalize to prevent exceeding total (Redistribute remainder)
            total_allocated = sum(per_entity_entity_budgets)
            if total_allocated > entity_budget:
                overshoot = total_allocated - entity_budget
                for i in sorted(range(N), key=lambda i: per_entity_entity_budgets[i], reverse=True):
                    if overshoot <= 0:
                        break
                    if per_entity_entity_budgets[i] > 1:
                        reduction = min(overshoot, per_entity_entity_budgets[i] - 1)
                        per_entity_entity_budgets[i] -= reduction
                        overshoot -= reduction
            # Community budget: proportional allocation
            per_entity_community_budgets = [
                max(1, int(community_budget * s / total_score)) for s in scores
            ]
            total_allocated_c = sum(per_entity_community_budgets)
            if total_allocated_c > community_budget:
                overshoot_c = total_allocated_c - community_budget
                for i in sorted(range(N), key=lambda i: per_entity_community_budgets[i], reverse=True):
                    if overshoot_c <= 0:
                        break
                    if per_entity_community_budgets[i] > 1:
                        reduction = min(overshoot_c, per_entity_community_budgets[i] - 1)
                        per_entity_community_budgets[i] -= reduction
                        overshoot_c -= reduction
        else:
            # Legacy: equal distribution
            per_entity_entity_budget_base = entity_budget // N
            per_entity_entity_remainder = entity_budget % N
            per_entity_entity_budgets = [
                per_entity_entity_budget_base + (1 if i < per_entity_entity_remainder else 0)
                for i in range(N)
            ]
            per_entity_community_budget_base = community_budget // N
            per_entity_community_remainder = community_budget % N
            per_entity_community_budgets = [
                per_entity_community_budget_base + (1 if i < per_entity_community_remainder else 0)
                for i in range(N)
            ]

        # --- Helper for Qdrant API requests ---
        async with aiohttp.ClientSession() as session:
            qdrant_headers = {}
            if QDRANT_API_KEY:
                qdrant_headers["api-key"] = QDRANT_API_KEY

            # --- Step 4: Search text blocks from the documents collection ---
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
    
            # --- Hybrid re-ranking (C3): combine dense + keyword scores ---
            query_tokens: Optional[set] = None
            if HYBRID_SEARCH_ENABLED:
                query_tokens = _tokenize(request.question)

            candidates: list = []  # (combined_score, dense_score, text, token_count)
            for point in (qdrant_result.get("result") or []):
                payload = point.get("payload") or {}
                original_element = payload.get("original_element") or {}
                text = original_element.get("text")
                if not text:
                    continue
                dense_score = float(point.get("score", 0.0))
                if HYBRID_SEARCH_ENABLED and query_tokens:
                    kw_score = _keyword_score(query_tokens, text)
                    combined = _hybrid_score(dense_score, kw_score, HYBRID_SEARCH_ALPHA)
                else:
                    combined = dense_score
                token_count = await llm.count_tokens(text, MODEL_NAME)
                candidates.append((combined, dense_score, text, token_count))

            # Sort by combined score descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            for _, _, text, token_count in candidates:
                if remaining_text_budget <= 0:
                    break
                if token_count <= remaining_text_budget:
                    text_pool.append(text)
                    remaining_text_budget -= token_count
                else:
                    continue  # skip oversized blocks, try next

            text_tokens_used = text_budget - remaining_text_budget
    
            # --- Step 5: Graph search for each valid entity ---
            entity_pools: list = []
            community_pools: list = []
    
            async def _process_entity(i: int, entity: dict) -> tuple[list, list]:
                """Process one entity: steps 5.1–5.5."""
                per_entity_e_budget = per_entity_entity_budgets[i]
                per_entity_c_budget = per_entity_community_budgets[i]
                entity_pool_i: list = []
                community_pool_i: list = []
                ee_budget_rem = per_entity_e_budget
                ec_budget_rem = per_entity_c_budget
                entity_title = entity["title"]
                entity_type = entity["type"]
    
                # 5.1 Qdrant entity_embeddings search for entry point
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
                    # per-entity fallback (step 6)
                    fallback_pool = await _fallback_search(
                        session=session,
                        qdrant_headers=qdrant_headers,
                        question_embedding=question_embedding,
                        budget=ec_budget_rem,
                    )
                    return ([], fallback_pool)
    
                # 5.2 Neo4j get entry node
                try:
                    neo4j_results = await doc_manager.run_cypher(
                        "MATCH (e:Entity {title: $title, type: $type}) "
                        "RETURN e.title AS title, e.type AS type, e.description AS description",
                        title=entry["title"],
                        type=entry["type"],
                    )
                except Exception as e:
                    logger.error("Neo4j connection error: %s", e)
                    raise HTTPException(status_code=500, detail="Neo4j connection error") from e
    
                if not neo4j_results:
                    # Entity not found in Neo4j
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
    
                # 5.3 Add entry node to entity pool
                entry_desc = entry_node.get("description") or "No description"
                entry_text = f"[{entry_node['title']}] ({entry_node['type']}): {entry_desc}"
                token_count = await llm.count_tokens(entry_text, MODEL_NAME)
                entity_set_i = {(entity_title, entity_type)}
                if token_count <= ee_budget_rem:
                    entity_pool_i.append(entry_text)
                    ee_budget_rem -= token_count
                entity_set_i.add((entry_node["title"], entry_node["type"]))
    
                # 5.4 Find related entities (Strategy B: 1-hop + 2-hop expansion)
                try:
                    related_results = doc_manager.get_related_entities_2hop(
                        entity_title=entry["title"],
                        entity_type=entry["type"],
                        max_degree=20,
                        limit_1hop=20,
                        limit_2hop=20,
                    )
                except Exception as e:
                    logger.error("Neo4j connection error (RELATED 2-hop): %s", e)
                    raise HTTPException(status_code=500, detail="Neo4j connection error") from e
    
                if related_results and ee_budget_rem > 0:
                    # Relation embeddings (parallel)
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
    
                    # Cosine similarity sort, weighting hop_depth: 1-hop → ×1.0, 2-hop → ×0.6
                    for rel in related_with_embs:
                        if rel.get("rel_embedding") is not None:
                            rel["score"] = cosine_similarity(rel["rel_embedding"], question_embedding)
                        else:
                            rel["score"] = 0.0
                        # Down-weight 2-hop results
                        if rel.get("hop_depth", 1) >= 2:
                            rel["score"] *= 0.6
    
                    related_with_embs.sort(key=lambda r: r["score"], reverse=True)
    
                    for rel in related_with_embs:
                        if ee_budget_rem <= 0:
                            break
                        desc = rel.get("description") or "No description"
                        rel_desc = rel.get("rel_description") or ""
                        hop_str = f" [depth:{rel.get('hop_depth', 1)}]" if rel.get("hop_depth", 1) >= 2 else ""
                        rel_text = f"[{rel['title']}] ({rel['type']}){hop_str}: {desc} | Relation: {rel_desc}"
                        token_count = await llm.count_tokens(rel_text, MODEL_NAME)
                        if token_count <= ee_budget_rem:
                            entity_pool_i.append(rel_text)
                            ee_budget_rem -= token_count
                        entity_set_i.add((rel["title"], rel["type"]))
    
                # 5.4b Find community siblings (Strategy C: sibling entities)
                if ec_budget_rem > 0 and entity_set_i:
                    try:
                        siblings_info = doc_manager.get_community_siblings(
                            entity_title=entry["title"],
                            entity_type=entry["type"],
                            max_siblings=10,
                        )
                    except Exception as e:
                        logger.error("Neo4j connection error (community siblings): %s", e)
                        siblings_info = {}
    
                    if siblings_info:
                        # Add community summary if available and budget permits
                        community_title = siblings_info.get("community_title")
                        community_summary = siblings_info.get("community_summary")
                        if community_title and community_summary and ec_budget_rem > 0:
                            comm_text = f"[{community_title}]: {community_summary}"
                            token_count = await llm.count_tokens(comm_text, MODEL_NAME)
                            if token_count <= ec_budget_rem:
                                community_pool_i.append(comm_text)
                                ec_budget_rem -= token_count
    
                        # Add sibling entities to entity pool (they are entities, use entity budget)
                        siblings = siblings_info.get("siblings") or []
                        sibling_texts = []
                        for sb in siblings:
                            if ee_budget_rem <= 0:
                                break
                            key = (sb["title"], sb["type"])
                            if key in entity_set_i:
                                continue
                            sb_desc = sb.get("description") or "No description"
                            sb_text = f"[{sb['title']}] ({sb['type']}) [sibling]: {sb_desc}"
                            token_count = await llm.count_tokens(sb_text, MODEL_NAME)
                            if token_count <= ee_budget_rem:
                                sibling_texts.append(sb_text)
                                ee_budget_rem -= token_count
                                entity_set_i.add(key)
                        entity_pool_i.extend(sibling_texts)
    
                # 5.5 Find leaf Community
                if ec_budget_rem > 0 and entity_set_i:
                    titles_list = [t for t, _ in entity_set_i]
                    types_list = [tp for _, tp in entity_set_i]
                    try:
                        # Find leaf communities (have IS_CHILD_OF but not IS_PARENT_OF)
                        communities_raw = await doc_manager.run_cypher(
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
                        # Count entities for each community
                        comm_with_counts = []
                        for comm in communities_raw:
                            try:
                                count_res = await doc_manager.run_cypher(
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
                """Step 6: Fallback community search."""
                fallback_pool: list = []
    
                # 6.1 Qdrant search for root community (level=0)
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
    
                # 6.2 Neo4j get Community
                try:
                    neo4j_res = await doc_manager.run_cypher(
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
    
                # 6.3 Add root community to pool
                summary = root.get("summary") or "No summary"
                root_text = f"[{root['title']}]: {summary}"
                token_count = await llm.count_tokens(root_text, MODEL_NAME)
                if token_count <= budget:
                    fallback_pool.append(root_text)
                    budget -= token_count
    
                # 6.4 Find child communities
                try:
                    children_raw = await doc_manager.run_cypher(
                        "MATCH (root:Community {id: $community_id})-[:IS_PARENT_OF]->(child:Community) "
                        "RETURN child.id AS id, child.title AS title, child.summary AS summary, child.level AS level",
                        community_id=community_id,
                    )
                except Exception as e:
                    logger.error("Neo4j connection error (children): %s", e)
                    raise HTTPException(status_code=500, detail="Neo4j connection error") from e
    
                if children_raw and budget > 0:
                    # 6.5 Get embeddings for child communities from Qdrant (parallel)
    
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
                                    child["score"] = cosine_similarity(vector, question_embedding)
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
    
                    # 6.6 Add child communities to pool
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
    
            # Step 5 (continued): parallel processing of all entities
    
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
    
            # --- Step 7: Merge pools (round-robin) ---
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
    
                # Count used tokens
                total_entity_tokens = await _count_tokens_list(merged_entity_pool)
                total_community_tokens = await _count_tokens_list(merged_community_pool)
                total_text_tokens = text_tokens_used
    
            else:
                # Global fallback
                merged_entity_pool = []
                fallback_community_pool = await _fallback_search(
                    session=session,
                    qdrant_headers=qdrant_headers,
                    question_embedding=question_embedding,
                    budget=community_budget,
                )
    
                # Count tokens for fallback community pool
                merged_community_pool = fallback_community_pool
                total_entity_tokens = 0
                total_community_tokens = await _count_tokens_list(merged_community_pool)
                total_text_tokens = text_tokens_used

    # --- Step 8: Build response ---
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