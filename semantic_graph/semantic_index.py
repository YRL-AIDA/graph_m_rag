import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).parent))

from clasterization import create_communities
from config import MODEL_NAME
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
from prompts import COMMUNITY_REPORT_PROMPT
from Qdrant_extractor.config import QDRANT_API_KEY, QDRANT_URL
from Qdrant_extractor.dataframe_builder import build_chunks_dataframe
from Qdrant_extractor.qdrant_adapter import QdrantStreamAdapter

load_dotenv()

MAX_CLUSTER_SIZE = 10
USE_LCC = False


config = ManagerConfig(
    uri='neo4j://' + os.environ['URL'],
    user=os.environ['USER_NEO4J'],
    password=os.environ['PASSWORD'],
    name_db=os.environ['NAME_DB']
)

doc_manager = Manager(config)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Chunk Processing Service")

# --- Константы ---
# Типы сущностей лучше вынести в константу и писать в верхнем регистре
ENTITY_TYPES = ['ORGANIZATION', 'PERSON', 'GEO', 'EVENT']


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
    examples = await create_communities(relations_df,max_cluster_size=MAX_CLUSTER_SIZE,use_lcc=USE_LCC,seed=256)
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9595)