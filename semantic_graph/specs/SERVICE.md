# Service: semantic_graph

## Identity
- **Зона ответственности**: Извлечение именованных сущностей и связей из текстовых чанков с помощью LLM, сохранение семантического графа знаний в Neo4j, иерархическая кластеризация графа (алгоритм Leiden), генерация LLM-отчётов по сообществам (community reports).
- **Порт**: `9595`
- **Хранилище**: Neo4j (граф семантических знаний: узлы `Entity`, `Community`; связи `RELATED`, `CONSISTS_OF`, `IS_CHILD_OF`, `IS_PARENT_OF`)

## Dependencies

| Сервис | Протокол | Назначение |
|--------|----------|------------|
| **Neo4j** | Direct driver (`neo4j` Python driver) | Хранение сущностей (`Entity`), связей (`RELATED`), сообществ (`Community`), иерархии сообществ (`IS_CHILD_OF`/`IS_PARENT_OF`), принадлежности сущностей к сообществам (`CONSISTS_OF`) |
| **Qdrant** | REST (`QdrantStreamAdapter` — HTTP scroll API) | Источник текстовых чанков документов; используется модулем `Qdrant_extractor/` для построения DataFrame чанков по `document_id` |
| **LLM API** | REST (OpenAI-совместимый `AsyncOpenAI`) | Извлечение сущностей/связей из текста (`graphrag.py`), суммаризация описаний, генерация отчётов сообществ (`create_community_report.py`) |
| **Tokenizer Service** | REST (`aiohttp` POST на `TOKENIZER_URL`) | Подсчёт числа токенов в тексте для обрезки контекста по лимитам |

## API

| Method | Path | Feature Spec | Purpose |
|--------|------|-------------|---------|
| `POST` | `/process-document` | (pending) | Загрузка чанков документа из Qdrant → извлечение сущностей и связей LLM → сохранение в Neo4j |
| `GET` | `/clastrize_graph` | (pending) | Кластеризация всего графа связей (Leiden) → создание/обновление узлов `Community` в Neo4j |
| `GET` | `/create_community_report` | (pending) | Генерация LLM-отчётов по всем сообществам → запись отчётов в существующие узлы `Community` |
| `GET` | `/compute_entity_embeddings` | [compute-entity-embeddings.md](compute-entity-embeddings.md) | Вычисление 2048-мерных эмбеддингов для Entity с изменившимся описанием, сохранение в Qdrant `entity_embeddings`, обновление `embedding_updated_at` в Neo4j |
| `GET` | `/compute_community_embeddings` | [compute-community-embeddings.md](compute-community-embeddings.md) | Вычисление 2048-мерных эмбеддингов для Community с изменившимся `summary`, сохранение в Qdrant `community_embeddings`, обновление `embedding_updated_at` в Neo4j |

### `POST /process-document`

**Request Body** (`DocumentRequest`):
```json
{
  "document_id": "<file_hash>"
}
```

**Response** (`200 OK`):
```json
{
  "document_id": "<file_hash>",
  "statistics": {
    "total_chunks": 42,
    "processing_time_ms": 15000,
    "status": "completed",
    "nodes_created": 10,
    "nodes_updated": 5,
    "relationships_added": 8
  }
}
```

**Status values**:
- `completed` — сущности извлечены и сохранены
- `completed_without_entities` — документ обработан, но LLM не нашёл сущностей (не ошибка, 200)

**Error responses**:
- `500 Qdrant connection error` — не удалось подключиться к Qdrant
- `404 Document not found or empty` — документ не найден в Qdrant или датафрейм чанков пуст
- `500 Graph extraction failed` — ошибка на этапе извлечения графа (исключая `ValueError` при пустых сущностях)
- `500 Failed to save data to Graph DB` — ошибка сохранения в Neo4j

### `GET /clastrize_graph`

**Response** (`200 OK`):
```json
{
  "document_id": "None",
  "statistics": {
    "total_chunks": 0,
    "processing_time_ms": 5000,
    "status": "completed",
    "communities_created": 15,
    "parent_relations_created": 12,
    "entity_relations_created": 45
  }
}
```

### `GET /compute_entity_embeddings`

**Response** (`200 OK`):
```json
{
  "document_id": "None",
  "statistics": {
    "total_chunks": 0,
    "processing_time_ms": 5000,
    "status": "completed",
    "embeddings_added": 10,
    "embeddings_updated": 5,
    "embeddings_skipped": 0,
    "embeddings_failed": 0
  }
}
```

**Error responses**:
- `500 Qdrant unavailable` — Qdrant недоступен при проверке/создании коллекции
- `500 Qdrant collection dimension mismatch` — коллекция существует, но размерность ≠ 2048
- `500 Qdrant collection distance metric mismatch` — коллекция существует, но метрика ≠ Cosine
- `500 Neo4j unavailable` — Neo4j недоступен при чтении кандидатов

### `GET /create_community_report`

**Response** (`200 OK`):
```json
{
  "document_id": "None",
  "statistics": {
    "total_chunks": 0,
    "processing_time_ms": 30000,
    "status": "completed",
    "updated": 15,
    "skipped": 0,
    "not_found": 0
  }
}
```

### `GET /compute_community_embeddings`

**Response** (`200 OK`):
```json
{
  "document_id": "None",
  "statistics": {
    "total_chunks": 0,
    "processing_time_ms": 8000,
    "status": "completed",
    "embeddings_added": 5,
    "embeddings_updated": 3,
    "embeddings_skipped": 0,
    "embeddings_failed": 0
  }
}
```

**Error responses**:
- `500 Qdrant unavailable` — Qdrant недоступен при проверке/создании коллекции
- `500 Qdrant collection dimension mismatch` — коллекция существует, но размерность ≠ 2048
- `500 Qdrant collection distance metric mismatch` — коллекция существует, но метрика ≠ Cosine
- `500 Neo4j unavailable` — Neo4j недоступен при чтении кандидатов или записи `embedding_updated_at`

## Data Model

Ссылки на Data Model Specs (все ожидают создания, см. Exceptions):
- [entity.md](models/entity.md) — `EntityCreate`, `RelationshipCreate`, `EntitiesRequest`, `EntitiesResponse`, `DocumentRequest` (Pydantic, `dtype/entity.py`)
- [document.md](models/document.md) — `Document`, `create_graph_from_mineru_result` (dtype/document.py)
- [region.md](models/region.md) — `Region`, `Style`, `BBox` (используется `Document` для парсинга MinerU-результата, `dtype/region.py`)

### Модель данных Neo4j (семантический граф)

| Узел / Связь | Метки / Тип | Ключевые свойства | Уникальное ограничение |
|--------------|-------------|-------------------|----------------------|
| **Entity** | `:Entity` | `title`, `type`, `description`, `degree`, `frequency`, `text_unit_ids`, `data`, `created_at`, `updated_at` | `(title, type)` |
| **Community** | `:Community` | `id` (uuid4), `community` (int), `level` (int), `parent` (int), `title`, `size`, `period`, `entity_ids`, `children` | — (по `id`) |
| **Community** (после отчёта) | `:Community` | + `summary`, `full_content`, `full_content_json`, `rating`, `rating_explanation`, `findings`, `report_updated_at` | — |
| **RELATED** | `(Entity)-[:RELATED]->(Entity)` | `id` (sha256[:16]), `weight`, `description`, `combined_degree`, `text_unit_ids`, `created_at`, `updated_at` | — |
| **CONSISTS_OF** | `(Community)-[:CONSISTS_OF]->(Entity)` | — | — |
| **IS_CHILD_OF** | `(Community)-[:IS_CHILD_OF]->(Community)` | — | — |
| **IS_PARENT_OF** | `(Community)-[:IS_PARENT_OF]->(Community)` | — | — |

Идентификатор сущности: составной ключ `"TITLE|TYPE"` (например, `"ACME CORP|ORGANIZATION"`).

## Pipelines

Ссылки на Pipeline Specs (все ожидают создания, см. Exceptions):
- [entity-extraction.md](pipelines/entity-extraction.md) — полный пайплайн извлечения сущностей (`run_extraction_pipeline_async` в `graphrag.py`). Этапы: (1) параллельное извлечение графа из чанков через LLM, (2) слияние результатов, (3) суммаризация описаний, (4) финализация датафреймов.
- [clustering.md](pipelines/clustering.md) — пайплайн кластеризации (`create_communities` в `clasterization.py`). Этапы: (1) иерархический Leiden через `graspologic_native`, (2) агрегация entity_ids и relationship_ids по сообществам, (3) построение дерева parent-child, (4) формирование выходного списка записей.
- [community-report.md](pipelines/community-report.md) — пайплайн генерации отчётов (`run_community_reports_pipeline_async` в `create_community_report.py`). Этапы: (1) подготовка узлов и рёбер (`explode_communities`, `_prep_nodes`, `_prep_edges`), (2) построение локального контекста для каждого сообщества (`build_local_context`), (3) генерация отчётов по уровням иерархии с параллелизацией внутри уровня (`summarize_communities`), (4) финализация датафрейма (`finalize_community_reports`).

## Configuration

### Переменные окружения (`.env`)

| Переменная | Обязательная | Описание | Пример |
|------------|-------------|----------|--------|
| `USER_NEO4J` | Да | Пользователь Neo4j | `neo4j` |
| `PASSWORD` | Да | Пароль Neo4j | `neo4j123` |
| `URL` | Да | Neo4j Bolt-хост:порт (без схемы) | `localhost:7687` |
| `NAME_DB` | Да | Имя базы данных Neo4j | `neo4j` |

### Константы конфигурации (`config.py`)

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Модель LLM для извлечения сущностей и генерации отчётов |
| `N4G_URL` | `http://192.168.19.148:9998` | Дополнительный URL (не используется в основном API) |
| `LLM_API_KEY` | `EMPTY` | API-ключ для LLM (для локальных моделей — `EMPTY`) |
| `LLM_URL` | `http://localhost:9886/v1` | OpenAI-совместимый endpoint LLM |
| `TOKENIZER_URL` | `http://localhost:9886/tokenize` | Endpoint сервиса токенизации |
| `QDRANT_URL` | `http://localhost:6333/` | URL Qdrant |
| `QDRANT_API_KEY` | `None` | API-ключ Qdrant (опционально) |
| `DOCUMENT_ID_FIELD` | `file_hash` | Поле в payload Qdrant для идентификации документа |
| `MAX_CLUSTER_SIZE` | `10` | Максимальный размер кластера в Leiden |
| `USE_LCC` | `False` | Использовать только наибольшую связную компоненту |
| `CLUSTERIZATION_SEED` | `256` | Seed для воспроизводимости кластеризации |
| `ENTITY_TYPES` | `['ORGANIZATION', 'PERSON', 'GEO', 'EVENT']` | Типы сущностей для промпта извлечения |
| `API_HOST` | `0.0.0.0` | Хост FastAPI-сервера |
| `API_PORT` | `9595` | Порт FastAPI-сервера |
| `EMBEDDING_BASE_URL` | `http://192.168.19.127:10115/embedding` | Базовый URL embedding-сервиса |
| `EMBEDDING_TIMEOUT` | `30` | Таймаут запроса к embedding-сервису в секундах |
| `EMBEDDING_MAX_CONCURRENCY` | `8` | Максимальное количество одновременных вызовов embedding-сервиса |
| `ENTITY_EMBEDDINGS_COLLECTION` | `entity_embeddings` | Имя коллекции Qdrant для эмбеддингов Entity |
| `ENTITY_EMBEDDINGS_BATCH_SIZE` | `100` | Размер батча для upsert в Qdrant |
| `ENTITY_EMBEDDINGS_NAMESPACE` | `UUID("a7f1b2c3-...")` | Namespace UUID для детерминированных point ID |
| `COMMUNITY_EMBEDDINGS_COLLECTION` | `community_embeddings` | Имя коллекции Qdrant для эмбеддингов Community |
| `COMMUNITY_EMBEDDINGS_BATCH_SIZE` | `100` | Размер батча для upsert эмбеддингов Community в Qdrant |
| `COMMUNITY_EMBEDDINGS_NAMESPACE` | `UUID("b8e2c3d4-...")` | Namespace UUID для детерминированных point ID Community |

### Константы форматирования промптов (`config.py`)

| Параметр | Значение | Описание |
|----------|---------|----------|
| `TUPLE_DELIMITER` | `<\|>` | Разделитель полей внутри записи ответа LLM |
| `RECORD_DELIMITER` | `##` | Разделитель записей в ответе LLM |
| `COMPLETION_DELIMITER` | `<\|COMPLETE\|>` | Маркер завершения ответа LLM |
| `CONTINUE_PROMPT` | (строка) | Промпт для продолжения извлечения (gleaning) |
| `LOOP_PROMPT` | (строка) | Промпт для проверки необходимости повторного прохода |

### Промпты (`config.py`, инлайн — нарушение P7)

| Переменная | Назначение | Должен быть в |
|------------|-----------|--------------|
| `GRAPH_EXTRACTION_PROMPT` | Извлечение сущностей и связей из текста | `prompts/graph-extraction.md` |
| `SUMMARIZE_PROMPT` | Суммаризация повторяющихся описаний сущностей/связей | `prompts/summarize.md` |
| `COMMUNITY_REPORT_PROMPT` | Генерация отчёта по сообществу | `prompts/community-report.md` |

## Invariants

1. **P9 — Семантический граф глобален**: кластеризация (`GET /clastrize_graph`) и генерация отчётов (`GET /create_community_report`) работают со всем графом Neo4j, а не в разрезе отдельного документа. Повторный вызов этих эндпоинтов создаёт/обновляет сообщества через `MERGE`, но не очищает старые данные.
2. **P10 — Мягкое удаление**: сущности (`Entity`) и связи (`RELATED`) при удалении документа должны помечаться как `archived`, а не удаляться физически (на текущий момент флаг `archived` в коде не реализован — см. Exceptions).
3. **P4 — Два независимых графа Neo4j**: `semantic_graph` работает только с узлами `Entity`/`Community` и связями `RELATED`/`CONSISTS_OF`/`IS_CHILD_OF`/`IS_PARENT_OF`. Запросы не затрагивают узлы документального графа (`Document`, `Region:*`).
4. **Дедупликация сущностей**: сущности уникальны по паре `(title, type)`. Повторная загрузка тех же `text_unit_ids` пропускается (метод `_text_unit_ids_already_exist`).
5. **P5 — Асинхронность I/O операций**: все сетевые вызовы (LLM, Neo4j, Qdrant) используют `async/await`. Параллелизация через `asyncio.gather`. Синхронные вызовы к Qdrant (`build_chunks_dataframe`) выполняются в основном потоке — потенциальная проблема (см. Exceptions).
6. **Порядок выполнения пайплайна**: `POST /process-document` → `GET /clastrize_graph` → `GET /create_community_report`. Каждый следующий шаг зависит от результатов предыдущего.
7. **LLM-формат ответа**: извлечение сущностей ожидает строгий формат: `("entity"<|>NAME<|>TYPE<|>DESCRIPTION)` и `("relationship"<|>SRC<|>TGT<|>DESC<|>WEIGHT)`, разделённые `##`, завершающиеся `<|COMPLETE|>`.
8. **Нормализация имён**: имена сущностей приводятся к UPPER CASE при парсинге ответа LLM.
9. **Embedding-обновление ONLY после Qdrant upsert**: Neo4j-поле `embedding_updated_at` обновляется строго после успешного upsert в Qdrant (не наоборот), чтобы предотвратить ситуацию, где Neo4j считает embedding актуальным, а в Qdrant его нет.
10. **Neo4j APOC**: для обновления `text_unit_ids` через `apoc.coll.toSet` требуется установленный плагин APOC в Neo4j.
11. **Constraint на Entity**: уникальность пары `(title, type)` гарантируется constraint'ом `entity_title_type_unique`, создаваемым при инициализации `Manager`.

## Exceptions

Обоснованные отклонения от Constitution (документационная спецификация — фиксация текущего состояния):

1. **P7 — Промпты инлайн в `config.py`**: промпты `GRAPH_EXTRACTION_PROMPT`, `SUMMARIZE_PROMPT`, `COMMUNITY_REPORT_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT` хранятся как строковые константы в `semantic_graph/config.py` (строки 179–468), а не в отдельных файлах `prompts/*.md` как того требует P7. Обоснование: переходный период, спецификация документирует «как есть». Требуется рефакторинг: вынести промпты в `prompts/graph-extraction.md`, `prompts/summarize.md`, `prompts/community-report.md`.

2. **N3 — Конфигурация не Pydantic Settings**: конфигурация сервиса определена как модульные константы в `config.py`, а не через Pydantic `BaseSettings` в `config/settings.py`. Часть параметров (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NAME_DB) читается из переменных окружения напрямую через `os.environ` в `semantic_index.py`, а не централизованно через Settings-класс. Обоснование: переходный период; требуется рефакторинг.

3. **Дубликаты `dtype/` с `documet_index`**: модели `Document`, `Region`, `Style`, `BBox` (`dtype/document.py`, `dtype/region.py`) дублируют или пересекаются с аналогичными моделями из сервиса `documet_index`. Это нарушает P1 (сервисные границы) и P6 (единый источник истины для моделей). Обоснование: код `semantic_graph` частично использует `Document` для парсинга MinerU-результатов (`neo4j_service.py`), но эти модели не задействованы в основных API-эндпоинтах.

4. **`req2.txt` — дубликат `requirements.txt`**: файл `semantic_graph/req2.txt` содержит идентичный список зависимостей (233 строки), что и `semantic_graph/requirements.txt`. Это неиспользуемый мусорный файл, подлежащий удалению.

5. **Независимые модули с пересекающейся ответственностью**: `semantic_graph/neo4j_service.py` (`DocumentIndexService`) реализует логику индексации документов MinerU, которая дублирует функциональность `documet_index`. Не используется в основном пайплайне `semantic_index.py`, но импортирует `dtype.Document`. Нарушение P1 и P4.

6. **Синхронный Qdrant-клиент**: `build_chunks_dataframe` и `QdrantStreamAdapter` используют синхронный `requests` для скроллинга Qdrant, что блокирует event loop FastAPI. Согласно P5, должно быть асинхронно или через `run_in_executor`. Обоснование: переходный период.

7. **P10 — Мягкое удаление не реализовано**: флаг `archived` на узлах `Entity` и связях `RELATED` в коде `manager.py` не проставлен, физического удаления также нет — удаление документов из семантического графа в текущей версии не поддерживается. Обоснование: функциональность удаления документов не реализована.
8. **P10 — Мягкое удаление Community не реализовано**: флаг `archived` на узлах `Community` отсутствует. Cypher-запрос в `get_communities_needing_embedding()` не фильтрует по `archived`. При реализации мягкого удаления потребуется добавить условие `AND (c.archived IS NULL OR c.archived = false)`.
