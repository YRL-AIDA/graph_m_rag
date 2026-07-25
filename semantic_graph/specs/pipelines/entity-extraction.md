# Pipeline: Entity Extraction

## Purpose
Извлечение сущностей и связей из текстовых чанков документа с помощью LLM, суммаризация описаний и запись в Neo4j Semantic Graph. Запускается через эндпоинт `POST /process-document` в `semantic_graph/semantic_index.py`. Это per-document операция (в отличие от кластеризации, которая работает со всем графом).

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Load Chunks from Qdrant | `document_id` (file_hash) | `semantic_index.py:process_document()` → `QdrantStreamAdapter(base_url=QDRANT_URL)` → `build_chunks_dataframe(adapter, doc_id=document_id)` в `Qdrant_extractor/dataframe_builder.py`. Scroll всех коллекций Qdrant через REST API (`/collections/{name}/points/scroll`, `limit=100`), фильтр `must: [{key: "file_hash", match: {value: doc_id}}]`. Фильтрация: только элементы с `original_element.type == "text"`, текст берётся из `original_element.text`. Сортировка по `document_id`, `page`, `element_index`. Формируется `input_df` с колонками `chunk_id` (формат `{file_hash}\|{element_index}`) и `text`. | `input_df: pd.DataFrame` с колонками `id`, `text`. Пустой DataFrame → HTTP 404. | sequential (REST scroll) |
| 2. LLM Entity & Relationship Extraction | `input_df`, `entity_types` (из `config.ENTITY_TYPES`: `['ORGANIZATION', 'PERSON', 'GEO', 'EVENT']`) | `graphrag.py:run_extraction_pipeline_async()` → Для каждого чанка параллельно: `AsyncGraphExtractor.extract(text, entity_types, source_id=chunk_id)`. **Per-chunk цикл**: (a) Первый вызов LLM с промптом `GRAPH_EXTRACTION_PROMPT` (из `config.py`, включает примеры few-shot). (b) Gleaning loop (до `max_gleanings` итераций, по умолчанию 0 — gleaning отключён на уровне `/process-document`): CONTINUE_PROMPT → проверка COMPLETION_DELIMITER → LOOP_PROMPT (Y/N). (c) Парсинг ответа `_parse_result()`: разбиение по `RECORD_DELIMITER="##"`, внутри записи — по `TUPLE_DELIMITER="<\|>"`. entity: `("entity"<\|><name><\|><type><\|><description>)`. relationship: `("relationship"<\|><source><\|><target><\|><description><\|><weight>)`. Имена приводятся к верхнему регистру, очищаются от HTML-экранирования (`clean_str`). Связи фильтруются: остаются только те, где source и target есть в entities данного чанка. | `entity_dfs: List[pd.DataFrame]`, `relationship_dfs: List[pd.DataFrame]` | parallel (asyncio.gather по чанкам) |
| 3. Merge & Deduplicate | `entity_dfs`, `relationship_dfs` | `graphrag.py:merge_entities()` — groupby `["title", "type"]`, агрегация `description` в список, `text_unit_ids` из `source_id`, `frequency=count(source_id)`. `merge_relationships()` — groupby `["source", "target"]`, агрегация `description` в список, `text_unit_ids`, `weight=sum`. Формирование составных ключей: `entity_key = f"{title}\|{type}"`. `filter_orphan_relationships()` — удаление связей с несуществующими узлами. Если `merged_entities` пуст — `ValueError` → HTTP 200 со статусом `completed_without_entities`. | `merged_entities: pd.DataFrame`, `valid_relationships: pd.DataFrame` | sequential |
| 4. Summarize Descriptions | `merged_entities`, `valid_relationships` | `graphrag.py:AsyncGraphSummarizer.summarize_all()` — Параллельная суммаризация: для каждой сущности и каждой связи запускается `_summarize_item()`. Алгоритм: если 1 описание — вернуть как есть. Если несколько — итеративное слияние с контролем токенов (`max_input_tokens=8000`), промпт `SUMMARIZE_PROMPT` ("concatenate all into single comprehensive description"). | `entity_summaries: pd.DataFrame`, `relationship_summaries: pd.DataFrame` | parallel (asyncio.gather по всем элементам) |
| 5. Build Degree Map & Finalize | `final_relationships` | `graphrag.py:_build_degree_map()` — подсчёт степени для каждой сущности (уникальные пары source-target). `finalize_entities()` — дедупликация, назначение `degree`, `human_readable_id`. `finalize_relationships()` — дедупликация, `combined_degree = degree(source) + degree(target)`. | `final_entities: pd.DataFrame`, `final_relationships: pd.DataFrame` | sequential |
| 6. Save to Neo4j | `final_entities`, `final_relationships` | `semantic_index.py:process_document()` → `doc_manager.add_entities_batch(EntitiesRequest)`. Конвертация DataFrame → `EntityCreate`/`RelationshipCreate` Pydantic-модели. `Manager.add_entities_batch()` → для каждой сущности: `_create_or_update_entity_tx()` (MERGE по `title`+`type`, обновление `text_unit_ids`, `frequency`, `description`, `degree`). Для каждой связи: `_create_relationship_tx()` (проверка существования, MERGE, обновление `weight`, `description`, `text_unit_ids`, `combined_degree`). Стабильный ID связи: `sha256(source|target|description)[:16]`. | `EntitiesResponse: {nodes_created, nodes_updated, relationships_added}` | sequential (внутри write-транзакции Neo4j) |

## Data Flow Diagram

```
POST /process-document { document_id }
    │
    ▼
[Stage 1] Qdrant: scroll all collections, filter by file_hash
    │ only original_element.type == "text"
    │ chunk_id = "{file_hash}|{element_index}"
    ▼
[Stage 2] LLM: parallel extraction per chunk
    │ prompt: GRAPH_EXTRACTION_PROMPT (config.py)
    │ model: Qwen/Qwen3-4B-Instruct-2507
    │ parse: RECORD_DELIMITER + TUPLE_DELIMITER
    │
    ├──► entities: (title, type, description, source_id)
    └──► relationships: (source, target, description, weight, source_id)
    │
    ▼
[Stage 3] Merge: groupby title+type → aggregate descriptions, text_unit_ids, frequency
    │ filter orphan relationships
    ▼
[Stage 4] LLM: parallel summarization
    │ prompt: SUMMARIZE_PROMPT (config.py)
    │ max_input_tokens: 8000, max_summary_length: 8000
    ▼
[Stage 5] Build degree map → finalize (dedup + degree assignment)
    ▼
[Stage 6] Neo4j: MERGE Entity nodes, MERGE RELATED relationships
    │ constraint: entity_title_type_unique (title, type)
    ▼
Response: { document_id, statistics: { total_chunks, processing_time_ms, nodes_created, ... } }
```

## LLM Interactions

### Extraction Prompt
- **Файл**: должен быть `semantic_graph/prompts/graph-extraction.md` (сейчас определён как `GRAPH_EXTRACTION_PROMPT` в `config.py`, строки 195-315 — нарушение P7 Constitution).
- **Формат**: `-Goal-` → `-Steps-` (4 шага) → `-Examples-` (3 few-shot примера) → `-Real Data-` с подстановкой `{entity_types}` и `{input_text}`.
- **Выходной формат**: строки, разделённые `##`, каждая строка — tuple через `<|>`. Entity: `("entity"<|><name><|><type><|><description>)`. Relationship: `("relationship"<|><source><|><target><|><description><|><weight>)`. Терминатор: `<|COMPLETE|>`.
- **Модель**: `Qwen/Qwen3-4B-Instruct-2507` (определена в `config.MODEL_NAME`).

### Continuation Prompts (gleaning)
- `CONTINUE_PROMPT` (config.py:317): запрос пропущенных сущностей.
- `LOOP_PROMPT` (config.py:318): Y/N проверка на оставшиеся сущности.
- **Примечание**: в `/process-document` gleaning отключён (`max_gleanings=0`).

### Summarization Prompt
- **Файл**: должен быть `semantic_graph/prompts/summarize.md` (сейчас `SUMMARIZE_PROMPT` в `config.py`, строки 179-193 — нарушение P7 Constitution).
- **Формат**: инструкция объединить несколько описаний в одно, разрешить противоречия, лимит `{max_length}` слов, third person.
- **Модель**: та же `Qwen/Qwen3-4B-Instruct-2507`.

## Performance Constraints

- **LLM extraction model**: `Qwen/Qwen3-4B-Instruct-2507`.
- **Max gleanings**: 0 (отключено в `/process-document`).
- **Summarization**: `max_summary_length=8000`, `max_input_tokens=8000`.
- **Parallelism**: все чанки извлекаются параллельно (asyncio.gather), все суммаризации — параллельно.
- **Token counting**: через отдельный сервис `TOKENIZER_URL` (`/tokenize`), fallback: `len(text)//4`.
- **Qdrant scroll**: `limit=100` на страницу.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Load Chunks | Qdrant недоступен | HTTP 500 "Qdrant connection error" |
| 1. Load Chunks | Документ не найден / пустой DataFrame | HTTP 404 "Document not found or empty" |
| 2. LLM Extraction | Ошибка LLM для отдельного чанка | `AsyncLLMClient.generate()` возвращает None → `_empty_dfs()` (пустые DataFrame для чанка) |
| 2. LLM Extraction | Все чанки дали пустой результат | `ValueError("No valid entities detected")` → HTTP 200 со статусом `completed_without_entities` |
| 3. Merge | Пустой merged_entities | ValueError → HTTP 200 `completed_without_entities` |
| 4. Summarization | Ошибка LLM для отдельного элемента | `_call_llm_summarize()` возвращает `""` |
| 5. Finalize | — | Последовательная операция, ошибок не ожидается |
| 6. Save to Neo4j | Ошибка Neo4j | HTTP 500 "Failed to save data to Graph DB" |

Общая обработка: внешний `try/except` в `process_document()` → HTTP 500.
