# Feature: Извлечение сущностей и связей из документа (process-document)

## Motivation
Пайплайн извлечения семантического графа знаний из отдельного документа. Принимает `document_id` (`file_hash`), извлекает чанки из Qdrant, через LLM выделяет сущности (`Entity`) и связи (`RELATED`) в каждом чанке, агрегирует, суммаризирует описания и сохраняет результат в Neo4j (semantic knowledge graph). Это per-document операция — первый шаг построения графа знаний перед глобальной кластеризацией и отчётами сообществ.

## Behaviour

### Input
- **Метод**: `POST`
- **Path**: `/process-document`
- **Content-Type**: `application/json`
- **Тело запроса**: Pydantic-модель `DocumentRequest` (см. `dtype/entity.py`):
  ```json
  {
    "document_id": "<file_hash>"
  }
  ```
- **Валидация**: `document_id` — непустая строка (MD5-хеш содержимого PDF). Пустое тело или отсутствие поля → HTTP 422 (FastAPI auto-validation).

### Processing
1. **Загрузка данных из Qdrant**:
   - Создаётся `QdrantStreamAdapter(base_url=QDRANT_URL, api_key=QDRANT_API_KEY)`.
   - Вызывается синхронная `build_chunks_dataframe(adapter, doc_id=document_id)`, которая фильтрует точки Qdrant по `file_hash == document_id` и возвращает DataFrame с колонками `chunk_id`, `text`.
   - Если Qdrant недоступен → `HTTPException(500, "Qdrant connection error")`.
   - Если DataFrame пуст (документ не найден или не содержит чанков) → `HTTPException(404, "Document not found or empty")`.
   - DataFrame переименовывается: `chunk_id` → `id`, выбираются колонки `['id', 'text']` → `input_df`.

2. **Извлечение графа знаний (LLM pipeline)**:
   - Вызывается `run_extraction_pipeline_async(input_df, MODEL_NAME, MODEL_NAME, ENTITY_TYPES, max_gleanings=0, max_summary_length=8000, max_input_tokens=8000)` из `graphrag.py`.
   - **Этап 2.1 — Извлечение**: для каждого чанка (строка `input_df`) параллельно (`asyncio.gather`) вызывается `AsyncGraphExtractor.extract(text, entity_types, chunk_id)`:
     - Формируется промпт через `GRAPH_EXTRACTION_PROMPT.format(input_text=..., entity_types=...)` с подстановкой строки из `",".join(entity_types)`.
     - LLM возвращает структурированный текст с сущностями в формате `("entity"<|>NAME<|>TYPE<|>desc)` и связями `("relationship"<|>SRC<|>TGT<|>desc<|>weight)`, разделёнными `##`.
     - Gleaning-цикл: если `max_gleanings > 0`, после первого ответа отправляется `CONTINUE_PROMPT`, затем `LOOP_PROMPT` с проверкой ответа `Y/N`. При текущей конфигурации `max_gleanings=0`, цикл не выполняется.
     - Результат парсится в два DataFrame: `entities_df` (колонки: `title`, `type`, `description`, `source_id`) и `relationships_df` (колонки: `source`, `target`, `description`, `source_id`, `weight`). Ключи связей формируются как `title|type`.
     - Имена сущностей и типы приводятся к верхнему регистру через `clean_str`.
   - **Этап 2.2 — Слияние**: результаты всех чанков объединяются:
     - `merge_entities()`: группировка по `(title, type)`, агрегация `description` в список, `text_unit_ids` из `source_id`, `frequency` = количество упоминаний.
     - `merge_relationships()`: группировка по `(source, target)`, агрегация `description` в список, `text_unit_ids` из `source_id`, `weight` суммируется.
     - `filter_orphan_relationships()`: удаляются связи, чьи `source`/`target` не найдены в множестве ключей сущностей `title|type`.
     - Если `merged_entities` пуст → `ValueError("Graph Extraction failed: No valid entities detected.")`.
   - **Этап 2.3 — Суммаризация**: для каждой сущности и связи параллельно вызывается `AsyncGraphSummarizer._summarize_item()`:
     - Список описаний дедуплицируется (`set`), сортируется.
     - Если описание одно — возвращается как есть.
     - Иначе описания буферизируются с учётом лимита `max_input_tokens` и суммаризируются через LLM с промптом `SUMMARIZE_PROMPT.format(entity_name=..., description_list=..., max_length=...)`.
   - **Этап 2.4 — Финализация**: `finalize_entities()` добавляет `degree` (из карты степеней связей), дедуплицирует; `finalize_relationships()` добавляет `combined_degree`.
   - Возвращает кортеж `(final_entities, final_relationships)` — pandas DataFrame.
   - Если `ValueError` → возвращается ответ `{status: 'completed_without_entities'}` (HTTP 200).
   - Если другое исключение → `HTTPException(500, "Graph extraction failed")`.

3. **Сохранение в Neo4j**:
   - DataFrame сущностей и связей конвертируются в список словарей через `.to_dict(orient='records')`.
   - Каждый словарь оборачивается в Pydantic-модели `EntityCreate` и `RelationshipCreate`.
   - Вызывается `Manager.add_entities_batch(EntitiesRequest(entities=[...], relationships=[...]))`:
     - Для каждой сущности: MERGE по `(title, type)`. Если существует — дополняются `text_unit_ids`, `description`, `frequency`, `degree`. Иначе — CREATE.
     - Для каждой связи: MATCH `source` и `target` Entity, MERGE связь `RELATED` с weight, description, combined_degree.
   - Возвращается `EntitiesResponse(nodes_created, nodes_updated, relationships_added)`.
   - При ошибке → `HTTPException(500, "Failed to save data to Graph DB")`.

4. **Формирование ответа**:
   - Унифицированный ответ через `_build_response(doc_id, total_chunks, start_time, status, extra_stats)`:
     - `processing_time_ms` = `(time.time() - start_time) * 1000`.
     - Статистика Neo4j добавляется в `statistics` через `extra_stats`.

### Output
- **HTTP 200** — успешная обработка:
  ```json
  {
    "document_id": "<file_hash>",
    "statistics": {
      "total_chunks": <int>,
      "processing_time_ms": <int>,
      "status": "completed" | "completed_without_entities",
      "nodes_created": <int>,
      "nodes_updated": <int>,
      "relationships_added": <int>
    }
  }
  ```
- **HTTP 404** — документ не найден или пуст: `{"detail": "Document not found or empty"}`
- **HTTP 500** — ошибка Qdrant, LLM или Neo4j: `{"detail": "Qdrant connection error" | "Graph extraction failed" | "Failed to save data to Graph DB"}`

## API Contract

```openapi
POST /process-document
summary: Извлечение сущностей и связей из документа через LLM и сохранение в Neo4j
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/DocumentRequest'
responses:
  '200':
    description: Обработка завершена (с сущностями или без)
    content:
      application/json:
        schema:
          type: object
          properties:
            document_id:
              type: string
            statistics:
              type: object
              properties:
                total_chunks:
                  type: integer
                processing_time_ms:
                  type: integer
                status:
                  type: string
                  enum: [completed, completed_without_entities]
                nodes_created:
                  type: integer
                nodes_updated:
                  type: integer
                relationships_added:
                  type: integer
  '404':
    description: Документ не найден или не содержит чанков
    content:
      application/json:
        schema:
          type: object
          properties:
            detail:
              type: string
  '500':
    description: Ошибка соединения или обработки
    content:
      application/json:
        schema:
          type: object
          properties:
            detail:
              type: string
```

## Data Flow
```
Client (app)
  │
  │ POST /process-document {"document_id": "<file_hash>"}
  ▼
semantic_index.py
  │
  ├─[1] QdrantStreamAdapter → build_chunks_dataframe()
  │     └─ Qdrant (read): filter by file_hash, return chunks DataFrame
  │
  ├─[2] graphrag.run_extraction_pipeline_async()
  │     ├─ AsyncGraphExtractor.extract() × N chunks (parallel)
  │     │   └─ LLM API (AsyncOpenAI): GRAPH_EXTRACTION_PROMPT
  │     ├─ merge_entities() + merge_relationships()
  │     ├─ AsyncGraphSummarizer.summarize_all() (parallel)
  │     │   └─ LLM API (AsyncOpenAI): SUMMARIZE_PROMPT
  │     └─ finalize_entities() + finalize_relationships()
  │
  ├─[3] Manager.add_entities_batch()
  │     └─ Neo4j (write): MERGE Entity, MERGE RELATED
  │
  └─ Response: {"document_id": ..., "statistics": {...}}
```

## LLM Interactions

| Промпт | Файл (должен быть) | Использование |
|--------|-------------------|---------------|
| `GRAPH_EXTRACTION_PROMPT` | `prompts/graph-extraction.md` | Извлечение сущностей и связей из текста чанка. Переменные: `{entity_types}`, `{input_text}`. Выход: структурированный текст с `("entity"<|>...<|>...)` и `("relationship"<|>...<|>...)`, разделённый `##`, завершается `<|COMPLETE|>`. |
| `CONTINUE_PROMPT` | `prompts/continue.md` | Запрос на продолжение извлечения пропущенных сущностей. Без переменных. |
| `LOOP_PROMPT` | `prompts/loop.md` | Запрос на решение Y/N о необходимости ещё одного цикла gleaning. Без переменных. |
| `SUMMARIZE_PROMPT` | `prompts/summarize.md` | Суммаризация списка описаний одной сущности/связи. Переменные: `{entity_name}`, `{description_list}`, `{max_length}`. Выход: свободный текст. |

**Текущее состояние (нарушение P7)**: все промпты инлайн в `config.py`. В соответствии с Constitution P7 они должны быть вынесены в `semantic_graph/prompts/*.md`.

## LLM Model Requirements
- **Тип модели**: text-only (чат-модель)
- **Минимальный размер контекста**: 8000 токенов (настраивается через `max_input_tokens`)
- **Язык выхода**: английский (промпты на английском, сущности и описания извлекаются на английском согласно промпту)
- **Требования к формату выхода**:
  - `GRAPH_EXTRACTION_PROMPT`: свободный текст со строгим синтаксисом кортежей и разделителей
  - `SUMMARIZE_PROMPT`: свободный текст (описание до `max_summary_length` слов)
  - `CONTINUE_PROMPT`, `LOOP_PROMPT`: свободный текст / одиночная буква Y/N
- **Конкретная модель**: `MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'` (задаётся в `config.py`)
- **API**: OpenAI-совместимый эндпоинт (`LLM_URL = 'http://localhost:9886/v1'`)
- **Tokenizer**: отдельный HTTP-эндпоинт (`TOKENIZER_URL = 'http://localhost:9886/tokenize'`)

## Error Handling

| Сценарий | Код | Поведение |
|----------|-----|-----------|
| Qdrant недоступен | 500 | `"Qdrant connection error"`, логирование ошибки |
| Документ не найден в Qdrant | 404 | `"Document not found or empty"` |
| DataFrame чанков пуст | 404 | `"Document not found or empty"` |
| LLM не вернул ни одной сущности во всех чанках | 200 | `status: "completed_without_entities"`, логирование warning |
| LLM API недоступен / ошибка | 500 | `"Graph extraction failed"`, логирование ошибки |
| Neo4j недоступен / ошибка записи | 500 | `"Failed to save data to Graph DB"`, логирование ошибки |
| Ответ LLM пустой (`None`) | — | Пропуск чанка (возвращается пустой DataFrame для чанка) |
| Некорректный JSON/формат ответа LLM | — | В `generate_structured` возвращается `None`; в `generate` — пустая строка → пустой DataFrame |
| Превышение лимита токенов при суммаризации | — | Описания обрабатываются итеративно: буферизируются с учётом `max_input_tokens`, промежуточные результаты сжимаются через LLM |

## Testing

### Тест-кейсы
1. **Успешное извлечение**: документ с 3 чанками, содержащими текст с организациями и персонами → HTTP 200, `status: "completed"`, `nodes_created > 0`, `relationships_added > 0`.
2. **Документ без сущностей**: документ с текстом без организаций/персон/geo/events → HTTP 200, `status: "completed_without_entities"`.
3. **Пустой документ**: document_id, которого нет в Qdrant → HTTP 404.
4. **Qdrant недоступен**: неверный QDRANT_URL → HTTP 500, `"Qdrant connection error"`.
5. **LLM возвращает мусор**: мок LLM с невалидным ответом → HTTP 200, `status: "completed_without_entities"` или 500 (зависит от того, на каком этапе упадёт).
6. **Neo4j недоступен**: неверные креды Neo4j → HTTP 500, `"Failed to save data to Graph DB"`.
7. **Дедупликация сущностей**: один и тот же чанк подан дважды → сущности не дублируются, `nodes_updated > 0`.
8. **Множественные чанки параллельно**: 10 чанков → все обрабатываются через `asyncio.gather`, время обработки ~ O(1 чанк).

### Подход к тестированию
- **Unit**: парсинг ответа LLM (`_parse_result`) с предзаписанными строками; `merge_entities`, `merge_relationships`, `filter_orphan_relationships` на фиктивных DataFrame.
- **Integration**: эндпоинт с мокнутыми `AsyncLLMClient`, `QdrantStreamAdapter`, `Manager` → проверка кодов ответа и структуры.
- **API**: httpx-запросы к поднятому FastAPI с мокнутыми зависимостями.

## Dependencies
- **Qdrant** (read): получение чанков документа через `QdrantStreamAdapter` + `build_chunks_dataframe` (`Qdrant_extractor/`)
- **LLM API** (OpenAI-совместимый): `AsyncOpenAI` через `LLM_URL` для генерации
- **Tokenizer API**: HTTP POST на `TOKENIZER_URL` для подсчёта токенов
- **Neo4j** (write): `Manager` → `Neo4jConnection` для MERGE сущностей и связей
- **Pydantic**: `DocumentRequest`, `EntityCreate`, `RelationshipCreate`, `EntitiesRequest`, `EntitiesResponse`
- **pandas**: DataFrames для чанков, сущностей, связей на всех этапах пайплайна

## Exceptions

### P7 — Промпты инлайн (НАРУШЕНИЕ)
Промпты `GRAPH_EXTRACTION_PROMPT`, `SUMMARIZE_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT` определены как строковые константы в `config.py`, а не в отдельных файлах `prompts/*.md`. Нарушение Constitution P7. Требуется вынос в:
- `prompts/graph-extraction.md`
- `prompts/summarize.md`
- `prompts/continue.md`
- `prompts/loop.md`

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует флага `archived` вместо физического удаления сущностей при удалении документа. В текущем коде механизм мягкого удаления (`archived`) отсутствует — нет поля `archived` на узлах `Entity`, нет фильтрации по `archived` в запросах `get_entities()`, `get_entity_relationships()`. При полном удалении документа через `delete_all_documents()` все узлы удаляются физически (`DETACH DELETE`).

### P9 — Глобальный граф (НЕ НАРУШЕНИЕ)
Операция `process-document` является per-document, что прямо разрешено Constitution P9: «process-document (извлечение сущностей) — per-document операция». Кластеризация и отчёты — глобальные операции — не затрагиваются этим эндпоинтом.
