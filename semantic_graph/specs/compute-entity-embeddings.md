# Feature: Вычисление эмбеддингов сущностей (compute_entity_embeddings)

## Motivation
При построении семантического графа знаний сущности (`Entity`) накапливают описания (`description`), извлечённые LLM из текстовых чанков. Для задач семантического поиска, кластеризации и рекомендаций необходимо иметь векторные представления (эмбеддинги) этих сущностей. Эндпоинт решает задачу **пакетного вычисления и сохранения** эмбеддингов: для всех сущностей, у которых изменилось описание, генерируется 2048-мерный embedding через `EmbeddingClient` (обёртка над embedding-сервисом), результат сохраняется в коллекцию Qdrant `entity_embeddings`, а факт обновления (таймстемп) фиксируется в Neo4j на узле `Entity` (поле `embedding_updated_at`). Сущности с уже актуальными эмбеддингами пропускаются. Это глобальная операция (Constitution P9) — работает со всем графом, без фильтрации по документу.

Архитектурное решение хранить эмбеддинги в Qdrant, а не в Neo4j, обусловлено:
- Qdrant — векторная БД, оптимизированная для ANN-поиска и векторных операций
- Neo4j не нагружается тяжёлыми `list[float]`-свойствами (экономятся ресурсы при B-tree-операциях, бэкапах, репликации)
- Разделение зон ответственности: Neo4j — графовые связи, Qdrant — векторные представления

## Behaviour

### Input
- **Метод**: `GET`
- **Path**: `/compute_entity_embeddings`
- **Параметры**: нет. Параметры подключения к embedding-сервису (`EMBEDDING_BASE_URL`, `EMBEDDING_TIMEOUT`, `EMBEDDING_MAX_CONCURRENCY`), Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`), и имя коллекции (`ENTITY_EMBEDDINGS_COLLECTION`), а также namespace UUID (`ENTITY_EMBEDDINGS_NAMESPACE`) для генерации детерминированных идентификаторов точек в Qdrant заданы в конфигурации сервиса.
- **Валидация**: отсутствует (эндпоинт без параметров).

### Processing

**Предварительные условия**:
- В Qdrant по адресу `QDRANT_URL` доступна (или будет создана) коллекция `entity_embeddings` с конфигурацией: размерность векторов 2048, метрика `COSINE`.
- В Neo4j на узлах `Entity` существует поле `embedding_updated_at` (тип `DateTime`). Поле `embedding` (тип `list[float]`) на узлах `Entity` **отсутствует** — эмбеддинги хранятся исключительно в Qdrant.
- При первом запуске `embedding_updated_at IS NULL` на всех существующих узлах — все сущности с `description IS NOT NULL` будут классифицированы как «новые» (embedding отсутствует в Qdrant).
- `EmbeddingClient` (из `app.src.qwen3_emb_client`) инициализирован с `base_url=EMBEDDING_BASE_URL` и `timeout=EMBEDDING_TIMEOUT`.

**Общая схема**:

```
Neo4j (read candidates) → EmbeddingClient (batched, via asyncio.to_thread) → Qdrant (upsert) → Neo4j (write embedding_updated_at)
```

**Пошаговый алгоритм**:

1. **Проверка / создание коллекции Qdrant**:
   - Асинхронный HTTP-запрос (через `aiohttp`) к Qdrant REST API:
     ```
     GET {QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}
     ```
   - Если коллекция существует:
     - Проверить размерность вектора (`config.params.vectors.size`). Если не равна 2048 → немедленная ошибка **HTTP 500**: «Qdrant collection dimension mismatch».
     - Проверить метрику (`config.params.vectors.distance`). Если не равна `Cosine` → немедленная ошибка **HTTP 500**: «Qdrant collection distance metric mismatch».
   - Если коллекция не существует (HTTP 404 от Qdrant):
     - Создать коллекцию:
       ```
       PUT {QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}
       Body: {
         "vectors": {
           "size": 2048,
           "distance": "Cosine"
         }
       }
       ```
     - Если создание не удалось (Qdrant недоступен, ошибка конфигурации) → **HTTP 500**.

2. **Получение кандидатов из Neo4j**:
   - Вызывается метод `Manager.get_entities_needing_embedding()`.
   - Cypher-запрос:
     ```cypher
     MATCH (e:Entity)
     WHERE e.description IS NOT NULL
       AND (e.embedding_updated_at IS NULL OR e.updated_at > e.embedding_updated_at)
     RETURN e.title AS title, e.type AS type, e.description AS description,
            e.embedding_updated_at AS embedding_updated_at,
            e.updated_at AS updated_at
     ```
   - Возвращается список словарей (`list[dict]`) с ключами: `title`, `type`, `description`, `embedding_updated_at`, `updated_at`.
   - **Важно**: поле `embedding` больше не читается из Neo4j — оно не существует на узлах `Entity`.
   - Если результат пуст → ответ HTTP 200 со всеми нулями (см. Output).

3. **Классификация сущностей** (происходит во время итерации, шаг 4):
   - `is_new` = (`embedding_updated_at IS NULL`) → инкрементируется счётчик `embeddings_added`.
   - `is_updated` = (`embedding_updated_at IS NOT NULL AND updated_at > embedding_updated_at`) → инкрементируется счётчик `embeddings_updated`.
   - `is_skipped` = (`embedding_updated_at IS NOT NULL AND updated_at <= embedding_updated_at`) → **не должно попасть в выборку** (отсеивается Cypher-запросом на шаге 2).
   - Сущности без описания (`description IS NULL`) отсеиваются Cypher-запросом на шаге 2.

4. **Параллельный вызов embedding-сервиса через EmbeddingClient**:
   - Для каждой сущности из выборки вызывается `emb_client.get_text_embedding(entity["description"])` — синхронный метод клиента.
   - Синхронный вызов оборачивается в `asyncio.to_thread()` для неблокирующей работы в async-эндпоинте FastAPI.
   - `EmbeddingClient` внутри формирует тело запроса `{"messages": [{"type": "text", "text": "<description>"}]}`, отправляет HTTP POST на `{EMBEDDING_BASE_URL}/embed` (через синхронную библиотеку `requests`), парсит ответ и возвращает `List[float]` из 2048 значений.
   - Параллелизация: `asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)` (по умолчанию 8) — ограничивает количество одновременных потоков `asyncio.to_thread`.
   - **Обработка ошибок на уровне отдельной сущности**:
     - `ValueError` (от клиента: невалидный JSON, отсутствие `data[0].embedding`) → сущность пропускается, `embeddings_failed += 1`.
     - `requests.RequestException` (от клиента: HTTP-ошибка 4xx/5xx, таймаут, connection error) → сущность пропускается, `embeddings_failed += 1`.
     - Размерность эмбеддинга не равна 2048 → сущность пропускается, `embeddings_failed += 1`.

5. **Сохранение эмбеддингов в Qdrant**:
   - Для всех успешно полученных embedding-векторов формируется батч точек Qdrant.
- Идентификатор точки (point ID): UUID, сгенерированный как `uuid5(ENTITY_EMBEDDINGS_NAMESPACE, "{TITLE}|{TYPE}")` (детерминированный UUID на основе namespace UUID из конфигурации).
- **Аутентификация Qdrant**: если в конфигурации задан `QDRANT_API_KEY` (не пустой), он передаётся в HTTP-заголовке `api-key` при upsert-запросах к Qdrant. Для публичных/read-only проверок коллекции заголовок не используется.
   - Payload каждой точки:
     ```json
     {
       "entity_title": "<title>",
       "entity_type": "<type>",
       "entity_id": "<title>|<type>",
       "description": "<description>"
     }
     ```
     Поля `entity_title` и `entity_type` в payload позволяют фильтровать/искать в Qdrant и однозначно находить Entity в Neo4j: `MATCH (e:Entity {title: $entity_title, type: $entity_type})`.
   - **Upsert в Qdrant** — асинхронный HTTP-запрос (через `aiohttp`, Constitution P5):
     ```
     PUT {QDRANT_URL}/collections/{ENTITY_EMBEDDINGS_COLLECTION}/points
     Query-параметр: ?wait=true
     Body: {
       "points": [
         {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
           "vector": [0.1, 0.2, ...],
           "payload": {
             "entity_title": "ACME CORP",
             "entity_type": "ORGANIZATION",
             "entity_id": "ACME CORP|ORGANIZATION",
             "description": "ACME Corp is a multinational..."
           }
         }
       ]
     }
     ```
   - **Важно**: используется `?wait=true` для синхронного подтверждения записи — без этого Qdrant может подтвердить запрос до фактической записи, и последующий поиск может не найти точку.
   - **Атомарность**: upsert идемпотентен:
     - Если точка с таким ID уже существует → обновляется вектор и payload
     - Если точки нет → создаётся новая
   - **Батчинг**: точки отправляются батчами по `ENTITY_EMBEDDINGS_BATCH_SIZE` (по умолчанию 100) для эффективности.
   - **Обработка ошибок Qdrant upsert**:
     - HTTP-ошибка (`4xx/5xx`), таймаут → все сущности в батче помечаются как `embeddings_failed`. Не прерывает обработку остальных батчей.
     - Размерность вектора не 2048 (проверка на клиенте до отправки) → сущность пропускается, `embeddings_failed += 1`.

6. **Обновление Neo4j** (`embedding_updated_at`):
   - После каждого успешного Qdrant upsert-батча вызывается `Manager.set_entity_embedding_updated_at(neo4j_batch)`, где `neo4j_batch` — список словарей `[{"title": ..., "type": ...}, ...]` для всех сущностей батча.
   - Cypher внутри транзакции:
     ```cypher
     UNWIND $entities AS row
     MATCH (e:Entity {title: row.title, type: row.type})
     SET e.embedding_updated_at = datetime()
     ```
   - **Важно**: Neo4j обновляется ТОЛЬКО после успешного Qdrant upsert для данного батча. Обратный порядок (сначала Neo4j) недопустим: если Qdrant-запрос упадёт, `embedding_updated_at` окажется установленным, а embedding в Qdrant отсутствует — система будет считать embedding актуальным, хотя его нет.
   - **Обработка ошибок**: если Neo4j-запрос для батча упал (сессионная ошибка драйвера, ServiceUnavailable, TransientError) → выбрасывается `HTTPException(500)`, обработка останавливается. Embedding'и в Qdrant для этого батча уже сохранены и орфанятся; при следующем вызове эндпоинта будут перезаписаны (upsert идемпотентен).
   - Выполняется в рамках синхронной транзакции Neo4j (`session.execute_write`).

7. **Формирование ответа**:
   - Используется функция `_build_response('None', 0, start_time, 'completed', neo4j_stats)` (переиспользование существующего хелпера из `semantic_index.py`).
   - `neo4j_stats` содержит: `embeddings_added`, `embeddings_updated`, `embeddings_skipped`, `embeddings_failed`.

### Output
- **HTTP 200** — операция завершена (все сущности обработаны или кандидатов нет):
  ```json
  {
    "document_id": "None",
    "statistics": {
      "total_chunks": 0,
      "processing_time_ms": <int>,
      "status": "completed",
      "embeddings_added": <int>,
      "embeddings_updated": <int>,
      "embeddings_skipped": <int>,
      "embeddings_failed": <int>
    }
  }
  ```
  Где:
  | Поле | Описание |
  |------|----------|
  | `embeddings_added` | Количество узлов, у которых `embedding_updated_at` отсутствовал (NULL) — embedding создан в Qdrant, `embedding_updated_at` проставлен в Neo4j |
  | `embeddings_updated` | Количество узлов, у которых `embedding_updated_at` существовал, но был пересчитан (`updated_at > embedding_updated_at`) — embedding обновлён в Qdrant, `embedding_updated_at` обновлён в Neo4j |
  | `embeddings_skipped` | Количество узлов, у которых embedding актуален (`updated_at <= embedding_updated_at`). Всегда 0 при корректной выборке (отсеиваются Cypher). |
  | `embeddings_failed` | Количество узлов, для которых вычисление или сохранение эмбеддинга завершилось ошибкой (ошибка embedding-сервиса, ошибка Qdrant upsert, ошибка Neo4j-записи) |

- **HTTP 500** — ошибка на уровне инфраструктуры:
  - Qdrant недоступен при проверке/создании коллекции
  - Neo4j недоступен при чтении кандидатов
  - Несовпадение размерности/метрики коллекции Qdrant

## API Contract

```openapi
GET /compute_entity_embeddings
summary: Вычисление 2048-мерных эмбеддингов для Entity с изменившимся описанием, сохранение их в Qdrant и обновление embedding_updated_at в Neo4j
responses:
  '200':
    description: Вычисление эмбеддингов завершено (все обработаны или кандидатов нет)
    content:
      application/json:
        schema:
          type: object
          properties:
            document_id:
              type: string
              description: Всегда "None" (глобальная операция)
            statistics:
              type: object
              properties:
                total_chunks:
                  type: integer
                  description: Всегда 0
                processing_time_ms:
                  type: integer
                status:
                  type: string
                  enum: [completed]
                embeddings_added:
                  type: integer
                  description: Эмбеддингов добавлено (embedding_updated_at ранее отсутствовал)
                embeddings_updated:
                  type: integer
                  description: Эмбеддингов обновлено (описание изменилось — embedding перезаписан в Qdrant)
                embeddings_skipped:
                  type: integer
                  description: Эмбеддингов пропущено (актуальны)
                embeddings_failed:
                  type: integer
                  description: Эмбеддингов не удалось вычислить или сохранить (ошибка embedding-сервиса, Qdrant или Neo4j)
  '500':
    description: Ошибка инфраструктуры (Qdrant, Neo4j, embedding-сервис недоступен, несовпадение конфигурации коллекции)
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
  │ GET /compute_entity_embeddings
  ▼
semantic_index.py
  │
  ├─[1] Qdrant: проверка/создание коллекции entity_embeddings
  │     ├─ GET  {QDRANT_URL}/collections/entity_embeddings
  │     │   ├─ 200: проверить vectors.size=2048, vectors.distance=Cosine
  │     │   └─ 404: PUT {QDRANT_URL}/collections/entity_embeddings
  │     │         Body: {"vectors": {"size": 2048, "distance": "Cosine"}}
  │     └─ Ошибка → HTTP 500
  │
  ├─[2] Manager.get_entities_needing_embedding()
  │     └─ Neo4j (read): MATCH (e:Entity)
  │        WHERE description IS NOT NULL
  │        AND (embedding_updated_at IS NULL OR updated_at > embedding_updated_at)
  │        RETURN title, type, description, embedding_updated_at, updated_at
  │
  ├─[3] Классификация:
  │     ├─ is_new:       embedding_updated_at IS NULL
  │     └─ is_updated:   embedding_updated_at IS NOT NULL (перезапись в Qdrant)
  │
  ├─[4] Для каждой сущности (параллельно, Semaphore(EMBEDDING_MAX_CONCURRENCY)):
  │     ├─ asyncio.to_thread(emb_client.get_text_embedding, description)
  │     │   └─ EmbeddingClient (sync, requests):
  │     │       POST → {EMBEDDING_BASE_URL}/embed
  │     │       Тело: {"messages": [{"type": "text", "text": "<description>"}]}
  │     │       Ответ: EmbedSuccessResponse → List[float] из 2048 значений
  │     │       Ошибка: requests.HTTPError, requests.Timeout, ValueError
  │     │              → пропуск, +1 failed
  │     └─ Накопление результатов в буфер для батчевого Qdrant upsert
  │
  ├─[5] Qdrant upsert (батчами по ENTITY_EMBEDDINGS_BATCH_SIZE):
  │     └─ PUT {QDRANT_URL}/collections/entity_embeddings/points?wait=true
  │        Body: {"points": [
          {"id": "<uuid5-генерированный идентификатор>", "vector": [...],
  │           "payload": {"entity_title": "...", "entity_type": "...",
  │                       "entity_id": "...", "description": "..."}}
  │        ]}
  │        Ошибка: Qdrant 4xx/5xx, таймаут → все точки батча → +N failed
  │
  └─[6] Neo4j (write): для успешно сохранённых в Qdrant:
        └─ MATCH (e:Entity {title: $title, type: $type})
           SET e.embedding_updated_at = datetime()
           Ошибка: сущность не найдена, ошибка Neo4j → +1 failed (embedding в Qdrant остаётся)
           Успех → +1 added / +1 updated
```

## LLM Interactions
Отсутствуют. Вычисление эмбеддингов — операция вызова `EmbeddingClient` (обёртка над embedding-сервисом, не LLM). Промпты не используются.

## LLM Model Requirements
Не применимо. LLM не используется. Embedding-модель (`qwen3-emb`) работает на стороне внешнего сервиса и не требует конфигурации со стороны `semantic_graph`.

## Error Handling

| Сценарий | Код | Поведение |
|----------|-----|-----------|
| Qdrant недоступен при проверке/создании коллекции | 500 | Unhandled exception → HTTP 500. Detail: «Qdrant unavailable: {error}» |
| Qdrant коллекция существует, но размерность ≠ 2048 | 500 | Unhandled exception → HTTP 500. Detail: «Qdrant collection dimension mismatch: expected 2048, got {N}» |
| Qdrant коллекция существует, но метрика ≠ Cosine | 500 | Unhandled exception → HTTP 500. Detail: «Qdrant collection distance metric mismatch: expected Cosine, got {metric}» |
| Ошибка создания коллекции Qdrant | 500 | Unhandled exception → HTTP 500. Detail: «Failed to create Qdrant collection: {error}» |
| Neo4j недоступен при чтении кандидатов | 500 | Unhandled exception → HTTP 500 |
| Кандидатов нет (все эмбеддинги актуальны) | 200 | `embeddings_added: 0`, `embeddings_updated: 0`, `embeddings_skipped: 0`, `embeddings_failed: 0` |
| EmbeddingClient: requests.HTTPError для отдельной сущности | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется с `title`, `type`, HTTP-статусом. |
| EmbeddingClient: requests.Timeout для отдельной сущности | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется. |
| EmbeddingClient: ValueError (невалидный ответ/парсинг) | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется. |
| Размерность эмбеддинга не равна 2048 | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется с фактической размерностью. |
| Qdrant ошибка upsert для батча (4xx/5xx, таймаут) | 200 | Все сущности батча пропускаются, `embeddings_failed += N`. Ошибка логируется с размером батча и HTTP-статусом/ошибкой. |
| Qdrant ошибка upsert для отдельной точки в батче | 200 | Qdrant REST API `PUT .../points` обрабатывает батч атомарно — либо весь батч принят, либо нет. Частичный отказ не предусмотрен протоколом. |
| Neo4j недоступен при записи `SET e.embedding_updated_at` (ошибка сессии/драйвера) | 500 | Unhandled exception → HTTP 500. Detail: «Neo4j write error: {error}». Обработка останавливается. Embedding'и в Qdrant для батча уже сохранены и орфанятся. |
| Сущность не найдена в Neo4j при записи | 200 | `embeddings_failed += 1` (гонка данных — удалена между чтением и записью). Embedding в Qdrant орфанится. |
| Qdrant upsert успешен, но Neo4j `SET embedding_updated_at` упал | 200 | `embeddings_failed += 1`. Embedding в Qdrant орфанится, но при следующем вызове эндпоинта `embedding_updated_at IS NULL` → сущность будет переобработана (upsert перезапишет embedding). Система самовосстанавливается. |

**Приоритет отказоустойчивости**: орфанный embedding в Qdrant (без `embedding_updated_at` в Neo4j) безопаснее, чем `embedding_updated_at` без embedding'а в Qdrant. Поэтому порядок: **сначала Qdrant upsert, потом Neo4j**. Система самовосстанавливается при следующем вызове эндпоинта.

## Testing

### Тест-кейсы

1. **Успешное вычисление для новых сущностей**: 5 Entity без `embedding_updated_at`, у всех есть `description` → HTTP 200, `embeddings_added: 5`, `embeddings_updated: 0`, `embeddings_failed: 0`. После запроса:
   - В Qdrant коллекции `entity_embeddings` — 5 точек с id `"<TITLE>|<TYPE>"`, vector[2048], payload с `entity_title`, `entity_type`, `entity_id`, `description`
   - В Neo4j все 5 узлов имеют `embedding_updated_at` (DateTime), поле `embedding` отсутствует

2. **Успешное обновление**: 3 Entity с `embedding_updated_at`, у всех `updated_at > embedding_updated_at` → HTTP 200, `embeddings_added: 0`, `embeddings_updated: 3`, `embeddings_failed: 0`. После запроса:
   - В Qdrant 3 точки обновлены (вектор и payload перезаписаны)
   - В Neo4j `embedding_updated_at` обновлён на всех 3 узлах

3. **Смешанный сценарий**: 2 новых + 3 обновляемых + 1 с актуальным embedding → HTTP 200, `embeddings_added: 2`, `embeddings_updated: 3`, `embeddings_skipped: 0`, `embeddings_failed: 0`. Сущность с актуальным embedding не попадает в выборку.

4. **Все эмбеддинги актуальны**: 10 Entity, у всех `embedding_updated_at IS NOT NULL` и `updated_at <= embedding_updated_at` → HTTP 200, `embeddings_added: 0`, `embeddings_updated: 0`, `embeddings_failed: 0`.

5. **Сущности без описания**: 5 Entity с `description IS NULL` (независимо от `embedding_updated_at`) → не попадают в выборку, HTTP 200, все счётчики 0.

6. **Пустой граф**: 0 Entity → HTTP 200, все счётчики 0.

7. **Создание коллекции Qdrant с нуля**: коллекция `entity_embeddings` отсутствует в Qdrant. Вызов эндпоинта → коллекция создана (2048-dim, Cosine), embedding'и вычислены и сохранены → HTTP 200.

8. **Qdrant коллекция с неверной размерностью**: коллекция `entity_embeddings` существует, но с размерностью 1024 → HTTP 500, detail: «Qdrant collection dimension mismatch».

9. **Qdrant коллекция с неверной метрикой**: коллекция `entity_embeddings` существует, но с метрикой `Euclid` → HTTP 500, detail: «Qdrant collection distance metric mismatch».

10. **Частичный отказ embedding-сервиса**: 10 сущностей, мок `EmbeddingClient.get_text_embedding`, который для 2 сущностей выбрасывает `requests.HTTPError` → HTTP 200, `embeddings_added + embeddings_updated: 8`, `embeddings_failed: 2`.

11. **Невалидный ответ embedding-сервиса**: мок `EmbeddingClient.get_text_embedding` выбрасывает `ValueError` → сущность пропускается, `embeddings_failed += 1`.

12. **Размерность эмбеддинга не совпадает**: мок `EmbeddingClient.get_text_embedding` возвращает вектор из 1024 float-значений → сущность пропускается, `embeddings_failed += 1`.

13. **Ошибка Qdrant upsert для батча**: мок Qdrant возвращает 500 на `PUT .../points` → все сущности батча помечаются как `embeddings_failed`. Остальные батчи обрабатываются нормально.

14. **Neo4j ошибка при записи `embedding_updated_at` после успешного Qdrant upsert**: после успешного Qdrant upsert, мок Neo4j возвращает ошибку сессии/драйвера → HTTP 500. Embedding'и в Qdrant для всего батча остаются (орфанятся). При повторном вызове эндпоинта сущности переобрабатываются (upsert перезапишет embedding'и).

15. **Neo4j недоступен**: неверные креды → HTTP 500.

16. **Qdrant недоступен**: неверный URL → HTTP 500 (при шаге 1 — проверка коллекции).

17. **Параллелизм**: 20 сущностей, `Semaphore(8)` → все вызовы `asyncio.to_thread(emb_client.get_text_embedding, ...)` выполнены, одновременно не более 8 активных потоков.

18. **Параллельный Qdrant upsert с корректным батчингом**: 250 сущностей, `ENTITY_EMBEDDINGS_BATCH_SIZE=100` → 3 батча (100 + 100 + 50). Проверка, что все точки попали в Qdrant, и все `embedding_updated_at` проставлены.

19. **Повторный вызов без изменений**: После успешного первого вызова (все `embedding_updated_at` установлены) повторный вызов → HTTP 200, `embeddings_added: 0`, `embeddings_updated: 0` (все отсеяны Cypher-запросом).

20. **Повторный вызов после изменения описаний**: Обновить `description` и `updated_at` у 3 сущностей, вызвать эндпоинт → HTTP 200, `embeddings_updated: 3`. Проверить, что векторы в Qdrant изменились (upsert перезаписал старые).

21. **Проверка payload Qdrant**: После успешного вычисления через Qdrant REST API получить точку по id (UUID5-сгенерированный идентификатор) и проверить, что payload содержит `entity_title`, `entity_type`, `entity_id`, `description`.

22. **Идемпотентность Qdrant upsert**: Повторно отправить тот же вектор для существующей точки → вектор и payload перезаписаны. Точка не дублируется.

23. **Проверка matchability Qdrant ↔ Neo4j**: По payload точки (`entity_title`, `entity_type`) выполнить поиск в Neo4j: `MATCH (e:Entity {title: $entity_title, type: $entity_type})` — должен найти ровно 1 узел.

### Подход к тестированию
- **Unit**: `Manager.get_entities_needing_embedding()` на мокнутом Neo4j-соединении; `Manager.set_entity_embedding_updated_at()` с проверкой параметров Cypher-запроса; классификация `is_new`/`is_updated` на фиктивных словарях; `EmbeddingClient.get_text_embedding()` с мокнутым `requests.Session`.
- **Integration**: эндпоинт с мокнутым `Manager`, мокнутым `EmbeddingClient` и мокнутым `aiohttp.ClientSession` (для Qdrant); проверка корректности статистики при разных наборах входных данных.
- **API**: httpx-запрос к эндпоинту с мокнутыми зависимостями → проверка структуры ответа и HTTP-кодов.
- **Pipeline**: сквозной тест с реальным Neo4j (testcontainers), реальным Qdrant (testcontainers или docker-compose) и мокнутым `EmbeddingClient`. Проверка полного цикла: Neo4j → embedding service → Qdrant upsert → Neo4j `embedding_updated_at`.

## Dependencies
- **Neo4j** (read/write): `Manager.get_entities_needing_embedding()` — чтение кандидатов; `Manager.set_entity_embedding_updated_at()` — запись `embedding_updated_at` через `SET e.embedding_updated_at = datetime()`.
- **Qdrant** (read/write):
  - **read**: `GET /collections/{collection}` — проверка существования, размерности и метрики коллекции `entity_embeddings`
  - **write**: `PUT /collections/{collection}` — создание коллекции (если не существует)
  - **write**: `PUT /collections/{collection}/points?wait=true` — upsert точек (векторы + payload) батчами
  - Протокол: REST API через `aiohttp` (Constitution P5: async для I/O-bound операций)
  - URL: `QDRANT_URL` из `config.py`
- **EmbeddingClient** (из `app.src.qwen3_emb_client`): синхронный HTTP-клиент на базе `requests` для вызова embedding-сервиса.
  - Инициализация: `EmbeddingClient(base_url=EMBEDDING_BASE_URL, timeout=EMBEDDING_TIMEOUT)`.
  - Метод: `get_text_embedding(text: str) -> List[float]` — POST `{base_url}/embed` с телом `{"messages": [{"type": "text", "text": "<description>"}]}`, парсинг `EmbedSuccessResponse.messages[0].embedding`.
  - Асинхронная обёртка: `asyncio.to_thread(emb_client.get_text_embedding, description)` для неблокирующей работы в async-эндпоинте FastAPI.
  - Модель: `qwen3-emb`, 2048-мерные векторы.
- **uuid**: модуль стандартной библиотеки Python для генерации детерминированного UUID5 point ID в Qdrant на основе `ENTITY_EMBEDDINGS_NAMESPACE`.
- **aiohttp** (HTTP-клиент): асинхронные HTTP-запросы к Qdrant REST API (Constitution P5: async для I/O-bound операций).
- **asyncio**: `Semaphore` для ограничения параллелизма потоков (`EMBEDDING_MAX_CONCURRENCY`), `asyncio.gather` для параллельного выполнения, `asyncio.to_thread` для адаптации синхронного `EmbeddingClient`.

## Exceptions

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует, чтобы `archived`-сущности исключались из обработки. В текущей схеме Neo4j:
- Поле `archived` на узлах `Entity` отсутствует.
- Cypher-запрос `get_entities_needing_embedding()` не фильтрует по `archived`.
- При реализации мягкого удаления потребуется добавить условие `AND (e.archived IS NULL OR e.archived = false)` в Cypher-запрос на шаге 2.

### N3 — Конфигурация не Pydantic Settings (НАРУШЕНИЕ)
Новые параметры `EMBEDDING_BASE_URL`, `EMBEDDING_TIMEOUT`, `EMBEDDING_MAX_CONCURRENCY`, `ENTITY_EMBEDDINGS_COLLECTION`, `ENTITY_EMBEDDINGS_BATCH_SIZE` добавляются как модульные константы в `config.py`, а не через Pydantic `BaseSettings` в `config/settings.py`. Обоснование: переходный период — вся конфигурация сервиса `semantic_graph` в настоящее время использует модульные константы (см. SERVICE.md, Exceptions #2). После рефакторинга конфигурации на Pydantic Settings эти параметры должны быть перенесены в Settings-класс.

### P5 — EmbeddingClient синхронный с asyncio.to_thread (НЕ НАРУШЕНИЕ)
`EmbeddingClient` из `app.src.qwen3_emb_client` использует синхронную библиотеку `requests` (задокументированное нарушение P5 в самом `EmbeddingClient`). Для соответствия Constitution P5 на уровне эндпоинта синхронный вызов `emb_client.get_text_embedding()` оборачивается в `asyncio.to_thread()`, что предотвращает блокировку event loop. Ограничение параллелизма реализовано через `asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)`, контролирующее количество одновременных потоков. Это компромиссное решение на период до миграции `EmbeddingClient` на `httpx.AsyncClient` (запланировано в `docs/CONSTITUTION_COMPLIANCE_PLAN.md`).

### P5 — Асинхронный Qdrant-клиент для записи (НЕ НАРУШЕНИЕ)
Существующий `QdrantStreamAdapter` (`Qdrant_extractor/qdrant_adapter.py`) использует синхронный `requests` для scroll-чтения. Для операций записи (создание коллекции, upsert точек) в рамках этого эндпоинта используется новый асинхронный клиент на `aiohttp` — прямое взаимодействие с Qdrant REST API. Это соответствует Constitution P5 (асинхронность для I/O-bound операций). Существующий синхронный `QdrantStreamAdapter` не модифицируется и продолжает использоваться в `POST /process-document` для чтения чанков (задокументированное нарушение P5 в SERVICE.md, Exceptions #6).

### P9 — Глобальный граф (НЕ НАРУШЕНИЕ)
Операция является always-full-graph, что прямо разрешено Constitution P9: глобальные операции в `semantic_graph` работают со всем графом Neo4j.

### P7 — Промпты инлайн (НЕ ПРИМЕНИМО)
Эндпоинт не использует LLM. Embedding-сервис (`qwen3-emb`) — это отдельный embedding-only сервис, вызываемый через `EmbeddingClient`. Промпты отсутствуют. Нарушения P7 нет.

### P1 — Сервисные границы и Qdrant (НЕ НАРУШЕНИЕ)
Согласно Constitution P1, Qdrant является самостоятельным сервисом с выделенным хранилищем (порт 6333). `semantic_graph` взаимодействует с Qdrant через REST API — это соответствует правилу «только через API или выделенный клиентский модуль». Использование коллекции `entity_embeddings` в Qdrant не пересекается с коллекцией `documents`, используемой сервисом `app` для векторного поиска чанков.
