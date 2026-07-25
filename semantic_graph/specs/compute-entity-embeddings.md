# Feature: Вычисление эмбеддингов сущностей (compute_entity_embeddings)

## Motivation
При построении семантического графа знаний сущности (`Entity`) накапливают описания (`description`), извлечённые LLM из текстовых чанков. Для задач семантического поиска, кластеризации и рекомендаций необходимо иметь векторные представления (эмбеддинги) этих сущностей. Эндпоинт решает задачу **пакетного вычисления и сохранения** эмбеддингов: для всех сущностей, у которых изменилось описание, генерируется 2048-мерный embedding через внешний HTTP-сервис, результат сохраняется в коллекцию Qdrant `entity_embeddings`, а факт обновления (таймстемп) фиксируется в Neo4j на узле `Entity` (поле `embedding_updated_at`). Сущности с уже актуальными эмбеддингами пропускаются. Это глобальная операция (Constitution P9) — работает со всем графом, без фильтрации по документу.

Архитектурное решение хранить эмбеддинги в Qdrant, а не в Neo4j, обусловлено:
- Qdrant — векторная БД, оптимизированная для ANN-поиска и векторных операций
- Neo4j не нагружается тяжёлыми `list[float]`-свойствами (экономятся ресурсы при B-tree-операциях, бэкапах, репликации)
- Разделение зон ответственности: Neo4j — графовые связи, Qdrant — векторные представления

## Behaviour

### Input
- **Метод**: `GET`
- **Path**: `/compute_entity_embeddings`
- **Параметры**: нет. Параметры подключения к embedding-сервису (`EMBEDDING_BASE_URL`, `EMBEDDING_TIMEOUT`, `EMBEDDING_MAX_CONCURRENCY`), Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`), и имя коллекции (`ENTITY_EMBEDDINGS_COLLECTION`) заданы в конфигурации сервиса.
- **Валидация**: отсутствует (эндпоинт без параметров).

### Processing

**Предварительные условия**:
- В Qdrant по адресу `QDRANT_URL` доступна (или будет создана) коллекция `entity_embeddings` с конфигурацией: размерность векторов 2048, метрика `COSINE`.
- В Neo4j на узлах `Entity` существует поле `embedding_updated_at` (тип `DateTime`). Поле `embedding` (тип `list[float]`) на узлах `Entity` **отсутствует** — эмбеддинги хранятся исключительно в Qdrant.
- При первом запуске `embedding_updated_at IS NULL` на всех существующих узлах — все сущности с `description IS NOT NULL` будут классифицированы как «новые» (embedding отсутствует в Qdrant).
- Embedding-сервис доступен по HTTP и принимает запросы по адресу `{EMBEDDING_BASE_URL}/embed`.

**Общая схема**:

```
Neo4j (read candidates) → Embedding Service (batched) → Qdrant (upsert) → Neo4j (write embedding_updated_at)
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

4. **Параллельный вызов embedding-сервиса**:
   - Для каждой сущности из выборки:
     - Формируется тело запроса: `{"messages": [{"type": "text", "text": "<description>"}]}`.
     - Отправляется асинхронный HTTP POST (через `aiohttp` — Constitution P5) на `{EMBEDDING_BASE_URL}/embed`.
     - Таймаут запроса: `EMBEDDING_TIMEOUT` (по умолчанию 30 сек).
     - Ожидаемый ответ:
       ```json
       {"data": [{"embedding": [0.1, 0.2, ...]}]}
       ```
       Из ответа извлекается `data[0].embedding` — список из 2048 float-значений.
   - Параллелизация: `asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)` (по умолчанию 8), так как embedding-сервис CPU-bound.
   - **Обработка ошибок на уровне отдельной сущности**:
     - HTTP-ошибка (`4xx/5xx`), таймаут, невалидный JSON или отсутствие `data[0].embedding` → сущность пропускается, инкрементируется счётчик `embeddings_failed`.

5. **Сохранение эмбеддингов в Qdrant**:
   - Для всех успешно полученных embedding-векторов формируется батч точек Qdrant.
   - Идентификатор точки (point ID): строка `"{TITLE}|{TYPE}"` (например, `"ACME CORP|ORGANIZATION"`). Этот ID совпадает с уникальным constraint в Neo4j (`entity_title_type_unique`) — однозначная связь Qdrant ↔ Neo4j.
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
           "id": "ACME CORP|ORGANIZATION",
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
   - Для всех сущностей, чьи embedding'и были успешно сохранены в Qdrant, вызывается метод `Manager.set_entity_embedding_updated_at()`:
     ```cypher
     MATCH (e:Entity {title: $title, type: $type})
     SET e.embedding_updated_at = datetime()
     ```
   - **Важно**: Neo4j обновляется ТОЛЬКО после успешного Qdrant upsert. Обратный порядок (сначала Neo4j) недопустим: если Qdrant-запрос упадёт, `embedding_updated_at` окажется установленным, а embedding в Qdrant отсутствует — система будет считать embedding актуальным, хотя его нет.
   - Если Neo4j-запрос для отдельной сущности упал (после успешного Qdrant upsert) → сущность помечается как `embeddings_failed`, но embedding в Qdrant остаётся (орфанится; при следующем вызове эндпоинта будет перезаписан).
   - Выполняется в рамках асинхронной транзакции Neo4j.

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
  - Embedding-сервис полностью недоступен (ошибка на первом же запросе)
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
  │     ├─ aiohttp POST → {EMBEDDING_BASE_URL}/embed
  │     │   Тело: {"messages": [{"type": "text", "text": "<description>"}]}
  │     │   Ответ: {"data": [{"embedding": [0.1, ..., 2048 значений]}]}
  │     │   Ошибка: HTTP 4xx/5xx, таймаут, невалидный JSON, размерность≠2048
  │     │          → пропуск, +1 failed
  │     └─ Накопление результатов в буфер для батчевого Qdrant upsert
  │
  ├─[5] Qdrant upsert (батчами по ENTITY_EMBEDDINGS_BATCH_SIZE):
  │     └─ PUT {QDRANT_URL}/collections/entity_embeddings/points?wait=true
  │        Body: {"points": [
  │          {"id": "<TITLE>|<TYPE>", "vector": [...],
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
Отсутствуют. Вычисление эмбеддингов — операция вызова внешнего embedding-сервиса (не LLM). Промпты не используются.

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
| Embedding-сервис недоступен (connection error до запросов) | 500 | Unhandled exception → HTTP 500 (нет смысла продолжать — все последующие запросы тоже не пройдут) |
| Embedding-сервис вернул 4xx/5xx для отдельной сущности | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется с `title`, `type`, HTTP-статусом. |
| Таймаут запроса к embedding-сервису | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется. |
| Ответ embedding-сервиса — невалидный JSON | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется. |
| Ответ embedding-сервиса — отсутствует `data[0].embedding` | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется. |
| Размерность эмбеддинга не равна 2048 | 200 | Сущность пропускается, `embeddings_failed += 1`. Ошибка логируется с фактической размерностью. |
| Qdrant ошибка upsert для батча (4xx/5xx, таймаут) | 200 | Все сущности батча пропускаются, `embeddings_failed += N`. Ошибка логируется с размером батча и HTTP-статусом/ошибкой. |
| Qdrant ошибка upsert для отдельной точки в батче | 200 | Qdrant REST API `PUT .../points` обрабатывает батч атомарно — либо весь батч принят, либо нет. Частичный отказ не предусмотрен протоколом. |
| Neo4j недоступен при записи `SET e.embedding_updated_at` | 500 | Если ошибка уровня сессии/драйвера (ServiceUnavailable) → retry 3 раза с exponential backoff. Если retry исчерпаны → HTTP 500 (дальнейшие записи невозможны). |
| Neo4j ошибка записи для отдельной сущности (TransientError) | 200 | Retry до 3 раз (exponential backoff). Если retry исчерпаны → `embeddings_failed += 1`. Не прерывает обработку остальных сущностей. Embedding в Qdrant уже сохранён (орфанится, при следующем вызове будет перезаписан). |
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

10. **Частичный отказ embedding-сервиса**: 10 сущностей, мок на embedding-сервис, который для 2 сущностей возвращает 500 → HTTP 200, `embeddings_added + embeddings_updated: 8`, `embeddings_failed: 2`.

11. **Невалидный ответ embedding-сервиса**: мок возвращает `{"data": []}` → сущность пропускается, `embeddings_failed += 1`.

12. **Размерность эмбеддинга не совпадает**: мок возвращает вектор из 1024 float-значений → сущность пропускается, `embeddings_failed += 1`.

13. **Ошибка Qdrant upsert для батча**: мок Qdrant возвращает 500 на `PUT .../points` → все сущности батча помечаются как `embeddings_failed`. Остальные батчи обрабатываются нормально.

14. **Neo4j ошибка при записи `embedding_updated_at` после успешного Qdrant upsert**: после успешного Qdrant upsert, мок Neo4j возвращает ошибку для одной сущности → `embeddings_failed += 1`. Embedding в Qdrant остаётся. При повторном вызове эндпоинта эта сущность переобрабатывается (embedding обновляется через upsert).

15. **Neo4j недоступен**: неверные креды → HTTP 500.

16. **Embedding-сервис недоступен полностью**: неверный URL → HTTP 500 (после первого же неудачного запроса).

17. **Qdrant недоступен**: неверный URL → HTTP 500 (при шаге 1 — проверка коллекции).

18. **Параллелизм**: 20 сущностей, `Semaphore(8)` → все запросы к embedding-сервису выполнены, одновременно не более 8 активных соединений.

19. **Параллельный Qdrant upsert с корректным батчингом**: 250 сущностей, `ENTITY_EMBEDDINGS_BATCH_SIZE=100` → 3 батча (100 + 100 + 50). Проверка, что все точки попали в Qdrant, и все `embedding_updated_at` проставлены.

20. **Повторный вызов без изменений**: После успешного первого вызова (все `embedding_updated_at` установлены) повторный вызов → HTTP 200, `embeddings_added: 0`, `embeddings_updated: 0` (все отсеяны Cypher-запросом).

21. **Повторный вызов после изменения описаний**: Обновить `description` и `updated_at` у 3 сущностей, вызвать эндпоинт → HTTP 200, `embeddings_updated: 3`. Проверить, что векторы в Qdrant изменились (upsert перезаписал старые).

22. **Проверка payload Qdrant**: После успешного вычисления через Qdrant REST API получить точку по id `"<TITLE>|<TYPE>"` и проверить, что payload содержит `entity_title`, `entity_type`, `entity_id`, `description`.

23. **Идемпотентность Qdrant upsert**: Повторно отправить тот же вектор для существующей точки → вектор и payload перезаписаны. Точка не дублируется.

24. **Проверка matchability Qdrant ↔ Neo4j**: По payload точки (`entity_title`, `entity_type`) выполнить поиск в Neo4j: `MATCH (e:Entity {title: $entity_title, type: $entity_type})` — должен найти ровно 1 узел.

### Подход к тестированию
- **Unit**: `Manager.get_entities_needing_embedding()` на мокнутом Neo4j-соединении; `Manager.set_entity_embedding_updated_at()` с проверкой параметров Cypher-запроса; классификация `is_new`/`is_updated` на фиктивных словарях; функция проверки/создания коллекции Qdrant на мокнутом `aiohttp.ClientSession`.
- **Integration**: эндпоинт с мокнутым `Manager` и мокнутым embedding-сервисом (через `aioresponses` или `pytest-aiohttp`); мокнутый Qdrant REST API; проверка корректности статистики при разных наборах входных данных.
- **API**: httpx-запрос к эндпоинту с мокнутыми зависимостями → проверка структуры ответа и HTTP-кодов.
- **Pipeline**: сквозной тест с реальным Neo4j (testcontainers), реальным Qdrant (testcontainers или docker-compose) и мокнутым embedding-сервисом. Проверка полного цикла: Neo4j → embedding service → Qdrant upsert → Neo4j `embedding_updated_at`.

## Dependencies
- **Neo4j** (read/write): `Manager.get_entities_needing_embedding()` — чтение кандидатов; `Manager.set_entity_embedding_updated_at()` — запись `embedding_updated_at` через `SET e.embedding_updated_at = datetime()`.
- **Qdrant** (read/write):
  - **read**: `GET /collections/{collection}` — проверка существования, размерности и метрики коллекции `entity_embeddings`
  - **write**: `PUT /collections/{collection}` — создание коллекции (если не существует)
  - **write**: `PUT /collections/{collection}/points?wait=true` — upsert точек (векторы + payload) батчами
  - Протокол: REST API через `aiohttp` (Constitution P5: async для I/O-bound операций)
  - URL: `QDRANT_URL` из `config.py`
- **Embedding-сервис** (HTTP): внешний сервис на `{EMBEDDING_BASE_URL}/embed`, модель `qwen3-emb`, 2048-мерные векторы. Протокол: POST JSON `{"messages": [{"type": "text", "text": "<description>"}]}` → `{"data": [{"embedding": [...]}]}`.
- **aiohttp** (HTTP-клиент): асинхронные HTTP-запросы к embedding-сервису и Qdrant REST API (Constitution P5: async для I/O-bound операций).
- **asyncio**: `Semaphore` для ограничения параллелизма, `asyncio.gather` для параллельного выполнения запросов.

## Exceptions

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует, чтобы `archived`-сущности исключались из обработки. В текущей схеме Neo4j:
- Поле `archived` на узлах `Entity` отсутствует.
- Cypher-запрос `get_entities_needing_embedding()` не фильтрует по `archived`.
- При реализации мягкого удаления потребуется добавить условие `AND (e.archived IS NULL OR e.archived = false)` в Cypher-запрос на шаге 2.

### N3 — Конфигурация не Pydantic Settings (НАРУШЕНИЕ)
Новые параметры `EMBEDDING_BASE_URL`, `EMBEDDING_TIMEOUT`, `EMBEDDING_MAX_CONCURRENCY`, `ENTITY_EMBEDDINGS_COLLECTION`, `ENTITY_EMBEDDINGS_BATCH_SIZE` добавляются как модульные константы в `config.py`, а не через Pydantic `BaseSettings` в `config/settings.py`. Обоснование: переходный период — вся конфигурация сервиса `semantic_graph` в настоящее время использует модульные константы (см. SERVICE.md, Exceptions #2). После рефакторинга конфигурации на Pydantic Settings эти параметры должны быть перенесены в Settings-класс.

### P5 — Асинхронный Qdrant-клиент для записи (НЕ НАРУШЕНИЕ)
Существующий `QdrantStreamAdapter` (`Qdrant_extractor/qdrant_adapter.py`) использует синхронный `requests` для scroll-чтения. Для операций записи (создание коллекции, upsert точек) в рамках этого эндпоинта используется новый асинхронный клиент на `aiohttp` — прямое взаимодействие с Qdrant REST API. Это соответствует Constitution P5 (асинхронность для I/O-bound операций). Существующий синхронный `QdrantStreamAdapter` не модифицируется и продолжает использоваться в `POST /process-document` для чтения чанков (задокументированное нарушение P5 в SERVICE.md, Exceptions #6).

### P9 — Глобальный граф (НЕ НАРУШЕНИЕ)
Операция является always-full-graph, что прямо разрешено Constitution P9: глобальные операции в `semantic_graph` работают со всем графом Neo4j.

### P7 — Промпты инлайн (НЕ ПРИМЕНИМО)
Эндпоинт не использует LLM. Embedding-сервис (`qwen3-emb`) — это отдельный embedding-only сервис, вызываемый через HTTP API. Промпты отсутствуют. Нарушения P7 нет.

### P1 — Сервисные границы и Qdrant (НЕ НАРУШЕНИЕ)
Согласно Constitution P1, Qdrant является самостоятельным сервисом с выделенным хранилищем (порт 6333). `semantic_graph` взаимодействует с Qdrant через REST API — это соответствует правилу «только через API или выделенный клиентский модуль». Использование коллекции `entity_embeddings` в Qdrant не пересекается с коллекцией `documents`, используемой сервисом `app` для векторного поиска чанков.
