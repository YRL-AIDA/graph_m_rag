# Feature: Семантический поиск по графу знаний (search-graph)

## Motivation
Эндпоинт выполняет семантический поиск релевантных элементов графа знаний для ответа на вопрос пользователя. По строке вопроса находятся релевантные текстовые блоки (из коллекции Qdrant `documents`), сущности и их связи (из семантического графа в Neo4j + векторной коллекции Qdrant `entity_embeddings`), а также сообщества (из Neo4j + Qdrant `community_embeddings`). Результат — три отсортированных пула текстовых описаний (`text_units`, `entities`, `communities`), каждый ограничен заданной долей от максимального размера вывода в токенах (`max_tokens`). Эндпоинт предназначен для использования сервисом `app` на этапе вопросно-ответного интерфейса (Q&A) — найденные блоки контекста подаются в LLM вместе с вопросом пользователя для генерации ответа. Это read-only операция (Constitution P9 — глобальный поиск без фильтрации по документу).

## Behaviour

### Input
- **Метод**: `POST`
- **Path**: `/search`
- **Content-Type**: `application/json`
- **Тело запроса**: Pydantic-модель `SearchRequest` (см. `dtype/search.py`):

```json
{
  "question": "<строка вопроса пользователя>",
  "max_tokens": 4000,
  "proportions": {
    "text_units": 0.5,
    "entities": 0.25,
    "communities": 0.25
  },
  "documents_filter": "text_only"
}
```

**Поля**:

| Поле | Тип | Обязательное | По умолчанию | Описание |
|------|-----|-------------|--------------|----------|
| `question` | `str` | Да | — | Строка вопроса пользователя (непустая). Валидация: `min_length=1`. |
| `max_tokens` | `int` | Да | — | Максимальный суммарный размер вывода в токенах. Валидация: `gt=0`. |
| `proportions` | `Dict[str, float]` | Нет | `{"text_units": 0.5, "entities": 0.25, "communities": 0.25}` | Доли токенов для каждого типа объектов. Ключи: `text_units`, `entities`, `communities`. Валидация: сумма значений == 1.0 ± 0.001, каждое значение ≥ 0. |
| `documents_filter` | `str` | Нет | `"text_only"` | Режим фильтрации коллекции `documents`: `"text_only"` — только `element_type == "text"`, `"all"` — без фильтра по `element_type`. Валидация: одно из `{"text_only", "all"}`. |

**Валидация**:
- `question` — непустая строка → HTTP 422 (FastAPI auto)
- `max_tokens` — положительное целое (`gt=0`) → HTTP 422
- `proportions` — если передан, сумма значений `text_units + entities + communities` должна быть равна 1.0 ± 0.001 (float tolerance); значения ≥ 0; ключи строго `{"text_units", "entities", "communities"}` → HTTP 422 с сообщением `"proportions must sum to 1.0"` или `"proportions values must be non-negative"`
- `documents_filter` — одно из `{"text_only", "all"}` → HTTP 422

### Processing

**Общая схема**: LLM-извлечение сущностей из вопроса → вычисление эмбеддингов → поиск текстовых блоков (Qdrant `documents`) → для каждой сущности: поиск точки входа (Qdrant `entity_embeddings`) + обход графа (Neo4j: точка входа, связи `RELATED`, сообщества `CONSISTS_OF`) → fallback-путь (если сущностей нет или не найдены в графе) → round-robin объединение пулов → формирование ответа со статистикой.

#### Шаг 0: Инициализация и расчёт бюджетов токенов
- Извлечь доли из `proportions` (или использовать значения по умолчанию):
  ```
  proportions = request.proportions or {"text_units": 0.5, "entities": 0.25, "communities": 0.25}
  ```
- Вычислить бюджеты токенов:
  ```
  text_budget       = int(max_tokens * proportions["text_units"])
  entity_budget     = int(max_tokens * proportions["entities"])
  community_budget  = int(max_tokens * proportions["communities"])
  ```
- Все бюджеты — целые числа (округление вниз за счёт `int()`).
- Запомнить `start_time = time.time()`.

#### Шаг 1: Предобработка вопроса и извлечение сущностей
1.1. Привести `question` к нижнему регистру: `question_lower = question.lower()`.

1.2. Выполнить извлечение сущностей из вопроса через LLM:
- Определить параметры LLM для query extraction:
  - `extraction_model = QUERY_EXTRACTION_MODEL_NAME or MODEL_NAME` (отдельная константа, fallback на основную модель)
  - `extraction_llm_url = QUERY_EXTRACTION_LLM_URL or LLM_URL` (отдельный URL, fallback на основной)
  - `extraction_api_key = QUERY_EXTRACTION_API_KEY or LLM_API_KEY` (отдельный ключ, fallback на основной)
  - `extraction_tokenizer_url = QUERY_EXTRACTION_TOKENIZER_URL or TOKENIZER_URL` (отдельный URL токенизатора, fallback на основной)
- Создать экземпляр `AsyncLLMClient(base_url=extraction_llm_url, tokenizer_url=extraction_tokenizer_url, api_key=extraction_api_key)`.
- Создать экземпляр `AsyncGraphExtractor(llm_client, model=extraction_model, max_gleanings=0)`.
- Вызвать `await extractor.extract(text=question_lower, entity_types=ENTITY_TYPES, source_id="search_query")`.
- `extract` внутри использует промпт `GRAPH_EXTRACTION_PROMPT.format(input_text=question_lower, entity_types=",".join(ENTITY_TYPES))`, вызывает LLM через `AsyncLLMClient.generate()`, парсит ответ через `_parse_result()`, возвращает кортеж `(entities_df, relationships_df)` — pandas DataFrame.
- Из `entities_df` извлечь список сущностей: каждая — `{"title": str, "type": str, "description": str}` (поля DataFrame: `title`, `type`, `description`).
- Если LLM вернул пустой ответ (`response_text` is None/empty) или `entities_df` пуст (0 строк) → `entities_from_question = []`.
- Если LLM API выбросил исключение (HTTP-ошибка, таймаут) → перехватить, залогировать warning, `entities_from_question = []`.
- Сохранить `len(entities_from_question)` для статистики.
- **Примечание для VL-моделей**: если `QUERY_EXTRACTION_MODEL_NAME` указывает на мультимодальную (VL) модель, в текущей версии эндпоинта изображение не передаётся — модель получает только текст вопроса. Архитектура (отдельный `AsyncLLMClient` с собственными параметрами подключения) позволяет в будущем расширить запрос на поддержку изображений без изменения сигнатуры эндпоинта.

#### Шаг 2: Вычисление эмбеддингов
2.1. Вычислить эмбеддинг полного вопроса (нижний регистр):
- `question_embedding = await asyncio.to_thread(emb_client.get_text_embedding, question_lower)`.
- `emb_client` — глобальный экземпляр `EmbeddingClient` (из `app.src.qwen3_emb_client`), инициализированный при запуске сервиса с `base_url=EMBEDDING_BASE_URL` и `timeout=EMBEDDING_TIMEOUT`.
- Выход: `List[float]` из 2048 значений.
- Если `emb_client.get_text_embedding` выбросил исключение (`requests.HTTPError`, `requests.Timeout`, `ValueError`) → `HTTPException(500, "Failed to compute embedding for question")`.

2.2. Для каждой сущности из `entities_from_question` (списка словарей с `title`, `type`) вычислить эмбеддинг:
- Сформировать текст для эмбеддинга: `f"{entity_title} ({entity_type})"`.
- Вызвать `await asyncio.to_thread(emb_client.get_text_embedding, entity_text)`.
- Все вызовы выполняются параллельно через `asyncio.gather()`.
- Результат для каждой сущности: словарь `{"title": ..., "type": ..., "embedding": List[float]}`.
- Если вычисление эмбеддинга для конкретной сущности упало → эта сущность исключается из дальнейшей обработки (логируется warning с `title`, `type` и причиной ошибки). Остальные продолжают обработку.
- После фильтрации получаем `valid_entities` — список словарей с ключами `title`, `type`, `embedding`.

#### Шаг 3: Распределение долей токенов между сущностями
3.1. Если `len(valid_entities) == 0`:
- Бюджет сущностей присоединяется к бюджету сообществ: `community_budget += entity_budget`, `entity_budget = 0`.
- Установить флаг `fallback_used = True`.
- Переход к шагу 6 (fallback-путь).

3.2. Если `len(valid_entities) > 0` (N сущностей):
- `fallback_used = False`.
- Бюджет каждой сущности для entity-пула:
  ```
  per_entity_entity_budget_base = entity_budget // N
  per_entity_entity_remainder  = entity_budget % N
  # Первые remainder сущностей получают +1 токен
  ```
- Бюджет каждой сущности для community-пула:
  ```
  per_entity_community_budget_base = community_budget // N
  per_entity_community_remainder  = community_budget % N
  ```
- Распределить remainder по одной единице первым K сущностям.

#### Шаг 4: Поиск текстовых блоков из коллекции `documents` (Qdrant)
4.1. Сформировать Qdrant search-запрос к коллекции `documents` через `aiohttp` (Constitution P5 — асинхронный I/O):
- URL: `POST {QDRANT_URL}/collections/documents/points/search`
- Тело запроса:
  ```json
  {
    "vector": <question_embedding>,
    "limit": 100,
    "with_payload": true,
    "with_vector": false,
    "filter": <filter object or null>
  }
  ```
- `filter`: если `documents_filter == "text_only"` → `{"must": [{"key": "element_type", "match": {"value": "text"}}]}`. Если `"all"` → фильтр не включается в тело запроса.
- `limit: 100` — достаточно большое значение, чтобы заведомо хватило для заполнения бюджета; блоки будут отбираться по токен-бюджету, а не по количеству.
- Если Qdrant недоступен (connection error, HTTP 4xx/5xx) → `HTTPException(500, "Qdrant documents collection unavailable")`.

4.2. Обработать результаты поиска:
- Qdrant возвращает `{"result": [{"id": ..., "score": ..., "payload": {...}}, ...]}`, отсортированные по убыванию `score` (cosine similarity).
- `remaining_text_budget = text_budget`.
- `text_pool = []`.
- Для каждой точки `result` в порядке убывания `score`:
  - Извлечь текстовое содержимое: `text = point.payload.original_element.text` (основной источник текста — поле `text` внутри `original_element`).
  - Если `text` is None или пустая строка → пропустить точку.
  - Посчитать количество токенов: `token_count = await llm.count_tokens(text, MODEL_NAME)`.
    - `llm` — экземпляр `AsyncLLMClient` (создан на шаге 1.2).
    - Если tokenizer недоступен → используется fallback `len(text) // 4` (встроен в `AsyncLLMClient.count_tokens`).
  - Если `token_count <= remaining_text_budget`:
    - Добавить `text` в `text_pool`.
    - `remaining_text_budget -= token_count`.
  - Иначе: прекратить обход (break).
  - Если `remaining_text_budget <= 0` → прекратить обход (break).
- Сохранить `text_tokens_used = text_budget - remaining_text_budget`.

#### Шаг 5: Поиск по графу (Entity + Community) для каждой валидной сущности

Для каждой сущности `i` из `valid_entities` (индекс `0..N-1`) выполнить шаги 5.1–5.5 **параллельно** (через `asyncio.gather`). Каждая сущность обрабатывается независимо.

Инициализация для сущности `i`:
- `per_entity_entity_budget = per_entity_entity_budget_base + (1 if i < per_entity_entity_remainder else 0)`
- `per_entity_community_budget = per_entity_community_budget_base + (1 if i < per_entity_community_remainder else 0)`
- `entity_pool_i = []`
- `community_pool_i = []`
- `entity_entity_budget_remaining = per_entity_entity_budget`
- `entity_community_budget_remaining = per_entity_community_budget`

**5.1. Поиск точки входа в Qdrant коллекции `entity_embeddings`:**
- Сформировать Qdrant search-запрос через `aiohttp`:
  ```
  POST {QDRANT_URL}/collections/entity_embeddings/points/search
  Body: {
    "vector": <entity.embedding>,
    "limit": 1,
    "with_payload": true,
    "with_vector": false
  }
  ```
- Если Qdrant недоступен → `HTTPException(500, "Qdrant entity_embeddings collection unavailable")`.
- Из результата `result[0]` (если массив не пуст) извлечь:
  - `score` (float)
  - `payload.entity_title` (str)
  - `payload.entity_type` (str)
- Если `result` пуст (0 точек) → `entities_search_miss_count += 1`. Бюджет entity-пула присоединяется к community-пулу для этой сущности: `entity_community_budget_remaining += entity_entity_budget_remaining`, `entity_entity_budget_remaining = 0`. Переход к шагу 5.5 (поиск Community через fallback для этой сущности — шаг 6, но в контексте бюджета именно этой сущности).

**5.2. Получение ноды входа из Neo4j:**
- Выполнить Cypher-запрос через `doc_manager` (глобальный экземпляр `Manager`, инициализированный при запуске сервиса):
  ```cypher
  MATCH (e:Entity {title: $title, type: $type})
  RETURN e.title AS title, e.type AS type, e.description AS description
  ```
  Параметры: `title = entity_title`, `type = entity_type`.
- Если Entity не найдена в Neo4j (пустой результат) → `entities_miss_in_neo4j += 1`. Эмбеддинги в Qdrant могли устареть. Переход к шагу 5.5.
- Если Neo4j недоступен → `HTTPException(500, "Neo4j connection error")`.

**5.3. Добавление ноды входа в entity-пул:**
- Сформировать текст: `f"[{entry.title}] ({entry.type}): {entry.description}"`.
- Если `entry.description` is None → `"No description"`.
- Посчитать токены: `token_count = await llm.count_tokens(entry_text, MODEL_NAME)`.
- Если `token_count <= entity_entity_budget_remaining`:
  - Добавить `entry_text` в `entity_pool_i`.
  - `entity_entity_budget_remaining -= token_count`.

**5.4. Поиск и добавление связанных сущностей (RELATED):**
- Выполнить Cypher-запрос:
  ```cypher
  MATCH (entry:Entity {title: $title, type: $type})-[r:RELATED]-(related:Entity)
  RETURN related.title AS title, related.type AS type,
         related.description AS description,
         r.description AS rel_description
  ```
  Примечание: связь ненаправленная `-[r:RELATED]-` (в обе стороны от entry).
- Параметры: `title = entity_title`, `type = entity_type`.
- Если результат пуст → `related_entities = []`, переход к шагу 5.5.

- Для каждой связанной пары (`title`, `type`, `description`, `rel_description`) из результата:
  - Сформировать текст для эмбеддинга: `rel_description if rel_description else ""`.
  - Вызвать `await asyncio.to_thread(emb_client.get_text_embedding, rel_text)`.
  - Все вызовы — параллельно через `asyncio.gather`.
  - Если эмбеддинг для какой-то связи не удалось вычислить → для этой связи `score = 0.0`, сущность не исключается.

- Для каждой связанной сущности, для которой эмбеддинг вычислен:
  - Вычислить cosine similarity: `score = cosine_similarity(rel_embedding, question_embedding)`.
  - Функция `cosine_similarity(a, b)` определена локально:
    ```python
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    ```
- Отсортировать `related_entities` по убыванию `score`. Сущности с `score = 0.0` (fallback при ошибке эмбеддинга) оказываются в конце.

- Для каждой связанной сущности в порядке убывания score:
  - Сформировать текст: `f"[{title}] ({type}): {description} | Relation: {rel_description}"`.
  - Если `description` is None → `"No description"`.
  - Посчитать токены: `token_count = await llm.count_tokens(rel_text, MODEL_NAME)`.
  - Если `token_count <= entity_entity_budget_remaining`:
    - Добавить `rel_text` в `entity_pool_i`.
    - `entity_entity_budget_remaining -= token_count`.
  - Иначе: пропустить.
  - Если `entity_entity_budget_remaining <= 0` → прекратить обход (break).

- Собрать множество всех Entity, участвовавших в обработке:
  ```
  entity_set_i = {(entity_title, entity_type)}
  for each related entity in processed:
      entity_set_i.add((title, type))
  ```

**5.5. Поиск сообществ (Community) для сущности:**
- Найти все нижнеуровневые (leaf) Community, содержащие хотя бы одну Entity из `entity_set_i`:
  - Cypher-запрос:
    ```cypher
    MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)
    WHERE e.title IN $titles AND e.type IN $types
    RETURN DISTINCT c.id AS id, c.title AS title, c.summary AS summary,
           c.level AS level
    ```
    Параметры:
    - `titles = [t for t, _ in entity_set_i]`
    - `types = [tp for _, tp in entity_set_i]`
  - Набор результатов фильтруется на стороне Python:
    - Community является «нижнеуровневым» (leaf), если существует хотя бы одна связь `IS_CHILD_OF` от этого Community к родителю, и **не** существует связей `IS_PARENT_OF` от этого Community к дочерним. Это сообщества, которые являются листьями в иерархии (имеют родителя, но не имеют детей).
    - Cypher для проверки leaf (можно выполнить одним запросом через `OPTIONAL MATCH`):
      ```cypher
      MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)
      WHERE e.title IN $titles AND e.type IN $types
      OPTIONAL MATCH (c)-[:IS_PARENT_OF]->(child:Community)
      OPTIONAL MATCH (c)-[:IS_CHILD_OF]->(parent:Community)
      WITH c, collect(DISTINCT child) AS children, collect(DISTINCT parent) AS parents
      WHERE size(children) = 0 AND size(parents) > 0
      RETURN DISTINCT c.id AS id, c.title AS title, c.summary AS summary, c.level AS level
      ```
    - Если Cypher-движок не поддерживает агрегацию в WHERE подобным образом, используется альтернативный подход:
      1. Выполнить запрос без фильтрации на leaf.
      2. Для каждого Community выполнить проверку отдельными запросами:
         - `MATCH (c:Community {id: $id})-[:IS_PARENT_OF]->(:Community) RETURN count(*) > 0 AS has_children`
         - `MATCH (c:Community {id: $id})-[:IS_CHILD_OF]->(:Community) RETURN count(*) > 0 AS has_parent`
      3. Оставить только те, где `has_children == False AND has_parent == True`.
  - Если сообществ не найдено → `community_pool_i` остаётся пустым, сохранить `community_space_left_i = entity_community_budget_remaining`.

- Для каждого найденного leaf Community подсчитать `count_ent` — количество Entity из `entity_set_i`, которые входят в это Community:
  - Cypher-запрос:
    ```cypher
    MATCH (c:Community {id: $community_id})-[:CONSISTS_OF]->(e:Entity)
    WHERE e.title IN $titles AND e.type IN $types
    RETURN count(e) AS count_ent
    ```
  - Параметры: `community_id`, `titles`, `types`.

- Отсортировать найденные Community по убыванию `count_ent`.

- Для каждого Community в порядке убывания `count_ent`:
  - Сформировать текст: `f"[{community.title}]: {community.summary}"`.
  - Если `summary` is None → `"No summary"`.
  - Посчитать токены: `token_count = await llm.count_tokens(comm_text, MODEL_NAME)`.
  - Если `token_count <= entity_community_budget_remaining`:
    - Добавить `comm_text` в `community_pool_i`.
    - `entity_community_budget_remaining -= token_count`.
  - Иначе: пропустить.
  - Если `entity_community_budget_remaining <= 0` → прекратить обход (break).

- Сохранить статистику для сущности `i`:
  - `entity_space_left_i = entity_entity_budget_remaining`
  - `community_space_left_i = entity_community_budget_remaining`
  - `entities_found_i = len(entity_pool_i)`
  - `communities_found_i = len(community_pool_i)`

#### Шаг 6: Fallback-путь (если сущности не найдены ИЛИ для сущностей без матча в Qdrant `entity_embeddings`)

Этот шаг выполняется в двух случаях:
- **Глобальный fallback**: `len(valid_entities) == 0` (шаг 3.1).
- **Per-entity fallback**: для конкретной сущности `i`, если Qdrant `entity_embeddings` не вернул результат (шаг 5.1).

В случае per-entity fallback обработка ведётся в контексте бюджета этой сущности (`entity_community_budget_remaining`).

В случае глобального fallback используется `community_budget` (увеличенный на `entity_budget`).

Ниже описан алгоритм для случая с бюджетом `budget = entity_community_budget_remaining` (per-entity) или `budget = community_budget` (глобальный).

**6.1. Поиск ближайшего Community уровня 0 (корневое сообщество):**
- Сформировать Qdrant search-запрос через `aiohttp`:
  ```
  POST {QDRANT_URL}/collections/community_embeddings/points/search
  Body: {
    "vector": <question_embedding>,
    "limit": 1,
    "with_payload": true,
    "with_vector": false,
    "filter": {
      "must": [{"key": "level", "match": {"value": 0}}]
    }
  }
  ```
- Если Qdrant недоступен → `HTTPException(500, "Qdrant community_embeddings collection unavailable")`.
- Из результата `result[0]` (если не пуст) извлечь `payload.community_id`.
- Если результат пуст → `community_qdrant_miss_count += 1`. Fallback не дал результатов — пул остаётся пустым.

**6.2. Получение Community из Neo4j:**
- Cypher-запрос:
  ```cypher
  MATCH (c:Community {id: $community_id})
  RETURN c.title AS title, c.summary AS summary
  ```
  Параметр: `community_id` (строка UUID).
- Если Community не найден → `community_miss_count += 1`. Пул остаётся пустым.
- Если Neo4j недоступен → `HTTPException(500, "Neo4j connection error")`.

**6.3. Добавление корневого Community в пул:**
- Сформировать текст: `f"[{community.title}]: {community.summary}"`.
- Если `summary` is None → `"No summary"`.
- Посчитать токены: `token_count = await llm.count_tokens(comm_text, MODEL_NAME)`.
- Если `token_count <= budget`:
  - Добавить `comm_text` в `fallback_community_pool`.
  - `budget -= token_count`.
- Иначе: корневое сообщество не добавляется, но дочерние всё равно ищутся.

**6.4. Поиск дочерних Community:**
- Cypher-запрос:
  ```cypher
  MATCH (root:Community {id: $community_id})-[:IS_PARENT_OF]->(child:Community)
  RETURN child.id AS id, child.title AS title, child.summary AS summary, child.level AS level
  ```
  Параметр: `community_id`.

**6.5. Сортировка дочерних Community по релевантности вопросу:**
- Для каждого дочернего Community получить его эмбеддинг из Qdrant:
  - ID точки в Qdrant: `uuid5(COMMUNITY_EMBEDDINGS_NAMESPACE, str(child.id))`.
  - Использовать Qdrant Point API: `GET {QDRANT_URL}/collections/community_embeddings/points/{point_id}` с параметром `?with_vector=true&with_payload=false`.
  - Все запросы выполняются параллельно через `aiohttp` + `asyncio.gather`.
  - Если точка не найдена (HTTP 404) → `score = 0.0` для этого дочернего Community.
  - Если запрос упал → `score = 0.0`.

- Для каждого дочернего Community, для которого получен эмбеддинг:
  - `score = cosine_similarity(child_embedding, question_embedding)`.
  - Если эмбеддинг не найден → `score = 0.0`.

- Отсортировать дочерние Community по убыванию `score`.

**6.6. Добавление дочерних Community в пул:**
- Для каждого дочернего Community в порядке убывания `score`:
  - Сформировать текст: `f"[{child.title}]: {child.summary}"`.
  - Если `summary` is None → `"No summary"`.
  - Посчитать токены: `token_count = await llm.count_tokens(child_text, MODEL_NAME)`.
  - Если `token_count <= budget`:
    - Добавить `child_text` в `fallback_community_pool`.
    - `budget -= token_count`.
  - Иначе: пропустить.
  - Если `budget <= 0` → прекратить обход.

- Сохранить `fallback_community_space_left = budget`.
- Для per-entity fallback — сохранить `community_pool_i = fallback_community_pool`.

#### Шаг 7: Объединение пулов

**7.1. Объединение entity-пулов (если сущности были найдены):**
- Round-robin слияние всех `entity_pool_i` (для `i` от 0 до N-1, исключая сущности с miss):
  ```python
  merged_entity_pool = []
  max_len = max((len(pool) for pool in entity_pools), default=0)
  for round_idx in range(max_len):
      for pool in entity_pools:
          if round_idx < len(pool):
              merged_entity_pool.append(pool[round_idx])
  ```
- Где `entity_pools` — список `entity_pool_i` для каждой из N валидных сущностей (в порядке их обнаружения на шаге 2.2).
- Для сущностей, у которых был miss в Qdrant (`entities_search_miss_count > 0`), их entity-пул пуст — они не влияют на round-robin.

**7.2. Объединение community-пулов (если сущности были найдены):**
- Аналогичный round-robin для `community_pools` → `merged_community_pool`.
- `community_pools` включает пулы от всех валидных сущностей (включая те, для которых сработал per-entity fallback — их `community_pool_i` уже заполнен через шаг 6).

**7.3. Если срабатывал глобальный fallback (шаг 3.1, `len(valid_entities) == 0`):**
- `merged_entity_pool = []`.
- `merged_community_pool = fallback_community_pool` (результат шага 6.6).
- `entity_tokens_used = 0`.
- `community_tokens_used = community_budget - fallback_community_space_left`.

**7.4. Итоговый подсчёт использованных токенов:**
- `total_entity_tokens_used` = сумма токенов всех элементов в `merged_entity_pool`.
- `total_community_tokens_used` = сумма токенов всех элементов в `merged_community_pool`.
- `total_text_tokens_used` = `text_budget - remaining_text_budget` (с шага 4.2).

#### Шаг 8: Формирование ответа

- Вычислить `processing_time_ms = int((time.time() - start_time) * 1000)`.
- Сформировать словарь ответа (Pydantic-модель `SearchResponse`, см. ниже).
- Вернуть HTTP 200.

### Output
- **HTTP 200** — поиск выполнен успешно (даже если все пулы пусты):

```json
{
  "text_units": [
    "<текстовый блок 1>",
    "<текстовый блок 2>"
  ],
  "entities": [
    "[ORG_NAME] (ORGANIZATION): Description text | Relation: related via partnership",
    "[PERSON_NAME] (PERSON): Description text | Relation: works at"
  ],
  "communities": [
    "[Community 3]: Summary of community findings",
    "[Community 7]: Another community summary"
  ],
  "statistics": {
    "processing_time_ms": 2450,
    "tokens_used": {
      "text_units": 1800,
      "entities": 920,
      "communities": 880
    },
    "tokens_remaining": {
      "text_units": 200,
      "entities": 80,
      "communities": 120
    },
    "entities_extracted_from_question": 3,
    "entities_matched_in_graph": 2,
    "entities_search_misses": 1,
    "fallback_used": false,
    "total_items": {
      "text_units": 5,
      "entities": 4,
      "communities": 3
    }
  }
}
```

**Поля ответа**:

| Поле | Тип | Описание |
|------|-----|----------|
| `text_units` | `List[str]` | Текстовые блоки из коллекции `documents`, отсортированные по убыванию cosine similarity с эмбеддингом вопроса |
| `entities` | `List[str]` | Объединённые (round-robin) текстовые описания сущностей и их связей из графа знаний |
| `communities` | `List[str]` | Объединённые (round-robin) текстовые описания сообществ (community reports) |
| `statistics.processing_time_ms` | `int` | Время обработки запроса в миллисекундах |
| `statistics.tokens_used` | `Dict[str, int]` | Фактически использовано токенов для каждого типа (`text_units`, `entities`, `communities`) |
| `statistics.tokens_remaining` | `Dict[str, int]` | Осталось неиспользованных токенов для каждого типа |
| `statistics.entities_extracted_from_question` | `int` | Количество сущностей, извлечённых LLM из вопроса (шаг 1.2) |
| `statistics.entities_matched_in_graph` | `int` | Количество сущностей, для которых найден матч в графе (Qdrant + Neo4j) — `len(valid_entities) - entities_search_misses` |
| `statistics.entities_search_misses` | `int` | Количество сущностей, для которых не найден матч в Qdrant `entity_embeddings` (шаг 5.1) |
| `statistics.fallback_used` | `bool` | Был ли задействован глобальный fallback-путь (community уровня 0) из-за отсутствия извлечённых сущностей |
| `statistics.total_items` | `Dict[str, int]` | Общее количество элементов, включённых в каждый пул |

## API Contract

```openapi
POST /search
summary: Семантический поиск по графу знаний — релевантные текстовые блоки, сущности со связями и сообщества для заданного вопроса
requestBody:
  required: true
  content:
    application/json:
      schema:
        type: object
        required: [question, max_tokens]
        properties:
          question:
            type: string
            minLength: 1
            description: Строка вопроса пользователя
          max_tokens:
            type: integer
            exclusiveMinimum: 0
            description: Максимальный суммарный размер вывода в токенах
          proportions:
            type: object
            description: Доли токенов для text_units, entities, communities (сумма = 1.0). По умолчанию {text_units: 0.5, entities: 0.25, communities: 0.25}
            properties:
              text_units:
                type: number
                minimum: 0
                default: 0.5
              entities:
                type: number
                minimum: 0
                default: 0.25
              communities:
                type: number
                minimum: 0
                default: 0.25
          documents_filter:
            type: string
            enum: [text_only, all]
            default: text_only
            description: Режим фильтрации коллекции documents. text_only — только element_type == text. all — без фильтра.
responses:
  '200':
    description: Поиск выполнен успешно
    content:
      application/json:
        schema:
          type: object
          properties:
            text_units:
              type: array
              items:
                type: string
              description: Текстовые блоки из коллекции documents
            entities:
              type: array
              items:
                type: string
              description: Описания сущностей и их связей
            communities:
              type: array
              items:
                type: string
              description: Описания сообществ (community summaries)
            statistics:
              type: object
              properties:
                processing_time_ms:
                  type: integer
                tokens_used:
                  type: object
                  properties:
                    text_units:
                      type: integer
                    entities:
                      type: integer
                    communities:
                      type: integer
                tokens_remaining:
                  type: object
                  properties:
                    text_units:
                      type: integer
                    entities:
                      type: integer
                    communities:
                      type: integer
                entities_extracted_from_question:
                  type: integer
                entities_matched_in_graph:
                  type: integer
                entities_search_misses:
                  type: integer
                fallback_used:
                  type: boolean
                total_items:
                  type: object
                  properties:
                    text_units:
                      type: integer
                    entities:
                      type: integer
                    communities:
                      type: integer
  '422':
    description: Ошибка валидации входных параметров
    content:
      application/json:
        schema:
          type: object
          properties:
            detail:
              type: string
  '500':
    description: Ошибка инфраструктуры (Neo4j, Qdrant, Embedding API недоступны)
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
  │ POST /search {"question": "...", "max_tokens": 4000, ...}
  ▼
semantic_index.py — search_graph()
  │
  ├─[Шаг 1] LLM: извлечение сущностей из вопроса
  │     ├─ AsyncLLMClient (aiohttp к TOKENIZER_URL, AsyncOpenAI к LLM_URL)
  │     ├─ AsyncGraphExtractor.extract() с GRAPH_EXTRACTION_PROMPT
  │     └─ Результат: список сущностей [{title, type, description}, ...] или []
  │
  ├─[Шаг 2] EmbeddingClient: эмбеддинг вопроса + эмбеддинги сущностей (parallel)
  │     ├─ asyncio.to_thread(emb_client.get_text_embedding, question_lower)
  │     │   └─ POST {EMBEDDING_BASE_URL}/embed → question_embedding[2048]
  │     └─ asyncio.gather(... asyncio.to_thread(emb_client.get_text_embedding, f"{title} ({type})") ...)
  │         └─ POST {EMBEDDING_BASE_URL}/embed → сущность.embedding[2048]
  │
  ├─[Шаг 4] Qdrant: поиск текстовых блоков (documents)
  │     ├─ POST {QDRANT_URL}/collections/documents/points/search
  │     │   vector: question_embedding, limit: 100, filter: element_type == "text" (опц.)
  │     ├─ Tokenizer (AsyncLLMClient.count_tokens): подсчёт токенов для каждого блока
  │     │   └─ POST {TOKENIZER_URL}/tokenize (fallback: len(text)//4)
  │     └─ text_pool[] (по text_budget)
  │
  ├─[Шаг 5] Для каждой сущности (parallel via asyncio.gather):
  │   │
  │   ├─[5.1] Qdrant: поиск точки входа в entity_embeddings (limit=1)
  │   │   └─ POST {QDRANT_URL}/collections/entity_embeddings/points/search
  │   │
  │   ├─[5.2] Neo4j: MATCH (e:Entity {title, type}) → title, type, description
  │   │
  │   ├─[5.3] Tokenizer: подсчёт токенов → добавление в entity_pool_i
  │   │
  │   ├─[5.4] Neo4j: MATCH (entry)-[r:RELATED]-(related) → related entities
  │   │   ├─ EmbeddingClient (parallel): эмбеддинги rel_description
  │   │   │   └─ POST {EMBEDDING_BASE_URL}/embed
  │   │   ├─ Cosine similarity (question_embedding × rel_embedding)
  │   │   ├─ Сортировка по убыванию score
  │   │   └─ Tokenizer: подсчёт токенов → добавление в entity_pool_i (по budget)
  │   │
  │   └─[5.5] Neo4j: поиск leaf Community через CONSISTS_OF
  │         ├─ Community с условиями: имеет IS_CHILD_OF, не имеет IS_PARENT_OF
  │         ├─ Neo4j: COUNT entities per community (count_ent)
  │         ├─ Сортировка по убыванию count_ent
  │         └─ Tokenizer: подсчёт токенов → добавление в community_pool_i (по budget)
  │
  ├─[Шаг 6] Fallback (если сущности не извлечены или miss в Qdrant):
  │   ├─ Qdrant: поиск community_embeddings (level=0, limit=1)
  │   │   └─ POST {QDRANT_URL}/collections/community_embeddings/points/search
  │   ├─ Neo4j: MATCH (c:Community {id}) → title, summary
  │   ├─ Neo4j: MATCH (root)-[:IS_PARENT_OF]->(child)
  │   ├─ Qdrant (parallel): GET /collections/community_embeddings/points/{point_id}?with_vector=true
  │   ├─ Cosine similarity (question_embedding × child_embedding)
  │   ├─ Сортировка по убыванию score
  │   └─ Tokenizer: подсчёт токенов → fallback_community_pool[] (по budget)
  │
  ├─[Шаг 7] Объединение:
  │   ├─ Round-robin merge всех entity_pool_i → merged_entity_pool[]
  │   └─ Round-robin merge всех community_pool_i → merged_community_pool[]
  │
  └─[Шаг 8] Ответ: SearchResponse {text_units, entities, communities, statistics}
```

## LLM Interactions

| Промпт | Файл (должен быть) | Использование |
|--------|-------------------|---------------|
| `GRAPH_EXTRACTION_PROMPT` | `prompts/graph-extraction.md` | Извлечение сущностей из вопроса пользователя (тот же промпт, что в process-document). Переменные: `{entity_types}`, `{input_text}`. Выход: структурированный текст с `("entity"<|>NAME<|>TYPE<|>desc)`, разделённый `##`, завершается `<|COMPLETE|>`. |

**Примечание**: в поиске используется только извлечение сущностей (`entity`), связи (`relationship`) из ответа LLM игнорируются — важен только список найденных сущностей для поиска точек входа в граф. `max_gleanings=0` — без циклов дозапроса, только один вызов LLM.

**Модель для query extraction**: LLM-модель для извлечения сущностей из пользовательского вопроса конфигурируется **отдельно** от основной модели, используемой для наполнения базы знаний (`process-document`). Это позволяет использовать более лёгкую/быструю модель для инференса на лету или мультимодальную (VL) модель для случаев, когда вопрос сопровождается изображением. Конфигурация задаётся отдельными константами в `config.py`:

## LLM Model Requirements

### Модель для query extraction (извлечение сущностей из вопроса)

- **Тип модели**: text-only или VL (Vision-Language, мультимодальная). Архитектура эндпоинта поддерживает оба типа — выбор определяется значением `QUERY_EXTRACTION_MODEL_NAME` в конфигурации.
- **Минимальный размер контекста**: 4096 токенов (вопрос пользователя + промпт extraction — обычно суммарно < 2000 токенов для типичного вопроса). Для VL-моделей контекст измеряется в токенах суммарно для текста и изображения.
- **Язык выхода**: английский (сущности извлекаются на английском согласно `GRAPH_EXTRACTION_PROMPT`)
- **Требования к формату выхода**: свободный текст со строгим синтаксисом кортежей: `("entity"<|>NAME<|>TYPE<|>desc)`, разделённых `##`, завершающихся `<|COMPLETE|>`. Связи (`relationship`) в ответе игнорируются.
- **Конкретная модель**: задаётся через `QUERY_EXTRACTION_MODEL_NAME` в `config.py`. По умолчанию (если не задана) используется `MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'`.
- **API**: OpenAI-совместимый эндпоинт (`QUERY_EXTRACTION_LLM_URL`; fallback: `LLM_URL = 'http://localhost:9886/v1'`)
- **API-ключ**: `QUERY_EXTRACTION_API_KEY` (fallback: `LLM_API_KEY`)
- **Tokenizer**: отдельный HTTP-эндпоинт (`QUERY_EXTRACTION_TOKENIZER_URL`; fallback: `TOKENIZER_URL = 'http://localhost:9886/tokenize'`). Для VL-моделей токенизатор должен поддерживать multimodal-подсчёт токенов (текст + изображение); если query extraction модель — text-only, используется стандартный текстовый токенизатор.

### Примечание по VL-моделям
Если `QUERY_EXTRACTION_MODEL_NAME` указывает на VL-модель, текущая версия эндпоинта передаёт модели только текст вопроса (изображение не прикрепляется). Это обеспечивает совместимость архитектуры для будущего расширения, когда вопрос сможет сопровождаться скриншотом/изображением. VL-модель в text-only режиме должна корректно обрабатывать запросы без прикреплённого изображения.


**Текущее состояние (нарушение P7)**: промпт `GRAPH_EXTRACTION_PROMPT` определён как строковая константа в `config.py`. В соответствии с Constitution P7 должен быть вынесен в `semantic_graph/prompts/graph-extraction.md`.

## Error Handling

| Сценарий | Код | Поведение |
|----------|-----|-----------|
| Qdrant недоступен (коллекция `documents`) | 500 | `"Qdrant documents collection unavailable"`. Логирование ошибки. |
| Qdrant недоступен (коллекция `entity_embeddings`) | 500 | `"Qdrant entity_embeddings collection unavailable"`. Логирование ошибки. |
| Qdrant недоступен (коллекция `community_embeddings`) | 500 | `"Qdrant community_embeddings collection unavailable"`. Логирование ошибки. |
| Neo4j недоступен (любой запрос) | 500 | `"Neo4j connection error"`. Логирование ошибки. |
| Embedding API недоступен для вопроса | 500 | `"Failed to compute embedding for question"`. Логирование ошибки. |
| LLM API недоступен при entity extraction | 200 | Сущности не извлечены, `entities_extracted_from_question: 0`. Срабатывает глобальный fallback. Warning в логах. |
| LLM вернул пустой ответ или невалидный формат | 200 | `entities_from_question = []`, `entities_extracted_from_question: 0`. Срабатывает глобальный fallback. Warning в логах. |
| Tokenizer недоступен | — | Используется fallback: `len(text) // 4` (встроен в `AsyncLLMClient.count_tokens`). Warning в логах. |
| Пустой результат поиска в `documents` (`result: []`) | 200 | `text_units: []`, `tokens_used.text_units: 0`. |
| Сущность извлечена, но не найдена в Qdrant `entity_embeddings` | 200 | `entities_search_misses += 1`. Бюджет entity-пула этой сущности передаётся в community-пул. Для этой сущности срабатывает per-entity fallback (шаг 6). |
| Сущность найдена в Qdrant, но не найдена в Neo4j | 200 | Пропуск сущности. `entities_miss_in_neo4j` (внутренний счётчик). Бюджет передаётся в community-пул. |
| Ошибка вычисления эмбеддинга отдельной сущности (Embedding API) | 200 | Сущность исключается из `valid_entities`. Остальные продолжают обработку. Warning в логах с `title`, `type`, причиной. |
| Ошибка вычисления эмбеддинга связи `RELATED` | 200 | Для этой связи `score = 0.0`, связанная сущность помещается в конец сортировки. Warning в логах. |
| Community не найден в Neo4j (гонка данных: удалён между Qdrant и Neo4j запросами) | 200 | Пропуск community, пул не пополняется. `community_miss_count += 1` (внутренний счётчик). |
| Дочерний Community без эмбеддинга в Qdrant (точка не найдена) | 200 | `score = 0.0` для этого community, помещается в конец сортировки. |
| Пустой результат поиска community в Qdrant (fallback, шаг 6.1) | 200 | `community_qdrant_miss_count += 1` (внутренний счётчик). `merged_community_pool` остаётся пустым. |
| Все пулы пусты (ничего не найдено) | 200 | Все списки пустые, `statistics.total_items` = все нули, `statistics.fallback_used: true`. |
| `proportions` не в сумме дают 1.0 | 422 | `"proportions must sum to 1.0"` |
| `proportions` содержит отрицательные значения | 422 | `"proportions values must be non-negative"` |
| `documents_filter` не `"text_only"` или `"all"` | 422 | Стандартная ошибка валидации Pydantic (pattern mismatch). |
| `max_tokens` ≤ 0 | 422 | Стандартная ошибка валидации Pydantic (`gt=0`). |
| `question` пустая строка | 422 | Стандартная ошибка валидации Pydantic (`min_length=1`). |

## Testing

### Тест-кейсы
1. **Успешный полный поиск**: вопрос с 3 сущностями (ORGANIZATION, PERSON, GEO), все компоненты доступны → HTTP 200, непустые `text_units`, `entities`, `communities`. `entities_extracted_from_question: 3`, `entities_matched_in_graph: 3`, `entities_search_misses: 0`.
2. **Поиск без сущностей в вопросе**: вопрос без организаций/персон/geo/events (например, "what is the weather?") → LLM не извлекает сущностей → HTTP 200, `entities_extracted_from_question: 0`, `fallback_used: true`, `entities: []`, непустые `communities` (через fallback).
3. **Сущности извлечены, но не найдены в Qdrant `entity_embeddings`**: мок LLM возвращает 2 сущности, мок Qdrant `entity_embeddings` возвращает пустой результат → HTTP 200, `entities_extracted_from_question: 2`, `entities_matched_in_graph: 0`, `entities_search_misses: 2`, `entities: []`, `fallback_used: true`.
4. **Частичный miss**: 3 сущности извлечены, 2 найдены в Qdrant, 1 — miss → HTTP 200, `entities_extracted_from_question: 3`, `entities_matched_in_graph: 2`, `entities_search_misses: 1`. Для miss-сущности срабатывает per-entity fallback.
5. **Проверка round-robin объединения**: 2 сущности, entity_pool_0 = ["A1", "A2"], entity_pool_1 = ["B1"] → `merged_entity_pool = ["A1", "B1", "A2"]`.
6. **Проверка round-robin с 3 сущностями разной длины**: pool_0 = ["A1", "A2", "A3"], pool_1 = ["B1", "B2"], pool_2 = ["C1"] → `merged = ["A1", "B1", "C1", "A2", "B2", "A3"]`.
7. **Проверка токен-бюджетирования**: `max_tokens=500` с пропорциями по умолчанию → `text_budget=250`, `entity_budget=125`, `community_budget=125`. Суммарно использовано ≤ 500 токенов. Каждый пул не превышает свой бюджет.
8. **Проверка пропорций**: `proportions = {"text_units": 0.7, "entities": 0.2, "communities": 0.1}` при `max_tokens=1000` → `text_budget=700`, `entity_budget=200`, `community_budget=100`.
9. **Проверка `documents_filter: "all"`**: в Qdrant search-запросе отсутствует фильтр по `element_type` → возвращаются блоки всех типов.
10. **Проверка `documents_filter: "text_only"`** (default): фильтр `{"must": [{"key": "element_type", "match": {"value": "text"}}]}` активен.
11. **Частичная недоступность Tokenizer**: мок `AsyncLLMClient.count_tokens` выбрасывает исключение → используется fallback `len(text) // 4`. Warning в логах.
12. **Пустой граф Neo4j**: 0 Entity, 0 Community → HTTP 200, `entities: []`, `communities: []`, `text_units` может быть непустым (если документы есть).
13. **Невалидные proportions**: сумма = 0.5 → HTTP 422, `"proportions must sum to 1.0"`.
14. **Невалидные proportions**: отрицательное значение → HTTP 422, `"proportions values must be non-negative"`.
15. **Невалидный documents_filter**: `"invalid"` → HTTP 422.
16. **Qdrant коллекция `entity_embeddings` не существует**: → HTTP 500 с сообщением `"Qdrant entity_embeddings collection unavailable"`.
17. **Дочерние Community без эмбеддингов в Qdrant**: у 3 дочерних community нет точек в Qdrant → `score = 0.0`, все три идут в конец сортировки по убыванию score.
- **LLM API** (query extraction): `AsyncLLMClient` + `AsyncGraphExtractor` для извлечения сущностей из вопроса:
  - Конфигурация: `QUERY_EXTRACTION_MODEL_NAME`, `QUERY_EXTRACTION_LLM_URL`, `QUERY_EXTRACTION_API_KEY`, `QUERY_EXTRACTION_TOKENIZER_URL` (из `config.py`; fallback на `MODEL_NAME`, `LLM_URL`, `LLM_API_KEY`, `TOKENIZER_URL` соответственно)
  - `AsyncOpenAI` клиент к `extraction_llm_url`
  - Промпт: `GRAPH_EXTRACTION_PROMPT` (из `config.py`, нарушение P7)
  - Поддерживаются text-only и VL (Vision-Language) модели; в текущей версии изображение не передаётся — модель получает только текст вопроса

21. **Связь `RELATED` без описания**: `rel_description IS NULL` → для эмбеддинга используется пустая строка `""`.
22. **Entity без описания**: `entry.description IS NULL` → текст `"[TITLE] (TYPE): No description"`.

### Подход к тестированию
- **Unit**: `cosine_similarity()` на фиксированных векторах; расчёт бюджетов токенов и распределение remainder; round-robin merge на фиктивных списках; валидация Pydantic-модели `SearchRequest` (включая `proportions` validator); парсинг ответа LLM (`_parse_result`) с тестовыми строками.
- **Integration**: эндпоинт с мокнутыми `AsyncLLMClient`, `EmbeddingClient`, `Manager` (Neo4j), `aiohttp.ClientSession` (Qdrant) → проверка кодов ответа, структуры `SearchResponse`, корректности статистики при разных комбинациях входных данных и сценариев ошибок.
- **API**: httpx-запросы к поднятому FastAPI с мокнутыми зависимостями → проверка структуры ответа и HTTP-кодов.
- **Pipeline**: сквозной тест с реальным Neo4j (testcontainers), реальным Qdrant (testcontainers/docker-compose), мокнутым `EmbeddingClient` и мокнутым `AsyncLLMClient`. Проверка полного цикла: вопрос → LLM extraction → embedding → Qdrant search → Neo4j graph traversal → ответ.

## Dependencies
- **Neo4j** (read): `Manager` (глобальный экземпляр `doc_manager`):
  - `MATCH (e:Entity {title, type})` — получение ноды входа
  - `MATCH (entry)-[r:RELATED]-(related)` — поиск связанных сущностей
  - `MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)` — поиск сообществ
  - `MATCH (c:Community {id})-[:IS_PARENT_OF]->(child)` — поиск дочерних сообществ
  - `MATCH (c:Community {id})-[:IS_CHILD_OF]->(parent)` — проверка leaf-условий
- **Qdrant** (read/search):
  - `POST /collections/documents/points/search` — поиск текстовых блоков
  - `POST /collections/entity_embeddings/points/search` — поиск точек входа Entity
  - `POST /collections/community_embeddings/points/search` — поиск корневого Community (уровень 0) и дочерних (если batch search)
  - `GET /collections/community_embeddings/points/{id}?with_vector=true` — получение эмбеддингов дочерних Community
  - Протокол: REST API через `aiohttp` (Constitution P5)
  - Аутентификация: `QDRANT_API_KEY` в заголовке `api-key` (если задан)
- **LLM API**: `AsyncLLMClient` + `AsyncGraphExtractor` для извлечения сущностей из вопроса:
  - `AsyncOpenAI` клиент к `LLM_URL`
  - Промпт: `GRAPH_EXTRACTION_PROMPT` (из `config.py`, нарушение P7)
- **Embedding API**: `EmbeddingClient.get_text_embedding()` (синхронный, через `asyncio.to_thread`):
  - POST `{EMBEDDING_BASE_URL}/embed`
  - Модель: `qwen3-emb`, 2048-мерные векторы
- **Tokenizer API**: HTTP POST на `QUERY_EXTRACTION_TOKENIZER_URL` через `AsyncLLMClient.count_tokens()`:
  - Тело: `{"model": QUERY_EXTRACTION_MODEL_NAME, "prompt": text}`
  - Ответ: `{"count": <int>}`
  - Fallback: `len(text) // 4` при недоступности
- **aiohttp**: асинхронные HTTP-запросы к Qdrant REST API (Constitution P5)
- **asyncio**: `asyncio.gather` для параллельных операций, `asyncio.to_thread` для синхронного `EmbeddingClient`, `asyncio.Semaphore` для ограничения параллелизма embedding-вызовов (`EMBEDDING_MAX_CONCURRENCY`)
- **Pydantic**: `SearchRequest`, `SearchResponse`, `TokensBreakdown`, `SearchStatistics` (новые модели в `dtype/search.py`)
- **math**: стандартный модуль для `sqrt` в `cosine_similarity()`
- **uuid**: стандартный модуль для `uuid5` при формировании ID точек Qdrant для community embeddings

## Pydantic Models (добавляются в `dtype/search.py`)

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    max_tokens: int = Field(..., gt=0, description="Максимальный размер вывода в токенах")
    proportions: Optional[Dict[str, float]] = Field(
        default=None,
        description="Доли токенов: text_units, entities, communities (сумма = 1.0)"
    )
    documents_filter: Optional[str] = Field(
        default="text_only",
        pattern="^(text_only|all)$",
        description="Фильтр: text_only или all"
    )

    @validator("proportions")
    def validate_proportions(cls, v):
        if v is not None:
            required_keys = {"text_units", "entities", "communities"}
            if set(v.keys()) != required_keys:
                raise ValueError(
                    f"proportions must contain exactly: {required_keys}"
                )
            total = sum(v.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"proportions must sum to 1.0, got {total}"
                )
            if any(p < 0 for p in v.values()):
                raise ValueError("proportions values must be non-negative")
        return v


class TokensBreakdown(BaseModel):
    text_units: int = 0
    entities: int = 0
    communities: int = 0


class SearchStatistics(BaseModel):
    processing_time_ms: int = 0
    tokens_used: TokensBreakdown = Field(default_factory=TokensBreakdown)
    tokens_remaining: TokensBreakdown = Field(default_factory=TokensBreakdown)
    entities_extracted_from_question: int = 0
    entities_matched_in_graph: int = 0
    entities_search_misses: int = 0
    fallback_used: bool = False
    total_items: TokensBreakdown = Field(default_factory=TokensBreakdown)


class SearchResponse(BaseModel):
    text_units: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    communities: List[str] = Field(default_factory=list)
    statistics: SearchStatistics = Field(default_factory=SearchStatistics)
```

## Exceptions

### P7 — Промпты инлайн (НАРУШЕНИЕ)
Промпт `GRAPH_EXTRACTION_PROMPT` для извлечения сущностей из вопроса определён как строковая константа в `config.py`. В соответствии с Constitution P7 должен быть вынесен в `prompts/graph-extraction.md`. Это наследуемое нарушение (затрагивает также `POST /process-document`), задокументированное в SERVICE.md Exceptions #1.

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует флага `archived` для мягкого удаления сущностей из семантического графа. Поскольку `archived` не реализован на узлах `Entity`, `Community` и связях `RELATED`, поиск может возвращать элементы, относящиеся к «удалённым» документам. Фильтрация по `archived` не применяется ни в одном из Neo4j-запросов. Это наследуемое нарушение, задокументированное в SERVICE.md Exceptions #7.

### P5 — Синхронный Qdrant scroll в `QdrantStreamAdapter` (НЕ ЗАТРАГИВАЕТ)
Существующий `QdrantStreamAdapter` использует синхронный `requests` для scroll-чтения (задокументированное нарушение P5 в SERVICE.md Exceptions #6). Эндпоинт `/search` не использует `QdrantStreamAdapter` — все Qdrant-запросы выполняются асинхронно через `aiohttp` (search, а не scroll), что соответствует Constitution P5.

### N3 — Конфигурация не Pydantic Settings (НАРУШЕНИЕ)
Параметры конфигурации сервиса используют модульные константы вместо Pydantic Settings. Для эндпоинта `/search` в `config.py` должны быть добавлены следующие константы:
- `QUERY_EXTRACTION_MODEL_NAME: Optional[str]` — имя модели для извлечения сущностей из вопроса (None → fallback на `MODEL_NAME`)
- `QUERY_EXTRACTION_LLM_URL: Optional[str]` — URL LLM API для query extraction (None → fallback на `LLM_URL`)
- `QUERY_EXTRACTION_API_KEY: Optional[str]` — API-ключ для query extraction (None → fallback на `LLM_API_KEY`)
- `QUERY_EXTRACTION_TOKENIZER_URL: Optional[str]` — URL токенизатора для query extraction (None → fallback на `TOKENIZER_URL`)
- `COMMUNITY_EMBEDDINGS_COLLECTION` и `COMMUNITY_EMBEDDINGS_NAMESPACE` — уже существующие константы, используемые для поиска в Qdrant

После рефакторинга конфигурации на Pydantic Settings все эти параметры должны быть перенесены в Settings-класс (см. SERVICE.md Exceptions #2).

### P9 — Глобальная операция (НЕ НАРУШЕНИЕ)
Поиск по графу — read-only операция над всем графом (без фильтра по `document_id`), что разрешено Constitution P9 для операций, не модифицирующих данные. В отличие от `process-document` (per-document), `/search` работает глобально, поскольку задача поиска требует обзора всего корпуса знаний.

### N3 — Конфигурация не Pydantic Settings (НАРУШЕНИЕ)
Параметры `COMMUNITY_EMBEDDINGS_COLLECTION` и `COMMUNITY_EMBEDDINGS_NAMESPACE` должны быть добавлены как константы в `config.py` (поскольку конфигурация сервиса `semantic_graph` в настоящее время использует модульные константы, см. SERVICE.md Exceptions #2). После рефакторинга конфигурации на Pydantic Settings эти параметры должны быть перенесены в Settings-класс.

### P4 — Два независимых графа Neo4j (НЕ НАРУШЕНИЕ)
Все Cypher-запросы в `/search` обращаются только к узлам и связям семантического графа (`Entity`, `Community`, `RELATED`, `CONSISTS_OF`, `IS_CHILD_OF`, `IS_PARENT_OF`). Узлы документального графа (`Document`, `Region:*`, `ORDER`, `PARENT`) не затрагиваются.
