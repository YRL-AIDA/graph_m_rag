# Pipeline: Question Answering (Q&A)

## Purpose
Пайплайн ответа на вопрос по конкретному документу. Принимает вопрос пользователя и `file_hash` документа, выполняет семантический поиск по векторной БД, обогащает контекст из графа структуры документа, опционально применяет реранкинг и генерирует ответ через LLM. Запускается через эндпоинт `POST /ask-document` в `app/src/api.py`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Auto-Index Check | `file_hash`, `collection_name` | `api.py:check_document_indexed()` — поиск в Qdrant по фильтру `file_hash` с dummy-вектором `[0.0]*2048`, `limit=1`. Если не найден → `api.py:index_document_by_hash()` — поиск MinerU result в MinIO по префиксу `mineru_results/{file_hash}_`, скачивание, вызов `compute_embeddings_for_elements()`. | `is_indexed: bool` | sequential |
| 2. Question Embedding | `question: str` | `emb_client.get_text_embedding(question)` — вызов эмбеддинг-сервиса `POST /embed` с `Message(type="text", text=question)`. | `question_embedding: List[float]` (2048-dim) | sequential |
| 3. Qdrant Search | `question_embedding`, `file_hash` | `qdrant_client.search()` с фильтром `models.Filter(must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))])`. Лимит: если `use_reranker=True` → `max(limit*3, rerank_top_n)`, иначе → `limit`. Использует `client.query_points()`. | `search_results: List[ScoredPoint]` | sequential |
| 4. Neo4j Context Enrichment | `search_results`, `file_hash` | `api.py:ask_document()` — для каждого результата с `element_type in ("image", "table", "image_caption", "image_footnote", "table_caption", "table_footnote")`: вызов `DocumentIndexService.get_related_context(file_hash, element_type, text)` → `Manager.get_related_context()`. Для caption/footnote элементов ищет родительский image/table узел (Cypher: `ORDER*` traversal). Для image/table элементов ищет дочерние caption/footnote узлы. Результаты добавляются как отдельные answer-объекты с `is_related_context=True` и пониженным score (×0.9 для parent, ×0.85 для sibling). | `answers: List[dict]` (обогащённые neo4j_context) | sequential |
| 5. Image Download | `answers` | `api.py:ask_document()` — для каждого answer с `element_type in ("image", "table", "image_caption", "image_footnote", "table_caption", "table_footnote")` и наличием `img_path`: скачивание изображения из MinIO `minio_client.get_object()`, кодирование в base64. Для image создаётся `Message(type="image/text")` с текстом captions/footnotes. Для остальных — `Message(type="text")`. | `answers` с заполненным `image_base64`, `documents_to_rerank: List[Message]` | sequential |
| 6. Reranking (optional) | `question`, `documents_to_rerank` | Если `use_reranker=True`: `reranker_client.rerank(query_text=question, messages=documents_to_rerank, instruction="Retrieve images or text relevant to the user's query.")` → HTTP POST на `{RERANKER_BASE_URL}/rerank`. Результаты сортируются по `score` (по убыванию), перезаписывают `answers` с полями `reranker_score`, `original_score`. | `answers` (переупорядоченные) | sequential |
| 7. LLM Answer Generation (optional) | `answers`, `question` | Если `use_llm=True` и `answers` не пуст: формирование мультимодального сообщения для LLM: **system_message** (`ModelMessageDict(role='system')`) с инструкцией на русском языке. **user_message** (`ModelMessageDict(role='user')`) — сначала добавляются все изображения (`add_img_content_base64()`), затем текстовый контекст с маркерами `[БЛОК N]`, Neo4j-контекстом (`→ СВЯЗАННЫЙ ЭЛЕМЕНТ`, `→ ПОДПИСЬ`, `→ СНОСКА`), затем вопрос и инструкция. Вызов `llm_client.send_message()` с параметрами `max_tokens=9182`, `temperature=0.3`, `top_p=0.9`. Модель: `Qwen/Qwen3-VL-32B-Thinking` (определена в `LLMClient.__init__`). | `llm_answer: str` (или None при ошибке) | sequential |

## Data Flow Diagram

```
User Question + file_hash
    │
    ▼
[Stage 1] Auto-Index Check
    │ is_indexed?
    ├── no ──► MinIO: load mineru result → compute embeddings → Qdrant upsert
    │
    ▼
[Stage 2] Embedding Service: /embed (text)
    │ question_embedding (2048-dim)
    ▼
[Stage 3] Qdrant: search by file_hash filter
    │ search_results (limit * N)
    ▼
[Stage 4] Neo4j: get_related_context() per result
    │ enriched answers (parent elements, sibling captions/footnotes)
    ▼
[Stage 5] MinIO: download images → base64
    │ image_base64 in answers
    ▼
[Stage 6] (optional) Reranker: POST /rerank
    │ reordered answers by relevance
    ▼
[Stage 7] (optional) LLM: Qwen3-VL-32B-Thinking
    │ multimodal context (text + images) → generated answer
    ▼
QuestionResponse: { status, answers, llm_answer, indexed, collection_name }
```

## LLM Interactions

- **Prompt**: встроен в код `api.py:ask_document()` как строковые константы (нарушение P7 Constitution — должен быть вынесен в `app/prompts/qa-system.md` и `app/prompts/qa-user.md`).
- **System prompt** (русский): инструкция ассистенту — отвечать только на основе контекста, анализировать изображения и подписи, цитировать фрагменты, быть точным, сохранять язык вопроса.
- **User prompt** (русский): структурированный контекст (изображения → текст → Neo4j контекст) + вопрос + инструкция.
- **Модель**: `Qwen/Qwen3-VL-32B-Thinking` (определена в `LLMClient.__init__`).
- **Параметры генерации**: `max_tokens=9182`, `temperature=0.3`, `top_p=0.9`.
- **LLM Requirements**: мультимодальная модель (текст + изображения), поддержка OpenAI-compatible API (`client.chat.completions.create`).

## Performance Constraints

- **Embedding timeout**: 300 секунд.
- **Qdrant search**: вектор 2048-dim, лимит до `limit * 3` (с реранкингом).
- **Reranker timeout**: 300 секунд.
- **LLM timeout**: определяется OpenAI клиентом (по умолчанию без таймаута).
- **Max tokens (LLM)**: 9182.
- **Neo4j context enrichment**: N Cypher-запросов, где N — количество результатов поиска. Может быть медленным при большом `limit`.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Auto-Index | Документ не найден в MinIO, ошибка Qdrant | `QuestionResponse(status="error")` с сообщением об ошибке индексации |
| 2. Question Embedding | Ошибка эмбеддинг-сервиса | HTTP 500 |
| 3. Qdrant Search | Ошибка Qdrant | HTTP 500 |
| 4. Neo4j Context | Neo4j недоступен, ошибка запроса | Логируется `logger.warning`, `neo4j_context=None`, pipeline продолжается без обогащения |
| 5. Image Download | Ошибка MinIO для отдельного изображения | Логируется `logger.error`/`logger.warning`, `image_base64=None`, элемент остаётся текстовым |
| 6. Reranking | Ошибка реранкера | Логируется `logger.warning`, используются исходные результаты поиска |
| 7. LLM Answer | Ошибка LLM | Логируется `logger.error`, `llm_answer` содержит сообщение об ошибке, `QuestionResponse` всё равно возвращается с answers |

Общая обработка: внешний `try/except` → HTTP 500.
