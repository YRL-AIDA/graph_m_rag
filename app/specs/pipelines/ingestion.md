# Pipeline: Document Ingestion

## Purpose
Полный пайплайн приёма и обработки PDF-документа: от загрузки файла через HTTP до сохранения структурированного контента во всех хранилищах системы (MinIO, Qdrant, Neo4j Document Graph, Neo4j Semantic Graph). Запускается через эндпоинт `POST /upload-pdf` в `app/src/api.py`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Validate & Hash | `UploadFile` (PDF, multipart/form-data) | `api.py:upload_pdf()` — проверка расширения `.pdf`, проверка имени файла на path traversal (`Path(file.filename).name`), проверка на пустой файл. Вычисление MD5-хеша содержимого: `hashlib.md5(content).hexdigest()`. | `file_hash: str`, `safe_filename: str`, `content: bytes` | sequential |
| 2. Duplicate Check | `file_hash` | `api.py:upload_pdf()` — проверка существования PDF в MinIO по префиксу `pdfs/{file_hash}_` через `minio_client.list_objects()`. Если PDF уже существует, возвращается `PDFUploadResponse(status="already_processed")` с `embeddings_computed=0`. | Ранний возврат (HTTP 200, status=already_processed) или продолжение pipeline | sequential |
| 3. Upload PDF to MinIO | `content: bytes` | `api.py:upload_pdf()` — загрузка PDF в MinIO по ключу `pdfs/{file_hash}_{safe_filename}/{safe_filename}` через `minio_client.upload()`. | `pdf_s3_key: str` | sequential |
| 4. Temp File Creation | `content: bytes` | `api.py:upload_pdf()` — создание временной директории `/tmp/pdf_processing/`, запись `content` во временный файл `{file_hash}_{safe_filename}`. | `temp_file_path: str` | sequential |
| 5. MinerU Processing | `temp_file_path` | `api.py:process_with_mineru()` → `MinerUClient.process_document()` с параметрами: `backend="pipeline"`, `method="auto"`, `lang="en"`, `formula_enable=True`, `table_enable=True`. HTTP POST на `{MINERU_HOST}:{MINERU_PORT}/process`. | `mineru_result: Dict` (содержит `results.result.results.content_list` и `results.result.results.images_base64`) | sequential |
| 6. Temp File Cleanup | — | `api.py:upload_pdf()` — удаление временного файла `os.remove(temp_file_path)` в блоке `finally`. | — | sequential |
| 7. Store MinerU Result in MinIO | `mineru_result` | `api.py:upload_pdf()` — сериализация `mineru_result` (с добавлением `metadata.file_hash`, `metadata.original_filename`, `metadata.processed_at`) в JSON, загрузка в MinIO по ключу `mineru_results/{file_hash}_{safe_filename}/result.json`. Функция `convert_to_serializable()` для обработки datetime. | `mineru_result_key: str` | sequential |
| 8. Save Images to MinIO | `mineru_result.results.result.results.images_base64` | `api.py:upload_pdf()` — итерация по `images_base64`, декодирование base64 и загрузка каждого изображения в MinIO по ключу `images/{img_key}` с content-type `image/jpeg`. | — | sequential |
| 9. Compute Embeddings & Qdrant Upsert | `content_list: List[Dict]`, `file_hash` | `api.py:compute_embeddings_for_elements()` — Итерация по элементам с инкрементальным `region_id` (начинается с 0). Для каждого элемента по типу: **text**: если `text_level==1` → `element_type="title"`, иначе text; эмбеддинг текста через `emb_client.get_text_embedding()`. **image**: скачивание из MinIO → base64 → `emb_client.get_image_text_embedding_base64()` для основного изображения + `emb_client.get_text_embedding()` отдельно для caption и footnote. **table**: `emb_client.get_text_embedding()` для caption, footnote, и тела таблицы. **equation**: `emb_client.get_text_embedding()` для LaTeX. **discarded**: пропускается. Каждый эмбеддинг сохраняется в MinIO (`embeddings/{file_hash}/region_{region_id}.json`) и буферизуется в списки `embeddings_list`, `texts_list`, `metadata_list`. После обработки всех элементов: создание коллекции Qdrant (если не существует) через `qdrant_client.create_collection(vector_size=2048)`, затем `qdrant_client.save_embeddings()` — запись всех точек батчем. | `embeddings_computed: int`, точки в Qdrant с payload: `region_id`, `element_index`, `element_type`, `file_hash`, `created_at`, `original_element` | sequential (элементы последовательно, Qdrant upsert — батчем) |
| 10. Create Neo4j Document Graph | `mineru_result`, `file_hash` | `api.py:upload_pdf()` → `documet_index.create_neo4j_graph()` → `DocumentIndexService.create_graph_from_mineru_result()` → `documet_index/dtype/document.py:create_graph_from_mineru_result()` — парсинг content_list, создание `Region` узлов с типами: `title`, `text`, `image`, `image_caption`, `image_footnote`, `table`, `table_caption`, `table_footnote`, `equation`. Построение `ORDER` связей (последовательное чтение) и `PARENT` связей (иерархия заголовков через стек). Запись в Neo4j через `Manager.add_document()` (один большой Cypher-запрос). | `neo4j_graph_created: bool` | sequential |
| 11. Call Semantic Graph Extraction | `file_hash` | `api.py:upload_pdf()` — синхронный HTTP POST `requests.post("http://localhost:9595/process-document", json={"document_id": file_hash})`. Выполняется только если `NEO4J_AVAILABLE=True`. Ошибка логируется, но не прерывает pipeline. | — | sequential (fire-and-forget) |

## Data Flow Diagram

```
User Upload (PDF)
    │
    ▼
[Stage 1] Validate & Hash ──► file_hash
    │
    ▼
[Stage 2] Duplicate Check ──► already_processed? → HTTP 200 (early return)
    │ new document
    ▼
[Stage 3] MinIO: Upload PDF ──► pdfs/{hash}_{name}/{name}
    │
    ▼
[Stage 4] Temp file: /tmp/pdf_processing/{hash}_{name}
    │
    ▼
[Stage 5] MinerU: POST /process ──► content_list + images_base64
    │
    ▼
[Stage 6] Cleanup temp file
    │
    ▼
[Stage 7] MinIO: Store MinerU result ──► mineru_results/{hash}_{name}/result.json
    │
    ▼
[Stage 8] MinIO: Save images ──► images/{img_key}
    │
    ▼
[Stage 9] Embedding Service: /embed (text + image/text)
    │
    ├──► MinIO: embeddings/{hash}/region_{N}.json
    └──► Qdrant: points in collection "documents" (batch upsert)
    │
    ▼
[Stage 10] Neo4j Document Graph: Document → Region:* via ORDER + PARENT
    │
    ▼
[Stage 11] HTTP POST localhost:9595/process-document → Semantic Graph extraction (async fire-and-forget)
```

## LLM Interactions

На данном этапе LLM не используется напрямую. Вызов LLM происходит опосредованно через Stage 11 (semantic_graph extraction) и внутри MinerU (Stage 5, при backend="vlm" используется VLM-модель).

- `mineru` — VLM модель `MinerU2.5-2509-1.2B` (Qwen2VL) для backend="vlm" (Stage 5).
- `semantic_graph` — вызов `/process-document` запускает LLM extraction (см. `semantic_graph/specs/pipelines/entity-extraction.md`).

## Performance Constraints

- **File size**: ограничений нет, но файл полностью читается в память (`await file.read()`).
- **MinerU timeout**: 300 секунд (5 минут) в `MinerUClient.process_document()` через `timeout=300`.
- **Embedding service timeout**: 300 секунд на запрос (`EmbeddingClient.__init__(timeout=300)`).
- **Embedding vector size**: 2048 измерений (`qdrant_client.create_collection(vector_size=2048)`).
- **Qdrant**: коллекция создаётся с `Distance.COSINE`.
- **Память**: все эмбеддинги буферизуются в памяти (списки `embeddings_list`, `texts_list`, `metadata_list`) до батчевого upsert в Qdrant.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Validate | Неверное расширение, пустой файл, path traversal | HTTP 400, pipeline прерывается |
| 2. Duplicate Check | Ошибка MinIO при проверке | Исключение пробрасывается как HTTP 500 |
| 3. Upload PDF | Ошибка MinIO | Исключение → HTTP 500 |
| 4. Temp File | Ошибка записи | `IOError` → HTTP 500 |
| 5. MinerU | ConnectionError, Timeout, HTTP error | `HTTPException(status_code=500)` в `process_with_mineru()` |
| 6. Cleanup | Ошибка удаления | Логируется предупреждение, не прерывает pipeline |
| 7. Store MinerU Result | Ошибка MinIO | Исключение → HTTP 500 |
| 8. Save Images | Ошибка MinIO для отдельного изображения | Логируется `logger.error`, изображение пропускается |
| 9. Compute Embeddings | Ошибка эмбеддинга для отдельного элемента | Логируется `logger.error`, элемент пропускается, `region_id` инкрементируется. Ошибка Qdrant upsert логируется, но не прерывает pipeline |
| 10. Neo4j Document Graph | Ошибка Neo4j, документ уже существует | Логируется `logger.error`, `neo4j_graph_created=False`, pipeline продолжается (документ доступен в MinIO и Qdrant) |
| 11. Semantic Graph | HTTP ошибка или таймаут | Логируется, pipeline НЕ прерывается |

Общая обработка: внешний `try/except` в `upload_pdf()` ловит все исключения и возвращает HTTP 500. Блок `finally` гарантирует удаление временного файла.
