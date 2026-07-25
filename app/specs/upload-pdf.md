# Feature: Загрузка PDF и полный ingestion pipeline

## Motivation
Основной эндпоинт загрузки документа в систему. Принимает PDF-файл, загружает его в MinIO, отправляет в MinerU для парсинга структурированного контента, вычисляет эмбеддинги для каждого элемента, сохраняет их в Qdrant, строит граф структуры документа в Neo4j (через `documet_index`), и инициирует извлечение семантического графа знаний (через `semantic_graph`).

## Behaviour

### Input
- **Method**: `POST /upload-pdf`
- **Content-Type**: `multipart/form-data`
- **Параметр**: `file: UploadFile = File(...)` — PDF-файл
- **Валидация**:
  - Имя файла непустое и заканчивается на `.pdf` (нижний регистр)
  - Защита от path traversal: `Path(file.filename).name` должен совпадать с `file.filename`
  - Контент файла непустой

### Processing
1. **Валидация файла**: проверка расширения `.pdf`, path traversal, непустой контент
2. **Хеширование**: вычисление MD5-хеша содержимого (`file_hash = hashlib.md5(content).hexdigest()`)
3. **Проверка дубликата**: поиск в MinIO объектов с префиксом `pdfs/{file_hash}_`. Если найдены — возврат `status: "already_processed"` (embeddings_computed=0, без обработки MinerU)
4. **Загрузка PDF в MinIO**: ключ `pdfs/{file_unique_id}/{safe_filename}`, где `file_unique_id = {file_hash}_{safe_filename}`
5. **Создание временного файла**: `/tmp/pdf_processing/{file_hash}_{safe_filename}`, запись контента
6. **Вызов MinerU** (синхронный): `process_with_mineru(file_path)` → `mineru_client.process_document(file_path, backend="pipeline", method="auto", lang="en", formula_enable=True, table_enable=True)`
7. **Удаление временного файла**: в `finally`-блоке сразу после MinerU
8. **Сохранение результата MinerU в MinIO**: ключ `mineru_results/{file_unique_id}/result.json`, добавление метаданных (`file_hash`, `original_filename`, `processed_at`)
9. **Сохранение изображений из MinerU**: `images_base64` → MinIO c ключом `images/{img_key}`, content-type `image/jpeg`
10. **Извлечение элементов**: `mineru_result["results"]["result"]["results"]["content_list"]`
11. **Вычисление эмбеддингов** (синхронный вызов `compute_embeddings_for_elements(elements, file_hash)`):
    - **text элементы** (`text_level == 1` → `title`, иначе `text (level N)`): текстовый эмбеддинг через `emb_client.get_text_embedding()`
    - **image элементы**: загрузка изображения из MinIO, мультимодальный эмбеддинг через `emb_client.get_image_text_embedding_base64()`, отдельные текстовые эмбеддинги для caption и footnote
    - **table элементы**: текстовые эмбеддинги для caption, footnote и тела таблицы
    - **equation элементы**: текстовый эмбеддинг для LaTeX-контента
    - **discarded элементы**: пропускаются
    - Каждый эмбеддинг:
      - Добавляется в батч для Qdrant (вектор + текст + метаданные с `region_id`, `element_index`, `element_type`, `file_hash`, `original_element`)
      - Сохраняется в MinIO как `embeddings/{file_hash}/region_{region_id}.json`
      - `region_id` инкрементируется (начинается с 0) для синхронизации с Neo4j
12. **Сохранение в Qdrant**: создание коллекции (если не существует, размер вектора — длина первого эмбеддинга), пакетное сохранение всех эмбеддингов через `qdrant_client.save_embeddings()`
13. **Создание Neo4j document graph** (опционально): если `NEO4J_AVAILABLE`, вызов `create_neo4j_graph(mineru_result, file_hash)`. При неудаче ingestion не падает.
14. **Вызов semantic_graph** (опционально): если `NEO4J_AVAILABLE`, синхронный POST на **hardcoded** `http://localhost:9595/process-document` с `{"document_id": file_hash}`. Результат логируется через `print()`.
15. **Возврат ответа**: `PDFUploadResponse` с временем обработки

### Output
- **200 OK**: `PDFUploadResponse`
  ```json
  {
    "status": "success" | "already_processed",
    "message": "...",
    "file_hash": "md5hex",
    "s3_path": "pdfs/{file_hash}_{filename}/{filename}",
    "mineru_result_path": "mineru_results/{file_hash}_{filename}/result.json",
    "embeddings_computed": 42,
    "processing_time": 12.34,
    "neo4j_graph_created": true | false
  }
  ```
- **400 Bad Request**: неверное расширение, path traversal, пустой файл
- **500 Internal Server Error**: ошибка MinerU, MinIO, эмбеддингов, Qdrant

## API Contract

```openapi
post: /upload-pdf
summary: Upload PDF document, process with MinerU, compute embeddings, create Neo4j graphs
requestBody:
  required: true
  content:
    multipart/form-data:
      schema:
        type: object
        required:
          - file
        properties:
          file:
            type: string
            format: binary
            description: PDF file with .pdf extension
responses:
  '200':
    description: PDF processed successfully or already exists
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/PDFUploadResponse'
  '400':
    description: Validation error (not a PDF, empty file, path traversal)
  '500':
    description: Processing error (MinerU, MinIO, embedding, Qdrant failure)
```

## Data Flow
```
Пользователь → POST /upload-pdf (multipart: PDF)
  → MinIO: загрузка исходного PDF (pdfs/{file_hash}_*/)
  → MinerU (REST): парсинг PDF → структурированный контент
  → MinIO: сохранение mineru_results/{file_hash}_*/result.json
  → MinIO: сохранение images/{img_key} (из MinerU)
  → MinIO: загрузка изображений для эмбеддингов
  → Embedding Service (REST): текстовые и image/text эмбеддинги
  → MinIO: сохранение embeddings/{file_hash}/region_{region_id}.json
  → Qdrant (direct driver): создание коллекции + сохранение векторов
  → Neo4j (через documet_index): создание Document Graph (ORDER/PARENT)
  → semantic_graph (REST): POST http://localhost:9595/process-document
```

## LLM Interactions
- В данном эндпоинте вызовов LLM нет. MinerU использует внутренние VLM-модели (`backend="pipeline"`).
- Semantic graph (вызывается асинхронно, без ожидания результата) использует LLM для извлечения сущностей — см. спецификацию `semantic_graph`.

## LLM Model Requirements
- Не применимо к данному эндпоинту напрямую. Модели используются опосредованно через MinerU (VLM для распознавания) и semantic_graph (LLM для извлечения сущностей).

## Error Handling
| Failure mode | Обработка |
|---|---|
| Неверное расширение файла | 400 Bad Request |
| Path traversal в имени файла | 400 Bad Request |
| Пустой файл | 400 Bad Request |
| Дубликат (уже существует) | 200 OK со статусом `already_processed` |
| Ошибка создания временного файла | 500 Internal Server Error |
| Ошибка MinerU | 500 Internal Server Error (raise HTTPException) |
| Ошибка загрузки изображения из MinIO при эмбеддингах | Логируется, пропуск элемента |
| Ошибка вычисления эмбеддинга | Логируется, пропуск элемента |
| Ошибка сохранения в Qdrant | Логируется, не прерывает обработку |
| Ошибка создания Neo4j graph | Логируется, не прерывает обработку (neo4j_graph_created=false) |
| Ошибка вызова semantic_graph | Логируется через print(), не прерывает обработку |
| Отсутствие content_list в результате MinerU | Логируется (warning), embeddings_computed=0 |

## Testing
1. **Загрузка валидного PDF**: успешный ingestion, все хранилища содержат данные
2. **Повторная загрузка того же PDF**: возврат `already_processed`, без повторной обработки
3. **Загрузка не-PDF файла** (например, `.txt`): 400 Bad Request
4. **Загрузка пустого файла**: 400 Bad Request
5. **Path traversal в имени**: `../etc/passwd.pdf` → 400 Bad Request
6. **Недоступность MinerU**: 500 Internal Server Error
7. **Недоступность Neo4j** (`NEO4J_AVAILABLE=False`): ingestion завершается успешно, но `neo4j_graph_created=false`
8. **PDF с изображениями/таблицами/формулами**: проверка что все типы элементов получают эмбеддинги
9. **PDF с discarded элементами**: discarded пропускаются, не влияют на подсчёт

## Dependencies
- `minio_client` (MinioClient)
- `mineru_client` (MinerUClient)
- `emb_client` (EmbeddingClient)
- `qdrant_client` (QdrantClientWrapper)
- `documet_index` (DocumentIndexService, create_neo4j_graph) — опционально
- `semantic_graph` (HTTP POST) — опционально, hardcoded URL
- `hashlib`, `os`, `time`, `json`, `pathlib`

## Exceptions
- **P5 (Синхронные вызовы в async)**: Эндпоинт объявлен `async def`, но внутри синхронные вызовы `process_with_mineru()`, `compute_embeddings_for_elements()`, `create_neo4j_graph()`, `requests.post()` — без `run_in_executor`. Блокирует event loop на время обработки (десятки секунд).
- **P1 (Сервисные границы)**: Вызов semantic_graph использует hardcoded URL `http://localhost:9595/process-document` вместо настроек (Pydantic Settings). Поле `document_id` передаётся как `file_hash`, а не как `text_unit_ids` (производный от `file_hash`), что противоречит документированному контракту I1 в Constitution.
- **P7 (Промпты в коде)**: Не применимо напрямую (нет LLM-промптов), но результат вызова `requests.post()` логируется через `print()`.
- **N2 (Структура модулей)**: Бизнес-логика ingestion'а встроена непосредственно в `api.py` (около 500 строк функций `process_with_mineru`, `compute_embeddings_for_elements`, и тела `upload_pdf`), вместо выделенного `manager.py`.
