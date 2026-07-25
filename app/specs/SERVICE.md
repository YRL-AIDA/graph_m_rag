# Service: app

## Identity
- Зона ответственности: Оркестрация ingestion pipeline (загрузка PDF → MinerU → эмбеддинги → Neo4j → semantic_graph), вопросно-ответный интерфейс (Q&A) с мультимодальным контекстом, PDF-рендеринг страниц с bbox-подсветкой, управление коллекциями Qdrant, управление документами (CRUD), health-check зависимых сервисов
- Порт: 8000 (по Constitution; `run_api()` в коде использует 9191 — см. Exceptions)
- Хранилище: MinIO (S3) — исходные PDF, результаты MinerU, эмбеддинги, изображения

## Dependencies
| Сервис | Протокол | Назначение |
|--------|----------|------------|
| mineru | REST (HTTP multipart/form-data) | Парсинг PDF → структурированный контент (текст, изображения, таблицы, формулы) |
| qdrant | direct driver (qdrant_client) | Векторная БД: хранение и поиск эмбеддингов элементов документа |
| neo4j (document graph) | direct driver через documet_index | Граф структуры документа: узлы `Document`, `Region:*`, связи `ORDER`, `PARENT` |
| semantic_graph | REST (HTTP POST) | Семантический граф знаний: извлечение сущностей, связей, сообществ |
| minio | direct driver (minio-py) | Объектное хранилище: PDF, mineru_results, embeddings, images |
| embedding (внешний сервис) | REST (HTTP POST) | Генерация текстовых и мультимодальных (image/text) эмбеддингов (qwen3-emb) |
| reranker (внешний сервис) | REST (HTTP POST) | Переранжирование результатов поиска по релевантности запросу |
| LLM (внешний сервис) | REST (OpenAI-compatible API) | Генерация ответа на вопрос на основе контекста (Qwen3-VL-32B) |

## API
| Method | Path | Feature Spec | Purpose |
|--------|------|-------------|---------|
| GET | / | — (будет feature spec) | Корневой эндпоинт: информация о сервисе и список эндпоинтов |
| GET | /health | — (будет feature spec) | Health-check: проверка доступности S3, embedding, mineru |
| POST | /upload-pdf | — (будет feature spec) | Загрузка PDF: хеширование, S3, MinerU, эмбеддинги, Neo4j graph, semantic graph |
| GET | /uploaded-files | — (будет feature spec) | Список загруженных PDF (из MinIO, префикс `pdfs/`) |
| GET | /collections | — (будет feature spec) | Список коллекций Qdrant с количеством точек |
| POST | /collections | — (будет feature spec) | Создание коллекции Qdrant (имя, vector_size, distance) |
| DELETE | /collections/{collection_name} | — (будет feature spec) | Удаление коллекции Qdrant по имени |
| GET | /collections/{collection_name}/files | — (будет feature spec) | Список файлов (file_hash) в конкретной коллекции Qdrant |
| GET | /ask-document | — (будет feature spec) | Web-интерфейс (HTML-страница) для задания вопросов к документу |
| POST | /ask-document | — (будет feature spec) | Поиск по документу: эмбеддинг вопроса → Qdrant → (опционально) rerank → (опционально) LLM-ответ |
| GET | /pdf/{file_hash} | — (будет feature spec) | Получение PDF-файла из MinIO по file_hash (inline просмотр) |
| GET | /api/pdf/{file_hash}/info | — (будет feature spec) | Метаданные PDF: имя файла, размер, дата изменения |
| GET | /api/pdf/{file_hash}/page/{page_number} | — (будет feature spec) | Рендеринг страницы PDF в PNG с опциональной bbox-подсветкой (PyMuPDF) |
| GET | /api/pdf/{file_hash}/mineru-bboxes | — (будет feature spec) | Bounding boxes из MinerU-результата с цветовой разметкой по типам элементов |
| DELETE | /documents/{file_hash} | — (будет feature spec) | Удаление документа: Qdrant точки + Neo4j graph + MinIO файлы (PDF, mineru, embeddings) |
| DELETE | /documents/all | — (будет feature spec) | Удаление ВСЕХ документов: все точки Qdrant + все графы Neo4j + все файлы MinIO |

## Data Model
Ссылки на Data Model Specs (будут созданы):
- `models/embeddings.md` — модели `Message`, `EmbedRequest`, `EmbedSuccessResponse` (schemas/embeddings.py)
- `models/reranker.md` — модели `Message`, `RerankRequest`, `RerankResponse`, `ResponseMessage` (schemas/reranker.py)
- `models/data_model.md` — модели `CollectionCreateRequest`, `CollectionInfo`, `CollectionsListResponse`, `QuestionRequest`, `QuestionResponse`, `UploadedFileInfo`, `UploadedFilesListResponse` (utils/data_model.py)
- `models/api_models.md` — модели `PDFUploadResponse`, `HealthCheckResponse` (api.py)

## Pipelines
Ссылки на Pipeline Specs (будут созданы):
- `pipelines/ingestion.md` — полный пайплайн загрузки документа: PDF validation → MD5 hash → MinIO upload → MinerU processing → embedding computation → Qdrant save → Neo4j document graph → semantic_graph extraction
- `pipelines/qa.md` — пайплайн вопросно-ответного поиска: question embedding → Qdrant search (filtered by file_hash) → Neo4j context enrichment → optional reranker → optional LLM answer generation

## Configuration
Переменные окружения (Pydantic Settings, класс `Settings` в `config/settings.py`, файл `.env`):

**S3/MinIO** (`S3Settings`):
- `S3_URL` (default: `http://localhost:9000`) — S3 endpoint URL
- `S3_ACCESS_KEY` (default: `minio`) — S3 access key
- `S3_SECRET_KEY` (default: `minio123`) — S3 secret key
- `S3_VERIFY_TLS` (default: `False`) — проверка TLS
- `S3_BUCKET_NAME` (default: `pdf-processing`) — имя бакета
- `MINIO_ROOT_USER` (default: `minioadmin`) — MinIO root user
- `MINIO_ROOT_PASSWORD` (default: `minioadmin`) — MinIO root password
- `MINIO_ENDPOINT` (default: `minio:9000`) — MinIO internal endpoint (Docker)
- `MINIO_BUCKET` (default: `pdf-processing`) — MinIO bucket name

**Qdrant** (`QdrantSettings`):
- `QDRANT_HOST` (default: `localhost`) — Qdrant host
- `QDRANT_PORT` (default: `6333`) — Qdrant HTTP port
- `QDRANT_GRPC_PORT` (default: `6334`) — Qdrant gRPC port
- `QDRANT_API_KEY` (default: `None`) — Qdrant API key
- `QDRANT_COLLECTION_NAME` (default: `documents`) — коллекция по умолчанию

**MinerU** (`MinerUSettings`):
- `MINERU_HOST` (default: `http://localhost`) — MinerU host
- `MINERU_PORT` (default: `8001`) — MinerU port
- `MINERU_TIMEOUT` (default: `300`) — таймаут запроса (сек)
- `MINERU_MAX_FILE_SIZE` (default: `52428800`, 50MB) — макс. размер файла
- `MINERU_CACHE_TTL` (default: `3600`) — TTL кэша (сек)
- `MODELSCOPE_CACHE` (default: `/app/models`) — директория кэша моделей
- `MINERU_BACKEND` (default: `pipeline`) — backend: pipeline или vlm
- `MINERU_METHOD` (default: `auto`) — метод: auto, txt, ocr
- `MINERU_LANG` (default: `ru`) — язык документа
- `MINERU_FORMULA_ENABLE` (default: `True`) — обработка формул
- `MINERU_TABLE_ENABLE` (default: `True`) — обработка таблиц

**Embedding** (`EmbeddingSettings`):
- `EMBEDDING_BASE_URL` (default: `http://192.168.19.127:10115/embedding`) — URL сервиса эмбеддингов
- `EMBEDDING_TIMEOUT` (default: `30`) — таймаут запроса (сек)
- `EMBEDDING_MODEL` (default: `qwen3-emb`) — модель эмбеддингов

**Reranker** (`RerankerSettings`):
- `RERANKER_BASE_URL` (default: `http://192.168.19.127:10115/reranker`) — URL сервиса реранкера
- `RERANKER_TIMEOUT` (default: `30`) — таймаут запроса (сек)
- `RERANKER_TOP_N` (default: `100`) — количество результатов после реранка

**LLM** (`LLMSettings`):
- `LLM_BASE_URL` (default: `http://192.168.19.127:8888/v1`) — URL LLM API (OpenAI-совместимый)
- `LLM_API_KEY` (default: `EMPTY`) — API ключ
- `LLM_MODEL_NAME` (default: `Qwen/Qwen3-VL-32B-Thinking`) — название модели
- `LLM_MAX_TOKENS` (default: `2048`) — макс. токенов в ответе
- `LLM_TEMPERATURE` (default: `0.7`) — температура генерации

**App** (`AppSettings`):
- `HOST` (default: `0.0.0.0`) — хост приложения
- `PORT` (default: `8000`) — порт приложения
- `DEBUG` (default: `False`) — режим отладки
- `LOG_LEVEL` (default: `INFO`) — уровень логирования
- `MAX_FILE_SIZE` (default: `52428800`, 50MB) — макс. размер загружаемого файла
- `CACHE_TTL` (default: `3600`) — TTL кэша (сек)
- `TEMP_DIR` (default: `/tmp/pdf_processing`) — временная директория
- `CORS_ORIGINS` (default: `["*"]`) — разрешённые CORS origins

## Invariants
- Единый идентификатор документа — MD5-хеш (`file_hash`) содержимого PDF. Используется во всех хранилищах (MinIO, Qdrant, Neo4j)
- Сквозной идентификатор региона — `file_hash|region_id`. `region_id` совпадает в Qdrant payload (`region_id`) и в Neo4j Document Graph (`Region` node)
- Коллекция Qdrant по умолчанию: `documents`, вектор 2048-dim, distance COSINE
- Дубликаты PDF (по MD5) не перезагружаются — возвращается статус `already_processed`
- Neo4j document index — опциональная зависимость (обрабатывается через `try/except ImportError` при импорте `documet_index`). Если модуль недоступен, создание графа пропускается, но ingestion не падает
- CORS: разрешены все origins (`*`), все methods, все headers
- Временные файлы PDF удаляются сразу после обработки MinerU (в `finally`-блоке)
- Файлы PDF проходят валидацию: расширение `.pdf`, непустой контент, защита от path traversal (Path.name)

## Exceptions
- **Переходный период**: Директория `specs/` создаётся впервые — Feature Specs, Data Model Specs и Pipeline Specs для существующих эндпоинтов и пайплайнов отсутствуют. Данный SERVICE.md — документирование текущего поведения.
- **Отклонение от Constitution P1 (порт)**: `run_api()` в `api.py` использует порт `9191` (hardcoded), в то время как Constitution и `AppSettings.PORT` декларируют `8000`. Порт 8000 указан в Identity согласно Constitution; фактический порт запуска определяется вызовом `run_api()`.
- **Отклонение от Constitution P5 (асинхронность)**: Функции `compute_embeddings_for_elements()`, `process_with_mineru()`, `check_document_indexed()`, `index_document_by_hash()` — синхронные. Эндпоинты `/upload-pdf` и `/ask-document` объявлены как `async def`, но внутри содержат синхронные вызовы без `run_in_executor`, что блокирует event loop. Некоторые эндпоинты (GET `/uploaded-files`, GET `/collections`, POST/DELETE `/collections`, DELETE `/documents`) объявлены как синхронные `def`, а не `async def`.
- **Отклонение от Constitution P6 (Pydantic-контракты)**: Ответы `POST /collections`, `DELETE /collections/{collection_name}`, `DELETE /documents/{file_hash}`, `DELETE /documents/all` возвращают `Dict[str, Any]` вместо строгих Pydantic-моделей. Поле `answers` в `QuestionResponse` типизировано как `List[Dict[str, Any]]` вместо строгой модели. Поле `services` в `HealthCheckResponse` — `Dict[str, str]`.
- **Отклонение от Constitution P7 (промпты в коде)**: LLM-промпт для `/ask-document` (system message и user prompt) определён как строковая константа в коде `api.py` (строки 1589-1676), а не вынесен в отдельный файл `prompts/`.
- **Отклонение от Constitution P5/P2 (hardcoded URL)**: Вызов `semantic_graph` в `/upload-pdf` использует hardcoded `http://localhost:9595/process-document` вместо переменной окружения или настроек. Не использует `file_hash` согласованно — `document_id` передаётся как `file_hash`, но Constitution требует `text_unit_ids` для семантического графа (производный от `file_hash`).
- **Отклонение от Constitution N2 (структура модулей)**: Модели данных расположены в `app/src/utils/data_model.py` и `app/src/schemas/`, а не в `dtype/` как предписано структурой. Отсутствуют `manager.py` — бизнес-логика встроена непосредственно в `api.py`.
