# Feature: Вопросно-ответный поиск по документу (Ask Document)

## Motivation
Обеспечивает семантический поиск по содержимому загруженного документа с опциональным переранжированием, Neo4j-контекстным обогащением и генерацией ответа через LLM. Состоит из двух эндпоинтов: `GET /ask-document` (web-интерфейс) и `POST /ask-document` (API поиска).

## Behaviour

### Input

#### GET /ask-document
- Без параметров
- Возвращает статический HTML-файл `app/static/ask-document.html`

#### POST /ask-document
- **Content-Type**: `application/json`
- **Тело**: `QuestionRequest` (Pydantic)

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `file_hash` | `str` | обязательное | MD5-хеш документа |
| `question` | `str` | обязательное | Текст вопроса |
| `limit` | `int` | 10 | Количество возвращаемых результатов |
| `collection_name` | `Optional[str]` | None | Имя коллекции Qdrant (если None — используется дефолтная) |
| `use_llm` | `bool` | False | Генерировать ответ через LLM |
| `use_reranker` | `bool` | False | Применять реранкер |

### Processing

1. **Выбор коллекции**: `client = get_qdrant_client(collection_name)` или дефолтный `qdrant_client`
2. **Проверка индексации** (`check_document_indexed`):
   - Создаётся `Filter` Qdrant: `file_hash == request.file_hash`
   - Поиск с dummy-вектором `[0.0]*2048`, limit=1
   - Возвращает `True` если найдена хотя бы одна точка
3. **Авто-индексация при отсутствии** (`index_document_by_hash`):
   - Поиск `mineru_results/` в MinIO с префиксом `file_hash_`
   - Загрузка mineru result, извлечение `content_list`
   - Вызов `compute_embeddings_for_elements(elements, file_hash)` — синхронный
   - Временная подмена `qdrant_client.collection_name` на целевую коллекцию
   - При неудаче — возврат `status: "error"` без HTTP-ошибки
4. **Эмбеддинг вопроса**: `emb_client.get_text_embedding(question)` — синхронный
5. **Поиск в Qdrant**:
   - Фильтр по `file_hash`
   - `search_limit = max(limit * 3, rerank_top_n)` если `use_reranker`, иначе `limit`
   - Вызов `client.search(query_vector, limit=search_limit, filter_condition)`
6. **Neo4j контекстное обогащение** (только если `NEO4J_AVAILABLE`):
   - Инициализация `DocumentIndexService()`
   - Для элементов типов `image`, `table`, `image_caption`, `image_footnote`, `table_caption`, `table_footnote`:
     - Вызов `neo4j_service.get_related_context(file_hash, element_type, text)`
     - При наличии `parent_element` — создаётся отдельный answer с `is_related_context=true`, загрузкой изображения из MinIO
     - При наличии `sibling_captions` — отдельные answers с загрузкой изображений
     - При наличии `sibling_footnotes` — отдельные answers с загрузкой изображений
   - Закрытие Neo4j-соединения
7. **Загрузка изображений для результатов**:
   - Для `image` элементов: загрузка из MinIO, base64-кодирование, создание мультимодального `Message` для реранкера
   - Для `image_caption/image_footnote/table_caption/table_footnote`: загрузка ассоциированного изображения при наличии `img_path`
   - Остальные типы: только текст
8. **Реранкинг** (опционально, если `use_reranker` и есть документы):
   - Вызов `reranker_client.rerank(query_text=question, messages=documents_to_rerank)` — синхронный
   - Сортировка по `score`, фильтрация `message_id < limit`
   - Замена `score` на `reranker_score`, сохранение `original_score`
9. **LLM-ответ** (опционально, если `use_llm` и есть answers):
   - **System message** (строковая константа в коде, русский): инструкция анализировать только контекст, быть точным и лаконичным
   - **User message**: изображения (как base64 в `image_content`), затем текстовый контекст из всех answers (включая Neo4j enrichment), вопрос пользователя с инструкцией
   - Вызов `llm_client.send_message(messages=[system_message, user_message], max_tokens=9182, temperature=0.3, top_p=0.9)` — синхронный
   - Модель: Qwen3VL-32B (мультимодальная, OpenAI-совместимый API)
10. **Возврат ответа**: `QuestionResponse` с answers, llm_answer (если сгенерирован)

### Output

#### GET /ask-document
- **200 OK**: `text/html` — содержимое файла `app/static/ask-document.html`
- **404 Not Found**: HTML-файл не найден

#### POST /ask-document
- **200 OK**: `QuestionResponse`
  ```json
  {
    "status": "success" | "error",
    "message": "Found N relevant chunks...",
    "file_hash": "md5hex",
    "question": "...",
    "answers": [
      {
        "text": "chunk text",
        "score": 0.95,
        "element_type": "text",
        "element_index": 0,
        "page_idx": 0,
        "img_path": null,
        "image_base64": null,
        "bbox": [x1, y1, x2, y2],
        "neo4j_context": null,
        "is_related_context": false,
        "reranker_score": 0.91,
        "original_score": 0.95,
        "related_to_element_type": "..."
      }
    ],
    "indexed": true,
    "collection_name": "documents",
    "llm_answer": "Сгенерированный ответ..."
  }
  ```
- **500 Internal Server Error**: ошибка обработки вопроса

## API Contract

```openapi
get: /ask-document
summary: Serve the Ask Document web interface (HTML page)
responses:
  '200':
    description: HTML page for asking questions about documents
    content:
      text/html:
        schema:
          type: string
  '404':
    description: Web interface file not found

post: /ask-document
summary: Ask a question about a specific document by file_hash
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/QuestionRequest'
responses:
  '200':
    description: Search results with optional LLM answer
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/QuestionResponse'
  '500':
    description: Internal processing error
```

## Data Flow
```
Пользователь → POST /ask-document (JSON: file_hash, question)
  → Qdrant: проверка индексации (search с фильтром по file_hash)
  → [если не indexed] MinIO: загрузка mineru_result
  → [если не indexed] Embedding Service: вычисление эмбеддингов элементов
  → [если не indexed] Qdrant: сохранение эмбеддингов
  → Embedding Service: эмбеддинг вопроса
  → Qdrant: поиск ближайших векторов с фильтром file_hash
  → MinIO: загрузка изображений для image/table/caption/footnote результатов
  → Neo4j (documet_index): контекстное обогащение (parent_element, sibling_captions, sibling_footnotes)
  → [опционально] Reranker Service: переранжирование результатов
  → [опционально] LLM: генерация ответа (мультимодальный: текст + изображения)
```

## LLM Interactions

Промпт для генерации ответа определён как строковая константа в коде `api.py` (строки 1589-1676), **не вынесен** в `prompts/`.

### System Prompt (в коде)
Русскоязычная инструкция ассистенту:
- Использовать ТОЛЬКО информацию из предоставленного контекста
- Если ответ не найден — честно сообщить
- Анализировать изображения, таблицы и диаграммы вместе с подписями
- Цитировать конкретные фрагменты контекста
- Быть точным и лаконичным
- Указывать на противоречия в контексте
- Сохранять язык ответа как язык вопроса

### User Prompt (в коде)
Формируется динамически:
1. Изображения из answers (base64) — в начало сообщения
2. Текстовый контекст: каждый answer как `[БЛОК N] (тип: ...)` с Neo4j-контекстом (СВЯЗАННЫЙ ЭЛЕМЕНТ, ПОДПИСЬ, СНОСКА)
3. Вопрос пользователя с инструкцией проанализировать изображения и текст

### LLM Model Requirements
- **Тип модели**: multimodal (text+image) — Qwen3-VL-32B-Thinking
- **Минимальный размер контекста**: ~9182 токенов на выходе (max_tokens), входной контекст зависит от количества answers и размера изображений
- **Язык выхода**: русский (соответствует языку вопроса)
- **Требования к формату выхода**: свободный текст
- **Параметры**: temperature=0.3, top_p=0.9
- Конкретная модель и API URL задаются в `config/settings.py` (`LLMSettings`)

## Error Handling
| Failure mode | Обработка |
|---|---|
| Документ не индексирован и не может быть проиндексирован | Возврат `status: "error"` (HTTP 200) |
| Документ не найден в MinIO при авто-индексации | `index_document_by_hash` возвращает False |
| Ошибка эмбеддинга вопроса | 500 Internal Server Error |
| Ошибка поиска Qdrant | 500 Internal Server Error |
| Ошибка инициализации Neo4j | Логируется, `neo4j_service = None`, контекстное обогащение пропускается |
| Ошибка получения Neo4j контекста | Логируется (warning), элемент пропускается |
| Ошибка загрузки изображения | Логируется (warning), image_base64 остаётся null |
| Ошибка реранкера | Логируется (warning), используются оригинальные результаты поиска |
| Ошибка LLM | Логируется (error), llm_answer = текст ошибки |
| HTML-файл web-интерфейса не найден | 404 Not Found |

## Testing
1. **GET /ask-document**: возврат HTML-страницы (200), отсутствие файла (404)
2. **POST /ask-document с индексированным документом**: поиск возвращает релевантные answers
3. **POST с неиндексированным документом**: авто-индексация → успешный поиск
4. **POST с несуществующим file_hash**: возврат `status: "error"`
5. **POST с use_llm=true**: llm_answer не null, содержит осмысленный ответ
6. **POST с use_reranker=true**: answers содержат `reranker_score` и `original_score`
7. **POST с limit**: количество answers ≤ limit
8. **Граничный случай**: пустой content_list в MinerU result → 0 answers
9. **Граничный случай**: Neo4j недоступен → поиск работает без контекстного обогащения
10. **Граничный случай**: реранкер недоступен → fallback на оригинальные результаты

## Dependencies
- `qdrant_client` (QdrantClientWrapper) — поиск и проверка индексации
- `emb_client` (EmbeddingClient) — эмбеддинг вопроса
- `reranker_client` (RerankerClient) — переранжирование (опционально)
- `llm_client` (LLMClient) — генерация ответа (опционально)
- `minio_client` (MinioClient) — загрузка mineru_result и изображений
- `DocumentIndexService` (documet_index) — Neo4j контекстное обогащение (опционально)
- `pathlib.Path` — поиск HTML-файла
- `FastAPI.responses.FileResponse`, `HTMLResponse`

## Exceptions
- **P5 (Синхронные вызовы в async)**: Синхронный `emb_client.get_text_embedding()`, `qdrant_client.search()`, `reranker_client.rerank()`, `llm_client.send_message()`, `index_document_by_hash()`, `check_document_indexed()` — все вызываются синхронно внутри синхронного `def ask_document()` без `run_in_executor` (GET /ask-document — `async def`, POST /ask-document — `def`)
- **P6 (Dict вместо Pydantic)**: Поле `answers` в `QuestionResponse` имеет тип `List[Dict[str, Any]]` вместо строгой Pydantic-модели `Answer`. Поля `neo4j_context`, `parent_element` и т.д. — все `dict` без типизации.
- **P7 (Промпты в коде)**: System prompt и user prompt для LLM (строки 1589-1676 в api.py) определены как строковые константы, а не вынесены в `prompts/ask-document-system.md` и `prompts/ask-document-user.md`.
- **N2 (Структура модулей)**: Бизнес-логика Q&A (~450 строк) встроена в `api.py`, отсутствует `manager.py`.
- **P5/P1 (hardcoded URL)**: Не в данном эндпоинте, но связанная функция `check_document_indexed` использует dummy-вектор `[0.0]*2048`, что является хардкодом размерности.
- **P10 (Мягкое удаление)**: Не применимо к данному эндпоинту, но авто-индексация не учитывает флаг `archived` сущностей.
