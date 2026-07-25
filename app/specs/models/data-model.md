# Data Model: Core Application Models

## Purpose
Центральные модели данных приложения `app`, используемые в API-эндпоинтах для:
- Управления коллекциями Qdrant (создание, просмотр, удаление)
- Вопросно-ответного поиска по документам (вопрос → результаты → LLM-ответ)
- Инвентаризации загруженных файлов (список, метаданные)

Модели живут в `app/src/utils/data_model.py` и импортируются непосредственно в `api.py`.

## Schema

```python
from typing import Dict, List, Any, Optional
from pydantic import BaseModel


# ====== Collection Management ======

class CollectionCreateRequest(BaseModel):
    """Запрос на создание коллекции Qdrant."""
    collection_name: str                          # Имя коллекции
    vector_size: int = 2048                       # Размерность вектора (по умолчанию 2048)
    distance: str = "COSINE"                      # Метрика расстояния: COSINE, DOT, EUCLID


class CollectionInfo(BaseModel):
    """Информация об одной коллекции Qdrant."""
    name: str                                     # Имя коллекции
    vectors_count: Optional[int] = None            # Количество векторов в коллекции
    points_count: Optional[int] = None             # Количество точек в коллекции


class CollectionsListResponse(BaseModel):
    """Ответ со списком всех коллекций Qdrant."""
    status: str                                   # Статус операции ("success" / "error")
    message: str                                  # Сообщение
    collections: List[CollectionInfo]              # Список коллекций
    total_count: int = 0                          # Общее количество коллекций


# ====== Question Answering ======

class QuestionRequest(BaseModel):
    """Запрос на поиск ответа по документу."""
    file_hash: str                                # MD5-хеш документа
    question: str                                 # Текст вопроса
    limit: int = 10                               # Максимальное количество результатов поиска
    collection_name: Optional[str] = None          # Имя коллекции Qdrant (если None — используется default)
    use_llm: bool = False                         # Генерировать ли ответ LLM на основе контекста
    use_reranker: bool = False                    # Применять ли реранкер к результатам поиска


class QuestionResponse(BaseModel):
    """Ответ на вопрос по документу."""
    status: str                                   # Статус операции ("success" / "error")
    message: str                                  # Сообщение
    file_hash: str                                # MD5-хеш документа
    question: str                                 # Исходный вопрос
    answers: List[Dict[str, Any]]                 # Список результатов поиска (Qdrant hits)
    indexed: bool                                 # Проиндексирован ли документ
    collection_name: Optional[str] = None          # Имя использованной коллекции
    llm_answer: Optional[str] = None               # LLM-сгенерированный ответ (если use_llm=True)


# ====== File Inventory ======

class UploadedFileInfo(BaseModel):
    """Информация о загруженном файле."""
    file_hash: str                                # MD5-хеш содержимого файла
    filename: str                                 # Исходное имя файла
    upload_date: Optional[str] = None              # Дата загрузки (ISO-формат)
    file_size: Optional[int] = None                # Размер файла в байтах
    status: Optional[str] = None                   # Статус обработки


class UploadedFilesListResponse(BaseModel):
    """Ответ со списком загруженных файлов."""
    status: str                                   # Статус операции ("success" / "error")
    message: str                                  # Сообщение
    files: List[UploadedFileInfo]                  # Список файлов
    total_count: int = 0                          # Общее количество файлов
```

## Storage
- **Хранение**: in-memory / HTTP transport only. Модели являются DTO (Data Transfer Objects) — сериализуются в JSON для HTTP-ответов и десериализуются из JSON в теле HTTP-запросов.
- Данные, представляемые этими моделями, извлекаются из:
  - **Qdrant** — количество точек, имена коллекций (`CollectionInfo`, `CollectionsListResponse`)
  - **MinIO** — список файлов с префиксом `pdfs/`, метаданные объектов (`UploadedFileInfo`, `UploadedFilesListResponse`)
  - **Qdrant search results** — результаты векторного поиска (`QuestionResponse.answers`)

## Relationships
- **`CollectionCreateRequest`** → эндпоинт `POST /collections` (`api.py`) → передаётся в `qdrant_client_api.py` для `create_collection()`
- **`CollectionInfo`** → часть `CollectionsListResponse`; заполняется из `qdrant_client.get_collections()`
- **`CollectionsListResponse`** → эндпоинт `GET /collections` (`api.py`) — возвращается как JSON-ответ
- **`QuestionRequest`** → эндпоинт `POST /ask-document` (`api.py`) — десериализуется из тела запроса (JSON)
- **`QuestionResponse`** → эндпоинт `POST /ask-document` (`api.py`) — сериализуется в JSON-ответ
- **`UploadedFileInfo`** → часть `UploadedFilesListResponse`; заполняется из `minio_client.list_objects(prefix="pdfs/")`
- **`UploadedFilesListResponse`** → эндпоинт `GET /uploaded-files` (`api.py`) — возвращается как JSON-ответ
- **Связь с `reranker` моделями**: результаты Qdrant-поиска (`QuestionResponse.answers`) преобразуются в `reranker.Message` список для реранкинга (когда `use_reranker=True`)
- **Связь с `LLMClient`**: `QuestionResponse.answers` передаются как контекст в LLM для генерации `llm_answer`

## Constraints
- **`CollectionCreateRequest.collection_name`** — обязательное поле, не валидируется на допустимые символы (допустимость имени определяется Qdrant)
- **`CollectionCreateRequest.vector_size`** — должно соответствовать размерности эмбеддингов (фактически 2048); не валидируется на соответствие модели эмбеддингов
- **`CollectionCreateRequest.distance`** — допустимые значения Qdrant: `"COSINE"`, `"DOT"`, `"EUCLID"`; не валидируется на уровне Pydantic (нет Literal/Enum), проверяется Qdrant при создании коллекции
- **`QuestionRequest.file_hash`** — обязательное поле; не валидируется формат MD5-хеша (32 hex символа)
- **`QuestionRequest.limit`** — целое число, по умолчанию 10; передаётся в Qdrant `search()` как `limit`
- **`QuestionRequest.use_llm`** и **`use_reranker`** — булевы флаги, определяющие ветвление логики в эндпоинте `/ask-document`
- **`QuestionResponse.answers`** — `List[Dict[str, Any]]` — использование `Dict[str, Any]` вместо строгой Pydantic-модели (ослабленная типизация)
- **`QuestionResponse.indexed`** — обязательное поле; `True` если документ найден в Qdrant
- **`UploadedFileInfo.file_hash`** — обязательное, извлекается из имени объекта MinIO (первый сегмент пути `pdfs/{file_hash}_*/`)
- **`UploadedFileInfo.upload_date`**, **`file_size`**, **`status`** — все опциональны (`Optional[...]`), могут отсутствовать в ответе

### Граничные случаи
- **Пустая коллекция Qdrant**: `points_count` и `vectors_count` будут `None` или `0`
- **Несуществующий `file_hash`** в `QuestionRequest`: поиск в Qdrant вернёт пустой список, `indexed=False`
- **Одновременное `use_llm=True` без результатов поиска**: LLM-запрос с пустым контекстом (поведение зависит от LLM)
- **Коллекция с нестандартной размерностью**: `CollectionCreateRequest(vector_size=128)` — Qdrant создаст с размерностью 128, но эмбеддинги будут 2048-dim → несоответствие при вставке
- **Не-ASCII символы в `collection_name`**: не валидируются, поведение зависит от Qdrant
- **Отсутствующие файлы в MinIO**: `GET /uploaded-files` вернёт пустой список с `total_count=0`

## Examples

### Создание коллекции Qdrant
```python
# Запрос
req = CollectionCreateRequest(
    collection_name="documents",
    vector_size=2048,
    distance="COSINE"
)
# Валидный JSON-ответ от POST /collections:
# {"status": "success", "collection_name": "documents"}
```

### Ответ со списком коллекций
```python
response = CollectionsListResponse(
    status="success",
    message="Found 2 collections",
    collections=[
        CollectionInfo(name="documents", vectors_count=150, points_count=150),
        CollectionInfo(name="test_collection", vectors_count=0, points_count=0)
    ],
    total_count=2
)
```

### Запрос вопроса к документу
```python
# JSON-тело запроса POST /ask-document:
req = QuestionRequest(
    file_hash="a1b2c3d4e5f6...",
    question="Какой рост экономики прогнозируется?",
    limit=5,
    collection_name="documents",
    use_llm=True,
    use_reranker=True
)
```

### Ответ на вопрос
```python
response = QuestionResponse(
    status="success",
    message="Found 5 relevant passages",
    file_hash="a1b2c3d4e5f6...",
    question="Какой рост экономики прогнозируется?",
    answers=[
        {"id": 42, "score": 0.95, "payload": {"text": "Экономика вырастет на 3.2%", "region_id": "r001"}},
        {"id": 17, "score": 0.89, "payload": {"text": "Прогноз роста ВВП: 3-4%", "region_id": "r002"}}
    ],
    indexed=True,
    collection_name="documents",
    llm_answer="Согласно документу, экономика прогнозируется вырасти на 3.2% в текущем году."
)
```

### Список загруженных файлов
```python
response = UploadedFilesListResponse(
    status="success",
    message="Found 3 files",
    files=[
        UploadedFileInfo(
            file_hash="a1b2c3d4",
            filename="report.pdf",
            upload_date="2026-07-22",
            file_size=1048576,
            status="processed"
        ),
        UploadedFileInfo(
            file_hash="e5f6g7h8",
            filename="manual.pdf",
            upload_date="2026-07-21",
            file_size=524288,
            status="processed"
        )
    ],
    total_count=3
)
```

### Невалидный QuestionRequest
```python
# Вызовет ValidationError — отсутствует file_hash:
QuestionRequest(question="Как дела?")
# → file_hash: Field required
```

## Exceptions
- **Отклонение от Constitution P6 (размещение моделей)**: Модели расположены в `app/src/utils/data_model.py`, а не в `dtype/` как предписано структурой N2. Каталог `dtype/` отсутствует в сервисе `app`.
- **Отклонение от Constitution P6 (слабые контракты)**: Поле `QuestionResponse.answers` типизировано как `List[Dict[str, Any]]` вместо строгой Pydantic-модели (например, `SearchHit` с полями `id`, `score`, `payload`, `vector`). Это ослабляет контракт между API и клиентом. Аналогично, ответы эндпоинтов `POST /collections`, `DELETE /collections/{collection_name}`, `DELETE /documents/{file_hash}`, `DELETE /documents/all` возвращают `Dict[str, Any]` вместо Pydantic-моделей.
- **Отклонение от Constitution P5 (синхронность)**: Эндпоинты `GET /uploaded-files`, `GET /collections`, `POST/DELETE /collections`, `DELETE /documents` объявлены как синхронные `def`, а не `async def`.
