# Feature: Управление коллекциями Qdrant

## Motivation
Предоставляет API для просмотра, создания и удаления коллекций Qdrant, а также получения списка файлов в конкретной коллекции. Необходим для управления векторным хранилищем из внешнего интерфейса.

## Behaviour

### Input

#### GET /collections
- Без параметров

#### POST /collections
- **Content-Type**: `application/json`
- **Тело**: `CollectionCreateRequest` (Pydantic)

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `collection_name` | `str` | обязательное | Имя коллекции |
| `vector_size` | `int` | 2048 | Размерность вектора |
| `distance` | `str` | "COSINE" | Метрика расстояния: COSINE, DOT, EUCLID |

#### DELETE /collections/{collection_name}
- **Path parameter**: `collection_name: str` — имя коллекции

#### GET /collections/{collection_name}/files
- **Path parameter**: `collection_name: str` — имя коллекции

### Processing

#### GET /collections
1. `get_qdrant_client()` → вызов `client.list_collections()`
2. Для каждой коллекции: `get_qdrant_client(collection_name=col_name)` → `client.client.get_collection(col_name)`
3. Извлечение `vectors_count` и `points_count` из коллекции
4. Возврат `CollectionsListResponse` со списком `CollectionInfo`

#### POST /collections
1. `get_qdrant_client(collection_name=request.collection_name)`
2. Проверка существования: `client.client.collection_exists(request.collection_name)`
3. Если существует — возврат `{"status": "already_exists", ...}` (HTTP 200)
4. Маппинг distance-строки в Qdrant Distance enum: `COSINE → COSINE, DOT → DOT, EUCLID → EUCLID`
5. `client.create_collection(vector_size, distance)`
6. При успехе — `{"status": "success", ...}`, при неудаче — 500

#### DELETE /collections/{collection_name}
1. `get_qdrant_client(collection_name=collection_name)`
2. Проверка существования: `client.client.collection_exists(collection_name)`
3. Если не существует — возврат `{"status": "not_found", ...}` (HTTP 200)
4. `client.delete_collection()`
5. При успехе — `{"status": "success", ...}`, при неудаче — 500

#### GET /collections/{collection_name}/files
1. `get_qdrant_client(collection_name=collection_name)`
2. Проверка существования коллекции: если нет — возврат `{"status": "success", "files": []}` (HTTP 200)
3. Scroll по всем точкам коллекции (limit=100, with_payload=True, with_vectors=False):
   - Извлечение `file_hash` из payload каждой точки
   - Дедупликация по `file_hash` (через `seen_hashes`)
   - Для каждого уникального `file_hash`: поиск в MinIO (`pdfs/{file_hash}_`) для получения имени файла и даты
   - Если PDF не найден в MinIO — `filename = "unknown_{file_hash[:8]}"`
4. Возврат `UploadedFilesListResponse` со списком `UploadedFileInfo`

### Output

#### GET /collections
- **200 OK**: `CollectionsListResponse`
  ```json
  {
    "status": "success",
    "message": "Found 3 collections",
    "collections": [
      {
        "name": "documents",
        "vectors_count": 150,
        "points_count": 150
      }
    ],
    "total_count": 3
  }
  ```
- **500 Internal Server Error**: ошибка получения списка

#### POST /collections
- **200 OK**: `Dict[str, Any]` (не Pydantic)
  ```json
  {
    "status": "success" | "already_exists",
    "message": "...",
    "collection_name": "...",
    "vector_size": 2048,
    "distance": "COSINE"
  }
  ```
- **500 Internal Server Error**: ошибка создания

#### DELETE /collections/{collection_name}
- **200 OK**: `Dict[str, Any]` (не Pydantic)
  ```json
  {
    "status": "success" | "not_found",
    "message": "...",
    "collection_name": "..."
  }
  ```
- **500 Internal Server Error**: ошибка удаления

#### GET /collections/{collection_name}/files
- **200 OK**: `UploadedFilesListResponse`
  ```json
  {
    "status": "success",
    "message": "Found 5 unique files in collection 'documents'",
    "files": [
      {
        "file_hash": "abc123",
        "filename": "report.pdf",
        "upload_date": "2026-01-01T00:00:00"
      }
    ],
    "total_count": 5
  }
  ```
- **500 Internal Server Error**: ошибка получения файлов

## API Contract

```openapi
get: /collections
summary: Get list of all Qdrant collections with their info
responses:
  '200':
    description: List of collections
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/CollectionsListResponse'
  '500':
    description: Error retrieving collections list

post: /collections
summary: Create a new Qdrant collection
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/CollectionCreateRequest'
responses:
  '200':
    description: Collection created or already exists
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [success, already_exists]
            message:
              type: string
            collection_name:
              type: string
            vector_size:
              type: integer
            distance:
              type: string
  '500':
    description: Error creating collection

delete: /collections/{collection_name}
summary: Delete a Qdrant collection
parameters:
  - name: collection_name
    in: path
    required: true
    schema:
      type: string
responses:
  '200':
    description: Collection deleted or not found
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [success, not_found]
            message:
              type: string
            collection_name:
              type: string
  '500':
    description: Error deleting collection

get: /collections/{collection_name}/files
summary: Get list of unique files (by file_hash) in a Qdrant collection
parameters:
  - name: collection_name
    in: path
    required: true
    schema:
      type: string
responses:
  '200':
    description: List of files in collection
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/UploadedFilesListResponse'
  '500':
    description: Error retrieving collection files
```

## Data Flow
```
Пользователь → GET /collections
  → Qdrant (direct driver): list_collections + get_collection

Пользователь → POST /collections (JSON: collection_name, vector_size, distance)
  → Qdrant (direct driver): collection_exists + create_collection

Пользователь → DELETE /collections/{collection_name}
  → Qdrant (direct driver): collection_exists + delete_collection

Пользователь → GET /collections/{collection_name}/files
  → Qdrant (direct driver): collection_exists + scroll (все точки)
  → MinIO: list_objects (pdfs/{file_hash}_) + stat_object (дата)
```

## LLM Interactions
- Вызовов LLM нет.

## LLM Model Requirements
- Не применимо.

## Error Handling
| Failure mode | Обработка |
|---|---|
| Qdrant недоступен (list_collections) | 500 Internal Server Error |
| Коллекция не найдена при get_collection | Логируется (warning), `CollectionInfo` без counts |
| Коллекция уже существует при создании | HTTP 200, `status: "already_exists"` |
| Неверный distance при создании | Маппинг через `.get(distance.upper(), COSINE)`, невалидные значения не вызывают ошибку |
| Ошибка создания коллекции | 500 Internal Server Error |
| Коллекция не существует при удалении | HTTP 200, `status: "not_found"` |
| Ошибка удаления коллекции | 500 Internal Server Error |
| Коллекция не существует при files | HTTP 200 с пустым списком файлов |
| Ошибка scroll | Возврат пустого списка (fallback) |
| PDF не найден в MinIO для file_hash | `filename = "unknown_{prefix}"`, `upload_date = "unknown"` |

## Testing
1. **GET /collections с существующими коллекциями**: возврат списка
2. **GET /collections с пустой БД**: возврат пустого списка
3. **POST /collections с валидными параметрами**: `status: "success"`
4. **POST /collections с уже существующим именем**: `status: "already_exists"`
5. **POST /collections с разными distance**: COSINE, DOT, EUCLID
6. **DELETE /collections/{name} существующей**: `status: "success"`
7. **DELETE /collections/{name} несуществующей**: `status: "not_found"`
8. **GET /collections/{name}/files с индексированными документами**: корректный список
9. **GET /collections/{name}/files с несуществующей коллекцией**: пустой список
10. **Граничный случай**: Qdrant недоступен → 500 на всех эндпоинтах

## Dependencies
- `qdrant_client` (QdrantClientWrapper) — все операции с коллекциями
- `minio_client` (MinioClient) — поиск PDF для `/collections/{name}/files`
- `qdrant_client.http.models.Distance` — enum для маппинга distance
- Модели: `CollectionCreateRequest`, `CollectionInfo`, `CollectionsListResponse`, `UploadedFileInfo`, `UploadedFilesListResponse`

## Exceptions
- **P6 (Dict вместо Pydantic)**: Ответы `POST /collections` и `DELETE /collections/{collection_name}` возвращают `Dict[str, Any]` (response_model=Dict[str, Any]) вместо строгих Pydantic-моделей.
- **P5 (Синхронные эндпоинты)**: Все эндпоинты объявлены как синхронные `def`, а не `async def`. Qdrant-клиент выполняет синхронные HTTP/gRPC вызовы, которые блокируют event loop.
- **N2 (Структура модулей)**: Логика встроена в `api.py`, отсутствует `manager.py`.
- **P6 (Структура моделей)**: Модели `CollectionCreateRequest`, `CollectionInfo`, `CollectionsListResponse`, `UploadedFileInfo`, `UploadedFilesListResponse` находятся в `app/src/utils/data_model.py`, а не в `app/src/dtype/` как предписано Constitution N2.
