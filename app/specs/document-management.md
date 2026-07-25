# Feature: Управление документами (CRUD)

## Motivation
Предоставляет эндпоинты для просмотра списка загруженных файлов, удаления отдельного документа со всеми связанными данными и полной очистки системы. Обеспечивает управление жизненным циклом документов.

## Behaviour

### Input

#### GET /uploaded-files
- Без параметров

#### DELETE /documents/{file_hash}
- **Path parameter**: `file_hash: str` — MD5-хеш документа для удаления

#### DELETE /documents/all
- Без параметров

### Processing

#### GET /uploaded-files
1. `minio_client.list_objects(bucket, prefix="pdfs/")` — список всех PDF-объектов в MinIO
2. Для каждого объекта:
   - Парсинг пути: `pdfs/{dir_name}/{file_name}` или `pdfs/{dir_name}`
   - Извлечение `file_hash` из `dir_name` (формат `{hash}_{filename}`) — первая часть до `_`
   - Дедупликация по `file_hash` (через `seen_hashes`)
   - Получение даты загрузки: `minio_client.client.stat_object(bucket, pdf_path).last_modified`
3. Возврат `UploadedFilesListResponse` со списком `UploadedFileInfo`
   - **Внимание**: в коде используется `file_name=` (не `filename=`) и `s3_path=` как параметры конструктора `UploadedFileInfo`, хотя модель определяет только `filename` и не имеет поля `s3_path`. Это может приводить к ошибкам валидации Pydantic в рантайме.

#### DELETE /documents/{file_hash}
1. **Удаление из Qdrant**: `qdrant_client.delete_points_by_file_hash(file_hash)` — удаление всех точек с данным `file_hash`
2. **Удаление из Neo4j** (опционально, если `NEO4J_AVAILABLE`):
   - `DocumentIndexService()` → `neo4j_service.delete_graph(file_hash)` → `neo4j_service.close()`
   - При ошибке: `results["neo4j_deleted"] = False`, операция продолжается
   - Если Neo4j недоступен: `results["neo4j_deleted"] = None`
3. **Удаление из MinIO**:
   - PDF: `list_objects(prefix=f"pdfs/{file_hash}")` → `remove_object()` для каждого
   - MinerU результаты: `list_objects(prefix=f"mineru_results/{file_hash}")` → `remove_object()`
   - Эмбеддинги: `list_objects(prefix=f"embeddings/{file_hash}")` → `remove_object()`
   - Счётчик `minio_files_removed`
4. **Определение общего статуса**:
   - `all_success = qdrant_deleted AND (neo4j_deleted is None OR neo4j_deleted) AND minio_deleted`
   - `status = "success"` если все успешно, иначе `"partial"`
5. Возврат `Dict[str, Any]` с результатами по каждому хранилищу

#### DELETE /documents/all
1. **Удаление всего из Qdrant**: `qdrant_client.delete_all_points()`
2. **Удаление всего из Neo4j** (опционально):
   - `DocumentIndexService()` → `neo4j_service.delete_all_graphs()` → `neo4j_service.close()`
   - При ошибке: `results["neo4j_deleted"] = False`
3. **Удаление всего из MinIO**:
   - `list_objects(bucket)` — список ВСЕХ объектов
   - Фильтрация только по managed-префиксам: `pdfs/`, `mineru_results/`, `embeddings/`
   - `remove_object()` для каждого
4. Определение общего статуса (аналогично одиночному удалению)
5. Возврат `Dict[str, Any]` с предупреждением: `"warning": "This operation deleted ALL documents from the system"`

### Output

#### GET /uploaded-files
- **200 OK**: `UploadedFilesListResponse`
  ```json
  {
    "status": "success",
    "files": [
      {
        "file_name": "report.pdf",
        "file_hash": "abc123",
        "s3_path": "pdfs/abc123_report.pdf/report.pdf",
        "upload_date": "2026-01-01T00:00:00"
      }
    ]
  }
  ```
- **500 Internal Server Error**: ошибка получения списка

#### DELETE /documents/{file_hash}
- **200 OK**: `Dict[str, Any]`
  ```json
  {
    "file_hash": "abc123",
    "status": "success" | "partial",
    "qdrant_deleted": true,
    "neo4j_deleted": true,
    "minio_deleted": true,
    "minio_files_removed": 5
  }
  ```
- **500 Internal Server Error**: ошибка удаления

#### DELETE /documents/all
- **200 OK**: `Dict[str, Any]`
  ```json
  {
    "status": "success" | "partial",
    "qdrant_deleted": true,
    "neo4j_deleted": true,
    "minio_deleted": true,
    "minio_files_removed": 42,
    "warning": "This operation deleted ALL documents from the system"
  }
  ```
- **500 Internal Server Error**: ошибка удаления

## API Contract

```openapi
get: /uploaded-files
summary: Get list of all uploaded PDF files with their hashes
responses:
  '200':
    description: List of uploaded files from MinIO
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/UploadedFilesListResponse'
  '500':
    description: Error retrieving file list

delete: /documents/{file_hash}
summary: Delete a document and all related data from Qdrant, Neo4j, and MinIO
parameters:
  - name: file_hash
    in: path
    required: true
    schema:
      type: string
responses:
  '200':
    description: Document deletion results (per storage)
    content:
      application/json:
        schema:
          type: object
          properties:
            file_hash:
              type: string
            status:
              type: string
              enum: [success, partial]
            qdrant_deleted:
              type: boolean
            neo4j_deleted:
              type: boolean
            minio_deleted:
              type: boolean
            minio_files_removed:
              type: integer
  '500':
    description: Error deleting document

delete: /documents/all
summary: Delete ALL documents from Qdrant, Neo4j, and MinIO (destructive operation)
responses:
  '200':
    description: All documents deletion results
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [success, partial]
            qdrant_deleted:
              type: boolean
            neo4j_deleted:
              type: boolean
            minio_deleted:
              type: boolean
            minio_files_removed:
              type: integer
            warning:
              type: string
  '500':
    description: Error deleting all documents
```

## Data Flow

```
GET /uploaded-files:
  → MinIO: list_objects(pdfs/) → парсинг путей → stat_object (дата)

DELETE /documents/{file_hash}:
  → Qdrant: delete_points_by_file_hash(file_hash)
  → Neo4j (documet_index): delete_graph(file_hash) [опционально]
  → MinIO: удаление pdfs/{file_hash}* + mineru_results/{file_hash}* + embeddings/{file_hash}*

DELETE /documents/all:
  → Qdrant: delete_all_points()
  → Neo4j (documet_index): delete_all_graphs() [опционально]
  → MinIO: удаление всех объектов с префиксами pdfs/, mineru_results/, embeddings/
```

## LLM Interactions
- Вызовов LLM нет.

## LLM Model Requirements
- Не применимо.

## Error Handling
| Failure mode | Обработка |
|---|---|
| Ошибка MinIO при list_objects | 500 Internal Server Error |
| Ошибка stat_object для конкретного файла | `upload_date = "unknown"`, файл включается в список |
| Ошибка удаления из Qdrant | `qdrant_deleted = False`, операция продолжается (partial) |
| Ошибка удаления из Neo4j | `neo4j_deleted = False`, операция продолжается (partial) |
| Neo4j недоступен | `neo4j_deleted = None`, учитывается как успех |
| Ошибка удаления отдельных файлов MinIO | Логируется (warning), счётчик уменьшается, операция продолжается |
| Ошибка удаления всех объектов MinIO | `minio_deleted = False` |
| Файл не имеет формата `{hash}_{name}` при парсинге | `file_hash = "unknown"` |
| Неверные параметры конструктора UploadedFileInfo | Pydantic validation error (потенциальный баг: `file_name` и `s3_path` передаются, но не определены в модели) |

## Testing
1. **GET /uploaded-files с загруженными PDF**: корректный список с хешами и датами
2. **GET /uploaded-files с пустым MinIO**: пустой список (статус success)
3. **DELETE /documents/{file_hash} существующего**: все три хранилища очищены, status "success"
4. **DELETE /documents/{file_hash} несуществующего**: MinIO удаляет 0 файлов, Qdrant удаляет 0 точек
5. **DELETE /documents/{file_hash} с недоступным Neo4j**: `neo4j_deleted = None`, остальное очищено
6. **DELETE /documents/{file_hash} с недоступным Qdrant**: `qdrant_deleted = False`, status "partial"
7. **DELETE /documents/all**: полная очистка всех хранилищ, warning в ответе
8. **DELETE /documents/all с пустой системой**: minio_files_removed = 0
9. **Граничный случай**: частичный сбой MinIO (некоторые файлы не удаляются) → status "partial", счётчик меньше ожидаемого
10. **Граничный случай**: MinIO содержит объекты вне managed-префиксов → они не удаляются при DELETE /documents/all

## Dependencies
- `minio_client` (MinioClient) — list_objects, stat_object, remove_object
- `qdrant_client` (QdrantClientWrapper) — delete_points_by_file_hash, delete_all_points
- `DocumentIndexService` (documet_index) — delete_graph, delete_all_graphs (опционально)
- Модели: `UploadedFileInfo`, `UploadedFilesListResponse`

## Exceptions
- **P6 (Dict вместо Pydantic)**: Ответы `DELETE /documents/{file_hash}` и `DELETE /documents/all` используют `response_model=Dict[str, Any]` вместо строгой Pydantic-модели `DocumentDeleteResponse`.
- **P5 (Синхронные эндпоинты)**: Все эндпоинты объявлены как синхронные `def`, а не `async def`. Операции MinIO и Qdrant выполняются синхронно, блокируя event loop.
- **P5 (Управление ресурсами Neo4j)**: `DocumentIndexService` создаётся и закрывается внутри try/except, но если `delete_graph` выбрасывает исключение, `close()` не вызывается (нет finally-блока).
- **P10 (Мягкое удаление)**: Удаление из Neo4j выполняется через `delete_graph()` (физическое удаление узлов Document/Region), что противоречит принципу мягкого удаления (P10). Однако это относится к document graph, а P10 регулирует semantic graph (Entity, Community), что допустимо.
- **P6 (Модели данных)**: `UploadedFileInfo` в коде `get_uploaded_files` инстанциируется с полями `file_name` и `s3_path`, которых нет в Pydantic-модели (модель имеет `filename`, `file_hash`, `upload_date`, `file_size`, `status`). Это ошибка несоответствия кода модели.
- **N2 (Структура модулей)**: Логика встроена в `api.py`.
