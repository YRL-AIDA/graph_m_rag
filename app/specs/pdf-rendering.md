# Feature: Рендеринг PDF-страниц и Bounding Boxes

## Motivation
Предоставляет эндпоинты для визуализации загруженного PDF-документа: получение оригинального PDF, рендеринг страниц в PNG с опциональной подсветкой bounding boxes (из результатов MinerU), получение метаданных и списка bbox-ов с цветовой разметкой по типам элементов.

## Behaviour

### Input

#### GET /pdf/{file_hash}
- **Path parameter**: `file_hash: str` — MD5-хеш PDF

#### GET /api/pdf/{file_hash}/info
- **Path parameter**: `file_hash: str` — MD5-хеш PDF

#### GET /api/pdf/{file_hash}/page/{page_number}
- **Path parameter**: `file_hash: str` — MD5-хеш PDF
- **Path parameter**: `page_number: int` — номер страницы (0-indexed)
- **Query parameter** (опционально): `bboxes: Optional[str]` — JSON-строка массива bbox-объектов для подсветки
  ```json
  [{"bbox": [x1, y1, x2, y2], "color": "#FF0000", "label": "text"}]
  ```

#### GET /api/pdf/{file_hash}/mineru-bboxes
- **Path parameter**: `file_hash: str` — MD5-хеш PDF
- **Query parameter** (опционально): `page_idx: Optional[int]` — фильтр по индексу страницы

### Processing

#### GET /pdf/{file_hash}
1. Поиск в MinIO: `list_objects(prefix=f"pdfs/{file_hash}")` — берётся первый совпавший объект
2. Если не найден — 404 Not Found
3. Загрузка PDF из MinIO: `get_object(bucket, pdf_path)` → bytes
4. Возврат `Response(content=pdf_data, media_type="application/pdf")` с `Content-Disposition: inline`

#### GET /api/pdf/{file_hash}/info
1. Поиск PDF в MinIO (`pdfs/{file_hash}`)
2. Если не найден — 404 Not Found
3. `stat_object(bucket, pdf_path)` → размер, дата изменения
4. Возврат JSON: `{status: "success", file_hash, file_name, s3_path, size, last_modified}`

#### GET /api/pdf/{file_hash}/page/{page_number}
1. Поиск PDF в MinIO (`pdfs/{file_hash}`), 404 если не найден
2. Загрузка PDF bytes из MinIO
3. Открытие через PyMuPDF: `fitz.open(stream=pdf_data, filetype="pdf")`
4. Валидация page_number: `0 ≤ page_number < len(doc)`, иначе 404
5. Рендеринг страницы: `page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))` → 2x zoom
6. Конвертация в PIL Image: `Image.frombytes("RGB", ...)`
7. **Отрисовка bbox-ов** (если передан `bboxes` параметр):
   - Парсинг JSON из query-параметра, игнорирование `JSONDecodeError`
   - Для каждого bbox:
     - Конвертация координат: MinerU использует нормализованные координаты (0-1000) → перевод в координаты страницы PDF (`scale_x = page_width / 1000`) → масштабирование под разрешение рендера (`img_scale = pix.width / page_width`)
     - Отрисовка прямоугольника: `draw.rectangle([x1, y1, x2, y2], outline=color, width=3)`
     - Полупрозрачная заливка: `Image.alpha_composite(img, overlay)` с `fill_color = (r, g, b, 50)`
     - Лейбл (если есть): текст над bbox на тёмном фоне
   - Цвет по умолчанию: `#FF0000` (красный)
8. Сохранение в PNG bytes
9. Закрытие PDF-документа: `doc.close()` (обычный вызов, не в finally — может не закрыться при ошибке рендеринга)
10. Возврат `Response(content=img_bytes, media_type="image/png")` с `Content-Disposition: inline`

#### GET /api/pdf/{file_hash}/mineru-bboxes
1. Поиск mineru_result в MinIO: `list_objects(prefix="mineru_results/")`, фильтр по `f"mineru_results/{file_hash}_"`, первый совпавший
2. Если не найден — 404 Not Found
3. Загрузка mineru_result JSON из MinIO
4. Извлечение элементов: `mineru_result["results"]["result"]["results"]["content_list"]`
5. Фильтрация по `page_idx` (если параметр передан): `elements = [e for e in elements if e.get("page_idx") == page_idx]`
6. Формирование bbox-списка:
   - Пропуск элементов без bbox или с некорректным bbox (не 4 координаты)
   - Определение типа элемента:
     - `text` с `text_level == 1` → `title` (фиолетовый `#9b59b6`)
     - `text` → `text` (синий `#3498db`)
     - `image` → `image` (зелёный `#27ae60`) + отдельные bbox для `image_caption` (`#2ecc71`) и `image_footnote` (`#1abc9c`)
     - `table` → `table` (оранжевый `#e67e22`) + отдельные bbox для `table_caption` (`#f39c12`) и `table_footnote` (`#d35400`)
     - `equation` → `equation` (красный `#e74c3c`)
     - `discarded` → `discarded` (серый `#95a5a6`)
   - Текстовый превью: первые 100 символов с эмодзи-префиксами (📑, 🖼️, 📷, 📝, 📊, 📋, ∫, 🗑️)
7. Возврат JSON: `{status: "success", file_hash, page_idx, total_elements, bboxes: [...]}`

### Output

#### GET /pdf/{file_hash}
- **200 OK**: `application/pdf` — содержимое PDF (inline)
- **404 Not Found**: PDF с данным хешем не найден
- **500 Internal Server Error**: ошибка загрузки PDF

#### GET /api/pdf/{file_hash}/info
- **200 OK**:
  ```json
  {
    "status": "success",
    "file_hash": "abc123",
    "file_name": "report.pdf",
    "s3_path": "pdfs/abc123_report.pdf/report.pdf",
    "size": 1048576,
    "last_modified": "2026-01-01T00:00:00"
  }
  ```
- **404 Not Found**: PDF не найден
- **500 Internal Server Error**: ошибка получения метаданных

#### GET /api/pdf/{file_hash}/page/{page_number}
- **200 OK**: `image/png` — рендер страницы (inline)
- **404 Not Found**: PDF не найден или страница вне диапазона
- **500 Internal Server Error**: ошибка рендеринга

#### GET /api/pdf/{file_hash}/mineru-bboxes
- **200 OK**:
  ```json
  {
    "status": "success",
    "file_hash": "abc123",
    "page_idx": 0,
    "total_elements": 15,
    "bboxes": [
      {
        "element_index": 0,
        "element_type": "title",
        "bbox": [100, 50, 900, 100],
        "page_idx": 0,
        "color": "#9b59b6",
        "label": "title_0",
        "text_preview": "📑 Title: Introduction to GraphRAG",
        "is_title": true
      }
    ]
  }
  ```
- **404 Not Found**: MinerU результат не найден
- **500 Internal Server Error**: ошибка получения bbox-ов

## API Contract

```openapi
get: /pdf/{file_hash}
summary: Get PDF file by file_hash for inline browser viewing
parameters:
  - name: file_hash
    in: path
    required: true
    schema:
      type: string
responses:
  '200':
    description: PDF file content
    content:
      application/pdf:
        schema:
          type: string
          format: binary
  '404':
    description: PDF file not found
  '500':
    description: Error retrieving PDF

get: /api/pdf/{file_hash}/info
summary: Get PDF file metadata (name, size, last modified)
parameters:
  - name: file_hash
    in: path
    required: true
    schema:
      type: string
responses:
  '200':
    description: PDF metadata
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
            file_hash:
              type: string
            file_name:
              type: string
            s3_path:
              type: string
            size:
              type: integer
            last_modified:
              type: string
  '404':
    description: PDF file not found
  '500':
    description: Error getting PDF info

get: /api/pdf/{file_hash}/page/{page_number}
summary: Render PDF page as PNG image with optional bbox highlights
parameters:
  - name: file_hash
    in: path
    required: true
    schema:
      type: string
  - name: page_number
    in: path
    required: true
    schema:
      type: integer
  - name: bboxes
    in: query
    required: false
    schema:
      type: string
    description: JSON array of bboxes to highlight [{"bbox":[x1,y1,x2,y2],"color":"#FF0000","label":"text"}]
responses:
  '200':
    description: PNG image of the rendered page
    content:
      image/png:
        schema:
          type: string
          format: binary
  '404':
    description: PDF not found or page out of range
  '500':
    description: Error rendering page

get: /api/pdf/{file_hash}/mineru-bboxes
summary: Get all bounding boxes from MinerU results with color-coded element types
parameters:
  - name: file_hash
    in: path
    required: true
    schema:
      type: string
  - name: page_idx
    in: query
    required: false
    schema:
      type: integer
responses:
  '200':
    description: Bounding boxes from MinerU
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
            file_hash:
              type: string
            page_idx:
              type: integer
            total_elements:
              type: integer
            bboxes:
              type: array
              items:
                type: object
  '404':
    description: MinerU result not found
  '500':
    description: Error getting bboxes
```

## Data Flow
```
Пользователь → GET /pdf/{file_hash}
  → MinIO: list_objects(pdfs/{file_hash}) + get_object → bytes → PDF response

Пользователь → GET /api/pdf/{file_hash}/info
  → MinIO: list_objects + stat_object → file size, last_modified

Пользователь → GET /api/pdf/{file_hash}/page/{page_number}?bboxes=[...]
  → MinIO: get_object(pdf) → bytes
  → PyMuPDF (fitz): отрисовка страницы с matrix(2.0, 2.0)
  → PIL (Pillow): наложение bbox-подсветки
  → PNG response

Пользователь → GET /api/pdf/{file_hash}/mineru-bboxes?page_idx=0
  → MinIO: list_objects(mineru_results/) + get_object → JSON
  → Извлечение content_list, фильтрация по page_idx
  → Форматирование bbox с цветами и preview
```

## LLM Interactions
- Вызовов LLM нет.

## LLM Model Requirements
- Не применимо.

## Error Handling
| Failure mode | Обработка |
|---|---|
| PDF не найден в MinIO | 404 Not Found |
| Страница вне диапазона (page_number < 0 или ≥ len(doc)) | 404 Not Found |
| Невалидный JSON в параметре bboxes | Логируется (warning), bbox-подсветка пропускается |
| MinerU результат не найден | 404 Not Found |
| Элемент без bbox или некорректный bbox | Пропуск элемента |
| Ошибка MinIO при загрузке | 500 Internal Server Error |
| Ошибка PyMuPDF при открытии PDF | 500 Internal Server Error |
| Ошибка PIL при отрисовке | 500 Internal Server Error |
| doc.close() не вызван при ошибке рендеринга | Потенциальная утечка памяти (не в finally-блоке) |

## Testing
1. **GET /pdf/{file_hash} с существующим PDF**: возврат PDF (200, application/pdf)
2. **GET /pdf/{file_hash} с несуществующим**: 404
3. **GET /api/pdf/{file_hash}/info**: корректные size и last_modified
4. **GET /api/pdf/{file_hash}/page/0**: PNG-изображение первой страницы (200)
5. **GET /api/pdf/{file_hash}/page/999** (за пределами): 404 с сообщением о количестве страниц
6. **GET /api/pdf/{file_hash}/page/0?bboxes=[...]**: PNG с корректно нарисованными прямоугольниками
7. **GET /api/pdf/{file_hash}/page/0?bboxes=invalid**: PNG без bbox (JSON не распарсился)
8. **GET /api/pdf/{file_hash}/mineru-bboxes**: список bbox с корректными цветами по типам
9. **GET /api/pdf/{file_hash}/mineru-bboxes?page_idx=0**: фильтрация по странице
10. **GET /api/pdf/{file_hash}/mineru-bboxes для несуществующего**: 404
11. **Граничный случай**: PDF без mineru_result → /mineru-bboxes возвращает 404
12. **Граничный случай**: изображение с caption и footnote → отдельные bbox-записи для каждого

## Dependencies
- `minio_client` (MinioClient) — все операции с объектами
- `fitz` (PyMuPDF) — рендеринг PDF страниц
- `PIL` (Pillow: Image, ImageDraw) — отрисовка bbox-подсветки
- `io`, `json`, `base64`
- `fastapi.responses.Response`

## Exceptions
- **P5 (Синхронные эндпоинты)**: Все эндпоинты объявлены как `async def`, но внутри синхронные вызовы MinIO, PyMuPDF, PIL — без `run_in_executor` (кроме загрузки PDF из MinIO, которая блокирует event loop на время скачивания).
- **P5 (Управление ресурсами)**: `doc.close()` для PDF-документа вызывается в обычном потоке управления, а не в `finally`-блоке. При ошибке рендеринга между открытием и закрытием документ может остаться открытым (утечка памяти).
- **P6 (Dict вместо Pydantic)**: Ответы `/api/pdf/{file_hash}/info` и `/api/pdf/{file_hash}/mineru-bboxes` возвращают dict без response_model.
- **N2 (Структура модулей)**: Логика рендеринга встроена в `api.py` (около 250 строк), отсутствует `manager.py`.
- **Hardcoded значения**: `fitz.Matrix(2.0, 2.0)` (2x zoom), координатный масштаб `1000.0` (нормализация MinerU) — хардкод в коде, не вынесены в настройки.
