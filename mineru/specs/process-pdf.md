# Feature: PDF Processing

## Motivation
Обработка PDF-документов и изображений через MinerU SDK — извлечение структурированного контента (текст, изображения, таблицы, формулы) с использованием OCR (pipeline) и VLM-доразметки. Это центральный эндпоинт сервиса `mineru`, обеспечивающий мультимодальный разбор документов для всей системы GraphRAG.

## Behaviour

### Input
**POST /process** — `multipart/form-data`.

| Параметр | Тип | Часть | По умолчанию | Описание |
|----------|-----|-------|-------------|----------|
| `file` | `UploadFile` | File (body) | — (обязательный) | PDF-файл или изображение для обработки |
| `backend` | `str` | Query | `"vlm"` | Бэкенд обработки: `"pipeline"` (OCR) или `"vlm"` (Vision-Language Model) |
| `method` | `str` | Query | `"auto"` | Метод распознавания: `"auto"`, `"txt"`, `"ocr"` (только для `backend=pipeline`) |
| `lang` | `str` | Query | `"ru"` | Язык документа |
| `formula_enable` | `bool` | Query | `True` | Включить распознавание формул (только для `backend=pipeline`) |
| `table_enable` | `bool` | Query | `True` | Включить распознавание таблиц (только для `backend=pipeline`) |
| `start_page` | `int` | Query | `0` | Начальная страница диапазона обработки (0-indexed) |
| `end_page` | `Optional[int]` | Query | `None` | Конечная страница (0-indexed, включительно). `None` — до последней страницы |

**Поддерживаемые форматы файлов**: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

Код проверяет расширение регистронезависимо (`Path.suffix.lower()`) и выбрасывает 400 на неподдерживаемый формат.

### Processing

Пошаговый алгоритм обработки запроса (`api.py:process_document`):

1. **Генерация task_id** — `uuid.uuid4()` для идентификации запроса (используется только как префикс временной директории, результат возвращается синхронно).

2. **Валидация формата файла** — проверка расширения файла (`Path(file.filename).suffix.lower()`) на вхождение в `allowed_extensions`. При несоответствии → `400 Bad Request`.

3. **Сохранение файла во временную директорию** — `tempfile.mkdtemp(prefix=f"mineru_{task_id}_")`, файл сохраняется через `shutil.copyfileobj`. При ошибке записи → `500`, временная директория удаляется.

4. **Чтение сохранённого файла в байты** — `open(file_path, "rb").read()` загружает весь файл в `bytes`. При ошибке чтения → `500`, временная директория удаляется.

5. **Ветвление по типу файла**:
   - **PDF** (`.pdf`): вызов `manager.process_pdf(file_bytes, config, temp_dir)` → см. *Pipeline-обработка PDF*.
   - **Изображение** (все остальные): вызов `manager.process_image(file_bytes)` → см. *Обработка изображения*.

6. **Проверка результата** — если `result["success"] == True` → `status = "completed"`, `message = "Документ обработан"`. Иначе → `status = "failed"`, `message = "Ошибка при обработке документа"`.

7. **Очистка временной директории** — `shutil.rmtree(temp_dir)` выполняется всегда (в блоке `finally`-стиля после всех try/except), ошибки очистки подавляются (`except: pass`).

8. **Формирование ответа** — `ProcessResponse(status, message, results={"result": result} if results else None)`.

#### Pipeline-обработка PDF (`manager.process_pdf`)

1. **Извлечение диапазона страниц** — `__convert_pdf_bytes_to_bytes_by_pypdf2` через `pypdf.PdfReader/PdfWriter`: извлекает страницы `[start_page_id, end_page_id]`. Если `end_page_id is None` — до последней страницы. При ошибке на отдельной странице — страница пропускается с warning-логом. Если в итоге не добавлено ни одной страницы — возвращаются исходные байты PDF.

2. **Ветвление по backend**:

   **Pipeline backend** (`backend="pipeline"`):
   - `pipeline_doc_analyze` — OCR-анализ документа: извлечение текста, блоков, изображений через модель `PDF-Extract-Kit-1.0`
   - `pipeline_result_to_middle_json` — преобразование результатов модели в промежуточный JSON (middle_json)
   - `pipeline_union_make(MakeMode.MM_MD)` — генерация Markdown-представления
   - `pipeline_union_make(MakeMode.CONTENT_LIST)` — генерация структурированного списка элементов (content_list)
   - Параметры `method`, `formula_enable`, `table_enable` влияют на логику pipeline-анализа

   **VLM backend** (`backend="vlm"`):
   - `vlm_doc_analyze` — анализ документа через VLM-модель `MinerU2.5-2509-1.2B` (Qwen2VLForConditionalGeneration). Модель загружается лениво при первом вызове через `get_vlm_client()` (LRU-кэш)
   - `vlm_union_make(MakeMode.MM_MD)` — генерация Markdown
   - `vlm_union_make(MakeMode.CONTENT_LIST)` — генерация content_list

3. **Конвертация извлечённых изображений в base64** — все файлы изображений из выходной директории (`local_image_dir`) с расширениями `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp` читаются и кодируются в base64 в словарь `images_base64`.

4. **Сборка результата**:
   ```python
   results = {
       "pdf_info": middle_json.get("pdf_info", {}),
       "middle_json": middle_json,
       "model_output": model_list,
       "markdown": md_content,
       "content_list": content_list,
       "output_dir": output_dir,
       "images_base64": images_base64
   }
   ```

#### Обработка изображения (`manager.process_image`)

1. Загрузка изображения через PIL (`Image.open`).
2. `client.two_step_extract(image)` — извлечение информации из изображения через VLM-клиент.
3. Конвертация исходного изображения в base64.
4. Возврат `{"success": True, "results": {"image_info": extracted_blocks, "image_base64": img_base64}, "format": "vlm"}`.

### Output

`ProcessResponse` (Pydantic `BaseModel`):
```python
class ProcessResponse(BaseModel):
    status: str                              # "completed" | "failed"
    message: str                             # Человекочитаемое сообщение
    results: Optional[Dict[str, Any]] = None # {"result": {...}} при успехе, None при ошибке
```

**Успешная обработка PDF** (`status="completed"`):
```json
{
    "status": "completed",
    "message": "Документ обработан",
    "results": {
        "result": {
            "success": true,
            "format": "pipeline",
            "results": {
                "pdf_info": {},
                "middle_json": {},
                "model_output": [],
                "markdown": "string",
                "content_list": [],
                "output_dir": "/tmp/mineru_xxx/",
                "images_base64": { "filename.png": "base64string" }
            }
        }
    }
}
```

**Ошибка обработки** (`status="failed"`):
```json
{
    "status": "failed",
    "message": "Ошибка при обработке документа",
    "results": {
        "result": {
            "success": false,
            "error": "Error description...",
            "format": "vlm"
        }
    }
}
```

## API Contract

```openapi
post: /process
summary: Обработка PDF/изображения через MinerU (OCR + VLM)
requestBody:
  required: true
  content:
    multipart/form-data:
      schema:
        type: object
        properties:
          file:
            type: string
            format: binary
            description: PDF-файл или изображение (.pdf, .png, .jpg, .jpeg, .tiff, .bmp)
          backend:
            type: string
            enum: [pipeline, vlm]
            default: vlm
            description: Бэкенд обработки
          method:
            type: string
            enum: [auto, txt, ocr]
            default: auto
            description: Метод обработки (только для pipeline)
          lang:
            type: string
            default: ru
            description: Язык документа
          formula_enable:
            type: boolean
            default: true
            description: Включить обработку формул
          table_enable:
            type: boolean
            default: true
            description: Включить обработку таблиц
          start_page:
            type: integer
            default: 0
            description: Начальная страница (0-indexed)
          end_page:
            type: integer
            nullable: true
            description: Конечная страница
responses:
  '200':
    description: Результат обработки документа
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ProcessResponse'
  '400':
    description: Неподдерживаемый формат файла
  '500':
    description: Ошибка сохранения/чтения/обработки файла
```

## Data Flow

```
Client (multipart/form-data)
  │
  ▼
api.py: process_document()
  ├─ 1. tempfile.mkdtemp() → Файловая система
  ├─ 2. shutil.copyfileobj() → Сохранение файла на диск
  ├─ 3. Чтение файла в bytes
  ├─ 4. Ветвление:
  │     ├─ PDF → manager.process_pdf()
  │     │   ├─ pypdf → Извлечение диапазона страниц
  │     │   ├─ MinerU Pipeline (PDF-Extract-Kit-1.0) или VLM (MinerU2.5)
  │     │   ├─ FileBasedDataWriter → Изображения в temp_dir/images/
  │     │   └─ base64 encode → images_base64 в памяти
  │     └─ Изображение → manager.process_image()
  │         └─ MinerU VLM-клиент → two_step_extract()
  ├─ 5. shutil.rmtree() → Очистка временной директории
  └─ 6. ProcessResponse → JSON
```

Хранилища, затронутые в процессе:
- **Файловая система**: временная директория `tempfile.mkdtemp` (создаётся и удаляется в рамках запроса), выходные изображения (`local_image_dir`)
- **MODELSCOPE_CACHE** (`/app/models`): чтение моделей `PDF-Extract-Kit-1.0` и `MinerU2.5-2509-1.2B`
- **GPU-память** (опционально): загрузка VLM-модели при первом вызове VLM-обработки
- **Не затрагивается**: MinIO, Qdrant, Neo4j

## LLM Interactions

Сервис `mineru` не взаимодействует с LLM напрямую (Constitution P7 не применим). Вместо этого:

- **VLM-модель** `MinerU2.5-2509-1.2B` (Qwen2VLForConditionalGeneration) — мультимодальная модель для анализа документов, загружается и вызывается через MinerU SDK (`mineru_vl_utils.MinerUClient`). Промпты инкапсулированы внутри `mineru` SDK (метод `two_step_extract` и `vlm_doc_analyze`) и не управляются на уровне сервиса.

## LLM Model Requirements

- **Тип модели**: multimodal (text+image) — VLM для анализа структуры документа
- **Модель**: `OpenDataLab/MinerU2.5-2509-1.2B` (Qwen2VL)
- **Загрузка**: ленивая, при первом вызове VLM-обработки (`get_vlm_client` с `@lru_cache`)
- **Размер**: ~1.2B параметров, загружается в GPU-память при наличии CUDA
- **Pipeline-модель**: `OpenDataLab/PDF-Extract-Kit-1.0` — OCR/структурный анализ, CPU-bound, не требует LLM

## Error Handling

| Код | Условие | Где обрабатывается | Сообщение |
|-----|---------|-------------------|-----------|
| `400` | Расширение файла не в `allowed_extensions` | `api.py:process_document`: проверка `file_ext not in allowed_extensions` | `"Неподдерживаемый формат файла. Разрешены: .pdf, .png, ..."` |
| `500` | Ошибка сохранения файла на диск | `api.py`: `except Exception` при `cpyfileobj` | `"Ошибка сохранения файла: {str(e)}"` |
| `500` | Ошибка чтения сохранённого файла | `api.py`: `except Exception` при `f.read()` | `"Ошибка чтения файла: {str(e)}"` |
| `500` | Ошибка обработки документа (инференс, GPU out-of-memory, ошибка модели) | `api.py`: `except Exception` вокруг `manager.process_pdf/process_image` | `"Ошибка чтения файла: {str(e)}"` — **баг**: сообщение об ошибке неверное, всегда «чтения файла», даже при ошибке инференса |
| — (200 `status=failed`) | Ошибка внутри `manager.process_pdf/process_image`, перехваченная на уровне менеджера | `manager.py`: `try/except` внутри методов → возврат `{"success": False, "error": str(e)}` | `message = "Ошибка при обработке документа"`, `results.result.error = str(e)` |

**Граничные случаи**:
- **Пустой PDF**: `pypdf` вернёт 0 страниц, `__convert_pdf_bytes_to_bytes_by_pypdf2` вернёт исходные байты (с warning-логом)
- **Повреждённый PDF**: `pypdf.PdfReader` выбросит исключение → перехватывается → возврат исходных байтов
- **Несуществующие страницы**: `start_page > total_pages` → `__convert_pdf_bytes_to_bytes_by_pypdf2` не добавит ни одной страницы → исходные байты
- **Файл без расширения**: `Path.suffix` вернёт `""` → не входит в `allowed_extensions` → `400`
- **Одновременные запросы**: `MinerUManager` — синглтон, VLM-модель общая. Одновременная обработка нескольких PDF через VLM может привести к конкуренции за GPU-память. Блокировки отсутствуют.
- **Отсутствие моделей в кэше**: `_init_models()` только логирует warning. Реальная ошибка возникнет при первом вызове `pipeline_doc_analyze` или `vlm_doc_analyze`.

**Известный баг**: на строке 139 `api.py` ошибка обработки документа оборачивается в `HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")` — сообщение фиксированное и не отражает реальную причину ошибки.

## Testing

### Unit-тесты

| # | Сценарий | Метод | Ожидаемый результат |
|---|----------|-------|-------------------|
| 1 | Отправка PDF с `backend=pipeline`, валидные параметры | `POST /process` | `200`, `status="completed"`, `results.result.format="pipeline"`, `results.result.results.markdown` непустой |
| 2 | Отправка PDF с `backend=vlm` | `POST /process` | `200`, `status="completed"`, `results.result.format="vlm"` |
| 3 | Отправка PNG-изображения | `POST /process` | `200`, `results.result.format="vlm"`, `results.result.results.image_info` непустой |
| 4 | Отправка файла с неподдерживаемым расширением (`.docx`) | `POST /process` | `400`, `detail="Неподдерживаемый формат файла..."` |
| 5 | Отправка без файла | `POST /process` (без `file`) | `422 Unprocessable Entity` (FastAPI-валидация) |
| 6 | `backend="pipeline"`, `method="ocr"`, `formula_enable=False` | `POST /process` | `200`, pipeline вызван с `parse_method="ocr"`, `formula_enable=False` |
| 7 | `start_page=1`, `end_page=2` для PDF из 5 страниц | `POST /process` | `200`, обработаны только страницы 1-2 |
| 8 | `start_page=100` для PDF из 3 страниц | `POST /process` | `200` (не 400), менеджер обработает пустой PDF с warning |

### Интеграционные тесты

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 9 | Обработка реального PDF с таблицами (pipeline) | В `content_list` присутствуют элементы типа `table` |
| 10 | Обработка PDF с формулами (pipeline, `formula_enable=True`) | Формулы распознаны в markdown-выводе |
| 11 | Отправка битого PDF (обрезанные байты) | Либо `200` `status="failed"` с ошибкой, либо `500` |
| 12 | VLM-обработка без предзагруженной модели | VLM-клиент инициализируется при первом запросе, последующие — используют кэш |

### Граничные случаи

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 13 | Файл с именем без расширения | `400` — `""` нет в `allowed_extensions` |
| 14 | Расширение в верхнем регистре (`.PDF`) | `200` — проверка `.lower()` |
| 15 | Ошибка записи на диск (диск заполнен) | `500`, временная директория удалена |
| 16 | Язык `lang="en"`, `backend="pipeline"` | `200`, OCR-модель использует английский языковой пакет |

## Dependencies

| Зависимость | Тип | Назначение |
|------------|-----|-----------|
| `mineru` SDK (`mineru.cli.common`, `mineru.backend.pipeline.*`, `mineru.backend.vlm.*`) | Библиотека | Pipeline и VLM-обработка документов |
| `mineru_vl_utils.MinerUClient` | Библиотека | VLM-клиент для извлечения информации из изображений |
| `pypdf` (`PdfReader`, `PdfWriter`) | Библиотека | Извлечение диапазона страниц PDF |
| `PIL` (Pillow) | Библиотека | Открытие и обработка изображений |
| `modelscope` (`AutoProcessor`, `Qwen2VLForConditionalGeneration`) | Библиотека | Загрузка и инференс VLM-модели |
| `MODELSCOPE_CACHE` (env var) | Файловая система | Кэш моделей MinerU (по умолчанию `/app/models`) |
| GPU (CUDA 12.4/12.8) | Аппаратный | Ускорение VLM-инференса (опционально) |

## Exceptions

1. **Синхронная обработка** — метод `process_document` и `MinerUManager.process_pdf/process_image` синхронные. Отклонения от Constitution P5 нет: инференс MinerU является CPU/GPU-bound вычислением, для которого синхронный код допустим. Однако `api.py:process_document` синхронный, что потенциально блокирует event loop; Constitution рекомендует `run_in_executor` для CPU-bound.

2. **Pydantic-модели в `api.py` вместо `dtype/`** — `ProcessResponse` и `StatusResponse` определены непосредственно в `api.py`, а не в отдельной `dtype/` директории. Отклонение от Constitution N2 (структура модулей). **Причина**: переходный период, сервис документируется «как есть»; планируется рефакторинг в `dtype/` с Data Model Specs.

3. **`ProcessingConfig` — `dataclass`, не Pydantic** — конфигурация обработки в `manager.py` определена как `@dataclass`, а не `BaseModel`. Отклонение от Constitution P6. **Фактическое влияние**: `ProcessingConfig` используется только внутри сервиса (между `api.py` и `manager.py`) без сериализации; строгая Pydantic-валидация не критична, но противоречит принципу.

4. **README расходится с кодом** — эндпоинты `/status/{task_id}`, `/download/{file_path}` упомянуты в корневом эндпоинте `GET /` и README, но не реализованы в коде. In-memory словарь `tasks` и модель `StatusResponse` объявлены, но не используются. **Причина**: README описывает желаемую асинхронную архитектуру; фактически обработка синхронная.

5. **Баг в обработке ошибок** — `HTTPException(status_code=500, detail=f"Ошибка чтения файла: {str(e)}")` на строке 139 всегда формирует сообщение «Ошибка чтения файла», даже если ошибка произошла на этапе инференса. **Причина**: копипаста с блока чтения файла (строки 119-121).

6. **Отсутствие rate limiting / таймаута** — нет ограничений на количество одновременных запросов, нет таймаута обработки (хотя `config/settings.py` декларирует `timeout: 30`, но он не используется в коде `api.py` и `manager.py`). При большом PDF или медленном GPU запрос может висеть неопределённо долго.
