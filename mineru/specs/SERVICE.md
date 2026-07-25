# Service: MinerU PDF Processing

## Identity
- Зона ответственности: PDF → structured content (OCR, VLM, разбор структуры документа)
- Порт: 8000 (внутренний)
- Хранилище: Файловая система (кэш моделей — `MODELSCOPE_CACHE`, по умолчанию `/app/models`)

## Dependencies
| Сервис | Протокол | Назначение |
|--------|----------|------------|
| Файловая система (`MODELSCOPE_CACHE`) | Локальный доступ на чтение | Кэш предзагруженных моделей: PDF-Extract-Kit-1.0 (pipeline), MinerU2.5-2509-1.2B (VLM) |
| GPU (CUDA 12.8) | Аппаратный | Ускорение инференса моделей (опционально — возможен запуск на CPU) |
| ModelScope | HTTP (только при сборке образа) | Загрузка моделей через `snapshot_download` на этапе `docker build` |
| PyTorch (CUDA 12.4 runtime) | Библиотека | Инференс VLM-модели (Qwen2VLForConditionalGeneration) |

## API
| Method | Path | Feature Spec | Purpose |
|--------|------|-------------|---------|
| GET | `/` | — | Корневой эндпоинт: информация о сервисе и список доступных эндпоинтов |
| GET | `/health` | — | Проверка здоровья сервиса |
| POST | `/process` | [process-pdf.md](process-pdf.md) | Обработка PDF или изображения (OCR, VLM, извлечение структуры) |

### POST /process — контракт

**Request**: `multipart/form-data`

| Параметр | Тип | Query/File | По умолчанию | Описание |
|----------|-----|-----------|-------------|----------|
| `file` | `UploadFile` | File (body) | — (обязательный) | PDF-файл или изображение |
| `backend` | `str` | Query | `"vlm"` | Бэкенд обработки: `"pipeline"` или `"vlm"` |
| `method` | `str` | Query | `"auto"` | Метод обработки: `"auto"`, `"txt"`, `"ocr"` |
| `lang` | `str` | Query | `"ru"` | Язык документа |
| `formula_enable` | `bool` | Query | `true` | Включить обработку формул |
| `table_enable` | `bool` | Query | `true` | Включить обработку таблиц |
| `start_page` | `int` | Query | `0` | Начальная страница (0-indexed) |
| `end_page` | `int \| null` | Query | `null` | Конечная страница (0-indexed, включительно). Если `null` — до последней страницы |

**Поддерживаемые форматы файлов**: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

**Response** `200 OK` (ProcessResponse):
```json
{
  "status": "completed" | "failed",
  "message": "string",
  "results": {
    "result": {
      "success": true,
      "format": "pipeline" | "vlm",
      "results": {
        "pdf_info": {},
        "middle_json": {},
        "model_output": [],
        "markdown": "string",
        "content_list": [],
        "output_dir": "string",
        "images_base64": { "filename": "base64string" }
      }
    }
  }
}
```

**Коды ошибок**:
| Код | Условие |
|-----|---------|
| `400` | Неподдерживаемый формат файла (расширение не в списке разрешённых) |
| `500` | Ошибка сохранения временного файла на диск |
| `500` | Ошибка чтения сохранённого файла |
| `500` | Ошибка обработки документа (ошибка инференса, нехватка памяти GPU и т.д.) |

## Data Model
- Ссылки на Data Model Specs: отсутствуют. Pydantic-модели `ProcessResponse` и `StatusResponse` определены непосредственно в `api.py` (см. Exceptions).

## Pipelines
- Ссылки на Pipeline Specs: [pdf-processing.md](pipelines/pdf-processing.md) — сквозной пайплайн обработки PDF (pipeline + vlm backends, извлечение контента, конвертация изображений в base64)

## Configuration

### Переменные окружения
| Переменная | Назначение | По умолчанию |
|-----------|------------|-------------|
| `MODELSCOPE_CACHE` | Путь к директории с кэшем моделей MinerU | `/app/models` |
| `PYTHONPATH` | Путь поиска Python-модулей | `/app` (в Docker) |
| `PYTHONUNBUFFERED` | Отключение буферизации stdout/stderr | `1` (в Docker) |

### Settings-класс (`config/settings.py`, Pydantic Settings)
| Поле | Тип | По умолчанию | Назначение |
|------|-----|-------------|------------|
| `host` | `str` | `"http://localhost"` | Хост сервиса (для клиентских ссылок) |
| `port` | `str` | `"8001"` | Порт сервиса |
| `timeout` | `int` | `30` | Таймаут обработки (секунды) |
| `max_file_size` | `int` | `52428800` (50 MB) | Максимальный размер загружаемого файла (байты) |
| `cache_ttl` | `int` | `3600` | Время жизни кэша (секунды) |

**Файл конфигурации**: `.env` (автоматически загружается Pydantic Settings)

### Docker
| Параметр | Значение |
|----------|----------|
| Базовый образ | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` |
| Python | 3.11 |
| PyTorch | CUDA 12.4 (совместим с драйвером 12.8) |
| Рабочая директория | `/app` |
| Пользователь | `appuser` (uid 1000) |
| Порты | `EXPOSE 8000` |
| CMD | `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1` |

### Модели (предзагружаются на этапе сборки)
| Модель | Назначение | Путь в кэше |
|--------|-----------|------------|
| `OpenDataLab/PDF-Extract-Kit-1.0` | Pipeline-обработка PDF (OCR, разбор структуры) | `{MODELSCOPE_CACHE}/OpenDataLab/PDF-Extract-Kit-1.0` |
| `OpenDataLab/MinerU2.5-2509-1.2B` | VLM-обработка документов и изображений | `{MODELSCOPE_CACHE}/OpenDataLab/MinerU2.5-2509-1.2B` |

## Invariants
1. **Singleton Manager**: `MinerUManager` реализован как синглтон (`__new__`). Единственный экземпляр инициализирует модели при первом создании.
2. **Lazy VLM-клиент**: VLM-модель (Qwen2VLForConditionalGeneration) загружается в память только при первом вызове VLM-обработки (`get_vlm_client` с `@lru_cache`), а не на старте сервиса.
3. **Модели только проверяются на старте**: `_init_models()` при инициализации менеджера **не загружает** модели в память — только проверяет наличие директорий в кэше.
4. **Pipeline-бэкенд не требует GPU**: при `backend=pipeline` GPU-память не используется (только CPU), VLM-модель не инициализируется.
5. **Временные файлы**: входной PDF сохраняется во временную директорию (`tempfile.mkdtemp`), которая удаляется после завершения обработки (успешной или с ошибкой).
6. **Page range**: при `backend=pipeline` или `backend=vlm` применяется `__convert_pdf_bytes_to_bytes_by_pypdf2` — извлечение указанного диапазона страниц через pypdf перед передачей в MinerU.
7. **Изображения — base64**: все извлечённые изображения конвертируются в base64 и включаются в ответ (поле `results.images_base64`).
8. **Безопасность**: CORS middleware разрешает все origins (`allow_origins=["*"]`).
9. **Обработка ошибок страниц**: при извлечении страниц через pypdf, если отдельная страница не может быть импортирована, она пропускается с warning-логом, а не прерывает всю обработку. Если все страницы пропущены — возвращаются исходные байты PDF.

## Exceptions
1. **Нет `dtype/` директории**: Pydantic-модели `ProcessResponse` и `StatusResponse` определены непосредственно в `api.py`, а не вынесены в отдельный модуль `dtype/`. Отклонение от Constitution N2. **Причина**: переходный период — сервис документируется «как есть»; рефакторинг в отдельную `dtype/` с соответствующими Data Model Specs планируется в будущем.
2. **Нет `prompts/` директории**: в сервисе `mineru` отсутствуют LLM-взаимодействия — вся обработка выполняется через MinerU SDK (OCR, VLM-инференс). Отклонения от Constitution P7 нет — правило не применимо, так как промпты не используются.
3. **Нет `specs/` директории (до текущего коммита)**: спецификации для сервиса отсутствовали. **Причина**: переходный период согласно Constitution 5.3 — `SERVICE.md` создаётся данным spec-файлом как документация текущего поведения.
4. **Синхронный менеджер**: `MinerUManager.process_pdf()` и `process_image()` — синхронные методы. Отклонения от Constitution P5 нет: инференс MinerU (pipeline/VLM) является CPU/GPU-bound вычислением, для которого синхронный код допустим. Порт 8000 в CONFIG.md расходится с кодом по умолчанию `8001` в `run_api()`, но Dockerfile явно задаёт `--port 8000`, поэтому эффективный порт = 8000.
5. **Нет эндпоинтов `/status/{task_id}`, `/download/{file_path}`, `/cleanup/{task_id}`**: эти эндпоинты упомянуты в `README.md`, но **не реализованы** в `api.py`. Обработка в `POST /process` выполняется синхронно — ответ содержит полный результат. In-memory словарь `tasks` в `api.py` объявлен, но не используется ни одним эндпоинтом. **Причина**: README.md описывает желаемое, а не реализованное поведение; фактический API синхронный.
