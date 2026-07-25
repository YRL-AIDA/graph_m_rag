# Pipeline: PDF Processing (MinerU)

## Purpose
Обработка PDF-документа (или изображения) через MinerU: оптическое распознавание (OCR) или визуально-языковую модель (VLM), структурирование контента (текст, изображения, таблицы, формулы), возврат в формате JSON с `content_list` и `images_base64`. Запускается через эндпоинт `POST /process` в `mineru/api.py`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Receive & Validate | `file: UploadFile` (multipart/form-data), query params: `backend` (default `"vlm"`), `method` (default `"auto"`), `lang` (default `"ru"`), `formula_enable` (default `True`), `table_enable` (default `True`), `start_page` (default `0`), `end_page` (default `None`) | `mineru/api.py:process_document()` — генерация `task_id=uuid4()`, проверка расширения файла (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`), сохранение во временную директорию `/tmp/mineru_{task_id}_/`. Создание `ProcessingConfig`. Чтение файла как bytes. | `task_id`, `file_bytes`, `config: ProcessingConfig`, `temp_dir` | sequential |
| 2. Route by Backend | `file_bytes`, `config` | `MinerUManager.process_pdf()` (для `.pdf`) или `MinerUManager.process_image()` (для изображений). Выбор backend: `config.backend == "pipeline"` или иначе (vlm). | — | sequential |
| 3a. Pipeline Backend: Page Extraction | `pdf_bytes`, `config` | `__convert_pdf_bytes_to_bytes_by_pypdf2()` — извлечение страниц через `pypdf.PdfReader`/`PdfWriter` в диапазоне `[start_page_id, end_page_id]`. Ошибки отдельных страниц логируются и пропускаются. | `pdf_bytes` (урезанный PDF) | sequential |
| 3b. Pipeline Backend: Document Analysis | `pdf_bytes`, `lang`, `method`, `formula_enable`, `table_enable` | `mineru.backend.pipeline.pipeline_analyze.doc_analyze()` — batch-анализ PDF: детекция layout, OCR (если `method="ocr"` или автоопределение), распознавание формул/таблиц. Модель: `PDF-Extract-Kit-1.0` (из `MODELSCOPE_CACHE`). | `infer_results`, `all_image_lists`, `all_pdf_docs`, `lang_list`, `ocr_enabled_list` | sequential |
| 3c. Pipeline Backend: Middle JSON Conversion | `infer_results`, `images`, `pdf_doc` | `mineru.backend.pipeline.model_json_to_middle_json.result_to_middle_json()` — преобразование результатов модели в промежуточный JSON (middle_json). | `middle_json` с ключом `pdf_info` | sequential |
| 3d. Pipeline Backend: Content Extraction | `middle_json["pdf_info"]` | `mineru.backend.pipeline.pipeline_middle_json_mkcontent.union_make(mode=MakeMode.CONTENT_LIST)` — извлечение структурированного `content_list` (список элементов с типами: text, image, table, equation, discarded). Параллельно: `union_make(mode=MakeMode.MM_MD)` — генерация Markdown. | `content_list: List[Dict]`, `md_content: str` | sequential |
| 4a. VLM Backend: Page Extraction | `pdf_bytes`, `config` | Аналогично Stage 3a — `__convert_pdf_bytes_to_bytes_by_pypdf2()`. | `pdf_bytes` | sequential |
| 4b. VLM Backend: VLM Analysis | `pdf_bytes`, `image_writer`, `backend` | `mineru.backend.vlm.vlm_analyze.doc_analyze()` — постраничный анализ PDF через VLM-модель. Модель: `MinerU2.5-2509-1.2B` (Qwen2VL) из `MODELSCOPE_CACHE`, загружается через `MinerUManager.get_vlm_client()` (lru_cache). VLM-клиент создаётся через `mineru_vl_utils.MinerUClient` с backend="transformers". | `middle_json`, `infer_result` | sequential |
| 4c. VLM Backend: Content Extraction | `middle_json["pdf_info"]` | `mineru.backend.vlm.vlm_middle_json_mkcontent.union_make(mode=MakeMode.CONTENT_LIST)` и `union_make(mode=MakeMode.MM_MD)`. | `content_list`, `md_content` | sequential |
| 5. Image Base64 Encoding | `local_image_dir` | `MinerUManager.process_pdf()` — чтение всех изображений из `local_image_dir` (фильтр по расширениям: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp`), кодирование в base64. | `images_base64: Dict[str, str]` | sequential |
| 6. Build Response | `middle_json`, `content_list`, `md_content`, `images_base64`, `model_output` | `MinerUManager.process_pdf()` — сборка результата: `results.pdf_info`, `results.middle_json`, `results.model_output`, `results.markdown`, `results.content_list`, `results.output_dir`, `results.images_base64`. `api.py:process_document()` — оборачивание в `ProcessResponse`. | `ProcessResponse: {status, message, results: {result: {...}}}` | sequential |
| 7. Cleanup | `temp_dir` | `api.py:process_document()` — удаление временной директории `shutil.rmtree(temp_dir)`. | — | sequential |

## Data Flow Diagram

```
POST /process (multipart PDF + query params)
    │
    ▼
[Stage 1] Validate extension, save to temp dir, read bytes
    │
    ▼
[Stage 2] Route: .pdf → process_pdf(), image → process_image()
    │
    ├──► Pipeline Backend (backend="pipeline"):
    │       │
    │       ▼
    │    [3a] Extract pages (pypdf)
    │       ▼
    │    [3b] doc_analyze (PDF-Extract-Kit-1.0): layout, OCR, formulas, tables
    │       ▼
    │    [3c] result_to_middle_json
    │       ▼
    │    [3d] union_make → content_list + markdown
    │
    └──► VLM Backend (default, backend="vlm"):
            │
            ▼
         [4a] Extract pages (pypdf)
            ▼
         [4b] vlm_doc_analyze (MinerU2.5-2509-1.2B, Qwen2VL)
            ▼
         [4c] union_make → content_list + markdown
    │
    ▼
[Stage 5] Read images from local dir → base64
    │
    ▼
[Stage 6] Build response + [Stage 7] Cleanup temp dir
    │
    ▼
ProcessResponse: { status, message, results: { result: { pdf_info, middle_json, model_output, markdown, content_list, images_base64 } } }
```

## LLM Interactions

- **VLM Backend**: модель `MinerU2.5-2509-1.2B` (Qwen2VLForConditionalGeneration) используется для визуального анализа страниц PDF. Загружается через `transformers` + `modelscope`, инференс через `mineru_vl_utils.MinerUClient`.
- **Pipeline Backend**: модель `PDF-Extract-Kit-1.0` — традиционный pipeline (layout detection + OCR), без LLM.
- **Промпты**: не требуются (модели MinerU предобучены на задачу extraction).

### LLM Model Requirements
- **VLM model**: `Qwen2VLForConditionalGeneration` (MinerU2.5-2509-1.2B), ~1.2B параметров.
- **Device**: `device_map="auto"` (автоматическое распределение по GPU/CPU).
- **Dtype**: `torch_dtype="auto"`.
- **Pipeline model**: `PDF-Extract-Kit-1.0`, вес модели — несколько GB, требует GPU для приемлемой производительности.

## Performance Constraints

- **Модели загружаются один раз**: `MinerUManager` — singleton (`__new__`), VLM клиент — `lru_cache(maxsize=1)`.
- **VLM инференс**: постраничный, время пропорционально количеству страниц.
- **Pipeline инференс**: batch-обработка всего документа.
- **Память**: модели загружаются в GPU-память при старте сервиса. PDF-Extract-Kit-1.0 требует несколько GB VRAM.
- **Temp files**: создаются в `/tmp`, требуют достаточно места для изображений и промежуточных результатов.
- **Image encoding**: все изображения результата конвертируются в base64 и включаются в JSON-ответ — может быть очень большим для документов с многими изображениями.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Validate | Неподдерживаемый формат | HTTP 400 |
| 1. Receive | Ошибка сохранения временного файла | `shutil.rmtree(temp_dir)` → HTTP 500 |
| 1. Receive | Ошибка чтения файла | `shutil.rmtree(temp_dir)` → HTTP 500 |
| 3a/4a. Page Extraction | Ошибка отдельной страницы | Страница пропускается (`logger.warning`), pipeline продолжается |
| 3a/4a. Page Extraction | Все страницы пропущены | Возвращаются исходные pdf_bytes |
| 3b. Pipeline Analysis | Ошибка модели | Исключение → `MinerUManager.process_pdf()` возвращает `{"success": False, "error": ...}` |
| 4b. VLM Analysis | Ошибка VLM | Исключение → `{"success": False, "error": ...}` |
| 5. Image Encoding | Ошибка чтения отдельного изображения | Пропускается |
| 7. Cleanup | Ошибка удаления | `pass` (блок `try/except: pass`) |

**Важно**: `api.py:process_document()` ловит исключения из `manager.process_pdf()` и возвращает HTTP 500. Однако `manager.process_pdf()` сам ловит исключения и возвращает `{"success": False, "error": str(e)}` — таким образом HTTP 200 может содержать failed-результат (двойная обработка ошибок).
