# Pipeline: Community Report Generation

## Purpose
Генерация иерархических отчётов (summaries) по сообществам (Community), обнаруженным на этапе кластеризации. Для каждого сообщества на каждом уровне иерархии LLM генерирует структурированный отчёт: заголовок, резюме, рейтинг влиятельности, детальные находки. Это **always-full-graph** операция (Constitution P9), запускается через эндпоинт `GET /create_community_report` в `semantic_graph/semantic_index.py`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Load Data from Neo4j | — | `semantic_index.py:create_community_report()` → `doc_manager.get_entity_relationships()` (все RELATED связи), `doc_manager.get_entities()` (все Entity узлы), `doc_manager.get_community()` (все Community узлы с CONSISTS_OF связями и children). | `relationships: pd.DataFrame`, `entities: pd.DataFrame`, `communities: pd.DataFrame` | sequential |
| 2. Prepare Nodes & Edges | `entities`, `communities` | `create_community_report.py:run_community_reports_pipeline_async()` → `explode_communities()` — разворачивание `entity_ids` в строках communities, merge с entities по `id`. `_prep_nodes()` — заполнение `node_details` (dict из `id`, `title`, `description`, `degree`). `_prep_edges()` — заполнение `edge_details` (dict из `id`, `source`, `target`, `description`, `combined_degree`). | `nodes: pd.DataFrame`, `edges: pd.DataFrame` | sequential |
| 3. Build Local Context | `nodes`, `edges` | `build_local_context()` → для каждого уровня иерархии (от высшего к низшему, `get_levels()`): `_prepare_reports_at_level()` — фильтрация узлов/рёбер по уровню, группировка по community, агрегация `all_context`. `parallel_sort_context_batch()` — параллельный вызов `sort_context()`: сортировка рёбер по `combined_degree` (desc), итеративное добавление узлов/рёбер в контекстную строку (CSV-формат) с контролем `max_context_tokens=8000` через `llm.count_tokens()`. | `local_contexts: pd.DataFrame` с колонками: `community_id`, `context_string`, `context_size`, `context_exceed_flag` | parallel (per context, sequential внутри sort_context) |
| 4. Build Level Context (Hierarchical) | `local_contexts`, `community_hierarchy` | `summarize_communities()` → для каждого уровня (от высшего к низшему): `build_level_context()`. **Логика**: (a) Разделение на valid (`context_exceed_flag=False`) и invalid контексты. (b) Для invalid: если есть отчёты с нижнего уровня — строится mixed context через `build_mixed_context()`: замена локального контекста на full_content дочерних отчётов (агрегация через CSV). (c) Для оставшихся — `sort_and_trim_context()` (сортировка и обрезка). (d) UNION valid + mixed + trimmed. | `level_contexts: List[pd.DataFrame]` (по одному на уровень) | sequential per level |
| 5. Generate Reports via LLM | `level_contexts` | `summarize_communities()` → для каждого уровня: `_process_rows_async()` с `max_concurrent=4`. Для каждой строки → `_generate_report()` → `AsyncCommunityReportExtractor.extract()` → `llm.generate_structured()` с `response_format={"type": "json_object"}`, парсинг в `CommunityReportResponse` (Pydantic: `title`, `summary`, `findings: List[FindingModel]`, `rating: float`, `rating_explanation`). Промпт: `COMMUNITY_REPORT_PROMPT` с подстановкой `{input_text}` (контекст сообщества) и `{max_report_length}`. | `reports: List[dict]` (поля: `community_id`, `full_content`, `level`, `rating`, `title`, `explanation`, `summary`, `findings`, `full_content_json`) | parallel (semaphore max_concurrent=4) |
| 6. Finalize Reports | `reports`, `communities` | `finalize_community_reports()` — merge reports с communities по `community` (добавление `id`, `parent`, `children`, `size`, `period`). Формирование `human_readable_id`. Выборка по `COMMUNITY_REPORTS_FINAL_COLUMNS`. | `final_reports: pd.DataFrame` | sequential |
| 7. Write to Neo4j | `final_reports` | `semantic_index.py:create_community_report()` → `doc_manager.update_community_reports(community_reports, batch_size=500)`. UNWIND-запрос: MATCH Community по `id`, SET полей отчёта (`title`, `summary`, `full_content`, `rating`, `explanation`, `findings`, `full_content_json`, `report_updated_at`). | Статистика: `{updated, skipped, not_found}` | sequential (batch write) |

## Data Flow Diagram

```
GET /create_community_report
    │
    ▼
[Stage 1] Neo4j: load all entities, relationships, communities
    │
    ▼
[Stage 2] Prepare: explode community→entity mapping, build node_details, edge_details
    │
    ▼
[Stage 3] Build Local Context (per community, parallel)
    │ sort_context: sort edges by degree, build CSV string, check token limit
    │ max_context_tokens = 8000
    ▼
[Stage 4] Build Level Context (per hierarchy level, bottom-up)
    │ valid contexts → keep as-is
    │ exceeded contexts → try mixed context (child reports)
    │ remaining → trim context
    ▼
[Stage 5] LLM: generate_structured (parallel, max_concurrent=4)
    │ prompt: COMMUNITY_REPORT_PROMPT (config.py)
    │ model: Qwen/Qwen3-4B-Instruct-2507
    │ response_model: CommunityReportResponse (Pydantic, JSON)
    │ output: title, summary, findings[], rating, rating_explanation
    ▼
[Stage 6] Finalize: merge reports with community metadata
    ▼
[Stage 7] Neo4j: update Community nodes with report fields (batch_size=500)
    ▼
Response: { document_id: "None", statistics: { updated, skipped, not_found } }
```

## LLM Interactions

### Community Report Prompt
- **Файл**: должен быть `semantic_graph/prompts/community-report.md` (сейчас `COMMUNITY_REPORT_PROMPT` в `config.py`, строки 322-468 — нарушение P7 Constitution).
- **Формат**: `# Goal` → `# Report Structure` (TITLE, SUMMARY, IMPACT SEVERITY RATING, RATING EXPLANATION, DETAILED FINDINGS) → `# Grounding Rules` (data references) → `# Example Input/Output` → `# Real Data` с подстановкой `{input_text}` и `{max_report_length}`.
- **Выходной формат**: JSON с полями: `title` (str), `summary` (str), `rating` (float 0-10), `rating_explanation` (str), `findings` (list of `{summary, explanation}`).
- **Парсинг**: `AsyncLLMClient.generate_structured()` — `response_format={"type": "json_object"}`, парсинг через `re.search(r"```(?:json)?\s*(\{.*\})\s*```")` с fallback на прямую загрузку JSON. Валидация через `CommunityReportResponse.model_validate()`.
- **Модель**: `Qwen/Qwen3-4B-Instruct-2507` (из `config.MODEL_NAME`).

## Performance Constraints

- **Model**: `Qwen/Qwen3-4B-Instruct-2507`.
- **Max context tokens**: 8000 (`max_input_length`).
- **Max report length**: 2000 слов (`max_report_length`).
- **Max concurrent LLM calls**: 4 (`max_concurrent`).
- **Neo4j batch size**: 500 (report update).
- **Token counting**: через `TOKENIZER_URL`, fallback `len(text)//4`.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Load Data | Neo4j недоступен | Исключение → HTTP 500 |
| 2. Prepare | Пустые данные | Pipeline продолжается, на выходе пустой DataFrame |
| 3. Build Context | Ошибка токенизатора | Fallback `len(text)//4` |
| 4. Level Context | Все контексты exceed token limit | `build_mixed_context()` пытается заменить на дочерние отчёты; если всё ещё превышает — обрезает |
| 5. LLM Generate | Ошибка LLM для отдельного сообщества | `_generate_report()` возвращает None, сообщество пропускается (логируется warning) |
| 5. LLM Generate | Некорректный JSON в ответе | `generate_structured()` возвращает None, сообщество пропускается |
| 6. Finalize | — | Операции pandas, ошибок не ожидается |
| 7. Write to Neo4j | Community не найден по id | Записывается в `not_found`, не прерывает pipeline |

Общая обработка: внешний `try/except` в `create_community_report()` → HTTP 500.

**Примечание (Constitution P9)**: генерация отчётов — always-full-graph операция. При добавлении новых документов требуется перезапуск после `clastrize_graph`. При частичной ошибке LLM некоторые сообщества могут остаться без отчётов.
