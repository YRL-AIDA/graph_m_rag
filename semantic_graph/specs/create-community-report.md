# Feature: Генерация отчётов по сообществам (create_community_report)

## Motivation
Генерация иерархических LLM-отчётов для каждого сообщества (`Community`) семантического графа знаний. На основе ранее построенных сообществ (через `/clastrize_graph`), их сущностей и связей, для каждого сообщества формируется контекст (сущности + связи + отчёты дочерних сообществ), который подаётся в LLM для генерации структурированного отчёта с оценкой важности (rating) и детальными находками (findings). Результат записывается в узлы `Community` Neo4j как свойства `summary`, `full_content`, `rating`, `findings`, `full_content_json`. Это глобальная операция (Constitution P9).

## Behaviour

### Input
- **Метод**: `GET`
- **Path**: `/create_community_report`
- **Параметры**: нет. Все параметры (`MODEL_NAME`, `COMMUNITY_REPORT_PROMPT`, `max_input_length=8000`, `max_report_length=2000`, `max_concurrent=4`) заданы в `config.py` и коде эндпоинта.
- **Валидация**: отсутствует.

### Processing
Вызывается `run_community_reports_pipeline_async(relationships, entities, communities, MODEL_NAME, COMMUNITY_REPORT_PROMPT, max_input_length=8000, max_report_length=2000, max_concurrent=4)`:

**1. Загрузка данных из Neo4j** (в эндпоинте, до вызова пайплайна):
   - `doc_manager.get_entity_relationships()` → DataFrame всех связей `RELATED`.
   - `doc_manager.get_entities()` → DataFrame всех сущностей `Entity` (колонки: `id` (формат `title|type`), `title`, `type`, `description`, `degree`, `data`, `updated_at`, `created_at`).
   - `doc_manager.get_community()` → DataFrame всех сообществ `Community` с дополнительными подзапросами на `CONSISTS_OF` (entity_ids) и `IS_PARENT_OF` (children).

**2. Этап 1 — Подготовка узлов и рёбер**:
   - `explode_communities(communities, entities)`: джойн сообществ (explode по `entity_ids`) с entities по `id` → каждый узел получает поля `community` и `level`.
   - `_prep_nodes(nodes)`: заполнение `description = "No Description"` для NaN, формирование `node_details` как dict с `id`, `title`, `description`, `degree`.
   - `_prep_edges(relationships)`: заполнение `description = "No Description"` для NaN, формирование `edge_details` как dict с `id`, `source`, `target`, `description`, `combined_degree`.

**3. Этап 2 — Построение локального контекста**:
   - `build_local_context(nodes, edges, llm, model, max_context_tokens)`:
     - Определяются уровни иерархии сообществ (`get_levels()` — убывающий порядок).
     - Для каждого уровня вызывается `_prepare_reports_at_level()`:
       - Фильтрация узлов уровня, фильтрация рёбер где оба конца в узлах уровня.
       - Агрегация рёбер по source и target → слияние с узлами.
       - Группировка по `community` → `all_context` как список записей с `id`, `degree`, `node_details`, `edge_details`.
       - `parallel_sort_context_batch()`: для каждого сообщества вызывается `sort_context()`:
         - Сортировка рёбер по убыванию `combined_degree`.
         - Итеративное добавление узлов и рёбер в контекстную строку (CSV-формат через `pd.DataFrame.to_csv()`).
         - Контроль размера через `llm.count_tokens()` — остановка при превышении `max_context_tokens`.
         - Если есть `sub_community_reports` — добавляются как "Reports" секция.
       - Вычисляется `context_size` и флаг `context_exceed_flag`.
   - Результат: DataFrame `local_contexts` с колонками `community`, `all_context`, `context_string`, `context_size`, `context_exceed_flag`, `level`.

**4. Этап 3 — Генерация отчётов по уровням иерархии**:
   - `summarize_communities(nodes, communities, local_contexts, build_level_context, extractor, llm, model, max_input_length, max_concurrent)`:
     - Построение `community_hierarchy`: explode `children`, rename `children` → `sub_community`.
     - Для каждого уровня (от верхнего к нижнему) вызывается `build_level_context()`:
       - Разделение на `valid_context_df` (контекст умещается в лимит) и `invalid_context_df` (превышен).
       - Если `invalid_context_df` пуст → отдаются валидные как есть.
       - Если есть отчёты с предыдущего уровня → вызов `build_mixed_context()`:
         - Сортировка дочерних контекстов по убыванию `context_size`.
         - Для превышающих лимит: попытка заменить локальный контекст на `full_content` дочернего отчёта.
         - Итеративный перебор, пока контекст не уложится в `max_context_tokens`.
         - Если не уложились — только дочерние отчёты как контекст.
       - Если отчётов нет → простая сортировка и обрезка контекста.
     - **Генерация отчётов**: для каждого сообщества уровня параллельно (с семафором `max_concurrent=4`) вызывается `_generate_report()`:
       - `AsyncCommunityReportExtractor.extract(context_string)`:
         - `COMMUNITY_REPORT_PROMPT.format(input_text=..., max_report_length=...)`
         - LLM-вызов через `generate_structured()` с `response_format={"type": "json_object"}`, парсинг в `CommunityReportResponse`.
       - Результат: dict с `community`, `level`, `full_content`, `rating`, `title`, `rating_explanation`, `summary`, `findings`, `full_content_json`.

**5. Этап 4 — Финализация**:
   - `finalize_community_reports(reports_df, communities_df)`: джойн с сообществами по `community` → добавление `id`, `parent`, `children`, `size`, `period`.
   - Возвращается DataFrame с колоннами из `COMMUNITY_REPORTS_FINAL_COLUMNS`.

**6. Сохранение в Neo4j**:
   - `Manager.update_community_reports(community_reports)`: батчами по 500 обновляются узлы `Community` (MATCH по `id`, SET `title`, `summary`, `full_content`, `rating`, `rating_explanation`, `findings`, `full_content_json`).
   - Возвращается `{"updated": N, "skipped": N, "not_found": N}`.

**7. Формирование ответа**:
   - `_build_response('None', 0, start_time, 'completed', neo4j_status)`.

### Output
- **HTTP 200** — успешная генерация:
  ```json
  {
    "document_id": "None",
    "statistics": {
      "total_chunks": 0,
      "processing_time_ms": <int>,
      "status": "completed",
      "updated": <int>,
      "skipped": <int>,
      "not_found": <int>
    }
  }
  ```
- **HTTP 500** — ошибка Neo4j, LLM или обработки (unhandled exception).

## API Contract

```openapi
GET /create_community_report
summary: Генерация LLM-отчётов по всем сообществам графа знаний и сохранение в Neo4j
responses:
  '200':
    description: Отчёты успешно сгенерированы
    content:
      application/json:
        schema:
          type: object
          properties:
            document_id:
              type: string
              description: Всегда "None" (глобальная операция)
            statistics:
              type: object
              properties:
                total_chunks:
                  type: integer
                  description: Всегда 0
                processing_time_ms:
                  type: integer
                status:
                  type: string
                  enum: [completed]
                updated:
                  type: integer
                  description: Количество обновлённых узлов Community
                skipped:
                  type: integer
                  description: Количество пропущенных (нет id)
                not_found:
                  type: integer
                  description: Количество не найденных в графе Community
  '500':
    description: Ошибка Neo4j, LLM или внутренняя ошибка пайплайна
    content:
      application/json:
        schema:
          type: object
          properties:
            detail:
              type: string
```

## Data Flow
```
Client (app)
  │
  │ GET /create_community_report
  ▼
semantic_index.py
  │
  ├─[1] Manager.get_entity_relationships() → Neo4j (read): все связи RELATED
  ├─[1] Manager.get_entities()             → Neo4j (read): все сущности Entity
  ├─[1] Manager.get_community()            → Neo4j (read): все сообщества Community + CONSISTS_OF + IS_PARENT_OF
  │
  ├─[2] create_community_report.run_community_reports_pipeline_async()
  │     ├─ Этап 1: explode_communities, _prep_nodes, _prep_edges
  │     ├─ Этап 2: build_local_context() → sort_context() [LLM: count_tokens]
  │     │          └─ Tokenizer API (count_tokens)
  │     ├─ Этап 3: summarize_communities() → build_level_context() → _generate_report()
  │     │          ├─ Tokenizer API (count_tokens)
  │     │          └─ LLM API: COMMUNITY_REPORT_PROMPT → generate_structured (JSON)
  │     └─ Этап 4: finalize_community_reports()
  │
  ├─[3] Manager.update_community_reports()
  │     └─ Neo4j (write): MATCH Community SET report fields
  │
  └─ Response: {"document_id": "None", "statistics": {...}}
```

## LLM Interactions

| Промпт | Файл (должен быть) | Использование |
|--------|-------------------|---------------|
| `COMMUNITY_REPORT_PROMPT` | `prompts/community-report.md` | Генерация структурированного отчёта по сообществу. Переменные: `{input_text}` (CSV-строка с сущностями, связями и дочерними отчётами), `{max_report_length}` (ограничение длины в словах). Выход: JSON с полями `title`, `summary`, `rating`, `rating_explanation`, `findings[]` (каждый с `summary`, `explanation`). Парсится в `CommunityReportResponse` (Pydantic). |

**Текущее состояние (нарушение P7)**: промпт `COMMUNITY_REPORT_PROMPT` определён инлайн в `config.py` (строки 322–468). В соответствии с Constitution P7 должен быть вынесен в `semantic_graph/prompts/community-report.md`.

**Особенность**: промпт дублирован в `config.py` дважды — основное тело и повтор внутри Example Input/Output. Фактически `{input_text}` подставляется в последний блок `# Real Data`.

## LLM Model Requirements
- **Тип модели**: text-only (чат-модель с поддержкой JSON mode)
- **Минимальный размер контекста**: 8000 токенов (настраивается через `max_input_length`, по умолчанию 8000)
- **Язык выхода**: английский (промпт и ожидаемый ответ на английском)
- **Требования к формату выхода**: строгий JSON через `response_format={"type": "json_object"}`. Парсится в Pydantic-модель `CommunityReportResponse`:
  - `title: str`
  - `summary: str`
  - `rating: float` (0–10)
  - `rating_explanation: str`
  - `findings: list[FindingModel]` (каждый: `summary: str`, `explanation: str`)
- **Конкретная модель**: `MODEL_NAME = 'Qwen/Qwen3-4B-Instruct-2507'` (задаётся в `config.py`)
- **API**: OpenAI-совместимый эндпоинт (`LLM_URL = 'http://localhost:9886/v1'`)
- **Tokenizer**: отдельный HTTP-эндпоинт (`TOKENIZER_URL = 'http://localhost:9886/tokenize'`)

## Error Handling

| Сценарий | Код | Поведение |
|----------|-----|-----------|
| Neo4j недоступен при чтении | 500 | Unhandled exception → FastAPI 500 |
| Нет сообществ (граф не кластеризован) | 200 | `get_community()` возвращает пустой DataFrame → pipeline возвращает пустой результат → `updated: 0` |
| Нет сущностей | 200 | Пустые entities → `explode_communities` даст пустой DataFrame → нет контекстов → `updated: 0` |
| LLM API недоступен | 500 | Unhandled exception в `generate_structured` → FastAPI 500 |
| LLM вернул невалидный JSON | — | Логируется ошибка, `generate_structured` возвращает `None` → `_generate_report` возвращает `None` → сообщество пропускается |
| LLM вернул `None` (пустой ответ) | — | `extract()` возвращает `CommunityReportsResult(structured_output=None, output="")` → пропуск |
| Превышение контекста (контекст > max_input_tokens) | — | `build_mixed_context()` пытается заменить локальный контекст дочерними отчётами; если всё ещё превышает — используются только дочерние отчёты как CSV |
| Контекст не умещается даже с дочерними отчётами | — | `build_mixed_context()` возвращает максимально возможную строку дочерних отчётов, обрезая по токенам |
| Neo4j недоступен при записи | 500 | Unhandled exception в `update_community_reports` → FastAPI 500 |
| Сообщество с `community_id == -1` | — | Исключается на этапе `explode_communities`: `nodes.loc[nodes[COMMUNITY_ID] != -1]` |

## Testing

### Тест-кейсы
1. **Успешная генерация**: 5 сообществ, 20 сущностей, 30 связей → HTTP 200, `updated >= 1`, все отчёты содержат `title`, `summary`, `rating`, `findings`.
2. **Пустой граф**: нет сообществ → HTTP 200, `updated: 0`.
3. **Граф без кластеризации**: сущности есть, сообществ нет → HTTP 200, `updated: 0`.
4. **LLM возвращает невалидный JSON**: мок LLM с `{"invalid": "json"` → сообщество пропускается, отчёт не создаётся.
5. **LLM недоступен**: неверный `LLM_URL` → HTTP 500.
6. **Neo4j недоступен**: неверные креды → HTTP 500.
7. **Контроль контекста**: сообщество с 200+ сущностями → контекст обрезается по `max_input_tokens`.
8. **Иерархические отчёты**: сообщества на уровнях 0, 1, 2 → отчёты верхнего уровня включают `full_content` дочерних.
9. **Параллельная генерация**: `max_concurrent=4` → проверка, что семафор ограничивает одновременные запросы к LLM.

### Подход к тестированию
- **Unit**: `sort_context()` с фиктивными контекстами и мокнутым `llm.count_tokens()`; `_prep_nodes()`, `_prep_edges()` на DataFrame с NaN; `finalize_community_reports()`.
- **Integration**: `AsyncCommunityReportExtractor.extract()` с мокнутым `AsyncLLMClient` → проверка парсинга `CommunityReportResponse`.
- **Pipeline**: `run_community_reports_pipeline_async()` на фиктивных DataFrames с мокнутым LLM.
- **API**: httpx-запрос к эндпоинту с мокнутыми `Manager` и `AsyncLLMClient`.

## Dependencies
- **Neo4j** (read/write): `Manager.get_entity_relationships()`, `Manager.get_entities()`, `Manager.get_community()`, `Manager.update_community_reports()`
- **LLM API** (OpenAI-совместимый): `AsyncOpenAI` через `LLM_URL` — вызов `generate_structured()` с `response_format={"type": "json_object"}`
- **Tokenizer API**: HTTP POST на `TOKENIZER_URL` для подсчёта токенов контекста
- **Pydantic**: `CommunityReportResponse`, `FindingModel` — target-модели для structured output
- **pandas**: DataFrames на всех этапах — подготовка узлов/рёбер, построение контекста, агрегация, финализация
- **asyncio**: `Semaphore` для ограничения параллельных LLM-вызовов, `asyncio.gather`

## Exceptions

### P7 — Промпты инлайн (НАРУШЕНИЕ)
Промпт `COMMUNITY_REPORT_PROMPT` определён как строковая константа в `config.py` (строки 322–468, ~145 строк), а не в отдельном файле `prompts/community-report.md`. Промпт также содержит дублирование текста (основная инструкция и повтор в секции `# Real Data`). Требуется вынос в `prompts/community-report.md`.

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует, чтобы `archived`-сущности исключались из отчётов. В текущем коде:
- `get_entities()`, `get_entity_relationships()`, `get_community()` не фильтруют по `archived`.
- Поле `archived` отсутствует в схеме Neo4j.
- При генерации отчётов участвуют все сущности, включая те, что могли бы быть помечены как удалённые.

### P9 — Глобальный граф (НЕ НАРУШЕНИЕ)
Операция является always-full-graph, что прямо разрешено Constitution P9: «create_community_report — always-full-graph операция, запускается после накопления изменений».
