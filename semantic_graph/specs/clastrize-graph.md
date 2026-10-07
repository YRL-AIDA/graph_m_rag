# Feature: Иерархическая кластеризация графа (clastrize_graph)

## Motivation
Построение иерархии сообществ (communities) на всём семантическом графе знаний. После накопления сущностей (`Entity`) и связей (`RELATED`) от одного или нескольких документов, этот эндпоинт запускает алгоритм иерархической кластеризации Лейдена (Leiden) для разбиения графа на вложенные сообщества. Результат — узлы `Community` в Neo4j с иерархическими связями `IS_CHILD_OF` / `IS_PARENT_OF` и связями `CONSISTS_OF` к сущностям. Это глобальная операция (Constitution P9) — всегда работает со всем графом, без фильтрации по документу.

## Behaviour

### Input
- **Метод**: `GET`
- **Path**: `/clastrize_graph`
- **Параметры**: нет. Все параметры кластеризации (`MAX_CLUSTER_SIZE`, `USE_LCC`, `CLUSTERIZATION_SEED`) заданы в `config.py`.
- **Валидация**: отсутствует (эндпоинт без параметров).

### Processing
1. **Получение связей из Neo4j**:
   - Вызывается `Manager.get_entity_relationships()` — Cypher-запрос, возвращающий все связи `RELATED` между узлами `Entity` как pandas DataFrame с колонками: `source` (формат `title|type`), `target` (формат `title|type`), `weight`, `description`, `combined_degree`, `text_unit_ids`, `id`.
   - При отсутствии связей возвращается пустой DataFrame с корректной схемой типов.

2. **Кластеризация**:
   - Вызывается `create_communities(relations_df, max_cluster_size=MAX_CLUSTER_SIZE, use_lcc=USE_LCC, seed=CLUSTERIZATION_SEED)` из `clasterization.py`:
     - **Шаг 2.1 — `cluster_graph()`**: нормализация рёбер (направление: `lo = min(source, target)`, `hi = max(source, target)`, дедупликация с `keep='last'`), приведение к неориентированному графу.
     - **Шаг 2.2 — `hierarchical_leiden()`**: вызов нативной библиотеки `graspologic_native.hierarchical_leiden()` с параметрами: `max_cluster_size=10`, `resolution=1.0`, `randomness=0.001`, `use_modularity=True`, `iterations=1`.
     - **Шаг 2.3 — Построение иерархии**: для каждого уровня кластеризации формируются списки узлов в каждом сообществе. Результат — `Communities = list[(level, community_id, parent_cluster_id, [node_ids])]`.
     - **Шаг 2.4 — Агрегация**: DataFrame сообществ, explode по `title` (сущностям). Для каждого уровня иерархии вычисляются:
       - `entity_ids`: список id сущностей в сообществе (группировка по `community`).
       - `relationship_ids`: список id внутрисообщенных связей (inner join relationships с сообществом по source и target, фильтрация где `community_x == community_y`).
       - `text_unit_ids`: агрегация `text_unit_ids` из внутрисообщенных связей.
     - **Шаг 2.5 — Финальные поля**: каждому сообществу присваивается `id = uuid4()`, `human_readable_id = community`, `title = "Community {community}"`, `children` (из обратной группировки по `parent`), `period = сегодняшняя дата ISO`, `size = len(entity_ids)`.
     - **Шаг 2.6 — Санация**: numpy-типы конвертируются в нативные Python через `_sanitize_row()`.
     - Возвращается `list[dict]` с колонками из `COMMUNITIES_FINAL_COLUMNS`: `id`, `human_readable_id`, `community`, `level`, `parent`, `children`, `title`, `entity_ids`, `relationship_ids`, `text_unit_ids`, `period`, `size`.

3. **Сохранение в Neo4j**:
   - Вызывается `Manager.insert_communities_to_neo4j(examples)` (двухэтапная загрузка):
     - **Этап 1**: батчами по 1000 создаются узлы `Community` через `MERGE` с полями: `id` (uuid4), `human_readable_id`, `title`, `community`, `level`, `parent`, `size`, `period`.
     - **Этап 2**: батчами по 1000 создаются связи:
       - `IS_CHILD_OF` / `IS_PARENT_OF` между Community (если `parent != -1`).
       - `CONSISTS_OF` между Community и Entity (MERGE Entity по `title|type`).
   - Возвращается словарь статистики: `communities_created`, `parent_relations_created`, `entity_relations_created`.

4. **Формирование ответа**:
   - Унифицированный ответ через `_build_response('None', 0, start_time, 'completed', neo4j_status)`.

### Output
- **HTTP 200** — успешная кластеризация:
  ```json
  {
    "document_id": "None",
    "statistics": {
      "total_chunks": 0,
      "processing_time_ms": <int>,
      "status": "completed",
      "communities_created": <int>,
      "parent_relations_created": <int>,
      "entity_relations_created": <int>
    }
  }
  ```
- **HTTP 500** — ошибка Neo4j или кластеризации (не обрабатывается явно, всплывёт как unhandled exception → FastAPI 500).

## API Contract

```openapi
GET /clastrize_graph
summary: Иерархическая кластеризация Лейдена всего графа знаний и сохранение сообществ в Neo4j
responses:
  '200':
    description: Кластеризация успешно завершена
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
                communities_created:
                  type: integer
                  description: Количество созданных узлов Community
                parent_relations_created:
                  type: integer
                  description: Количество созданных связей IS_CHILD_OF/IS_PARENT_OF
                entity_relations_created:
                  type: integer
                  description: Количество созданных связей CONSISTS_OF
  '500':
    description: Ошибка Neo4j или внутренняя ошибка кластеризации
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
  │ GET /clastrize_graph
  ▼
semantic_index.py
  │
  ├─[1] Manager.get_entity_relationships()
  │     └─ Neo4j (read): MATCH (source:Entity)-[r:RELATED]->(target:Entity)
  │        возвращает DataFrame с source, target, weight, id, description, combined_degree, text_unit_ids
  │
  ├─[2] clasterization.create_communities()
  │     ├─ cluster_graph()
  │     │   ├─ Нормализация рёбер (неориентированный граф, дедупликация)
  │     │   └─ graspologic_native.hierarchical_leiden()
  │     │       (CPU-bound, без I/O)
  │     ├─ Агрегация entity_ids, relationship_ids, text_unit_ids
  │     │   через pandas groupby + merge
  │     └─ Формирование финальных полей (uuid4, period, size, children, title)
  │
  ├─[3] Manager.insert_communities_to_neo4j()
  │     └─ Neo4j (write): MERGE Community, IS_CHILD_OF, IS_PARENT_OF, CONSISTS_OF
  │
  └─ Response: {"document_id": "None", "statistics": {...}}
```

## LLM Interactions
Отсутствуют. Кластеризация — чисто алгоритмическая операция (Leiden algorithm + pandas), не использующая LLM.

## LLM Model Requirements
Не применимо. LLM не используется.

## Error Handling

| Сценарий | Код | Поведение |
|----------|-----|-----------|
| Neo4j недоступен при чтении связей | 500 | Unhandled exception → FastAPI 500 Internal Server Error |
| Граф пуст (нет связей) | 200 | Пустой DataFrame → `hierarchical_leiden` с пустым списком рёбер → 0 сообществ → `communities_created: 0` |
| Ошибка в `graspologic_native` | 500 | Unhandled exception → FastAPI 500 |
| Neo4j недоступен при записи | 500 | Unhandled exception → FastAPI 500 |
| `LCC` включён, но граф не связный | — | `stable_lcc()` закомментирован в коде (`USE_LCC = False` по умолчанию), LCC-фильтрация не применяется |

## Testing

### Тест-кейсы
1. **Успешная кластеризация**: 50 сущностей, 100 связей → HTTP 200, `communities_created > 0`, сообщества на нескольких уровнях иерархии.
2. **Пустой граф**: 0 связей → HTTP 200, `communities_created: 0`.
3. **Граф из одного ребра**: 2 сущности, 1 связь → HTTP 200, минимум 1 сообщество 0-го уровня.
4. **Neo4j недоступен**: неверные креды → HTTP 500.
5. **Детерминированность**: одинаковый граф + `CLUSTERIZATION_SEED=256` → одинаковые сообщества при повторных вызовах.
6. **Иерархия**: граф из 100+ сущностей → сообщества на уровнях 0, 1, 2+ (проверка `level` и `parent`).

### Подход к тестированию
- **Unit**: `cluster_graph()` на фиктивном DataFrame рёбер; `_sanitize_row()` с numpy-типами.
- **Integration**: эндпоинт с мокнутым `Manager` (возвращает предзаданный DataFrame связей) → проверка структуры ответа.
- **Pipeline**: сквозной тест на тестовых данных Neo4j (testcontainers).

## Dependencies
- **Neo4j** (read/write): `Manager.get_entity_relationships()`, `Manager.insert_communities_to_neo4j()`
- **graspologic_native** (CPU): `hierarchical_leiden()` — нативная библиотека для кластеризации Лейдена
- **pandas**: DataFrames для связей и агрегации сообществ
- **numpy**: типы данных в промежуточных вычислениях (санируются в `_sanitize_row`)
- **uuid**: генерация уникальных id для узлов Community

## Exceptions

### P7 — Промпты инлайн (НЕ ПРИМЕНИМО)
Кластеризация не использует LLM, поэтому промпты отсутствуют. Нарушения P7 нет.

### P10 — Мягкое удаление не реализовано (НАРУШЕНИЕ)
Constitution P10 требует, чтобы удалённые сущности помечались `archived` и исключались из кластеризации. В текущем коде:
- `get_entity_relationships()` возвращает ВСЕ связи `RELATED` без фильтрации по `archived`.
- Поле `archived` на узлах `Entity` отсутствует.
- При полной очистке (`delete_all_documents`) используется `DETACH DELETE`.
- Добавление нового документа без перестроения сообществ может привести к неконсистентности, если какой-то документ был удалён.

### P9 — Глобальный граф (НЕ НАРУШЕНИЕ)
Операция является always-full-graph, что прямо разрешено Constitution P9: «clastrize_graph — always-full-graph операция, запускается после накопления изменений».
