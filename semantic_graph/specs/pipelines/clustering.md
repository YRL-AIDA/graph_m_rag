# Pipeline: Graph Clustering (Community Detection)

## Purpose
Иерархическая кластеризация полного семантического графа знаний (всех Entity узлов и RELATED связей) с помощью алгоритма Leiden для обнаружения сообществ. Результат — иерархия Community узлов в Neo4j. Это **always-full-graph** операция (Constitution P9), запускается через эндпоинт `GET /clastrize_graph` в `semantic_graph/semantic_index.py`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Load Graph from Neo4j | — | `semantic_index.py:clastrize_graph()` → `doc_manager.get_entity_relationships()` → `Manager.get_entity_relationships()` (Cypher: `MATCH (source:Entity)-[r:RELATED]->(target:Entity) RETURN ...`). Возвращает DataFrame с колонками: `source` (формат `title\|type`), `target` (формат `title\|type`), `weight`, `description`, `combined_degree`, `text_unit_ids`, `id`. | `relations_df: pd.DataFrame` | sequential |
| 2. Hierarchical Leiden Clustering | `relations_df`, `max_cluster_size=10`, `use_lcc=False`, `seed=256` | `clasterization.py:create_communities()` → `cluster_graph()` → `_compute_leiden_communities()` → `hierarchical_leiden()`. **Алгоритм**: (a) Нормализация направления рёбер: `lo = min(source, target)`, `hi = max(source, target)`, дедупликация. (b) Сортировка edge list: `[(source, target, weight), ...]`. (c) Вызов `graspologic_native.hierarchical_leiden()` с параметрами: `max_cluster_size=10`, `resolution=1.0`, `randomness=0.001`, `use_modularity=True`, `iterations=1`. (d) Группировка результатов по уровням иерархии: `{level: {node_id: community_id}}`. (e) Построение parent mapping. (f) Формирование списка `Communities = [(level, community_id, parent, [node_titles]), ...]`. | `clusters: Communities` (list of tuples) | sequential (Leiden — CPU-bound) |
| 3. Build Communities DataFrame | `clusters` | `clasterization.py:create_communities()` — преобразование `clusters` в DataFrame, `explode("title")`, агрегация: (a) `entity_ids` — groupby `community` → список entity titles. (b) `relationship_ids` — для каждого уровня: merge relationships с communities (inner join по source→title и target→title), фильтрация intra-community рёбер, groupby, дедупликация списков. (c) Объединение: `all_grouped.merge(entity_ids)`. (d) Генерация полей: `id=uuid4()`, `human_readable_id=community`, `title="Community {N}"`, `parent`, `children` (groupby parent). (e) Добавление `period=сегодня`, `size=len(entity_ids)`. (f) Санитизация numpy типов `_sanitize_row()`. | `communities_rows: List[dict]` (соответствует `COMMUNITIES_FINAL_COLUMNS`) | sequential |
| 4. Write Communities to Neo4j | `communities_rows` | `semantic_index.py:clastrize_graph()` → `doc_manager.insert_communities_to_neo4j(communities_rows, batch_size=1000)`. **Двухэтапная загрузка**: **Этап 1**: UNWIND батчами, MERGE Community узлов с полями: `id` (uuid4), `short_id` (community_id), `title`, `community_id`, `community_level`, `community_parent`, `size`, `period`. **Этап 2**: создание связей (теми же батчами): (a) `IS_CHILD_OF`/`IS_PARENT_OF` между Community узлами по `community_parent`. (b) `CONSISTS_OF` между Community и Entity (разбор `entity_id_str` по `\|` на `title` и `type`, MERGE Entity узла). | Статистика: `{communities_created, parent_relations_created, entity_relations_created}` | sequential (batch write-транзакции) |

## Data Flow Diagram

```
GET /clastrize_graph
    │
    ▼
[Stage 1] Neo4j: MATCH (Entity)-[RELATED]->(Entity)
    │ relations_df: source, target, weight, description, text_unit_ids
    ▼
[Stage 2] Leiden Clustering (graspologic_native.hierarchical_leiden)
    │ max_cluster_size=10, resolution=1.0, use_modularity=True
    │
    ├──► Level 0: base communities
    ├──► Level 1: super-communities
    └──► Level N: top-level communities
    │
    ▼
[Stage 3] Build DataFrame: explode titles → aggregate entity_ids, relationship_ids
    │ add id (uuid4), title, parent, children, size, period
    ▼
[Stage 4] Neo4j: 2-phase batch write
    │ Phase 1: MERGE Community nodes (batch_size=1000)
    │ Phase 2: MERGE IS_CHILD_OF/IS_PARENT_OF + CONSISTS_OF relationships
    ▼
Response: { document_id: "None", statistics: { communities_created, parent_relations_created, entity_relations_created } }
```

## LLM Interactions

LLM на данном этапе **не используется**. Кластеризация — чисто алгоритмическая операция (Leiden algorithm через `graspologic_native`).

## Performance Constraints

- **Leiden parameters**: `max_cluster_size=10`, `resolution=1.0`, `randomness=0.001`, `use_modularity=True`, `iterations=1`.
- **use_lcc**: `False` — кластеризация всего графа без фильтрации largest connected component.
- **Seed**: `256` (фиксированный для детерминизма).
- **Neo4j batch size**: 1000 записей на транзакцию.
- **Leiden implementation**: нативный Rust-биндинг `graspologic_native` (CPU-bound, однопоточный).
- **Память**: весь граф загружается в pandas DataFrame. Может быть проблематично для очень больших графов.

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Load Graph | Neo4j недоступен | Исключение → HTTP 500 |
| 1. Load Graph | Пустой граф (нет Entity/RELATED) | Пустой DataFrame → Leiden обработает пустой edge list (вернёт пустой результат) |
| 2. Leiden | Ошибка graspologic_native | Исключение → HTTP 500 |
| 3. Build DataFrame | — | Операции pandas, ошибок не ожидается |
| 4. Write to Neo4j | Ошибка Neo4j на любом батче | Исключение → HTTP 500, частично записанные данные могут остаться |

**Примечание**: Constitution P9 — кластеризация работает со всем графом. Нет механизма инкрементального обновления. При ошибке на Stage 4 часть Community узлов может быть записана, часть — нет (транзакции на уровне батчей, не глобальная).
