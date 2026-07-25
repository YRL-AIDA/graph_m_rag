# Pipeline: Document to Graph (MinerU JSON → Neo4j Document Graph)

## Purpose
Преобразование структурированного результата MinerU (JSON с `content_list`) в граф структуры документа в Neo4j. Создаёт `Document` и `Region:*` узлы со связями `ORDER` (последовательность чтения) и `PARENT` (иерархия заголовков). Вызывается из ingestion pipeline (Stage 10) через `documet_index/create_neo4j_graph()`.

## Stages

| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. Parse MinerU JSON | `mineru_result: Dict`, `document_name: str` (file_hash) | `documet_index/dtype/document.py:create_graph_from_mineru_result(mineru_result, document_name)` — извлечение `content_list` из `mineru_result["content_list"]` (ожидается, что передан уже `results.result.results`). | `content_list: List[Dict]` | sequential |
| 2. Create Region Objects | `content_list` | `create_graph_from_mineru_result()` — итерация с инкрементальным `element_index`. Для каждого элемента по типу: **text**: если `text_level==1` → `label="title"`, текст `f"Title: {text}"`; иначе `label="text"`, текст `f"Text: {text}"`. Элементы с пустым текстом пропускаются. **image**: создаётся основной узел `label="image"` с `image=img_path`; если есть `image_caption` → дополнительный узел `label="image_caption"`; если есть `image_footnote` → `label="image_footnote"`. **table**: основной узел `label="table"` с текстом `f"Table: \| {captions} \| {body} \| {footnotes}"`; дополнительные узлы для caption и footnote. **equation**: `label="equation"`, текст `f"Equation: {latex}"`. **discarded**: пропускается. Каждый `Region` содержит: `text`, `image` (путь), `bbox` (BBox), `style` (Style(-1)), `order` (element_index), `label`, `element_data`. | `regions: List[Region]` (отсортированы по `order`) | sequential |
| 3. Build ORDER Edges | `regions` | `Document.get_graph()` — построение рёбер порядка чтения: `order_edges = [(-1, 0)] + [(i, i+1) for i in range(N-1)]`, где `-1` означает `Document` узел. Если N=0 → пустой список. | `order_edges: List[Tuple[int, int]]` | sequential |
| 4. Build PARENT Edges | `regions` | `Document.get_graph()` — построение иерархических связей на основе стека `tmp_parent_list_id` (начинается с `[-1]`). Для каждого region: (a) Если `reg.is_content()` → привязка к текущему parent (верхушка стека). (b) Если заголовок (title): `is_include_by_id(parent, child)` проверяет `regions[parent].style > regions[child].style`; пока условие ложно — pop из стека; затем создаётся PARENT-связь, и заголовок добавляется в стек. (c) `-1` (Document) всегда включает любой элемент. **Важно**: `Style(-1)` инициализируется для всех регионов, поэтому логика `style > style` всегда даёт `False` — элементы помещаются под последний заголовок в стеке. | `parent_edges: List[Tuple[int, int]]` | sequential |
| 5. Write to Neo4j | `Document`, `regions`, `order_edges`, `parent_edges` | `documet_index/manager.py:Manager.add_document()` — (a) Проверка `is_document_exist(name)` через `OPTIONAL MATCH (d:Document) RETURN '{name}' in d.name`. (b) Если не существует: формирование одного большого Cypher-запроса: `CREATE (d:Document {name: '{name}'})`, затем для каждого региона — `CREATE (reg{N}:Region:{label} {text, image, bbox, style, order, element_data})`, затем `CREATE (node1)-[:ORDER]->(node2)` и `CREATE (node1)-[:PARENT]->(node2)`. Все текстовые поля экранируются (одинарные кавычки → `\'`). | `True` (документ добавлен) или `False` (уже существует) | sequential (один Cypher-запрос) |

## Data Flow Diagram

```
mineru_result + file_hash
    │
    ▼
[Stage 1] Parse: extract content_list from mineru_result
    │
    ▼
[Stage 2] Create Region objects per element type
    │
    ├──► text (text_level==1 → title, else → text)
    ├──► image → image + image_caption + image_footnote
    ├──► table → table + table_caption + table_footnote
    ├──► equation → equation
    └──► discarded → SKIP
    │
    ▼
[Stage 3] Build ORDER edges: sequential chain [-1→0, 0→1, ..., N-1]
    │ -1 = Document node
    ▼
[Stage 4] Build PARENT edges: stack-based hierarchy
    │ is_content()? → attach to current parent
    │ is_title()?   → pop stack until valid parent, then push to stack
    ▼
[Stage 5] Neo4j: single Cypher query
    │ CREATE (d:Document {name})
    │ CREATE (regN:Region:label {...})
    │ CREATE (:Document)-[:ORDER]->(:Region)
    │ CREATE (...)-[:PARENT]->(...)
    ▼
Result: Document + Region:* nodes with ORDER + PARENT relationships
```

## LLM Interactions

LLM на данном этапе **не используется**. Преобразование MinerU JSON → Neo4j Graph — чисто алгоритмическая операция.

## Performance Constraints

- **Cypher**: один большой запрос (не батчевый), количество операций = O(regions).
- **Экранирование**: строковые поля экранируются через `.replace("'", "\\'")`. Сложный текст может вызвать проблемы с Cypher-синтаксисом.
- **Duplicate check**: один запрос перед созданием. **Важно**: текущая реализация `is_document_exist` использует `OPTIONAL MATCH (d:Document) RETURN '{name}' in d.name as exist` — это может работать некорректно (сравнивает строку, а не свойство).

## Error Recovery

| Stage | Failure Mode | Recovery |
|-------|-------------|----------|
| 1. Parse | content_list отсутствует или пуст | Пустой список → пустой граф, `add_document` создаст только Document узел |
| 2. Create Regions | Некорректный bbox (не 4 элемента) | `BBox(0,0,0,0)` как fallback |
| 3-4. Build Edges | N=0 (нет регионов) | Пустые списки рёбер |
| 5. Write to Neo4j | Документ уже существует | `add_document()` возвращает `False`, логируется warning |
| 5. Write to Neo4j | Ошибка Neo4j | Исключение → пробрасывается в вызывающий код (ingestion pipeline логирует и продолжает) |

**Примечание**: В ingestion pipeline (`api.py:upload_pdf()`) ошибка Neo4j на Stage 10 не прерывает pipeline — документ остаётся доступным в MinIO и Qdrant. `neo4j_graph_created` выставляется в `False`.
