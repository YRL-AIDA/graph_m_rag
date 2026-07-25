# Data Model: Document & DocumentIndexService

## Purpose
Модель `Document` представляет PDF-документ, полученный из MinerU, и предоставляет парсинг его структурированного контента в граф Neo4j (document graph). Функция `create_graph_from_mineru_result` — отдельно стоящий пайплайн, строящий список `Region`-объектов из JSON-результата MinerU. Класс `DocumentIndexService` — верхнеуровневый сервис, оркестрирующий запись графа через `Manager`.

**Зона ответственности**: граф структуры документа — узлы `Document`, `Region:*`, связи `ORDER` (порядок чтения) и `PARENT` (иерархия заголовков).

## Schema

### Document

```python
class Document:
    def __init__(self, json_data: Dict[str, Any], name: str, mode: str)
```

| Поле | Тип | Назначение |
|------|-----|-----------|
| `json_data` | `Dict[str, Any]` | Полный JSON-результат MinerU. При инициализации извлекается вложенный путь `json_data["results"]["result"]["results"]` |
| `name` | `str` | Имя документа — `file_hash` (MD5 содержимого PDF, P2 Constitution) |
| `mode` | `str` | Режим парсинга. Допустимо только `"mineru"`, иначе `ValueError` |

**Свойства**:

| Свойство | Тип | Назначение |
|----------|-----|-----------|
| `regions` | `List[Region]` | Доступ только для чтения. При `mode="mineru"` вызывает `create_graph_from_mineru_result(json_data, name)` |

**Методы**:

| Метод | Сигнатура | Назначение |
|-------|-----------|------------|
| `get_graph` | `() -> Dict[str, Any]` | Строит структуру графа: сортирует `self.regions` по `order`, вычисляет ORDER- и PARENT-рёбра, возвращает словарь `{"nodes": {...}, "edges": {...}}` |

### create_graph_from_mineru_result

```python
def create_graph_from_mineru_result(
    mineru_result: Dict[str, Any],
    document_name: str
) -> List[Region]:
```

| Параметр | Тип | Назначение |
|----------|-----|-----------|
| `mineru_result` | `Dict[str, Any]` | Полный JSON-результат MinerU (содержит ключ `content_list`) |
| `document_name` | `str` | Имя документа (`file_hash`) |
| **Возврат** | `List[Region]` | Список объектов `Region`, отсортированных по `order` |

### DocumentIndexService

```python
class DocumentIndexService:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, name_db: Optional[str] = None)
```

| Параметр | Источник по умолчанию | Назначение |
|----------|----------------------|-----------|
| `uri` | `f"neo4j://{os.environ.get('URL', 'localhost:7687')}"` | Neo4j connection URI |
| `user` | `os.environ.get('USER_NEO4J', 'neo4j')` | Имя пользователя |
| `password` | `os.environ.get('PASSWORD', '')` | Пароль |
| `name_db` | `os.environ.get('NAME_DB', 'neo4j')` | Имя БД Neo4j |

**Методы**:

| Метод | Возврат | Назначение |
|-------|---------|------------|
| `create_graph_from_mineru_result(mineru_result, file_hash)` | `bool` | Создаёт `Document` → `Manager.add_document()`. `True` при успехе, `False` если документ уже существует |
| `delete_graph(file_hash)` | `bool` | Удаляет граф документа через `Manager.delete_document()` |
| `is_document_indexed(file_hash)` | `bool` | Проверяет наличие через `Manager.is_document_exist()` |
| `get_status()` | `Dict[str, Any]` | `{"node_count": int}` |
| `get_related_context(file_hash, element_type, text)` | `Dict[str, Any]` | Запрос связанного контекста: для caption/footnote — родительский image/table; для image/table — дочерние caption/footnote |
| `delete_all_graphs()` | `bool` | DETACH DELETE всех узлов |
| `close()` | — | Закрытие Neo4j-соединения |
| `__enter__` / `__exit__` | — | Context manager |

## Processing (алгоритм)

### `create_graph_from_mineru_result`

1. Извлечь `content_list` из `mineru_result["content_list"]` (fallback к пустому списку, если ключ отсутствует).
2. Перебор элементов `content_list` с переменной `element_index = 0` — последовательным счётчиком:
   - **Пропустить**: не-`dict` элементы, тип `discarded`.
   - **`text`** (с `text_level == 1`): создать `Region` с `label="title"`, `text=f"Title: {text}"`, `element_data=text`. Если `text.strip() == ""` — пропустить.
   - **`text`** (с `text_level != 1`): создать `Region` с `label="text"`, `text=f"Text: {text}"`, `element_data=text`.
   - **`image`**: создать основной `Region` с `label="image"`, `text=""`, `image=img_path`, `element_data=img_path`. Если есть `image_caption` — создать отдельный `Region` с `label="image_caption"`, `text=f"Image Caption: ..."`. Если есть `image_footnote` — аналогично с `label="image_footnote"`, `text=f"Image Footnote: ..."`.
   - **`table`**: создать основной `Region` с `label="table"`, `text=f"Table: ..."` (конкатенация caption, body, footnotes). Если есть `table_caption`/`table_footnote` — отдельные `Region`-узлы аналогично image.
   - **`equation`**: создать `Region` с `label="equation"`, `text=f"Equation: {text}"`.
   - **Прочие типы**: как generic text — `label=element_type`, `text=f"{element_type}: {text}"`.
   - Все `Region` получают `style=Style(-1)`.
   - `element_index` инкрементируется только при создании узла.
3. Отсортировать итоговый список по `order`.
4. Вернуть `List[Region]`.

### `Document.get_graph`

1. Получить `regions = self.regions`.
2. Отсортировать `regions` по `order`.
3. Построить **ORDER-рёбра**: цепочка `[(-1, 0)] + [(i, i+1) for i in range(N-1)]`. Индекс `-1` обозначает узел `Document`.
4. Построить **PARENT-рёбра** (стековый алгоритм иерархии заголовков):
   - Стек `tmp_parent_list_id = [-1]` (корень — Document).
   - Для каждого региона с индексом `id_reg`:
     - Если `reg.is_content()` (`label in CONTENT_LABELS`) — привязать к текущей вершине стека (`parent_edges.append((test_parent_id, id_reg))`), **не** добавлять в стек.
     - Иначе (структурный элемент — title, equation, caption и т.д.):
       - Пока `not is_include_by_id(test_parent_id, id_reg)` — выталкивать из стека.
       - `is_include_by_id(parent_id, child_id)`: если `parent_id == -1` → всегда `True`; иначе `regions[parent_id].style < regions[child_id].style` (стиль родителя «меньше» стиля ребёнка — **инвертированный** оператор `__lt__`: `font_size_родителя > font_size_ребёнка + error_rate`).
       - Добавить ребро `(test_parent_id, id_reg)`, положить `id_reg` в стек.
5. Вернуть словарь `{"nodes": {"document": {"name": self.name}, "regions": {id: reg.to_dict() for ...}}, "edges": {"order": [...], "parental": [...]}}`.

### `DocumentIndexService.create_graph_from_mineru_result`

1. Создать `Document(json_data=mineru_result, name=file_hash, mode='mineru')`.
2. Вызвать `self.manager.add_document(document)`.
3. `Manager.add_document()`: проверяет `is_document_exist(name)`; если документ новый — вызывает `document.get_graph()`, сериализует в один Cypher-запрос (CREATE узлов Document + Region:label, затем CREATE связей ORDER и PARENT) через `conn.query()`.
4. Вернуть `True` (успех) или `False` (дубликат).

## Storage

Neo4j (Document Graph).

- **Узлы**:
  - `Document` — свойства: `{name: file_hash}`
  - `Region:{label}` — мульти-лейбл: всегда `Region` плюс конкретный тип (`title`, `text`, `image`, `image_caption`, `image_footnote`, `table`, `table_caption`, `table_footnote`, `equation`, или произвольный из generic-обработки).
  - Свойства Region-узла: `text` (str), `image` (str — путь к файлу), `bbox` (str — JSON-строка `{"x1":...,"y1":...,"x2":...,"y2":...}`), `style` (str — JSON-строка `{"font_size":..., "error_rate":...}`), `order` (int), `element_data` (str).

- **Связи**:
  - `(:Document) -[:ORDER]-> (:Region)` — первый элемент
  - `(:Region) -[:ORDER]-> (:Region)` — цепочка порядка чтения
  - `(:Document) -[:PARENT]-> (:Region)` — заголовок/элемент верхнего уровня
  - `(:Region) -[:PARENT]-> (:Region)` — иерархия: заголовок → подзаголовок, заголовок → контент

## Relationships

### ORDER
Образует **единую линейную цепочку** от `Document` через все `Region`-узлы в порядке их следования в `content_list` (после сортировки по `order`).

### PARENT
**Иерархическое дерево**, определяемое на основе сравнения стилей (размер шрифта заголовков):
- `Document` (`id=-1`) — корень, вмещает всё.
- `Region` с бóльшим `font_size` становится родителем последующих `Region` с меньшим `font_size`.
- Контентные элементы (`is_content() == True`: `text`, `table`, `image`) привязываются к текущему заголовку и не меняют стек.
- Алгоритм: LIFO-стек (последний заголовок с бóльшим шрифтом является текущим контекстом).

## Constraints

| Ограничение | Описание |
|-------------|----------|
| **Уникальность документа** | `Document.name` (`file_hash`) — уникальный идентификатор. `Manager.add_document()` проверяет существование перед созданием |
| **mode валидация** | Допустимо только `"mineru"`, иначе `ValueError('mode in ("mineru", ...)')` |
| **Пустой content_list** | `create_graph_from_mineru_result` возвращает пустой список `[]` если `content_list` отсутствует или пуст |
| **ORDER при N=0** | Если нет регионов, `order_edges = []` |
| **BBox fallback** | Если `bbox` не из 4 элементов — `BBox(0, 0, 0, 0)` |
| **Пустой текст пропускается** | Элементы с `text.strip() == ""` не создают Region |
| **Экранирование кавычек** | `Manager.add_document()` экранирует `'` → `\\'` в текстовых полях перед подстановкой в Cypher |
| **Дедупликация** | `is_document_exist(name)` → `False` если документ уже в БД |

## Examples

### Входные данные (MinerU content_list, фрагмент)

```python
mineru_result = {
    "content_list": [
        {"type": "text", "text": "Chapter 1", "text_level": 1, "bbox": [72, 100, 200, 120], "page_idx": 0},
        {"type": "text", "text": "This is a paragraph.", "text_level": 2, "bbox": [72, 140, 500, 180], "page_idx": 0},
        {"type": "image", "img_path": "/tmp/img1.png", "bbox": [72, 200, 300, 400], "page_idx": 0,
         "image_caption": ["Fig. 1: Diagram"], "image_footnote": ["Source: Author"]},
        {"type": "discarded", "text": "ignored"},
    ]
}
```

### Выход: `create_graph_from_mineru_result` → `List[Region]`

```python
[
    Region(text="Title: Chapter 1", image="", bbox=BBox(72, 100, 200, 120), style=Style(-1), order=0, label="title", element_data="Chapter 1"),
    Region(text="Text: This is a paragraph.", image="", bbox=BBox(72, 140, 500, 180), style=Style(-1), order=1, label="text", element_data="This is a paragraph."),
    Region(text="", image="/tmp/img1.png", bbox=BBox(72, 200, 300, 400), style=Style(-1), order=2, label="image", element_data="/tmp/img1.png"),
    Region(text="Image Caption: Fig. 1: Diagram", image="/tmp/img1.png", bbox=BBox(72, 200, 300, 400), style=Style(-1), order=3, label="image_caption", element_data="Fig. 1: Diagram"),
    Region(text="Image Footnote: Source: Author", image="/tmp/img1.png", bbox=BBox(72, 200, 300, 400), style=Style(-1), order=4, label="image_footnote", element_data="Source: Author"),
]
```

### `Document.get_graph()` — структура графа

```python
{
    "nodes": {
        "document": {"name": "abc123"},
        "regions": {
            0: {"label": "title", "text": "Title: Chapter 1", "image": "", "bbox": {...}, "style": {...}, "order": 0, "element_data": "Chapter 1"},
            1: {"label": "text",  "text": "Text: This is a paragraph.", ...},
            2: {"label": "image", "text": "", "image": "/tmp/img1.png", ...},
            3: {"label": "image_caption", ...},
            4: {"label": "image_footnote", ...},
        }
    },
    "edges": {
        "order": [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)],
        "parental": [(-1, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    }
}
```

PARENT-рёбра здесь: `title` (index 0) становится родителем всех последующих элементов, потому что `Style(-1) < Style(-1)` всегда `False` (одинаковый font_size), и стек не меняется на контентных элементах — все привязываются к последнему заголовку.

### Иерархия с разными размерами шрифта (псевдо)

Если бы у title был `Style(24)`, а у последующего header — `Style(18)`:
- `is_include_by_id` работает инвертированно: `Style(24) < Style(18)` → `24 > 18 + 1` → `True`. Значит, родитель вмещает ребёнка.
- При `Style(12)` для подзаголовка: `Style(18) < Style(12)` → `18 > 12 + 1` → `True`.
- При появлении заголовка с `Style(24)` снова: `Style(12) < Style(24)` → `12 > 24 + 1` → `False` → всплытие стека до подходящего родителя.

## Error Handling

| Ситуация | Поведение |
|----------|-----------|
| `mode != "mineru"` | `ValueError('mode in ("mineru", ...)')` при инициализации `Document` |
| `mode != "mineru"` в `regions` | `ValueError('there is not parser for this mode ')` (защита на будущее) |
| `content_list` отсутствует в `mineru_result` | `create_graph_from_mineru_result` использует пустой список, возвращает `[]` |
| Элемент не является `dict` | Пропускается (`continue`) |
| `bbox` не из 4 элементов | Fallback к `BBox(0, 0, 0, 0)` |
| Документ уже существует | `Manager.add_document()` возвращает `False`, логирует WARNING |
| Ошибка Neo4j-запроса | `Manager.query()` логирует ERROR и пробрасывает исключение |
| `DocumentIndexService.create_graph_from_mineru_result` исключение | Логирует ERROR, пробрасывает исключение |
| `Manager.delete_document`: документ не найден | Возвращает `False`, логирует ERROR и перехватывает исключение |
| `Manager.delete_all_documents`: ошибка | Возвращает `False`, логирует ERROR |

## Testing

| Тест-кейс | Проверка |
|-----------|----------|
| `Document` с `mode="mineru"`, валидным `json_data` | `regions` возвращает `List[Region]` |
| `Document` с `mode="unknown"` | `ValueError` при инициализации |
| `create_graph_from_mineru_result` с пустым `content_list` | Возвращает `[]` |
| `create_graph_from_mineru_result` с `text` без `text_level` | `label="text"` |
| `create_graph_from_mineru_result` с `text`, `text_level=1` | `label="title"`, text префикс `"Title: "` |
| `create_graph_from_mineru_result` с `text`, пустая строка | Region не создаётся |
| `create_graph_from_mineru_result` с `image` без caption/footnote | Один Region с `label="image"` |
| `create_graph_from_mineru_result` с `image` + caption + footnote | Три Region: `image`, `image_caption`, `image_footnote` |
| `create_graph_from_mineru_result` с `table` + caption + footnote | Три Region |
| `create_graph_from_mineru_result` с `equation` | `label="equation"`, текст с префиксом |
| `create_graph_from_mineru_result` с `discarded` | Пропускается |
| `create_graph_from_mineru_result` с неизвестным типом | `label=<element_type>`, текст с префиксом |
| `create_graph_from_mineru_result` с `bbox` из 3 элементов | `BBox(0, 0, 0, 0)` |
| `Document.get_graph()`: 0 регионов | `order_edges=[]`, `parental_edges=[]` |
| `Document.get_graph()`: все один стиль | Все `parental` к последнему заголовку |
| `Document.get_graph()`: иерархия заголовков | Проверка стека: родитель с бóльшим шрифтом |
| `DocumentIndexService.create_graph_from_mineru_result`: новый документ | `True` |
| `DocumentIndexService.create_graph_from_mineru_result`: повторный вызов | `False` |
| `DocumentIndexService.delete_graph`: существующий | `True` |
| `DocumentIndexService.delete_graph`: несуществующий | `False` |
| `Manager.add_document`: экранирование `'` | Cypher не ломается |

## Dependencies

| Модуль/Библиотека | Назначение |
|-------------------|------------|
| `documet_index/dtype/region.py` | `BBox`, `Style`, `Region`, `CONTENT_LABELS` |
| `documet_index/manager.py` | `Manager`, `ManagerConfig`, `Neo4jConnection` |
| `documet_index/neo4j_service.py` | `DocumentIndexService`, `create_neo4j_graph` |
| `neo4j` (GraphDatabase) | Neo4j Bolt-драйвер |
| `python-dotenv` | Загрузка `.env` для `DocumentIndexService` |
| `os`, `logging`, `json` | Стандартная библиотека |

## Exceptions

1. **Нарушение P6 (Pydantic)**: `Document` — plain Python класс с `__init__`, а не Pydantic `BaseModel`. Типы параметров (`json_data: Dict`, `name: str`, `mode: str`) не валидируются на уровне конструктора (только `mode` проверяется через условный оператор, но без декораторов валидации). **Обоснование**: переходный период — код написан до принятия Constitution. Должно быть исправлено: преобразовать в Pydantic-модель с `@field_validator` для `mode` и `json_data`.

2. **Нарушение P6 (Pydantic)**: `ManagerConfig` — plain Python класс. **Обоснование**: переходный период.

3. **Нарушение P5 (асинхронность)**: `Manager.query()` использует синхронный `session.run()`. Весь `DocumentIndexService` синхронный. **Обоснование**: переходный период — драйвер `neo4j` поддерживает async API. Требуется миграция на `AsyncGraphDatabase`.

4. **Мёртвый код в `Document.__parser_mineru`**: после `return create_graph_from_mineru_result(json_data, self.name)` (строка 25) идёт недостижимый код (строки 27–39) — старая реализация парсинга, оставленная для истории. **Рекомендация**: удалить при рефакторинге.

5. **Cypher-инъекция**: `Manager.add_document()` строит Cypher-запрос конкатенацией строк с ручным экранированием `' → \\'`. Поля `text`, `image`, `element_data` не проходят параметризацию. **Рекомендация**: перейти на параметризованные запросы Neo4j (`$param`).
