# Data Model: Document (MinerU Parser)

## Purpose

Модели для парсинга результата MinerU (JSON с `content_list`) и построения графа структуры документа: извлечение регионов (текст, заголовки, изображения, таблицы, формулы), построение порядка чтения (`ORDER`) и иерархии заголовков (`PARENT`).

**Используется в**: `semantic_graph/neo4j_service.py` (`DocumentIndexService`), не в основном пайплайне `semantic_index.py`. Парсинг MinerU-результата для индексации документа в Neo4j (document graph).

**Важно**: эти модели **дублируют** `documet_index/dtype/document.py` — см. [Exceptions](#exceptions).

## Schema

### Document

```python
class Document:
    """Парсер результата MinerU для построения графа структуры документа."""

    def __init__(self, json_data: dict, name: str, mode: str):
        """
        Args:
            json_data: Полный ответ MinerU (содержит 'results' -> 'result' -> 'results')
            name:     Имя документа (file_hash, P2)
            mode:     Режим парсинга ('mineru' — единственный поддерживаемый)
        Raises:
            ValueError: если mode не равен 'mineru'
        """
        self.mode = mode
        self.name = name
        self.json_data = json_data["results"]["result"]["results"]

    @property
    def regions(self) -> List[Region]:
        """Извлекает список Region из MinerU content_list через create_graph_from_mineru_result."""
        if self.mode == 'mineru':
            return self.__parser_mineru(self.json_data)
        raise ValueError('there is not parser for this mode')

    def __parser_mineru(self, json_data: dict) -> List[Region]:
        """
        Вызывает create_graph_from_mineru_result(json_data, self.name).
        Фактический код делегирует парсинг внешней функции, оставляя собственный
        цикл (строки 26-38) недостижимым (dead code после return).
        """
        regions = []
        return create_graph_from_mineru_result(json_data, self.name)
        # Код ниже недостижим (dead code) — return выше прерывает выполнение
        ...

    def get_graph(self) -> dict:
        """
        Строит граф документа: ORDER-рёбра (порядок чтения) и PARENT-рёбра (иерархия заголовков).
        Returns:
            {
                "nodes": {
                    "document": {"name": str},
                    "regions": {id_reg: region_dict, ...}
                },
                "edges": {
                    "order": [(parent_id, child_id), ...],
                    "parental": [(parent_id, child_id), ...]
                }
            }
        """
```

### create_graph_from_mineru_result

```python
def create_graph_from_mineru_result(
    mineru_result: Dict[str, Any],
    document_name: str
) -> List[Region]:
    """
    Строит список Region из content_list MinerU для всех типов элементов.

    Обрабатываемые типы элементов:
    - text / title:  text с text_level == 1 → label="title", иначе label="text"
    - image:         label="image" + опционально "image_caption", "image_footnote"
    - table:         label="table" + опционально "table_caption", "table_footnote"
    - equation:      label="equation"
    - discarded:     пропускается
    - прочие:        label = element_type, text = "{type}: {text}"

    После создания список сортируется по order и возвращается (без рёбер,
    в отличие от Document.get_graph()).

    Args:
        mineru_result:  Словарь MinerU (с ключом "content_list")
        document_name:  Имя документа

    Returns:
        Список Region, отсортированный по order
    """
```

### Вспомогательные классы внутри функции

Функция `create_graph_from_mineru_result` содержит локальные переменные и вложенную функцию `is_include_by_id` для построения иерархии заголовков (parent edges). Эти структуры не являются экспортируемыми моделями, но образуют промежуточное представление графа:

```python
# Локально внутри функции
order_edges: List[Tuple[int, int]]   # Рёбра порядка чтения: (-1, 0) + (i, i+1)
parent_edges: List[Tuple[int, int]]  # Рёбра иерархии: заголовок → подчинённые
tmp_parent_list_id: List[int]        # Стек заголовков для отслеживания вложенности; -1 = Document
```

## Storage

- **In-memory**: `Document`, `Region`, `Style`, `BBox` — обычные классы Python (не Pydantic), создаются в рантайме при вызове `DocumentIndexService` из `neo4j_service.py`.
- **Neo4j** (document graph): результат `get_graph()` записывается в узлы `Document`, `Region:*` и связи `ORDER`, `PARENT` через `documet_index.Manager`. Сами модели `Document`, `Region` не пишутся напрямую в БД.

## Relationships

| Модель | Связана с | Характер связи |
|--------|-----------|---------------|
| `Document` | `Region` (из `dtype/region.py`) | 1:N — документ содержит множество регионов |
| `Document.name` | P2 `file_hash` | Идентификатор документа во всех хранилищах |
| `create_graph_from_mineru_result` | `Region`, `BBox`, `Style` | Фабрика регионов из MinerU JSON |
| `Document.get_graph()` | `documet_index.Manager` | Результат — графовая структура для записи в Neo4j document graph |

## Constraints

| Ограничение | Детали |
|-------------|--------|
| `mode` | Только `"mineru"`. Любое другое значение → `ValueError` |
| `json_data` | Ожидает структуру `{"results": {"result": {"results": ...}}}` (вложенный путь MinerU) |
| `content_list` | Каждый элемент обязан быть `dict`. Не-dict элементы пропускаются |
| Типы элементов | `discarded` пропускается; пустые тексты (`""`, только пробелы) пропускаются для `text` и `equation` |
| `bbox` | Ожидается список из 4 чисел `[x1, y1, x2, y2]`. Если длина ≠ 4 → `BBox(0, 0, 0, 0)` |
| `element_index` | Сквозной счётчик, не зависит от оригинального индекса `i` в `content_list` (из-за пропуска `discarded` и пустых) |
| Dead code | Собственный цикл в `Document.__parser_mineru` (строки 26-38) недостижим — `return` на строке 25 прерывает выполнение |

## Examples

### MinerU JSON (фрагмент content_list)

```json
{
  "results": {
    "result": {
      "results": {
        "content_list": [
          {
            "type": "text",
            "text": "Глава 1. Введение",
            "text_level": 1,
            "bbox": [72, 100, 540, 120],
            "page_idx": 0
          },
          {
            "type": "text",
            "text": "Это первый параграф документа.",
            "bbox": [72, 140, 540, 160],
            "page_idx": 0
          },
          {
            "type": "image",
            "img_path": "images/page_0_img_0.jpeg",
            "bbox": [100, 200, 500, 400],
            "page_idx": 0,
            "image_caption": ["Рис. 1: Архитектура системы"],
            "image_footnote": []
          },
          {
            "type": "table",
            "img_path": "images/page_0_table_0.jpeg",
            "table_body": "<table>...</table>",
            "table_caption": ["Таблица 1: Результаты"],
            "table_footnote": ["* данные за 2023 год"],
            "bbox": [100, 420, 500, 600],
            "page_idx": 0
          },
          {
            "type": "discarded",
            "text": "колонтитул",
            "bbox": [0, 0, 0, 0],
            "page_idx": 0
          }
        ]
      }
    }
  }
}
```

### Создание Document и построение графа

```python
doc = Document(json_data=mineru_response, name="a1b2c3d4e5f6", mode="mineru")
regions = doc.regions
# regions[0]: Region(label="title", text="Title: Глава 1. Введение", order=0)
# regions[1]: Region(label="text",  text="Text: Это первый параграф документа.", order=1)
# regions[2]: Region(label="image", image="images/page_0_img_0.jpeg", order=2)
# regions[3]: Region(label="image_caption", text="Image Caption: Рис. 1: Архитектура системы", order=3)
# regions[4]: Region(label="table", text="Table: | ['Таблица 1: Результаты'] | <table>...</table> | ['* данные за 2023 год']", order=4)
# regions[5]: Region(label="table_caption", text="Table Caption: Таблица 1: Результаты", order=5)
# regions[6]: Region(label="table_footnote", text="Table Footnote: * данные за 2023 год", order=6)
# discarded — пропущен

graph = doc.get_graph()
# graph["nodes"]["document"] → {"name": "a1b2c3d4e5f6"}
# graph["edges"]["order"] → [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
# graph["edges"]["parental"] → [(-1, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
```

## Exceptions

| Отклонение | Обоснование |
|------------|-------------|
| **Дубликат `documet_index/dtype/document.py` (P1, P6)** | Файл `semantic_graph/dtype/document.py` побайтово идентичен `documet_index/dtype/document.py`. Это нарушает P1 (сервисные границы — document-парсинг принадлежит `documet_index`) и P6 (единый источник истины). Код `semantic_graph` использует `Document` в `neo4j_service.py` (`DocumentIndexService`), который сам дублирует функциональность `documet_index`. Требуется удаление дубликата и использование моделей из `documet_index`. |
| **Не-Pydantic модели (P6)** | `Document` и `create_graph_from_mineru_result` — обычные классы, не Pydantic. Это исключение оправдано тем, что `Document` — внутренний парсер, а не контракт между сервисами. Однако при рефакторинге следует рассмотреть перенос логики в `documet_index`. |
| **Dead code** | `Document.__parser_mineru` содержит недостижимый цикл (строки 26-38) после `return` на строке 25. Мёртвый код должен быть удалён при рефакторинге. |
| **Несоответствие сигнатуры `Region` в dead code** | Недостижимый код `__parser_mineru` вызывает `Region(text, bbox, Style, order, label)` с 5 аргументами, в то время как актуальный `Region.__init__` требует 7 (`text, image, bbox, style, order, label, element_data`). Это не вызывает ошибок только потому, что код недостижим. |
