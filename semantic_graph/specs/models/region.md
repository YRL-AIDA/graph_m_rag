# Data Model: Region, Style, BBox

## Purpose

Базовые классы для представления элементов документа (регионов), извлечённых из результата парсинга MinerU. Используются `Document` при построении графа структуры документа: каждый регион описывает один структурный элемент (текст, заголовок, изображение, таблицу, формулу) с его координатами, стилем и порядковым номером.

**Используется в**: `semantic_graph/dtype/document.py` → `Document.get_graph()`, `create_graph_from_mineru_result`.

**Важно**: эти классы **дублируют** `documet_index/dtype/region.py` — см. [Exceptions](#exceptions).

## Schema

### BBox

```python
class BBox:
    """Ограничивающий прямоугольник (bounding box) элемента на странице."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1  # Левая координата
        self.y1 = y1  # Верхняя координата
        self.x2 = x2  # Правая координата
        self.y2 = y2  # Нижняя координата

    @property
    def width(self) -> float:
        """Ширина bbox: x2 - x1."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Высота bbox: y2 - y1."""
        return self.y2 - self.y1
```

### Style

```python
class Style:
    """Стилевые атрибуты региона (размер шрифта, с допуском на ошибку)."""

    def __init__(self, font_size: float):
        self.font_size = font_size  # Размер шрифта (из MinerU); -1 используется как "неизвестно"
        self.error_rate = 1         # Допуск ошибки для сравнения размеров шрифта

    def __lt__(self, other: "Style") -> bool:
        """
        Сравнение стилей для построения иерархии заголовков.
        Style A < Style B, если font_size(A) > font_size(B) + error_rate.
        То есть более крупный шрифт (заголовок) считается "меньше" в смысле иерархии:
        он может быть родителем для элементов с более мелким шрифтом.

        Пример:
            Style(16) < Style(12) → True   (16 > 12 + 1 → родитель)
            Style(12) < Style(14) → False  (12 не больше 14 + 1)
            Style(13) < Style(12) → False  (13 > 12 + 1 = 13, не строго больше)
        """
        if self.font_size > other.font_size + self.error_rate:
            return True
        return False
```

### Region

```python
class Region:
    """Элемент документа — текст, заголовок, изображение, таблица или формула."""

    def __init__(
        self,
        text: str,            # Текстовое содержимое (с префиксом типа: "Text: ...", "Title: ...")
        image: str,           # Путь к файлу изображения (для image/table элементов), иначе ""
        bbox: BBox,           # Координаты элемента на странице
        style: Style,         # Стиль (размер шрифта)
        order: int,           # Порядковый номер в документе (после фильтрации)
        label: str,           # Метка типа: "title", "text", "image", "image_caption",
                              #   "image_footnote", "table", "table_caption",
                              #   "table_footnote", "equation" или оригинальный element_type
        element_data: str     # Оригинальные данные элемента (текст, путь к изображению, тип)
    ):
        self.text = text
        self.image = image
        self.bbox = bbox
        self.style = style
        self.order = order
        self.label = label
        self.element_data = element_data

    def is_content(self) -> bool:
        """
        Проверяет, является ли регион контентным (текст, таблица, изображение),
        а не заголовком. Используется при построении parent-рёбер: контент
        прикрепляется к текущему заголовку.

        Returns:
            True, если label входит в CONTENT_LABELS (['text', 'table', 'image'] из config.py)
        """
        return self.label in CONTENT_LABELS

    def to_dict(self) -> dict:
        """
        Сериализация региона в словарь для graph-представления.
        Returns:
            {
                "label": str,
                "text": str,
                "image": str,
                "bbox": {"x1": float, "y1": float, "x2": float, "y2": float} | {},
                "style": {"font_size": float, "error_rate": float} | {},
                "order": int,
                "element_data": str
            }
        """
```

### CONTENT_LABELS (из `config.py`)

```python
CONTENT_LABELS = ['text', 'table', 'image']
```

Импортируется в `region.py` через `from config import CONTENT_LABELS`. Определяет, какие типы регионов считаются контентом (прикрепляются к родительскому заголовку), а какие — заголовками (создают новый уровень иерархии).

## Storage

- **In-memory**: `BBox`, `Style`, `Region` — обычные Python-классы (не Pydantic), создаются в рантайме при парсинге MinerU-результата.
- **Neo4j**: не пишутся напрямую. Результат `Region.to_dict()` используется в `Document.get_graph()` и затем записывается в Neo4j через `documet_index.Manager`.

## Relationships

| Модель | Связана с | Характер связи |
|--------|-----------|---------------|
| `Region` | `BBox` | 1:1 — каждый регион имеет один bounding box |
| `Region` | `Style` | 1:1 — каждый регион имеет один стиль |
| `Region` | `Document` | N:1 — документ содержит множество регионов |
| `Style.__lt__` | `Style` | Сравнение для построения иерархии заголовков |
| `Region.is_content()` | `CONTENT_LABELS` | Проверка принадлежности label к контентным типам |

## Constraints

| Ограничение | Детали |
|-------------|--------|
| `BBox` координаты | Числовые значения, знак не проверяется. width/height могут быть отрицательными при некорректных данных |
| `Style.font_size` | Может быть `-1` (неизвестно). Используется во всех регионах из `create_graph_from_mineru_result` |
| `Style.error_rate` | Жёстко закодирован как `1`, не настраивается |
| `Style.__lt__` | Нестандартная семантика: A < B означает "A может быть родителем B". Путает стандартное поведение `<` |
| `Region.element_data` | Содержимое зависит от типа: для text — исходный текст, для image — img_path, для equation — строка `"equation"` (не формула), для table — составная строка из caption+body+footnote |
| `Region.label` | Для неподдерживаемых типов используется оригинальный `element_type` из MinerU |
| `CONTENT_LABELS` | Импортируется из `config.py`, а не определён локально (в отличие от `documet_index/dtype/region.py`, где список захардкожен) |
| Пустой bbox | При некорректном bbox (длина ≠ 4) создаётся `BBox(0, 0, 0, 0)`. `to_dict()` возвращает `{}` если `self.bbox` falsy |

## Examples

### Создание Region

```python
from semantic_graph.dtype.region import BBox, Style, Region

# Текстовый регион
region = Region(
    text="Text: Это параграф документа.",
    image="",
    bbox=BBox(72, 140, 540, 160),
    style=Style(-1),
    order=1,
    label="text",
    element_data="Это параграф документа."
)

region.is_content()  # → True (label="text" ∈ CONTENT_LABELS)
region.to_dict()
# {
#     "label": "text",
#     "text": "Text: Это параграф документа.",
#     "image": "",
#     "bbox": {"x1": 72, "y1": 140, "x2": 540, "y2": 160},
#     "style": {"font_size": -1, "error_rate": 1},
#     "order": 1,
#     "element_data": "Это параграф документа."
# }

# Заголовок
title_region = Region(
    text="Title: Глава 1",
    image="",
    bbox=BBox(72, 100, 540, 120),
    style=Style(-1),
    order=0,
    label="title",
    element_data="Глава 1"
)
title_region.is_content()  # → False (label="title" ∉ CONTENT_LABELS)
```

### Сравнение Style

```python
h1 = Style(18)   # Крупный заголовок
h2 = Style(14)   # Подзаголовок
body = Style(10) # Основной текст

h1 < h2     # True  — h1 может быть родителем h2 (18 > 14 + 1 = 15)
h2 < body   # True  — h2 может быть родителем body (14 > 10 + 1 = 11)
h1 < body   # True  — h1 может быть родителем body (18 > 10 + 1 = 11)
body < h1   # False — body не может быть родителем h1 (10 не больше 18 + 1)
```

### BBox

```python
bbox = BBox(100, 200, 400, 350)
bbox.width   # → 300
bbox.height  # → 150
```

## Exceptions

| Отклонение | Обоснование |
|------------|-------------|
| **Дубликат `documet_index/dtype/region.py` (P1, P6)** | `semantic_graph/dtype/region.py` дублирует `documet_index/dtype/region.py` с единственным отличием: импорт `CONTENT_LABELS` из `config.py` вместо локального определения. Это нарушает P1 (сервисные границы — модели регионов принадлежат `documet_index`) и P6 (единый источник истины). Требуется удаление дубликата и импорт из `documet_index`. |
| **Не-Pydantic модели (P6)** | `BBox`, `Style`, `Region` — обычные классы Python. Это исключение оправдано тем, что они являются внутренними структурами парсера, а не контрактами между сервисами. При рефакторинге (перенос в `documet_index`) может быть рассмотрена миграция на Pydantic. |
| **Нестандартная семантика `Style.__lt__`** | Оператор `<` переопределён с неочевидной семантикой («может быть родителем»), что нарушает принцип наименьшего удивления. Вместо `a < b` следовало бы использовать явный метод `a.can_be_parent_of(b)`. Документируется «как есть». |
| **`element_data` для equation содержит тип, а не данные** | В `create_graph_from_mineru_result` для equation `element_data=element_type` (строка `"equation"`), а не сам текст формулы. Вероятно, баг — должен быть `element_data=text`. |
