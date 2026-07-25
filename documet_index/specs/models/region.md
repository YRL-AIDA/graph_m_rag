# Data Model: Region, BBox, Style

## Purpose

Модели `Region`, `BBox` и `Style` — plain Python классы, представляющие элементы структурированного контента PDF-документа, их координаты и стилевые атрибуты. Используются как узлы графа структуры документа (Neo4j document graph) и как промежуточное представление при парсинге MinerU JSON.

**`CONTENT_LABELS`** — модульная константа, определяющая какие типы регионов считаются контентными (а не структурными).

## Schema

### BBox

```python
class BBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float)
```

Bounding box региона — координаты левого верхнего (`x1`, `y1`) и правого нижнего (`x2`, `y2`) углов.

| Поле | Тип | Описание |
|------|-----|----------|
| `x1` | `float` / `int` | Левая граница |
| `y1` | `float` / `int` | Верхняя граница |
| `x2` | `float` / `int` | Правая граница |
| `y2` | `float` / `int` | Нижняя граница |

**Свойства** (только для чтения):

| Свойство | Тип | Формула |
|----------|-----|---------|
| `width` | `float` | `x2 - x1` |
| `height` | `float` | `y2 - y1` |

**Замечание о типах**: поля не типизированы на уровне конструктора — никакой валидации `float`/`int` не производится. В реальных данных значения приходят как `int` или `float` из JSON MinerU.

### Style

```python
class Style:
    def __init__(self, font_size: float)
```

Стилевые атрибуты региона. На данный момент содержит только размер шрифта.

| Поле | Тип | Значение по умолчанию | Описание |
|------|-----|----------------------|----------|
| `font_size` | `float` / `int` | — | Размер шрифта |
| `error_rate` | `float` | `1.0` | Порог погрешности при сравнении стилей |

**Операторы**:

| Оператор | Сигнатура | Семантика |
|----------|-----------|-----------|
| `__lt__` | `(self, other: Style) -> bool` | **Инвертированное сравнение**: возвращает `True`, если `self.font_size > other.font_size + self.error_rate`. То есть `style_a < style_b` означает «шрифт `a` крупнее шрифта `b` с учётом погрешности» |

**Семантика `__lt__`**: оператор `<` используется для определения иерархии заголовков. Родитель «меньше» ребёнка, если у родителя **крупнее** шрифт. Это означает, что родитель включает ребёнка в свой контекст. `error_rate = 1` добавляет порог в 1 пункт, чтобы незначительные колебания размера шрифта не создавали ложную иерархию.

Примеры:
- `Style(24) < Style(18)` → `24 > 18 + 1` → `True` (родитель с font_size=24 вмещает ребёнка с font_size=18)
- `Style(18) < Style(24)` → `18 > 24 + 1` → `False` (родитель с font_size=18 НЕ вмещает ребёнка с font_size=24)
- `Style(12) < Style(11)` → `12 > 11 + 1` → `False` (разница слишком мала — считается одинаковым уровнем)
- `Style(-1) < Style(-1)` → `-1 > -1 + 1` → `False` (одинаковый стиль)

### Region

```python
class Region:
    def __init__(self, text: str, image: str, bbox: BBox, style: Style,
                 order: int, label: str, element_data: str)
```

Элемент контента PDF-документа — узел в графе структуры.

| Поле | Тип | Описание |
|------|-----|----------|
| `text` | `str` | Текстовое содержимое региона. Для структурных типов — с префиксом (`"Title: ..."`, `"Text: ..."`, `"Equation: ..."`, `"Image Caption: ..."`, `"Table Caption: ..."` и т.д.). Для `image` — пустая строка |
| `image` | `str` | Путь к файлу изображения. Для image/table/caption/footnote — соответствующий `img_path`. Для текстовых — `""` |
| `bbox` | `BBox` | Координаты bounding box на странице PDF |
| `style` | `Style` | Стилевые атрибуты (font_size, error_rate) |
| `order` | `int` | Порядковый номер в линейной последовательности `content_list` (сквозной счётчик `element_index`) |
| `label` | `str` | Тип региона: `"title"`, `"text"`, `"image"`, `"image_caption"`, `"image_footnote"`, `"table"`, `"table_caption"`, `"table_footnote"`, `"equation"`, `"header"`, или произвольная строка для неизвестных типов |
| `element_data` | `str` | Исходные данные элемента без префикса. Для `title`/`text` — чистый текст; для `image` — `img_path`; для `caption`/`footnote` — текст подписи; для `equation` — `"equation"` (строка-тип) |

**Методы**:

| Метод | Сигнатура | Описание |
|-------|-----------|----------|
| `is_content` | `() -> bool` | `True`, если `self.label in CONTENT_LABELS` (т.е. `label` входит в `['text', 'table', 'image']`) |
| `to_dict` | `() -> Dict[str, Any]` | Сериализация в словарь для последующей записи в Neo4j |

### CONTENT_LABELS

```python
CONTENT_LABELS = ['text', 'table', 'image']
```

Модульная константа — список лейблов, соответствующих «контентным» (листовым) элементам документа. Используется в `Region.is_content()`. Элементы с этими лейблами в иерархии PARENT-связей привязываются к текущему заголовку, но сами не становятся заголовками (не добавляются в стек).

**Историческая заметка**: закомментированная предыдущая версия (`# CONTENT_LABELS = ['text', 'table', 'list', 'figure', 'interline_equation']`) в строке 1 отражает более широкий набор из более ранней версии парсера (pdf_info), не используется.

## Storage

Neo4j (Document Graph), свойства узла `Region:{label}`:

| Свойство Neo4j | Источник из Region | Формат |
|----------------|-------------------|--------|
| `text` | `self.text` | `str` |
| `image` | `self.image` | `str` |
| `bbox` | `self.bbox` | `str` — JSON: `{"x1":..., "y1":..., "x2":..., "y2":...}` |
| `style` | `self.style` | `str` — JSON: `{"font_size":..., "error_rate":...}` |
| `order` | `self.order` | `int` |
| `element_data` | `self.element_data` | `str` |

При сериализации через `to_dict()`:
- `bbox` сериализуется только если `self.bbox` не `None`/`False`; иначе `{}`.
- `style` сериализуется только если `self.style` не `None`/`False`; иначе `{}`.
- Оба преобразуются в JSON-строку в `Manager.add_document()` через `json.dumps()`.

## Relationships

Модели `Region`, `BBox`, `Style` не определяют связи напрямую — это plain Python классы данных. Связи (`ORDER`, `PARENT`) строятся внешним кодом:

- `Document.get_graph()` строит `order_edges` и `parent_edges` как списки кортежей `(parent_index, child_index)`, где `-1` обозначает узел `Document`.
- `Manager.add_document()` транслирует эти кортежи в Cypher `CREATE` с соответствующими отношениями.

**Участие `Style.__lt__` в построении PARENT**: оператор `<` используется в алгоритме `Document.get_graph()` → `is_include_by_id()` для определения, может ли родительский регион вмещать дочерний.

## Constraints

| Ограничение | Описание |
|-------------|----------|
| **Типы полей не валидируются** | Конструкторы `BBox`, `Style`, `Region` не проверяют типы аргументов — ответственность вызывающего кода |
| **error_rate жёстко задан** | `Style.error_rate = 1` — не настраивается извне |
| **CONTENT_LABELS — модульная константа** | Изменение списка требует пересмотра логики `is_content()` и поведения PARENT-алгоритма |
| **label не ограничен перечислением** | `Region.label` — свободная строка, не `Literal`/`Enum`. Допустимы любые значения |
| **bbox может быть «пустым»** | При ошибке парсинга координат создаётся `BBox(0, 0, 0, 0)` |

## Examples

### BBox

```python
bbox = BBox(72, 100, 200, 150)
assert bbox.width == 128     # 200 - 72
assert bbox.height == 50     # 150 - 100
```

### Style

```python
# Заголовок главы (крупный шрифт)
h1_style = Style(24)
# Подзаголовок (средний шрифт)
h2_style = Style(18)
# Основной текст (мелкий шрифт)
text_style = Style(10)

assert h1_style < h2_style    # True   (24 > 18 + 1: h1 вмещает h2)
assert h2_style < text_style  # True   (18 > 10 + 1: h2 вмещает text)
assert h2_style < h1_style    # False  (18 > 24 + 1: False — h2 не вмещает h1)
assert text_style < h1_style  # False  (10 > 24 + 1: False)

# Оба с font_size=-1 (используется по умолчанию в парсере)
s1 = Style(-1)
s2 = Style(-1)
assert not (s1 < s2)          # -1 > -1 + 1 → False (одинаковый уровень)
```

### Region

```python
region = Region(
    text="Title: Chapter 1",
    image="",
    bbox=BBox(72, 100, 200, 120),
    style=Style(24),
    order=0,
    label="title",
    element_data="Chapter 1"
)

assert region.is_content() == False  # "title" not in CONTENT_LABELS
assert region.to_dict() == {
    "label": "title",
    "text": "Title: Chapter 1",
    "image": "",
    "bbox": {"x1": 72, "y1": 100, "x2": 200, "y2": 120},
    "style": {"font_size": 24, "error_rate": 1},
    "order": 0,
    "element_data": "Chapter 1"
}
```

### Контентный vs структурный регион

```python
text_region = Region(text="Text: Hello", image="", bbox=BBox(0,0,100,20),
                     style=Style(10), order=1, label="text", element_data="Hello")
assert text_region.is_content() == True  # "text" in CONTENT_LABELS

image_caption = Region(text="Image Caption: Fig 1", image="/tmp/img.png",
                       bbox=BBox(0,0,100,20), style=Style(-1), order=2,
                       label="image_caption", element_data="Fig 1")
assert image_caption.is_content() == False  # "image_caption" not in CONTENT_LABELS
```

## Error Handling

| Ситуация | Поведение |
|----------|-----------|
| `BBox(x1, y1, x2, y2)` с нечисловыми аргументами | Ошибки времени выполнения при вычислении `width`/`height` (арифметические операции над нечисловыми типами) |
| `Style(font_size=<нечисло>)` | Ошибка при сравнении `<` (арифметика с `error_rate`) |
| `Region` с несовместимыми типами полей | Не проверяется — silent corruption возможно |
| `Region.to_dict()` при `bbox=None` | `{}` (пустой словарь) |
| `Region.to_dict()` при `style=None` | `{}` (пустой словарь) |

## Testing

| Тест-кейс | Проверка |
|-----------|----------|
| `BBox(10, 20, 30, 50)` | `width == 20`, `height == 30` |
| `BBox(0, 0, 0, 0)` | `width == 0`, `height == 0` |
| `Style(24) < Style(18)` | `True` |
| `Style(18) < Style(24)` | `False` |
| `Style(12) < Style(11)` | `False` (разница <= error_rate) |
| `Style(-1) < Style(-1)` | `False` |
| `Region(label="text").is_content()` | `True` |
| `Region(label="title").is_content()` | `False` |
| `Region(label="image_caption").is_content()` | `False` |
| `Region(label="unknown").is_content()` | `False` |
| `Region.to_dict()` с валидными bbox и style | Словарь корректной структуры |
| `Region.to_dict()` с `bbox=None` | `"bbox": {}` |
| `Region.to_dict()` с `style=None` | `"style": {}` |
| Все лейблы из `CONTENT_LABELS` покрыты тестом `is_content() == True` | `['text', 'table', 'image']` |

## Dependencies

| Модуль | Назначение |
|--------|------------|
| `builtins` | Стандартные типы Python (`str`, `int`, `float`, `list`, `bool`) |
| `typing` | — (не используется, plain Python) |

## Exceptions

1. **Нарушение P6 (Pydantic)**: `BBox`, `Style`, `Region` — plain Python классы с `__init__`, а не Pydantic `BaseModel`. Поля не типизированы в рантайме, валидация отсутствует. Класс `Region` не использует даже `@dataclass` — полностью ручной `__init__`. **Обоснование**: переходный период — код написан до принятия Constitution. Должно быть исправлено:
   - `BBox` → Pydantic `BaseModel` с валидацией `x1, y1, x2, y2` как `float`/`int`
   - `Style` → Pydantic `BaseModel` с `font_size: float`, `error_rate: float = 1.0`
   - `Region` → Pydantic `BaseModel` с валидаторами полей, `is_content()` как `@property` или `@computed_field`
   - `CONTENT_LABELS` → вынести в `config/settings.py` или Enum

2. **Инвертированная семантика `__lt__`**: оператор `<` определён нестандартным образом (возвращает `True`, когда `self.font_size` **больше** `other.font_size`). Это противоречит математическому контракту `<` (меньше значит меньше). При рефакторинге на Pydantic пересмотреть: либо сделать отдельный метод `includes(other)`, либо переопределить оператор `>` с ожидаемой семантикой.

3. **Отсутствие `__eq__`**: классы `BBox`, `Style`, `Region` не определяют `__eq__`/`__hash__`, что делает их поведение при сравнении на равенство зависимым от `id()` (identity). При рефакторинге добавить `__eq__` на основе значений полей.

4. **Смешанные типы для `element_data`**: поле объявлено как `str`, но в коде `create_graph_from_mineru_result` для `equation` присваивается `element_data=element_type` (строка `"equation"`), а в `Manager.add_document()` применяется `str(element_data)` для надёжности. **Рекомендация**: унифицировать — либо всегда содержательный текст, либо валидировать тип на входе в `Region.__init__`.
