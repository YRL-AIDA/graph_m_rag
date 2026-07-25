# Data Model: Entity & Relationship

## Purpose

Модели для извлечения именованных сущностей и связей из текстовых чанков через LLM, передачи их в API `POST /process-document` и сохранения в Neo4j (семантический граф знаний). Все модели определены как Pydantic-наследники и используются как контракты между модулем извлечения (`graphrag.py`), API-эндпоинтом (`api.py`) и менеджером Neo4j (`semantic_index.py`).

## Schema

### EntityCreate

```python
from pydantic import BaseModel
from typing import List, Optional

class EntityCreate(BaseModel):
    """Сущность, извлечённая LLM из текстового чанка."""
    title: str                              # Имя сущности (нормализуется в UPPER CASE)
    type: str                               # Тип сущности (ORGANIZATION, PERSON, GEO, EVENT, ...)
    text_unit_ids: Optional[List[str]] = [] # Список ID чанков (chunk_id), из которых извлечена сущность
    frequency: Optional[int] = 0            # Частота упоминания в чанках
    description: Optional[str] = None       # Описание сущности, суммаризованное из всех упоминаний
    degree: Optional[int] = 0               # Степень узла в графе (число связей)
```

### RelationshipCreate

```python
class RelationshipCreate(BaseModel):
    """Связь между двумя сущностями, извлечённая LLM."""
    source: str                       # Имя исходной сущности (title в UPPER CASE)
    target: str                       # Имя целевой сущности (title в UPPER CASE)
    text_unit_ids: Optional[List[str]] = []  # ID чанков, из которых извлечена связь
    weight: Optional[float] = 1.0     # Сила связи (0-10, из ответа LLM)
    description: Optional[str] = None # Описание характера связи
    combined_degree: Optional[int] = 0 # Суммарная степень source + target узлов
```

### EntitiesRequest

```python
class EntitiesRequest(BaseModel):
    """Тело запроса на сохранение извлечённых сущностей и связей."""
    entities: List[EntityCreate]           # Список извлечённых сущностей
    relationships: List[RelationshipCreate] # Список извлечённых связей
```

### EntitiesResponse

```python
class EntitiesResponse(BaseModel):
    """Ответ после сохранения в Neo4j."""
    nodes_created: int        # Количество новых созданных Entity-узлов
    nodes_updated: int        # Количество обновлённых Entity-узлов
    relationships_added: int  # Количество добавленных RELATED-связей
```

### DocumentRequest

```python
class DocumentRequest(BaseModel):
    """Тело запроса POST /process-document."""
    document_id: str  # file_hash документа (P2 — единый идентификатор)
```

### ErrorResponse

В текущем коде `dtype/entity.py` класс `ErrorResponse` **отсутствует**. Ошибки возвращаются как стандартные HTTPException с JSON-телом `{"detail": "..."}`. При необходимости может быть добавлен:

```python
class ErrorResponse(BaseModel):
    error: str       # Краткий код ошибки
    detail: str      # Человекочитаемое описание
    status_code: int # HTTP-код
```

## Storage

- **Neo4j**: узлы `Entity` с меткой `:Entity`, связи `RELATED` между `Entity`-узлами.
- **Уникальный constraint**: `(title, type)` — пара имя + тип образует составной ключ сущности.
- **Идентификатор сущности в Neo4j**: строка `"TITLE|TYPE"` (например, `"ACME CORP|ORGANIZATION"`).
- **in-memory**: Pydantic-модели `EntityCreate`, `RelationshipCreate` обрабатываются в рантайме, преобразуются в pandas DataFrame перед записью в Neo4j (внутренний формат пайплайна `graphrag.py`).

## Relationships

| Модель | Связана с | Характер связи |
|--------|-----------|---------------|
| `EntityCreate` | `RelationshipCreate` | `source` и `target` в `RelationshipCreate` ссылаются на `title` из `EntityCreate` |
| `EntitiesRequest` | `EntityCreate`, `RelationshipCreate` | Аггрегирует списки для пакетной записи |
| `EntitiesResponse` | Neo4j `Entity` / `RELATED` | Отражает результат записи в граф |
| `DocumentRequest` | `EntityCreate.text_unit_ids` | `document_id` используется для фильтрации чанков в Qdrant; извлечённые `text_unit_ids` связывают сущности с документом |

## Constraints

| Модель | Поле | Ограничение |
|--------|------|-------------|
| `EntityCreate` | `title` | Обязательное. При парсинге ответа LLM приводится к UPPER CASE. Уникально в комбинации с `type` (Neo4j constraint `entity_title_type_unique`) |
| `EntityCreate` | `type` | Обязательное. Допустимые типы определяются константой `ENTITY_TYPES` в `config.py` |
| `EntityCreate` | `text_unit_ids` | Список строк-идентификаторов чанков. При повторной загрузке тех же `text_unit_ids` дубликаты сущностей пропускаются |
| `RelationshipCreate` | `source` | Обязательное. Имя сущности, обязано существовать в списке `entities` того же запроса |
| `RelationshipCreate` | `target` | Обязательное. Имя сущности, обязано существовать в списке `entities` того же запроса |
| `RelationshipCreate` | `weight` | float, по умолчанию 1.0. Из ответа LLM парсится как числовая оценка силы связи |
| `EntitiesRequest` | `entities` | Может быть пустым списком (документ без сущностей → статус `completed_without_entities`) |
| `DocumentRequest` | `document_id` | Обязательное. Соответствует `file_hash` в Qdrant и MinIO (P2) |

## Examples

### Валидный запрос EntitiesRequest

```json
{
  "entities": [
    {
      "title": "ACME CORP",
      "type": "ORGANIZATION",
      "text_unit_ids": ["doc123_chunk0", "doc123_chunk5"],
      "frequency": 4,
      "description": "ACME Corp is a multinational technology company specializing in AI solutions.",
      "degree": 3
    },
    {
      "title": "JOHN SMITH",
      "type": "PERSON",
      "text_unit_ids": ["doc123_chunk0"],
      "frequency": 1,
      "description": "John Smith is the CEO of ACME Corp.",
      "degree": 1
    }
  ],
  "relationships": [
    {
      "source": "JOHN SMITH",
      "target": "ACME CORP",
      "text_unit_ids": ["doc123_chunk0"],
      "weight": 9.0,
      "description": "John Smith is the CEO of ACME Corp",
      "combined_degree": 4
    }
  ]
}
```

### Валидный ответ EntitiesResponse

```json
{
  "nodes_created": 2,
  "nodes_updated": 0,
  "relationships_added": 1
}
```

### Валидный запрос DocumentRequest

```json
{
  "document_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
}
```

### Документ без сущностей (EntitiesRequest с пустыми списками)

```json
{
  "entities": [],
  "relationships": []
}
```

Статус ответа: `200 OK` со статусом `completed_without_entities` (не ошибка).

## Exceptions

| Отклонение | Обоснование |
|------------|-------------|
| **Отсутствие `ErrorResponse` в коде** | Класс не определён в `dtype/entity.py`, хотя упоминается в задании. Ошибки возвращаются через FastAPI `HTTPException`. При необходимости модель должна быть добавлена для единообразия (P6). |
| **Несоответствие имён полей заданию** | В задании указаны поля `file_hash`, `entity_name`, `entity_type`, `source_chunk`, `region_id`, но в коде используются `title`, `type`, `text_unit_ids`, `description`. Спецификация документирует **фактический код**. |
| **DataFrame как промежуточный формат** | Внутри пайплайна `graphrag.py` Pydantic-модели преобразуются в pandas DataFrame. Это допустимое исключение из P6 для внутренних вычислений (Constitution §5.2). |
