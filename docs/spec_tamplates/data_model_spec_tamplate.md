# Data Model: {Имя модели}

## Purpose
- Для чего используется эта модель

## Schema
```python
class {ModelName}(BaseModel):
    field: type  # описание
```

## Storage
- Где хранится: Qdrant payload / Neo4j node / MinIO object / in-memory

## Relationships
- Связи с другими моделями

## Constraints
- Уникальность, обязательность полей, валидация

## Examples
- Примеры валидных и невалидных данных