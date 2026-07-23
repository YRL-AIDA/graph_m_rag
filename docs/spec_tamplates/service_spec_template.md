# Service: {Имя сервиса}

## Identity
- Зона ответственности: ...
- Порт: {port}
- Хранилище: {database / file system / none}

## Dependencies
| Сервис | Протокол | Назначение |
|--------|----------|------------|
| ...    | REST/gRPC/driver | ... |

## API
| Method | Path | Feature Spec | Purpose |
|--------|------|-------------|---------|
| POST   | /upload-pdf | [document-ingestion.md](document-ingestion.md) | Загрузка PDF |

## Data Model
- Ссылки на Data Model Specs: ...

## Pipelines
- Ссылки на Pipeline Specs: ...

## Configuration
- Переменные окружения: ...

## Invariants
- Список инвариантов, которые этот сервис гарантирует

## Exceptions
- Обоснованные отклонения от Constitution (если есть)