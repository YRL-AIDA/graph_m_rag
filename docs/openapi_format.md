# Формат OpenAPI-фрагментов в Feature Spec

Для каждого API-эндпоинта Feature Spec обязан содержать секцию `## API Contract` с OpenAPI-фрагментом в формате YAML внутри блока `openapi`. Фрагмент описывает:

- HTTP method и path
- `summary` — краткое описание
- `requestBody` — схема запроса (ссылка на Pydantic-модель из `dtype/`)
- `responses` — минимум `200`, `400`, `500`
