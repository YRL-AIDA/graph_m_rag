# Описание проверок

#### API Contracts (автоматически)
1. Генерируется OpenAPI spec из FastAPI (`/openapi.json`)
2. Для каждого эндпоинта в коде проверяется наличие Feature Spec с OpenAPI-фрагментом
3. method, path и status codes в OpenAPI-фрагменте сверяются со сгенерированным spec
4. Расхождение = CI failure

#### Структурная проверка spec-файлов (автоматически)
1. Каждая Pydantic-модель из `dtype/` должна иметь Data Model Spec в `specs/models/`
2. Каждый spec-файл должен содержать все обязательные секции шаблона
3. Отсутствие обязательной секции = CI warning (на переходный период) → CI failure (после полного покрытия)

#### Pipeline/Behaviour проверка (ручной review)
- Pipeline Specs и Data Model Specs проверяются вручную при code review
- Reviewer обязан сверить описанное поведение с реализацией