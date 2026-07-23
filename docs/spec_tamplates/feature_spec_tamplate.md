# Feature: {Название фичи}

## Motivation
- Какую проблему решает, почему это нужно

## Behaviour
### Input
- Что принимает на вход, формат, валидация

### Processing
- Пошаговое описание логики обработки

### Output
- Что возвращает, формат, коды ошибок

## API Contract
<!-- Только для эндпоинтов. Валидируется CI против сгенерированного OpenAPI. -->

```openapi
{method}: {path}
summary: ...
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/{RequestModel}'
responses:
  '200':
    description: ...
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/{ResponseModel}'
  '400':
    description: Invalid request
  '500':
    description: Internal processing error
```

## Data Flow
- Какие хранилища затрагивает, в каком порядке
- Диаграмма (mermaid или описание текстом)

## LLM Interactions
- Ссылки на файлы промптов: `prompts/{prompt-name}.md`
- Требования к промпту (вход/выход/ограничения)

## LLM Model Requirements
- **Тип модели**: text-only / multimodal (text+image) / reasoning
- **Минимальный размер контекста**: {N} токенов
- **Язык выхода**: русский / английский / multilingual
- **Требования к формату выхода**: JSON / свободный текст / структурированный список
- Конкретная модель задаётся в конфигурации сервиса (Pydantic Settings)

## Error Handling
- Какие ошибки возможны, как обрабатываются

## Testing
- Ключевые тест-кейсы (сценарии, граничные случаи)

## Dependencies
- Какие сервисы/модули использует

## Exceptions
- Обоснованные отклонения от Constitution (если есть)