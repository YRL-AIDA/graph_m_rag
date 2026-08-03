# Feature: GleanerClient — NER-only клиент к Gleaner

## Motivation
Клиент к Gleaner (`urchade/gliner_large-v2.1`) через кастомный FastAPI с OpenAI-совместимым эндпоинтом. Gleaner — bi-encoder (DeBERTa-based) span detector, **не** generative модель, поэтому не запускается через VLLM. Поддерживает только NER; RE не реализовано по дизайну.

## Behaviour

### Constructor
```python
class GleanerClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,   # env: GLINER_BASE_URL, default "http://<remote_host>:<port>/v1"
        api_key: str,     # env: GLINER_API_KEY, default "EMPTY"
        model: str,       # env: GLINER_MODEL, default "urchade/gliner_large-v2.1"
    )
```
Транспорт: `openai.AsyncOpenAI` (Gleaner FastAPI сервер предоставляет OpenAI-совместимый эндпоинт `POST /v1/chat/completions`).

### extract_entities(text, entity_types) → list[PredictedEntity]
1. Формирует запрос: типы сущностей в system prompt, текст в user message.
2. System message: `Entity types: {", ".join(entity_types)}`. User message: `{text}`.
3. Вызывает `await client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], model=self.model)`.
4. Парсит ответ: ожидает JSON-список `[{"name": str, "type": str}]` в `message.content`.
5. Возвращает `list[PredictedEntity]`.

### extract_relations
НЕ реализован. Вызов → `NotImplementedError("GleanerClient does not support RE")`. Gleaner — span detector, не generative модель.

### Error Handling
| Сценарий | Поведение |
|----------|-----------|
| Connection error / timeout | `ConnectionError`, логируется |
| Response не парсится (невалидный JSON) | `[]`, warning в лог |
| Пустой ответ (API вернул `[]`) | `[]` |

## Testing
1. `extract_entities` с мокнутым `AsyncOpenAI`: валидный Gleaner JSON-ответ → корректный `list[PredictedEntity]`.
2. Невалидный JSON в ответе → возвращает `[]`.
3. Вызов `extract_relations` → `NotImplementedError` с ожидаемым сообщением.

Все LLM/API-вызовы мокаются (Constitution T3). Фикстура ответа: `tests/fixtures/model_responses/gliner_response.json`.

## Dependencies
- `openai>=1.0` (`AsyncOpenAI`)
- `PredictedEntity`, `PredictedRelation`, `BaseModelClient` — определены в `models/base_client.py` (см. `general_experiment.md` §2 Unified Model Interface)
- Удалённый Gleaner FastAPI сервер: `deploy/gliner_server.py`

## Exceptions
- **§5.2 Experiments/Spikes**: код в изолированной директории `tests/`, спецификация написана до реализации.
