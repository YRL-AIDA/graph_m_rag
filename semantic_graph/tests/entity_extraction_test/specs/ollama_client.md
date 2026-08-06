# Feature: OllamaClient — клиент моделей через нативный Ollama API

## Motivation
Универсальный клиент для подключения к Ollama-серверу через нативный REST API (`POST /api/chat`). Позволяет использовать любые модели, загруженные в Ollama (qwen3-coder, llama3, mistral и т.д.), для экспериментов NER/RE. Клиент конфигурируется через параметры конструктора и переменные окружения: URL сервера, имя модели, режим рассуждений (`think`), максимальное число токенов генерации, размер контекста, пути к файлам промптов. Поддерживает как раздельные вызовы NER/RE, так и combined-режим (NER+RE одним вызовом).

## Behaviour

### Constructor
```python
class OllamaClient(BaseModelClient):
    def __init__(
        self,
        base_url: str | None = None,        # env: OLLAMA_BASE_URL, default "http://192.168.55.242:7869"
        model: str | None = None,            # env: OLLAMA_MODEL, default "qwen3-coder:30b"
        think: bool | str | None = None,     # env: OLLAMA_THINK, default False
        num_predict: int | None = None,      # env: OLLAMA_NUM_PREDICT, default 256
        num_ctx: int | None = None,          # env: OLLAMA_NUM_CTX, default 4096
        ner_prompt_path: str = "",           # путь к файлу NER-промпта
        re_prompt_path: str = "",            # путь к файлу RE-промпта
        combined_prompt_path: str = "",      # путь к файлу combined-промпта
        keep_alive: str | None = None,       # env: OLLAMA_KEEP_ALIVE, default "5m"
    )
```

Все параметры конструктора переопределяются через переменные окружения, если не переданы явно.

Внутренний транспорт: `httpx.AsyncClient(base_url=base_url, timeout=...)`.

Промпты загружаются из файлов при инициализации через `Path(...).read_text()` (Constitution P7):
- `self._ner_prompt = Path(ner_prompt_path).read_text()` если путь передан, иначе `""`
- `self._re_prompt = Path(re_prompt_path).read_text()` если путь передан, иначе `""`
- `self._combined_prompt = Path(combined_prompt_path).read_text()` если путь передан, иначе `""`

### extract_entities(text: str, entity_types: list[str]) → list[PredictedEntity]
1. Проверить, что `_ner_prompt` не пуст. Если пуст — warning, вернуть `[]`.
2. Форматировать NER-промпт: `self._ner_prompt.format(input_text=text, entity_types=",".join(entity_types))`
3. Вызвать `self._chat(messages=[{"role": "user", "content": prompt}])`
4. Распарсить ответ через `_parse_ner_response(response, entity_types)`:
   - Извлечь строки формата `("entity"<|>NAME<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
   - Для каждой строки: `NAME` → `PredictedEntity.name`, `TYPE` → валидация типа через `normalize_type()`
   - Если `normalize_type` вернул `None` — пропустить сущность, warning в лог
5. Вернуть `list[PredictedEntity]`

### extract_relations(text, entities, relation_types) → list[PredictedRelation]
1. Проверить, что `_re_prompt` не пуст. Если пуст — warning, вернуть `[]`.
2. Построить `entities_list` из переданных сущностей: строки `NAME|TYPE`, разделённые `\n`
3. Форматировать RE-промпт: `self._re_prompt.format(input_text=text, entities_list=entities_list, relation_types=",".join(relation_types))`
4. Вызвать `self._chat(messages=[{"role": "user", "content": prompt}])`
5. Распарсить ответ через `_parse_re_response(response, relation_types)`:
   - Извлечь строки `("relationship"<|>SRC<|>TGT<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
   - Для каждой строки: `SRC` → `PredictedRelation.head`, `TGT` → `PredictedRelation.tail`, `TYPE` → валидация типа через `normalize_type()`
6. Вернуть `list[PredictedRelation]`

### extract_entities_and_relations(text, entity_types, relation_types) → tuple[list[PredictedEntity], list[PredictedRelation]]
1. Проверить, что `_combined_prompt` не пуст. Если пуст — warning, вернуть `([], [])`.
2. Форматировать combined-промпт: `self._combined_prompt.format(input_text=text, entity_types=",".join(entity_types), relation_types=",".join(relation_types))`
3. Вызвать `self._chat(messages=[{"role": "user", "content": prompt}])` — **один вызов**
4. Из одного ответа распарсить и сущности, и отношения:
   - Строки с префиксом `("entity"<|>...)` → `PredictedEntity`
   - Строки с префиксом `("relationship"<|>...)` → `PredictedRelation`
   - Терминатор `<|COMPLETE|>` общий
5. Применить валидацию типов: `normalize_type()` для сущностей и отношений
6. Вернуть кортеж `(entities, relations)`

### Метод _chat(messages: list[dict]) → str | None
Базовый метод для отправки запроса к Ollama `/api/chat`:

```python
async def _chat(self, messages: list[dict]) -> str | None:
    """Отправить запрос к Ollama /api/chat и вернуть content ответа."""
    payload = {
        "model": self.model,
        "messages": messages,
        "stream": False,
        "keep_alive": self.keep_alive,
    }
    # think — только если не False
    if self.think is not False:
        payload["think"] = self.think
    # options: num_predict, num_ctx
    options = {}
    if self.num_predict is not None:
        options["num_predict"] = self.num_predict
    if self.num_ctx is not None:
        options["num_ctx"] = self.num_ctx
    if options:
        payload["options"] = options

    response = await self._client.post("/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content")
```

Формат тела запроса:
```json
{
    "model": "qwen3-coder:30b",
    "messages": [{"role": "user", "content": "..."}],
    "stream": false,
    "think": false,
    "keep_alive": "5m",
    "options": {
        "num_predict": 256,
        "num_ctx": 4096
    }
}
```

Формат ответа:
```json
{
    "model": "qwen3-coder:30b",
    "created_at": "2025-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "..."},
    "done": true,
    "done_reason": "stop"
}
```

### Параметр `think`
Управляет режимом рассуждений (reasoning) для моделей, которые его поддерживают:
- `False` (по умолчанию) — рассуждения отключены
- `True` — включить рассуждения (дефолтный уровень модели)
- `"low"`, `"medium"`, `"high"`, `"max"` — конкретный уровень

### Type Validation
Все методы парсинга применяют `normalize_type()` из `metrics.metrics` (так же, как QwenClient):
- Для каждого `PredictedEntity`: `normalize_type(entity.type, entity_types)`, при `None` — пропустить
- Для каждого `PredictedRelation`: `normalize_type(relation.type, relation_types)`, при `None` — пропустить

### Error handling
| Сценарий | Поведение |
|----------|-----------|
| Connection error / timeout | `ConnectionError`, логируется, запись считается failed |
| HTTP ошибка (4xx, 5xx) | `ConnectionError`, логируется |
| Ответ без ключа `message.content` | `None` → пустой `[]`, warning в лог |
| Ответ непарсибельный (нет `("entity"` / `("relationship"`) | Пустой `[]` (или `([], [])`), warning в лог |
| Частично парсибельный ответ (часть строк валидна) | Извлечь валидные строки, пропустить невалидные, warning |
| Тип не прошёл `normalize_type` (вернулся `None`) | Пропустить запись, warning в лог |
| `_ner_prompt` пуст при вызове `extract_entities` | `[]`, warning в лог |
| `_re_prompt` пуст при вызове `extract_relations` | `[]`, warning в лог |
| `_combined_prompt` пуст при вызове `extract_entities_and_relations` | `([], [])`, warning в лог |

## Data Flow
```
OllamaClient
  ├── self._client: httpx.AsyncClient ── HTTP POST /api/chat ──► Ollama Server (default: 192.168.55.242:7869)
  ├── ner_prompt: str        (из prompts/ner_prompt.md)
  ├── re_prompt: str         (из prompts/re_prompt.md)
  └── combined_prompt: str   (из prompts/combined_prompt.md)

extract_entities:
  text + entity_types → format(ner_prompt) → _chat(messages) → _parse_ner_response(response, entity_types) → normalize_type → list[PredictedEntity]

extract_relations:
  text + entities + relation_types → format(re_prompt) → _chat(messages) → _parse_re_response(response, relation_types) → normalize_type → list[PredictedRelation]

extract_entities_and_relations:
  text + entity_types + relation_types → format(combined_prompt) → _chat(messages) → _parse_combined_response → normalize_type → (list[PredictedEntity], list[PredictedRelation])
```

## LLM Interactions

| Промпт | Файл | Переменные |
|--------|------|------------|
| NER Prompt | `prompts/ner_prompt.md` | `{input_text}`, `{entity_types}` |
| RE Prompt | `prompts/re_prompt.md` | `{input_text}`, `{entities_list}`, `{relation_types}` |
| Combined Prompt | `prompts/combined_prompt.md` | `{input_text}`, `{entity_types}`, `{relation_types}` |

Все промпты загружаются из файлов (Constitution P7). Клиент переиспользует те же промпт-файлы, что и QwenClient. Требования к формату выхода: кортежи с разделителем `<|>`, строки разделены `##`, терминатор `<|COMPLETE|>`.

Формат ответа модели, ожидаемый от Ollama — идентичен формату QwenClient (строки с `("entity"<|>...)` и `("relationship"<|>...)`).

## Ollama API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/tags` | Список доступных моделей (для диагностики) |
| POST | `/api/chat` | Генерация ответа чата (основной) |

Клиент использует только `POST /api/chat`.

## Environment Variables

| Переменная | Значение по умолчанию | Описание |
|-----------|----------------------|----------|
| `OLLAMA_BASE_URL` | `http://192.168.55.242:7869` | URL Ollama-сервера |
| `OLLAMA_MODEL` | `qwen3-coder:30b` | Имя модели в Ollama |
| `OLLAMA_THINK` | `False` | Режим рассуждений: `False`, `True`, `"low"`, `"medium"`, `"high"`, `"max"` |
| `OLLAMA_NUM_PREDICT` | `256` | Максимальное число токенов генерации |
| `OLLAMA_NUM_CTX` | `4096` | Размер контекстного окна |
| `OLLAMA_KEEP_ALIVE` | `"5m"` | Время удержания модели в памяти после запроса |

## Testing
Все тесты с замоканным `httpx.AsyncClient` (Constitution T3):
1. `extract_entities` с валидным ответом → корректный `list[PredictedEntity]`
2. `extract_relations` с валидным ответом → корректный `list[PredictedRelation]`
3. `extract_entities_and_relations` с combined-ответом → корректный кортеж
4. _chat вернул `None` (пустой content) → `[]`; combined → `([], [])`
5. Непарсибельный ответ → `[]`, warning
6. Частично парсибельный ответ → извлечены только валидные строки
7. HTTP-ошибка (500) → `ConnectionError`
8. Валидация типов: сущность с неизвестным типом отбрасывается
9. Пустой NER/RE/combined prompt → `[]`, warning (без вызова API)

Фикстуры: `tests/fixtures/model_responses/ollama_ner_response.txt`, `tests/fixtures/model_responses/ollama_re_response.txt`, `tests/fixtures/model_responses/ollama_combined_response.txt`.

Мокирование: `httpx.AsyncClient.post` возвращает `httpx.Response(200, json={"message": {"content": "..."}})` с телом ответа из фикстуры.

## Dependencies
- `httpx>=0.27` (`AsyncClient`)
- `pytest-httpx>=0.30` (для тестирования)
- Типы `BaseModelClient`, `PredictedEntity`, `PredictedRelation` — из `testdata/base_loader.py`
- `metrics.metrics` → `normalize_type`
- Промпты: `prompts/ner_prompt.md`, `prompts/re_prompt.md`, `prompts/combined_prompt.md`

## Exceptions
- **Constitution §5.2 Experiments/Spikes**: клиент находится в изолированной директории `tests/entity_extraction_test/`, не является production-кодом.
- **Constitution §P9**: нет взаимодействия с Neo4j — клиент изолирован от семантического графа.
- **§P7 (промпты в файлах)**: соблюдается — все промпты загружаются из файлов через `Path(...).read_text()`.
- Ollama API — нативный (не OpenAI-совместимый), транспорт отличается от QwenClient (`httpx.AsyncClient` вместо `AsyncLLMClient`). Но формат промптов и парсинг ответов идентичен QwenClient для унификации.
