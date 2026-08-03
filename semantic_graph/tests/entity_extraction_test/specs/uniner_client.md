# Feature: UniNerClient — клиент модели UniNer

## Motivation
Клиент к `Universal-NER/UniNER-7B-all` через кастомный FastAPI сервер. NER-only модель (7B generative LLM). RE не поддерживается. Используется в экспериментах E1, E5 матрицы (см. `general_experiment.md`).

## Behaviour

### Constructor
```python
class UniNerClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,     # ENV: UNINER_BASE_URL, default "http://<remote_host>:<port>"
        timeout: float = 30.0,  # ENV: UNINER_TIMEOUT
    )
```
Транспорт — `httpx.AsyncClient`, инициализируется в конструкторе: `httpx.AsyncClient(base_url=base_url, timeout=timeout)`.

### extract_entities(text: str, entity_types: list[str]) → list[PredictedEntity]
1. Сформировать тело запроса: `{"text": text, "entity_types": entity_types}`. Промпт для UniNer генерируется на серверной стороне FastAPI-приложения (клиент не формирует промпт).
2. Вызвать `await self.client.post("/extract", json={"text": text, "entity_types": entity_types})`
3. Распарсить ответ: `response.json()["entities"]` — список `[{"name": str, "type": str}]`, преобразовать в `list[PredictedEntity]`
4. Если `entities` — пустой список — вернуть `[]`
5. Вернуть `list[PredictedEntity]`

### extract_relations(...) → list[PredictedRelation]
Не реализован. Выбрасывает `NotImplementedError` с сообщением `"UniNerClient не поддерживает RE"`. Поведение наследуется от `BaseModelClient.extract_relations` (см. `general_experiment.md`).

## Data Flow
```
Caller → UniNerClient.extract_entities(text, entity_types)
  → request body construction
  → httpx.AsyncClient.post("/extract", json={...})
  → response.json()["entities"] → parse JSON list
  → list[PredictedEntity]
```
Хранилища не затрагиваются.

FastAPI-эндпоинт (реализуется на серверной стороне, приведено для справки):
- **POST /extract**, request: `{"text": str, "entity_types": list[str]}`, response (200): `{"entities": [{"name": str, "type": str}]}`

## LLM Interactions
Промпт генерируется inline на серверной стороне (не вынесен в файл клиента — см. Exceptions).

## LLM Model Requirements
- **Тип модели**: text-only generative LLM
- **Размер**: 7B параметров
- **Язык выхода**: английский (названия сущностей — язык исходного текста)
- **Формат выхода**: structured list в свободном тексте (`<entity_type>: <entity_name>`)

## Error Handling
| Ситуация | Поведение |
|----------|-----------|
| Connection error / timeout (`httpx.ConnectError`, `httpx.TimeoutException`) | `ConnectionError`, ошибка логируется |
| HTTP ошибка (4xx, 5xx) | Лог + `ConnectionError` |
| Unparseable response (не JSON или нет ключа `"entities"`) | Возврат `[]`, warning в лог |
| Empty response (`"entities": []`) | Возврат `[]` |
| Вызов `extract_relations` | `NotImplementedError`, не логируется (ожидаемое поведение) |

## Testing
| Тест | Вход | Ожидаемый результат |
|------|------|---------------------|
| `extract_entities` с валидным ответом FastAPI | Мок `httpx.AsyncClient`, ответ `httpx.Response(200, json={"entities": [{"name": "Apple", "type": "Organization"}, {"name": "Steve Jobs", "type": "Person"}]})` | `[PredictedEntity(name="Apple", type="Organization"), PredictedEntity(name="Steve Jobs", type="Person")]` |
| Пустой ответ (entities=[]) | Мок возвращает `httpx.Response(200, json={"entities": []})` | `[]` |
| HTTP ошибка (500) | Мок возвращает `httpx.Response(500, json={"error": "internal error"})` | `ConnectionError`, ошибка в лог |
| Невалидный JSON-ответ | Мок возвращает `httpx.Response(200, content=b"not json")` | `[]`, warning в лог |
| Ответ без ключа `"entities"` | Мок возвращает `httpx.Response(200, json={"other_key": "value"})` | `[]`, warning в лог |
| `extract_relations` | Любые аргументы | `NotImplementedError("UniNerClient не поддерживает RE")` |

## Dependencies
- `httpx>=0.27` (`AsyncClient`)
- `pytest-httpx>=0.30` (для тестирования)
- Типы `BaseModelClient`, `PredictedEntity`, `PredictedRelation` — из `general_experiment.md` (реализуются в `models/base_client.py`)

## Exceptions
- **Constitution §P7 (промпты в файлах)**: промпт UniNer — однострочный шаблон, генерируемый inline на серверной стороне. Вынос в файл для такого тривиального случая избыточен. Экспериментальный код (§5.2).
- **§P9**: нет взаимодействия с Neo4j — эксперимент изолирован.
