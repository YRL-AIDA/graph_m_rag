# Feature: QwenClient — клиент модели Qwen3-VL-32B-Thinking

## Motivation
Клиент к Qwen (`Qwen/Qwen3-VL-32B-Thinking`) через OpenAI-совместимый API для экспериментов E3 (combined NER+RE) и E4 (separate NER/RE), а также как RE-провайдер в гибридных экспериментах E5, E6. Переиспользует `AsyncLLMClient` из `semantic_graph/graphrag.py` для сетевых вызовов (Constitution P5).

## Behaviour

### Constructor
```python
class QwenClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,              # env: QWEN_BASE_URL, default "http://192.168.19.127:8888/v1"
        api_key: str,               # env: QWEN_API_KEY, default "EMPTY"
        model: str,                 # env: QWEN_MODEL, default "Qwen/Qwen3-VL-32B-Thinking"
        ner_prompt_path: str,       # путь к prompts/ner_prompt.md
        re_prompt_path: str,        # путь к prompts/re_prompt.md
        combined_prompt_path: str,  # путь к prompts/combined_prompt.md
    )
```
Внутренний транспорт: `AsyncLLMClient(base_url, api_key)`. Промпты загружаются из файлов при инициализации через `Path(...).read_text()` (Constitution P7):
- `self.ner_prompt = Path(ner_prompt_path).read_text()`
- `self.re_prompt = Path(re_prompt_path).read_text()`
- `self.combined_prompt = Path(combined_prompt_path).read_text()`

### extract_entities(text: str, entity_types: list[str]) → list[PredictedEntity]
1. Форматировать NER-промпт: `prompt.format(input_text=text, entity_types=",".join(entity_types))`
2. Вызвать `await self.llm.generate(messages=[{"role": "user", "content": prompt}], model=self.model)`
3. Распарсить ответ через `_parse_ner_response(response, entity_types)`:
   - Извлечь строки формата `("entity"<|>NAME<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
   - Для каждой строки: `NAME` → `PredictedEntity.name`, `TYPE` → `PredictedEntity.type`
   - Применить валидацию типа (см. секцию Type Validation)
4. Вернуть `list[PredictedEntity]`

### extract_relations(text: str, entities: list[PredictedEntity], relation_types: list[str]) → list[PredictedRelation]
1. Построить `entities_list` из переданных сущностей: строки `NAME|TYPE`, разделённые `\n`
2. Форматировать RE-промпт: `prompt.format(input_text=text, entities_list=entities_list, relation_types=",".join(relation_types))`
3. Вызвать `await self.llm.generate(...)` (аналогично NER)
4. Распарсить ответ через `_parse_re_response(response, relation_types)`:
   - Извлечь строки `("relationship"<|>SRC<|>TGT<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
   - Для каждой строки: `SRC` → `PredictedRelation.head`, `TGT` → `PredictedRelation.tail`, `TYPE` → `PredictedRelation.type`
   - Применить валидацию типа (см. секцию Type Validation)
5. Вернуть `list[PredictedRelation]`

### extract_entities_and_relations(text: str, entity_types: list[str], relation_types: list[str]) → tuple[list[PredictedEntity], list[PredictedRelation]]
1. Форматировать combined-промпт: `prompt.format(input_text=text, entity_types=",".join(entity_types), relation_types=",".join(relation_types))`
2. Вызвать `await self.llm.generate(messages=[{"role": "user", "content": prompt}], model=self.model)` — **один вызов**
3. Из одного ответа распарсить и сущности, и отношения:
   - Строки с префиксом `("entity"<|>...)` → `PredictedEntity`
   - Строки с префиксом `("relationship"<|>...)` → `PredictedRelation`
   - Терминатор `<|COMPLETE|>` общий для обоих типов
4. Применить валидацию типов (см. ниже):
   - Для `PredictedEntity`: `normalize_type(entity.type, entity_types)`, при `None` — пропустить, warning
   - Для `PredictedRelation`: `normalize_type(relation.type, relation_types)`, при `None` — пропустить, warning
5. Вернуть кортеж `(entities, relations)`

### Type Validation
Все методы парсинга (`_parse_ner_response`, `_parse_re_response`, парсинг в `extract_entities_and_relations`) обязаны применять валидацию типов сущностей/отношений:

```python
def _parse_ner_response(self, response: str, entity_types: list[str]) -> list[PredictedEntity]:
    ...

def _parse_re_response(self, response: str, relation_types: list[str]) -> list[PredictedRelation]:
    ...
```

После извлечения каждой сущности/отношения:
1. Для `PredictedEntity`: вызвать `normalize_type(entity.type, entity_types)` (импортируется из `metrics.metrics`).
   - Если вернулся `None` — пропустить сущность, залогировать warning.
   - Если вернулся канонический тип — заменить `entity.type` на канонический.
2. Для `PredictedRelation`: вызвать `normalize_type(relation.type, relation_types)`.
   - Если `None` — пропустить отношение, залогировать warning.
   - Если канонический тип — заменить `relation.type` на канонический.

`normalize_type` и `TYPE_SYNONYMS` импортируются из `metrics.metrics`:
```python
from metrics.metrics import normalize_type, TYPE_SYNONYMS
```

Сигнатуры `extract_entities` и `extract_relations` пробрасывают `entity_types` / `relation_types` в соответствующие методы парсинга.

### Combined mode (E3)
Combined-режим (NER+RE одним вызовом) реализован в `QwenClient` через метод `extract_entities_and_relations()`. `ExperimentRunner` вызывает этот метод для E3 (`re_mode="combined_single_call"`).

### Error handling
| Сценарий | Поведение |
|----------|-----------|
| Connection error / timeout | `ConnectionError`, логируется, запись считается failed |
| `llm.generate()` вернул `None` | Пустой `[]` (или `([], [])` для combined), warning в лог |
| Ответ непарсибельный (нет `("entity"` / `("relationship"`) | Пустой `[]` (или `([], [])`), warning в лог |
| Частично парсибельный ответ (часть строк валидна) | Извлечь валидные строки, пропустить невалидные, warning |
| Тип сущности/отношения не прошёл `normalize_type` (вернулся `None`) | Пропустить запись, warning в лог |

## Data Flow
```
QwenClient
  ├── self.llm: AsyncLLMClient(graphrag.py) ── HTTP ──► Qwen API (192.168.19.127:8888/v1)
  ├── ner_prompt: str      (из prompts/ner_prompt.md)
  ├── re_prompt: str       (из prompts/re_prompt.md)
  └── combined_prompt: str (из prompts/combined_prompt.md)

extract_entities:
  text + entity_types → format(ner_prompt) → llm.generate() → _parse_ner_response(response, entity_types) → normalize_type → list[PredictedEntity]

extract_relations:
  text + entities + relation_types → format(re_prompt) → llm.generate() → _parse_re_response(response, relation_types) → normalize_type → list[PredictedRelation]

extract_entities_and_relations (E3 combined_single_call):
  text + entity_types + relation_types → format(combined_prompt) → llm.generate() → parse combined → normalize_type → (list[PredictedEntity], list[PredictedRelation])
```

## LLM Interactions
| Промпт | Файл | Переменные |
|--------|------|------------|
| NER Prompt | `prompts/ner_prompt.md` | `{input_text}`, `{entity_types}` |
| RE Prompt | `prompts/re_prompt.md` | `{input_text}`, `{entities_list}`, `{relation_types}` |
| Combined Prompt | `prompts/combined_prompt.md` | `{input_text}`, `{entity_types}`, `{relation_types}` |

Все промпты загружаются из файлов (Constitution P7). Требования к формату выхода: кортежи с разделителем `<|>`, строки разделены `##`, терминатор `<|COMPLETE|>`.

Формат combined-ответа — тот же, что и для раздельных вызовов, но в одном теле:
```
("entity"<|>Apple Inc.<|>Organization)
##
("relationship"<|>Apple Inc.<|>Steve Jobs<|>founded_by)
##
("entity"<|>Steve Jobs<|>Person)
<|COMPLETE|>
```

## LLM Model Requirements
- **Тип модели**: multimodal (text+image), используется как text-only
- **Минимальный размер контекста**: 4096 токенов
- **Язык выхода**: английский
- **Формат выхода**: структурированный текст (кортежи `("entity"<|>NAME<|>TYPE)` / `("relationship"<|>SRC<|>TGT<|>TYPE)`)
- **Конкретная модель**: `Qwen/Qwen3-VL-32B-Thinking`
- **API**: OpenAI-совместимый (`http://192.168.19.127:8888/v1`)

## Testing
Все тесты с мокнутым `AsyncLLMClient` (Constitution T3):
1. `extract_entities` с валидным ответом → корректный `list[PredictedEntity]`
2. `extract_relations` с валидным ответом → корректный `list[PredictedRelation]`
3. `extract_entities_and_relations` с combined-ответом → корректный кортеж `(entities, relations)`
4. `llm.generate()` вернул `None` → оба метода возвращают `[]`; `extract_entities_and_relations` возвращает `([], [])`
5. Непарсибельный ответ (мусор без `("entity"`) → `[]`, warning
6. Частично парсибельный ответ → извлечены только валидные строки
7. Комбинированный ответ (NER+RE в одном теле) парсится раздельно, поскольку формат сущностей и отношений различается префиксами
8. Валидация типов: сущность с неизвестным типом (не прошедшим `normalize_type`) отбрасывается, warning в лог
9. Валидация типов: тип из `TYPE_SYNONYMS` (например `"organization"`) маппится в канонический (`"Org"`)

Фикстуры: `tests/fixtures/model_responses/qwen_ner_response.txt`, `tests/fixtures/model_responses/qwen_re_response.txt`, `tests/fixtures/model_responses/qwen_combined_response.txt`.

## Dependencies
- `openai>=1.0` — `AsyncOpenAI` (внутри `AsyncLLMClient`)
- `semantic_graph/graphrag.py` → `AsyncLLMClient`
- Типы из `models/base_client.py`: `BaseModelClient`, `PredictedEntity`, `PredictedRelation`
- Промпты: `prompts/ner_prompt.md`, `prompts/re_prompt.md`, `prompts/combined_prompt.md`
- `metrics.metrics` → `normalize_type`, `TYPE_SYNONYMS`

## Exceptions
- **§5.2 Experiments/Spikes**: клиент находится в изолированной директории `tests/entity_extraction_test/`, не является production-кодом.
- **§P9**: нет взаимодействия с Neo4j — клиент изолирован от семантического графа.
