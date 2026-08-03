# Feature: QwenClient — клиент модели Qwen3-VL-32B-Thinking

## Motivation
Клиент к Qwen (`Qwen/Qwen3-VL-32B-Thinking`) через OpenAI-совместимый API для экспериментов E3 (combined NER+RE) и E4 (separate NER/RE), а также как RE-провайдер в гибридных экспериментах E5, E6. Переиспользует `AsyncLLMClient` из `semantic_graph/graphrag.py` для сетевых вызовов (Constitution P5).

## Behaviour

### Constructor
```python
class QwenClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,        # env: QWEN_BASE_URL, default "http://192.168.19.127:8888/v1"
        api_key: str,          # env: QWEN_API_KEY, default "EMPTY"
        model: str,            # env: QWEN_MODEL, default "Qwen/Qwen3-VL-32B-Thinking"
        ner_prompt_path: str,  # путь к prompts/ner_prompt.md
        re_prompt_path: str,   # путь к prompts/re_prompt.md
    )
```
Внутренний транспорт: `AsyncLLMClient(base_url, api_key)`. Промпты загружаются из файлов при инициализации через `Path(...).read_text()` (Constitution P7).

### extract_entities(text: str, entity_types: list[str]) → list[PredictedEntity]
1. Форматировать NER-промпт: `prompt.format(input_text=text, entity_types=",".join(entity_types))`
2. Вызвать `await self.llm.generate(messages=[{"role": "user", "content": prompt}], model=self.model)`
3. Распарсить ответ: извлечь строки формата `("entity"<|>NAME<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
4. Для каждой строки: `NAME` → `PredictedEntity.name`, `TYPE` → `PredictedEntity.type`
5. Вернуть `list[PredictedEntity]`

### extract_relations(text: str, entities: list[PredictedEntity], relation_types: list[str]) → list[PredictedRelation]
1. Построить `entities_list` из переданных сущностей: строки `NAME|TYPE`, разделённые `\n`
2. Форматировать RE-промпт: `prompt.format(input_text=text, entities_list=entities_list, relation_types=",".join(relation_types))`
3. Вызвать `await self.llm.generate(...)` (аналогично NER)
4. Распарсить ответ: извлечь строки `("relationship"<|>SRC<|>TGT<|>TYPE)`, разделённые `##`, завершающиеся `<|COMPLETE|>`
5. Для каждой строки: `SRC` → `PredictedRelation.head`, `TGT` → `PredictedRelation.tail`, `TYPE` → `PredictedRelation.type`
6. Вернуть `list[PredictedRelation]`

### Combined mode (E3)
Combined-режим (NER+RE одним вызовом) реализуется на уровне `ExperimentRunner`, а не в `QwenClient`. Раннер формирует combined-промпт, вызывает `extract_entities` (`llm.generate`), парсит и сущности, и отношения из одного ответа. `QwenClient` предоставляет только раздельные методы `extract_entities` и `extract_relations`.

### Error handling
| Сценарий | Поведение |
|----------|-----------|
| Connection error / timeout | `ConnectionError`, логируется, запись считается failed |
| `llm.generate()` вернул `None` | Пустой `[]`, warning в лог |
| Ответ непарсибельный (нет `("entity"` / `("relationship"`) | Пустой `[]`, warning в лог |
| Частично парсибельный ответ (часть строк валидна) | Извлечь валидные строки, пропустить невалидные, warning |

## Data Flow
```
QwenClient
  ├── self.llm: AsyncLLMClient(graphrag.py) ── HTTP ──► Qwen API (192.168.19.127:8888/v1)
  ├── ner_prompt: str (из prompts/ner_prompt.md)
  └── re_prompt: str  (из prompts/re_prompt.md)

extract_entities:
  text + entity_types → format(ner_prompt) → llm.generate() → parse → list[PredictedEntity]

extract_relations:
  text + entities + relation_types → format(re_prompt) → llm.generate() → parse → list[PredictedRelation]
```

## LLM Interactions
| Промпт | Файл | Переменные |
|--------|------|------------|
| NER Prompt | `prompts/ner_prompt.md` | `{input_text}`, `{entity_types}` |
| RE Prompt | `prompts/re_prompt.md` | `{input_text}`, `{entities_list}`, `{relation_types}` |

Оба промпта загружаются из файлов (Constitution P7). Требования к формату выхода: кортежи с разделителем `<|>`, строки разделены `##`, терминатор `<|COMPLETE|>`.

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
3. `llm.generate()` вернул `None` → оба метода возвращают `[]`
4. Непарсибельный ответ (мусор без `("entity"`) → `[]`, warning
5. Частично парсибельный ответ → извлечены только валидные строки
6. Комбинированный ответ (NER+RE в одном теле) парсится раздельно, поскольку формат сущностей и отношений различается префиксами

Фикстуры: `tests/fixtures/model_responses/qwen_ner_response.txt`, `tests/fixtures/model_responses/qwen_re_response.txt`.

## Dependencies
- `openai>=1.0` — `AsyncOpenAI` (внутри `AsyncLLMClient`)
- `semantic_graph/graphrag.py` → `AsyncLLMClient`
- Типы из `models/base_client.py`: `BaseModelClient`, `PredictedEntity`, `PredictedRelation`
- Промпты: `prompts/ner_prompt.md`, `prompts/re_prompt.md`

## Exceptions
- **§5.2 Experiments/Spikes**: клиент находится в изолированной директории `tests/entity_extraction_test/`, не является production-кодом.
- **§P9**: нет взаимодействия с Neo4j — клиент изолирован от семантического графа.
