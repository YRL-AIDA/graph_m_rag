# Feature: CoNLL04 Dataset Loader

## Motivation
Загрузчик датасета CoNLL04 из HuggingFace в унифицированный формат `DatasetRecord` для экспериментального фреймворка NER/RE. Нормализует токен-базированные аннотации в символьные спаны, совместимые с моделями.

## Behaviour

### Source
- `load_dataset("DFKI-SLT/conll04")` — HuggingFace `datasets`
- Сплиты: `train` (922 записи), `validation` (231), `test` (288)

### Input (per record)
```json
{"tokens": list[str], "entities": [{type, start, end}], "relations": [{type, head, tail}]}
```

### Processing (normalization algorithm)
1. `text = " ".join(tokens)` — склейка токенов через пробел.
2. **Entities**: для каждой `{type, start, end}`:
   - `name = " ".join(tokens[start:end+1])`
   - `start_char` / `end_char` вычисляются через кумулятивные длины токенов с одиночными пробелами:
     ```
     start_char = sum(len(tokens[i]) for i in range(start)) + start
     end_char   = start_char + len(name) - 1
     ```
   - Пример single-token: `tokens=["John", "lives", "in", "NYC"]`, entity `{Peop, 0, 0}` → `name="John"`, `start_char=0`, `end_char=3`.
   - Пример multi-token: `tokens=["New", "York"]`, entity `{Loc, 0, 1}` → `name="New York"`, `start_char=0`, `end_char=7`.
3. **Relations**: для каждой `{type, head, tail}`:
   - `head_idx = head`, `tail_idx = tail` (прямые индексы в gold-списке сущностей)
   - `head_start`, `head_end`, `tail_start`, `tail_end` — копируются из `Entity` по индексам `head_idx`, `tail_idx`.
4. **ID**: `f"conll04_{split}_{idx}"` (split ∈ {train, validation, test}, idx — 0-based позиция в датасете).

### Output
`list[DatasetRecord]`. Модели `Entity`, `Relation`, `DatasetRecord` определены в `general_experiment.md` (§ 1. Unified Data Model).

### Types
- **Entity types**: `["Peop", "Loc", "Org", "Other"]`
- **Relation types**: `["Located_In", "Work_For", "OrgBased_In", "Live_In", "Kill"]`

### Base class
Наследует `DatasetLoader` ABC (файл `datasets/base_loader.py`, определён в `experiment_feature.md` § 2.2):
- `entity_types: list[str]` → `["Peop", "Loc", "Org", "Other"]`
- `relation_types: list[str]` → `["Located_In", "Work_For", "OrgBased_In", "Live_In", "Kill"]`
- `load(split: str) -> list[DatasetRecord]` — загружает и нормализует сплит
- `splits() -> list[str]` → `["train", "validation", "test"]`

## Error Handling

| Сценарий | Поведение |
|----------|-----------|
| HF-датасет недоступен (offline) | `ConnectionError` / `FileNotFoundError` от `datasets`, пробрасывается наверх |
| Неизвестный split | `ValueError(f"Unknown split '{split}'. Available: {self.splits()}")` |
| Пустой датасет после загрузки | `RuntimeWarning`, возврат `[]` |
| Сущность с `start > end` | `ValueError` — битые данные (не должно встречаться в CoNLL04) |

## Testing

| # | Тест-кейс | Ожидаемый результат |
|---|-----------|---------------------|
| 1 | `tokens=["John", "lives", "in", "NYC"]`, entity `{Peop, 0, 0}` | `name="John"`, `start_char=0`, `end_char=3` |
| 2 | `tokens=["New", "York"]`, entity `{Loc, 0, 1}` | `name="New York"`, `start_char=0`, `end_char=7` |
| 3 | `tokens=["A", "B", "C"]`, entity `{Org, 1, 2}` | `name="B C"`, `start_char=2`, `end_char=4` |
| 4 | Отношения: `head_idx`/`tail_idx` матчатся с entity-списком | Индексы прямого соответствия |
| 5 | `head_start`/`head_end`/`tail_start`/`tail_end` в `Relation` | Копии полей `start`/`end` Entity по `head_idx`/`tail_idx` |
| 6 | `entity_types` и `relation_types` | `["Peop", "Loc", "Org", "Other"]`, `["Located_In", ...]` |
| 7 | Размеры сплитов | train=922, validation=231, test=288 |
| 8 | ID формат `conll04_test_5` | Содержит split и 0-based индекс |

## Dependencies
- **Внутренние**: `datasets/base_loader.py` — `DatasetLoader` (ABC), плюс `Entity`, `Relation`, `DatasetRecord` (Pydantic-модели из `general_experiment.md`).
- **Внешние**: `datasets>=2.0`, `pydantic>=2.0`.

## Exceptions
— (нет отклонений от Constitution)
