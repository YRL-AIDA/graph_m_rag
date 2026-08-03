# Feature: SCIERC Dataset Loader

## Motivation
Загрузчик датасета SCIERC из локальных JSON-файлов, нормализующий multi-sentence документы в унифицированный формат `DatasetRecord`. SCIERC содержит научные аннотации с составными типами сущностей и RE на основе спанов — в отличие от CoNLL04, где отношения заданы индексами в золотом списке сущностей. Загрузчик изолирован: не зависит от HuggingFace `datasets`, работает с локальными файлами.

## Behaviour

### Source
Локальные JSON-файлы по пути `/home/ivan/work/ooo/graph_m_rag/datasets_/scierc/json/`:

| Сплит | Файл | Документов |
|-------|------|-----------|
| `train` | `train.json` | 350 |
| `validation` | `dev.json` | 50 |
| `test` | `test.json` | 100 |

### Input Format
Каждый документ — один JSON-объект (одна строка в файле):
```json
{
  "doc_key": "<str>",
  "sentences": [["tok1", "tok2", ...], [...]],
  "ner": [[[start, end, type], ...], [...]],
  "relations": [[[h_start, h_end, t_start, t_end, type], ...], [...]]
}
```
`ner[i]` и `relations[i]` соответствуют `sentences[i]` (параллельные массивы).

### Processing (нормализация)
Для каждого документа, для каждого предложения `i`:
1. `text = " ".join(sentences[i])` — токены склеиваются одиночным пробелом.
2. **NER → Entity**: для каждого `[start, end, type]` в `ner[i]`:
   - `name = " ".join(sentences[i][start:end+1])`
   - `start_char` / `end_char` вычисляются через кумулятивные длины токенов с пробелами между токенами (0-based, `end` inclusive — соответствует модели `Entity` из `general_experiment.md`).
3. **RE → Relation**: для каждого `[h_start, h_end, t_start, t_end, type]` в `relations[i]`:
   - `head_idx` / `tail_idx` определяются поиском NER-спана в `ner[i]` с совпадающими `start`/`end`.
   - Если соответствующий NER-спан **не найден** — отношение **пропускается** (не попадает в `DatasetRecord.relations`), в лог выводится `WARNING` с `doc_key`, индексом предложения и ожидаемыми/фактическими значениями спанов NER-аннотаций для этого предложения.
   - При совпадении — подставляются вычисленные `head_start`/`head_end`/`tail_start`/`tail_end` из соответствующих `Entity`.
4. `id = f"scierc_{doc_key}_sent{i}"`

### Output
`list[DatasetRecord]` — по одной записи на предложение. Модели `Entity`, `Relation`, `DatasetRecord` определены в `general_experiment.md`; данный загрузчик их импортирует, не переопределяет.

### Types
- **Entity types**: определяются **динамически** при первом вызове `load()` — собираются все уникальные значения `type` из NER-аннотаций **полного датасета** (всех трёх сплитов). Кэшируются в `self._entity_types`. Ожидаемые (не исчерпывающие): `Generic`, `Material`, `Method`, `Task`, `OtherScientificTerm`, `Metric`, `ORGANIZATION`, `PERSON`, `ORGANIZATION|PERSON`.
- **Relation types**: фиксированный список `["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "CONJUNCTION", "COMPARE", "EVALUATE-FOR", "PART-OF"]` (7 типов).

### Base Class
Наследует `DatasetLoader` (ABC из `testdata/base_loader.py`, контракт описан в `general_experiment.md`):
- `entity_types` (property) → динамически собранный список
- `relation_types` (property) → фиксированный список
- `load(split: str) -> list[DatasetRecord]` — загружает указанный сплит
- `splits() -> list[str]` → `["train", "validation", "test"]`

## Data Flow
```
datasets_/scierc/json/{split}.json  →  json.load (весь файл)
  →  for doc in data:
       for i in range(len(doc["sentences"])):
         normalize sentence i → DatasetRecord
  →  list[DatasetRecord]
```

## Error Handling
| Сценарий | Поведение |
|----------|-----------|
| Файл сплита не найден (`datasets_/scierc/json/train.json` отсутствует) | `FileNotFoundError` с путём к файлу |
| JSON не парсится (malformed) | `json.JSONDecodeError` с именем файла и номером строки |
| RE-спан не соответствует ни одному NER-спану | `WARNING` в лог, отношение пропускается, выполнение продолжается |
| Документ без предложений (`"sentences": []`) | Документ пропускается (0 записей), без ошибки |

## Testing
1. **Multi-sentence документ**: документ из 3 предложений → 3 отдельных `DatasetRecord` с ID `scierc_{doc_key}_sent0`, `_sent1`, `_sent2`.
2. **RE-маппинг — совпадение**: отношение с `[h_start, h_end]`, совпадающими с NER-спаном → корректно подставлены `head_idx`, `head_start`, `head_end`.
3. **RE-маппинг — несовпадение**: отношение с `head_start/head_end` не совпадает ни с одним NER-спаном → `WARNING` в лог, отношение пропущено.
4. **Динамические типы**: `entity_types` содержит все уникальные типы из NER-аннотаций полного датасета (включая составной `ORGANIZATION|PERSON`).
5. **Символьные индексы**: multi-token сущность внутри предложения → `start_char`/`end_char` корректны (учитывают кумулятивные длины с пробелами).
6. **Все 3 сплита**: `splits()` возвращает `["train", "validation", "test"]`, каждый сплит загружается без ошибок.

## Dependencies
- Стандартная библиотека: `json`, `logging`, `pathlib.Path`
- Импорт из `testdata/base_loader.py`: `Entity`, `Relation`, `DatasetRecord`, `DatasetLoader`

## Exceptions
- **Constitution §5.2 — Experiments/Spikes**: код находится в `semantic_graph/tests/entity_extraction_test/`, изолирован от основного пайплайна. Спецификация написана spec-first — более строгий подход.
- Нет взаимодействия с Neo4j, Qdrant, MinIO — фреймворк чисто оценочный.
