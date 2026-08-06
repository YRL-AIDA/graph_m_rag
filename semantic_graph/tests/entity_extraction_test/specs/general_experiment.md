# Feature: Общий контракт экспериментального фреймворка NER/RE

## Motivation
Создать изолированный экспериментальный фреймворк внутри `semantic_graph/tests/entity_extraction_test/` для воспроизводимой оценки F1 и скорости различных комбинаций NER/RE моделей на CoNLL04 и SCIERC. Фреймворк самодостаточен, за исключением переиспользования `AsyncLLMClient` из `graphrag.py` для Qwen. OllamaClient использует свой транспорт через `httpx.AsyncClient` для нативного Ollama API. Статус: Experiments/Spikes (Constitution §5.2).

## Overview

### Экспериментальная матрица (E1–E8)

| ID | NER model | RE model | RE call type | Datasets |
|----|-----------|----------|--------------|----------|
| E1 | UniNer | — | — | CoNLL04, SCIERC |
| E2 | Gleaner | — | — | CoNLL04, SCIERC |
| E3 | Qwen | Qwen | combined_single_call (один API-вызов Qwen для NER+RE) | CoNLL04, SCIERC |
| E4 | Qwen | Qwen | separate (NER затем RE) | CoNLL04, SCIERC |
| E5 | UniNer | Qwen | hybrid (NER=UniNer, RE=Qwen) | CoNLL04, SCIERC |
| E6 | Gleaner | Qwen | hybrid (NER=Gleaner, RE=Qwen) | CoNLL04, SCIERC |
| E7 | Ollama | Ollama | combined_single_call (один API-вызов Ollama для NER+RE) | CoNLL04, SCIERC |
| E8 | Ollama | Ollama | separate (NER затем RE) | CoNLL04, SCIERC |

### Модели

| Модель | HF ID | Тип | Инференс |
|--------|-------|-----|----------|
| UniNer | `Universal-NER/UniNER-7B-all` | Generative LLM (7B) | FastAPI |
| Gleaner | `urchade/gliner_large-v2.1` | Encoder bi-encoder (DeBERTa ~300M), НЕ generative | Кастомный FastAPI, OpenAI-совместимый эндпоинт |
| Qwen | `Qwen/Qwen3-VL-32B-Thinking` | Generative VLM (32B) | Уже запущен: `http://192.168.19.127:8888/v1` |
| Ollama | Любая модель в Ollama (по умолчанию `qwen3-coder:30b`) | Generative LLM | Нативный Ollama API: `http://192.168.55.242:7869` |

## Behaviour

### 1. Unified Data Model

Pydantic-модели, используемые ВСЕМИ dataloader-ами на выходе и ВСЕМИ model clients на входе/выходе:

```python
class Entity(BaseModel):
    name: str         # Текст сущности (токены, склеенные пробелом)
    type: str         # Тип сущности
    start: int        # Стартовый символьный индекс (0-based) в поле text
    end: int          # Конечный символьный индекс (0-based, inclusive) в поле text

class Relation(BaseModel):
    head_idx: int     # Индекс head-сущности в списке entities
    tail_idx: int     # Индекс tail-сущности в списке entities
    head_start: int   # Стартовый символьный индекс head-сущности
    head_end: int     # Конечный символьный индекс head-сущности
    tail_start: int   # Стартовый символьный индекс tail-сущности
    tail_end: int     # Конечный символьный индекс tail-сущности
    type: str         # Тип отношения

class DatasetRecord(BaseModel):
    id: str                   # Уникальный ID записи
    text: str                 # Полный текст записи
    entities: list[Entity]    # Gold-сущности
    relations: list[Relation] # Gold-отношения
```

### 2. Unified Model Interface

Общие типы и ABC, реализуемые ВСЕМИ клиентами моделей:

```python
class PredictedEntity(BaseModel):
    name: str   # Название сущности (для матчинга с PredictedRelation.head/tail)
    type: str   # Тип сущности

class PredictedRelation(BaseModel):
    head: str    # name head-сущности (матчинг с PredictedEntity.name)
    tail: str    # name tail-сущности
    type: str    # Тип отношения

class BaseModelClient(ABC):
    @abstractmethod
    async def extract_entities(
        self, text: str, entity_types: list[str]
    ) -> list[PredictedEntity]:
        """Извлечь и классифицировать сущности из текста."""
        ...

    async def extract_relations(
        self,
        text: str,
        entities: list[PredictedEntity],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        """Извлечь отношения между заданными сущностями.
        NER-only модели (UniNer, Gleaner) НЕ реализуют. По умолчанию — NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает RE")
```

### 3. Name and Type Normalization

Две утилитарные функции, используемые везде — и в метриках, и в клиентах:

```python
def normalize_name(name: str) -> str:
    """Нормализовать имя сущности: lower(), strip(), схлопнуть множественные пробелы в один."""
    return " ".join(name.lower().strip().split())

def normalize_type(typ: str, allowed_types: list[str]) -> str | None:
    """Сопоставить тип, возвращённый моделью, с каноническим типом из датасета.

    Алгоритм:
    1. Точное совпадение (case-insensitive) с одним из allowed_types → вернуть канонический тип.
    2. Поиск в TYPE_SYNONYMS: если typ есть в словаре синонимов → вернуть канонический тип.
    3. Иначе → None (тип не распознан — сущность/отношение отбрасывается).
    """
    ...
```

**Словарь `TYPE_SYNONYMS`** — mapping от вариантов написания к каноническим типам CoNLL04:

| Варианты (lowercase) | Канонический тип |
|----------------------|------------------|
| `"person"`, `"people"`, `"human"` | `"Peop"` |
| `"location"`, `"place"`, `"loc"` | `"Loc"` |
| `"organization"`, `"organisation"`, `"company"`, `"corporation"`, `"corp"` | `"Org"` |
| `"other"`, `"miscellaneous"`, `"misc"` | `"Other"` |

Для **SciERC** синонимы не нужны (типы специфичны: `"Task"`, `"Method"`, `"Metric"`, `"Material"`, `"OtherScientificTerm"`, `"Generic"`), только case-insensitive match с `allowed_types`.

### 4. Type Validation in Clients

Все клиенты (`QwenClient`, `UniNerClient`, `GleanerClient`, `OllamaClient`) при парсинге ответа модели **обязаны**:

1. **Для каждого извлечённого `PredictedEntity`**: вызвать `normalize_type(ent.type, entity_types)`.
   - Если вернулся `None` — пропустить эту сущность (не включать в результат).
   - Если вернулся канонический тип — заменить `ent.type` на канонический.

2. **Для каждого извлечённого `PredictedRelation`**: вызвать `normalize_type(rel.type, relation_types)`.
   - Если `None` — пропустить отношение.

3. **Логировать `warning`** для каждой отброшенной сущности/отношения с указанием исходного типа и списка доступных типов (`entity_types`/`relation_types`).

### 5. E3 Combined Mode

E3 использует выделенный combined-промпт из файла `prompts/combined_prompt.md`.

`QwenClient` предоставляет метод:

```python
async def extract_entities_and_relations(
    self, text: str, entity_types: list[str], relation_types: list[str]
) -> tuple[list[PredictedEntity], list[PredictedRelation]]:
    """Делает ОДИН вызов llm.generate() и парсит из одного ответа и сущности, и отношения."""
    ...
```

**Формат combined-ответа**: строки `("entity"<|>NAME<|>TYPE)` и `("relationship"<|>SRC<|>TGT<|>TYPE)`, разделённые `##`, с единым терминатором `<|COMPLETE|>`.

**Пример ответа модели**:
```
("entity"<|>John Smith<|>person)<|NEWLINE|>("entity"<|>Microsoft<|>organization)<|NEWLINE|>##<|NEWLINE|>("relationship"<|>John Smith<|>Microsoft<|>Work_For)<|NEWLINE|><|COMPLETE|>
```

**Парсинг**: используются разные регулярные выражения для сущностей и отношений, извлекая их из одного тела ответа. Валидация типов (`normalize_type`) применяется и к сущностям, и к отношениям.

### 6. ExperimentResult

```python
class ExperimentResult(BaseModel):
    experiment_id: str        # "E1"–"E6"
    model_ner: str            # "uniner"|"gliner"|"qwen"|"none"
    model_re: str             # "qwen"|"combined_qwen"|"none"
    dataset: str              # "conll04"|"scierc"
    split: str                # "test"|"validation"|"train"
    task_type: str            # "ner_only"|"re_only"|"ner_re"
    config: dict              # Параметры запуска
    metrics: dict[str, float] # precision_ner, recall_ner, f1_ner,
                              # precision_re, recall_re, f1_re,
                              # avg_response_time_sec, total_samples,
                              # ner_entity_count
    timestamp: str            # ISO 8601
```

**Поля `metrics`**:
- `precision_ner`, `recall_ner`, `f1_ner` — NER-метрики
- `precision_re`, `recall_re`, `f1_re` — RE-метрики
- `avg_response_time_sec` — среднее время ответа модели (в секундах)
- `total_samples` — количество обработанных записей
- `ner_entity_count` — среднее количество предсказанных NER-сущностей на запись (после фильтрации типов через `normalize_type`)

## Data Flow

```
DatasetLoader → records: list[DatasetRecord]
     │
     ▼
Model Clients:
  QwenClient    → extract_entities / extract_relations → list[PredictedEntity] / list[PredictedRelation]
  UniNerClient  → extract_entities → list[PredictedEntity]
  GleanerClient → extract_entities → list[PredictedEntity]
  HybridClient  → делегирует NER → ner_client, RE → re_client
  OllamaClient → extract_entities / extract_relations → list[PredictedEntity] / list[PredictedRelation]
     │
     ▼
Metrics:
  compute_ner_f1(gold_entities, pred_entities) → {precision, recall, f1}

    NER Entity matching:
    - gold и pred сущности сопоставляются по нормализованным (name, type).
    - gold.name и pred.name проходят через `normalize_name()`.
    - gold.type и pred.type проходят через `normalize_type()` с allowed_types=entity_types датасета.
    - Совпадение — только если оба нормализованных значения равны.
    - Жадный алгоритм: одна gold-сущность на одну pred-сущность.

    Из совпадений вычисляются precision, recall, F1.

  compute_re_f1(gold_entities, gold_relations, pred_entities, pred_relations) → {precision, recall, f1}

    RE Entity matching:
    - gold и pred сущности сопоставляются по нормализованным (name, type).
    - gold.name и pred.name проходят через `normalize_name()`.
    - gold.type и pred.type проходят через `normalize_type()` с allowed_types=entity_types датасета.
    - Совпадение — только если оба нормализованных значения равны.
    - Жадный алгоритм: одна gold-сущность на одну pred-сущность.

    После матчинга сущностей, отношения сопоставляются по парам (head_idx, tail_idx, type)
    с использованием замапленных индексов. Вычисляются precision, recall, F1.

  compute_avg_response_time(timings: list[float]) → float
     │
     ▼
ExperimentRunner:
  Для всех (dataset, split, experiment):
    load → extract → compute metrics → save ExperimentResult
  Вывод: results/<timestamp>_experiments.json + results/<timestamp>_summary.md
```

## Dependencies

**Внутренние**: `semantic_graph/graphrag.py` — `AsyncLLMClient` (переиспользуется только QwenClient).

**Внешние**: `pydantic>=2.0`, `pydantic-settings>=2.0`, `openai>=1.0`, `aiohttp>=3.8`, `datasets>=2.0`, `pandas>=1.5`, `numpy>=1.24`.

**Тестирование**: `pytest>=7.0`, `pytest-asyncio>=0.21`.

## Testing

Все LLM-вызовы мокаются. CI: `pytest semantic_graph/tests/entity_extraction_test/tests/`. Live-вызовы API — только вручную, вне CI.

## Exceptions

Отклонения от Constitution:
- **§5.2 Experiments/Spikes**: код в изолированной директории `tests/`, допускается реализация без полного spec-first (данная спецификация написана до реализации — более строгий подход).
- **§P6**: фреймворк оперирует списками Pydantic-моделей (`list[DatasetRecord]`, `list[PredictedEntity]`), а не DataFrames — осознанно, экспериментальный формат.
- **§P9**: нет взаимодействия с Neo4j, фреймворк изолирован от семантического графа.
- Единственная связь с основным кодом: импорт `AsyncLLMClient` из `semantic_graph/graphrag.py`.
