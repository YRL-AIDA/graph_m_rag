# Feature: Общий контракт экспериментального фреймворка NER/RE

## Motivation
Создать изолированный экспериментальный фреймворк внутри `semantic_graph/tests/entity_extraction_test/` для воспроизводимой оценки F1 и скорости различных комбинаций NER/RE моделей на CoNLL04 и SCIERC. Фреймворк самодостаточен, за исключением переиспользования `AsyncLLMClient` из `graphrag.py` для Qwen. Статус: Experiments/Spikes (Constitution §5.2).

## Overview

### Экспериментальная матрица (E1–E6)

| ID | NER model | RE model | RE call type | Datasets |
|----|-----------|----------|--------------|----------|
| E1 | UniNer | — | — | CoNLL04, SCIERC |
| E2 | Gleaner | — | — | CoNLL04, SCIERC |
| E3 | Qwen | Qwen | combined (NER+RE одним промптом) | CoNLL04, SCIERC |
| E4 | Qwen | Qwen | separate (NER затем RE) | CoNLL04, SCIERC |
| E5 | UniNer | Qwen | hybrid (NER=UniNer, RE=Qwen) | CoNLL04, SCIERC |
| E6 | Gleaner | Qwen | hybrid (NER=Gleaner, RE=Qwen) | CoNLL04, SCIERC |

### Модели

| Модель | HF ID | Тип | Инференс |
|--------|-------|-----|----------|
| UniNer | `Universal-NER/UniNER-7B-all` | Generative LLM (7B) | FastAPI |
| Gleaner | `urchade/gliner_large-v2.1` | Encoder bi-encoder (DeBERTa ~300M), НЕ generative | Кастомный FastAPI, OpenAI-совместимый эндпоинт |
| Qwen | `Qwen/Qwen3-VL-32B-Thinking` | Generative VLM (32B) | Уже запущен: `http://192.168.19.127:8888/v1` |

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

### 3. ExperimentResult

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
                              # avg_response_time_sec, total_samples
    timestamp: str            # ISO 8601
```

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
     │
     ▼
Metrics:
  compute_ner_f1(gold_entities, pred_entities) → {precision, recall, f1}
  compute_re_f1(gold_entities, gold_relations, pred_entities, pred_relations) → {precision, recall, f1}
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
