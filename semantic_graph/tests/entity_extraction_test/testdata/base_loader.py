"""Базовый модуль: общие Pydantic-модели, интерфейсы и ABC для NER/RE экспериментов.

Все компоненты экспериментального фреймворка (dataloader, model clients, runner)
используют типы и ABC, определённые в этом модуле.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# 1. Unified Data Model — золотые (gold) сущности/отношения из датасетов
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Сущность из золотой разметки датасета."""

    name: str   # Текст сущности (токены, склеенные пробелом)
    type: str   # Тип сущности
    start: int  # Стартовый символьный индекс (0-based) в поле text
    end: int    # Конечный символьный индекс (0-based, inclusive) в поле text


class Relation(BaseModel):
    """Отношение из золотой разметки датасета."""

    head_idx: int   # Индекс head-сущности в списке entities
    tail_idx: int   # Индекс tail-сущности в списке entities
    head_start: int  # Стартовый символьный индекс head-сущности
    head_end: int    # Конечный символьный индекс head-сущности
    tail_start: int  # Стартовый символьный индекс tail-сущности
    tail_end: int    # Конечный символьный индекс tail-сущности
    type: str        # Тип отношения


class DatasetRecord(BaseModel):
    """Одна запись датасета (текст + золотая разметка)."""

    id: str                   # Уникальный ID записи
    text: str                 # Полный текст записи
    entities: list[Entity]    # Gold-сущности
    relations: list[Relation]  # Gold-отношения


# ---------------------------------------------------------------------------
# 2. Unified Model Interface — предсказанные сущности/отношения от моделей
# ---------------------------------------------------------------------------

class PredictedEntity(BaseModel):
    """Сущность, предсказанная моделью NER."""

    name: str   # Название сущности (для матчинга с PredictedRelation.head/tail)
    type: str   # Тип сущности


class PredictedRelation(BaseModel):
    """Отношение, предсказанное моделью RE."""

    head: str    # name head-сущности (матчинг с PredictedEntity.name)
    tail: str    # name tail-сущности
    type: str    # Тип отношения


class BaseModelClient(ABC):
    """Абстрактный клиент модели NER/RE.

    Все клиенты (QwenClient, UniNerClient, GleanerClient, HybridClient)
    реализуют этот интерфейс.
    """

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

        NER-only модели (UniNer, Gleaner) НЕ реализуют.
        По умолчанию — NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает RE")


# ---------------------------------------------------------------------------
# 3. DatasetLoader ABC — загрузчики датасетов
# ---------------------------------------------------------------------------

class DatasetLoader(ABC):
    """Абстрактный загрузчик датасета.

    Конкретные реализации: Conll04Loader, SciERCLoader.
    """

    @property
    @abstractmethod
    def entity_types(self) -> list[str]:
        """Возвращает список типов сущностей для датасета."""
        ...

    @property
    @abstractmethod
    def relation_types(self) -> list[str]:
        """Возвращает список типов отношений для датасета."""
        ...

    @property
    @abstractmethod
    def relation_type_descriptions(self) -> str:
        """Возвращает форматированное описание типов отношений с указанием
        entity-типов head и tail (например, "Work_For: Person -> Organization")
        плюс описание None-отношения."""
        ...

    @property
    def allowed_relation_types(self) -> list[str]:
        """relation_types + ["None"] — полный список допустимых значений TYPE
        для подстановки в промпт. None используется в промпте, но игнорируется
        при подсчёте метрик."""
        return self.relation_types + ["None"]


    @abstractmethod
    def load(self, split: str) -> list[DatasetRecord]:
        """Загружает и нормализует указанный сплит датасета."""
        ...

    @abstractmethod
    def splits(self) -> list[str]:
        """Возвращает список доступных сплитов."""
        ...


# ---------------------------------------------------------------------------
# 4. ExperimentResult — результат одного эксперимента
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Результат одного эксперимента (E1–E6) на конкретном датасете/сплите."""

    experiment_id: str         # "E1"–"E6"
    model_ner: str             # "uniner"|"gliner"|"qwen"|"none"
    model_re: str              # "qwen"|"combined_qwen"|"none"
    dataset: str               # "conll04"|"scierc"
    split: str                 # "test"|"validation"|"train"
    task_type: str             # "ner_only"|"re_only"|"ner_re"
    config: dict               # Параметры запуска
    metrics: dict[str, float]  # precision_ner, recall_ner, f1_ner,
                               # precision_re, recall_re, f1_re,
                               # avg_response_time_sec, total_samples
    timestamp: str             # ISO 8601
