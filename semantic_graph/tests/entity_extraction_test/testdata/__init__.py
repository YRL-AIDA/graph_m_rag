"""Пакет testdata — загрузчики датасетов и модели данных для NER/RE экспериментов.

Экспортирует все публичные символы из base_loader.py.
"""

from .base_loader import (
    BaseModelClient,
    DatasetLoader,
    DatasetRecord,
    Entity,
    ExperimentResult,
    PredictedEntity,
    PredictedRelation,
    Relation,
)

__all__ = [
    "BaseModelClient",
    "DatasetLoader",
    "DatasetRecord",
    "Entity",
    "ExperimentResult",
    "PredictedEntity",
    "PredictedRelation",
    "Relation",
]
