"""Загрузчик датасета CoNLL-04 для NER/RE экспериментов.

Использует HuggingFace datasets (DFKI-SLT/conll04).
Сплиты: train (922), validation (231), test (288).

Типы сущностей: Peop, Loc, Org, Other
Типы отношений: Located_In, Work_For, OrgBased_In, Live_In, Kill
"""

from __future__ import annotations

import logging
import warnings

import os as _os
import sys as _sys

# Workaround: локальный пакет testdata конфликтует с PyPI datasets.
# Убираем entity_extraction_test из sys.path и локальный модуль из sys.modules
# на время импорта, чтобы загрузить HF-версию.
_entity_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_saved_path_entries = [p for p in _sys.path if _os.path.abspath(p if p else '') == _entity_dir]
_new_path = [p for p in _sys.path if _os.path.abspath(p if p else '') != _entity_dir]
_sys.path[:] = _new_path
_cached_datasets = _sys.modules.pop('testdata', None)
try:
    from datasets import load_dataset  # type: ignore[import-untyped]
finally:
    _sys.path[:] = _new_path + _saved_path_entries
    if _cached_datasets is not None:
        _sys.modules['testdata'] = _cached_datasets

from .base_loader import DatasetLoader, DatasetRecord, Entity, Relation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы датасета
# ---------------------------------------------------------------------------

_CONLL04_ENTITY_TYPES: list[str] = ["Peop", "Loc", "Org", "Other"]
_CONLL04_RELATION_TYPES: list[str] = [
    "Located_In",
    "Work_For",
    "OrgBased_In",
    "Live_In",
    "Kill",
]
_CONLL04_SPLITS: list[str] = ["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Conll04Loader
# ---------------------------------------------------------------------------


class Conll04Loader(DatasetLoader):
    """Загрузчик датасета CoNLL-04.

    Нормализует записи HuggingFace-датасета DFKI-SLT/conll04 в единый формат
    DatasetRecord: текст + gold-сущности (Entity) + gold-отношения (Relation).
    """

    # ------------------------------------------------------------------ #
    #  Properties (read-only типы сущностей и отношений)
    # ------------------------------------------------------------------ #

    @property
    def entity_types(self) -> list[str]:
        """Типы сущностей CoNLL-04: Peop, Loc, Org, Other."""
        return _CONLL04_ENTITY_TYPES

    @property
    def relation_types(self) -> list[str]:
        """Типы отношений CoNLL-04: Located_In, Work_For, OrgBased_In, Live_In, Kill."""
        return _CONLL04_RELATION_TYPES

    # ------------------------------------------------------------------ #
    #  Splits
    # ------------------------------------------------------------------ #

    def splits(self) -> list[str]:
        """Доступные сплиты: train, validation, test."""
        return _CONLL04_SPLITS

    # ------------------------------------------------------------------ #
    #  Normalize
    # ------------------------------------------------------------------ #

    def _normalize_record(self, record: dict, split: str, idx: int) -> DatasetRecord:
        """Нормализует одну запись CoNLL04 из токен-формата в символьные спаны.

        Args:
            record: Словарь в формате HuggingFace CoNLL04: {"tokens": [...],
                    "entities": [{"type":..., "start":..., "end":...}, ...],
                    "relations": [{"type":..., "head":..., "tail":...}, ...]}.
            split: Имя сплита (train, validation, test).
            idx: Индекс записи в сплите.

        Returns:
            Нормализованная запись DatasetRecord.

        Raises:
            ValueError: Если у сущности start > end.
        """
        tokens: list[str] = record["tokens"]
        hf_entities: list[dict] = record["entities"]
        hf_relations: list[dict] = record["relations"]

        # 1. Склейка текста
        text = " ".join(tokens)

        # 2. Предвычисление кумулятивных стартов для токенов,
        #    чтобы по формуле start_char = sum(len(tokens[i]) for i<start) + start
        #    вычислять за O(1) на каждую сущность
        token_starts: list[int] = []
        pos = 0
        for t in tokens:
            token_starts.append(pos)
            pos += len(t) + 1  # +1 за пробел между токенами

        # 3. Нормализация сущностей
        entities: list[Entity] = []
        for ent in hf_entities:
            ent_type: str = ent["type"]
            start: int = ent["start"]
            end: int = ent["end"]

            # Проверка на битые данные: start > end
            if start > end:
                raise ValueError(
                    f"Entity with start > end: start={start}, end={end}, "
                    f"type={ent_type} in record {idx}"
                )

            # Имя сущности — склейка токенов через пробел
            name = " ".join(tokens[start : end + 1])

            # Символьные границы по формуле спецификации:
            #   start_char = sum(len(tokens[i]) for i in range(start)) + start
            #   end_char   = start_char + len(name) - 1
            start_char = token_starts[start]
            end_char = start_char + len(name) - 1

            entities.append(
                Entity(name=name, type=ent_type, start=start_char, end=end_char)
            )

        # 4. Нормализация отношений
        relations: list[Relation] = []
        for rel in hf_relations:
            rel_type: str = rel["type"]
            head_idx: int = rel["head"]
            tail_idx: int = rel["tail"]

            head_entity = entities[head_idx]
            tail_entity = entities[tail_idx]

            relations.append(
                Relation(
                    head_idx=head_idx,
                    tail_idx=tail_idx,
                    head_start=head_entity.start,
                    head_end=head_entity.end,
                    tail_start=tail_entity.start,
                    tail_end=tail_entity.end,
                    type=rel_type,
                )
            )

        # 5. Формирование ID записи
        record_id = f"conll04_{split}_{idx}"

        return DatasetRecord(
            id=record_id,
            text=text,
            entities=entities,
            relations=relations,
        )

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #

    def load(self, split: str) -> list[DatasetRecord]:
        """Загружает и нормализует указанный сплит датасета.

        Args:
            split: Имя сплита – train, validation или test.

        Returns:
            Список нормализованных записей DatasetRecord.

        Raises:
            ValueError: Если передан неизвестный сплит.
            ConnectionError: Если HF-датасет недоступен (пробрасывается из datasets).
        """
        # --- Валидация сплита ---
        if split not in _CONLL04_SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Available: {self.splits()}"
            )

        # --- Загрузка из HuggingFace ---
        # Исключения (ConnectionError, FileNotFoundError) пробрасываются наверх.
        hf_dataset = load_dataset("DFKI-SLT/conll04", split=split)

        # --- Проверка на пустой датасет ---
        if len(hf_dataset) == 0:  # type: ignore[arg-type]
            warnings.warn(
                f"CoNLL-04 split '{split}' is empty after loading.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        # --- Нормализация записей ---
        records: list[DatasetRecord] = []
        for idx, hf_record in enumerate(hf_dataset):  # type: ignore[arg-type]
            records.append(self._normalize_record(hf_record, split, idx))  # type: ignore[arg-type]

        return records
