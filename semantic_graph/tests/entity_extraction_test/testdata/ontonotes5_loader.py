"""Загрузчик датасета OntoNotes 5.0 (NER-only) для NER/RE экспериментов.

Загружает JSONL-файлы напрямую с HuggingFace (tner/ontonotes5, подмножество ontonotes5).
Сплиты: train (59,924), validation (8,528), test (8,262).

Типы сущностей (18): CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC,
                     MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT,
                     QUANTITY, TIME, WORK_OF_ART
Типы отношений: отсутствуют (NER-only датасет).
"""

from __future__ import annotations

import logging
import warnings
from urllib.request import urlopen

from .base_loader import DatasetLoader, DatasetRecord, Entity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы датасета
# ---------------------------------------------------------------------------

_ONTONOTES5_ENTITY_TYPES: list[str] = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]

#: Отношения отсутствуют (NER-only датасет).
_ONTONOTES5_RELATION_TYPES: list[str] = []

_ONTONOTES5_RELATION_DESCRIPTIONS: str = (
    "None: Этот датасет не содержит отношений."
)

_ONTONOTES5_SPLITS: list[str] = ["train", "validation", "test"]

# ---------------------------------------------------------------------------
# Label Mapping (BIO → numeric ID), выверено по label.json из HF-репозитория
# ---------------------------------------------------------------------------

ONTONOTES5_LABEL_TO_ID: dict[str, int] = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36,
}

ID_TO_LABEL: dict[int, str] = {v: k for k, v in ONTONOTES5_LABEL_TO_ID.items()}

# ---------------------------------------------------------------------------
# URL-маппинг сплит → список JSONL-файлов на HF
# ---------------------------------------------------------------------------

_BASE_URL = "https://huggingface.co/datasets/tner/ontonotes5/resolve/main/dataset"

_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train00.json", "train01.json", "train02.json", "train03.json"],
    "validation": ["valid.json"],
    "test": ["test.json"],
}


# ---------------------------------------------------------------------------
# Ontonotes5Loader
# ---------------------------------------------------------------------------


class Ontonotes5Loader(DatasetLoader):
    """Загрузчик датасета OntoNotes 5.0 (NER-only).

    Нормализует записи из JSONL-файлов HuggingFace tner/ontonotes5 в единый
    формат DatasetRecord: текст + gold-сущности (Entity). Отношения отсутствуют.
    """

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def entity_types(self) -> list[str]:
        """Типы сущностей OntoNotes 5.0 (18 типов)."""
        return _ONTONOTES5_ENTITY_TYPES

    @property
    def relation_types(self) -> list[str]:
        """Типы отношений: пустой список (NER-only датасет)."""
        return _ONTONOTES5_RELATION_TYPES

    @property
    def relation_type_descriptions(self) -> str:
        """Описание типов отношений: только None (NER-only датасет)."""
        return _ONTONOTES5_RELATION_DESCRIPTIONS

    # ------------------------------------------------------------------ #
    #  Splits
    # ------------------------------------------------------------------ #

    def splits(self) -> list[str]:
        """Доступные сплиты: train, validation, test."""
        return _ONTONOTES5_SPLITS

    # ------------------------------------------------------------------ #
    #  Helpers: загрузка JSONL
    # ------------------------------------------------------------------ #

    @staticmethod
    def _download_jsonl(url: str) -> list[dict]:
        """Загружает один JSONL-файл (по одному JSON-объекту на строку)
        и возвращает список словарей.

        Args:
            url: Полный URL файла.

        Returns:
            Список словарей из JSONL-файла.

        Raises:
            OSError: При сетевых ошибках.
            ValueError: При ошибках парсинга JSON.
        """
        import json as _json

        records: list[dict] = []
        with urlopen(url) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line:
                    records.append(_json.loads(line))
        return records

    @staticmethod
    def _load_split_jsonl(split: str) -> list[dict]:
        """Загружает все JSONL-файлы для указанного сплита.

        Args:
            split: Имя сплита (train, validation, test).

        Returns:
            Объединённый список словарей из всех файлов сплита.
        """
        records: list[dict] = []
        for filename in _SPLIT_FILES[split]:
            url = f"{_BASE_URL}/{filename}"
            records.extend(Ontonotes5Loader._download_jsonl(url))
        return records

    # ------------------------------------------------------------------ #
    #  BIO → Entity span conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bio_tags_to_entities(
        tags: list[int],
        tokens: list[str],
        token_starts: list[int],
    ) -> list[Entity]:
        """Преобразует BIO-тэги в список Entity с символьными спанами.

        Args:
            tags: Числовые BIO-тэги (по ID_TO_LABEL).
            tokens: Список токенов.
            token_starts: Кумулятивные стартовые позиции токенов в склеенном тексте.

        Returns:
            Список Entity с вычисленными символьными границами.
        """
        entities: list[Entity] = []
        n = len(tags)
        i = 0

        while i < n:
            label = ID_TO_LABEL[tags[i]]

            if label.startswith("B-"):
                ent_type = label[2:]  # часть после "B-"
                start_idx = i
                end_idx = i + 1  # exclusive, Python-slice convention

                # Поглощаем все последующие I-тэги того же типа
                while end_idx < n:
                    next_label = ID_TO_LABEL.get(tags[end_idx], "O")
                    if next_label == f"I-{ent_type}":
                        end_idx += 1
                    else:
                        break

                # Имя сущности — склейка токенов через пробел
                name = " ".join(tokens[start_idx:end_idx])

                # Символьные границы
                start_char = token_starts[start_idx]
                end_char = start_char + len(name) - 1  # inclusive end

                entities.append(
                    Entity(name=name, type=ent_type, start=start_char, end=end_char)
                )

                i = end_idx
            else:
                i += 1

        return entities

    # ------------------------------------------------------------------ #
    #  Normalize
    # ------------------------------------------------------------------ #

    def _normalize_record(self, record: dict, split: str, idx: int) -> DatasetRecord:
        """Нормализует одну запись OntoNotes 5.0 из токен-формата в символьные спаны.

        Args:
            record: Словарь {"tokens": [...], "tags": [...]} из JSONL-файла.
            split: Имя сплита (train, validation, test).
            idx: Индекс записи в сплите.

        Returns:
            Нормализованная запись DatasetRecord.
        """
        tokens: list[str] = record["tokens"]
        tags: list[int] = record["tags"]

        # 1. Склейка текста (единый пробел между токенами)
        text = " ".join(tokens)

        # 2. Предвычисление кумулятивных стартов для токенов
        #    token_starts[i] = sum(len(tokens[j]) + 1 for j in range(i))
        token_starts: list[int] = []
        pos = 0
        for t in tokens:
            token_starts.append(pos)
            pos += len(t) + 1  # +1 за пробел

        # 3. Преобразование BIO-тэгов в сущности
        entities = self._bio_tags_to_entities(tags, tokens, token_starts)

        # 4. Отношения — всегда пустой список (NER-only)
        record_id = f"ontonotes5_{split}_{idx}"

        return DatasetRecord(
            id=record_id,
            text=text,
            entities=entities,
            relations=[],
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
            OSError: Если HF-файлы недоступны (сетевые ошибки).
        """
        # --- Валидация сплита ---
        if split not in _ONTONOTES5_SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Available: {self.splits()}"
            )

        # --- Загрузка JSONL с HuggingFace ---
        logger.info("Loading OntoNotes 5.0 split '%s' from HF…", split)
        raw_records = self._load_split_jsonl(split)

        if not raw_records:
            warnings.warn(
                f"OntoNotes 5.0 split '{split}' is empty after loading.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        # --- Нормализация записей ---
        records: list[DatasetRecord] = []
        for idx, raw_record in enumerate(raw_records):
            records.append(self._normalize_record(raw_record, split, idx))

        return records
