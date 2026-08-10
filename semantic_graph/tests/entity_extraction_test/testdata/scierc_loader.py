"""Загрузчик датасета SciERC.

Нормализует сырые JSON-файлы SciERC (формат JSONL — один JSON-объект на строку)
в список DatasetRecord с вычислением символьных индексов сущностей.

NER/RE-спаны используют глобальные (cross-sentence) токеновые индексы документа;
при нормализации они преобразуются в локальные индексы предложения и затем
в символьные позиции текста предложения по формуле CoNLL04.

Спецификация: SciERC Loader — entity_extraction_test/testdata/scierc_loader.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base_loader import DatasetLoader, Entity, Relation, DatasetRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Фиксированный список типов отношений SciERC (7 типов)
SCIERC_RELATION_TYPES: list[str] = [
    "USED-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "CONJUNCTION",
    "COMPARE",
    "EVALUATE-FOR",
    "PART-OF",
]

# Форматированное описание типов отношений SciERC + None
SCIERC_RELATION_DESCRIPTIONS: str = (
    "USED-FOR: Method/Material -> Task\n"
    "FEATURE-OF: OtherScientificTerm/Metric -> Method/Task (Свойство или метрика описывает метод или задачу)\n"
    "HYPONYM-OF: Method/Task -> Method/Task (Первая сущность является подтипом или примером второй)\n"
    "CONJUNCTION: Any -> Any (Сущности просто перечисляются через 'и'/'или' без строгой семантической связи)\n"
    "COMPARE: Method/Task -> Method/Task (Сущности сравниваются в тексте)\n"
    "EVALUATE-FOR: Metric/Method -> Task/Material (Метрика используется для оценки задачи ИЛИ метод оценивается на датасете)\n"
    "PART-OF: Method/Task -> Method/Task (Первая сущность является компонентом второй)\n"
    "None: Если между парой сущностей нет ни одного из вышеперечисленных отношений"
)


# Маппинг имени сплита → имя JSON-файла
SPLIT_FILE_MAP: dict[str, str] = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}


# ---------------------------------------------------------------------------
# SciERCLoader
# ---------------------------------------------------------------------------

class SciERCLoader(DatasetLoader):
    """Загрузчик датасета SciERC.

    Читает JSONL-файлы из data_path/json/ (train.json, dev.json, test.json),
    нормализует глобальные токеновые NER/RE аннотации в символьные индексы
    Entity/Relation для каждого предложения.

    Типы сущностей собираются динамически при первом обращении к entity_types
    (ленивая инициализация — сканируются все три JSON-файла).
    """

    def __init__(
        self,
        data_path: str = "/home/ivan/work/ooo/graph_m_rag/datasets_/scierc",
    ) -> None:
        self._data_path = Path(data_path)
        self._json_dir = self._data_path / "json"
        self._entity_types: list[str] | None = None  # Ленивая инициализация
        self._relation_types: list[str] = list(SCIERC_RELATION_TYPES)

    # ------------------------------------------------------------------
    # Properties (ABC DatasetLoader)
    # ------------------------------------------------------------------

    @property
    def entity_types(self) -> list[str]:
        """Типы сущностей SciERC — собираются динамически при первом обращении.

        Сканирует все три сплита (train/validation/test), извлекает уникальные
        значения type из NER-аннотаций и кэширует результат в self._entity_types.
        """
        if self._entity_types is None:
            self._entity_types = self._collect_entity_types()
        return self._entity_types

    @property
    def relation_types(self) -> list[str]:
        """Фиксированный список типов отношений SciERC (7 типов)."""
        return self._relation_types

    @property
    def relation_type_descriptions(self) -> str:
        """Форматированное описание типов отношений SciERC + None."""
        return SCIERC_RELATION_DESCRIPTIONS

    # ------------------------------------------------------------------
    # Public API (ABC DatasetLoader)
    # ------------------------------------------------------------------

    def splits(self) -> list[str]:
        """Возвращает список доступных сплитов."""
        return ["train", "validation", "test"]

    def load(self, split: str) -> list[DatasetRecord]:
        """Загружает указанный сплит и нормализует в список DatasetRecord.

        Args:
            split: Один из "train", "validation", "test".

        Returns:
            Список DatasetRecord — по одной записи на предложение.

        Raises:
            ValueError: Если передан неизвестный сплит.
            FileNotFoundError: Если JSON-файл сплита не найден.
            json.JSONDecodeError: Если JSON не парсится (malformed).
        """
        if split not in SPLIT_FILE_MAP:
            raise ValueError(
                f"Неизвестный сплит '{split}'. Доступные: {self.splits()}"
            )

        file_path = self._json_dir / SPLIT_FILE_MAP[split]
        data = self._read_jsonl(file_path)

        records: list[DatasetRecord] = []
        for doc in data:
            records.extend(self._normalize_document(doc))

        return records

    # ------------------------------------------------------------------
    # Private: сбор entity_types (ленивая инициализация)
    # ------------------------------------------------------------------

    def _collect_entity_types(self) -> list[str]:
        """Сканирует все три JSON-файла, собирает уникальные типы NER.

        Выполняется однократно при первом обращении к entity_types.
        """
        types: set[str] = set()
        for split in self.splits():
            file_path = self._json_dir / SPLIT_FILE_MAP[split]
            data = self._read_jsonl(file_path)
            for doc in data:
                for ner_spans in doc.get("ner", []):
                    for span in ner_spans:
                        # span = [start, end, type]
                        types.add(span[2])
        return sorted(types)

    # ------------------------------------------------------------------
    # Private: чтение JSONL
    # ------------------------------------------------------------------

    @staticmethod
    def _read_jsonl(file_path: Path) -> list[dict]:
        """Читает JSONL-файл. Каждая строка — один полный JSON-объект.

        Raises:
            FileNotFoundError: Файл не существует.
            json.JSONDecodeError: Некорректный JSON на одной из строк.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        data: list[dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Ошибка парсинга JSON в файле {file_path.name}, "
                        f"строка {line_num}: {e.msg}",
                        e.doc,
                        e.pos,
                    ) from e
        return data

    # ------------------------------------------------------------------
    # Private: нормализация одного документа
    # ------------------------------------------------------------------

    def _normalize_document(self, doc: dict) -> list[DatasetRecord]:
        """Нормализует один документ SciERC в список DatasetRecord.

        NER/RE-спаны используют глобальные (cross-sentence) токеновые индексы.
        Для каждого предложения вычисляется offset — суммарное количество токенов
        предыдущих предложений — и глобальные индексы преобразуются в локальные.

        Для каждого предложения i:
          1. text = " ".join(sentences[i])
          2. NER → Entity: глобальные индексы → локальные → start_char/end_char
          3. RE → Relation: матчинг по глобальным (start, end), индексы Entity
          4. id = f"scierc_{doc_key}_sent{i}"

        Документы без предложений (sentences: []) пропускаются (0 записей).
        """
        doc_key: str = doc["doc_key"]
        sentences: list[list[str]] = doc["sentences"]

        if not sentences:
            # Документ без предложений — пропускаем
            return []

        ner_annotations: list[list[list]] = doc["ner"]
        relations_annotations: list[list[list]] = doc["relations"]

        records: list[DatasetRecord] = []
        offset: int = 0  # Глобальный индекс первого токена текущего предложения

        for sent_idx, sent_tokens in enumerate(sentences):
            if not sent_tokens:
                offset += 0  # Пустое предложение не добавляет токенов
                continue

            # 1. Текст: токены склеиваются одиночным пробелом
            text = " ".join(sent_tokens)

            # 2. NER → Entity
            #    Глобальные индексы преобразуются в локальные вычитанием offset
            ner_spans = ner_annotations[sent_idx]
            entities = self._normalize_entities(
                sent_tokens, ner_spans, offset
            )

            # 3. RE → Relation
            #    Матчинг по глобальным (start, end) в ner_spans
            rel_spans = relations_annotations[sent_idx]
            relations = self._normalize_relations(
                doc_key, sent_idx, ner_spans, entities, rel_spans
            )

            # 4. Формирование записи
            record_id = f"scierc_{doc_key}_sent{sent_idx}"
            records.append(
                DatasetRecord(
                    id=record_id,
                    text=text,
                    entities=entities,
                    relations=relations,
                )
            )

            # Обновление offset для следующего предложения
            offset += len(sent_tokens)

        return records

    # ------------------------------------------------------------------
    # Private: NER → Entity
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_entities(
        tokens: list[str],
        ner_spans: list[list],
        offset: int,
    ) -> list[Entity]:
        """Преобразует NER-аннотации одного предложения в список Entity.

        Глобальные токеновые индексы (start, end) из ner_spans преобразуются
        в локальные индексы предложения вычитанием offset, затем вычисляются
        символьные индексы по формуле CoNLL04:

          start_char = sum(len(tokens[j]) for j in range(start_local)) + start_local
          end_char   = start_char + len(name) - 1

        Args:
            tokens: Токены текущего предложения.
            ner_spans: Список NER-аннотаций [[start_global, end_global, type], ...].
            offset: Суммарное количество токенов в предыдущих предложениях.
        """
        entities: list[Entity] = []
        for span in ner_spans:
            start_global, end_global, ner_type = span[0], span[1], span[2]
            # Преобразование глобальных индексов в локальные
            start_local = start_global - offset
            end_local = end_global - offset
            name = " ".join(tokens[start_local:end_local + 1])
            # Символьные индексы: кумулятивные длины токенов + пробелы между ними
            start_char = sum(len(tokens[j]) for j in range(start_local)) + start_local
            end_char = start_char + len(name) - 1
            entities.append(
                Entity(name=name, type=ner_type, start=start_char, end=end_char)
            )
        return entities

    # ------------------------------------------------------------------
    # Private: RE → Relation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_relations(
        doc_key: str,
        sent_idx: int,
        ner_spans: list[list],
        entities: list[Entity],
        rel_spans: list[list],
    ) -> list[Relation]:
        """Преобразует RE-аннотации одного предложения в список Relation.

        Для каждого отношения [h_start, h_end, t_start, t_end, type]
        (глобальные индексы):
          - Ищет NER-спан в ner_spans с совпадающими глобальными (start, end).
          - Если не найден → WARNING в лог, отношение пропускается.
          - При совпадении → подставляются start_char/end_char из Entity.
          - Если несколько одинаковых спанов — берётся первое совпадение.
        """
        # Маппинг (start_global, end_global) → индекс в entities
        ner_span_to_idx: dict[tuple[int, int], int] = {}
        for idx, span in enumerate(ner_spans):
            key = (span[0], span[1])
            if key not in ner_span_to_idx:
                ner_span_to_idx[key] = idx

        relations: list[Relation] = []
        for rel_span in rel_spans:
            h_start, h_end, t_start, t_end, rel_type = (
                rel_span[0], rel_span[1], rel_span[2], rel_span[3], rel_span[4]
            )
            head_key = (h_start, h_end)
            tail_key = (t_start, t_end)

            if head_key not in ner_span_to_idx or tail_key not in ner_span_to_idx:
                logger.warning(
                    "RE-спан не соответствует NER-спану: doc_key=%s, sent_idx=%d, "
                    "rel=(h_span=(%d,%d), t_span=(%d,%d), type=%s), "
                    "ner_spans=%s",
                    doc_key,
                    sent_idx,
                    h_start,
                    h_end,
                    t_start,
                    t_end,
                    rel_type,
                    [(s[0], s[1]) for s in ner_spans],
                )
                continue

            head_idx = ner_span_to_idx[head_key]
            tail_idx = ner_span_to_idx[tail_key]

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

        return relations
