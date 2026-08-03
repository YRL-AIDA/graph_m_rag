"""Тесты для SciERCLoader — строго по спецификации scierc_loader.md (§Testing).

Тестируют: multi-sentence документы, RE-маппинг (совпадение/несовпадение),
динамический сбор entity_types, символьные индексы и загрузку всех 3 сплитов.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

# Путь для импорта testdata из entity_extraction_test
sys.path.insert(0, str(Path(__file__).parent.parent))

from testdata.scierc_loader import SciERCLoader

# ---------------------------------------------------------------------------
# Тестовые данные
# ---------------------------------------------------------------------------

# Документ для теста 1: 3 предложения
DOC_3SENT = {
    "doc_key": "test_doc_1",
    "sentences": [
        ["Token", "A1", "B1"],
        ["Token", "A2", "B2"],
        ["Token", "A3", "B3"],
    ],
    "ner": [
        [[0, 0, "Generic"]],   # sent 0: Token
        [[1, 2, "Material"]],  # sent 1: A2 B2
        [[0, 2, "Method"]],    # sent 2: Token A3 B3
    ],
    "relations": [
        [],
        [],
        [],
    ],
}

# Документ для теста 2: RE-маппинг — совпадение
# NER: (0,1)="Method" для "The algorithm", (3,4)="Generic" для "method X"
# RE: (0,1)-(3,4)="USED-FOR" — индексы совпадают с NER-спанами
DOC_RE_MATCH = {
    "doc_key": "test_doc_2",
    "sentences": [
        ["The", "algorithm", "uses", "method", "X"],
    ],
    "ner": [
        [[0, 1, "Method"], [3, 4, "Generic"]],
    ],
    "relations": [
        [[0, 1, 3, 4, "USED-FOR"]],
    ],
}

# Документ для теста 3: RE-маппинг — несовпадение
# NER: только (0,0)="Generic" (токен "E1")
# RE: (1,1)-(2,2)="FEATURE-OF" — head (1,1) не совпадает с NER (0,0)
DOC_RE_MISMATCH = {
    "doc_key": "test_doc_3",
    "sentences": [
        ["E1", "E2", "E3"],
    ],
    "ner": [
        [[0, 0, "Generic"]],
    ],
    "relations": [
        [[1, 1, 2, 2, "FEATURE-OF"]],
    ],
}

# Документ для теста 4: составной тип ORGANIZATION|PERSON
DOC_COMPOUND_TYPE = {
    "doc_key": "test_doc_4",
    "sentences": [
        ["John", "Smith", "works"],
    ],
    "ner": [
        [[0, 1, "ORGANIZATION|PERSON"]],
    ],
    "relations": [
        [],
    ],
}

# Документ для теста 5: multi-token сущность, символьные индексы
# Предложение: ["Token", "A2", "B2"]
# Токен 0: "Token" (len=5)
# Токен 1: "A2" (len=2)
# Токен 2: "B2" (len=2)
# Сущность [1,2] → name="A2 B2" (len=5)
# Формула: start_char = sum(len(tokens[:1])) + 1 = 5 + 1 = 6
#          end_char = start_char + len(name) - 1 = 6 + 5 - 1 = 10
DOC_CHAR_INDICES = {
    "doc_key": "test_doc_5",
    "sentences": [
        ["Token", "A2", "B2"],
    ],
    "ner": [
        [[1, 2, "Material"]],
    ],
    "relations": [
        [],
    ],
}

# Документ для теста 6: пустой документ (без предложений) — проверка error handling
DOC_EMPTY_SENTENCES = {
    "doc_key": "test_empty",
    "sentences": [],
    "ner": [],
    "relations": [],
}

# Все документы для train.json
TRAIN_DOCS = [
    DOC_3SENT,
    DOC_RE_MATCH,
    DOC_RE_MISMATCH,
    DOC_COMPOUND_TYPE,
    DOC_CHAR_INDICES,
    DOC_EMPTY_SENTENCES,
]

# Простой документ для dev.json и test.json
DUMMY_DOC = {
    "doc_key": "dummy",
    "sentences": [["Hello", "world"]],
    "ner": [[[0, 1, "Task"]]],
    "relations": [[]],
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(file_path: Path, docs: list[dict]) -> None:
    """Записать список документов в JSONL-файл (один JSON-объект на строку)."""
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


@pytest.fixture
def scierc_data_dir(tmp_path: Path) -> Path:
    """Создаёт временную структуру scierc/json/ с train.json, dev.json, test.json.

    train.json содержит все тестовые документы (5 + пустой документ).
    dev.json и test.json содержат по одному простому документу — необходимо
    для теста 4 (entity_types сканирует все 3 сплита) и теста 6 (splits/load).
    """
    json_dir = tmp_path / "json"
    json_dir.mkdir()

    _write_jsonl(json_dir / "train.json", TRAIN_DOCS)
    _write_jsonl(json_dir / "dev.json", [DUMMY_DOC])
    _write_jsonl(json_dir / "test.json", [DUMMY_DOC])

    return tmp_path


# ---------------------------------------------------------------------------
# Тест 1: Multi-sentence документ
# ---------------------------------------------------------------------------


def test_multi_sentence_document(scierc_data_dir: Path) -> None:
    """Документ из 3 предложений → 3 отдельных DatasetRecord.

    ID: scierc_test_doc_1_sent0, _sent1, _sent2.
    """
    loader = SciERCLoader(data_path=str(scierc_data_dir))
    records = loader.load("train")

    # Фильтруем только записи test_doc_1
    doc1_records = [r for r in records if r.id.startswith("scierc_test_doc_1")]

    assert len(doc1_records) == 3, (
        f"Ожидалось 3 записи, получено {len(doc1_records)}"
    )

    expected_ids = {
        "scierc_test_doc_1_sent0",
        "scierc_test_doc_1_sent1",
        "scierc_test_doc_1_sent2",
    }
    actual_ids = {r.id for r in doc1_records}
    assert actual_ids == expected_ids, f"ID записей не совпадают: {actual_ids}"

    # Проверяем тексты предложений
    assert doc1_records[0].text == "Token A1 B1"
    assert doc1_records[1].text == "Token A2 B2"
    assert doc1_records[2].text == "Token A3 B3"


# ---------------------------------------------------------------------------
# Тест 2: RE-маппинг — совпадение
# ---------------------------------------------------------------------------


def test_re_mapping_match(scierc_data_dir: Path) -> None:
    """Отношение с [h_start, h_end], совпадающими с NER-спаном.

    Проверяет: head_idx, head_start, head_end, tail_idx, tail_start, tail_end
    корректно подставлены из соответствующих Entity.
    """
    loader = SciERCLoader(data_path=str(scierc_data_dir))
    records = loader.load("train")

    doc2_records = [r for r in records if r.id.startswith("scierc_test_doc_2")]
    assert len(doc2_records) == 1
    record = doc2_records[0]

    assert len(record.entities) == 2, f"Ожидалось 2 сущности, получено {len(record.entities)}"

    # NER: (0,1)="Method" для "The algorithm", (3,4)="Generic" для "method X"
    head_entity = record.entities[0]
    tail_entity = record.entities[1]

    assert head_entity.name == "The algorithm"
    assert head_entity.type == "Method"
    assert tail_entity.name == "method X"
    assert tail_entity.type == "Generic"

    # RE: (0,1)-(3,4)="USED-FOR"
    assert len(record.relations) == 1, (
        f"Ожидалось 1 отношение, получено {len(record.relations)}"
    )
    rel = record.relations[0]

    assert rel.head_idx == 0, f"head_idx: ожидалось 0, получено {rel.head_idx}"
    assert rel.tail_idx == 1, f"tail_idx: ожидалось 1, получено {rel.tail_idx}"
    assert rel.head_start == head_entity.start
    assert rel.head_end == head_entity.end
    assert rel.tail_start == tail_entity.start
    assert rel.tail_end == tail_entity.end
    assert rel.type == "USED-FOR"


# ---------------------------------------------------------------------------
# Тест 3: RE-маппинг — несовпадение
# ---------------------------------------------------------------------------


def test_re_mapping_mismatch(scierc_data_dir: Path, caplog) -> None:
    """Отношение с head_start/head_end, НЕ совпадающими ни с одним NER-спаном.

    Отношение пропускается, в лог выводится WARNING.
    """
    loader = SciERCLoader(data_path=str(scierc_data_dir))

    # Включаем захват WARNING-сообщений
    with caplog.at_level(logging.WARNING):
        records = loader.load("train")

    doc3_records = [r for r in records if r.id.startswith("scierc_test_doc_3")]
    assert len(doc3_records) == 1
    record = doc3_records[0]

    assert len(record.entities) == 1, (
        f"Ожидалась 1 сущность, получено {len(record.entities)}"
    )

    # Отношения должны быть пропущены — RE-спан не совпадает с NER
    assert len(record.relations) == 0, (
        f"Ожидалось 0 отношений (RE-спан не совпадает), получено {len(record.relations)}"
    )

    # Проверяем наличие WARNING в логе
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 1, "Ожидалось WARNING-сообщение в логе"

    # Проверяем содержание WARNING — должно упоминать doc_key
    warning_messages = " ".join(r.message for r in warning_records)
    assert "test_doc_3" in warning_messages, (
        f"WARNING должен содержать doc_key 'test_doc_3', сообщение: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# Тест 4: Динамические типы
# ---------------------------------------------------------------------------


def test_entity_types_include_compound(scierc_data_dir: Path) -> None:
    """entity_types содержит все уникальные типы, включая составной ORGANIZATION|PERSON.

    Типы собираются динамически при первом обращении к property (лениво).
    """
    loader = SciERCLoader(data_path=str(scierc_data_dir))
    types = loader.entity_types

    # Проверяем, что ORGANIZATION|PERSON присутствует
    assert "ORGANIZATION|PERSON" in types, (
        f"entity_types должен содержать 'ORGANIZATION|PERSON', получено: {types}"
    )

    # Проверяем другие ожидаемые типы из тестовых данных
    assert "Generic" in types
    assert "Material" in types
    assert "Method" in types
    assert "Task" in types

    # Проверяем кэширование: повторный вызов возвращает тот же объект
    types_again = loader.entity_types
    assert types is types_again, "Повторный вызов entity_types должен вернуть кэшированный список"


# ---------------------------------------------------------------------------
# Тест 5: Символьные индексы
# ---------------------------------------------------------------------------


def test_character_indices_multi_token(scierc_data_dir: Path) -> None:
    """Multi-token сущность внутри предложения.

    Предложение: ["Token", "A2", "B2"]
    Сущность [1,2] → name="A2 B2", start_char=6, end_char=10.

    Вычисление:
      - Токен 0: "Token" (len=5), после него — пробел.
      - start_char = 5 (кумулятивная длина до токена 1) + 1 (start_local) = 6.
      - name = "A2 B2" (len=5).
      - end_char = 6 + 5 - 1 = 10.
    """
    loader = SciERCLoader(data_path=str(scierc_data_dir))
    records = loader.load("train")

    doc5_records = [r for r in records if r.id.startswith("scierc_test_doc_5")]
    assert len(doc5_records) == 1
    record = doc5_records[0]

    assert len(record.entities) == 1, (
        f"Ожидалась 1 сущность, получено {len(record.entities)}"
    )
    entity = record.entities[0]

    assert entity.name == "A2 B2", f"name: ожидалось 'A2 B2', получено '{entity.name}'"
    assert entity.type == "Material"
    assert entity.start == 6, (
        f"start_char: ожидалось 6, получено {entity.start}"
    )
    assert entity.end == 10, (
        f"end_char: ожидалось 10 (start=6 + len('A2 B2')=5 - 1), получено {entity.end}"
    )


# ---------------------------------------------------------------------------
# Тест 6: Все 3 сплита
# ---------------------------------------------------------------------------


def test_all_splits(scierc_data_dir: Path) -> None:
    """splits() возвращает ["train", "validation", "test"], каждый сплит загружается."""
    loader = SciERCLoader(data_path=str(scierc_data_dir))

    splits = loader.splits()
    assert splits == ["train", "validation", "test"], (
        f"Ожидались сплиты ['train', 'validation', 'test'], получено {splits}"
    )

    # Проверяем загрузку каждого сплита
    for split in splits:
        records = loader.load(split)
        assert isinstance(records, list), (
            f"load('{split}') должен вернуть list, получено {type(records)}"
        )
        assert len(records) > 0, (
            f"load('{split}') вернул пустой список"
        )

    # Проверяем, что train содержит ожидаемое количество записей
    train_records = loader.load("train")
    # 5 документов с предложениями (test_doc_1: 3, test_doc_2: 1, test_doc_3: 1,
    # test_doc_4: 1, test_doc_5: 1) + test_empty (пропущен) = 7 записей
    assert len(train_records) == 7, (
        f"Ожидалось 7 записей в train, получено {len(train_records)}"
    )


# ---------------------------------------------------------------------------
# Error handling: неизвестный сплит
# ---------------------------------------------------------------------------


def test_load_unknown_split_raises_valueerror(scierc_data_dir: Path) -> None:
    """load() с неизвестным сплитом → ValueError."""
    loader = SciERCLoader(data_path=str(scierc_data_dir))

    with pytest.raises(ValueError, match="Неизвестный сплит"):
        loader.load("unknown_split")


# ---------------------------------------------------------------------------
# Error handling: файл не найден
# ---------------------------------------------------------------------------


def test_load_file_not_found_raises(scierc_data_dir: Path) -> None:
    """load() при отсутствующем JSON-файле → FileNotFoundError."""
    loader = SciERCLoader(data_path=str(scierc_data_dir))

    # Удаляем файл train.json
    (scierc_data_dir / "json" / "train.json").unlink()

    with pytest.raises(FileNotFoundError, match="Файл не найден"):
        loader.load("train")


# ---------------------------------------------------------------------------
# Error handling: документ без предложений
# ---------------------------------------------------------------------------


def test_empty_sentences_document_skipped(scierc_data_dir: Path) -> None:
    """Документ без предложений (sentences=[]) пропускается, не вызывая ошибок."""
    loader = SciERCLoader(data_path=str(scierc_data_dir))
    records = loader.load("train")

    # Записи с doc_key "test_empty" не должны присутствовать
    empty_ids = [r.id for r in records if "test_empty" in r.id]
    assert len(empty_ids) == 0, (
        f"Документ без предложений должен быть пропущен, найдены записи: {empty_ids}"
    )
