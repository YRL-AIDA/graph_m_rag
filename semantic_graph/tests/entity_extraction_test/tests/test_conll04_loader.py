"""Тесты для Conll04Loader — загрузчика датасета CoNLL-04.

Покрытие согласно спецификации conll04_loader.md §Testing:
  TC1 – нормализация single-token сущности
  TC2 – нормализация multi-token сущности (New York)
  TC3 – нормализация multi-token сущности в середине списка (B C)
  TC4 – head_idx/tail_idx в Relation соответствуют entity-списку
  TC5 – head_start/head_end/tail_start/tail_end копируются из Entity
  TC6 – свойства entity_types и relation_types
  TC7 – размеры сплитов (интеграционный, @pytest.mark.integration)
  TC8 – формат ID записи (conll04_{split}_{idx})

Плюс error-handling тесты:
  - неизвестный split → ValueError
  - сущность с start > end → ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Добавляем entity_extraction_test/ в sys.path, чтобы импортировать testdata.conll04_loader
# Path(__file__).parent          = .../entity_extraction_test/tests/
# Path(__file__).parent.parent   = .../entity_extraction_test/
sys.path.insert(0, str(Path(__file__).parent.parent))

from testdata.conll04_loader import Conll04Loader


# ─────────────────────────────────────────────────────────────────────────────
# Фикстуры
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def loader() -> Conll04Loader:
    """Создаёт экземпляр Conll04Loader без вызова HF."""
    return Conll04Loader()


# ════════════════════════════════════════════════════════════════════════════
# TC1: single-token entity
#   tokens=["John", "lives", "in", "NYC"], entity {Peop, 0, 0}
#   → name="John", start_char=0, end_char=3
# ════════════════════════════════════════════════════════════════════════════


def test_normalize_single_token_entity(loader: Conll04Loader) -> None:
    """TC1: одиночный токен — символьные границы совпадают с токеном."""
    record = {
        "tokens": ["John", "lives", "in", "NYC"],
        "entities": [{"type": "Peop", "start": 0, "end": 0}],
        "relations": [],
    }
    result = loader._normalize_record(record, "test", 0)

    assert result.id == "conll04_test_0"
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.name == "John"
    assert entity.type == "Peop"
    assert entity.start == 0
    assert entity.end == 3  # len("John") - 1 = 3


# ════════════════════════════════════════════════════════════════════════════
# TC2: multi-token entity
#   tokens=["New", "York"], entity {Loc, 0, 1}
#   → name="New York", start_char=0, end_char=7
# ════════════════════════════════════════════════════════════════════════════


def test_normalize_multi_token_entity(loader: Conll04Loader) -> None:
    """TC2: несколько токенов — склейка через пробел, границы с учётом пробела."""
    record = {
        "tokens": ["New", "York"],
        "entities": [{"type": "Loc", "start": 0, "end": 1}],
        "relations": [],
    }
    result = loader._normalize_record(record, "test", 1)

    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.name == "New York"
    assert entity.type == "Loc"
    assert entity.start == 0
    # "New York": start=0, len("New York")=8, end=0+8-1=7
    assert entity.end == 7


# ════════════════════════════════════════════════════════════════════════════
# TC3: multi-token entity в середине
#   tokens=["A", "B", "C"], entity {Org, 1, 2}
#   → name="B C", start_char=2, end_char=4
# ════════════════════════════════════════════════════════════════════════════


def test_normalize_mid_multi_token_entity(loader: Conll04Loader) -> None:
    """TC3: сущность не с начала — кумулятивные длины токенов + пробелы."""
    record = {
        "tokens": ["A", "B", "C"],
        "entities": [{"type": "Org", "start": 1, "end": 2}],
        "relations": [],
    }
    result = loader._normalize_record(record, "test", 2)

    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.name == "B C"
    assert entity.type == "Org"
    # token_starts: ["A"→0, "B"→2 (0+1+1), "C"→4 (2+1+1)]
    # start_char = token_starts[1] = 2
    # end_char = 2 + len("B C") - 1 = 2 + 3 - 1 = 4
    assert entity.start == 2
    assert entity.end == 4


# ════════════════════════════════════════════════════════════════════════════
# TC4: отношения — head_idx/tail_idx матчатся с entity-списком
# ════════════════════════════════════════════════════════════════════════════


def test_relation_indices_match_entity_list(loader: Conll04Loader) -> None:
    """TC4: индексы head_idx/tail_idx — прямые индексы в списке entities."""
    record = {
        "tokens": ["John", "works", "at", "Microsoft", "in", "Seattle"],
        "entities": [
            {"type": "Peop", "start": 0, "end": 0},
            {"type": "Org", "start": 3, "end": 3},
            {"type": "Loc", "start": 5, "end": 5},
        ],
        "relations": [
            {"type": "Work_For", "head": 0, "tail": 1},
            {"type": "Located_In", "head": 1, "tail": 2},
        ],
    }
    result = loader._normalize_record(record, "test", 0)

    assert len(result.relations) == 2

    # Первое отношение: John → Microsoft
    rel0 = result.relations[0]
    assert rel0.type == "Work_For"
    assert rel0.head_idx == 0
    assert rel0.tail_idx == 1
    # head_idx=0 → entities[0] = "John", tail_idx=1 → entities[1] = "Microsoft"
    assert result.entities[rel0.head_idx].name == "John"
    assert result.entities[rel0.tail_idx].name == "Microsoft"

    # Второе отношение: Microsoft → Seattle
    rel1 = result.relations[1]
    assert rel1.type == "Located_In"
    assert rel1.head_idx == 1
    assert rel1.tail_idx == 2
    assert result.entities[rel1.head_idx].name == "Microsoft"
    assert result.entities[rel1.tail_idx].name == "Seattle"


# ════════════════════════════════════════════════════════════════════════════
# TC5: head_start/head_end/tail_start/tail_end в Relation
#   — копии полей start/end Entity по head_idx/tail_idx
# ════════════════════════════════════════════════════════════════════════════


def test_relation_copies_entity_boundaries(loader: Conll04Loader) -> None:
    """TC5: поля head_start/head_end/tail_start/tail_end копируются из Entity."""
    record = {
        "tokens": ["John", "works", "at", "Microsoft"],
        "entities": [
            {"type": "Peop", "start": 0, "end": 0},   # John: start=0, end=3
            {"type": "Org", "start": 3, "end": 3},     # Microsoft: start=15, end=23
        ],
        "relations": [
            {"type": "Work_For", "head": 0, "tail": 1},
        ],
    }
    result = loader._normalize_record(record, "test", 0)

    head_entity = result.entities[0]  # John
    tail_entity = result.entities[1]  # Microsoft

    rel = result.relations[0]
    assert rel.head_start == head_entity.start
    assert rel.head_end == head_entity.end
    assert rel.tail_start == tail_entity.start
    assert rel.tail_end == tail_entity.end

    # Проверяем конкретные значения для наглядности
    # "John" → start=0, end=3
    # "John works at Microsoft": token_starts=[0,5,11,14]
    #   Microsoft: start_char=14, end_char=14+9-1=22
    # Но фактически после склейки "John works at Microsoft"
    # token_starts: "John"=0, "works"=5 (0+4+1), "at"=11 (5+5+1), "Microsoft"=14 (11+2+1)
    # Microsoft start=14, end=14+9-1=22
    assert head_entity.start == 0
    assert head_entity.end == 3
    assert tail_entity.start == 14
    assert tail_entity.end == 22


# ════════════════════════════════════════════════════════════════════════════
# TC6: свойства entity_types и relation_types
# ════════════════════════════════════════════════════════════════════════════


def test_entity_types(loader: Conll04Loader) -> None:
    """TC6 (часть 1): entity_types возвращает корректный список типов."""
    types = loader.entity_types
    assert types == ["Peop", "Loc", "Org", "Other"]


def test_relation_types(loader: Conll04Loader) -> None:
    """TC6 (часть 2): relation_types возвращает корректный список типов."""
    types = loader.relation_types
    assert types == ["Located_In", "Work_For", "OrgBased_In", "Live_In", "Kill"]


# ════════════════════════════════════════════════════════════════════════════
# TC7: размеры сплитов (интеграционный тест)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.slow
def test_split_sizes(loader: Conll04Loader) -> None:
    """TC7: интеграционный тест — реальная загрузка из HuggingFace.

    Проверяет размеры сплитов CoNLL-04:
      train=922, validation=231, test=288.
    """
    expected = {"train": 922, "validation": 231, "test": 288}

    for split, expected_size in expected.items():
        records = loader.load(split)
        assert len(records) == expected_size, (
            f"Split '{split}': expected {expected_size}, got {len(records)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# TC8: формат ID — conll04_{split}_{idx}
# ════════════════════════════════════════════════════════════════════════════


def test_record_id_format(loader: Conll04Loader) -> None:
    """TC8: ID содержит split и 0-based индекс, формат conll04_{split}_{idx}."""
    record = {
        "tokens": ["Hello", "world"],
        "entities": [],
        "relations": [],
    }

    # Проверяем ID для разных сплитов и индексов
    for split in ("train", "validation", "test"):
        result = loader._normalize_record(record, split, 5)
        expected_id = f"conll04_{split}_5"
        assert result.id == expected_id, (
            f"Expected ID '{expected_id}', got '{result.id}'"
        )

    # Проверяем нулевой индекс
    result0 = loader._normalize_record(record, "test", 0)
    assert result0.id == "conll04_test_0"

    # Проверяем большой индекс
    result_large = loader._normalize_record(record, "train", 999)
    assert result_large.id == "conll04_train_999"


# ════════════════════════════════════════════════════════════════════════════
# Error handling: неизвестный split → ValueError
# ════════════════════════════════════════════════════════════════════════════


def test_unknown_split_raises_valueerror(loader: Conll04Loader) -> None:
    """Неизвестный сплит в load() → ValueError с информативным сообщением."""
    with pytest.raises(ValueError, match="Unknown split"):
        loader.load("nonexistent")


def test_unknown_split_message_contains_available(loader: Conll04Loader) -> None:
    """Сообщение ValueError содержит список доступных сплитов."""
    with pytest.raises(ValueError, match="Available"):
        loader.load("invalid_split")


# ════════════════════════════════════════════════════════════════════════════
# Error handling: сущность с start > end → ValueError
# ════════════════════════════════════════════════════════════════════════════


def test_entity_start_greater_than_end_raises_valueerror(loader: Conll04Loader) -> None:
    """Сущность с start > end → ValueError (битые данные)."""
    record = {
        "tokens": ["A", "B", "C"],
        "entities": [{"type": "Other", "start": 2, "end": 1}],  # start > end
        "relations": [],
    }
    with pytest.raises(ValueError, match="start > end"):
        loader._normalize_record(record, "test", 0)
