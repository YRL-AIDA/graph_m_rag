"""Тесты ExperimentRunner с замоканными LLM-клиентами (Constitution T3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Добавляем entity_extraction_test/ в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner.config import ExperimentSettings
from runner.experiment_runner import ExperimentRunner
from testdata.base_loader import (
    DatasetRecord,
    Entity,
    ExperimentResult,
    PredictedEntity,
    PredictedRelation,
    Relation,
)


# ---------------------------------------------------------------------------
# Helpers — минимальные записи датасета
# ---------------------------------------------------------------------------


def _make_minimal_record(
    record_id: str = "test_0",
    text: str = "Apple is in Cupertino.",
) -> DatasetRecord:
    """Создать минимальную запись датасета с одной сущностью."""
    return DatasetRecord(
        id=record_id,
        text=text,
        entities=[Entity(name="Apple", type="ORG", start=0, end=4)],
        relations=[],
    )


def _make_record_with_relation() -> DatasetRecord:
    """Запись с сущностями и отношением."""
    return DatasetRecord(
        id="test_0",
        text="Apple is based in Cupertino.",
        entities=[
            Entity(name="Apple", type="ORG", start=0, end=4),
            Entity(name="Cupertino", type="LOC", start=17, end=25),
        ],
        relations=[
            Relation(
                head_idx=0,
                tail_idx=1,
                head_start=0,
                head_end=4,
                tail_start=17,
                tail_end=25,
                type="located_in",
            ),
        ],
    )


def _make_mock_loader(
    records: list[DatasetRecord] | None = None,
    entity_types: list[str] | None = None,
    relation_types: list[str] | None = None,
) -> MagicMock:
    """Создать mock DatasetLoader с заданными записями и типами."""
    loader = MagicMock()
    loader.load.return_value = records or [_make_minimal_record()]
    loader.entity_types = entity_types or ["ORG", "PERSON"]
    loader.relation_types = relation_types or ["located_in"]
    loader.splits.return_value = ["test", "validation", "train"]
    return loader


def _make_ner_mock() -> AsyncMock:
    """Mock NER-клиент: возвращает одну сущность Apple/ORG."""
    mock = AsyncMock()
    mock.extract_entities = AsyncMock(
        return_value=[PredictedEntity(name="Apple", type="ORG")],
    )
    return mock


def _make_re_mock() -> AsyncMock:
    """Mock RE-клиент: возвращает одно отношение."""
    mock = AsyncMock()
    mock.extract_relations = AsyncMock(
        return_value=[
            PredictedRelation(head="Apple", tail="Cupertino", type="located_in"),
        ],
    )
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_e1_mocked(mock_get_loader: MagicMock) -> None:
    """E1 с замоканным UniNerClient: проверка структуры ExperimentResult."""
    mock_get_loader.return_value = _make_mock_loader()

    settings = ExperimentSettings(results_dir="/tmp/test_results_e1")
    runner = ExperimentRunner(settings)

    ner_client = _make_ner_mock()

    result = await runner.run_experiment(
        experiment_id="E1",
        ner_client=ner_client,
        re_client=None,
        re_mode="none",
        dataset_name="conll04",
        split="test",
    )

    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E1"
    assert result.model_ner != "none"  # resolved from client type
    assert result.model_re == "none"
    assert result.task_type == "ner_only"
    assert result.dataset == "conll04"
    assert result.split == "test"
    assert "f1_ner" in result.metrics
    assert "f1_re" in result.metrics
    assert "avg_response_time_sec" in result.metrics
    assert "total_samples" in result.metrics
    assert result.metrics["total_samples"] == 1  # успешно обработана

    ner_client.extract_entities.assert_called_once()


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_e3_mocked(mock_get_loader: MagicMock) -> None:
    """E3 с замоканным QwenClient combined mode."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_record_with_relation()],
        entity_types=["ORG", "LOC"],
        relation_types=["located_in"],
    )

    settings = ExperimentSettings(results_dir="/tmp/test_results_e3")
    runner = ExperimentRunner(settings)

    # Combined mode: один клиент делает и NER, и RE
    combined_client = _make_ner_mock()
    combined_client.extract_relations = AsyncMock(
        return_value=[
            PredictedRelation(head="Apple", tail="Cupertino", type="located_in"),
        ],
    )

    result = await runner.run_experiment(
        experiment_id="E3",
        ner_client=combined_client,
        re_client=combined_client,
        re_mode="combined",
        dataset_name="conll04",
        split="test",
    )

    assert result.experiment_id == "E3"
    assert result.model_ner != "none"
    assert result.model_re != "none"
    assert result.task_type == "ner_re"
    assert isinstance(result.metrics["f1_ner"], float)
    assert isinstance(result.metrics["f1_re"], float)

    # NER был вызван
    combined_client.extract_entities.assert_called()
    # RE тоже был вызван (combined mode)
    combined_client.extract_relations.assert_called()


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_e4_mocked(mock_get_loader: MagicMock) -> None:
    """E4 с замоканным QwenClient separate mode: extract_entities + extract_relations."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_record_with_relation()],
        entity_types=["ORG", "LOC"],
        relation_types=["located_in"],
    )

    settings = ExperimentSettings(results_dir="/tmp/test_results_e4")
    runner = ExperimentRunner(settings)

    ner_client = _make_ner_mock()
    re_client = _make_re_mock()

    result = await runner.run_experiment(
        experiment_id="E4",
        ner_client=ner_client,
        re_client=re_client,
        re_mode="separate",
        dataset_name="conll04",
        split="test",
    )

    assert result.experiment_id == "E4"
    assert result.model_ner != "none"
    assert result.model_re != "none"

    # Оба клиента вызваны
    ner_client.extract_entities.assert_called()
    re_client.extract_relations.assert_called()


@pytest.mark.asyncio
@patch("runner.experiment_runner.UniNerClient")
@patch("runner.experiment_runner.GleanerClient")
@patch("runner.experiment_runner.QwenClient")
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_full_cycle_mocked(
    mock_get_loader: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_gliner_cls: MagicMock,
    mock_uniner_cls: MagicMock,
) -> None:
    """Полный цикл E1–E6 с замоканными клиентами: save_results создаёт валидный JSON."""
    # Настраиваем mock-загрузчик
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_record_with_relation()],
        entity_types=["ORG", "LOC"],
        relation_types=["located_in"],
    )

    # Все клиенты возвращают осмысленные данные
    def _make_realistic_client_mock() -> MagicMock:
        client = MagicMock()
        client.extract_entities = AsyncMock(
            return_value=[
                PredictedEntity(name="Apple", type="ORG"),
                PredictedEntity(name="Cupertino", type="LOC"),
            ],
        )
        client.extract_relations = AsyncMock(
            return_value=[
                PredictedRelation(head="Apple", tail="Cupertino", type="located_in"),
            ],
        )
        return client

    mock_uniner_cls.return_value = _make_realistic_client_mock()
    mock_gliner_cls.return_value = _make_realistic_client_mock()
    mock_qwen_cls.return_value = _make_realistic_client_mock()

    settings = ExperimentSettings(
        results_dir="/tmp/test_results_full_cycle",
        datasets=["conll04"],
        splits=["test"],
        max_samples=1,
    )
    runner = ExperimentRunner(settings)

    results = await runner.run_all()
    assert len(results) == 6  # E1–E6
    for result in results:
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id in ("E1", "E2", "E3", "E4", "E5", "E6")

    # Сохраняем и проверяем JSON
    output_path = runner.save_results(results)
    assert output_path.exists()
    assert output_path.suffix == ".json"

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 6
    for item in data:
        assert "experiment_id" in item
        assert "metrics" in item
        assert "f1_ner" in item["metrics"]


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_error_handling(mock_get_loader: MagicMock) -> None:
    """Mock клиент, выбрасывающий ConnectionError → runner не падает, возвращает partial."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_minimal_record()],
    )

    settings = ExperimentSettings(results_dir="/tmp/test_results_error")
    runner = ExperimentRunner(settings)

    ner_client = AsyncMock()
    ner_client.extract_entities = AsyncMock(
        side_effect=ConnectionError("mocked connection error"),
    )

    result = await runner.run_experiment(
        experiment_id="E1",
        ner_client=ner_client,
        re_client=None,
        re_mode="none",
        dataset_name="conll04",
        split="test",
    )

    # Раннер не упал, вернул результат
    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E1"
    # Все метрики — 0 (все записи провалены)
    assert result.metrics["f1_ner"] == 0.0
    assert result.metrics["total_samples"] == 0  # successful=0
    assert result.metrics["failed_samples"] == 1  # one failed
