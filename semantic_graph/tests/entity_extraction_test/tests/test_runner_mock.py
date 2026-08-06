"""Тесты ExperimentRunner с замоканными LLM-клиентами (Constitution T3).

Включает тесты async-батчинга: batch_size, обработка ошибок в батчах,
ограничение конкурентности через Semaphore.
"""

from __future__ import annotations

import asyncio
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


def _make_n_records(n: int) -> list[DatasetRecord]:
    """Создать n минимальных записей с уникальными ID."""
    return [
        _make_minimal_record(
            record_id="test_{}".format(i),
            text="Record {}: Apple is in Cupertino.".format(i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Existing Tests (batch_size=1 — поведение как раньше)
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
    assert result.model_ner != "none"
    assert result.model_re == "none"
    assert result.task_type == "ner_only"
    assert result.dataset == "conll04"
    assert result.split == "test"
    assert "f1_ner" in result.metrics
    assert "f1_re" in result.metrics
    assert "avg_response_time_sec" in result.metrics
    assert "avg_batch_time_sec" in result.metrics
    assert "total_samples" in result.metrics
    assert result.metrics["total_samples"] == 1

    ner_client.extract_entities.assert_called_once()


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_e3_mocked(mock_get_loader: MagicMock) -> None:
    """E3 с замоканным QwenClient combined_single_call mode."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_record_with_relation()],
        entity_types=["ORG", "LOC"],
        relation_types=["located_in"],
    )

    settings = ExperimentSettings(results_dir="/tmp/test_results_e3")
    runner = ExperimentRunner(settings)

    combined_client = _make_ner_mock()
    combined_client.extract_entities_and_relations = AsyncMock(
        return_value=(
            [PredictedEntity(name="Apple", type="ORG"), PredictedEntity(name="Cupertino", type="LOC")],
            [PredictedRelation(head="Apple", tail="Cupertino", type="located_in")],
        ),
    )

    result = await runner.run_experiment(
        experiment_id="E3",
        ner_client=combined_client,
        re_client=combined_client,
        re_mode="combined_single_call",
        dataset_name="conll04",
        split="test",
    )

    assert result.experiment_id == "E3"
    assert result.model_ner != "none"
    assert result.model_re != "none"
    assert result.task_type == "ner_re"
    assert isinstance(result.metrics["f1_ner"], float)
    assert isinstance(result.metrics["f1_re"], float)

    combined_client.extract_entities_and_relations.assert_called_once()
    combined_client.extract_entities.assert_not_called()


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_e4_mocked(mock_get_loader: MagicMock) -> None:
    """E4 с замоканным QwenClient separate mode."""
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

    ner_client.extract_entities.assert_called()
    re_client.extract_relations.assert_called()


@pytest.mark.asyncio
@patch("runner.experiment_runner.UniNerClient")
@patch("runner.experiment_runner.GleanerClient")
@patch("runner.experiment_runner.OllamaClient")
@patch("runner.experiment_runner.QwenClient")
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_full_cycle_mocked(
    mock_get_loader: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_ollama_cls: MagicMock,
    mock_gliner_cls: MagicMock,
    mock_uniner_cls: MagicMock,
) -> None:
    """Полный цикл E1–E6 с замоканными клиентами."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_record_with_relation()],
        entity_types=["ORG", "LOC"],
        relation_types=["located_in"],
    )

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
    mock_ollama_cls.return_value = _make_realistic_client_mock()

    settings = ExperimentSettings(
        results_dir="/tmp/test_results_full_cycle",
        datasets=["conll04"],
        splits=["test"],
        max_samples=1,
    )
    runner = ExperimentRunner(settings)

    results = await runner.run_all()
    assert len(results) == 8
    for result in results:
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")

    output_path = runner.save_results(results)
    assert output_path.exists()
    assert output_path.suffix == ".json"

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 8
    for item in data:
        assert "experiment_id" in item
        assert "metrics" in item
        assert "f1_ner" in item["metrics"]


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_error_handling(mock_get_loader: MagicMock) -> None:
    """Mock клиент, выбрасывающий ConnectionError."""
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

    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E1"
    assert result.metrics["f1_ner"] == 0.0
    assert result.metrics["total_samples"] == 0
    assert result.metrics["failed_samples"] == 1


@pytest.mark.asyncio
@patch("runner.experiment_runner.UniNerClient")
@patch("runner.experiment_runner.GleanerClient")
@patch("runner.experiment_runner.OllamaClient")
@patch("runner.experiment_runner.QwenClient")
@patch.object(ExperimentRunner, "_get_loader")
async def test_runner_experiment_filter(
    mock_get_loader: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_ollama_cls: MagicMock,
    mock_gliner_cls: MagicMock,
    mock_uniner_cls: MagicMock,
) -> None:
    """Проверка что experiment_filter='E1' запускает только E1."""
    mock_get_loader.return_value = _make_mock_loader(
        records=[_make_minimal_record()],
    )

    mock_uniner = MagicMock()
    mock_uniner.extract_entities = AsyncMock(
        return_value=[PredictedEntity(name="Apple", type="ORG")],
    )
    mock_uniner_cls.return_value = mock_uniner
    mock_gliner_cls.return_value = MagicMock()
    mock_qwen_cls.return_value = MagicMock()

    settings = ExperimentSettings(
        results_dir="/tmp/test_results_filter",
        datasets=["conll04"],
        splits=["test"],
        max_samples=1,
    )
    runner = ExperimentRunner(settings)

    results = await runner.run_all(experiment_filter="E1")
    assert len(results) == 1
    assert results[0].experiment_id == "E1"


# ---------------------------------------------------------------------------
# Batching Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_batch_size_1_compatibility(mock_get_loader: MagicMock) -> None:
    """batch_size=1 (default): поведение идентично последовательному."""
    records = _make_n_records(5)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(batch_size=1, results_dir="/tmp/test_batch_1")
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

    assert result.metrics["total_samples"] == 5
    assert result.metrics["failed_samples"] == 0
    assert "avg_batch_time_sec" in result.metrics
    assert ner_client.extract_entities.call_count == 5


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_grouping_6_records_batch_size_3(
    mock_get_loader: MagicMock,
) -> None:
    """6 записей, batch_size=3 -> 2 батча по 3 записи."""
    records = _make_n_records(6)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(
        batch_size=3,
        max_concurrent_batches=4,
        results_dir="/tmp/test_batch_3",
    )
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

    assert result.metrics["total_samples"] == 6
    assert result.metrics["failed_samples"] == 0
    assert "avg_batch_time_sec" in result.metrics
    assert isinstance(result.metrics["avg_batch_time_sec"], float)
    assert ner_client.extract_entities.call_count == 6


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_last_batch_smaller(mock_get_loader: MagicMock) -> None:
    """5 записей, batch_size=3 -> батчи: [3, 2] (последний уменьшенный)."""
    records = _make_n_records(5)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(
        batch_size=3,
        results_dir="/tmp/test_batch_last",
    )
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

    assert result.metrics["total_samples"] == 5
    assert result.metrics["failed_samples"] == 0
    assert ner_client.extract_entities.call_count == 5


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_error_in_batch(mock_get_loader: MagicMock) -> None:
    """3 записи в батче, одна кидает ConnectionError — остальные обработаны."""
    records = _make_n_records(3)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(
        batch_size=3,
        results_dir="/tmp/test_batch_error",
    )
    runner = ExperimentRunner(settings)

    ner_client = AsyncMock()
    ner_client.extract_entities = AsyncMock(
        side_effect=[
            [PredictedEntity(name="Apple", type="ORG")],
            ConnectionError("mocked error"),
            [PredictedEntity(name="Apple", type="ORG")],
        ],
    )

    result = await runner.run_experiment(
        experiment_id="E1",
        ner_client=ner_client,
        re_client=None,
        re_mode="none",
        dataset_name="conll04",
        split="test",
    )

    assert result.metrics["total_samples"] == 2
    assert result.metrics["failed_samples"] == 1
    assert ner_client.extract_entities.call_count == 3


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_max_concurrent_1(mock_get_loader: MagicMock) -> None:
    """max_concurrent_batches=1: проверка отсутствия перекрытия батчей.

    Используем счётчик активных вызовов с asyncio.Lock.
    """
    records = _make_n_records(6)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(
        batch_size=2,
        max_concurrent_batches=1,
        results_dir="/tmp/test_batch_concurrent_1",
    )
    runner = ExperimentRunner(settings)

    concurrent_counter = 0
    max_concurrent_seen = 0
    lock = asyncio.Lock()

    async def _spy_extract_entities(text, entity_types):
        nonlocal concurrent_counter, max_concurrent_seen
        async with lock:
            concurrent_counter += 1
            if concurrent_counter > max_concurrent_seen:
                max_concurrent_seen = concurrent_counter
        await asyncio.sleep(0.01)
        async with lock:
            concurrent_counter -= 1
        return [PredictedEntity(name="Apple", type="ORG")]

    ner_client = AsyncMock()
    ner_client.extract_entities = AsyncMock(side_effect=_spy_extract_entities)

    result = await runner.run_experiment(
        experiment_id="E1",
        ner_client=ner_client,
        re_client=None,
        re_mode="none",
        dataset_name="conll04",
        split="test",
    )

    assert result.metrics["total_samples"] == 6
    assert result.metrics["failed_samples"] == 0
    assert max_concurrent_seen <= 2, (
        "Expected <= 2 concurrent calls with max_concurrent_batches=1 "
        "and batch_size=2, got {}".format(max_concurrent_seen)
    )


@pytest.mark.asyncio
@patch.object(ExperimentRunner, "_get_loader")
async def test_batching_avg_batch_time_metric(mock_get_loader: MagicMock) -> None:
    """Проверка наличия и корректности метрики avg_batch_time_sec."""
    records = _make_n_records(4)
    mock_get_loader.return_value = _make_mock_loader(records=records)

    settings = ExperimentSettings(
        batch_size=2,
        results_dir="/tmp/test_batch_metric",
    )
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

    assert "avg_batch_time_sec" in result.metrics
    assert isinstance(result.metrics["avg_batch_time_sec"], float)
    assert result.metrics["avg_batch_time_sec"] >= 0.0
