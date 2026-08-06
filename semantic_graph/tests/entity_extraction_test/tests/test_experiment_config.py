"""Тесты для ExperimentSettings — значения по умолчанию и переопределение через env."""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем entity_extraction_test/ в sys.path для импорта runner.config
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner.config import ExperimentSettings


def test_default_values() -> None:
    """Проверка всех значений по умолчанию."""
    settings = ExperimentSettings()

    assert settings.qwen_base_url == "http://192.168.19.127:8888/v1"
    assert settings.qwen_api_key == "EMPTY"
    assert settings.qwen_model == "qwen3-vl:32b"
    assert settings.uniner_base_url == "http://192.168.19.127:9898/v1"
    assert settings.gliner_base_url == "http://192.168.19.127:9596/v1"
    assert settings.gliner_api_key == "EMPTY"
    assert settings.gliner_model == "gliner-community/gliner_small-v2.5"
    assert settings.scierc_data_path == "/home/ivan/work/ooo/graph_m_rag/datasets_/scierc"
    assert settings.prompts_dir.endswith("prompts")
    assert settings.results_dir.endswith("results")
    assert settings.datasets == ["conll04", "scierc"]
    assert settings.splits == ["test"]
    assert settings.max_samples is None
    assert settings.ollama_base_url == "http://192.168.55.242:7869"
    assert settings.ollama_model == "qwen3-coder:30b"
    assert settings.ollama_think == False
    assert settings.ollama_num_predict == 256
    assert settings.ollama_num_ctx == 4096
    assert settings.ollama_keep_alive == "5m"



def test_env_override(monkeypatch) -> None:
    """EXPERIMENT_QWEN_BASE_URL=http://x:1/v1 → settings.qwen_base_url == 'http://x:1/v1'."""
    monkeypatch.setenv("EXPERIMENT_QWEN_BASE_URL", "http://x:1/v1")
    settings = ExperimentSettings()
    assert settings.qwen_base_url == "http://x:1/v1"


def test_env_override_max_samples(monkeypatch) -> None:
    """EXPERIMENT_MAX_SAMPLES=50 → settings.max_samples == 50."""
    monkeypatch.setenv("EXPERIMENT_MAX_SAMPLES", "50")
    settings = ExperimentSettings()
    assert settings.max_samples == 50


def test_env_override_datasets(monkeypatch) -> None:
    """EXPERIMENT_DATASETS='["conll04"]' → settings.datasets == ['conll04']."""
    monkeypatch.setenv("EXPERIMENT_DATASETS", '["conll04"]')
    settings = ExperimentSettings()
    assert settings.datasets == ["conll04"]
