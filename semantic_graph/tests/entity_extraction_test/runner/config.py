"""Настройки экспериментов NER/RE — Pydantic-settings модель.

Все поля переопределяются через переменные окружения с префиксом ``EXPERIMENT_``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class ExperimentSettings(BaseSettings):
    """Параметры запуска экспериментов NER/RE.

    Значения по умолчанию заданы для локального стенда.
    Переменные окружения с префиксом ``EXPERIMENT_`` переопределяют любое поле.
    """

    #qwen_base_url: str = "http://192.168.19.127:8888/v1"
    #qwen_api_key: str = "EMPTY"
    #qwen_model: str = "Qwen/Qwen3-VL-32B-Thinking"
    qwen_base_url: str = "http://192.168.19.127:8888/v1"
    qwen_api_key: str = "EMPTY"
    qwen_model: str = "Qwen/Qwen3-VL-32B-Thinking"
    uniner_base_url: str = "http://192.168.19.127:9898/v1"

    gliner_base_url: str = "http://192.168.19.127:9596/v1"
    gliner_api_key: str = "EMPTY"
    gliner_model: str = "gliner-community/gliner_small-v2.5"

    ollama_base_url: str = "http://192.168.55.242:7869"
    ollama_model: str = "qwen3-vl:32b"
    ollama_think: bool | str = 'low'
    ollama_num_predict: int = 5000
    ollama_num_ctx: int = 8000
    ollama_keep_alive: str = "5m"

    scierc_data_path: str = "/home/ivan/work/ooo/graph_m_rag/datasets_/scierc"

    prompts_dir: str = str(Path(__file__).parent.parent / "prompts")
    results_dir: str = str(Path(__file__).parent.parent / "results")

    datasets: list[str] = ["conll04", "scierc"]
    splits: list[str] = ["test"]
    max_samples: int | None = None

    batch_size: int = 5
    """Размер батча (записей, обрабатываемых параллельно). 1 = последовательно (как сейчас)."""

    max_concurrent_batches: int = 2
    """Максимальное количество параллельно выполняемых батчей.
    Эффективный лимит конкурентных API-вызовов = max_concurrent_batches * batch_size."""

    class Config:
        env_prefix = "EXPERIMENT_"
        env_file = ".env"
        extra = "ignore"
