"""Ядро оркестрации экспериментов NER/RE.

ExperimentRunner запускает эксперименты E1–E6 по матрице «эксперимент × датасет ×
сплит», агрегирует метрики и сохраняет результаты в JSON и Markdown.

Поддерживает async-батчинг: записи группируются в батчи и обрабатываются
параллельно через ``asyncio.gather()`` с семафором для ограничения конкурентности.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Добавляем entity_extraction_test/ в sys.path для внутрипроектных импортов --
_ENTITY_DIR = str(Path(__file__).resolve().parent.parent)
if _ENTITY_DIR not in sys.path:
    sys.path.insert(0, _ENTITY_DIR)

from clients.gliner_client import GleanerClient  # noqa: E402
from clients.hybrid_client import HybridClient  # noqa: E402
from clients.qwen_client import QwenClient  # noqa: E402
from clients.ollama_client import OllamaClient  # noqa: E402
from clients.uniner_client import UniNerClient  # noqa: E402
from metrics.metrics import (  # noqa: E402
    compute_avg_response_time,
    compute_ner_f1,
    compute_re_f1,
)
from testdata.base_loader import (  # noqa: E402
    BaseModelClient,
    DatasetLoader,
    DatasetRecord,
    ExperimentResult,
)
from testdata.conll04_loader import Conll04Loader  # noqa: E402
from testdata.scierc_loader import SciERCLoader  # noqa: E402

from .config import ExperimentSettings  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# ExperimentRunner
# =============================================================================


class ExperimentRunner:
    """Оркестратор экспериментов NER/RE.

    Запускает эксперименты по матрице (E1–E6 × датасеты × сплиты), собирает
    метрики и сохраняет результаты.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        """Инициализация с настройками эксперимента.

        Args:
            settings: Параметры запуска (URL моделей, пути к данным, фильтры).
        """
        self.settings = settings
        self._loaders: dict[str, DatasetLoader] = {}

    # -------------------------------------------------------------------------
    # Загрузчики датасетов (ленивая инициализация)
    # -------------------------------------------------------------------------

    def _get_loader(self, dataset_name: str) -> DatasetLoader:
        """Вернуть загрузчик датасета (ленивая инициализация с кэшированием).

        Args:
            dataset_name: ``"conll04"`` или ``"scierc"``.

        Returns:
            Загрузчик DatasetLoader.

        Raises:
            ValueError: Если имя датасета не поддерживается.
        """
        if dataset_name not in self._loaders:
            if dataset_name == "conll04":
                self._loaders[dataset_name] = Conll04Loader()
            elif dataset_name == "scierc":
                self._loaders[dataset_name] = SciERCLoader(
                    data_path=self.settings.scierc_data_path,
                )
            else:
                raise ValueError(f"Неизвестный датасет: {dataset_name}")
        return self._loaders[dataset_name]

    # -------------------------------------------------------------------------
    # run_experiment — один эксперимент на одном датасете/сплите
    # -------------------------------------------------------------------------

    async def run_experiment(  # noqa: C901
        self,
        experiment_id: str,
        ner_client: BaseModelClient,
        re_client: BaseModelClient | None,
        re_mode: str,
        dataset_name: str,
        split: str,
    ) -> ExperimentResult:
        """Запустить один эксперимент (E1–E6) на конкретном датасете и сплите.

        Args:
            experiment_id: Идентификатор эксперимента (``"E1"``–``"E6"``).
            ner_client: Клиент NER-модели.
            re_client: Клиент RE-модели (может быть ``None`` для ``re_mode="none"``
                       и ``"combined_single_call"``).
            re_mode: Режим извлечения отношений: ``"combined_single_call"``,
                     ``"separate"`` или ``"none"``.
            dataset_name: Имя датасета (``"conll04"``, ``"scierc"``).
            split: Сплит датасета (``"test"``, ``"validation"``, ``"train"``).

        Returns:
            :class:`ExperimentResult` с агрегированными метриками.

        Raises:
            RuntimeError: Если датасет не удалось загрузить.
        """
        # --- Загрузка датасета ---
        loader = self._get_loader(dataset_name)

        try:
            records = loader.load(split)
        except (FileNotFoundError, ConnectionError, OSError) as exc:
            raise RuntimeError(
                f"Не удалось загрузить датасет {dataset_name}/{split}: {exc}"
            ) from exc

        # --- Проверка на пустой датасет ---
        if not records:
            logger.warning(
                "Пустой датасет %s/%s — эксперимент %s пропущен",
                dataset_name,
                split,
                experiment_id,
            )
            return ExperimentResult(
                experiment_id=experiment_id,
                model_ner=self._resolve_model_name(ner_client),
                model_re=(
                    self._resolve_model_name(re_client) if re_client else "none"
                ),
                dataset=dataset_name,
                split=split,
                task_type=self._task_type(re_mode),
                config={
                    "ner_client": type(ner_client).__name__,
                    "re_client": type(re_client).__name__ if re_client else "none",
                    "re_mode": re_mode,
                    "entity_types": loader.entity_types,
                    "relation_types": loader.relation_types,
                },
                metrics={
                    "precision_ner": 0.0,
                    "recall_ner": 0.0,
                    "f1_ner": 0.0,
                    "precision_re": 0.0,
                    "recall_re": 0.0,
                    "f1_re": 0.0,
                    "ner_entity_count": 0.0,
                    "avg_response_time_sec": 0.0,
                    "avg_batch_time_sec": 0.0,
                    "total_samples": 0,
                    "failed_samples": 0,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # --- Ограничение количества записей ---
        entity_types = loader.entity_types
        relation_types = loader.relation_types
        limit = (
            min(self.settings.max_samples, len(records))
            if self.settings.max_samples is not None
            else len(records)
        )
        sample = records[:limit]

        # --- Группировка записей в батчи ---
        batch_size = max(self.settings.batch_size, 1)
        batches = [
            sample[i : i + batch_size]
            for i in range(0, len(sample), batch_size)
        ]

        # --- Параллельное выполнение батчей с семафором ---
        sem = asyncio.Semaphore(self.settings.max_concurrent_batches)

        async def _run_one_batch(
            batch_idx: int,
            batch_records: list[DatasetRecord],
        ) -> list[tuple[DatasetRecord, object, float]]:
            """Выполнить один батч: параллельный запуск _process_record для всех записей.

            Args:
                batch_idx: Порядковый номер батча (0-based) для логирования.
                batch_records: Список записей в батче.

            Returns:
                Список кортежей ``(record, result_or_exception, batch_time)``.
            """
            async with sem:
                logger.info(
                    "Батч %d/%d: %d записей",
                    batch_idx + 1,
                    len(batches),
                    len(batch_records),
                )
                t_batch_start = time.monotonic()
                tasks = [
                    self._process_record(
                        ner_client=ner_client,
                        re_client=re_client,
                        re_mode=re_mode,
                        record=r,
                        entity_types=entity_types,
                        relation_types=relation_types,
                    )
                    for r in batch_records
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                t_batch_end = time.monotonic()
                batch_time = t_batch_end - t_batch_start
                return [
                    (r, raw, batch_time)
                    for r, raw in zip(batch_records, raw_results)
                ]

        all_batch_results = await asyncio.gather(
            *[_run_one_batch(i, b) for i, b in enumerate(batches)],
        )

        # --- Агрегация результатов батчей ---
        ner_timings: list[float] = []
        re_timings: list[float] = []
        batch_timings: list[float] = []
        ner_precisions: list[float] = []
        ner_recalls: list[float] = []
        ner_f1s: list[float] = []
        re_precisions: list[float] = []
        re_recalls: list[float] = []
        re_f1s: list[float] = []
        ner_entity_counts: list[int] = []
        successful = 0
        failed = 0

        for batch_output in all_batch_results:
            for record, result, batch_time in batch_output:
                if isinstance(result, ConnectionError):
                    logger.exception(
                        "Ошибка соединения на записи %s (эксперимент %s)",
                        record.id,
                        experiment_id,
                    )
                    failed += 1
                    continue
                elif isinstance(result, Exception):
                    logger.warning(
                        "Неожиданная ошибка на записи %s (эксперимент %s): %s",
                        record.id,
                        experiment_id,
                        result,
                    )
                    failed += 1
                    continue

                pred_entities, pred_relations, n_time, r_time = result
                print(record.entities,pred_entities)
                ner_timings.append(n_time)
                if r_time is not None:
                    re_timings.append(r_time)
                batch_timings.append(batch_time)

                ner_entity_counts.append(len(pred_entities))

                # Метрики NER
                ner_metrics = compute_ner_f1(
                    record.entities, pred_entities, entity_types
                )
                ner_precisions.append(ner_metrics["precision"])
                ner_recalls.append(ner_metrics["recall"])
                ner_f1s.append(ner_metrics["f1"])

                # Метрики RE
                re_metrics = compute_re_f1(
                    gold_entities=record.entities,
                    gold_relations=record.relations,
                    pred_entities=pred_entities,
                    pred_relations=pred_relations,
                    entity_types=entity_types,
                    relation_types=relation_types,
                )
                re_precisions.append(re_metrics["precision"])
                re_recalls.append(re_metrics["recall"])
                re_f1s.append(re_metrics["f1"])

                successful += 1

        if successful == 0:
            logger.warning(
                "Все записи (%d) провалены для %s/%s (%s)",
                limit,
                dataset_name,
                split,
                experiment_id,
            )

        # --- Агрегация ---
        all_timings = ner_timings + re_timings

        return ExperimentResult(
            experiment_id=experiment_id,
            model_ner=self._resolve_model_name(ner_client),
            model_re=(
                self._resolve_model_name(re_client) if re_client else "none"
            ),
            dataset=dataset_name,
            split=split,
            task_type=self._task_type(re_mode),
            config={
                "ner_client": type(ner_client).__name__,
                "re_client": type(re_client).__name__ if re_client else "none",
                "re_mode": re_mode,
                "entity_types": entity_types,
                "relation_types": relation_types,
            },
            metrics={
                "precision_ner": _safe_mean(ner_precisions),
                "recall_ner": _safe_mean(ner_recalls),
                "f1_ner": _safe_mean(ner_f1s),
                "precision_re": _safe_mean(re_precisions),
                "recall_re": _safe_mean(re_recalls),
                "f1_re": _safe_mean(re_f1s),
                "ner_entity_count": _safe_mean([float(c) for c in ner_entity_counts]),
                "avg_response_time_sec": compute_avg_response_time(all_timings),
                "avg_batch_time_sec": compute_avg_response_time(batch_timings),
                "total_samples": successful,
                "failed_samples": failed,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # -------------------------------------------------------------------------
    # Обработка одной записи
    # -------------------------------------------------------------------------

    async def _process_record(
        self,
        ner_client: BaseModelClient,
        re_client: BaseModelClient | None,
        re_mode: str,
        record: DatasetRecord,
        entity_types: list[str],
        relation_types: list[str],
    ) -> tuple[
        list,  # pred_entities
        list,  # pred_relations
        float,  # ner_time
        float | None,  # re_time
    ]:
        """Обработать одну запись датасета: NER, (опционально) RE.

        Args:
            ner_client: NER-клиент.
            re_client: RE-клиент (игнорируется при ``re_mode="none"``
                       и ``"combined_single_call"``).
            re_mode: Режим RE.
            record: Запись датасета.
            entity_types: Типы сущностей датасета.
            relation_types: Типы отношений датасета.

        Returns:
            Кортеж ``(pred_entities, pred_relations, ner_time, re_time)``.

        Raises:
            ConnectionError: При ошибках сетевого соединения или таймауте.
        """
        # --- Шаг 0: combined_single_call — один вызов API для NER+RE ---
        if re_mode == "combined_single_call":
            t0 = time.monotonic()
            try:
                pred_entities, pred_relations = await ner_client.extract_entities_and_relations(
                    record.text, entity_types, relation_types
                )
            except NotImplementedError:
                logger.warning(
                    "combined_single_call не поддерживается клиентом %s на записи %s",
                    type(ner_client).__name__,
                    record.id,
                )
                pred_entities = []
                pred_relations = []
            except ConnectionError:
                raise
            except Exception:
                logger.warning(
                    "Непарсируемый combined-ответ на записи %s",
                    record.id,
                    exc_info=True,
                )
                pred_entities = []
                pred_relations = []
            t1 = time.monotonic()
            ner_time = t1 - t0
            re_time = None  # время уже учтено в ner_time
            return pred_entities, pred_relations, ner_time, re_time

        # --- Шаг 1: NER ---
        t0 = time.monotonic()
        try:
            pred_entities = await ner_client.extract_entities(
                record.text, entity_types
            )
        except NotImplementedError:
            logger.warning(
                "NER не поддерживается клиентом %s на записи %s",
                type(ner_client).__name__,
                record.id,
            )
            pred_entities = []
        except ConnectionError:
            raise
        except Exception:
            logger.warning(
                "Непарсируемый ответ NER на записи %s",
                record.id,
                exc_info=True,
            )
            pred_entities = []
        t1 = time.monotonic()
        ner_time = t1 - t0

        # --- Шаг 2: RE ---
        pred_relations: list = []
        re_time: float | None = None

        if re_mode == "separate" and re_client is not None:
            t2 = time.monotonic()
            try:
                pred_relations = await re_client.extract_relations(
                    record.text, pred_entities, relation_types
                )
            except NotImplementedError:
                logger.warning(
                    "RE не поддерживается клиентом %s в separate-режиме",
                    type(re_client).__name__,
                )
                pred_relations = []
            except ConnectionError:
                raise
            except Exception:
                logger.warning(
                    "Непарсируемый ответ RE (separate) на записи %s",
                    record.id,
                    exc_info=True,
                )
                pred_relations = []
            t3 = time.monotonic()
            re_time = t3 - t2

        # re_mode == "none" → пропускаем RE

        return pred_entities, pred_relations, ner_time, re_time

    # -------------------------------------------------------------------------
    # run_all — полная матрица экспериментов
    # -------------------------------------------------------------------------

    async def run_all(
        self, experiment_filter: str | None = None
    ) -> list[ExperimentResult]:
        """Запустить все эксперименты E1–E6 × датасеты × сплиты.

        Создаёт клиентов моделей, итерирует матрицу, вызывает
        :meth:`run_experiment` для каждой комбинации.

        Args:
            experiment_filter: Если указан (``"E1"``–``"E6"``), выполняются только
                               эксперименты с этим идентификатором.

        Returns:
            Список :class:`ExperimentResult` для всех успешно завершённых
            комбинаций.
        """
        # --- Создание клиентов ---
        prompts = Path(self.settings.prompts_dir)
        ollama_client = OllamaClient(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            think=self.settings.ollama_think,
            num_predict=self.settings.ollama_num_predict,
            num_ctx=self.settings.ollama_num_ctx,
            keep_alive=self.settings.ollama_keep_alive,
            ner_prompt_path=str(prompts / "ner_prompt.md"),
            re_prompt_path=str(prompts / "re_prompt.md"),
            combined_prompt_path=str(prompts / "combined_prompt.md"),
        )

        qwen_client = QwenClient(
            base_url=self.settings.qwen_base_url,
            api_key=self.settings.qwen_api_key,
            model=self.settings.qwen_model,
            ner_prompt_path=str(prompts / "ner_prompt.md"),
            re_prompt_path=str(prompts / "re_prompt.md"),
            combined_prompt_path=str(prompts / "combined_prompt.md"),
        )
        uniner_client = UniNerClient(base_url=self.settings.uniner_base_url)
        gliner_client = GleanerClient(
            base_url=self.settings.gliner_base_url,
            api_key=self.settings.gliner_api_key,
            model=self.settings.gliner_model,
        )

        # Hybrid-клиенты для E5, E6
        hybrid_uniner_qwen = HybridClient(
            ner_client=uniner_client, re_client=qwen_client
        )
        hybrid_gliner_qwen = HybridClient(
            ner_client=gliner_client, re_client=qwen_client
        )

        # --- Матрица экспериментов ---
        experiments: list[tuple[str, BaseModelClient, BaseModelClient | None, str]] = [
            ("E1", uniner_client, None, "none"),                          # UniNer NER only
            ("E2", gliner_client, None, "none"),                           # Gleaner NER only
            ("E3", qwen_client, None, "combined_single_call"),             # Qwen combined NER+RE (single call)
            ("E4", qwen_client, qwen_client, "separate"),                  # Qwen NER + Qwen RE
            ("E5", hybrid_uniner_qwen, qwen_client, "separate"),           # Hybrid UniNer+Qwen
            ("E6", hybrid_gliner_qwen, qwen_client, "separate"),           # Hybrid Gleaner+Qwen
            ("E7", ollama_client, None, "combined_single_call"),             # Ollama combined NER+RE (single call)
            ("E8", ollama_client, ollama_client, "separate"),                  # Ollama NER + Ollama RE
        ]

        # --- Фильтрация матрицы ДО выполнения ---
        if experiment_filter is not None:
            experiments = [
                (eid, ner, re, mode)
                for (eid, ner, re, mode) in experiments
                if eid == experiment_filter
            ]

        results: list[ExperimentResult] = []

        for exp_id, ner_client, re_client, re_mode in experiments:
            for dataset_name in self.settings.datasets:
                for split in self.settings.splits:
                    logger.info(
                        "Запуск %s: dataset=%s, split=%s, re_mode=%s",
                        exp_id,
                        dataset_name,
                        split,
                        re_mode,
                    )
                    t_exp_start = time.monotonic()
                    try:
                        result = await self.run_experiment(
                            experiment_id=exp_id,
                            ner_client=ner_client,
                            re_client=re_client,
                            re_mode=re_mode,
                            dataset_name=dataset_name,
                            split=split,
                        )
                        t_exp_end = time.monotonic()
                        t_exp_elapsed = t_exp_end - t_exp_start
                        results.append(result)
                        logger.info(
                            "%s/%s/%s — NER F1=%.4f  RE F1=%.4f | общее время=%.1fс  среднее/пример=%.2fс",
                            exp_id,
                            dataset_name,
                            split,
                            result.metrics["f1_ner"],
                            result.metrics["f1_re"],
                            t_exp_elapsed,
                            t_exp_elapsed / max(result.metrics["total_samples"], 1),
                        )
                    except RuntimeError as exc:
                        logger.error(
                            "Пропуск %s/%s/%s: %s",
                            exp_id,
                            dataset_name,
                            split,
                            exc,
                        )
                        continue

        return results

    # -------------------------------------------------------------------------
    # save_results — сохранение результатов
    # -------------------------------------------------------------------------

    def save_results(self, results: list[ExperimentResult]) -> Path:
        """Сохранить результаты в JSON и Markdown-отчёт.

        Артефакты:
        - ``results/<timestamp>_experiments.json`` — полный дамп результатов.
        - ``results/<timestamp>_summary.md`` — сводная таблица F1 и времени.

        Args:
            results: Список результатов экспериментов.

        Returns:
            Путь к сохранённому JSON-файлу.
        """
        results_dir = Path(self.settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # --- JSON ---
        json_path = results_dir / f"{timestamp}_experiments.json"
        json_data = [r.model_dump() for r in results]
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Результаты сохранены: %s", json_path)

        # --- Markdown-сводка ---
        md_path = results_dir / f"{timestamp}_summary.md"
        md_lines = [
            "# Сводка экспериментов NER/RE",
            "",
            f"Дата запуска: {timestamp}",
            f"Всего экспериментов: {len(results)}",
            "",
            "| Эксперимент | Датасет | Сплит | NER F1 | RE F1 | Entities | Avg Time (s) | Успешно | Провалено |",
            "|------------|---------|-------|--------|-------|----------|-------------|---------|----------|",
        ]

        for r in results:
            m = r.metrics
            md_lines.append(
                f"| {r.experiment_id} | {r.dataset} | {r.split} "
                f"| {m['f1_ner']:.4f} | {m['f1_re']:.4f} "
                f"| {m['ner_entity_count']:.1f} "
                f"| {m['avg_response_time_sec']:.4f} "
                f"| {m['total_samples']} | {m['failed_samples']} |"
            )

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        logger.info("Сводка сохранена: %s", md_path)

        return json_path

    # -------------------------------------------------------------------------
    # Вспомогательные методы
    # -------------------------------------------------------------------------

    @staticmethod
    def _task_type(re_mode: str) -> str:
        """Определить тип задачи по режиму RE.

        Args:
            re_mode: ``"none"``, ``"combined_single_call"`` или ``"separate"``.

        Returns:
            ``"ner_only"`` или ``"ner_re"``.
        """
        if re_mode == "none":
            return "ner_only"
        return "ner_re"

    @staticmethod
    def _resolve_model_name(client: BaseModelClient | None) -> str:
        """Сопоставить клиент → короткое имя модели.

        Args:
            client: Экземпляр клиента (или ``None``).

        Returns:
            Короткое имя: ``"uniner"``, ``"gliner"``, ``"qwen"``, ``"hybrid"``
            или ``"none"``.
        """
        if client is None:
            return "none"

        name_map: dict[str, str] = {
            "QwenClient": "qwen",
            "UniNerClient": "uniner",
            "GleanerClient": "gliner",
            "OllamaClient": "ollama",
            "HybridClient": "hybrid",
        }
        class_name = type(client).__name__
        return name_map.get(class_name, class_name.lower())


# =============================================================================
# Вспомогательные функции уровня модуля
# =============================================================================


def _safe_mean(values: list[float]) -> float:
    """Безопасное среднее арифметическое.

    Args:
        values: Список чисел.

    Returns:
        Среднее значение, либо ``0.0`` если список пуст.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# =============================================================================
# CLI (главный блок)
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов NER/RE (E1–E6)",
    )
    parser.add_argument(
        "--experiment",
        choices=[f"E{i}" for i in range(1, 9)],
        default=None,
        help="Фильтр по конкретному эксперименту (по умолчанию: все)",
    )
    parser.add_argument(
        "--dataset",
        choices=["conll04", "scierc"],
        default=None,
        help="Фильтр по датасету (по умолчанию: все из настроек)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default=None,
        help="Фильтр по сплиту (по умолчанию: все из настроек)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Ограничение количества записей (по умолчанию: из настроек)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Размер батча (по умолчанию: EXPERIMENT_BATCH_SIZE)",
    )
    return parser


async def _main() -> None:
    """Точка входа CLI: парсинг аргументов, настройка, запуск экспериментов."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # 1. Загрузка настроек из env
    settings = ExperimentSettings()

    # 2. Парсинг CLI и применение переопределений
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.max_samples is not None:
        settings.max_samples = args.max_samples
    if args.dataset is not None:
        settings.datasets = [args.dataset]
    if args.split is not None:
        settings.splits = [args.split]
    if args.batch_size is not None:
        settings.batch_size = args.batch_size

    logger.info(
        "Настройки: datasets=%s, splits=%s, max_samples=%s, batch_size=%s",
        settings.datasets,
        settings.splits,
        settings.max_samples,
        settings.batch_size,
    )

    # 3. Создание раннера и запуск
    runner = ExperimentRunner(settings)

    # Фильтрация --experiment передаётся в run_all() — матрица фильтруется ДО выполнения
    results = await runner.run_all(experiment_filter=args.experiment)

    # 4. Сохранение
    json_path = runner.save_results(results)
    logger.info("Готово. Результаты: %s", json_path)


if __name__ == "__main__":
    asyncio.run(_main())
