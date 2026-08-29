#!/usr/bin/env python3
"""
Evaluate full-dataset (MMLongBench-Doc) strategy grid results.

Same evaluation pipeline as ``evaluate_strategies.py`` (MMLongDocEval
``eval_score`` + ``eval_acc_and_f1``), but using the FULL MMLongBench-Doc
``samples.json`` as ground truth and reading per-strategy results from
``mmlong_strategy_grid_results/``.

Produces (inside the results directory):
  - evaluation_report.txt          — plain-text summary
  - <name>_scored.json             — per-strategy results with scores
  - strategy_summary.json          — per-strategy metrics
  - strategy_comparison.html       — HTML comparison table

Usage:
    python app/tests/mmlong_evaluate_strategies.py
    python app/tests/mmlong_evaluate_strategies.py --results-dir path/to/results
    python app/tests/mmlong_evaluate_strategies.py --sort-by f1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the evaluation/report logic from the SmallerDataset evaluator so the
# scoring pipeline stays in sync. Only the ground-truth path and defaults differ.
from evaluate_strategies import (  # type: ignore[import]
    SORT_BY_CHOICES,
    evaluate_strategy,
    read_json,
    sort_strategies,
    write_html_report,
    write_json,
    write_text_report,
)

# Full MMLongBench-Doc ground truth (from mmlongdoceval_via_api.py).
SAMPLES_PATH = (
    "/home/sunveil/Documents/projects/laba/graph-m-rag/"
    "data/MMLongBench-Doc/data/samples.json"
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "mmlong_strategy_grid_results"

DATASET_NAME = "MMLongBench-Doc"


def _fix_html_dataset(path: str, gt_data: List[dict]) -> None:
    """Replace the SmallerDataset description baked into the shared HTML report."""
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        html = f.read()

    num_q = len(gt_data)
    num_docs = len({i.get("doc_id") for i in gt_data})
    num_types = len({i.get("doc_type") for i in gt_data})

    html = html.replace(
        "Dataset: <strong>SmallerDataset</strong>",
        f"Dataset: <strong>{DATASET_NAME}</strong>",
    )
    html = html.replace(
        "(12 documents, 105 questions, 5 types).",
        f"({num_docs} documents, {num_q} questions, {num_types} types).",
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate full-dataset strategy grid results from "
                    "mmlong_strategy_grid_results/",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Directory containing strategy JSON files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=SORT_BY_CHOICES,
        default="accuracy",
        help=(
            "Metric to sort strategies by. "
            f"Choices: {', '.join(SORT_BY_CHOICES)}. (default: accuracy)"
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    samples_path = SAMPLES_PATH

    # --- Load full-dataset ground truth ---
    if not os.path.exists(samples_path):
        print(f"ERROR: samples.json not found at {samples_path}")
        sys.exit(1)

    gt_data: List[dict] = read_json(samples_path)
    ground_truth_map: Dict[Tuple[str, str], dict] = {}
    for item in gt_data:
        key = (item.get("doc_id", ""), item.get("question", ""))
        ground_truth_map[key] = item

    print(
        f"Loaded {len(ground_truth_map)} ground-truth entries from "
        f"{samples_path}"
    )

    # --- Discover strategy result files ---
    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    result_files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
        and not f.endswith("_scored.json")
        and f not in ("strategy_summary.json",)
    )

    if not result_files:
        print(f"No JSON result files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} strategy result files\n")

    # --- Evaluate each strategy ---
    all_metrics: List[Dict[str, Any]] = []

    for fname in result_files:
        strategy_name = Path(fname).stem
        file_path = os.path.join(results_dir, fname)
        print(f"Evaluating: {strategy_name} ...")

        results: List[dict] = read_json(file_path)
        metrics = evaluate_strategy(results, ground_truth_map, strategy_name)

        # Save scored results
        scored_path = os.path.join(results_dir, f"{strategy_name}_scored.json")
        write_json(scored_path, results)
        print(f"  Scored results → {scored_path}")

        print(
            f"  Accuracy: {metrics['accuracy']}% "
            f"({metrics['correct']}/{metrics['scored']}), "
            f"F1: {metrics['f1']:.3f}, "
            f"Avg score: {metrics['avg_score']:.3f}, "
            f"Avg time: {metrics['avg_elapsed']:.1f}s"
        )

        all_metrics.append(metrics)

    # --- Generate reports ---
    print("")

    text_path = os.path.join(results_dir, "evaluation_report.txt")
    write_text_report(text_path, all_metrics, sort_by=args.sort_by)

    json_path = os.path.join(results_dir, "strategy_summary.json")
    write_json(json_path, all_metrics)
    print(f"  JSON summary → {json_path}")

    html_path = os.path.join(results_dir, "strategy_comparison.html")
    write_html_report(html_path, all_metrics, sort_by=args.sort_by)
    _fix_html_dataset(html_path, gt_data)

    # --- Print top strategies ---
    print("")
    print("=" * 72)
    print(f"  TOP STRATEGIES  (sorted by: {args.sort_by})")
    print("=" * 72)

    sorted_metrics = sort_strategies(all_metrics, sort_by=args.sort_by)
    for i, m in enumerate(sorted_metrics[:5]):
        priority = m["accuracy"] / (max(m.get("avg_elapsed", 1), 1) + 1)
        print(
            f"  {i + 1}. {m['name']:<30s} "
            f"{m['accuracy']:>5.1f}%  "
            f"(priority: {priority:.1f}, "
            f"correct: {m['correct']}/{m['scored']}, "
            f"F1: {m['f1']:.3f}, "
            f"time: {m['avg_elapsed']:.1f}s)"
        )
    print("=" * 72)
    print(f"\nFull report: {text_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
