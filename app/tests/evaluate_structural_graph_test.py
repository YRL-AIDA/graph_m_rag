#!/usr/bin/env python3
"""
Evaluate all structural graph test results from structural_graph_results/ directory.

Computes per-strategy metrics using the official eval_score / eval_acc_and_f1
from MMLongDocEval — exactly the same pipeline as evaluate_strategies.py.

Produces:
  - A plain-text summary report     → structural_graph_results/evaluation_report.txt
  - A per-strategy scored JSON      → structural_graph_results/<name>_scored.json
  - A strategy comparison summary   → structural_graph_results/strategy_summary.json
  - An HTML comparison table        → structural_graph_results/strategy_comparison.html

Usage:
    python app/tests/evaluate_structural_graph_test.py
    python app/tests/evaluate_structural_graph_test.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path resolution & imports  (mirrors evaluate_strategies.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "structural_graph_results"
SAMPLES_PATH = SCRIPT_DIR / "SmallerDataset" / "samples.json"

sys.path.insert(0, str(SCRIPT_DIR))
from MMLongDocEval.eval_score import (  # type: ignore[import]
    eval_score,
    eval_acc_and_f1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Answer extraction  (identical to evaluate_strategies.py)
# ---------------------------------------------------------------------------

# Pattern to strip  ...  thinking blocks
_THINK_PATTERN = re.compile(r".*?", re.DOTALL)

# Try explicit "Extracted answer:" / "Answer:" lines
_EXTRACTED_PATTERN = re.compile(
    r"(?:Extracted\s+answer|Answer)\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def extract_pred_from_extracted_res(extracted_res: str) -> str:
    """Mirrors the extraction in smallerdataset_evaluation.py.

    Input example:
        "Answer format: Str\nExtracted answer: 42"
    Returns "42", or "Failed to extract" on failure.

    Uses rsplit to find the LAST "Extracted answer:" marker, avoiding
    accidental extraction of template placeholders like "[answer]" from
    the system prompt instructions that also contain these markers.
    """
    try:
        if "Extracted answer:" in extracted_res:
            # rsplit from the right to skip any template placeholder
            # occurrences of "Extracted answer:" and "Answer format:"
            after_extracted = extracted_res.rsplit("Extracted answer:", 1)[-1]
            pred = after_extracted.split("Answer format:")[0].strip()
            if pred:
                return pred
        return "Failed to extract"
    except (IndexError, AttributeError):
        return "Failed to extract"


def extract_pred_from_llm_answer(llm_answer: str) -> str:
    """Heuristic fallback: extract a short answer from raw LLM output.

    Strips  ...  blocks, then looks for explicit answer markers,
    then falls back to the last non-empty line.
    """
    if not llm_answer:
        return ""

    # Remove thinking block
    text = _THINK_PATTERN.sub("", llm_answer).strip()

    # Try explicit markers
    m = _EXTRACTED_PATTERN.search(text)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) < 200 or candidate.startswith("["):
            return candidate

    # Fallback: last non-empty line
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Per-strategy evaluation  (identical to evaluate_strategies.py)
# ---------------------------------------------------------------------------

def evaluate_strategy(
    results: List[dict],
    ground_truth_map: dict,
    strategy_name: str,
) -> Dict[str, Any]:
    """
    Score every result in a strategy using the same pipeline as
    smallerdataset_evaluation.py and compute breakdown metrics.

    Returns a dict with keys:
      name, total, completed, scored, accuracy, f1, avg_score,
      correct, avg_elapsed,
      single_page_acc, cross_page_acc, unanswerable_acc,
      by_source, by_doc_type
    """
    scored_samples: List[dict] = []  # samples that got a score
    elapsed_times: List[float] = []

    for r in results:
        if r.get("status") != "completed":
            continue
        if not r.get("llm_answer"):
            continue

        doc_id = r.get("doc_id", "")
        question = r.get("question", "")
        gt_info = ground_truth_map.get((doc_id, question), {})

        # --- Extract predicted answer ---
        extracted_res = r.get("extracted_res")
        if extracted_res is not None and str(extracted_res) not in ("None", ""):
            pred = extract_pred_from_extracted_res(str(extracted_res))
        else:
            llm_answer = r.get("llm_answer", "")
            pred = extract_pred_from_llm_answer(llm_answer)
            if not pred:
                continue

        # --- Score via eval_score ---
        correct_answer = gt_info.get("answer", r.get("answer", ""))
        answer_format = gt_info.get("answer_format", "Str")

        try:
            score = float(eval_score(correct_answer, pred, answer_format))
        except Exception:
            score = 0.0

        r["pred"] = pred
        r["score"] = score
        scored_samples.append(r)

        if r.get("elapsed"):
            elapsed_times.append(float(r["elapsed"]))

    # --- Populate evidence metadata for breakdown ---
    for s in scored_samples:
        doc_id = s.get("doc_id", "")
        question = s.get("question", "")
        gt_info = ground_truth_map.get((doc_id, question), {})

        try:
            s["evidence_pages"] = eval(str(gt_info.get("evidence_pages", "[]")))
        except Exception:
            s["evidence_pages"] = []

        try:
            s["evidence_sources"] = eval(str(gt_info.get("evidence_sources", "[]")))
        except Exception:
            s["evidence_sources"] = []

        s["answer"] = gt_info.get("answer", s.get("answer", ""))
        s["answer_format"] = gt_info.get("answer_format", s.get("answer_format", "Str"))
        s["doc_type"] = gt_info.get("doc_type", "Unknown")

    # --- Compute metrics via eval_acc_and_f1 ---
    acc, f1 = eval_acc_and_f1(scored_samples)
    acc_pct = round(acc * 100, 1)
    f1_val = round(f1, 3)

    avg_score = acc
    correct_count = sum(1 for s in scored_samples if s.get("score", 0) >= 1.0)
    avg_elapsed = (
        sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    )

    # --- Breakdown by page type ---
    single = [
        s for s in scored_samples
        if len(s.get("evidence_pages", [])) == 1
    ]
    cross = [
        s for s in scored_samples
        if len(s.get("evidence_pages", [])) != 1
        and str(s.get("answer", "")) != "Not answerable"
    ]
    unans = [
        s for s in scored_samples
        if str(s.get("answer", "")) == "Not answerable"
    ]

    single_acc, _ = eval_acc_and_f1(single)
    cross_acc, _ = eval_acc_and_f1(cross)
    unans_acc, _ = eval_acc_and_f1(unans)

    # --- Breakdown by evidence source ---
    source_dict: Dict[str, List[dict]] = defaultdict(list)
    for s in scored_samples:
        for src in s.get("evidence_sources", []):
            source_dict[src].append(s)

    by_source = {}
    for src, items in sorted(source_dict.items()):
        src_acc, _ = eval_acc_and_f1(items)
        by_source[src] = {
            "accuracy": round(src_acc, 3),
            "count": len(items),
        }

    # --- Breakdown by document type ---
    doctype_dict: Dict[str, List[dict]] = defaultdict(list)
    for s in scored_samples:
        doctype_dict[s.get("doc_type", "Unknown")].append(s)

    by_doc_type = {}
    for dt, items in sorted(doctype_dict.items()):
        dt_acc, _ = eval_acc_and_f1(items)
        by_doc_type[dt] = {
            "accuracy": round(dt_acc, 3),
            "count": len(items),
        }

    return {
        "name": strategy_name,
        "total": len(results),
        "completed": len([r for r in results if r.get("llm_answer")]),
        "scored": len(scored_samples),
        "accuracy": acc_pct,
        "f1": f1_val,
        "avg_score": round(avg_score, 3),
        "correct": correct_count,
        "avg_elapsed": round(avg_elapsed, 1),
        "single_page_acc": round(single_acc, 3),
        "single_page_count": len(single),
        "cross_page_acc": round(cross_acc, 3),
        "cross_page_count": len(cross),
        "unanswerable_acc": round(unans_acc, 3),
        "unanswerable_count": len(unans),
        "by_source": by_source,
        "by_doc_type": by_doc_type,
    }


# ---------------------------------------------------------------------------
# Report generation  (adapted for structural graph evaluation)
# ---------------------------------------------------------------------------

def write_text_report(
    path: str,
    metrics_list: List[Dict[str, Any]],
):
    """Write a plain-text evaluation report."""
    if not metrics_list:
        return

    lines: List[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  GRAPH M-RAG  —  Structural Graph Evaluation Report")
    lines.append(sep)
    lines.append("")

    # Overall ranking table
    sorted_metrics = sorted(
        metrics_list, key=lambda m: m["accuracy"], reverse=True
    )
    header = (
        f" {'Rank':<5} {'Strategy':<35} {'Acc%':>7} {'F1':>7} "
        f"{'Corr':>6} {'Scored':>6} {'Time':>6}"
    )
    lines.append(header)
    lines.append("-" * 72)

    for i, m in enumerate(sorted_metrics):
        line = (
            f" {i + 1:<5} {m['name']:<35} "
            f"{m['accuracy']:>6.1f}% {m['f1']:>6.3f} "
            f"{m['correct']:>5}/{m['scored']:<5} {m['avg_elapsed']:>5.1f}s"
        )
        lines.append(line)

    lines.append("")
    lines.append(sep)
    lines.append("  Detailed breakdown (all strategies)")
    lines.append(sep)
    lines.append("")

    for m in sorted_metrics:
        lines.append(
            f"--- {m['name']}  "
            f"(Accuracy: {m['accuracy']}%, F1: {m['f1']:.3f}, "
            f"Correct: {m['correct']}/{m['scored']}) ---"
        )
        lines.append(
            f"  Single-page:   {m['single_page_acc']:.3f}  "
            f"(n={m['single_page_count']})"
        )
        lines.append(
            f"  Cross-page:    {m['cross_page_acc']:.3f}  "
            f"(n={m['cross_page_count']})"
        )
        lines.append(
            f"  Unanswerable:  {m['unanswerable_acc']:.3f}  "
            f"(n={m['unanswerable_count']})"
        )

        if m.get("by_source"):
            lines.append("  By evidence source:")
            for src, info in m["by_source"].items():
                lines.append(
                    f"    {src:<40s} {info['accuracy']:.3f} "
                    f"(n={info['count']})"
                )

        if m.get("by_doc_type"):
            lines.append("  By document type:")
            for dt, info in m["by_doc_type"].items():
                lines.append(
                    f"    {dt:<40s} {info['accuracy']:.3f} "
                    f"(n={info['count']})"
                )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Text report → {path}")


def write_html_report(
    path: str,
    metrics_list: List[Dict[str, Any]],
):
    """Write a self-contained HTML comparison table."""
    if not metrics_list:
        return

    sorted_metrics = sorted(
        metrics_list, key=lambda m: m["accuracy"], reverse=True
    )
    max_acc = (
        max(m["accuracy"] for m in sorted_metrics)
        if sorted_metrics
        else 100.0
    )

    def row_color(acc: float) -> str:
        ratio = acc / max(max_acc, 0.01)
        if ratio >= 0.9:
            return "#d4edda"
        if ratio >= 0.7:
            return "#fff3cd"
        if ratio >= 0.5:
            return "#ffeeba"
        return "#f8d7da"

    table_rows = ""
    for i, m in enumerate(sorted_metrics):
        medal = ""
        if i == 0:
            medal = "  🥇"
        elif i == 1:
            medal = "  🥈"
        elif i == 2:
            medal = "  🥉"

        rb = row_color(m["accuracy"])
        table_rows += f"""
        <tr style="background:{rb}">
          <td style="text-align:right">{i + 1}</td>
          <td><strong>{m['name']}{medal}</strong></td>
          <td style="text-align:right"><strong>{m['accuracy']}%</strong></td>
          <td style="text-align:right">{m['f1']:.3f}</td>
          <td style="text-align:right">{m['correct']}/{m['scored']}</td>
          <td style="text-align:right">{m['avg_elapsed']:.1f}s</td>
          <td style="text-align:right">{m['single_page_acc']:.3f}</td>
          <td style="text-align:right">{m['cross_page_acc']:.3f}</td>
          <td style="text-align:right">{m['unanswerable_acc']:.3f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Graph M-RAG — Structural Graph Evaluation</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1300px; margin: 40px auto; padding: 0 20px; color: #333;
  }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left;
        font-size: 0.9em; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
  tr:hover {{ filter: brightness(0.95); }}
  .legend {{ display: flex; gap: 15px; margin: 15px 0; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 20px; height: 20px; border-radius: 3px;
                    border: 1px solid #ccc; }}
  .footer {{ margin-top: 30px; font-size: 0.8em; color: #888;
             border-top: 1px solid #eee; padding-top: 10px; }}
  code {{ font-size: 0.8em; color: #555; background: #f5f5f5;
          padding: 1px 4px; border-radius: 3px; }}
  .details {{ margin-top: 20px; }}
  .details summary {{ cursor: pointer; font-weight: bold; font-size: 1.1em;
                      margin-bottom: 10px; }}
</style>
</head>
<body>
<h1>Graph M-RAG &mdash; Structural Graph Evaluation</h1>
<p>
  Dataset: <strong>SmallerDataset</strong>
  (12 documents, 105 questions, 5 types).
  Evaluates strategies that use <code>use_structured_graph=True</code>
  including the <code>use_structural_parent_only</code> parent-only walk variant.
  Scores computed with <code>eval_score</code> + <code>eval_acc_and_f1</code>
  using proper <code>answer_format</code> (Int, Float, Str, List, None).
</p>

<div class="legend">
  <div class="legend-item">
    <div class="legend-swatch" style="background:#d4edda"></div> ≥90% of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#fff3cd"></div> ≥70% of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#ffeeba"></div> ≥50% of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#f8d7da"></div> <50% of best
  </div>
</div>

<h2>Summary Table</h2>
<table>
<thead>
<tr>
  <th>#</th><th>Strategy</th><th>Accuracy</th><th>F1</th>
  <th>Correct</th><th>Avg Time</th><th>Single-Page</th><th>Cross-Page</th>
  <th>Unanswerable</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Per-Strategy Details</h2>
"""

    for m in sorted_metrics:
        html += f"""
<details class="details">
<summary>{m['name']} — Accuracy: {m['accuracy']}%, F1: {m['f1']:.3f}
({m['correct']}/{m['scored']})</summary>
<table>
<tr><th>Metric</th><th>Value</th><th>Count</th></tr>
<tr><td>Single-page</td><td>{m['single_page_acc']:.3f}</td>
    <td>{m['single_page_count']}</td></tr>
<tr><td>Cross-page</td><td>{m['cross_page_acc']:.3f}</td>
    <td>{m['cross_page_count']}</td></tr>
<tr><td>Unanswerable</td><td>{m['unanswerable_acc']:.3f}</td>
    <td>{m['unanswerable_count']}</td></tr>
</table>"""

        if m.get("by_source"):
            html += (
                "<table><tr><th>Evidence Source</th>"
                "<th>Accuracy</th><th>Count</th></tr>"
            )
            for src, info in m["by_source"].items():
                html += (
                    f"<tr><td>{src}</td><td>{info['accuracy']:.3f}</td>"
                    f"<td>{info['count']}</td></tr>"
                )
            html += "</table>"

        if m.get("by_doc_type"):
            html += (
                "<table><tr><th>Document Type</th>"
                "<th>Accuracy</th><th>Count</th></tr>"
            )
            for dt, info in m["by_doc_type"].items():
                html += (
                    f"<tr><td>{dt}</td><td>{info['accuracy']:.3f}</td>"
                    f"<td>{info['count']}</td></tr>"
                )
            html += "</table>"

        html += "</details>\n"

    html += """
<div class="footer">
  Generated by <code>evaluate_structural_graph_test.py</code> —
  uses MMLongDocEval eval_score + eval_acc_and_f1 pipeline.
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  HTML report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate structural graph test results from structural_graph_results/",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Directory containing structural graph strategy JSON files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    samples_path = str(SAMPLES_PATH)

    # --- Load ground truth ---
    if not os.path.exists(samples_path):
        print(f"ERROR: samples.json not found at {samples_path}")
        sys.exit(1)

    gt_data: List[dict] = read_json(samples_path)
    ground_truth_map: Dict[Tuple[str, str], dict] = {}
    for item in gt_data:
        key = (item.get("doc_id", ""), item.get("question", ""))
        ground_truth_map[key] = item

    print(f"Loaded {len(ground_truth_map)} ground-truth entries from samples.json")

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
    write_text_report(text_path, all_metrics)

    json_path = os.path.join(results_dir, "strategy_summary.json")
    write_json(json_path, all_metrics)
    print(f"  JSON summary → {json_path}")

    html_path = os.path.join(results_dir, "strategy_comparison.html")
    write_html_report(html_path, all_metrics)

    # --- Print top strategies ---
    print("")
    print("=" * 72)
    print("  TOP STRUCTURAL GRAPH STRATEGIES")
    print("=" * 72)

    sorted_metrics = sorted(
        all_metrics, key=lambda m: m["accuracy"], reverse=True
    )
    for i, m in enumerate(sorted_metrics[:6]):
        print(
            f"  {i + 1}. {m['name']:<35s} "
            f"{m['accuracy']:>5.1f}%  "
            f"(correct: {m['correct']}/{m['scored']}, "
            f"F1: {m['f1']:.3f}, "
            f"time: {m['avg_elapsed']:.1f}s)"
        )
    print("=" * 72)
    print(f"\nFull report: {text_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
