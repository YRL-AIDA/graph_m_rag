#!/usr/bin/env python3
"""
Compare answers between two strategy grid result runs (v1 vs v2).

Uses the already-scored *_scored.json files from each directory.
Reads pred/score/llm_answer from the scored files and matches entries
by (doc_id, question) across runs.

Produces:
  - comparison_report.html  — per-strategy side-by-side tables
  - comparison_report.txt   — plain-text summary

Usage:
    python app/tests/compare_runs.py
    python app/tests/compare_runs.py -1 strategy_grid_results -2 strategy_grid_results_first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DIR_V1 = SCRIPT_DIR / "strategy_grid_results"
DIR_V2 = SCRIPT_DIR / "strategy_grid_result_full_second"
SAMPLES_PATH = SCRIPT_DIR / "SmallerDataset" / "samples.json"


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_strategies(
    strategy_name: str,
    v1_scored: List[dict],
    v2_scored: List[dict],
    ground_truth_map: Dict[Tuple[str, str], dict],
) -> Dict[str, Any]:
    """Compare per-question pred/score from two runs of the same strategy."""

    v1_by_key = {
        (r.get("doc_id", ""), r.get("question", "")): r
        for r in v1_scored
        if r.get("pred") is not None
    }
    v2_by_key = {
        (r.get("doc_id", ""), r.get("question", "")): r
        for r in v2_scored
        if r.get("pred") is not None
    }

    common_keys = sorted(set(v1_by_key.keys()) & set(v2_by_key.keys()))
    if not common_keys:
        return {}

    rows: List[Dict[str, Any]] = []
    scores_v1: List[float] = []
    scores_v2: List[float] = []

    for key in common_keys:
        doc_id, question = key
        gt_info = ground_truth_map.get(key, {})

        r1 = v1_by_key[key]
        r2 = v2_by_key[key]

        pred1 = str(r1.get("pred", "")) if r1.get("pred") is not None else ""
        pred2 = str(r2.get("pred", "")) if r2.get("pred") is not None else ""
        score1 = float(r1.get("score", 0) or 0)
        score2 = float(r2.get("score", 0) or 0)
        scores_v1.append(score1)
        scores_v2.append(score2)

        changed = abs(score1 - score2) > 0.01
        improved = (score2 - score1) > 0.01
        regressed = (score1 - score2) > 0.01

        # Truncate llm_answer for preview
        raw_v1 = r1.get("llm_answer", "") or ""
        raw_v2 = r2.get("llm_answer", "") or ""
        prev_v1 = raw_v1[:150].replace("\n", " ") if raw_v1 else ""
        prev_v2 = raw_v2[:150].replace("\n", " ") if raw_v2 else ""

        rows.append({
            "doc_id": doc_id,
            "question": question,
            "correct_answer": gt_info.get("answer", r1.get("answer", "")),
            "answer_format": gt_info.get("answer_format", "Str"),
            "pred_v1": pred1,
            "pred_v2": pred2,
            "score_v1": round(score1, 3),
            "score_v2": round(score2, 3),
            "delta": round(score2 - score1, 3),
            "changed": changed,
            "improved": improved,
            "regressed": regressed,
            "elapsed_v1": r1.get("elapsed", 0),
            "elapsed_v2": r2.get("elapsed", 0),
            "llm_preview_v1": prev_v1,
            "llm_preview_v2": prev_v2,
        })

    acc_v1 = round(sum(scores_v1) / len(scores_v1) * 100, 1) if scores_v1 else 0.0
    acc_v2 = round(sum(scores_v2) / len(scores_v2) * 100, 1) if scores_v2 else 0.0

    return {
        "name": strategy_name,
        "rows": rows,
        "common_count": len(common_keys),
        "accuracy_v1": acc_v1,
        "accuracy_v2": acc_v2,
        "acc_delta": round(acc_v2 - acc_v1, 1),
        "improved": sum(1 for r in rows if r["improved"]),
        "regressed": sum(1 for r in rows if r["regressed"]),
        "unchanged": sum(1 for r in rows if not r["changed"]),
        "avg_elapsed_v1": round(
            sum(r["elapsed_v1"] for r in rows) / len(rows), 1
        ) if rows else 0,
        "avg_elapsed_v2": round(
            sum(r["elapsed_v2"] for r in rows) / len(rows), 1
        ) if rows else 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_html_report(path: str, comparisons: List[Dict[str, Any]]) -> None:
    if not comparisons:
        return

    def _esc(text: str) -> str:
        return str(text).replace("&", "&").replace("<", "<").replace(">", ">")

    def _ell(text: str, n: int) -> str:
        t = str(text)
        return t if len(t) <= n else t[:n - 1] + "…"

    html_parts: List[str] = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Grid: v1 vs v2 Comparison</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1400px; margin: 40px auto; padding: 0 20px; color: #333; }
  h1 { border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
  h2 { color: #2c3e50; margin-top: 30px; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  th { background: #2c3e50; color: white; padding: 10px 8px; text-align: left;
        font-size: 0.9em; white-space: nowrap; }
  td { padding: 8px; border-bottom: 1px solid #ddd; font-size: 0.82em;
        vertical-align: top; }
  tr:hover { filter: brightness(0.95); }
  .improved { color: #155724; font-weight: bold; }
  .regressed { color: #721c24; font-weight: bold; }
  .neutral { color: #666; }
  .improved-bg { background: #d4edda; }
  .regressed-bg { background: #f8d7da; }
  .footer { margin-top: 30px; font-size: 0.8em; color: #888;
             border-top: 1px solid #eee; padding-top: 10px; }
  .details { margin-top: 20px; }
  .details summary { cursor: pointer; font-weight: bold; font-size: 1.1em;
                      margin-bottom: 10px; }
  .acc-up { color: #28a745; font-weight: bold; }
  .acc-down { color: #dc3545; font-weight: bold; }
  .pred-cell { max-width: 280px; word-break: break-word; font-size: 0.78em; }
  .q-cell { max-width: 280px; font-size: 0.78em; }
</style>
</head>
<body>
<h1>Strategy Grid — v1 vs v2 Per-Question Comparison</h1>
<p><strong>v1:</strong> strategy_grid_results &emsp;
<strong>v2:</strong> strategy_grid_results_first</p>
""")

    # --- Summary table ---
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<table><thead><tr>")
    for col_h in ["Strategy", "Q", "Acc v1", "Acc v2", "Δ Acc",
                   "↑ Improved", "↓ Regressed", "= Same", "Time v1", "Time v2"]:
        html_parts.append(f"<th>{col_h}</th>")
    html_parts.append("</tr></thead><tbody>")

    comparisons_sorted = sorted(comparisons, key=lambda c: c["acc_delta"], reverse=True)
    for c in comparisons_sorted:
        dclass = "acc-up" if c["acc_delta"] > 0 else ("acc-down" if c["acc_delta"] < 0 else "neutral")
        dsign = "+" if c["acc_delta"] > 0 else ""
        html_parts.append(
            f"<tr>"
            f"<td><strong>{c['name']}</strong></td>"
            f"<td style=\"text-align:right\">{c['common_count']}</td>"
            f"<td style=\"text-align:right\">{c['accuracy_v1']}%</td>"
            f"<td style=\"text-align:right\">{c['accuracy_v2']}%</td>"
            f"<td style=\"text-align:right;\" class=\"{dclass}\">{dsign}{c['acc_delta']}%</td>"
            f"<td style=\"text-align:right;\" class=\"improved\">{c['improved']}</td>"
            f"<td style=\"text-align:right;\" class=\"regressed\">{c['regressed']}</td>"
            f"<td style=\"text-align:right;\">{c['unchanged']}</td>"
            f"<td style=\"text-align:right;\">{c['avg_elapsed_v1']}s</td>"
            f"<td style=\"text-align:right;\">{c['avg_elapsed_v2']}s</td>"
            f"</tr>"
        )
    html_parts.append("</tbody></table>")

    # --- Per-strategy detail ---
    html_parts.append("<h2>Per-Strategy Detail</h2>")
    for c in comparisons_sorted:
        rows = c["rows"]
        if not rows:
            continue
        rows_by_delta = sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)
        ad = f"+{c['acc_delta']}%" if c['acc_delta'] > 0 else f"{c['acc_delta']}%"
        html_parts.append(
            f"<details class=\"details\"><summary>"
            f"{c['name']} — Acc: {c['accuracy_v1']}% → {c['accuracy_v2']}% "
            f"(Δ {ad}), ↑{c['improved']} ↓{c['regressed']} ={c['unchanged']}"
            f"</summary>"
        )
        html_parts.append("<table><thead><tr>")
        for col_h in ["#", "Doc", "Question", "Correct",
                       "v1 Predicted", "v2 Predicted",
                       "Score v1", "Score v2", "Δ"]:
            html_parts.append(f"<th>{col_h}</th>")
        html_parts.append("</tr></thead><tbody>")

        for idx, r in enumerate(rows_by_delta, 1):
            if r["improved"]:
                bg = "improved-bg"
            elif r["regressed"]:
                bg = "regressed-bg"
            else:
                bg = ""
            dclass = "improved" if r["improved"] else ("regressed" if r["regressed"] else "neutral")
            dsign = "+" if r["delta"] > 0 else ""

            html_parts.append(
                f"<tr class=\"{bg}\">"
                f"<td>{idx}</td>"
                f"<td style=\"font-size:0.72em\">{_ell(r['doc_id'], 30)}</td>"
                f"<td class=\"q-cell\">{_esc(_ell(r['question'], 100))}</td>"
                f"<td style=\"font-size:0.78em\">{_esc(_ell(r['correct_answer'], 80))}</td>"
                f"<td class=\"pred-cell\">{_esc(_ell(r['pred_v1'], 200))}</td>"
                f"<td class=\"pred-cell\">{_esc(_ell(r['pred_v2'], 200))}</td>"
                f"<td style=\"text-align:right\">{r['score_v1']:.2f}</td>"
                f"<td style=\"text-align:right\">{r['score_v2']:.2f}</td>"
                f"<td style=\"text-align:right;\" class=\"{dclass}\">{dsign}{r['delta']:.2f}</td>"
                f"</tr>"
            )
        html_parts.append("</tbody></table>")
        html_parts.append("</details>")

    html_parts.append("""
<div class="footer">
  Generated by <code>compare_runs.py</code> —
  compares <code>strategy_grid_results</code> (v1) with
  <code>strategy_grid_results_first</code> (v2).
  Uses pre-scored *_scored.json files. Green = score improved in v2; red = regressed.
</div>
</body>
</html>""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"  HTML report → {path}")


def write_text_report(path: str, comparisons: List[Dict[str, Any]]) -> None:
    if not comparisons:
        return

    lines: List[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  STRATEGY GRID COMPARISON — v1 (strategy_grid_results) vs v2 (strategy_grid_results_first)")
    lines.append(sep)
    lines.append("")

    comparisons_sorted = sorted(comparisons, key=lambda c: c["acc_delta"], reverse=True)
    header = (
        f" {'Strategy':<20} {'Q':>4} {'Acc v1':>7} {'Acc v2':>7} {'Δ':>7} "
        f"{'↑':>5} {'↓':>5} {'=':>5} {'T v1':>6} {'T v2':>6}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for c in comparisons_sorted:
        dsign = "+" if c["acc_delta"] > 0 else ""
        lines.append(
            f" {c['name']:<20} {c['common_count']:>4} "
            f"{c['accuracy_v1']:>6.1f}% {c['accuracy_v2']:>6.1f}% "
            f"{dsign}{c['acc_delta']:>6.1f}% "
            f"{c['improved']:>5} {c['regressed']:>5} {c['unchanged']:>5} "
            f"{c['avg_elapsed_v1']:>5.1f}s {c['avg_elapsed_v2']:>5.1f}s"
        )

    lines.append("")
    lines.append(sep)
    lines.append("  Per-Strategy Detail (top changes first)")
    lines.append(sep)
    lines.append("")

    for c in comparisons_sorted:
        rows = sorted(c["rows"], key=lambda r: abs(r["delta"]), reverse=True)
        lines.append(
            f"--- {c['name']}  ({c['accuracy_v1']}% → {c['accuracy_v2']}%, "
            f"↑{c['improved']} ↓{c['regressed']} ={c['unchanged']}) ---"
        )
        for idx, r in enumerate(rows, 1):
            flag = " ↑" if r["improved"] else (" ↓" if r["regressed"] else "  ")
            lines.append(
                f"  {idx:<3}{flag} Score: {r['score_v1']:.2f}→{r['score_v2']:.2f} "
                f"(Δ{r['delta']:+.2f}) | Q: {r['question'][:60]}"
            )
            lines.append(f"       Correct : {r['correct_answer']}")
            lines.append(f"       v1      : {r['pred_v1'][:120]}")
            lines.append(f"       v2      : {r['pred_v2'][:120]}")
            lines.append("")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Text report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scored answers from two strategy-grid runs",
    )
    parser.add_argument(
        "-1", "--dir-v1", default=str(DIR_V1),
        help=f"First results directory (default: {DIR_V1})",
    )
    parser.add_argument(
        "-2", "--dir-v2", default=str(DIR_V2),
        help=f"Second results directory (default: {DIR_V2})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for reports (default: dir-v1)",
    )
    args = parser.parse_args()

    dir_v1 = args.dir_v1
    dir_v2 = args.dir_v2
    out_dir = args.out_dir or dir_v1

    if not os.path.isdir(dir_v1):
        print(f"ERROR: v1 directory not found: {dir_v1}")
        sys.exit(1)
    if not os.path.isdir(dir_v2):
        print(f"ERROR: v2 directory not found: {dir_v2}")
        sys.exit(1)

    # Load ground truth for display
    gt_data: List[dict] = []
    if os.path.exists(str(SAMPLES_PATH)):
        gt_data = read_json(str(SAMPLES_PATH))
    ground_truth_map: Dict[Tuple[str, str], dict] = {}
    for item in gt_data:
        key = (item.get("doc_id", ""), item.get("question", ""))
        ground_truth_map[key] = item
    print(f"Loaded {len(ground_truth_map)} ground-truth entries")

    # Find overlapping SCORED strategy files
    def _scored_files(dirpath: str) -> set:
        return {
            Path(f).stem.replace("_scored", "")
            for f in os.listdir(dirpath)
            if f.endswith("_scored.json")
            and f != "strategy_summary.json"
        }

    v1_strats = _scored_files(dir_v1)
    v2_strats = _scored_files(dir_v2)
    overlapping = sorted(v1_strats & v2_strats)

    if not overlapping:
        print("No overlapping scored strategy files found.")
        sys.exit(0)

    print(f"Overlapping strategies ({len(overlapping)}): {overlapping}")
    print()

    comparisons: List[Dict[str, Any]] = []
    for strat in overlapping:
        print(f"Comparing: {strat} ...")
        v1_path = os.path.join(dir_v1, f"{strat}_scored.json")
        v2_path = os.path.join(dir_v2, f"{strat}_scored.json")

        if not os.path.exists(v1_path) or not os.path.exists(v2_path):
            print(f"  Skipping — missing scored file for {strat}")
            continue

        v1_data: List[dict] = read_json(v1_path)
        v2_data: List[dict] = read_json(v2_path)
        result = compare_strategies(strat, v1_data, v2_data, ground_truth_map)
        if result:
            print(
                f"  {result['common_count']} questions | "
                f"Acc: {result['accuracy_v1']}% → {result['accuracy_v2']}% "
                f"(Δ {result['acc_delta']:+.1f}%) | "
                f"↑{result['improved']} ↓{result['regressed']} ={result['unchanged']}"
            )
            comparisons.append(result)
        else:
            print("  No comparable questions found.")
        print()

    if not comparisons:
        print("No comparable data found.")
        sys.exit(0)

    text_path = os.path.join(out_dir, "comparison_report.txt")
    write_text_report(text_path, comparisons)

    html_path = os.path.join(out_dir, "comparison_report.html")
    write_html_report(html_path, comparisons)

    print(f"\nReports saved to {out_dir}/")


if __name__ == "__main__":
    main()
