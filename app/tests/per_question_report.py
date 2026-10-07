#!/usr/bin/env python3
"""
Generate a per-question HTML report for strategy_grid_results.

Each row is one (doc_id, question) pair. Columns show each strategy's
predicted answer and score, so all strategies' responses to the same
question are visible side-by-side.

Output: strategy_grid_results/per_question_report.html

Usage:
    python app/tests/per_question_report.py
    python app/tests/per_question_report.py -d strategy_grid_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR / "strategy_grid_results"
SAMPLES_PATH = SCRIPT_DIR / "SmallerDataset" / "samples.json"


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored_data(results_dir: str) -> Dict[str, Dict[Tuple[str, str], dict]]:
    """Load all *_scored.json files from results_dir.

    Returns: {strategy_name: {(doc_id, question): entry_dict}}
    where entry_dict has keys: pred, score, elapsed, llm_answer
    """
    strategy_data: Dict[str, Dict[Tuple[str, str], dict]] = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_scored.json"):
            continue
        strategy_name = fname.replace("_scored.json", "")
        if strategy_name == "strategy_summary":
            continue

        path = os.path.join(results_dir, fname)
        entries = read_json(path)

        by_question: Dict[Tuple[str, str], dict] = {}
        for entry in entries:
            key = (entry.get("doc_id", ""), entry.get("question", ""))
            by_question[key] = {
                "pred": str(entry.get("pred", "")),
                "score": float(entry.get("score", 0) or 0),
                "elapsed": entry.get("elapsed", 0),
                "llm_preview": (entry.get("llm_answer", "") or "")[:200],
            }
        strategy_data[strategy_name] = by_question

    return strategy_data


def load_ground_truth() -> Dict[Tuple[str, str], dict]:
    if not os.path.exists(str(SAMPLES_PATH)):
        return {}
    gt_items = read_json(str(SAMPLES_PATH))
    return {
        (item.get("doc_id", ""), item.get("question", "")): item
        for item in gt_items
    }


def load_strategy_summary(results_dir: str) -> List[Dict[str, Any]]:
    """Load strategy_summary.json for overall stats."""
    summary_path = os.path.join(results_dir, "strategy_summary.json")
    if not os.path.exists(summary_path):
        return []
    return read_json(summary_path)


# ---------------------------------------------------------------------------
# Pivot: merge all strategies into per-question rows
# ---------------------------------------------------------------------------

def build_per_question_rows(
    strategy_data: Dict[str, Dict[Tuple[str, str], dict]],
    ground_truth: Dict[Tuple[str, str], dict],
) -> Tuple[
    List[Dict[str, Any]],          # per_question rows
    List[str],                     # strategy names (column order)
    Dict[str, Dict[str, Any]],     # per-strategy aggregate stats
]:
    strategy_names = sorted(strategy_data.keys())

    # Collect all unique (doc_id, question) keys across all strategies
    all_keys: set = set()
    for sdata in strategy_data.values():
        all_keys.update(sdata.keys())

    # Group by doc_id for sectioning
    keys_by_doc: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for doc_id, q in all_keys:
        keys_by_doc[doc_id].append((doc_id, q))

    # Sort docs, then sort questions within each doc
    sorted_docs = sorted(keys_by_doc.keys())
    ordered_keys: List[Tuple[str, str]] = []
    for doc_id in sorted_docs:
        ordered_keys.extend(sorted(keys_by_doc[doc_id], key=lambda k: k[1]))

    # Build per-question rows
    rows: List[Dict[str, Any]] = []

    # Per-strategy aggregate counters
    strat_stats: Dict[str, Dict[str, Any]] = {
        s: {"correct": 0, "wrong": 0, "total": 0, "scores": []}
        for s in strategy_names
    }

    for idx, key in enumerate(ordered_keys, 1):
        doc_id, question = key
        gt = ground_truth.get(key, {})
        correct_answer = gt.get("answer", "")
        answer_format = gt.get("answer_format", "Str")
        doc_type = gt.get("doc_type", "")

        row: Dict[str, Any] = {
            "idx": idx,
            "doc_id": doc_id,
            "question": question,
            "correct_answer": correct_answer,
            "answer_format": answer_format,
            "doc_type": doc_type,
        }

        for sname in strategy_names:
            sentry = strategy_data[sname].get(key, {})
            pred = sentry.get("pred", "")
            score = sentry.get("score", 0.0)
            elapsed = sentry.get("elapsed", 0)
            llm_preview = sentry.get("llm_preview", "")

            if sentry:  # only count if this strategy has this question
                strat_stats[sname]["total"] += 1
                strat_stats[sname]["scores"].append(score)
                if score >= 0.99:
                    strat_stats[sname]["correct"] += 1
                elif score <= 0.01:
                    strat_stats[sname]["wrong"] += 1

            row[sname] = {
                "pred": pred,
                "score": score,
                "elapsed": elapsed,
                "llm_preview": llm_preview,
            }

        rows.append(row)

    # Compute accuracy per strategy
    for sname in strategy_names:
        scores = strat_stats[sname]["scores"]
        strat_stats[sname]["accuracy"] = (
            round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
        )

    return rows, strategy_names, strat_stats


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html(
    path: str,
    rows: List[Dict[str, Any]],
    strategy_names: List[str],
    strat_stats: Dict[str, Dict[str, Any]],
) -> None:
    def _esc(text: str) -> str:
        return str(text).replace("&", "&").replace("<", "<").replace(">", ">")

    def _cell_class(score: float) -> str:
        if score >= 0.99:
            return "score-good"
        elif score <= 0.01:
            return "score-bad"
        else:
            return "score-partial"

    def _score_label(score: float) -> str:
        if score >= 0.99:
            return "&#10003;"
        elif score <= 0.01:
            return "&#10007;"
        else:
            return f"{score:.2f}"

    html: List[str] = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-Question Strategy Report</title>
<style>
  :root {
    --good: #d4edda; --good-text: #155724;
    --bad: #f8d7da; --bad-text: #721c24;
    --partial: #fff3cd; --partial-text: #856404;
    --header: #2c3e50;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; padding: 0 20px 40px; color: #333; background: #fafafa; }
  h1 { padding: 20px 0 10px; border-bottom: 3px solid var(--header); }
  h2 { color: var(--header); margin: 25px 0 10px; }

  .summary-bar { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }
  .summary-card {
    flex: 0 0 180px; background: white; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid var(--header);
  }
  .summary-card .strat-name { font-weight: 700; font-size: 0.85em; color: #555; }
  .summary-card .acc { font-size: 1.6em; font-weight: 700; margin: 4px 0; }
  .summary-card .detail { font-size: 0.75em; color: #888; }
  .acc-high { color: var(--good-text); }
  .acc-mid { color: var(--partial-text); }
  .acc-low { color: var(--bad-text); }

  .controls { margin: 15px 0; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .controls button, .controls select {
    padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
    background: white; cursor: pointer; font-size: 0.85em;
  }
  .controls button:hover { background: #e9ecef; }
  .controls button.active { background: var(--header); color: white; border-color: var(--header); }

  table { border-collapse: collapse; width: max-content; min-width: 100%; margin: 15px 0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); background: white;
          border-radius: 6px; overflow: hidden; }
  th { background: var(--header); color: white; padding: 8px 6px; text-align: left;
        font-size: 0.78em; white-space: nowrap; position: sticky; top: 0; z-index: 2; }
  th.q-col { min-width: 280px; }
  th.pred-col, th.score-col { min-width: 90px; text-align: center; }
  td { padding: 5px 6px; border-bottom: 1px solid #eee; font-size: 0.76em;
        vertical-align: top; }
  tr:hover { background: #f8f9fa !important; }
  tr:nth-child(even) { background: #fdfdfd; }

  .score-good { background: var(--good); color: var(--good-text); font-weight: bold;
               text-align: center; border-radius: 3px; padding: 2px 6px;
               display: inline-block; min-width: 24px; }
  .score-bad { background: var(--bad); color: var(--bad-text); font-weight: bold;
              text-align: center; border-radius: 3px; padding: 2px 6px;
              display: inline-block; min-width: 24px; }
  .score-partial { background: var(--partial); color: var(--partial-text); font-weight: bold;
                  text-align: center; border-radius: 3px; padding: 2px 6px;
                  display: inline-block; min-width: 24px; }

  .pred-text { max-width: 180px; word-break: break-word; font-size: 0.80em; }
  .pred-bad { color: var(--bad-text); font-style: italic; }
  .q-text { font-size: 0.82em; line-height: 1.35; }
  .correct-text { font-weight: 600; color: #2c3e50; max-width: 160px; word-break: break-word; }

  .section-doc { margin: 20px 0 10px; padding: 10px 16px; background: #eef2f7;
                 border-radius: 6px; cursor: pointer; user-select: none; }
  .section-doc:hover { background: #dde5f0; }
  .section-doc h2 { margin: 0; font-size: 1.05em; display: flex; align-items: center; gap: 10px; }
  .section-doc .badge { font-size: 0.7em; background: var(--header); color: white;
                         padding: 2px 8px; border-radius: 10px; }

  .footer { margin: 30px 0; padding-top: 15px; border-top: 1px solid #ddd;
             font-size: 0.75em; color: #999; }

  .tooltip { position: relative; cursor: help; border-bottom: 1px dotted #999; }
  .tooltip .tooltip-text {
    visibility: hidden; width: 340px; background: #333; color: #eee;
    text-align: left; border-radius: 6px; padding: 8px 10px;
    position: absolute; z-index: 10; bottom: 125%; left: 50%;
    margin-left: -170px; font-size: 0.72em; line-height: 1.3;
    word-break: break-word; pointer-events: none;
  }
  .tooltip:hover .tooltip-text { visibility: visible; }

  @media (max-width: 900px) {
    .summary-card { flex: 0 0 140px; }
    th { font-size: 0.68em; }
    td { font-size: 0.7em; }
    .pred-text { max-width: 120px; }
  }
</style>
</head>
<body>
<h1>Per-Question Strategy Report</h1>
<p>Each row = one question. Each strategy column shows predicted answer + score.<br>
<strong>Source:</strong> <code>strategy_grid_results/</code><br>
<strong>Legend:</strong>
<span class="score-good">&check;</span> = correct (score &ge; 0.99) &emsp;
<span class="score-bad">&cross;</span> = wrong (score &le; 0.01) &emsp;
<span class="score-partial">0.XX</span> = partial credit
</p>
""")

    # --- Summary cards ---
    html.append('<div class="summary-bar">')
    for sname in strategy_names:
        stats = strat_stats[sname]
        acc = stats["accuracy"]
        correct = stats["correct"]
        total = stats["total"]
        acc_class = "acc-high" if acc >= 50 else ("acc-mid" if acc >= 25 else "acc-low")
        html.append(
            f'<div class="summary-card">'
            f'<div class="strat-name">{_esc(sname)}</div>'
            f'<div class="acc {acc_class}">{acc}%</div>'
            f'<div class="detail">{correct}/{total} correct</div>'
            f'</div>'
        )
    html.append('</div>')

    # --- Controls ---
    html.append('<div class="controls">')
    html.append('<button id="btn-all" class="active" onclick="filterScore(\'all\')">All Questions</button>')
    html.append('<button id="btn-correct" onclick="filterScore(\'correct\')">Has Correct</button>')
    html.append('<button id="btn-wrong" onclick="filterScore(\'wrong\')">Has Wrong</button>')
    html.append('<button id="btn-mixed" onclick="filterScore(\'mixed\')">Partial Only</button>')
    html.append('<span style="color:#888;">|</span>')
    html.append('<button id="toggle-all-cols" class="active" onclick="toggleCols(\'all\')">All Columns</button>')
    html.append('<button id="toggle-scores" onclick="toggleCols(\'scores\')">Scores Only</button>')
    html.append('<button id="toggle-answers" onclick="toggleCols(\'answers\')">Answers Only</button>')
    html.append('</div>')

    # --- Per-document sections ---
    rows_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_doc[row["doc_id"]].append(row)

    sorted_docs = sorted(rows_by_doc.keys())
    strategy_count = len(strategy_names)

    for doc_id in sorted_docs:
        doc_rows = rows_by_doc[doc_id]
        doc_type = doc_rows[0].get("doc_type", "")

        # Compute per-document per-strategy accuracy
        doc_acc: Dict[str, float] = {}
        for sname in strategy_names:
            scores = [
                row.get(sname, {}).get("score", 0.0)
                for row in doc_rows
                if sname in row
            ]
            doc_acc[sname] = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0

        html.append(
            f'<div class="section-doc" onclick="toggleSection(this)">'
            f'<h2>{_esc(doc_id)} '
            f'<span class="badge">{_esc(doc_type)}</span> '
            f'<span class="badge">{len(doc_rows)} questions</span>'
            f'</h2></div>'
        )
        html.append(f'<div class="section-content">')

        # Per-doc mini summary
        html.append('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;padding:0 6px;">')
        for sname in strategy_names:
            da = doc_acc[sname]
            daclass = "acc-high" if da >= 50 else ("acc-mid" if da >= 25 else "acc-low")
            html.append(
                f'<span style="font-size:0.72em;padding:2px 8px;border-radius:4px;'
                f'background:white;border:1px solid #ddd;">'
                f'<strong>{_esc(sname)}:</strong> <span class="{daclass}">{da}%</span>'
                f'</span>'
            )
        html.append('</div>')

        html.append('<table class="question-table"><thead><tr>')
        html.append('<th>#</th>')
        html.append('<th class="q-col">Question</th>')
        html.append('<th>Correct Answer</th>')

        for sname in strategy_names:
            html.append(f'<th class="pred-col">{_esc(sname)}</th>')
            html.append(f'<th class="score-col">Score</th>')

        html.append('</tr></thead><tbody>')

        for row in doc_rows:
            idx = row["idx"]
            question = row["question"]
            correct_answer = row["correct_answer"]
            answer_format = row["answer_format"]

            html.append('<tr>')
            html.append(f'<td style="color:#999;text-align:center;font-size:0.75em;">{idx}</td>')
            html.append(
                f'<td class="q-text">'
                f'{_esc(question)}'
                f'<br><span style="color:#999;font-size:0.75em;">format: {_esc(answer_format)}</span>'
                f'</td>'
            )
            html.append(f'<td class="correct-text">{_esc(str(correct_answer))}</td>')

            for sname in strategy_names:
                sentry = row.get(sname, {})
                pred = sentry.get("pred", "\u2014")
                score = sentry.get("score", -1)
                llm_preview = sentry.get("llm_preview", "")

                cell_class = _cell_class(score) if score >= 0 else ""
                score_str = _score_label(score) if score >= 0 else "\u2014"

                # Prediction cell with tooltip showing LLM preview
                display_pred = pred[:150] + ("..." if len(pred) > 150 else "")
                # Flag obviously bad predictions (extraction failures and
                # unfilled template placeholders like "[answer]", "[answer].")
                stripped = pred.strip().strip('"').strip("'").strip("`").rstrip(
                    ".,;:!?%"
                ).strip()
                is_bad = (
                    stripped == "—"
                    or stripped.lower().startswith("failed to extract")
                    or (
                        stripped.startswith("[")
                        and stripped.endswith("]")
                        and " " not in stripped
                    )
                )
                pred_extra_class = " pred-bad" if is_bad else ""
                html.append(f'<td class="pred-text{pred_extra_class}">')
                if pred and llm_preview:
                    html.append(
                        f'<span class="tooltip">{_esc(display_pred)}'
                        f'<span class="tooltip-text">{_esc(llm_preview)}</span>'
                        f'</span>'
                    )
                else:
                    html.append(_esc(display_pred))
                html.append('</td>')

                # Score cell
                html.append(
                    f'<td style="text-align:center;">'
                    f'<span class="{cell_class}">{score_str}</span>'
                    f'</td>'
                )

            html.append('</tr>')

        html.append('</tbody></table>')
        html.append('</div>')  # section-content

    # --- Footer ---
    html.append(f"""
<div class="footer">
  Generated by <code>per_question_report.py</code> &mdash;
  {len(rows)} questions &times; {strategy_count} strategies &mdash;
  Source: <code>strategy_grid_results/</code>
</div>
""")

    # --- JavaScript ---
    html.append("""
<script>
function filterScore(mode) {
  ['all','correct','wrong','mixed'].forEach(function(m) {
    document.getElementById('btn-'+m).classList.remove('active');
  });
  document.getElementById('btn-'+mode).classList.add('active');

  document.querySelectorAll('.question-table tbody tr').forEach(function(tr) {
    var cells = tr.querySelectorAll('td');
    var hasCorrect = false, hasWrong = false, hasPartial = false;
    // After col 0 (#), 1 (q), 2 (correct): pairs of (pred, score) starting at 3
    // Score cells are at even offsets from the pair start: 4, 6, 8, ...
    for (var i = 4; i < cells.length; i += 2) {
      var span = cells[i].querySelector('span');
      if (!span) continue;
      if (span.classList.contains('score-good')) hasCorrect = true;
      else if (span.classList.contains('score-bad')) hasWrong = true;
      else if (span.classList.contains('score-partial')) hasPartial = true;
    }
    if (mode === 'all') tr.style.display = '';
    else if (mode === 'correct') tr.style.display = hasCorrect ? '' : 'none';
    else if (mode === 'wrong') tr.style.display = hasWrong ? '' : 'none';
    else if (mode === 'mixed') tr.style.display = hasPartial ? '' : 'none';
  });
}

function toggleCols(mode) {
  ['all','scores','answers'].forEach(function(m) {
    document.getElementById('toggle-'+m).classList.remove('active');
  });
  document.getElementById('toggle-'+mode).classList.add('active');

  document.querySelectorAll('.question-table').forEach(function(table) {
    var theadTr = table.querySelector('thead tr');
    var ths = theadTr.querySelectorAll('th');
    // th indices: 0=#, 1=Q, 2=Correct, then pairs (pred, score) at 3,4,5,6,...
    var numExtraCols = (ths.length - 3);
    var half = numExtraCols / 2;

    for (var i = 0; i < half; i++) {
      var predThIdx = 3 + i * 2;
      var scoreThIdx = 4 + i * 2;
      var predTh = ths[predThIdx];
      var scoreTh = ths[scoreThIdx];

      if (mode === 'all') {
        if (predTh) predTh.style.display = '';
        if (scoreTh) scoreTh.style.display = '';
      } else if (mode === 'scores') {
        if (predTh) predTh.style.display = 'none';
        if (scoreTh) scoreTh.style.display = '';
      } else if (mode === 'answers') {
        if (predTh) predTh.style.display = '';
        if (scoreTh) scoreTh.style.display = 'none';
      }
    }

    // Now handle tbody cells
    table.querySelectorAll('tbody tr').forEach(function(tr) {
      var tds = tr.querySelectorAll('td');
      for (var i = 3; i < tds.length; i++) {
        var pairIdx = i - 3;
        // even = pred, odd = score
        if (mode === 'all') {
          tds[i].style.display = '';
        } else if (mode === 'scores') {
          tds[i].style.display = (pairIdx % 2 === 0) ? 'none' : '';  // hide pred, show score
        } else if (mode === 'answers') {
          tds[i].style.display = (pairIdx % 2 === 0) ? '' : 'none';  // show pred, hide score
        }
      }
    });
  });
}

function toggleSection(el) {
  var content = el.nextElementSibling;
  if (content.style.display === 'none') {
    content.style.display = '';
  } else {
    content.style.display = 'none';
  }
}
</script>
</body>
</html>""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"  HTML report -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-question HTML report for strategy_grid_results",
    )
    parser.add_argument(
        "-d", "--dir",
        default=str(DEFAULT_DIR),
        help=f"Results directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML path (default: <dir>/per_question_report.html)",
    )
    args = parser.parse_args()

    results_dir = args.dir
    if not os.path.isdir(results_dir):
        print(f"ERROR: directory not found: {results_dir}")
        sys.exit(1)

    out_path = args.output or os.path.join(results_dir, "per_question_report.html")

    print(f"Loading data from {results_dir} ...")

    # Load scored data
    strategy_data = load_scored_data(results_dir)
    strategy_names = sorted(strategy_data.keys())
    print(f"  Found {len(strategy_names)} strategies: {', '.join(strategy_names)}")

    if not strategy_data:
        print("ERROR: no *_scored.json files found")
        sys.exit(1)

    # Load ground truth
    ground_truth = load_ground_truth()
    print(f"  Loaded {len(ground_truth)} ground-truth entries")

    # Build per-question rows
    rows, strategy_names, strat_stats = build_per_question_rows(
        strategy_data, ground_truth
    )
    print(f"  Built {len(rows)} question rows across {len(set(r['doc_id'] for r in rows))} documents")

    # Generate HTML
    generate_html(out_path, rows, strategy_names, strat_stats)

    print(f"\nDone! Report size: {os.path.getsize(out_path):,} bytes")


if __name__ == "__main__":
    main()
