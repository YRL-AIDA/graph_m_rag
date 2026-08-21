#!/usr/bin/env python3
"""
Evaluate all strategy grid results from strategy_grid_results/ directory.

Computes per-strategy metrics using the official eval_score / eval_acc_and_f1
from MMLongDocEval — exactly the same pipeline as smallerdataset_evaluation.py.

Produces:
  - A plain-text summary report     → strategy_grid_results/evaluation_report.txt
  - A per-strategy scored JSON      → strategy_grid_results/<name>_scored.json
  - A strategy comparison summary   → strategy_grid_results/strategy_summary.json
  - An HTML comparison table        → strategy_grid_results/strategy_comparison.html

Usage:
    python app/tests/evaluate_strategies.py
    python app/tests/evaluate_strategies.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path resolution & imports  (mirrors smallerdataset_evaluation.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "strategy_grid_results"
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
# Strategy sorting
# ---------------------------------------------------------------------------

SORT_KEYS = {
    "accuracy": lambda m: m.get("accuracy", 0),
    "f1": lambda m: m.get("f1", 0),
    "avg_score": lambda m: m.get("avg_score", 0),
    "avg_elapsed": lambda m: -m.get("avg_elapsed", 0),
    "priority": lambda m: m.get("accuracy", 0) / (max(m.get("avg_elapsed", 1), 1) + 1),
}

SORT_BY_CHOICES = list(SORT_KEYS.keys())


def sort_strategies(
    metrics_list: List[Dict[str, Any]], sort_by: str = "accuracy"
) -> List[Dict[str, Any]]:
    key_fn = SORT_KEYS.get(sort_by, SORT_KEYS["accuracy"])
    return sorted(metrics_list, key=key_fn, reverse=True)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Template placeholders that the model sometimes echoes verbatim instead of
# filling in (e.g. "Extracted answer: [answer]"). These must never be scored.
_PLACEHOLDER_RE = re.compile(r"^\[[A-Za-z][A-Za-z0-9\s_.-]*\]$")

# Priority: [FINAL_ANSWER] marker (from updated system prompt), then legacy markers
_FINAL_ANSWER_PATTERN = re.compile(
    r"\[FINAL_ANSWER\]\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)

# Try explicit "Extracted answer:" / "Answer:" lines
_EXTRACTED_PATTERN = re.compile(
    r"(?:Extracted\s+answer|Answer)\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def _strip_thinking(text: str) -> str:
    """Remove everything up to and including the last ``</think>`` marker.

    The final answer always appears after the closing think tag; anything
    before it is chain-of-thought and must not be scored.
    """
    if not text:
        return ""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text.strip()


# Failure sentinels emitted by the extraction pipeline. Tolerates trailing
# punctuation (e.g. "Failed to extract.") which the exact-equality check below
# used to miss.
_FAILED_SENTINEL_RE = re.compile(r"^failed\s+to\s+extract\b", re.IGNORECASE)


def _is_failed_sentinel(value: str) -> bool:
    """True if ``value`` is a literal extraction-failure sentinel."""
    return bool(_FAILED_SENTINEL_RE.match((value or "").strip()))


def _is_placeholder(value: str) -> bool:
    """True if ``value`` is an unfilled template placeholder like ``[answer]``.

    Tolerates surrounding quotes and trailing punctuation (e.g. "[answer].",
    "[answer],", "[answer]%") which the model sometimes emits when echoing the
    template verbatim. These must never be scored.
    """
    v = (value or "").strip()
    if not v:
        return True
    v = v.strip('"').strip("'").strip("`").strip()
    v = v.rstrip(".,;:!?%").strip()
    if not v:
        return True
    return bool(_PLACEHOLDER_RE.match(v))


def _clean_extracted_value(raw: str) -> str:
    """Normalize a value found after an 'Extracted answer:' marker.

    Strips surrounding quotes/backticks and rejects placeholders and failure
    sentinels, returning "" when the value is unusable.
    """
    value = (raw or "").strip()
    value = value.strip('"').strip("'").strip("`").strip()
    if _is_placeholder(value) or _is_failed_sentinel(value):
        return ""
    return value


def extract_pred_from_extracted_res(extracted_res: str) -> str:
    """Extract the final answer from ``extracted_res``.

    Mirrors smallerdataset_evaluation.py but returns "" (instead of
    "Failed to extract") when no usable value is found, so the caller can
    fall back to ``llm_answer``. Scans markers right-to-left and takes the
    first real value, rejecting unfilled template placeholders like
    "[answer]" that appear in the reasoning text.
    """
    if not extracted_res:
        return ""
    if not isinstance(extracted_res, str):
        extracted_res = str(extracted_res)

    text = extracted_res.strip()
    if not text:
        return ""

    # Upstream pipeline sometimes stores a literal failure sentinel.
    if _is_failed_sentinel(text):
        return ""

    # Every "Extracted answer:" occurrence; template placeholders may appear
    # earlier in the reasoning, the real value after </think>.
    parts = text.split("Extracted answer:")
    for after in reversed(parts[1:]):
        value = after.split("Answer format:")[0].strip()
        cleaned = _clean_extracted_value(value)
        if cleaned:
            return cleaned

    return ""


def extract_pred_from_llm_answer(llm_answer: str) -> str:
    """Heuristic fallback: extract a short answer from raw LLM output.

    Priority order:
      1. [FINAL_ANSWER]: marker  (from updated system prompt)
      2. "Extracted answer:" / "Answer:" markers
      3. Python list string (e.g. ['item1', 'item2'])
      4. Last non-empty line as fallback

    Strips <think>...</think> blocks first.
    """
    if not llm_answer:
        return ""

    text = _strip_thinking(llm_answer)

    # Priority 1: [FINAL_ANSWER] marker from updated prompt
    m = _FINAL_ANSWER_PATTERN.search(text)
    if m:
        candidate = _clean_extracted_value(m.group(1))
        if candidate:
            return candidate

    # Priority 2: Legacy markers (Extracted answer: / Answer:)
    m = _EXTRACTED_PATTERN.search(text)
    if m:
        candidate = _clean_extracted_value(m.group(1))
        if candidate:
            return candidate

    # Priority 3: Detect Python list strings: ['val1', 'val2']
    # This helps with List-format answers where the LLM outputs a Python repr
    list_match = re.search(r"\[([^\]]+)\]", text)
    if list_match:
        # Only use if it looks like a proper list (multiple quoted items)
        inner = list_match.group(1)
        if inner.count("'") >= 4 or inner.count('"') >= 4:
            return list_match.group(0)  # return the full [...] string

    # Fallback: last non-empty, non-placeholder line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        candidate = _clean_extracted_value(line)
        if candidate:
            return candidate
    return ""


def _is_numeric(s: str) -> bool:
    """Return True if s parses as a number (optionally with $/% prefix/suffix)."""
    if not s:
        return False
    s = s.strip().lstrip("$").rstrip("%").replace(",", "").strip()
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_pred(pred: str, expected_answer: str, answer_format: str) -> str:
    """Normalize pred to match expected answer formatting conventions.

    - Float: auto-add/remove % sign to match expected
    - General: strip trailing punctuation that eval_score is strict about
    """
    if not pred or not expected_answer:
        return pred

    fmt = (answer_format or "").strip().lower()
    exp = str(expected_answer).strip()

    if fmt == "float":
        # Never touch non-numeric predictions (e.g. "Not answerable",
        # "Failed to extract") — appending "%" would corrupt them into
        # "Not answerable%".
        if not _is_numeric(pred):
            return pred
        exp_has_pct = "%" in exp
        pred_has_pct = "%" in pred
        if exp_has_pct and not pred_has_pct:
            return pred.strip() + "%"
        elif not exp_has_pct and pred_has_pct:
            return pred.strip().rstrip("%").strip()

    return pred


# ---------------------------------------------------------------------------
# Per-strategy evaluation  (combines smallerdataset_evaluation.py logic
#                           with eval_acc_and_f1 + show_results-style breakdown)
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
        # Path 1: extracted_res available (via-API pipeline) — use exact
        #         smallerdataset_evaluation.py logic
        # Path 2: only llm_answer (strategy grid results) — heuristic fallback
        extracted_res = r.get("extracted_res")
        pred = ""
        if extracted_res is not None and str(extracted_res) not in ("None", ""):
            pred = extract_pred_from_extracted_res(str(extracted_res))

        # Fall back to the raw LLM answer when extracted_res is missing or
        # yielded no usable value (e.g. "Failed to extract" or a placeholder).
        if not pred:
            llm_answer = r.get("llm_answer", "")
            pred = extract_pred_from_llm_answer(llm_answer)

        if not pred:
            continue

        # --- Score via eval_score (same as smallerdataset_evaluation.py) ---
        correct_answer = gt_info.get("answer", r.get("answer", ""))
        answer_format = gt_info.get("answer_format", "Str")

        # Normalize predicted answer format to match expected (e.g., % sign for Float)
        pred = _normalize_pred(pred, correct_answer, answer_format)

        try:
            score = float(eval_score(correct_answer, pred, answer_format))
        except Exception:
            score = 0.0

        r["pred"] = pred
        r["score"] = score
        scored_samples.append(r)

        if r.get("elapsed"):
            elapsed_times.append(float(r["elapsed"]))

    # --- Populate evidence metadata for breakdown (mirrors show_results) ---
    for s in scored_samples:
        doc_id = s.get("doc_id", "")
        question = s.get("question", "")
        gt_info = ground_truth_map.get((doc_id, question), {})

        # Parse evidence_pages via eval() — same as show_results line 178
        try:
            s["evidence_pages"] = eval(str(gt_info.get("evidence_pages", "[]")))
        except Exception:
            s["evidence_pages"] = []

        # Parse evidence_sources via eval() — same as show_results line 179
        try:
            s["evidence_sources"] = eval(str(gt_info.get("evidence_sources", "[]")))
        except Exception:
            s["evidence_sources"] = []

        # Ensure answer/doc_type are on the sample for show_results-style logic
        s["answer"] = gt_info.get("answer", s.get("answer", ""))
        s["answer_format"] = gt_info.get("answer_format", s.get("answer_format", "Str"))
        s["doc_type"] = gt_info.get("doc_type", "Unknown")

    # --- Compute metrics via eval_acc_and_f1 (same as smallerdataset_evaluation.py) ---
    acc, f1 = eval_acc_and_f1(scored_samples)
    acc_pct = round(acc * 100, 1)
    f1_val = round(f1, 3)

    avg_score = acc  # eval_acc_and_f1 uses mean of scores
    correct_count = sum(1 for s in scored_samples if s.get("score", 0) >= 1.0)
    avg_elapsed = (
        sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    )

    # --- Breakdown by page type (mirrors show_results lines 188-200) ---
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

    # --- Breakdown by evidence source (mirrors show_results lines 204-212) ---
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

    # --- Breakdown by document type (mirrors show_results lines 215-218) ---
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

    # --- Per-question breakdown (lightweight) ---
    per_question: List[Dict[str, Any]] = []
    for s in scored_samples:
        # Truncate llm_answer for report readability
        llm = s.get("llm_answer", "")
        llm_preview = llm[:200].replace("\n", " ") if llm else ""
        # Reconstruct the exact text context that was sent to the LLM
        # (the API joins context_blocks with "\n\n" before calling the model).
        context_blocks = s.get("context_blocks") or []
        context_text = (
            "\n\n".join(str(b) for b in context_blocks)
            if context_blocks else ""
        )
        per_question.append({
            "doc_id": s.get("doc_id", ""),
            "question": s.get("question", ""),
            "correct_answer": s.get("answer", ""),
            "predicted": s.get("pred", ""),
            "score": s.get("score", 0),
            "elapsed": s.get("elapsed", 0),
            "doc_type": s.get("doc_type", "Unknown"),
            "answer_format": s.get("answer_format", "Str"),
            "evidence_pages": s.get("evidence_pages", []),
            "llm_preview": llm_preview,
            "context_text": context_text,
        })

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
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_text_report(
    path: str,
    metrics_list: List[Dict[str, Any]],
    sort_by: str = "accuracy",
):
    """Write a plain-text evaluation report in show_results-style format."""
    if not metrics_list:
        return

    lines: List[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  GRAPH M-RAG  —  Strategy Grid Evaluation Report")
    lines.append(sep)
    lines.append("")

    sorted_metrics = sort_strategies(metrics_list, sort_by=sort_by)

    header = (
        f" {'Rank':<5} {'Strategy':<30} {'Acc%':>7} {'F1':>7} "
        f"{'Corr':>6} {'Scored':>6} {'Time':>6}"
    )
    lines.append(header)
    lines.append("-" * 72)

    for i, m in enumerate(sorted_metrics):
        line = (
            f" {i + 1:<5} {m['name']:<30} "
            f"{m['accuracy']:>6.1f}% {m['f1']:>6.3f} "
            f"{m['correct']:>5}/{m['scored']:<5} {m['avg_elapsed']:>5.1f}s"
        )
        lines.append(line)

    lines.append("")
    lines.append(sep)
    lines.append("  Detailed breakdown (top 5 strategies)")
    lines.append(sep)
    lines.append("")

    for m in sorted_metrics[:5]:
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

        # --- Per-question breakdown ---
        pq = m.get("per_question", [])
        if pq:
            pq_sorted = sorted(pq, key=lambda x: x["score"])
            lines.append("")
            lines.append(
                f"  {'─' * 66}"
            )
            lines.append(
                f"  Per-Question Breakdown ({len(pq)} questions)"
            )
            lines.append(
                f"  {'─' * 66}"
            )
            q_header = (
                f"  {'#':<4} {'Score':<7} {'Q (truncated)':<35} "
                f"{'Expected':<25} {'Predicted':<25}"
            )
            lines.append(q_header)
            lines.append(f"  {'-' * 66}")
            for idx, q in enumerate(pq_sorted, 1):
                q_text = q["question"][:50] if q["question"] else ""
                lines.append(
                    f"  {idx:<4} {q['score']:<7.2f} {q_text:<50}"
                )
                lines.append(
                    f"       Expected : {q['correct_answer']}"
                )
                lines.append(
                    f"       Predicted: {q['predicted']}"
                )
                lines.append("")
            lines.append("")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Text report → {path}")


def write_html_report(
    path: str,
    metrics_list: List[Dict[str, Any]],
    sort_by: str = "accuracy",
):
    """Write a self-contained HTML comparison table."""
    if not metrics_list:
        return

    sorted_metrics = sort_strategies(metrics_list, sort_by=sort_by)
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
<title>Graph M-RAG Strategy Grid Evaluation</title>
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
<h1>Graph M-RAG &mdash; Strategy Grid Evaluation</h1>
<p>
  Dataset: <strong>SmallerDataset</strong>
  (12 documents, 105 questions, 5 types).
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

        # --- Per-question breakdown table ---
        pq = m.get("per_question", [])
        if pq:
            pq_sorted = sorted(pq, key=lambda x: x["score"])
            html += (
                "<h4>Per-Question Breakdown</h4>"
                "<table>"
                "<tr><th>#</th><th>Doc ID</th><th>Question</th>"
                "<th>Expected</th><th>Predicted</th><th>Score</th>"
                "<th>Model Answer (preview)</th><th>Context</th></tr>"
            )
            for idx, q in enumerate(pq_sorted, 1):
                score_color = "#d4edda" if q["score"] >= 1.0 else (
                    "#f8d7da" if q["score"] < 0.5 else "#fff3cd"
                )
                context_text = q.get("context_text", "")
                if context_text:
                    context_cell = (
                        "<td style=\"font-size:0.8em\">"
                        "<details>"
                        "<summary style=\"cursor:pointer;color:#2c3e50;"
                        "font-weight:bold\">Показать контекст</summary>"
                        "<pre style=\"max-width:600px;max-height:400px;"
                        "overflow:auto;white-space:pre-wrap;"
                        "font-size:0.75em;background:#f5f5f5;padding:8px;"
                        "margin-top:4px\">"
                        f"{html_escape(context_text)}"
                        "</pre>"
                        "</details>"
                        "</td>"
                    )
                else:
                    context_cell = (
                        "<td style=\"font-size:0.8em;color:#999\">—</td>"
                    )
                html += (
                    f"<tr style=\"background:{score_color}\">"
                    f"<td>{idx}</td>"
                    f"<td style=\"font-size:0.8em\">{q['doc_id'][:30]}</td>"
                    f"<td style=\"font-size:0.8em\">{q['question'][:80]}</td>"
                    f"<td style=\"font-size:0.8em\">{q['correct_answer']}</td>"
                    f"<td style=\"font-size:0.8em\">{q['predicted']}</td>"
                    f"<td style=\"font-weight:bold\">{q['score']:.2f}</td>"
                    f"<td style=\"font-size:0.75em; max-width:350px\">"
                    f"{q['llm_preview'][:150]}</td>"
                    f"{context_cell}"
                    f"</tr>"
                )
            html += "</table>"

        html += "</details>\n"

    html += """
<div class="footer">
  Generated by <code>evaluate_strategies.py</code> —
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
        description="Evaluate strategy grid results from strategy_grid_results/",
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
            f"Choices: {', '.join(SORT_BY_CHOICES)}. "
            "(default: accuracy)"
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
    write_text_report(text_path, all_metrics, sort_by=args.sort_by)

    json_path = os.path.join(results_dir, "strategy_summary.json")
    write_json(json_path, all_metrics)
    print(f"  JSON summary → {json_path}")

    html_path = os.path.join(results_dir, "strategy_comparison.html")
    write_html_report(html_path, all_metrics, sort_by=args.sort_by)

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
