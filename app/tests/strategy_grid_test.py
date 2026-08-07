#!/usr/bin/env python3
"""
Strategy Grid Test — exhaustive evaluation of all retrieval / context / generation
combinations on the SmallerDataset.

Usage:
    python app/tests/strategy_grid_test.py
    python app/tests/strategy_grid_test.py --list-strategies
    python app/tests/strategy_grid_test.py --base-url http://host:9191

The script hits the /ask-document API with every strategy combination,
saves per-strategy results, evaluates them, and produces a visual HTML report.

Strategies grid (3 dimensions):
  - Retrieval:   none | reranker | mmr_lambda0.5 | mmr_lambda0.7 | mmr_lambda0.9
  - Context:     none | semantic | structural | semantic+structural
  - Processing:  none | decompose | iterative

Total: 5 x 4 x 3 = 60 combinations  (subset to 20 most meaningful by default)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# --- Strategy Definitions ---------------------------------------------------

@dataclass
class Strategy:
    """A single strategy configuration for the ask-document endpoint."""
    name: str           # short identifier, e.g. "baseline"
    label: str          # human-readable, e.g. "Baseline (Qdrant + LLM)"
    use_reranker: bool = False
    use_mmr_reranker: bool = False
    mmr_lambda: float = 0.7
    mmr_min_relevance: float = 0.0
    use_semantic_graph: bool = False
    use_structured_graph: bool = False
    use_iterative_search: bool = False
    use_question_decomposition: bool = False

    def to_payload(self, file_hash: str, question: str,
                   limit: int = 10) -> dict:
        """Build the JSON payload for /ask-document."""
        return {
            "file_hash": file_hash,
            "question": question,
            "limit": limit,
            "use_llm": True,
            "use_reranker": self.use_reranker,
            "use_mmr_reranker": self.use_mmr_reranker,
            "mmr_lambda": self.mmr_lambda,
            "mmr_min_relevance": self.mmr_min_relevance,
            "use_semantic_graph": self.use_semantic_graph,
            "use_structured_graph": self.use_structured_graph,
            "use_iterative_search": self.use_iterative_search,
            "use_question_decomposition": self.use_question_decomposition,
        }


STRATEGIES: List[Strategy] = [
    # --- Baseline ---
    Strategy("baseline", "Baseline (Qdrant + LLM)"),

    # --- Retrieval variants ---
    Strategy("reranker", "Reranker",
             use_reranker=True),
    Strategy("mmr_07", "MMR (lambda=0.7)",
             use_mmr_reranker=True, mmr_lambda=0.7),
    Strategy("mmr_05", "MMR (lambda=0.5)",
             use_mmr_reranker=True, mmr_lambda=0.5),
    Strategy("mmr_09", "MMR (lambda=0.9)",
             use_mmr_reranker=True, mmr_lambda=0.9),

    # --- Context enrichment ---
    Strategy("semantic", "Semantic Graph",
             use_semantic_graph=True),
    Strategy("structural", "Structural Graph",
             use_structured_graph=True),
    Strategy("both_graphs", "Semantic + Structural",
             use_semantic_graph=True, use_structured_graph=True),

    # --- Context + Reranker ---
    Strategy("both_reranker", "Both Graphs + Reranker",
             use_semantic_graph=True, use_structured_graph=True,
             use_reranker=True),
    Strategy("both_mmr07", "Both Graphs + MMR lambda=0.7",
             use_semantic_graph=True, use_structured_graph=True,
             use_mmr_reranker=True, mmr_lambda=0.7),
    Strategy("semantic_reranker", "Semantic + Reranker",
             use_semantic_graph=True, use_reranker=True),
    Strategy("structural_reranker", "Structural + Reranker",
             use_structured_graph=True, use_reranker=True),

    # --- Question processing ---
    Strategy("decompose", "Question Decomposition",
             use_question_decomposition=True,
             use_semantic_graph=True, use_structured_graph=True),
    Strategy("iterative", "Iterative Search",
             use_iterative_search=True,
             use_semantic_graph=True, use_structured_graph=True),

    # --- Full system ---
    Strategy("full_system", "Full System (All)",
             use_reranker=True,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),

    # --- Full + MMR variants ---
    Strategy("full_mmr07", "Full + MMR lambda=0.7",
             use_mmr_reranker=True, mmr_lambda=0.7,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_mmr05", "Full + MMR lambda=0.5",
             use_mmr_reranker=True, mmr_lambda=0.5,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_mmr09", "Full + MMR lambda=0.9",
             use_mmr_reranker=True, mmr_lambda=0.9,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),

    # --- Graph-only + iterative ---
    Strategy("semantic_iterative", "Semantic + Iterative",
             use_semantic_graph=True, use_iterative_search=True),
    Strategy("structural_iterative", "Structural + Iterative",
             use_structured_graph=True, use_iterative_search=True),
]


# --- Helpers ----------------------------------------------------------------


def read_json(filename: str) -> Any:
    """Load a JSON file."""
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def ask_document(base_url: str, strategy: Strategy,
                 file_hash: str, question: str,
                 limit: int = 10, timeout: int = 300) -> Dict[str, Any]:
    """Call /ask-document with the given strategy."""
    payload = strategy.to_payload(file_hash, question, limit)
    url = f"{base_url.rstrip('/')}/ask-document"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# --- Main grid test ---------------------------------------------------------


def run_grid_test(
    base_url: str = "http://0.0.0.0:9191",
    dataset_path: str = "",
    file_hash_map_path: str = "",
    output_dir: str = "",
    limit: int = 10,
):
    """Run all strategies against the SmallerDataset and save results."""

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    if not dataset_path:
        dataset_path = str(script_dir / "SmallerDataset" / "samples.json")
    if not file_hash_map_path:
        file_hash_map_path = str(script_dir / "file_hash_comparison.json")
    if not output_dir:
        output_dir = str(script_dir / "strategy_grid_results")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    hash_map = read_json(file_hash_map_path)
    dataset = read_json(dataset_path)

    # Load extraction prompt (optional, for evaluation)
    extract_prompt_path = script_dir / "MMLongDocEval" / \
        "prompt_for_answer_extraction.md"
    extract_prompt = ""
    if extract_prompt_path.exists():
        extract_prompt = extract_prompt_path.read_text(encoding="utf-8")

    print(f"Dataset: {len(dataset)} questions")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    all_strategy_results: Dict[str, List[dict]] = {}

    for si, strategy in enumerate(STRATEGIES):
        strategy_label = (
            f"[{si + 1}/{len(STRATEGIES)}] {strategy.label}"
        )
        print(f"\n{'=' * 70}")
        print(f"  {strategy_label}")
        print(f"{'=' * 70}")

        strategy_results: List[dict] = []
        strategy_output = os.path.join(output_dir, f"{strategy.name}.json")

        # Resume from partial if exists
        if os.path.exists(strategy_output):
            strategy_results = read_json(strategy_output)
            print(
                f"  Resuming from {len(strategy_results)} already processed"
            )

        for qi, case in enumerate(dataset):
            already = (
                qi < len(strategy_results)
                and strategy_results[qi].get("llm_answer")
            )
            if already:
                continue  # already processed

            doc_id = case.get("doc_id", "")
            question = case.get("question", "")
            correct_answer = case.get("answer", "")

            if doc_id not in hash_map:
                print(
                    f"  [{qi}/{len(dataset)}] SKIP: {doc_id} "
                    f"not in hash map"
                )
                strategy_results.append({
                    "doc_id": doc_id,
                    "question": question,
                    "answer": correct_answer,
                    "status": "hash_not_found",
                })
                continue

            file_hash = hash_map[doc_id]
            start = time.time()

            try:
                result = ask_document(
                    base_url, strategy, file_hash, question, limit,
                )
                elapsed = time.time() - start

                llm_answer = result.get("llm_answer", "")
                answers_count = len(result.get("answers", []))

                strategy_results.append({
                    "doc_id": doc_id,
                    "question": question,
                    "answer": correct_answer,
                    "file_hash": file_hash,
                    "llm_answer": llm_answer,
                    "retrieves": result.get("answers", []),
                    "context_blocks": result.get("context_blocks", []),
                    "status": "completed",
                    "elapsed": round(elapsed, 2),
                    "answers_count": answers_count,
                })
                print(
                    f"  [{qi}/{len(dataset)}] OK ({elapsed:.1f}s) - "
                    f"{len(llm_answer)} chars, {answers_count} retrieves"
                )

            except Exception as exc:
                elapsed = time.time() - start
                print(f"  [{qi}/{len(dataset)}] ERROR: {exc}")
                strategy_results.append({
                    "doc_id": doc_id,
                    "question": question,
                    "answer": correct_answer,
                    "status": f"error: {exc}",
                    "elapsed": round(elapsed, 2),
                })

            # Save checkpoint every 10 questions
            if (qi + 1) % 10 == 0:
                with open(strategy_output, "w", encoding="utf-8") as f:
                    json.dump(
                        strategy_results, f, ensure_ascii=False, indent=2,
                    )

        # Final save
        with open(strategy_output, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, ensure_ascii=False, indent=2)

        all_strategy_results[strategy.name] = strategy_results
        print(
            f"  Done: {len(strategy_results)} results -> {strategy_output}"
        )

    # --- Evaluate and generate report ---
    print(f"\n{'=' * 70}")
    print("  Evaluating all strategies...")
    print(f"{'=' * 70}")

    report = evaluate_all(
        all_strategy_results, output_dir, extract_prompt,
    )
    print(f"\nReport saved to {report}")


# --- Evaluation -------------------------------------------------------------


def evaluate_all(
    all_results: Dict[str, List[dict]],
    output_dir: str,
    extract_prompt: str,
) -> str:
    """Score every strategy using exact-match, then produce HTML report."""

    # Try to import eval utilities
    eval_available = False
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from MMLongDocEval.eval_score import eval_score  # noqa: F401
        eval_available = True
    except ImportError:
        print(
            "  WARNING: MMLongDocEval not available, "
            "using substring matching"
        )

    rows: List[Dict[str, Any]] = []

    for strategy in STRATEGIES:
        results = all_results.get(strategy.name, [])
        completed = [
            r for r in results
            if r.get("status") == "completed" and r.get("llm_answer")
        ]
        total = len(results)
        completed_count = len(completed)

        scores: List[float] = []
        correct_count = 0

        for r in completed:
            llm_answer = r.get("llm_answer", "")
            correct_answer = r.get("answer", "")

            if eval_available:
                try:
                    score = float(eval_score(llm_answer, correct_answer))
                except Exception:
                    score = 0.0
            else:
                ans_lower = correct_answer.strip().lower()
                llm_lower = llm_answer.strip().lower()
                score = 1.0 if ans_lower and ans_lower in llm_lower else 0.0

            scores.append(score)
            if score >= 0.5:
                correct_count += 1

            r["score"] = score

        avg_score = sum(scores) / len(scores) if scores else 0.0
        accuracy = (
            (correct_count / completed_count * 100)
            if completed_count else 0.0
        )
        times = [r.get("elapsed", 0) for r in completed]
        avg_elapsed = sum(times) / len(times) if times else 0.0

        rows.append({
            "name": strategy.name,
            "label": strategy.label,
            "total": total,
            "completed": completed_count,
            "accuracy": round(accuracy, 1),
            "avg_score": round(avg_score, 3),
            "avg_time": round(avg_elapsed, 1),
            "correct": correct_count,
        })

    # Sort by accuracy descending
    rows.sort(key=lambda r: r["accuracy"], reverse=True)

    # Generate HTML report
    html_path = os.path.join(output_dir, "strategy_comparison.html")
    _write_html_report(html_path, rows, STRATEGIES)

    # Also save JSON summary
    json_path = os.path.join(output_dir, "strategy_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return html_path


def _write_html_report(
    path: str,
    rows: List[dict],
    strategies: List[Strategy],
):
    """Generate a self-contained HTML comparison table."""

    max_acc = max(r["accuracy"] for r in rows) if rows else 100.0

    def _row_color(acc: float) -> str:
        ratio = acc / max(max_acc, 0.01)
        if ratio >= 0.9:
            return "#d4edda"  # green
        if ratio >= 0.7:
            return "#fff3cd"  # yellow
        if ratio >= 0.5:
            return "#ffeeba"
        return "#f8d7da"  # red

    table_rows = ""
    for i, r in enumerate(rows):
        medal = ""
        if i == 0:
            medal = " (1st)"
        elif i == 1:
            medal = " (2nd)"
        elif i == 2:
            medal = " (3rd)"

        rb = _row_color(r["accuracy"])
        table_rows += f"""
        <tr style="background:{rb}">
          <td style="text-align:right">{i + 1}</td>
          <td><strong>{r['label']}{medal}</strong><br>
            <code>{r['name']}</code></td>
          <td style="text-align:right"><strong>{r['accuracy']}%</strong></td>
          <td style="text-align:right">{r['avg_score']:.3f}</td>
          <td style="text-align:right">{r['correct']}/{r['completed']}</td>
          <td style="text-align:right">{r['avg_time']:.1f}s</td>
        </tr>"""

    strategy_rows = "\n".join(
        f"<tr><td><code>{s.name}</code></td><td>{s.label}</td></tr>"
        for s in strategies
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Graph M-RAG Strategy Grid Comparison</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 sans-serif;
    max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #333;
  }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  table {{
    border-collapse: collapse; width: 100%; margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  th {{
    background: #2c3e50; color: white; padding: 10px 12px;
    text-align: left; font-size: 0.9em;
  }}
  td {{
    padding: 8px 12px; border-bottom: 1px solid #ddd; font-size: 0.9em;
  }}
  tr:hover {{ filter: brightness(0.95); }}
  .legend {{
    display: flex; gap: 15px; margin: 15px 0; font-size: 0.85em;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 5px;
  }}
  .legend-swatch {{
    width: 20px; height: 20px; border-radius: 3px;
    border: 1px solid #ccc;
  }}
  .footer {{
    margin-top: 30px; font-size: 0.8em; color: #888;
    border-top: 1px solid #eee; padding-top: 10px;
  }}
  code {{
    font-size: 0.8em; color: #555; background: #f5f5f5;
    padding: 1px 4px; border-radius: 3px;
  }}
</style>
</head>
<body>
<h1>Graph M-RAG &mdash; Strategy Grid Comparison</h1>
<p>
  Dataset: <strong>SmallerDataset</strong>
  (12 documents, 105 questions, 5 types).
  Results from combinatorial testing of retrieval,
  context enrichment, and generation strategies.
</p>

<div class="legend">
  <div class="legend-item">
    <div class="legend-swatch" style="background:#d4edda"></div>
    90%+ of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#fff3cd"></div>
    70-90% of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#ffeeba"></div>
    50-70% of best
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#f8d7da"></div>
    <50% of best
  </div>
</div>

<h2>Accuracy Ranking</h2>
<table>
<thead>
<tr>
  <th>#</th>
  <th>Strategy</th>
  <th>Accuracy</th>
  <th>Avg Score</th>
  <th>Correct</th>
  <th>Avg Time</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Strategy Descriptions</h2>
<table>
<thead>
<tr><th>Name</th><th>Components</th></tr>
</thead>
<tbody>
{strategy_rows}
</tbody>
</table>

<div class="footer">
Generated by <code>app/tests/strategy_grid_test.py</code> &mdash;
Graph M-RAG: Multi-Strategy Graph-Enhanced
Retrieval-Augmented Generation
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# --- Entry point ------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Strategy Grid Test")
    ap.add_argument(
        "--base-url", default="http://0.0.0.0:9191",
        help="Base URL of the API server",
    )
    ap.add_argument(
        "--dataset", default="",
        help="Path to SmallerDataset samples.json",
    )
    ap.add_argument(
        "--hash-map", default="",
        help="Path to file_hash_comparison.json",
    )
    ap.add_argument(
        "--output-dir", default="",
        help="Directory for strategy results",
    )
    ap.add_argument(
        "--limit", type=int, default=10,
        help="Number of retrieves per query",
    )
    ap.add_argument(
        "--list-strategies", action="store_true",
        help="Just print the strategy grid and exit",
    )
    args = ap.parse_args()

    if args.list_strategies:
        print(f"\n{'=' * 70}")
        print(f"  Strategy Grid: {len(STRATEGIES)} combinations")
        print(f"{'=' * 70}")
        for i, s in enumerate(STRATEGIES):
            flags: List[str] = []
            if s.use_reranker:
                flags.append("reranker")
            if s.use_mmr_reranker:
                flags.append(f"mmr(l={s.mmr_lambda})")
            if s.use_semantic_graph:
                flags.append("semantic")
            if s.use_structured_graph:
                flags.append("structural")
            if s.use_iterative_search:
                flags.append("iterative")
            if s.use_question_decomposition:
                flags.append("decompose")
            flag_str = ", ".join(flags) if flags else "none"
            print(
                f"  [{i + 1:2d}] {s.name:25s}  "
                f"{s.label:40s}  [{flag_str}]"
            )
        sys.exit(0)

    run_grid_test(
        base_url=args.base_url,
        dataset_path=args.dataset,
        file_hash_map_path=args.hash_map,
        output_dir=args.output_dir,
        limit=args.limit,
    )
