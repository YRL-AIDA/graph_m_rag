#!/usr/bin/env python3
"""
Strategy Grid Test — runs all retrieval / context / generation
combinations on the SmallerDataset and saves raw results.

This is a TEST RUNNER only — it does NOT evaluate results.
Evaluation is handled by the separate evaluate_strategies.py script.

Usage:
    python app/tests/strategy_grid_test.py
    python app/tests/strategy_grid_test.py --list-strategies
    python app/tests/strategy_grid_test.py --base-url http://host:9191
    python app/tests/strategy_grid_test.py --strategies baseline,reranker

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

from evaluate_strategies import SORT_KEYS, SORT_BY_CHOICES  # type: ignore[import]


# ---------------------------------------------------------------------------
# Answer extraction (optional, mirrors smallerdataset_eval_via_api.py)
# ---------------------------------------------------------------------------

def extract_answer_qwen_api(question: str, output: str, prompt: str) -> str:
    """Extract short answer from LLM output using custom Qwen API."""
    try:
        import utils.qwen_qa_utils as custom_qwen  # type: ignore[import]
    except ImportError:
        return "Failed to extract"

    tt = custom_qwen.ModelMessageDict()
    tt.add_text_content(prompt)
    answer = f"Question: {question}\nAnalysis:{output}"
    tt.add_text_content(answer)
    result = custom_qwen.send_messasge(
        messages=[tt], base_url=os.environ.get(
            'QWEN_EXTRACT_URL', 'http://192.168.19.127:8888/v1'
        )
    )
    return result[1][0]


# ---------------------------------------------------------------------------
# Strategy Definitions
# ---------------------------------------------------------------------------

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
    use_structural_parent_only: bool = False
    use_iterative_search: bool = False
    use_question_decomposition: bool = False

    def to_payload(self, file_hash: str, question: str,
                   limit: int = 30) -> dict:
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
            "use_structural_parent_only": self.use_structural_parent_only,
            "use_iterative_search": self.use_iterative_search,
            "use_question_decomposition": self.use_question_decomposition,
        }

    def flags_summary(self) -> List[str]:
        """Human-readable list of active flags for this strategy."""
        flags: List[str] = []
        if self.use_reranker:
            flags.append("reranker")
        if self.use_mmr_reranker:
            flags.append(f"mmr(λ={self.mmr_lambda})")
        if self.use_semantic_graph:
            flags.append("semantic")
        if self.use_structured_graph:
            if self.use_structural_parent_only:
                flags.append("structural(parent-only)")
            else:
                flags.append("structural")
        if self.use_iterative_search:
            flags.append("iterative")
        if self.use_question_decomposition:
            flags.append("decompose")
        return flags if flags else ["baseline"]


# --- Strategy definitions: 20 combinations from strategy_reference.md ---

STRATEGIES: List[Strategy] = [

    # ----- Baseline -----
    Strategy("baseline", "Baseline (Qdrant + LLM)"),

    # ----- Retrieval variants -----
    Strategy("reranker", "Reranker",
             use_reranker=True),
#    Strategy("mmr_07", "MMR (λ=0.7)",
#             use_mmr_reranker=True, mmr_lambda=0.7),
#    Strategy("mmr_05", "MMR (λ=0.5)",
#             use_mmr_reranker=True, mmr_lambda=0.5),
    Strategy("mmr_09", "MMR (λ=0.9)",
             use_mmr_reranker=True, mmr_lambda=0.9),

    # ----- Single-graph context enrichment -----
    Strategy("semantic", "Semantic Graph",
             use_semantic_graph=True),
    Strategy("structural", "Structural Graph (ORDER + Parent)",
             use_structured_graph=True),
    Strategy("structural_parent_only", "Structural Graph (Parent Only)",
             use_structured_graph=True, use_structural_parent_only=True),

    # ----- Single-graph + Reranker -----
    Strategy("semantic_reranker", "Semantic + Reranker",
             use_semantic_graph=True, use_reranker=True),
    Strategy("structural_reranker", "Structural + Reranker",
             use_structured_graph=True, use_reranker=True),

    # ----- Dual-graph (activates BFS Crawler C10) -----
    Strategy("both_graphs", "Semantic + Structural",
             use_semantic_graph=True, use_structured_graph=True),
    Strategy("both_reranker", "Both Graphs + Reranker",
             use_semantic_graph=True, use_structured_graph=True,
             use_reranker=True),
    Strategy("both_mmr07", "Both Graphs + MMR λ=0.7",
             use_semantic_graph=True, use_structured_graph=True,
             use_mmr_reranker=True, mmr_lambda=0.7),

    # ----- Query processing variants -----
    Strategy("decompose", "Question Decomposition",
             use_semantic_graph=True, use_structured_graph=True,
             use_question_decomposition=True),
#    Strategy("iterative", "Iterative Search",
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True),

    # ----- Full system variants -----
    Strategy("full_system", "Full System (All)",
             use_reranker=True,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
#    Strategy("full_mmr07", "Full + MMR λ=0.7",
#             use_mmr_reranker=True, mmr_lambda=0.7,
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=False),
#    Strategy("full_mmr05", "Full + MMR λ=0.5",
#             use_mmr_reranker=True, mmr_lambda=0.5,
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True),
#    Strategy("full_mmr09", "Full + MMR λ=0.9",
#             use_mmr_reranker=True, mmr_lambda=0.9,
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True),

    # ----- Graph-only + Iterative -----
#    Strategy("semantic_iterative", "Semantic + Iterative",
#             use_semantic_graph=True, use_iterative_search=True),
#    Strategy("structural_iterative", "Structural + Iterative",
#             use_structured_graph=True, use_iterative_search=True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(filename: str) -> Any:
    """Load a JSON file."""
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def ask_document(base_url: str, strategy: Strategy,
                 file_hash: str, question: str,
                 limit: int = 30, timeout: int = 300) -> Dict[str, Any]:
    """Call /ask-document with the given strategy."""
    payload = strategy.to_payload(file_hash, question, limit)

    # Send all graph/processing flags as query params so FastAPI picks
    # them up directly (API uses OR-logic: query_param OR body_field).
    query_params = {
        "use_semantic_graph": str(strategy.use_semantic_graph).lower(),
        "use_structured_graph": str(strategy.use_structured_graph).lower(),
        "use_structural_parent_only": str(strategy.use_structural_parent_only).lower(),
        "use_iterative_search": str(strategy.use_iterative_search).lower(),
        "use_question_decomposition": str(strategy.use_question_decomposition).lower(),
    }
    qs = "&".join(f"{k}={v}" for k, v in query_params.items())
    url = f"{base_url.rstrip('/')}/ask-document?{qs}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _build_result_entry(
    doc_id: str,
    question: str,
    correct_answer: str,
    result: Dict[str, Any],
    file_hash: str,
    model_answer_time: float,
    model_extract_time: float,
    extracted_res: Optional[str],
) -> dict:
    """Build a single result entry dict from an API response."""
    answers = result.get("answers", [])
    retrieves = []
    for elem in answers:
        retrieves.append({
            "qdrant_id": elem.get("element_index", 0),
            "type": elem.get("element_type", "unknown"),
            "file_hash": file_hash,
            "content": elem.get("text", ""),
            "page_idx": elem.get("page_idx", 0),
            "score": elem.get("score", 0),
        })

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": correct_answer,
        "file_hash": file_hash,
        "response": result.get("llm_answer", ""),
        "llm_answer": result.get("llm_answer", ""),
        "extracted_res": extracted_res,
        "retrieves": retrieves,
        "context_blocks": result.get("context_blocks", []),
        "status": "completed",
        "elapsed": round(model_answer_time, 2),
        "answers_count": len(answers),
        "model_answer_time": round(model_answer_time, 2),
        "model_extract_time": round(model_extract_time, 2),
    }


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_grid_test(
    base_url: str = "http://0.0.0.0:9191",
    dataset_path: str = "",
    file_hash_map_path: str = "",
    output_dir: str = "",
    limit: int = 30,
    strategy_filter: Optional[List[str]] = None,
    skip_extraction: bool = False,
    sort_reference: str = "",
    sort_by: str = "accuracy",
):
    """Run all (or filtered) strategies against the SmallerDataset.

    Parameters
    ----------
    base_url : str
        Base URL of the API server.
    dataset_path : str
        Path to SmallerDataset samples.json.
    file_hash_map_path : str
        Path to file_hash_comparison.json (maps doc_id → file_hash).
    output_dir : str
        Directory for per-strategy JSON result files.
    limit : int
        Max Qdrant results per query.
    strategy_filter : list of str or None
        If provided, only run strategies whose `name` is in this list.
    skip_extraction : bool
        If True, skip the Qwen answer-extraction step.
    sort_reference : str
        Path to a strategy_summary.json from a previous evaluation.
        If provided, strategies are reordered by the metric in sort_by
        (highest first), so the most promising strategies run first.
    sort_by : str
        Metric to sort strategies by when sort_reference is provided.
        One of: accuracy, f1, avg_score, avg_elapsed, priority.
    """
    # --- Resolve paths ---
    script_dir = Path(__file__).resolve().parent
    if not dataset_path:
        dataset_path = str(script_dir / "SmallerDataset" / "samples.json")
    if not file_hash_map_path:
        file_hash_map_path = str(script_dir / "file_hash_comparison.json")
    if not output_dir:
        output_dir = str(script_dir / "strategy_grid_results")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load dataset ---
    hash_map = read_json(file_hash_map_path)
    dataset = read_json(dataset_path)

    # --- Load extraction prompt (optional) ---
    extract_prompt_path = (
        script_dir / "MMLongDocEval" / "prompt_for_answer_extraction.md"
    )
    extract_prompt = ""
    if not skip_extraction and extract_prompt_path.exists():
        extract_prompt = extract_prompt_path.read_text(encoding="utf-8")

    # --- Filter strategies ---
    selected = STRATEGIES
    if strategy_filter:
        filter_set = set(strategy_filter)
        selected = [s for s in STRATEGIES if s.name in filter_set]
        if not selected:
            print(f"ERROR: no strategies matched filter {strategy_filter}")
            sys.exit(1)
        missing = filter_set - {s.name for s in selected}
        if missing:
            print(f"WARNING: unknown strategy names: {missing}")

    # --- Optional: reorder strategies by reference metrics ---
    if sort_reference and os.path.exists(sort_reference):
        ref_data = read_json(sort_reference)
        name_to_metric: Dict[str, dict] = {
            m["name"]: m for m in ref_data if isinstance(m, dict)
        }
        key_fn = SORT_KEYS.get(sort_by, SORT_KEYS["accuracy"])
        selected = sorted(
            selected,
            key=lambda s: key_fn(name_to_metric.get(s.name, {})),
            reverse=True,
        )
        print(
            f"Reordered {len(selected)} strategies by '{sort_by}' "
            f"from {sort_reference}"
        )
        # Print order for visibility
        for i, s in enumerate(selected):
            metrics = name_to_metric.get(s.name, {})
            acc = metrics.get("accuracy", "?")
            f1_val = metrics.get("f1", "?")
            t = metrics.get("avg_elapsed", "?")
            print(
                f"  {i + 1:2d}. {s.name:25s} "
                f"acc={acc}% f1={f1_val} time={t}s"
            )
        print()

    # --- Header ---
    print(f"Dataset:     {len(dataset)} questions")
    print(f"Strategies:  {len(selected)} of {len(STRATEGIES)} total")
    if strategy_filter:
        print(f"Filter:      {strategy_filter}")
    print(f"Output:      {output_dir}")
    print(f"Extraction:  {'enabled' if extract_prompt else 'skipped'}")
    print("=" * 70)

    all_strategy_results: Dict[str, List[dict]] = {}

    for si, strategy in enumerate(selected):
        print(f"\n{'=' * 70}")
        print(f"  [{si + 1}/{len(selected)}] {strategy.label}")
        print(f"  Flags: {', '.join(strategy.flags_summary())}")
        print(f"{'=' * 70}")

        strategy_results: List[dict] = []
        strategy_output = os.path.join(output_dir, f"{strategy.name}.json")

        # Resume from partial results if present
        if os.path.exists(strategy_output):
            strategy_results = read_json(strategy_output)
            print(f"  Resuming from {len(strategy_results)} already processed")

        for qi, case in enumerate(dataset):
            # Skip if already processed (check qi index and llm_answer)
            if (qi < len(strategy_results)
                    and strategy_results[qi].get("llm_answer")):
                continue

            doc_id = case.get("doc_id", "")
            question = case.get("question", "")
            correct_answer = case.get("answer", "")

            if doc_id not in hash_map:
                print(f"  [{qi}/{len(dataset)}] SKIP: {doc_id} "
                      f"not in hash map")
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
                model_answer_time = time.time() - start

                llm_answer = result.get("llm_answer", "") or ""

                # --- Optional: Extract short answer via Qwen API ---
                extract_start = time.time()
                extracted_res = None
                if llm_answer and extract_prompt and not skip_extraction:
                    try:
                        extracted_res = extract_answer_qwen_api(
                            question, llm_answer, extract_prompt,
                        )
                    except Exception as exc:
                        print(f"    Extraction failed: {exc}")
                        extracted_res = "Failed to extract"
                model_extract_time = time.time() - extract_start

                entry = _build_result_entry(
                    doc_id=doc_id,
                    question=question,
                    correct_answer=correct_answer,
                    result=result,
                    file_hash=file_hash,
                    model_answer_time=model_answer_time,
                    model_extract_time=model_extract_time,
                    extracted_res=extracted_res,
                )
                strategy_results.append(entry)

                answers_count = entry["answers_count"]
                print(
                    f"  [{qi}/{len(dataset)}] OK "
                    f"(answer: {model_answer_time:.1f}s, "
                    f"extract: {model_extract_time:.1f}s) - "
                    f"{len(llm_answer)} chars, {answers_count} retrieves"
                )
                if extracted_res:
                    print(f">>> Extracted answer:\n{extracted_res}\n")

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
                    json.dump(strategy_results, f, ensure_ascii=False,
                              indent=2)

        # --- Final save for this strategy ---
        with open(strategy_output, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, ensure_ascii=False, indent=2)

        all_strategy_results[strategy.name] = strategy_results
        completed = sum(
            1 for r in strategy_results if r.get("status") == "completed"
        )
        errors = sum(
            1 for r in strategy_results
            if r.get("status", "").startswith("error:")
        )
        print(f"  Done: {len(strategy_results)} results "
              f"({completed} ok, {errors} errors) → {strategy_output}")

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print("  Grid test complete.")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nTo evaluate results, run:")
    print(f"  python app/tests/evaluate_strategies.py "
          f"--results-dir {output_dir}")


# ---------------------------------------------------------------------------
# List strategies (--list-strategies)
# ---------------------------------------------------------------------------

def print_strategy_table(strategies: List[Strategy]) -> None:
    """Print a formatted table of all strategies."""
    print(f"\n{'=' * 70}")
    print(f"  Strategy Grid: {len(strategies)} combinations")
    print(f"{'=' * 70}")
    for i, s in enumerate(strategies):
        flags = s.flags_summary()
        print(
            f"  [{i + 1:2d}] {s.name:25s}  "
            f"{s.label:40s}  [{', '.join(flags)}]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Strategy Grid Test — run retrieval/context/generation "
                    "combinations against the SmallerDataset."
    )
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
        help="Directory for strategy result files",
    )
    ap.add_argument(
        "--limit", type=int, default=30,
        help="Number of retrieves per query",
    )
    ap.add_argument(
        "--strategies", default="",
        help="Comma-separated list of strategy names to run "
             "(default: all 20). E.g.: --strategies baseline,reranker",
    )
    ap.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip Qwen answer-extraction step",
    )
    ap.add_argument(
        "--sort-reference", default="",
        help="Path to strategy_summary.json from a previous evaluation. "
             "If provided, strategies are reordered by --sort-by metric "
             "(highest first) so the best candidates run first.",
    )
    ap.add_argument(
        "--sort-by",
        choices=SORT_BY_CHOICES,
        default="accuracy",
        help=(
            "Metric to sort strategies by when --sort-reference is used. "
            f"Choices: {', '.join(SORT_BY_CHOICES)}. "
            "(default: accuracy)"
        ),
    )
    ap.add_argument(
        "--list-strategies", action="store_true",
        help="Print the strategy grid and exit",
    )
    args = ap.parse_args()

    if args.list_strategies:
        print_strategy_table(STRATEGIES)
        sys.exit(0)

    strategy_filter = None
    if args.strategies:
        strategy_filter = [
            name.strip() for name in args.strategies.split(",") if name.strip()
        ]

    run_grid_test(
        base_url=args.base_url,
        dataset_path=args.dataset,
        file_hash_map_path=args.hash_map,
        output_dir=args.output_dir,
        limit=args.limit,
        strategy_filter=strategy_filter,
        skip_extraction=args.skip_extraction,
        sort_reference=args.sort_reference,
        sort_by=args.sort_by,
    )
