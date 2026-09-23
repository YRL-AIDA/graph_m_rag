#!/usr/bin/env python3
"""
Full-dataset (MMLongBench-Doc) Strategy Grid Test.

Runs the same strategy grid as ``strategy_grid_test.py`` (retrieval / context /
processing combinations) but against the FULL MMLongBench-Doc dataset (1082
questions) instead of the 105-question SmallerDataset.

The strategies are defined locally (copied from ``strategy_grid_test.py``) so
this script is self-contained and can be tweaked independently of the
SmallerDataset runner.

This is a TEST RUNNER only — it does NOT evaluate results. It saves one JSON
file per strategy, ready for the evaluation step.

Paths (from ``mmlongdoceval_via_api.py``):

* dataset    : /home/sunveil/Documents/projects/laba/graph-m-rag/data/MMLongBench-Doc/data/samples.json
* hash map   : app/tests/file_hash_comparison.json
* output dir : app/tests/mmlong_strategy_grid_results

Usage:
    python app/tests/mmlong_strategy_grid_test.py
    python app/tests/mmlong_strategy_grid_test.py --list-strategies
    python app/tests/mmlong_strategy_grid_test.py --strategies baseline,reranker
    python app/tests/mmlong_strategy_grid_test.py --base-url http://host:9191

Note on evaluation: score these results with ``mmlong_evaluate_strategies.py``
(the full-dataset counterpart of ``evaluate_strategies.py``), e.g.:

    python app/tests/mmlong_evaluate_strategies.py --results-dir mmlong_strategy_grid_results
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

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Strategy sorting keys (inlined from evaluate_strategies.py)
# ---------------------------------------------------------------------------

SORT_KEYS = {
    "accuracy": lambda m: m.get("accuracy", 0),
    "f1": lambda m: m.get("f1", 0),
    "avg_score": lambda m: m.get("avg_score", 0),
    "avg_elapsed": lambda m: -m.get("avg_elapsed", 0),
    "priority": lambda m: m.get("accuracy", 0) / (max(m.get("avg_elapsed", 1), 1) + 1),
}

SORT_BY_CHOICES = list(SORT_KEYS.keys())


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
    limit: Optional[int] = None  # per-strategy retrieve limit; None → use CLI --limit (default 30)
    use_reranker: bool = False
    reranker_min_relevance: Optional[float] = None  # top-1 score threshold; None=server default, 0.0=disabled
    use_mmr_reranker: bool = False
    mmr_lambda: float = 0.7
    mmr_min_relevance: float = 0.0
    use_semantic_graph: bool = False
    semantic_min_relevance: Optional[float] = None  # min score for semantic embedding-search entities/communities; None=server default, 0.0=disabled
    use_structured_graph: bool = False
    use_structural_parent_only: bool = False
    use_bfs_crawler: Optional[bool] = None   # Unified BFS crawler (Strategy E); None = auto (both graphs)
    use_iterative_search: bool = False
    use_question_decomposition: bool = False
    use_structured_context: bool = True   # XML-structured context vs flat plain-text
    use_neo4j_enrichment: bool = True     # enrich Qdrant regions with Neo4j parent/caption/footnote

    def to_payload(self, file_hash: str, question: str,
                   limit: int = 30) -> dict:
        """Build the JSON payload for /ask-document."""
        effective_limit = self.limit if self.limit is not None else limit
        return {
            "file_hash": file_hash,
            "question": question,
            "limit": effective_limit,
            "use_llm": True,
            "use_reranker": self.use_reranker,
            "reranker_min_relevance": self.reranker_min_relevance,
            "use_mmr_reranker": self.use_mmr_reranker,
            "mmr_lambda": self.mmr_lambda,
            "mmr_min_relevance": self.mmr_min_relevance,
            "use_semantic_graph": self.use_semantic_graph,
            "semantic_min_relevance": self.semantic_min_relevance,
            "use_structured_graph": self.use_structured_graph,
            "use_structural_parent_only": self.use_structural_parent_only,
            "use_bfs_crawler": self.use_bfs_crawler,
            "use_iterative_search": self.use_iterative_search,
            "use_question_decomposition": self.use_question_decomposition,
            "use_structured_context": self.use_structured_context,
            "use_neo4j_enrichment": self.use_neo4j_enrichment,
        }

    def flags_summary(self) -> List[str]:
        """Human-readable list of active flags for this strategy."""
        flags: List[str] = []
        if self.use_reranker:
            flags.append("reranker")
            if self.reranker_min_relevance:
                flags.append(f"reranker_minrel={self.reranker_min_relevance}")
        if self.use_mmr_reranker:
            flags.append(f"mmr(λ={self.mmr_lambda})")
        if self.use_semantic_graph:
            flags.append("semantic")
            if self.semantic_min_relevance:
                flags.append(f"semantic_minrel={self.semantic_min_relevance}")
        if self.use_structured_graph:
            if self.use_structural_parent_only:
                flags.append("structural(parent-only)")
            else:
                flags.append("structural")
        if self.use_bfs_crawler:
            flags.append("bfs")
        if self.use_iterative_search:
            flags.append("iterative")
        if self.use_question_decomposition:
            flags.append("decompose")
        if self.limit is not None:
            flags.append(f"limit={self.limit}")
        if not self.use_structured_context:
            flags.append("flat")
        if not self.use_neo4j_enrichment:
            flags.append("no-neo4j")
        return flags if flags else ["baseline"]


# --- Strategy definitions: minimal full-dataset experiment grid (L = 15) ---
#
# The list below contains ONLY the configurations required for the article's
# core scientific claims (plans/mmlong_full_results_and_experiment_plan_ru.md,
# sections 1-2.3): a controlled factorial ablation GRAPH x RERANKER plus the
# graph-free retrieval controls.  All strategies share the SAME retrieval
# limit (L = 15) and the same context budgets / generation settings, so any
# difference between two runs comes only from the mechanism under study.
# Each strategy NAME is the configuration identifier used in the plan and in
# the manuscript tables, and every run is saved to --output-dir/<name>.json:
#
#                     | no reranker            | cross-encoder reranker |
#   ------------------+------------------------+------------------------+
#   no graph          | baseline               | reranker               |
#   semantic graph    | semantic               | semantic_reranker      |
#   structural graph  | structural             | structural_reranker    |
#   both graphs + BFS | both_graphs            | both_reranker          |
#
# The three remaining graph-free ablations complete family A:
#   qdrant_only ..... vector search WITHOUT Neo4j parent/caption enrichment
#                     (isolates the enrichment step);
#   flat ............ XML context serialisation off (isolates the XML block
#                     layout);
#   mmr_07 .......... MMR post-search reranker (lambda=0.7) instead of the
#                     cross-encoder (isolates the reranker choice).
#
# Result files are written to --output-dir as <name>.json and RESUME by
# question index; already-computed runs whose configuration is unchanged
# (qdrant_only, baseline, flat, reranker, mmr_07, structural_reranker,
# both_reranker) are reused automatically by the resume logic.
#
# The remaining cells of the original 18-configuration design
# (structural_parent_only, both_mmr07, the iterative/decomposition family D
# and the full system, other MMR lambdas) are kept commented out at the
# bottom of STRATEGIES: they are NOT needed for the base claim, and the paper
# covers their absence explicitly (plan, sections 3.5 and 6).

STRATEGIES: List[Strategy] = [

    # ============ A. Retrieval & context controls (no graph) ============
    # Clean vector baseline: Qdrant hits only, NO Neo4j parent/caption
    # enrichment and no graph context.
    Strategy("qdrant_only", "Qdrant only (no graph, no enrichment)",
             use_neo4j_enrichment=False, limit=15),
    # Default pipeline: Qdrant hits enriched with Neo4j parents/captions;
    # the "no graph, no reranker" cell of the factor grid.
    Strategy("baseline", "Baseline (Qdrant + Neo4j enrichment)",
             limit=15),
    # Context serialisation: flat plain text instead of XML blocks.
    Strategy("flat", "Flat context (plain text, no XML)",
             use_structured_context=False, limit=15),
    # Post-search re-ranking with a cross-encoder reranker; the "no graph"
    # cell of the factor grid WITH reranker.
    Strategy("reranker", "Reranker (cross-encoder)",
             use_reranker=True, reranker_min_relevance=0.0, limit=15),
    # Post-search re-ranking with MMR (relevance/diversity, lambda=0.7).
    Strategy("mmr_07", "MMR (lambda=0.7)",
             use_mmr_reranker=True, mmr_lambda=0.7, limit=15),

    # ============ B. Graph augmentation (no reranker) ============
    # "No reranker" row of the factor grid.
    # Semantic graph: entity enrichment + embedding search over
    # entities/communities.
    Strategy("semantic", "Semantic graph",
             use_semantic_graph=True, limit=15),
    # Structural graph: reading-order walk + parents.
    Strategy("structural", "Structural graph (ORDER + Parent)",
             use_structured_graph=True, limit=15),
    # Both graphs -> activates the unified BFS crawler.
    Strategy("both_graphs", "Semantic + Structural (BFS)",
             use_semantic_graph=True, use_structured_graph=True, limit=15),

    # ============ C. Cross-encoder reranker on top of graphs ============
    # "With reranker" row of the factor grid.
    Strategy("semantic_reranker", "Semantic + Reranker",
             use_semantic_graph=True, use_reranker=True,
             reranker_min_relevance=0.0, limit=15),
    Strategy("structural_reranker", "Structural + Reranker",
             use_structured_graph=True, use_reranker=True,
             reranker_min_relevance=0.0, limit=15),
    Strategy("both_reranker", "Both graphs + Reranker (BFS)",
             use_semantic_graph=True, use_structured_graph=True,
             use_reranker=True, reranker_min_relevance=0.0, limit=15),

    # ----- Optional cells of the original 18-config design (uncomment) -----
    # Not required for the article's core claims (plan, sections 3.5 and 6):
#    # Structural graph, PARENT edges only -> isolates the ORDER contribution.
#    Strategy("structural_parent_only", "Structural graph (Parent only)",
#             use_structured_graph=True, use_structural_parent_only=True,
#             limit=15),
#    # MMR analogue of both_reranker.
#    Strategy("both_mmr07", "Both graphs + MMR (BFS, lambda=0.7)",
#             use_semantic_graph=True, use_structured_graph=True,
#             use_mmr_reranker=True, mmr_lambda=0.7, limit=15),
#    # ----- D. Query processing & full system (optional) -----
#    Strategy("semantic_iterative", "Semantic + Iterative",
#             use_semantic_graph=True, use_iterative_search=True, limit=15),
#    Strategy("structural_iterative", "Structural + Iterative",
#             use_structured_graph=True, use_iterative_search=True, limit=15),
#    Strategy("iterative", "Both graphs + Iterative (BFS)",
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True, limit=15),
#    Strategy("decompose", "Both graphs + Decomposition (BFS)",
#             use_semantic_graph=True, use_structured_graph=True,
#             use_question_decomposition=True, limit=15),
#    Strategy("full_system", "Full system (graphs + reranker + iterative)",
#             use_reranker=True, reranker_min_relevance=0.0,
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True, limit=15),
#    Strategy("mmr_05", "MMR (lambda=0.5)",
#             use_mmr_reranker=True, mmr_lambda=0.5, limit=15),
#    Strategy("mmr_09", "MMR (lambda=0.9)",
#             use_mmr_reranker=True, mmr_lambda=0.9, limit=15),
#    Strategy("structural_mmr07", "Structural + MMR (lambda=0.7)",
#             use_structured_graph=True, use_mmr_reranker=True,
#             mmr_lambda=0.7, limit=15),
#    Strategy("both_mmr05", "Both graphs + MMR (lambda=0.5, BFS)",
#             use_semantic_graph=True, use_structured_graph=True,
#             use_mmr_reranker=True, mmr_lambda=0.5, limit=15),
#    Strategy("full_mmr07", "Full system + MMR (lambda=0.7)",
#             use_mmr_reranker=True, mmr_lambda=0.7,
#             use_semantic_graph=True, use_structured_graph=True,
#             use_iterative_search=True, limit=15),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(filename: str) -> Any:
    """Load a JSON file."""
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(path: str, data: Any) -> None:
    """Write ``data`` to ``path`` atomically.

    The payload is written to a sibling ``.tmp`` file first and then moved
    into place with ``os.replace``. A crash or Ctrl+C mid-write can therefore
    never leave a truncated JSON file behind — the previous checkpoint stays
    intact until the new one is fully on disk.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _salvage_json_array(path: str) -> Optional[List[dict]]:
    """Recover complete list entries from a JSON file that was truncated.

    Returns ``None`` when nothing usable can be recovered. This is only a
    safety net for files written by older non-atomic checkpoints (and for the
    interrupted writes that produced them); all new writes go through
    ``_save_json_atomic``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    # Fast path: the file may already be valid JSON.
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass

    # Find where every top-level object closes (brace depth returning to 0,
    # ignoring braces inside string literals).
    cut_points: List[int] = []
    depth = 0
    in_str = False
    esc = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cut_points.append(i)
        i += 1

    # Try the longest prefix first: the most complete entries we can keep.
    for end in reversed(cut_points):
        candidate = text[: end + 1].rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1]
        try:
            data = json.loads(candidate + "]")
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
    return None


def _load_existing_results(path: str) -> List[dict]:
    """Load a per-strategy results file for resume.

    Handles files that were truncated/corrupted by an interrupted write: the
    complete entries are salvaged and the missing tail is re-run instead of
    crashing. Keeps the result list 1:1 aligned with the dataset order.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = _salvage_json_array(path)
        if data is None:
            print(f"  WARNING: {os.path.basename(path)} is corrupt and could "
                  f"not be salvaged — restarting it from scratch")
            return []
        print(f"  WARNING: {os.path.basename(path)} was truncated/corrupt; "
              f"recovered {len(data)} complete entries and will resume from there")

    if not isinstance(data, list):
        print(f"  WARNING: {os.path.basename(path)} does not contain a JSON "
              f"list ({type(data).__name__}); restarting it from scratch")
        return []

    # Entries recorded as "completed" by an older run but with no model output
    # are failed generations: convert them so the resume below retries them and
    # the statistics count them as errors instead of silently dropping them.
    converted = 0
    for r in data:
        if (isinstance(r, dict)
                and str(r.get("status")) == "completed"
                and not r.get("llm_answer")
                and not r.get("response")):
            r["status"] = "error"
            r["error"] = True
            r["retryable"] = True
            r.setdefault("error_msg", "Empty LLM answer (no model output)")
            converted += 1
    if converted:
        print(f"  NOTE: {os.path.basename(path)} contains {converted} entries "
              f"marked 'completed' but with no model output — marking them as "
              f"errors so they are retried")
    return data


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
    # BFS is Optional[bool]: only send an explicit override when the strategy
    # pins it; otherwise the server keeps legacy auto behaviour (both graphs).
    if strategy.use_bfs_crawler is not None:
        query_params["use_bfs_crawler"] = str(strategy.use_bfs_crawler).lower()
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
# Full-dataset paths (from mmlongdoceval_via_api.py)
# ---------------------------------------------------------------------------

DEFAULT_DATASET = (
    "/home/sunveil/Documents/projects/laba/graph-m-rag/"
    "data/MMLongBench-Doc/data/samples.json"
)
DEFAULT_HASH_MAP = str(SCRIPT_DIR / "file_hash_comparison.json")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "mmlong_strategy_grid_results")


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
    """Run all (or filtered) strategies against the full MMLongBench-Doc dataset.

    Parameters
    ----------
    base_url : str
        Base URL of the API server.
    dataset_path : str
        Path to the full dataset samples.json.
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
    if not dataset_path:
        dataset_path = DEFAULT_DATASET
    if not file_hash_map_path:
        file_hash_map_path = DEFAULT_HASH_MAP
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # --- Load dataset ---
    hash_map = read_json(file_hash_map_path)
    dataset = read_json(dataset_path)

    # --- Load extraction prompt (optional) ---
    extract_prompt_path = (
        SCRIPT_DIR / "MMLongDocEval" / "prompt_for_answer_extraction.md"
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

        # Resume from partial results if present. The list stays 1:1 aligned
        # with the dataset order: entries that already produced an LLM answer
        # (or that will never be answerable) are skipped, while entries marked
        # as generation errors are retried and REPLACED in place on success.
        if os.path.exists(strategy_output):
            strategy_results = _load_existing_results(strategy_output)
            # Trim trailing junk from pre-fix misaligned resumes.
            if len(strategy_results) > len(dataset):
                strategy_results = strategy_results[: len(dataset)]
            res_completed = sum(
                1 for r in strategy_results if r.get("llm_answer")
            )
            res_errors = sum(
                1 for r in strategy_results
                if str(r.get("status", "")).startswith("error")
            )
            print(f"  Resuming from {len(strategy_results)} entries "
                  f"({res_completed} ok, {res_errors} errors to retry)")

        for qi, case in enumerate(dataset):
            # --- Resume decision -------------------------------------------------
            prev = strategy_results[qi] if qi < len(strategy_results) else None
            if prev is not None:
                prev_status = str(prev.get("status", ""))
                if (prev.get("llm_answer")
                        or prev.get("response")
                        or prev_status == "hash_not_found"):
                    # Already answered or permanently skipped — keep as is.
                    # NOTE: an entry whose status is "completed" but that has no
                    # model output is NOT skipped here — it is retried below,
                    # because an empty answer is a failed generation.
                    continue

            doc_id = case.get("doc_id", "")
            question = case.get("question", "")
            correct_answer = case.get("answer", "")

            entry: Optional[dict] = None

            if doc_id not in hash_map:
                print(f"  [{qi}/{len(dataset)}] SKIP: {doc_id} "
                      f"not in hash map")
                entry = {
                    "doc_id": doc_id,
                    "question": question,
                    "answer": correct_answer,
                    "status": "hash_not_found",
                }
            else:
                file_hash = hash_map[doc_id]
                start = time.time()

                try:
                    result = ask_document(
                        base_url, strategy, file_hash, question, limit,
                    )
                    model_answer_time = time.time() - start

                    llm_answer = result.get("llm_answer", "") or ""

                    # An API response without any model output is NOT a
                    # successful generation: mark it as an error so the entry
                    # is counted as such and retried on the next resume.
                    if not llm_answer and not result.get("response"):
                        raise RuntimeError(
                            "Empty LLM answer (model returned no output)"
                        )

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
                    # Generation/transport error: mark the entry explicitly so
                    # the evaluation statistics can count it and a later resume
                    # retries it in place.
                    elapsed = time.time() - start
                    print(f"  [{qi}/{len(dataset)}] ERROR: {exc}")
                    entry = {
                        "doc_id": doc_id,
                        "question": question,
                        "answer": correct_answer,
                        "status": "error",
                        "error": True,
                        "error_msg": str(exc),
                        "retryable": True,
                        "elapsed": round(elapsed, 2),
                    }

            if entry is not None:
                # Replace the old (error/partial) entry in place so index
                # alignment with the dataset is preserved; append only when the
                # list is shorter than the current question index.
                if prev is not None:
                    strategy_results[qi] = entry
                else:
                    strategy_results.append(entry)

            # Save checkpoint every 10 questions (atomic write)
            if (qi + 1) % 10 == 0:
                _save_json_atomic(strategy_output, strategy_results)

        # --- Final save for this strategy ---
        _save_json_atomic(strategy_output, strategy_results)

        all_strategy_results[strategy.name] = strategy_results
        completed = sum(
            1 for r in strategy_results if r.get("status") == "completed"
        )
        errors = sum(
            1 for r in strategy_results
            if str(r.get("status", "")).startswith("error")
        )
        print(f"  Done: {len(strategy_results)} results "
              f"({completed} ok, {errors} errors) → {strategy_output}")

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print("  Grid test complete.")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nTo evaluate results, run:")
    print(f"  python app/tests/mmlong_evaluate_strategies.py "
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

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Full-dataset (MMLongBench-Doc) strategy grid test — "
                    "runs retrieval/context/generation combinations against "
                    "the full 1082-question dataset."
    )
    ap.add_argument(
        "--base-url", default="http://0.0.0.0:9191",
        help="Base URL of the API server",
    )
    ap.add_argument(
        "--dataset", default="",
        help=f"Path to the full dataset samples.json "
             f"(default: {DEFAULT_DATASET})",
    )
    ap.add_argument(
        "--hash-map", default="",
        help=f"Path to file_hash_comparison.json "
             f"(default: {DEFAULT_HASH_MAP})",
    )
    ap.add_argument(
        "--output-dir", default="",
        help=f"Directory for per-strategy result files "
             f"(default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--limit", type=int, default=30,
        help="Number of retrieves per query",
    )
    ap.add_argument(
        "--strategies", default="",
        help="Comma-separated list of strategy names to run "
             "(default: all). E.g.: --strategies baseline,reranker",
    )
    ap.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip Qwen answer-extraction step",
    )
    ap.add_argument(
        "--sort-reference", default="",
        help="Path to a strategy_summary.json from a previous evaluation. "
             "If provided, strategies are reordered by --sort-by metric "
             "(highest first) so the best candidates run first.",
    )
    ap.add_argument(
        "--sort-by",
        choices=SORT_BY_CHOICES,
        default="accuracy",
        help=(
            "Metric to sort strategies by when --sort-reference is used. "
            f"Choices: {', '.join(SORT_BY_CHOICES)}. (default: accuracy)"
        ),
    )
    ap.add_argument(
        "--list-strategies", action="store_true",
        help="Print the strategy grid and exit",
    )
    args = ap.parse_args()

    if args.list_strategies:
        print_strategy_table(STRATEGIES)
        return 0

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
