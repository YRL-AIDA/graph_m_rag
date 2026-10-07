#!/usr/bin/env python3
"""
Single Strategy Consistency Test — runs one strategy against one question
10 times and produces an HTML report showing answer stability and timing.

Usage:
    python app/tests/consistency_test.py
    python app/tests/consistency_test.py --strategy baseline
    python app/tests/consistency_test.py --strategy full_system --question-index 5
    python app/tests/consistency_test.py --base-url http://host:9191 --runs 20
    python app/tests/consistency_test.py --list-questions
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Strategy definitions (same as strategy_grid_test.py)
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A single strategy configuration for the ask-document endpoint."""

    name: str
    label: str
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


STRATEGIES: List[Strategy] = [
    Strategy("baseline", "Baseline (Qdrant + LLM)"),
    Strategy("reranker", "Reranker", use_reranker=True),
    Strategy("mmr_07", "MMR (λ=0.7)", use_mmr_reranker=True, mmr_lambda=0.7),
    Strategy("mmr_05", "MMR (λ=0.5)", use_mmr_reranker=True, mmr_lambda=0.5),
    Strategy("mmr_09", "MMR (λ=0.9)", use_mmr_reranker=True, mmr_lambda=0.9),
    Strategy("semantic", "Semantic Graph", use_semantic_graph=True),
    Strategy("structural", "Structural Graph (ORDER + Parent)",
             use_structured_graph=True),
    Strategy("structural_parent_only", "Structural Graph (Parent Only)",
             use_structured_graph=True, use_structural_parent_only=True),
    Strategy("both_graphs", "Semantic + Structural",
             use_semantic_graph=True, use_structured_graph=True),
    Strategy("both_reranker", "Both Graphs + Reranker",
             use_semantic_graph=True, use_structured_graph=True,
             use_reranker=True),
    Strategy("both_mmr07", "Both Graphs + MMR λ=0.7",
             use_semantic_graph=True, use_structured_graph=True,
             use_mmr_reranker=True, mmr_lambda=0.7),
    Strategy("semantic_reranker", "Semantic + Reranker",
             use_semantic_graph=True, use_reranker=True),
    Strategy("structural_reranker", "Structural + Reranker",
             use_structured_graph=True, use_reranker=True),
    Strategy("decompose", "Question Decomposition",
             use_semantic_graph=True, use_structured_graph=True,
             use_question_decomposition=True),
    Strategy("iterative", "Iterative Search",
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_system", "Full System (All)",
             use_reranker=True,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_mmr07", "Full + MMR λ=0.7",
             use_mmr_reranker=True, mmr_lambda=0.7,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_mmr05", "Full + MMR λ=0.5",
             use_mmr_reranker=True, mmr_lambda=0.5,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("full_mmr09", "Full + MMR λ=0.9",
             use_mmr_reranker=True, mmr_lambda=0.9,
             use_semantic_graph=True, use_structured_graph=True,
             use_iterative_search=True),
    Strategy("semantic_iterative", "Semantic + Iterative",
             use_semantic_graph=True, use_iterative_search=True),
    Strategy("structural_iterative", "Structural + Iterative",
             use_structured_graph=True, use_iterative_search=True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(filename: str) -> Any:
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def ask_document(base_url: str, strategy: Strategy,
                 file_hash: str, question: str,
                 limit: int = 30, timeout: int = 300) -> Dict[str, Any]:
    """Call /ask-document with the given strategy."""
    payload = strategy.to_payload(file_hash, question, limit)
    query_params = {
        "use_semantic_graph": str(strategy.use_semantic_graph).lower(),
        "use_structured_graph": str(strategy.use_structured_graph).lower(),
        "use_structural_parent_only": str(
            strategy.use_structural_parent_only
        ).lower(),
        "use_iterative_search": str(strategy.use_iterative_search).lower(),
        "use_question_decomposition": str(
            strategy.use_question_decomposition
        ).lower(),
    }
    qs = "&".join(f"{k}={v}" for k, v in query_params.items())
    url = f"{base_url.rstrip('/')}/ask-document?{qs}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


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
        messages=[tt],
        base_url=os.environ.get(
            'QWEN_EXTRACT_URL', 'http://192.168.19.127:8888/v1'
        ),
    )
    return result[1][0]


def compute_stats(values: List[float]) -> dict:
    """Compute basic statistics for a list of numeric values."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (
        sorted_vals[n // 2 - 1] + sorted_vals[n // 2]
    ) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(variance ** 0.5, 2),
    }


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def generate_html_report(
    results: List[dict],
    strategy: Strategy,
    case: dict,
    file_hash: str,
    stats: dict,
) -> str:
    """Generate a self-contained HTML report."""
    question = case.get("question", "")
    correct_answer = case.get("answer", "")
    doc_id = case.get("doc_id", "")
    answer_format = case.get("answer_format", "Str")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build answer rows
    answer_rows = ""
    all_answers: List[str] = []
    all_extracted: List[str] = []
    for i, r in enumerate(results):
        llm_answer = r.get("llm_answer", "")
        extracted = r.get("extracted_res", "")
        elapsed = r.get("elapsed", 0)
        retrieves = r.get("retrieves_count", 0)

        if llm_answer:
            all_answers.append(llm_answer.strip().lower())
        if extracted:
            all_extracted.append(extracted.strip())

        status_icon = "✅" if r.get("status") == "completed" else "❌"
        row_class = ""
        if r.get("status", "").startswith("error"):
            row_class = ' class="error-row"'

        answer_rows += f"""
            <tr{row_class}>
                <td class="center">{status_icon}</td>
                <td class="center">{i + 1}</td>
                <td class="mono right">{elapsed:.1f}s</td>
                <td class="center">{retrieves}</td>
                <td>
                    <div class="answer-scroll">{_escape_html(llm_answer)}</div>
                </td>
            </tr>"""

    # Extracted answers table
    extracted_rows = ""
    if all_extracted:
        for i, ex in enumerate(all_extracted):
            extracted_rows += f"""
                <tr>
                    <td class="center">{i + 1}</td>
                    <td class="mono">{_escape_html(ex)}</td>
                </tr>"""
    else:
        extracted_rows = (
            '<tr><td colspan="2" class="center muted">'
            '(extraction skipped)</td></tr>'
        )

    # Answer uniqueness stats
    from collections import Counter
    raw_counter = Counter(all_answers) if all_answers else {}
    ext_counter = Counter(all_extracted) if all_extracted else {}

    unique_raw = len(raw_counter)
    unique_ext = len(ext_counter)
    total_runs = len(results)

    # Most common answer
    most_common_raw = raw_counter.most_common(1)
    most_common_raw_str = (
        _escape_html(most_common_raw[0][0][:200])
        if most_common_raw else "—"
    )
    most_common_raw_count = most_common_raw[0][1] if most_common_raw else 0

    most_common_ext = ext_counter.most_common(1)
    most_common_ext_str = (
        _escape_html(most_common_ext[0][0][:200])
        if most_common_ext else "—"
    )
    most_common_ext_count = most_common_ext[0][1] if most_common_ext else 0

    # Build distribution bars for raw answers
    dist_bars = ""
    for answer_text, count in raw_counter.most_common():
        pct = count / total_runs * 100 if total_runs else 0
        short = _escape_html(answer_text[:150])
        dist_bars += f"""
            <div class="dist-row">
                <span class="dist-label" title="{_escape_html(answer_text)}">{short}</span>
                <div class="dist-bar-wrap">
                    <div class="dist-bar" style="width:{pct:.0f}%"></div>
                    <span class="dist-count">{count}/{total_runs} ({pct:.0f}%)</span>
                </div>
            </div>"""

    # Timing stats
    t_stats = stats.get("time", {})
    r_stats = stats.get("retrieves", {})
    c_stats = stats.get("context_blocks", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Consistency Report — {_escape_html(strategy.label)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa; color: #1a1a2e; line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 24px; }}
  .card {{
    background: #fff; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    padding: 20px 24px; margin-bottom: 20px;
  }}
  .card h2 {{
    font-size: 1.15rem; margin-bottom: 14px;
    padding-bottom: 8px; border-bottom: 2px solid #e8ecf1;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .stat {{ text-align: center; padding: 12px 8px; background: #f8fafc; border-radius: 8px; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #2563eb; }}
  .stat-label {{ font-size: 0.8rem; color: #666; margin-top: 4px; }}
  .stat-label.warn {{ color: #d97706; }}
  .stat-label.good {{ color: #059669; }}
  .stat-label.bad {{ color: #dc2626; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e8ecf1; }}
  th {{ background: #f8fafc; font-weight: 600; color: #555; white-space: nowrap; }}
  .center {{ text-align: center; }}
  .right {{ text-align: right; }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; }}
  .muted {{ color: #999; }}

  .error-row {{ background: #fef2f2; }}
  .error-row td {{ color: #b91c1c; }}

  .answer-scroll {{
    max-height: 120px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
    font-size: 0.8rem; line-height: 1.4;
    background: #f9fafb; padding: 6px 10px; border-radius: 6px;
    border: 1px solid #e5e7eb;
  }}

  .dist-row {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px;
  }}
  .dist-label {{
    width: 200px; flex-shrink: 0;
    font-size: 0.8rem; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
    color: #333;
  }}
  .dist-bar-wrap {{
    flex: 1; display: flex; align-items: center; gap: 8px;
  }}
  .dist-bar {{
    height: 18px; background: linear-gradient(90deg, #3b82f6, #60a5fa);
    border-radius: 4px; min-width: 2px;
    transition: width .3s;
  }}
  .dist-count {{
    font-size: 0.78rem; color: #555; white-space: nowrap;
    min-width: 70px;
  }}

  .info-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; }}
  .info-label {{ font-weight: 600; color: #555; font-size: 0.85rem; }}
  .info-value {{ font-size: 0.85rem; color: #1a1a2e; word-break: break-word; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
    background: #e0e7ff; color: #3730a3;
  }}

  @media (max-width: 700px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    .dist-label {{ width: 100px; }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>🔬 Consistency Test Report</h1>
<p class="subtitle">
  Strategy: <strong>{_escape_html(strategy.label)}</strong>
  &nbsp;·&nbsp; {total_runs} runs
  &nbsp;·&nbsp; Generated {now_utc}
</p>

<!-- ====== STRATEGY & QUESTION INFO ====== -->
<div class="card">
  <h2>📋 Strategy & Question</h2>
  <div class="grid-2">
    <div>
      <div class="info-grid">
        <span class="info-label">Strategy:</span>
        <span class="info-value">{_escape_html(strategy.label)}</span>
        <span class="info-label">Flags:</span>
        <span class="info-value">
          {"".join(f'<span class="badge">{f}</span> ' for f in strategy.flags_summary())}
        </span>
        <span class="info-label">File Hash:</span>
        <span class="info-value mono">{_escape_html(file_hash)}</span>
        <span class="info-label">Doc ID:</span>
        <span class="info-value">{_escape_html(doc_id)}</span>
      </div>
    </div>
    <div>
      <div class="info-grid">
        <span class="info-label">Correct Answer:</span>
        <span class="info-value mono" style="font-weight:700;">{_escape_html(correct_answer)}</span>
        <span class="info-label">Format:</span>
        <span class="info-value">{_escape_html(answer_format)}</span>
      </div>
    </div>
  </div>
  <div style="margin-top:12px;">
    <span class="info-label">Question:</span>
    <div style="margin-top:4px; font-style:italic; color:#333;">
      {_escape_html(question)}
    </div>
  </div>
</div>

<!-- ====== CONSISTENCY STATS ====== -->
<div class="card">
  <h2>📊 Answer Consistency</h2>
  <div class="grid-3">
    <div class="stat">
      <div class="stat-value">{unique_raw}</div>
      <div class="stat-label">Unique raw answers</div>
    </div>
    <div class="stat">
      <div class="stat-value">{most_common_raw_count}/{total_runs}</div>
      <div class="stat-label {"good" if most_common_raw_count >= total_runs * 0.8 else "warn" if most_common_raw_count >= total_runs * 0.5 else "bad"}">
        Top answer frequency
      </div>
    </div>
    <div class="stat">
      <div class="stat-value">{unique_ext}</div>
      <div class="stat-label">Unique extracted answers</div>
    </div>
  </div>

  <h3 style="margin-top:20px; font-size:1rem;">Raw Answer Distribution</h3>
  {dist_bars if dist_bars else '<p class="muted">No answers recorded</p>'}
</div>

<!-- ====== TIMING STATS ====== -->
<div class="card">
  <h2>⏱️ Performance</h2>
  <div class="grid-3">
    <div class="stat">
      <div class="stat-value">{t_stats.get('mean', 0)}s</div>
      <div class="stat-label">Mean time</div>
    </div>
    <div class="stat">
      <div class="stat-value">{t_stats.get('min', 0)}s</div>
      <div class="stat-label">Min time</div>
    </div>
    <div class="stat">
      <div class="stat-value">{t_stats.get('max', 0)}s</div>
      <div class="stat-label">Max time</div>
    </div>
    <div class="stat">
      <div class="stat-value">{t_stats.get('std', 0)}s</div>
      <div class="stat-label">Std deviation</div>
    </div>
    <div class="stat">
      <div class="stat-value">{t_stats.get('median', 0)}s</div>
      <div class="stat-label">Median time</div>
    </div>
    <div class="stat">
      <div class="stat-value">{r_stats.get('mean', 0)}</div>
      <div class="stat-label">Avg retrieves</div>
    </div>
  </div>
  <p class="muted" style="margin-top:12px; font-size:0.8rem;">
    Context blocks: mean={c_stats.get('mean', 0)}, min={c_stats.get('min', 0)}, max={c_stats.get('max', 0)}, std={c_stats.get('std', 0)}
  </p>
</div>

<!-- ====== PER-RUN ANSWERS ====== -->
<div class="card">
  <h2>📝 Per-Run Answers ({total_runs} runs)</h2>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th></th>
        <th>#</th>
        <th>Time</th>
        <th>Retr</th>
        <th>LLM Answer</th>
      </tr>
    </thead>
    <tbody>
      {answer_rows}
    </tbody>
  </table>
  </div>
</div>

<!-- ====== EXTRACTED ANSWERS ====== -->
<div class="card">
  <h2>🔍 Extracted Answers</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Extracted</th></tr>
    </thead>
    <tbody>
      {extracted_rows}
    </tbody>
  </table>
</div>

<!-- ====== RAW RESULTS JSON ====== -->
<div class="card">
  <h2>📄 Raw Results (JSON)</h2>
  <details>
    <summary style="cursor:pointer; font-weight:600;">Click to expand</summary>
    <pre style="max-height:500px; overflow:auto; font-size:0.75rem;
                background:#1e1e2e; color:#cdd6f4; padding:12px;
                border-radius:6px; margin-top:10px;">{_escape_html(
                    json.dumps(results, ensure_ascii=False, indent=2)
                )}</pre>
  </details>
</div>

</div>
</body>
</html>"""
    return html


def _escape_html(text: str) -> str:
    """Escape text for safe HTML embedding."""
    if text is None:
        return ""
    return (
        str(text)
        .replace(chr(38), chr(38) + "amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_consistency_test(
    base_url: str = "http://0.0.0.0:9191",
    dataset_path: str = "",
    file_hash_map_path: str = "",
    output_dir: str = "",
    strategy_name: str = "structural_parent_only",
    question_index: int = 0,
    num_runs: int = 10,
    limit: int = 30,
    skip_extraction: bool = False,
    timeout: int = 300,
):
    """Run one strategy against one question N times.

    Parameters
    ----------
    base_url : str
        Base URL of the API server.
    dataset_path : str
        Path to SmallerDataset samples.json.
    file_hash_map_path : str
        Path to file_hash_comparison.json (maps doc_id to file_hash).
    output_dir : str
        Directory for the output HTML + JSON files.
    strategy_name : str
        Name of the strategy to test (must be in STRATEGIES).
    question_index : int
        Index into the dataset (0-based).
    num_runs : int
        Number of times to ask the same question.
    limit : int
        Max Qdrant results per query.
    skip_extraction : bool
        If True, skip the Qwen answer-extraction step.
    timeout : int
        Request timeout in seconds.
    """
    # --- Resolve paths ---
    script_dir = Path(__file__).resolve().parent
    if not dataset_path:
        dataset_path = str(script_dir / "SmallerDataset" / "samples.json")
    if not file_hash_map_path:
        file_hash_map_path = str(script_dir / "file_hash_comparison.json")
    if not output_dir:
        output_dir = str(script_dir / "consistency_results")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load dataset ---
    hash_map = read_json(file_hash_map_path)
    dataset = read_json(dataset_path)

    # --- Select strategy ---
    strategy_map = {s.name: s for s in STRATEGIES}
    if strategy_name not in strategy_map:
        print(f"ERROR: unknown strategy '{strategy_name}'")
        print(f"Available: {list(strategy_map.keys())}")
        sys.exit(1)
    strategy = strategy_map[strategy_name]

    # --- Select question ---
    if question_index < 0 or question_index >= len(dataset):
        print(
            f"ERROR: question_index {question_index} out of range "
            f"(0–{len(dataset) - 1})"
        )
        sys.exit(1)
    case = dataset[question_index]
    doc_id = case.get("doc_id", "")
    question = case.get("question", "")
    correct_answer = case.get("answer", "")
    answer_format = case.get("answer_format", "Str")

    if doc_id not in hash_map:
        print(f"ERROR: doc_id '{doc_id}' not found in file_hash_map")
        sys.exit(1)
    file_hash = hash_map[doc_id]

    # --- Load extraction prompt ---
    extract_prompt_path = (
        script_dir / "MMLongDocEval" / "prompt_for_answer_extraction.md"
    )
    extract_prompt = ""
    if not skip_extraction and extract_prompt_path.exists():
        extract_prompt = extract_prompt_path.read_text(encoding="utf-8")

    # --- Print summary ---
    print("=" * 70)
    print(f"  Consistency Test")
    print("=" * 70)
    print(f"  Strategy:      {strategy.label}")
    print(f"  Flags:         {', '.join(strategy.flags_summary())}")
    print(f"  Doc ID:        {doc_id}")
    print(f"  File Hash:     {file_hash}")
    print(f"  Question:      {question}")
    print(f"  Correct Ans:   {correct_answer} ({answer_format})")
    print(f"  Runs:          {num_runs}")
    print(f"  Output:        {output_dir}/")
    print("=" * 70)

    # --- Run iterations ---
    results: List[dict] = []
    timings: List[float] = []
    retrieves_counts: List[int] = []
    context_block_counts: List[int] = []

    for run_i in range(num_runs):
        print(f"\n  [{run_i + 1}/{num_runs}] ", end="", flush=True)
        start = time.time()

        try:
            api_result = ask_document(
                base_url, strategy, file_hash, question, limit, timeout,
            )
            elapsed = time.time() - start
            timings.append(elapsed)

            llm_answer = api_result.get("llm_answer", "")
            answers_list = api_result.get("answers", [])
            retrieves = len(answers_list)
            retrieves_counts.append(retrieves)

            # Count context blocks from the response
            context_blocks = api_result.get("context_blocks", [])
            ctx_count = len(context_blocks) if context_blocks else 0
            context_block_counts.append(ctx_count)

            # --- Optional extraction ---
            extract_start = time.time()
            extracted_res = None
            extract_elapsed = 0.0
            if llm_answer and extract_prompt and not skip_extraction:
                try:
                    extracted_res = extract_answer_qwen_api(
                        question, llm_answer, extract_prompt,
                    )
                    extract_elapsed = time.time() - extract_start
                except Exception as exc:
                    print(f"extraction err: {exc}", flush=True)
                    extracted_res = "Failed to extract"
                    extract_elapsed = time.time() - extract_start

            entry = {
                "run": run_i + 1,
                "question": question,
                "correct_answer": correct_answer,
                "answer_format": answer_format,
                "llm_answer": llm_answer,
                "extracted_res": extracted_res,
                "elapsed": round(elapsed, 2),
                "extract_elapsed": round(extract_elapsed, 2),
                "retrieves_count": retrieves,
                "context_blocks": ctx_count,
                "status": "completed",
            }
            results.append(entry)

            preview = (
                llm_answer[:120].replace("\n", " ")
                if llm_answer else "(empty)"
            )
            print(
                f"OK  {elapsed:.1f}s  retr={retrieves}  ctx={ctx_count}  "
                f"extr={extracted_res}  →  {preview}",
                flush=True,
            )

        except Exception as exc:
            elapsed = time.time() - start
            timings.append(elapsed)
            print(f"ERROR: {exc}", flush=True)
            results.append({
                "run": run_i + 1,
                "question": question,
                "correct_answer": correct_answer,
                "answer_format": answer_format,
                "llm_answer": "",
                "extracted_res": None,
                "elapsed": round(elapsed, 2),
                "extract_elapsed": 0,
                "retrieves_count": 0,
                "context_blocks": 0,
                "status": f"error: {exc}",
            })

    # Collect server-side timing breakdowns
    server_search_ms = [
        r["server_timing"].get("search_ms", 0)
        for r in results
        if r.get("server_timing") and isinstance(r["server_timing"], dict)
    ]
    server_enrichment_ms = [
        r["server_timing"].get("enrichment_ms", 0)
        for r in results
        if r.get("server_timing") and isinstance(r["server_timing"], dict)
    ]
    server_llm_ms = [
        r["server_timing"].get("llm_generation_ms", 0)
        for r in results
        if r.get("server_timing") and isinstance(r["server_timing"], dict)
    ]
    server_total_ms = [
        r["server_timing"].get("total_ms", 0)
        for r in results
        if r.get("server_timing") and isinstance(r["server_timing"], dict)
    ]

    # --- Compute stats ---
    stats = {
        "time": compute_stats(timings),
        "retrieves": compute_stats(
            [float(r) for r in retrieves_counts] if retrieves_counts else [0]
        ),
        "context_blocks": compute_stats(
            [float(c) for c in context_block_counts]
            if context_block_counts else [0]
        ),
        "server_timing": {
            "search_ms": compute_stats(
                [float(s) for s in server_search_ms]
            ) if server_search_ms else None,
            "enrichment_ms": compute_stats(
                [float(s) for s in server_enrichment_ms]
            ) if server_enrichment_ms else None,
            "llm_generation_ms": compute_stats(
                [float(s) for s in server_llm_ms]
            ) if server_llm_ms else None,
            "total_ms": compute_stats(
                [float(s) for s in server_total_ms]
            ) if server_total_ms else None,
        },
        "completed": sum(
            1 for r in results if r.get("status") == "completed"
        ),
        "errors": sum(
            1 for r in results
            if r.get("status", "").startswith("error")
        ),
    }

    # --- Save JSON ---
    json_path = os.path.join(
        output_dir,
        f"consistency_{strategy.name}_q{question_index}.json",
    )
    json_output = {
        "meta": {
            "strategy": strategy.name,
            "strategy_label": strategy.label,
            "flags": strategy.flags_summary(),
            "doc_id": doc_id,
            "file_hash": file_hash,
            "question": question,
            "correct_answer": correct_answer,
            "answer_format": answer_format,
            "num_runs": num_runs,
            "question_index": question_index,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "stats": stats,
        "runs": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON saved → {json_path}")

    # --- Generate HTML ---
    html = generate_html_report(results, strategy, case, file_hash, stats)
    html_path = os.path.join(
        output_dir,
        f"consistency_{strategy.name}_q{question_index}.html",
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {html_path}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  Complete: {stats['completed']}/{num_runs} ok, "
          f"{stats['errors']} errors")
    print(f"  Time:     mean={stats['time']['mean']}s  "
          f"min={stats['time']['min']}s  max={stats['time']['max']}s  "
          f"std={stats['time']['std']}s")
    print(f"  Retr:     mean={stats['retrieves']['mean']}")
    print(f"  Context:  mean={stats['context_blocks']['mean']}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# List questions helper
# ---------------------------------------------------------------------------

def list_questions(dataset_path: str = ""):
    """Print all questions from the dataset with their indices."""
    script_dir = Path(__file__).resolve().parent
    if not dataset_path:
        dataset_path = str(script_dir / "SmallerDataset" / "samples.json")
    dataset = read_json(dataset_path)

    print(f"\n{'=' * 70}")
    print(f"  Dataset Questions ({len(dataset)} total)")
    print(f"{'=' * 70}")
    for i, case in enumerate(dataset):
        doc_id = case.get("doc_id", "")
        question = case.get("question", "")
        answer = case.get("answer", "")
        fmt = case.get("answer_format", "Str")
        print(
            f"  [{i:2d}] {doc_id:40s}  {fmt:5s}  "
            f"ans={answer[:50]:50s}  Q: {question[:80]}..."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Consistency Test — ask one question N times with one "
                    "strategy and generate an HTML report."
    )
    ap.add_argument(
        "--base-url", default="http://0.0.0.0:9191",
        help="Base URL of the API server",
    )
    ap.add_argument(
        "--strategy", default="structural_parent_only",
        help="Strategy name (see --list-strategies)",
    )
    ap.add_argument(
        "--question-index", type=int, default=4,
        help="Index of the question in the SmallerDataset (0-based)",
    )
    ap.add_argument(
        "--runs", type=int, default=10,
        help="Number of times to ask the same question",
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
        help="Directory for output files",
    )
    ap.add_argument(
        "--limit", type=int, default=30,
        help="Qdrant result limit per query",
    )
    ap.add_argument(
        "--timeout", type=int, default=300,
        help="Request timeout in seconds",
    )
    ap.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip the answer extraction step",
    )
    ap.add_argument(
        "--list-strategies", action="store_true",
        help="List all available strategies and exit",
    )
    ap.add_argument(
        "--list-questions", action="store_true",
        help="List all questions in the dataset and exit",
    )

    args = ap.parse_args()

    if args.list_strategies:
        from strategy_grid_test import print_strategy_table
        print_strategy_table(STRATEGIES)
        sys.exit(0)

    if args.list_questions:
        list_questions(args.dataset)
        sys.exit(0)

    run_consistency_test(
        base_url=args.base_url,
        dataset_path=args.dataset,
        file_hash_map_path=args.hash_map,
        output_dir=args.output_dir,
        strategy_name=args.strategy,
        question_index=args.question_index,
        num_runs=args.runs,
        limit=args.limit,
        skip_extraction=args.skip_extraction,
        timeout=args.timeout,
    )
