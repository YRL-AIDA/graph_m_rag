"""
Feedback-driven iterative retrieval (C5).

Two-round retrieval where the LLM itself identifies missing information
after an initial answer and formulates a refinement query for a second pass.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.src.llm_client import ModelMessageDict

logger = logging.getLogger(__name__)


def _build_message(text: str) -> list[ModelMessageDict]:
    """Build a single-user-message list for LLMClient."""
    msg = ModelMessageDict(role="user")
    msg.add_text_content(text)
    return [msg]


def _call_llm(
    llm_client: Any,
    user_text: str,
    system_text: str | None = None,
) -> str:
    """Call the synchronous LLMClient and return the response text."""
    messages: list[ModelMessageDict] = []
    if system_text:
        sys_msg = ModelMessageDict(role="system")
        sys_msg.add_text_content(system_text)
        messages.append(sys_msg)
    messages.extend(_build_message(user_text))
    success, responses = llm_client.send_message(
        messages,
        temperature=0.1,
        max_tokens=2048,
    )
    if success and responses:
        return responses[0]
    return ""

_ITERATIVE_SYSTEM_PROMPT = """You are a research assistant performing iterative retrieval.
You have just answered a question using a set of retrieved context.
Now examine your answer and identify what KEY FACTS or DETAILS are STILL MISSING.

Rules:
1. If the answer is complete and covers all aspects of the question, respond with {"complete": true, "refinement_query": null}.
2. If information is missing, produce a SINGLE refinement query that would fill the gap.
   Make it precise and keyword-rich so a search engine can find the missing facts.
3. Do NOT repeat the original question. Focus on what is NOT yet covered.
4. Output ONLY valid JSON with keys "complete" (bool) and "refinement_query" (string|null).

Example:
Original question: "What is the profitability of Company X and their AI strategy?"
First answer mentions profitability but nothing about AI.
→ {"complete": false, "refinement_query": "Company X artificial intelligence strategy initiatives investments"}

Original question: "How does the engine cooling system work?"
First answer describes all components and flow completely.
→ {"complete": true, "refinement_query": null}
"""


@dataclass
class IterationResult:
    """Result of one retrieval round."""
    round_number: int
    context: list[str] = field(default_factory=list)
    answer: str = ""
    complete: bool = False
    refinement_query: str | None = None


async def iterative_retrieval(
    question: str,
    llm_client: Any,
    search_fn: Any,  # async callable(question: str) -> dict with "context" key
    *,
    max_rounds: int = 2,
    model_name: str = "gpt-4o",
) -> dict[str, Any]:
    """Run feedback-driven iterative retrieval.

    Args:
        question: The user's original question.
        llm_client: LLM client with `send_message(text, model, system_prompt)`.
        search_fn: Async callable that takes a question and returns a dict
                   with at least a "context" key (list of text blocks).
        max_rounds: Maximum number of retrieval rounds (default 2).
        model_name: LLM model name for answer generation and gap detection.

    Returns:
        Dict with keys:
          - "context": merged context from all rounds
          - "answer": final answer
          - "rounds": list of IterationResult
          - "total_rounds": number of rounds executed
    """
    rounds: list[IterationResult] = []
    all_context: list[str] = []

    for round_num in range(1, max_rounds + 1):
        query = rounds[-1].refinement_query if rounds else question
        logger.info("Iterative retrieval round %d/%d: %s", round_num, max_rounds, query[:100])

        # Search
        try:
            search_result = await search_fn(query)
        except Exception as exc:
            logger.error("Search failed in round %d: %s", round_num, exc)
            search_result = {}

        round_context: list[str] = search_result.get("context", []) if isinstance(search_result, dict) else []
        if not round_context and isinstance(search_result, list):
            round_context = search_result

        # Deduplicate against previously collected context
        new_context = [c for c in round_context if c not in all_context]
        all_context.extend(new_context)

        result = IterationResult(
            round_number=round_num,
            context=new_context,
        )
        rounds.append(result)

        # Generate answer from all accumulated context
        context_text = "\n---\n".join(all_context) if all_context else "No context available."
        answer_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Provide a thorough answer in ENGLISH. Start with the answer value on its own line, "
            f"then add supporting evidence on subsequent lines. "
            f"If the context is insufficient or you cannot give an accurate answer, "
            f"respond strictly with: \"Not answerable\" on a single line with no other text."
        )

        try:
            result.answer = _call_llm(llm_client, answer_prompt)
            if not result.answer:
                result.answer = "Unable to generate answer due to an error."
        except Exception as exc:
            logger.error("LLM answer generation failed in round %d: %s", round_num, exc)
            result.answer = "Unable to generate answer due to an error."

        if round_num >= max_rounds:
            result.complete = True
            break

        # Ask LLM to identify gaps
        gap_prompt = (
            f"Original question: {question}\n\n"
            f"Your answer:\n{result.answer}\n\n"
            f"Context used:\n{context_text[:3000]}...\n\n"
            f"Are there any missing facts? If so, what should we search for?"
        )
        try:
            gap_text = _call_llm(llm_client, gap_prompt, system_text=_ITERATIVE_SYSTEM_PROMPT)
            # Extract JSON from response (may be wrapped in markdown)
            gap_text = gap_text.strip()
            if gap_text.startswith("```"):
                gap_text = gap_text.split("\n", 1)[-1]
                gap_text = gap_text.rsplit("```", 1)[0]
            gap_data = json.loads(gap_text)
            result.complete = bool(gap_data.get("complete", False))
            result.refinement_query = gap_data.get("refinement_query") if not result.complete else None
        except Exception as exc:
            logger.warning("Gap detection failed in round %d: %s. Assuming complete.", round_num, exc)
            result.complete = True
            result.refinement_query = None

        if result.complete:
            break

    return {
        "context": all_context,
        "answer": rounds[-1].answer if rounds else "",
        "rounds": rounds,
        "total_rounds": len(rounds),
    }
