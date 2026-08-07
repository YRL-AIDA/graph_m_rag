"""
Multi-hop Question Decomposition (C6).

Decomposes a complex question into simpler sub-questions using LLM,
then searches each independently and merges results.

Strategy:
1. LLM prompt asks to break complex questions into sub-questions
2. Each sub-question is searched in Qdrant independently
3. Results are deduplicated by element_index+file_hash
4. Top results are taken as the merged candidate set
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from app.src.llm_client import ModelMessageDict

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a question decomposition agent. Your task is to break down "
    "a complex question into simpler sub-questions that can be answered "
    "independently from a document.\n\n"
    "Rules:\n"
    "1. If the question is simple (one fact, one entity), return a single "
    "sub-question identical to the original.\n"
    "2. If the question has multiple parts or requires information from "
    "different sections, split it into sub-questions.\n"
    "3. Each sub-question must be self-contained and answerable independently.\n"
    "4. Do NOT add information not present in the original question.\n"
    "5. Return ONLY a JSON array of strings. No explanation, no markdown.\n"
    "6. Maximum 5 sub-questions.\n\n"
    "Examples:\n"
    'Question: "What is the revenue of Apple in 2023?"\n'
    'Output: ["What is the revenue of Apple in 2023?"]\n\n'
    'Question: "How did the profit change after the acquisition?"\n'
    'Output: ["What was the profit before the acquisition?", '
    '"What was the profit after the acquisition?", '
    '"What acquisition occurred and when?"]\n\n'
    'Question: "Compare the risk factors mentioned in section A and section B."\n'
    'Output: ["What are the risk factors in section A?", '
    '"What are the risk factors in section B?"]\n\n'
    'Question: "What are the differences between product X and product Y '
    'in pricing and features?"\n'
    'Output: ["What is the pricing of product X?", '
    '"What are the features of product X?", '
    '"What is the pricing of product Y?", '
    '"What are the features of product Y?"]\n'
)


def decompose_question(
    question: str,
    llm_client: Any,
    model_name: str = "Qwen/Qwen3-VL-32B-Thinking",
) -> List[str]:
    """Decompose a complex question into simpler sub-questions using LLM.

    Returns a list of sub-questions. If decomposition fails or the question
    is already simple, returns [original_question].

    Args:
        question: The original question text.
        llm_client: An LLMClient instance with a `send_message` method.
        model_name: Model name to use for decomposition.

    Returns:
        List of sub-question strings. Never empty.
    """
    if not question or not question.strip():
        return [question] if question else []

    # Quick heuristic: short questions are likely single-hop
    word_count = len(question.split())
    if word_count <= 5:
        logger.debug(
            "Question too short for decomposition (%d words), using as-is",
            word_count,
        )
        return [question]

    user_msg = ModelMessageDict(role="user")
    user_msg.add_text_content(f"Question: {question}\nOutput:")

    system_msg = ModelMessageDict(role="system")
    system_msg.add_text_content(DECOMPOSE_SYSTEM_PROMPT)

    try:
        success, responses = llm_client.send_message(
            messages=[system_msg, user_msg],
            model=model_name,
            temperature=0.1,
            max_tokens=256,
        )

        if not success or not responses or not responses[0]:
            logger.warning(
                "Question decomposition LLM call failed, using original"
            )
            return [question]

        raw = responses[0].strip()
        logger.debug("LLM decomposition raw response: %s", raw[:300])

        # Extract JSON array from response (robust against markdown fences)
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in decomposition response")
            return [question]

        parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, list) or len(parsed) == 0:
            logger.warning("Decomposition produced empty/non-list result")
            return [question]

        # Clean and validate sub-questions
        sub_questions: List[str] = []
        for sq in parsed:
            if isinstance(sq, str) and sq.strip():
                cleaned = sq.strip().rstrip("?").strip()
                if len(cleaned) >= 3:  # minimum meaningful question
                    sub_questions.append(f"{cleaned}?")

        if not sub_questions:
            return [question]

        if len(sub_questions) == 1:
            logger.info("Question is single-hop, no decomposition needed")
            return [question]

        logger.info(
            "Decomposed question into %d sub-questions: %s",
            len(sub_questions),
            sub_questions,
        )
        return sub_questions

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse decomposition JSON: %s", e)
        return [question]
    except Exception as e:
        logger.warning(
            "Question decomposition failed: %s", e, exc_info=True
        )
        return [question]


def merge_search_results(
    results_list: List[List[Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Merge and deduplicate search results from multiple sub-queries.

    Deduplicates by (file_hash, element_index) tuple, keeping the highest
    score for each unique result. Then returns top `limit` by score.

    Args:
        results_list: List of search result lists, one per sub-question.
        limit: Maximum number of merged results to return.

    Returns:
        Deduplicated list of result dicts sorted by score descending.
    """
    seen: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for results in results_list:
        for r in results:
            payload = r.get("payload", {})
            file_hash = payload.get("file_hash", "")
            elem_idx = payload.get("element_index", -1)
            key = (file_hash, elem_idx)

            if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
                seen[key] = {
                    "text": r.get("text", ""),
                    "score": r.get("score", 0),
                    "element_type": payload.get("element_type", ""),
                    "element_index": elem_idx,
                    "page_idx": payload.get(
                        "original_element", {}
                    ).get("page_idx", 0),
                    "payload": payload,
                    "_sub_query_count": (
                        1
                        if key not in seen
                        else seen[key].get("_sub_query_count", 1) + 1
                    ),
                }

    # Sort by score descending
    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged[:limit]
