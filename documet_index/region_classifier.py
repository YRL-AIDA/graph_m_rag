"""
Region Type Classifier for structural document graph.

Re-classifies ambiguous region types (image vs table) that may have been
misclassified by the MinerU parser. Uses a two-stage approach:

1. **Heuristic rules** (fast, no LLM cost): checks text content, captions,
   table body structure, and bounding box proportions.
2. **Multimodal LLM** (optional, for truly ambiguous cases): sends the region
   image to a VLM for visual classification.

Integration point: [`create_graph_from_mineru_result`](documet_index/dtype/document.py:120)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic thresholds
# ---------------------------------------------------------------------------

# Minimum characters in table_body to confidently label as a table
_MIN_TABLE_BODY_LENGTH = 20

# Aspect ratio (width/height) thresholds: tables tend to be wider than images
_TABLE_ASPECT_RATIO_MIN = 1.2   # below this → likely image
_IMAGE_ASPECT_RATIO_MAX = 0.8   # above this → likely table

# Caption keyword hints that suggest a figure/image rather than a table
_IMAGE_CAPTION_HINTS = (
    "figure", "fig.", "image", "photo", "diagram", "chart",
    "graph", "plot", "illustration", "screenshot", "picture",
)

# Table-specific content indicators
_TABLE_CONTENT_HINTS = (
    "|", "\t", "col", "row", "header",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_region_type(
    element: Dict[str, Any],
    element_type: str,
    *,
    use_llm: bool = False,
    llm_client: Any = None,
) -> str:
    """Re-classify a region element if its type is ambiguous.

    Only ``image`` and ``table`` types are considered for reclassification.
    All other types are returned unchanged.

    Args:
        element:      Raw MinerU element dictionary (contains ``type``, ``bbox``,
                      ``text``, ``img_path``, ``table_body``,
                      ``image_caption``, ``table_caption``, etc.).
        element_type: Current type label from MinerU (``'image'`` or ``'table'``).
        use_llm:      If ``True``, fall back to the multimodal LLM when
                      heuristics cannot make a confident decision.
        llm_client:   An optional LLM client instance with a ``send_message``
                      method compatible with
                      [`LLMClient`](app/src/llm_client.py:46).

    Returns:
        Corrected type label — one of ``'image'``, ``'table'``, or the original
        ``element_type`` if no change was made.
    """
    if element_type not in ("image", "table"):
        return element_type

    # --- Stage 1: heuristics ---
    heuristic_result = _classify_by_heuristics(element)
    if heuristic_result is not None:
        if heuristic_result != element_type:
            logger.debug(
                "Heuristic reclassification: %s → %s (img=%s, tbl_body_len=%d)",
                element_type,
                heuristic_result,
                element.get("img_path", "")[:60],
                len(str(element.get("table_body", ""))),
            )
        return heuristic_result

    # --- Stage 2: LLM (optional) ---
    if use_llm and llm_client is not None:
        llm_result = _classify_by_llm(element, llm_client)
        if llm_result is not None:
            if llm_result != element_type:
                logger.info(
                    "LLM reclassification: %s → %s (img=%s)",
                    element_type,
                    llm_result,
                    element.get("img_path", "")[:60],
                )
            return llm_result

    # Keep original if neither method could reclassify
    return element_type


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


def _classify_by_heuristics(element: Dict[str, Any]) -> Optional[str]:
    """Return ``'image'``, ``'table'``, or ``None`` (ambiguous).

    Decision logic (ordered by confidence):

    1. If the element contains substantial ``table_body`` text **and**
       no ``image_caption`` → **table**.
    2. If the element has ``image_caption`` hints (e.g. "Figure") **and**
       no ``table_body`` → **image**.
    3. If ``table_body`` contains table-structure markers (``|``, ``\\t``)
       → **table**.
    4. Inspect bounding-box aspect ratio:
       - Wide & short (w/h ≥ 1.2) → **table**.
       - Tall & narrow (w/h ≤ 0.8) → **image**.
    5. Fallback: presence of any ``image_caption`` → **image**;
       presence of any ``table_body`` → **table**.
    6. Otherwise → **None** (ambiguous).
    """
    table_body = str(element.get("table_body", "")).strip()
    image_captions = element.get("image_caption", [])
    table_captions = element.get("table_caption", [])
    bbox = element.get("bbox", [0, 0, 0, 0])

    has_table_body = len(table_body) >= _MIN_TABLE_BODY_LENGTH
    has_image_caption = bool(image_captions)
    has_table_caption = bool(table_captions)

    # --- Rule 1: strong table signal ---
    if has_table_body and not has_image_caption:
        return "table"

    # --- Rule 2: strong image signal ---
    if has_image_caption and not has_table_body:
        caption_text = " ".join(image_captions).lower()
        if any(hint in caption_text for hint in _IMAGE_CAPTION_HINTS):
            return "image"

    # --- Rule 3: table structure markers ---
    if any(marker in table_body for marker in _TABLE_CONTENT_HINTS):
        return "table"

    # --- Rule 4: bbox aspect ratio ---
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width > 0 and height > 0:
            aspect = width / height
            if aspect >= _TABLE_ASPECT_RATIO_MIN:
                return "table"
            if aspect <= _IMAGE_ASPECT_RATIO_MAX:
                return "image"

    # --- Rule 5: weaker fallback signals ---
    if has_image_caption and not has_table_caption:
        return "image"
    if has_table_body:
        return "table"

    # --- Rule 6: truly ambiguous ---
    return None


# ---------------------------------------------------------------------------
# LLM-based classification
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a document element classifier. Your task is to determine "
    "whether a given region in a document is an **image** (figure, photo, "
    "chart, diagram, screenshot) or a **table** (structured data with rows "
    "and columns).\n\n"
    "Reply with exactly one word: 'image' or 'table'.\n"
    "Do not add any explanation, punctuation, or other text."
)


def _classify_by_llm(
    element: Dict[str, Any],
    llm_client: Any,
) -> Optional[str]:
    """Use a multimodal LLM to classify the element by its image.

    Returns ``'image'``, ``'table'``, or ``None`` on failure.
    """
    img_path = element.get("img_path", "")
    if not img_path:
        logger.debug("LLM classify skipped: no img_path for element")
        return None

    try:
        # Import here to avoid circular dependency at module load time
        from app.src.llm_client import ModelMessageDict  # type: ignore[import-untyped]

        message = ModelMessageDict(role="user")
        message.add_text_content(_CLASSIFY_SYSTEM_PROMPT)
        message.add_img_content(source="image_url", path_to_img=img_path)

        success, responses = llm_client.send_message(
            [message],
            temperature=0.0,
            max_tokens=8,
        )

        if not success or not responses:
            logger.warning("LLM classify failed for %s", img_path)
            return None

        answer = responses[0].strip().lower()
        if answer in ("image", "table"):
            return answer

        logger.debug("LLM classify: unexpected response '%s' for %s", answer, img_path)
        return None

    except Exception:
        logger.exception("LLM classify error for %s", img_path)
        return None
