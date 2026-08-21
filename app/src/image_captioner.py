"""
Caption generation for caption-less images and tables.

Generates a textual description for ``image``/``table`` elements that MinerU
did not annotate with a caption or footnote. The generated description is
injected back into the element's ``image_caption``/``table_caption`` field so
that it flows through the existing pipeline unchanged:

- **Qdrant index**: a separate caption text embedding is created, making the
  image/table discoverable by text queries.
- **Neo4j graph**: a caption region node is created next to the image/table
  node.
- **Context search**: the caption node is already handled by
  [`get_related_context`](documet_index/manager.py:323).

The module relies on the multimodal [`LLMClient`](app/src/llm_client.py:46)
and downloads the image bytes from MinIO before sending them to the model.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Prompt asking the VLM for a concise, search-friendly description.
_CAPTION_PROMPT = (
    "Describe the content of this image concisely in a single paragraph. "
    "Focus on the key visual information: what is depicted, the main "
    "objects, any visible text, numbers or labels, and the overall message. "
    "Return only the description text, without any preamble or formatting."
)


def _strip_thinking(text: str) -> str:
    """Remove a reasoning ``<think>...</think>`` block from a model reply.

    Thinking models (e.g. ``Qwen3-VL-32B-Thinking``) prepend chain-of-thought
    inside a ``<think>`` tag before the actual answer. Keep only the final
    answer that appears after the last closing tag.
    """
    if not text:
        return ""

    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()

    return text.strip()


def _has_nonempty_text(value: Any) -> bool:
    """Return True if ``value`` contains any non-whitespace text.

    Caption/footnote fields are lists of strings; MinerU may leave an empty
    stub like ``[""]`` or ``["  "]`` which should be treated as "no caption".
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(isinstance(v, str) and v.strip() for v in value)
    return bool(value)


def _image_object_candidates(img_path: str) -> List[str]:
    """Return candidate MinIO object names for an image ``img_path``.

    Images are uploaded to MinIO under ``images/{filename}``, while MinerU's
    ``img_path`` uses its own image-directory prefix (e.g. ``images/fig1.png``
    or ``document_images/fig1.png``). Try the raw path first, then the
    ``images/``-prefixed basename, then the bare basename.
    """
    candidates: List[str] = []
    if img_path:
        normalized = img_path.replace("\\", "/")
        candidates.append(normalized)
        basename = normalized.rsplit("/", 1)[-1]
        for candidate in (f"images/{basename}", basename):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _download_image_bytes(
    minio_client: Any,
    bucket_name: str,
    img_path: str,
) -> Optional[bytes]:
    """Download image bytes, trying several MinIO object-name candidates."""
    for object_name in _image_object_candidates(img_path):
        try:
            return minio_client.get_object(
                bucket_name=bucket_name,
                object_name=object_name,
            )
        except Exception:
            logger.debug(
                "Image object %s not found, trying next candidate",
                object_name,
            )
            continue
    return None


def generate_image_description(
    image_base64: str,
    llm_client: Any,
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Optional[str]:
    """Generate a textual description for a single image using the VLM.

    Args:
        image_base64: Base64-encoded image bytes (without data-URI prefix).
        llm_client: An LLM client with a ``send_message`` method compatible
            with [`LLMClient`](app/src/llm_client.py:46).
        temperature: Sampling temperature for the generation.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The generated description string, or ``None`` if generation failed.
    """
    try:
        # Import here to avoid a hard module-level dependency cycle.
        from app.src.llm_client import ModelMessageDict

        message = ModelMessageDict(role="user")
        message.add_text_content(_CAPTION_PROMPT)
        message.add_img_content_base64(image_base64)

        success, responses = llm_client.send_message(
            [message],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not success or not responses:
            logger.warning("Image caption generation failed (no response)")
            return None

        description = _strip_thinking(responses[0] or "")
        if not description:
            logger.warning("Image caption generation returned empty text")
            return None

        return description

    except Exception:
        logger.exception("Image caption generation raised an exception")
        return None


def generate_captions_for_captionless_images(
    elements: List[Dict[str, Any]],
    llm_client: Any,
    minio_client: Any,
    *,
    enabled: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 512,
    bucket_name: Optional[str] = None,
) -> int:
    """Fill missing captions for caption-less image/table elements in-place.

    Mutates the ``elements`` list by adding a generated ``image_caption``
    (or ``table_caption``) to every ``image``/``table`` element that has
    neither a non-empty caption nor a non-empty footnote. The same list is
    referenced by the MinerU result used later for Neo4j graph construction,
    so both Qdrant and Neo4j pick up the generated caption automatically.

    Args:
        elements: The MinerU ``content_list`` elements (mutated in place).
        llm_client: Multimodal LLM client used for description generation.
        minio_client: MinIO client used to download the image bytes.
        enabled: If ``False``, the function returns immediately without
            generating anything.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens for each generated description.
        bucket_name: MinIO bucket name. Defaults to ``minio_client.bucket_name``.

    Returns:
        Number of images/tables for which a description was generated.
    """
    if not enabled:
        return 0

    if llm_client is None or minio_client is None:
        logger.warning(
            "Image captioning skipped: LLM client or MinIO client unavailable"
        )
        return 0

    generated = 0

    for element in elements:
        if not isinstance(element, dict):
            continue

        element_type = element.get("type")
        if element_type == "image":
            caption_key = "image_caption"
            footnote_key = "image_footnote"
        elif element_type == "table":
            caption_key = "table_caption"
            footnote_key = "table_footnote"
        else:
            continue

        # Only caption elements that MinerU left without any textual annotation.
        # Treat empty stubs like [""] or ["  "] as "no caption" to avoid
        # skipping a genuinely caption-less element.
        if _has_nonempty_text(element.get(caption_key)) or _has_nonempty_text(
            element.get(footnote_key)
        ):
            continue

        img_path = element.get("img_path", "")
        if not img_path:
            logger.debug(
                "Skipping caption-less %s without img_path", element_type
            )
            continue

        try:
            image_data = _download_image_bytes(
                minio_client,
                bucket_name or minio_client.bucket_name,
                img_path,
            )
            if not image_data:
                logger.warning(
                    "Image not found in MinIO for %s (%s); tried %s",
                    element_type,
                    img_path,
                    _image_object_candidates(img_path),
                )
                continue

            image_base64 = base64.b64encode(image_data).decode("utf-8")

            description = generate_image_description(
                image_base64,
                llm_client,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if description:
                element[caption_key] = [description]
                generated += 1
                logger.info(
                    "Generated %s caption for %s (%d chars)",
                    element_type,
                    img_path,
                    len(description),
                )

        except Exception:
            logger.exception(
                "Failed to generate caption for %s %s", element_type, img_path
            )

    if generated:
        logger.info("Generated captions for %d caption-less images/tables", generated)

    return generated
