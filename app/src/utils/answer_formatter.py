"""
Answer Format Post-Processor

Extracts and normalizes Int, Float, and List answers from raw LLM response text.
Used to improve evaluation scores for structured answer types where the model
may produce verbose chain-of-thought instead of a clean answer value.
"""

import re
from typing import Optional, Union, List


def _strip_first_nonempty_line(text: str) -> str:
    """Return the first non-empty, non-formatting line of text."""
    if not text:
        return ""
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(
            ("#", ">", "```", "Okay", "Let", "First", "Based", "According", "The", "I")
        ):
            return stripped
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def extract_int(text: str) -> Optional[int]:
    """Extract an integer value from LLM response text."""
    if not text:
        return None
    first_line = _strip_first_nonempty_line(text)
    match = re.search(r'(?<![.\d])(-?\d+)(?![.\d])', first_line)
    if match:
        return int(match.group(1))
    match = re.search(r'(?<![.\d])(-?\d+)(?![.\d])', text)
    if match:
        return int(match.group(1))
    return None


def extract_float(text: str) -> Optional[float]:
    """Extract a float/decimal/percentage value from LLM response text."""
    if not text:
        return None
    first_line = _strip_first_nonempty_line(text)

    pct_match = re.search(r'([\d,]+(?:\.\d+)?)\s*%', first_line)
    if pct_match:
        return float(pct_match.group(1).replace(',', ''))

    dollar_match = re.search(r'\$[\s]*([\d,]+(?:\.\d+)?)', first_line)
    if dollar_match:
        return float(dollar_match.group(1).replace(',', ''))

    float_match = re.search(r'(?<!\d)(-?[\d,]+\.\d+)(?![\d.])', first_line)
    if float_match:
        return float(float_match.group(1).replace(',', ''))

    # Fall back to full text
    pct_match = re.search(r'([\d,]+(?:\.\d+)?)\s*%', text)
    if pct_match:
        return float(pct_match.group(1).replace(',', ''))

    dollar_match = re.search(r'\$[\s]*([\d,]+(?:\.\d+)?)', text)
    if dollar_match:
        return float(dollar_match.group(1).replace(',', ''))

    float_match = re.search(r'(?<!\d)(-?[\d,]+\.\d+)(?![\d.])', text)
    if float_match:
        return float(float_match.group(1).replace(',', ''))

    return None


def extract_list(text: str) -> Optional[List[str]]:
    """Extract a list of items from LLM response text."""
    if not text:
        return None
    first_line = _strip_first_nonempty_line(text)

    if first_line.count(',') >= 2:
        items = [item.strip().strip('"').strip("'") for item in first_line.split(',')]
        items = [re.sub(r'^(and|or)\s+', '', item).strip() for item in items]
        items = [item for item in items if item]
        if len(items) >= 2:
            return items

    if first_line.count(';') >= 1:
        items = [item.strip().strip('"').strip("'") for item in first_line.split(';')]
        items = [item for item in items if item]
        if len(items) >= 2:
            return items

    # Numbered/bullet lists
    lines = text.strip().splitlines()
    list_items = []
    for line in lines:
        stripped = line.strip()
        num_match = re.match(r'^\d+[.)]\s+(.+)', stripped)
        if num_match:
            list_items.append(num_match.group(1).strip())
        elif re.match(r'^[-*•]\s+(.+)', stripped):
            list_items.append(re.match(r'^[-*•]\s+(.+)', stripped).group(1).strip())

    if len(list_items) >= 2:
        return list_items

    if text.count(',') >= 2:
        items = [item.strip().strip('"').strip("'") for item in text.split(',')]
        items = [re.sub(r'^(and|or)\s+', '', item).strip() for item in items]
        items = [item for item in items if item and len(item) < 200]
        if len(items) >= 2:
            return items

    return None


def extract_str(text: str) -> Optional[str]:
    """Extract a string answer — return cleaned first meaningful line."""
    if not text:
        return None
    first_line = _strip_first_nonempty_line(text)
    if first_line.lower() in ("not answerable", "none", "n/a", "unknown", ""):
        return None
    return first_line


_REFUSAL_VARIANTS = (
    "not answerable", "fail to answer", "failed to answer", "unable to answer",
    "cannot answer", "cannot provide", "no answer", "i don't know",
    "none", "n/a", "unknown",
)


def _is_refusal(text: str) -> bool:
    """Return True if the text is (or starts with) a refusal / not-answerable signal."""
    if not text:
        return False
    lower = text.strip().lower()
    return lower in _REFUSAL_VARIANTS or any(
        lower.startswith(v) for v in _REFUSAL_VARIANTS
    )


def format_answer(text: str, answer_format: str) -> Union[str, int, float, List[str], None]:
    """Post-process LLM response based on expected answer format.

    Args:
        text: Raw LLM response text
        answer_format: One of 'Int', 'Float', 'List', 'Str', or 'None'

    Returns:
        Extracted value or original text if extraction fails.
    """
    if not text or not text.strip():
        return None

    fmt = (answer_format or '').strip().lower()

    # Early exit: refusal / "Not answerable" must be returned verbatim, not
    # parsed into a number or suffixed with "%" / "list".
    if _is_refusal(text):
        return "Not answerable"

    if fmt == 'int':
        result = extract_int(text)
        return result if result is not None else text.strip()

    elif fmt == 'float':
        result = extract_float(text)
        return result if result is not None else text.strip()

    elif fmt == 'list':
        result = extract_list(text)
        return result if result is not None else text.strip()

    elif fmt == 'str':
        result = extract_str(text)
        return result if result is not None else text.strip()

    elif fmt == 'none':
        lower = text.strip().lower()
        if lower.startswith('not answerable') or lower == 'not answerable':
            return 'Not answerable'
        return text.strip()

    else:
        return text.strip()
