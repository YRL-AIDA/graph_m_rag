"""Tests for QwenClient — entity extraction and relation extraction.

All LLM calls are mocked via AsyncMock on self.llm.generate (Constitution T3).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add entity_extraction_test/ to sys.path for project imports
# (also done by conftest, but explicit per spec)
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.qwen_client import QwenClient
from testdata.base_loader import PredictedEntity, PredictedRelation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "model_responses"


@pytest.fixture
def ner_response_text() -> str:
    """Load qwen_ner_response.txt fixture content."""
    return (FIXTURES_DIR / "qwen_ner_response.txt").read_text()


@pytest.fixture
def re_response_text() -> str:
    """Load qwen_re_response.txt fixture content."""
    return (FIXTURES_DIR / "qwen_re_response.txt").read_text()


@pytest.fixture
def qwen_client() -> QwenClient:
    """Create QwenClient with mocked AsyncLLMClient.generate.

    Sets placeholder prompts so extract_entities/extract_relations
    don’t short-circuit on empty prompts. The llm.generate callable
    is replaced with an AsyncMock — each test sets its return_value.
    """
    client = QwenClient()
    # Prompts must contain format placeholders so .format() succeeds
    client._ner_prompt = "{input_text} {entity_types}"
    client._re_prompt = "{input_text} {entities_list} {relation_types}"
    client.llm.generate = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Test: extract_entities valid response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_entities_valid_response(
    qwen_client: QwenClient,
    ner_response_text: str,
) -> None:
    """Mock generate() to return the NER fixture; assert correct entities."""
    qwen_client.llm.generate.return_value = ner_response_text

    result = await qwen_client.extract_entities(
        "Apple Inc. was founded by Steve Jobs in Cupertino.",
        ["Organization", "Person", "Location"],
    )

    assert isinstance(result, list)
    assert len(result) == 3

    assert result[0].name == "Apple Inc."
    assert result[0].type == "Organization"
    assert result[1].name == "Steve Jobs"
    assert result[1].type == "Person"
    assert result[2].name == "Cupertino"
    assert result[2].type == "Location"

    # All items are PredictedEntity instances
    for item in result:
        assert isinstance(item, PredictedEntity)


# ---------------------------------------------------------------------------
# Test: extract_relations valid response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_relations_valid_response(
    qwen_client: QwenClient,
    re_response_text: str,
) -> None:
    """Mock generate() to return the RE fixture; assert correct relations."""
    qwen_client.llm.generate.return_value = re_response_text

    entities = [
        PredictedEntity(name="Apple Inc.", type="Organization"),
        PredictedEntity(name="Steve Jobs", type="Person"),
        PredictedEntity(name="Cupertino", type="Location"),
    ]

    result = await qwen_client.extract_relations(
        "Apple Inc. was founded by Steve Jobs in Cupertino.",
        entities,
        ["founded_by", "located_in"],
    )

    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0].head == "Apple Inc."
    assert result[0].tail == "Steve Jobs"
    assert result[0].type == "founded_by"

    assert result[1].head == "Apple Inc."
    assert result[1].tail == "Cupertino"
    assert result[1].type == "located_in"

    for item in result:
        assert isinstance(item, PredictedRelation)


# ---------------------------------------------------------------------------
# Test: generate() returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_returns_none(qwen_client: QwenClient) -> None:
    """When llm.generate() returns None, both methods return empty lists."""
    qwen_client.llm.generate.return_value = None

    # NER
    entities = await qwen_client.extract_entities(
        "Some text", ["Organization"]
    )
    assert entities == []

    # RE
    relations = await qwen_client.extract_relations(
        "Some text",
        [PredictedEntity(name="A", type="Org")],
        ["rel_type"],
    )
    assert relations == []


# ---------------------------------------------------------------------------
# Test: unparseable response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unparseable_response(qwen_client: QwenClient) -> None:
    """A completely unparseable response yields an empty list."""
    qwen_client.llm.generate.return_value = "garbage text without proper format"

    result = await qwen_client.extract_entities(
        "Some text", ["Organization"]
    )
    assert result == []


# ---------------------------------------------------------------------------
# Test: partially parseable response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partially_parseable_response(qwen_client: QwenClient) -> None:
    """Response with 2 valid entity lines + 1 garbage line, ending with <|COMPLETE|>.

    Only the 2 valid lines should be extracted.
    """
    response = (
        '("entity"<|>Apple Inc.<|>Organization)\n'
        '##\n'
        'garbage line without proper format\n'
        '##\n'
        '("entity"<|>Steve Jobs<|>Person)\n'
        '<|COMPLETE|>'
    )
    qwen_client.llm.generate.return_value = response

    result = await qwen_client.extract_entities(
        "Some text", ["Organization", "Person"]
    )

    assert len(result) == 2
    assert result[0].name == "Apple Inc."
    assert result[0].type == "Organization"
    assert result[1].name == "Steve Jobs"
    assert result[1].type == "Person"


# ---------------------------------------------------------------------------
# Test: combined response (entities + relationships)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_combined_response(qwen_client: QwenClient) -> None:
    """Response with both entity and relationship lines.

    NER parser (_NER_LINE_RE) extracts only entity-prefixed lines.
    RE parser (_RE_LINE_RE) extracts only relationship-prefixed lines.
    """
    response = (
        '("entity"<|>Apple Inc.<|>Organization)\n'
        '##\n'
        '("relationship"<|>Apple Inc.<|>Steve Jobs<|>founded_by)\n'
        '##\n'
        '("entity"<|>Steve Jobs<|>Person)\n'
        '<|COMPLETE|>'
    )
    qwen_client.llm.generate.return_value = response

    # NER: should extract only the two entity lines
    entities = await qwen_client.extract_entities(
        "Some text", ["Organization", "Person"]
    )
    assert len(entities) == 2
    assert entities[0].name == "Apple Inc."
    assert entities[0].type == "Organization"
    assert entities[1].name == "Steve Jobs"
    assert entities[1].type == "Person"

    # RE: should extract only the one relationship line
    qwen_client.llm.generate.return_value = response
    relations = await qwen_client.extract_relations(
        "Some text",
        [
            PredictedEntity(name="Apple Inc.", type="Organization"),
            PredictedEntity(name="Steve Jobs", type="Person"),
        ],
        ["founded_by"],
    )
    assert len(relations) == 1
    assert relations[0].head == "Apple Inc."
    assert relations[0].tail == "Steve Jobs"
    assert relations[0].type == "founded_by"
