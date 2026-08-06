"""Tests for OllamaClient — entity extraction and relation extraction via native Ollama API.

All HTTP calls are mocked via AsyncMock on self._client.post (Constitution T3).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

# Add entity_extraction_test/ to sys.path for project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.ollama_client import OllamaClient
from testdata.base_loader import PredictedEntity, PredictedRelation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "model_responses"


@pytest.fixture
def ner_response_text() -> str:
    """Load ollama_ner_response.txt fixture content."""
    return (FIXTURES_DIR / "ollama_ner_response.txt").read_text()


@pytest.fixture
def re_response_text() -> str:
    """Load ollama_re_response.txt fixture content."""
    return (FIXTURES_DIR / "ollama_re_response.txt").read_text()


@pytest.fixture
def combined_response_text() -> str:
    """Load ollama_combined_response.txt fixture content."""
    return (FIXTURES_DIR / "ollama_combined_response.txt").read_text()


@pytest.fixture
def ollama_client() -> OllamaClient:
    """Create OllamaClient with mocked httpx.AsyncClient.

    Sets placeholder prompts so extract_entities/extract_relations
    don't short-circuit on empty prompts. The _client (httpx.AsyncClient)
    is replaced with an AsyncMock — each test sets _client.post return_value.
    """
    client = OllamaClient()
    # Prompts must contain format placeholders so .format() succeeds
    client._ner_prompt = "{input_text} {entity_types}"
    client._re_prompt = "{input_text} {entities_list} {relation_types}"
    client._combined_prompt = "{input_text} {entity_types} {relation_types}"
    client._client = AsyncMock()  # replace the real httpx.AsyncClient
    return client


# ---------------------------------------------------------------------------
# Test: extract_entities valid response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_entities_valid_response(
    ollama_client: OllamaClient,
    ner_response_text: str,
) -> None:
    """Mock _client.post to return the NER fixture; assert correct entities."""
    mock_response = httpx.Response(
        200, json={"message": {"content": ner_response_text}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    result = await ollama_client.extract_entities(
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
    ollama_client: OllamaClient,
    re_response_text: str,
) -> None:
    """Mock _client.post to return the RE fixture; assert correct relations."""
    mock_response = httpx.Response(
        200, json={"message": {"content": re_response_text}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    entities = [
        PredictedEntity(name="Apple Inc.", type="Organization"),
        PredictedEntity(name="Steve Jobs", type="Person"),
        PredictedEntity(name="Cupertino", type="Location"),
    ]

    result = await ollama_client.extract_relations(
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
# Test: extract_entities_and_relations (combined)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_entities_and_relations(
    ollama_client: OllamaClient,
    combined_response_text: str,
) -> None:
    """Mock _client.post to return the combined fixture; assert 3 entities + 2 relations."""
    mock_response = httpx.Response(
        200, json={"message": {"content": combined_response_text}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    entities, relations = await ollama_client.extract_entities_and_relations(
        "Apple Inc. was founded by Steve Jobs in Cupertino.",
        ["Organization", "Person", "Location"],
        ["founded_by", "located_in"],
    )

    # Check entities
    assert isinstance(entities, list)
    assert len(entities) == 3
    assert entities[0].name == "Apple Inc."
    assert entities[0].type == "Organization"
    assert entities[1].name == "Steve Jobs"
    assert entities[1].type == "Person"
    assert entities[2].name == "Cupertino"
    assert entities[2].type == "Location"
    for item in entities:
        assert isinstance(item, PredictedEntity)

    # Check relations
    assert isinstance(relations, list)
    assert len(relations) == 2
    assert relations[0].head == "Apple Inc."
    assert relations[0].tail == "Steve Jobs"
    assert relations[0].type == "founded_by"
    assert relations[1].head == "Apple Inc."
    assert relations[1].tail == "Cupertino"
    assert relations[1].type == "located_in"
    for item in relations:
        assert isinstance(item, PredictedRelation)


# ---------------------------------------------------------------------------
# Test: _chat returns empty content (None)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_returns_empty_content(
    ollama_client: OllamaClient,
) -> None:
    """When _chat returns None (no message.content), all methods return empty lists."""
    # Response with no "content" in "message"
    mock_response = httpx.Response(200, json={"message": {}})
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    # NER
    entities = await ollama_client.extract_entities(
        "Some text", ["Organization"]
    )
    assert entities == []

    # RE
    relations = await ollama_client.extract_relations(
        "Some text",
        [PredictedEntity(name="A", type="Org")],
        ["rel_type"],
    )
    assert relations == []

    # Combined
    ents, rels = await ollama_client.extract_entities_and_relations(
        "Some text",
        ["Organization"],
        ["rel_type"],
    )
    assert ents == []
    assert rels == []


# ---------------------------------------------------------------------------
# Test: unparseable response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unparseable_response(ollama_client: OllamaClient) -> None:
    """A completely unparseable response yields an empty list."""
    mock_response = httpx.Response(
        200,
        json={"message": {"content": "garbage text without proper format"}},
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    result = await ollama_client.extract_entities(
        "Some text", ["Organization"]
    )
    assert result == []


# ---------------------------------------------------------------------------
# Test: partially parseable response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partially_parseable_response(
    ollama_client: OllamaClient,
) -> None:
    """Response with 2 valid entity lines + 1 garbage line; assert only 2 extracted."""
    response = (
        '("entity"<|>Apple Inc.<|>Organization)\n'
        "##\n"
        "garbage line without proper format\n"
        "##\n"
        '("entity"<|>Steve Jobs<|>Person)\n'
        "<|COMPLETE|>"
    )
    mock_response = httpx.Response(
        200, json={"message": {"content": response}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    result = await ollama_client.extract_entities(
        "Some text", ["Organization", "Person"]
    )

    assert len(result) == 2
    assert result[0].name == "Apple Inc."
    assert result[0].type == "Organization"
    assert result[1].name == "Steve Jobs"
    assert result[1].type == "Person"


# ---------------------------------------------------------------------------
# Test: HTTP error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_error(ollama_client: OllamaClient) -> None:
    """HTTP 500 response should raise ConnectionError."""
    # httpx.Response with error status — raise_for_status() will raise HTTPStatusError,
    # which the client catches and re-raises as ConnectionError
    mock_response = httpx.Response(500)
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    with pytest.raises(ConnectionError):
        await ollama_client.extract_entities(
            "Some text", ["Organization"]
        )


# ---------------------------------------------------------------------------
# Test: empty prompt — no API call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_prompt() -> None:
    """Client with empty prompts should return [] without making an API call."""
    client = OllamaClient()
    client._ner_prompt = ""
    client._re_prompt = ""
    client._combined_prompt = ""
    client._client = AsyncMock()

    # NER — empty prompt
    entities = await client.extract_entities(
        "Some text", ["Organization"]
    )
    assert entities == []
    # _client.post must NOT be called
    client._client.post.assert_not_called()

    # RE — empty prompt
    relations = await client.extract_relations(
        "Some text",
        [PredictedEntity(name="A", type="Org")],
        ["rel_type"],
    )
    assert relations == []
    client._client.post.assert_not_called()

    # Combined — empty prompt
    ents, rels = await client.extract_entities_and_relations(
        "Some text", ["Organization"], ["rel_type"]
    )
    assert ents == []
    assert rels == []
    client._client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: type validation — unknown type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_type_validation_unknown_type(
    ollama_client: OllamaClient,
) -> None:
    """Entity with type not in entity_types → normalize_type returns None → skipped."""
    response = (
        '("entity"<|>Apple Inc.<|>UnknownType)\n'
        '<|COMPLETE|>'
    )
    mock_response = httpx.Response(
        200, json={"message": {"content": response}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    result = await ollama_client.extract_entities(
        "Some text", ["Organization", "Person", "Location"]
    )

    # UnknownType is not in allowed types and not in TYPE_SYNONYMS → skipped
    assert result == []


# ---------------------------------------------------------------------------
# Test: type validation — synonym mapping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_type_validation_synonym(
    ollama_client: OllamaClient,
) -> None:
    """Entity type "organization" maps to "Org" via normalize_type (TYPE_SYNONYMS)."""
    response = (
        '("entity"<|>Some Org<|>organization)\n'
        '<|COMPLETE|>'
    )
    mock_response = httpx.Response(
        200, json={"message": {"content": response}}
    )
    ollama_client._client.post = AsyncMock(return_value=mock_response)

    # NOTE: allowed_types must include "Org" for normalize_type to map
    # "organization" → TYPE_SYNONYMS["organization"] = "Org" → canonical
    result = await ollama_client.extract_entities(
        "Some text", ["Org", "Peop", "Loc"]
    )

    assert len(result) == 1
    assert result[0].name == "Some Org"
    assert result[0].type == "Org"
