"""Тесты для UniNerClient."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.uniner_client import UniNerClient
from testdata.base_loader import PredictedEntity


@pytest.mark.asyncio
async def test_extract_entities_valid():
    """Успешное извлечение сущностей — HTTP 200 с валидным JSON."""
    client = UniNerClient(base_url="http://mock")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "entities": [
            {"name": "Apple", "type": "Organization"},
            {"name": "Steve Jobs", "type": "Person"},
        ]
    }
    client.client.post = AsyncMock(return_value=mock_response)

    result = await client.extract_entities(
        text="Apple was founded by Steve Jobs.",
        entity_types=["Organization", "Person"],
    )

    assert result == [
        PredictedEntity(name="Apple", type="Organization"),
        PredictedEntity(name="Steve Jobs", type="Person"),
    ]


@pytest.mark.asyncio
async def test_extract_entities_empty():
    """Пустой список сущностей — HTTP 200 с entities=[]."""
    client = UniNerClient(base_url="http://mock")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"entities": []}
    client.client.post = AsyncMock(return_value=mock_response)

    result = await client.extract_entities(
        text="Some text without entities.",
        entity_types=["Person"],
    )

    assert result == []


@pytest.mark.asyncio
async def test_extract_entities_http_500(caplog):
    """HTTP 500 → ConnectionError и лог ошибки."""
    client = UniNerClient(base_url="http://mock")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )
    client.client.post = AsyncMock(return_value=mock_response)

    with pytest.raises(ConnectionError):
        await client.extract_entities(
            text="trigger error",
            entity_types=["Person"],
        )

    assert "HTTP-ошибка UniNer" in caplog.text


@pytest.mark.asyncio
async def test_extract_entities_invalid_json(caplog):
    """HTTP 200, но тело не JSON → возврат [] и warning."""
    client = UniNerClient(base_url="http://mock")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("not json")
    client.client.post = AsyncMock(return_value=mock_response)

    result = await client.extract_entities(
        text="some text",
        entity_types=["Person"],
    )

    assert result == []
    assert "Невалидный ответ UniNer" in caplog.text


@pytest.mark.asyncio
async def test_extract_entities_no_entities_key(caplog):
    """HTTP 200, но ключ 'entities' отсутствует → возврат [] и warning."""
    client = UniNerClient(base_url="http://mock")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"other_key": "value"}
    client.client.post = AsyncMock(return_value=mock_response)

    result = await client.extract_entities(
        text="some text",
        entity_types=["Person"],
    )

    assert result == []
    assert "Невалидный ответ UniNer" in caplog.text


@pytest.mark.asyncio
async def test_extract_relations_not_implemented():
    """extract_relations всегда выбрасывает NotImplementedError."""
    client = UniNerClient(base_url="http://mock")

    with pytest.raises(NotImplementedError, match="UniNerClient не поддерживает RE"):
        await client.extract_relations(
            text="some text",
            entities=[],
            relation_types=[],
        )
