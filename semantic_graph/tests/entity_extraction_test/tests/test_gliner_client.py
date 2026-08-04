"""Тесты для GleanerClient — NER-клиента на базе Gleaner (Gliner) FastAPI-сервера.

Все вызовы API замоканы (реальные HTTP-запросы не выполняются).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Добавляем entity_extraction_test/ в sys.path для импорта clients и testdata
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.gliner_client import GleanerClient
from testdata.base_loader import PredictedEntity, PredictedRelation


# ---------------------------------------------------------------------------
# Вспомогательная функция — загрузка фикстуры
# ---------------------------------------------------------------------------


def _load_gliner_response() -> dict:
    """Загрузить JSON-фикстуру ответа Gleaner API."""
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "model_responses"
        / "gliner_response.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Тесты extract_entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_entities_valid() -> None:
    """Успешное извлечение 3 сущностей из валидного JSON-ответа."""
    fixture = _load_gliner_response()
    content: str = fixture["choices"][0]["message"]["content"]

    # Собираем mock-ответ: choices[0].message.content = JSON-строка из фикстуры
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    with patch(
        "clients.gliner_client.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        client = GleanerClient(
            base_url="http://test.local/v1", api_key="test-key"
        )
        result = await client.extract_entities(
            text="Apple Inc. was founded by Steve Jobs in Cupertino.",
            entity_types=["Organization", "Person", "Location"],
        )

    assert len(result) == 3

    entities_by_name = {e.name: e for e in result}
    assert "Apple Inc." in entities_by_name
    assert entities_by_name["Apple Inc."].type == "Organization"
    assert "Steve Jobs" in entities_by_name
    assert entities_by_name["Steve Jobs"].type == "Person"
    assert "Cupertino" in entities_by_name
    assert entities_by_name["Cupertino"].type == "Location"

    # Проверяем, что все элементы — PredictedEntity
    for entity in result:
        assert isinstance(entity, PredictedEntity)


@pytest.mark.asyncio
async def test_extract_entities_invalid_json() -> None:
    """Невалидный JSON в ответе → возвращается пустой список."""
    mock_message = MagicMock()
    mock_message.content = "not valid json"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    with patch(
        "clients.gliner_client.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        client = GleanerClient(
            base_url="http://test.local/v1", api_key="test-key"
        )
        result = await client.extract_entities(
            text="Some text",
            entity_types=["Organization"],
        )

    assert result == []


# ---------------------------------------------------------------------------
# Тест extract_relations — RE не поддерживается
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_relations_not_implemented() -> None:
    """Вызов extract_relations выбрасывает NotImplementedError."""
    client = GleanerClient(
        base_url="http://test.local/v1", api_key="test-key"
    )

    with pytest.raises(NotImplementedError, match="GleanerClient does not support RE"):
        await client.extract_relations(
            text="some text",
            entities=[],
            relation_types=["Located_In"],
        )
