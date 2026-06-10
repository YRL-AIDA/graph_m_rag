"""Tests for structured LLM request/response flow (community reports)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from create_community_report import (
    AsyncCommunityReportExtractor,
    CommunityReportResponse,
    FindingModel,
)
from graphrag import AsyncLLMClient, remove_think_tags
from prompts import COMMUNITY_REPORT_PROMPT


SAMPLE_REPORT_JSON = {
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza.",
    "rating": 5.0,
    "rating_explanation": "Moderate impact due to potential unrest.",
    "findings": [
        {
            "summary": "Central location",
            "explanation": "Verdant Oasis Plaza is the central entity.",
        },
        {
            "summary": "Harmony Assembly role",
            "explanation": "Harmony Assembly organizes the march.",
        },
    ],
}


def _make_chat_response(content: str | None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_llm_with_content(content: str | None) -> AsyncLLMClient:
    client = AsyncLLMClient(base_url="http://test", tokenizer_url="http://test/tokenize")
    client.client = MagicMock()
    client.client.chat.completions.create = AsyncMock(
        return_value=_make_chat_response(content)
    )
    return client


# --- Pydantic models ---


class TestCommunityReportResponse:
    def test_valid_report_parses_all_fields(self):
        report = CommunityReportResponse.model_validate(SAMPLE_REPORT_JSON)

        assert report.title == SAMPLE_REPORT_JSON["title"]
        assert report.summary == SAMPLE_REPORT_JSON["summary"]
        assert report.rating == 5.0
        assert report.rating_explanation == SAMPLE_REPORT_JSON["rating_explanation"]
        assert len(report.findings) == 2
        assert isinstance(report.findings[0], FindingModel)

    def test_missing_required_field_raises(self):
        incomplete = {k: v for k, v in SAMPLE_REPORT_JSON.items() if k != "title"}
        with pytest.raises(ValidationError):
            CommunityReportResponse.model_validate(incomplete)

    def test_invalid_rating_type_raises(self):
        bad = {**SAMPLE_REPORT_JSON, "rating": "high"}
        with pytest.raises(ValidationError):
            CommunityReportResponse.model_validate(bad)

    def test_model_dump_json_roundtrip(self):
        report = CommunityReportResponse.model_validate(SAMPLE_REPORT_JSON)
        restored = CommunityReportResponse.model_validate_json(report.model_dump_json())
        assert restored == report


# --- remove_think_tags (preprocessing before JSON parse) ---


class TestRemoveThinkTags:
    def test_strips_reasoning_prefix_before_json(self):
        raw = 'reasoning here{"title": "x"}'
        assert remove_think_tags(raw) == '{"title": "x"}'

    def test_plain_json_unchanged(self):
        raw = '{"title": "x"}'
        assert remove_think_tags(raw) == raw

    def test_whitespace_trimmed(self):
        assert remove_think_tags('  {"a": 1}  ') == '{"a": 1}'


# --- AsyncLLMClient.generate_structured ---


@pytest.mark.asyncio
class TestGenerateStructured:
    async def test_parses_plain_json_response(self):
        client = _mock_llm_with_content(json.dumps(SAMPLE_REPORT_JSON))

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is not None
        assert result.title == SAMPLE_REPORT_JSON["title"]
        assert result.findings[0].summary == "Central location"

    async def test_parses_json_inside_markdown_fence(self):
        fenced = f"```json\n{json.dumps(SAMPLE_REPORT_JSON)}\n```"
        client = _mock_llm_with_content(fenced)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is not None
        assert result.rating == 5.0

    async def test_parses_json_after_think_tags(self):
        content = (
            "internal reasoning"
            + json.dumps(SAMPLE_REPORT_JSON)
        )
        client = _mock_llm_with_content(content)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is not None
        assert result.title == SAMPLE_REPORT_JSON["title"]

    async def test_passes_json_object_response_format(self):
        client = _mock_llm_with_content(json.dumps(SAMPLE_REPORT_JSON))

        await client.generate_structured(
            messages=[{"role": "user", "content": "prompt"}],
            model="test-model",
            response_model=CommunityReportResponse,
            temperature=0.1,
        )

        call_kwargs = client.client.chat.completions.create.await_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.1

    async def test_empty_content_returns_none(self):
        client = _mock_llm_with_content(None)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

    async def test_invalid_json_returns_none(self):
        client = _mock_llm_with_content("not valid json {{{")

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

    async def test_schema_mismatch_returns_none(self):
        bad_json = json.dumps({"title": "only title"})
        client = _mock_llm_with_content(bad_json)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

    async def test_api_exception_returns_none(self):
        client = AsyncLLMClient(base_url="http://test", tokenizer_url="http://test/tokenize")
        client.client = MagicMock()
        client.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None


# --- AsyncCommunityReportExtractor ---


@pytest.mark.asyncio
class TestAsyncCommunityReportExtractor:
    @pytest.fixture
    def extractor(self) -> AsyncCommunityReportExtractor:
        return AsyncCommunityReportExtractor(
            llm_client=_mock_llm_with_content(json.dumps(SAMPLE_REPORT_JSON)),
            model="test-model",
            extraction_prompt=COMMUNITY_REPORT_PROMPT,
            max_report_length=1000,
        )

    async def test_extract_returns_structured_and_text_output(self, extractor):
        result = await extractor.extract("Entities\nid,title\n1,FOO")

        assert result.structured_output is not None
        assert result.structured_output.title == SAMPLE_REPORT_JSON["title"]
        assert "# Verdant Oasis Plaza and Unity March" in result.output
        assert "## Central location" in result.output
        assert "Verdant Oasis Plaza is the central entity." in result.output

    async def test_extract_formats_prompt_with_context_and_length(self):
        llm = _mock_llm_with_content(json.dumps(SAMPLE_REPORT_JSON))
        extractor = AsyncCommunityReportExtractor(
            llm_client=llm,
            model="test-model",
            extraction_prompt=COMMUNITY_REPORT_PROMPT,
            max_report_length=500,
        )
        context = "Entities\nhuman_readable_id,title\n5,PLAZA"

        await extractor.extract(context)

        messages = llm.client.chat.completions.create.await_args.kwargs["messages"]
        prompt = messages[0]["content"]
        assert context in prompt
        assert "500" in prompt

    async def test_extract_returns_empty_text_when_llm_fails(self):
        extractor = AsyncCommunityReportExtractor(
            llm_client=_mock_llm_with_content(None),
            model="test-model",
            extraction_prompt=COMMUNITY_REPORT_PROMPT,
            max_report_length=1000,
        )

        result = await extractor.extract("some context")

        assert result.structured_output is None
        assert result.output == ""

    def test_get_text_output_markdown_structure(self):
        report = CommunityReportResponse.model_validate(SAMPLE_REPORT_JSON)
        extractor = AsyncCommunityReportExtractor(
            llm_client=MagicMock(),
            model="test-model",
            extraction_prompt=COMMUNITY_REPORT_PROMPT,
            max_report_length=1000,
        )

        text = extractor._get_text_output(report)

        assert text.startswith("# Verdant Oasis Plaza and Unity March\n\n")
        assert "The community revolves around the Verdant Oasis Plaza." in text
        assert text.count("## ") == 2
