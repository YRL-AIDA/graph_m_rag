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
from config import COMMUNITY_REPORT_PROMPT
from graphrag import AsyncLLMClient, remove_think_tags


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


def _log_test(check: str, expected: str, got: str) -> None:
    print(f"\n--- {check} ---")
    print(f"  Ожидается: {expected}")
    print(f"  Получено:  {got}")


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

        _log_test(
            "Валидный JSON парсится в CommunityReportResponse со всеми полями",
            f"title={SAMPLE_REPORT_JSON['title']!r}, rating=5.0, findings=2",
            f"title={report.title!r}, rating={report.rating}, findings={len(report.findings)}, "
            f"findings[0]={type(report.findings[0]).__name__}",
        )

    def test_missing_required_field_raises(self):
        incomplete = {k: v for k, v in SAMPLE_REPORT_JSON.items() if k != "title"}
        with pytest.raises(ValidationError) as exc_info:
            CommunityReportResponse.model_validate(incomplete)

        _log_test(
            "JSON без обязательного поля title вызывает ValidationError",
            "ValidationError",
            f"{type(exc_info.value).__name__} ({exc_info.value.error_count()} ошибок)",
        )

    def test_invalid_rating_type_raises(self):
        bad = {**SAMPLE_REPORT_JSON, "rating": "high"}
        with pytest.raises(ValidationError) as exc_info:
            CommunityReportResponse.model_validate(bad)

        _log_test(
            "Некорректный тип rating (строка вместо числа) вызывает ValidationError",
            "ValidationError",
            f"{type(exc_info.value).__name__} (rating={bad['rating']!r})",
        )

    def test_model_dump_json_roundtrip(self):
        report = CommunityReportResponse.model_validate(SAMPLE_REPORT_JSON)
        restored = CommunityReportResponse.model_validate_json(report.model_dump_json())
        assert restored == report

        _log_test(
            "Сериализация model_dump_json и обратный парсинг сохраняют модель",
            f"restored == original ({report.title!r})",
            f"restored == report: {restored == report}, title={restored.title!r}",
        )


# --- remove_think_tags (preprocessing before JSON parse) ---


class TestRemoveThinkTags:
    def test_strips_reasoning_prefix_before_json(self):
        raw = 'reasoning here jklkjlkjlkljlkj </think>\n{"title": "x"}'
        result = remove_think_tags(raw)
        assert result == '{"title": "x"}'

        _log_test(
            "remove_think_tags отрезает текст рассуждений перед JSON",
            '{"title": "x"}',
            result,
        )

    def test_plain_json_unchanged(self):
        raw = '{"title": "x"}'
        result = remove_think_tags(raw)
        assert result == raw

        _log_test(
            "Чистый JSON без префикса возвращается без изменений",
            raw,
            result,
        )

    def test_whitespace_trimmed(self):
        raw = '  {"a": 1}  '
        result = remove_think_tags(raw)
        assert result == '{"a": 1}'

        _log_test(
            "Пробелы по краям JSON обрезаются",
            '{"a": 1}',
            result,
        )


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

        _log_test(
            "generate_structured парсит обычный JSON-ответ LLM в Pydantic-модель",
            f"title={SAMPLE_REPORT_JSON['title']!r}, findings[0].summary='Central location'",
            f"title={result.title!r}, findings[0].summary={result.findings[0].summary!r}",
        )

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

        _log_test(
            "generate_structured извлекает JSON из markdown-блока ```json ... ```",
            "rating=5.0",
            f"rating={result.rating}",
        )

    async def test_parses_json_after_think_tags(self):
        content = (
            "internal reasoning </think>\n"
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

        _log_test(
            "generate_structured парсит JSON после текста рассуждений (think tags)",
            f"title={SAMPLE_REPORT_JSON['title']!r}",
            f"title={result.title!r}",
        )

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

        _log_test(
            "generate_structured передаёт в API response_format, model и temperature",
            "response_format={'type': 'json_object'}, model='test-model', temperature=0.1",
            f"response_format={call_kwargs['response_format']!r}, "
            f"model={call_kwargs['model']!r}, temperature={call_kwargs['temperature']}",
        )

    async def test_empty_content_returns_none(self):
        client = _mock_llm_with_content(None)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

        _log_test(
            "Пустой content от LLM возвращает None вместо модели",
            "None",
            repr(result),
        )

    async def test_invalid_json_returns_none(self):
        client = _mock_llm_with_content("not valid json {{{")

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

        _log_test(
            "Невалидный JSON в ответе LLM возвращает None",
            "None",
            repr(result),
        )

    async def test_schema_mismatch_returns_none(self):
        bad_json = json.dumps({"title": "only title"})
        client = _mock_llm_with_content(bad_json)

        result = await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            response_model=CommunityReportResponse,
        )

        assert result is None

        _log_test(
            "JSON без обязательных полей схемы возвращает None",
            "None (JSON не проходит валидацию CommunityReportResponse)",
            repr(result),
        )

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

        _log_test(
            "Исключение API (RuntimeError) перехватывается, возвращается None",
            "None",
            repr(result),
        )


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

        _log_test(
            "extract возвращает structured_output и markdown-текст отчёта",
            f"structured_output.title={SAMPLE_REPORT_JSON['title']!r}, "
            "output содержит заголовок и findings",
            f"structured_output.title={result.structured_output.title!r}, "
            f"output начинается с {result.output[:50]!r}...",
        )

    async def test_extract_formats_prompt_with_context_and_length(self):
        llm = _mock_llm_with_content(json.dumps(SAMPLE_REPORT_JSON))
        extractor = AsyncCommunityReportExtractor(
            llm_client=llm,
            model="test-model",
            extraction_prompt=COMMUNITY_REPORT_PROMPT,
            max_report_length=500,
        )
        context = "Entities\nid,title\nPLAZA|LOCATION,PLAZA"

        await extractor.extract(context)

        messages = llm.client.chat.completions.create.await_args.kwargs["messages"]
        prompt = messages[0]["content"]
        assert context in prompt
        assert "500" in prompt

        _log_test(
            "extract подставляет context и max_report_length в промпт для LLM",
            f"context и '500' присутствуют в промпте",
            f"context in prompt: {context in prompt}, '500' in prompt: {'500' in prompt}",
        )

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

        _log_test(
            "При ошибке LLM extract возвращает пустой output и structured_output=None",
            "structured_output=None, output=''",
            f"structured_output={result.structured_output!r}, output={result.output!r}",
        )

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

        _log_test(
            "_get_text_output формирует markdown с заголовком, summary и findings",
            "начинается с '# Verdant...', 2 секции '## '",
            f"начало={text[:40]!r}..., секций '## ': {text.count('## ')}",
        )
#@pytest.mark.asyncio
#class TestAsyncCommunityReportPyplines:
    