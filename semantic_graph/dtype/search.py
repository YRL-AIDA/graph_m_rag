"""Pydantic-модели для эндпоинта POST /search."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


class Proportions(BaseModel):
    """Доли токенов для каждого типа объектов поисковой выдачи."""

    text_units: float = Field(default=0.5, description="Доля токенов для текстовых блоков")
    entities: float = Field(default=0.25, description="Доля токенов для сущностей")
    communities: float = Field(default=0.25, description="Доля токенов для сообществ")


class TokensBreakdown(BaseModel):
    """Разбивка использованных/оставшихся токенов по типам пулов."""

    text_units: int = 0
    entities: int = 0
    communities: int = 0


class SearchStatistics(BaseModel):
    """Статистика выполнения поискового запроса."""

    processing_time_ms: int
    tokens_used: TokensBreakdown
    tokens_remaining: TokensBreakdown
    entities_extracted_from_question: int
    entities_matched_in_graph: int
    entities_search_misses: int
    fallback_used: bool
    total_items: TokensBreakdown


class SearchRequest(BaseModel):
    """Тело запроса POST /search."""

    question: str = Field(..., min_length=1, description="Строка вопроса пользователя")
    max_tokens: int = Field(..., gt=0, description="Максимальный суммарный размер вывода в токенах")
    proportions: Proportions = Field(
        default_factory=lambda: Proportions(text_units=0.5, entities=0.25, communities=0.25),
        description="Доли токенов для text_units, entities, communities (сумма = 1.0)",
    )
    documents_filter: str = Field(
        default="text_only",
        description="Режим фильтрации коллекции documents: text_only или all",
    )

    @model_validator(mode="after")
    def validate_proportions(self) -> "SearchRequest":
        """Валидация поля proportions: сумма значений должна быть 1.0 ± 0.001, все ≥ 0."""
        props = self.proportions

        # Проверка неотрицательности
        if props.text_units < 0 or props.entities < 0 or props.communities < 0:
            raise ValueError("proportions values must be non-negative")

        # Проверка суммы
        total = props.text_units + props.entities + props.communities
        if abs(total - 1.0) > 0.001:
            raise ValueError("proportions must sum to 1.0")

        # Проверка documents_filter
        if self.documents_filter not in {"text_only", "all"}:
            raise ValueError("documents_filter must be 'text_only' or 'all'")

        return self


class SearchResponse(BaseModel):
    """Ответ POST /search."""

    text_units: List[str] = Field(default_factory=list, description="Текстовые блоки из коллекции documents")
    entities: List[str] = Field(default_factory=list, description="Описания сущностей и их связей")
    communities: List[str] = Field(default_factory=list, description="Описания сообществ (community summaries)")
    statistics: SearchStatistics
