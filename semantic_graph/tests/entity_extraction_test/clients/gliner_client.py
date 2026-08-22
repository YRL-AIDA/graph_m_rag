"""Gleaner client for entity extraction via OpenAI-compatible API.

Gleaner — span-based NER model (Gliner) served via FastAPI with an
OpenAI-compatible POST /v1/chat/completions endpoint. Supports only NER
(entity extraction); RE is not implemented.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import openai

from testdata.base_loader import BaseModelClient, PredictedEntity, PredictedRelation

_ENTITY_DIR = str(Path(__file__).resolve().parent.parent)
if _ENTITY_DIR not in sys.path:
    sys.path.insert(0, _ENTITY_DIR)
from metrics.metrics import normalize_type
from metrics.entity_mappings import get_dataset_entity_types

logger = logging.getLogger(__name__)


class GleanerClient(BaseModelClient):
    """NER-клиент, работающий через Gleaner (Gliner) FastAPI-сервер.

    Gleaner предоставляет OpenAI-совместимый эндпоинт
    ``POST /v1/chat/completions``. Клиент отправляет типы сущностей
    в system prompt, текст — в user message и парсит JSON-ответ.

    Поддерживает только NER; вызов ``extract_relations`` всегда
    выбрасывает ``NotImplementedError``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        entity_types_source: str = "ontonotes5",
    ) -> None:
        self.base_url = base_url or os.getenv(
            "GLINER_BASE_URL", "http://<remote_host>:<port>/v1"
        )
        self.api_key = api_key or os.getenv("GLINER_API_KEY", "EMPTY")
        self.model = model or os.getenv(
            "GLINER_MODEL", "urchade/gliner_large-v2.1"
        )
        self.source_dataset = entity_types_source
        self.entity_types = get_dataset_entity_types(entity_types_source)

        _validate_url(self.base_url)

        self.client = openai.AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key
        )

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    async def extract_entities(
        self, text: str, entity_types: list[str] | None = None
    ) -> list[PredictedEntity]:
        """Извлечь и классифицировать сущности из текста.

        Формирует запрос: entity_types в system prompt, text — в user
        message. Ожидает JSON-список вида ``[{"name": ..., "type": ...}]``.

        Returns:
            Список ``PredictedEntity``. Пустой список при ошибках парсинга
            или пустом ответе API.
        """
        types = entity_types if entity_types else self.entity_types

        system_prompt = f"Entity types: {', '.join(types)}"

        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                model=self.model,
            )
        except openai.APIConnectionError as exc:
            logger.error("Gleaner connection error: %s", exc)
            raise ConnectionError(f"Gleaner API unreachable: {exc}") from exc
        except openai.APITimeoutError as exc:
            logger.error("Gleaner timeout: %s", exc)
            raise ConnectionError(f"Gleaner API timeout: {exc}") from exc

        raw_content = response.choices[0].message.content

        if not raw_content:
            return []

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning("Gleaner returned invalid JSON: %s", raw_content)
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "Gleaner returned non-list JSON: %s", raw_content
            )
            return []

        result: list[PredictedEntity] = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item and "type" in item:
                name = item["name"]
                raw_type = item["type"]
                canonical_type = normalize_type(raw_type, types)
                if canonical_type is not None:
                    result.append(PredictedEntity(name=name, type=canonical_type))
                else:
                    logger.warning(
                        "Gleaner: skipping entity %r with unknown type %r (allowed: %s)",
                        name, raw_type, types,
                    )

        return result

    # ------------------------------------------------------------------
    # RE — NOT supported
    # ------------------------------------------------------------------

    async def extract_relations(
        self,
        text: str,
        entities: list[PredictedEntity],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        """Gleaner — span-детектор, не генеративная модель. RE не поддерживается."""
        raise NotImplementedError("GleanerClient does not support RE")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_UNRESOLVED_TEMPLATE_RE = re.compile(r"<[^>]+>")


def _validate_url(url: str) -> None:
    """Проверяет, что URL не является шаблоном-заглушкой (содержит ``<...>``)."""
    if _UNRESOLVED_TEMPLATE_RE.search(url):
        raise ValueError(
            f"base_url содержит неразрешённый шаблон: {url!r}. "
            f"Установите переменную окружения GLINER_BASE_URL или "
            f"передайте реальный base_url."
        )
