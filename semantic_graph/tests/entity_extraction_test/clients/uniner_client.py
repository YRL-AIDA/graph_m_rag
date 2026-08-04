"""Клиент к UniNer NER-модели (Universal-NER/UniNER-7B-all) через кастомный FastAPI-сервер.

NER-only модель. RE не поддерживается.
Используется в экспериментах E1, E5 матрицы.
"""

from __future__ import annotations

import logging

import httpx

from testdata.base_loader import BaseModelClient, PredictedEntity, PredictedRelation

logger = logging.getLogger(__name__)


class UniNerClient(BaseModelClient):
    """Клиент модели UniNer.

    Отправляет текст и типы сущностей на серверный эндпоинт /extract.
    Промпт генерируется на серверной стороне — клиент не формирует промпт.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def extract_entities(
        self, text: str, entity_types: list[str]
    ) -> list[PredictedEntity]:
        """Извлечь и классифицировать сущности из текста через UniNer-сервер.

        Args:
            text: Исходный текст для NER.
            entity_types: Список типов сущностей для извлечения.

        Returns:
            Список предсказанных сущностей.

        Raises:
            ConnectionError: При ошибках соединения, таймауте или HTTP-ошибках (4xx, 5xx).
        """
        try:
            response = await self.client.post(
                "/extract", json={"text": text, "entity_types": entity_types}
            )
            response.raise_for_status()
            data = response.json()
            entities_raw = data["entities"]
        except httpx.ConnectError as e:
            logger.error("Ошибка подключения к UniNer: %s", e)
            raise ConnectionError from e
        except httpx.TimeoutException as e:
            logger.error("Таймаут запроса к UniNer: %s", e)
            raise ConnectionError from e
        except httpx.HTTPStatusError as e:
            logger.error("HTTP-ошибка UniNer: %s", e)
            raise ConnectionError from e
        except (KeyError, ValueError) as e:
            logger.warning("Невалидный ответ UniNer (отсутствует ключ 'entities' или невалидный JSON): %s", e)
            return []

        return [PredictedEntity(name=item["name"], type=item["type"]) for item in entities_raw]

    async def extract_relations(
        self,
        text: str,
        entities: list[PredictedEntity],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        """Выбросить NotImplementedError — UniNer не поддерживает RE.

        Поведение не логируется — это ожидаемое поведение для NER-only модели.
        """
        raise NotImplementedError("UniNerClient не поддерживает RE")
