"""HybridClient — делегирует NER одному клиенту (e.g. UniNer, Gleaner),
а RE — другому (e.g. Qwen).

Используется в экспериментах E5 (UniNer NER + Qwen RE) и E6 (Gleaner NER + Qwen RE).
"""

from __future__ import annotations

from testdata.base_loader import BaseModelClient, PredictedEntity, PredictedRelation


class HybridClient(BaseModelClient):
    """Гибридный клиент: NER извлекается через `ner_client`, RE — через `re_client`.

    Исключения от делегированных клиентов пробрасываются наверх без обработки.
    """

    def __init__(
        self,
        ner_client: BaseModelClient,
        re_client: BaseModelClient,
    ) -> None:
        self.ner_client = ner_client
        self.re_client = re_client

    async def extract_entities(
        self, text: str, entity_types: list[str]
    ) -> list[PredictedEntity]:
        """Делегирует извлечение сущностей NER-клиенту."""
        return await self.ner_client.extract_entities(text, entity_types)

    async def extract_relations(
        self,
        text: str,
        entities: list[PredictedEntity],
        relation_types: list[str],
        relation_type_descriptions: str | None = None,
        allowed_relation_types: str | None = None,
    ) -> list[PredictedRelation]:
        """Делегирует извлечение отношений RE-клиенту."""
        return await self.re_client.extract_relations(
            text, entities, relation_types,
            relation_type_descriptions=relation_type_descriptions,
            allowed_relation_types=allowed_relation_types,
        )
