"""QwenClient — клиент модели Qwen3-VL-32B-Thinking через OpenAI-совместимый API.

Использует AsyncLLMClient из semantic_graph/graphrag.py для сетевых вызовов.
Реализует BaseModelClient: extract_entities (NER) и extract_relations (RE).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from semantic_graph.graphrag import AsyncLLMClient
from testdata.base_loader import BaseModelClient, PredictedEntity, PredictedRelation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Регулярки для парсинга ответа модели
# ---------------------------------------------------------------------------

_NER_LINE_RE = re.compile(r'\("entity"<\|>([^<]+)<\|>([^<]+)\)')
_RE_LINE_RE = re.compile(r'\("relationship"<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)\)')
_COMPLETE_MARKER = "<|COMPLETE|>"
_SEPARATOR = "##"


class QwenClient(BaseModelClient):
    """Клиент модели Qwen через OpenAI-совместимый API.

    Поддерживает раздельные вызовы NER (extract_entities) и RE (extract_relations).
    Combined-режим (NER+RE одним вызовом) реализуется на уровне ExperimentRunner.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        ner_prompt_path: str = "",
        re_prompt_path: str = "",
    ) -> None:
        """Инициализация QwenClient.

        Параметры конструктора читаются из переменных окружения, если не переданы явно.

        Args:
            base_url: URL OpenAI-совместимого API (env: QWEN_BASE_URL).
            api_key: API-ключ (env: QWEN_API_KEY).
            model: Идентификатор модели (env: QWEN_MODEL).
            ner_prompt_path: Путь к файлу NER-промпта (prompts/ner_prompt.md).
            re_prompt_path: Путь к файлу RE-промпта (prompts/re_prompt.md).
        """
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL", "http://192.168.19.127:8888/v1"
        )
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
        self.model = model or os.environ.get(
            "QWEN_MODEL", "Qwen/Qwen3-VL-32B-Thinking"
        )

        # Транспорт: AsyncLLMClient из semantic_graph.graphrag
        self.llm = AsyncLLMClient(base_url=self.base_url, api_key=self.api_key)

        # Загрузка промптов из файлов (Constitution P7)
        if ner_prompt_path:
            self._ner_prompt = Path(ner_prompt_path).read_text()
        else:
            self._ner_prompt = ""
        if re_prompt_path:
            self._re_prompt = Path(re_prompt_path).read_text()
        else:
            self._re_prompt = ""

    # -----------------------------------------------------------------------
    # NER — извлечение сущностей
    # -----------------------------------------------------------------------

    async def extract_entities(
        self, text: str, entity_types: list[str]
    ) -> list[PredictedEntity]:
        """Извлечь и классифицировать сущности из текста.

        Args:
            text: Входной текст для NER.
            entity_types: Список типов сущностей для извлечения.

        Returns:
            Список предсказанных сущностей (PredictedEntity).
        """
        if not self._ner_prompt:
            logger.warning("NER prompt is empty, cannot extract entities")
            return []

        # 1. Форматирование NER-промпта
        prompt = self._ner_prompt.format(
            input_text=text,
            entity_types=",".join(entity_types),
        )

        # 2. Вызов LLM
        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )

        # 3. llm.generate() вернул None — пустой результат
        if response is None:
            logger.warning(
                "llm.generate() returned None for NER request (model=%s)", self.model
            )
            return []

        # 4. Парсинг ответа
        return self._parse_ner_response(response)

    def _parse_ner_response(self, response: str) -> list[PredictedEntity]:
        """Распарсить NER-ответ модели в список PredictedEntity.

        Формат ответа: строки '("entity"<|>NAME<|>TYPE)', разделённые '##',
        завершается '<|COMPLETE|>'.

        Args:
            response: Сырой текст ответа модели.

        Returns:
            Список PredictedEntity — только валидные строки.
        """
        entities: list[PredictedEntity] = []
        lines = response.split(_SEPARATOR)
        has_complete = _COMPLETE_MARKER in response
        any_valid = False

        for line in lines:
            line = line.strip()

            # Проверка терминатора — дальше не парсим
            if line == _COMPLETE_MARKER:
                break

            match = _NER_LINE_RE.search(line)
            if match:
                any_valid = True
                name = match.group(1).strip()
                ent_type = match.group(2).strip()
                entities.append(PredictedEntity(name=name, type=ent_type))
            elif line:
                # Непустая строка, но не соответствует формату — пропускаем
                logger.warning(
                    "Skipping unparseable NER line: %r (model=%s)",
                    line[:200],
                    self.model,
                )

        if not any_valid and not has_complete:
            logger.warning(
                "NER response has no valid entities and no <|COMPLETE|> terminator "
                "(model=%s, response_preview=%r)",
                self.model,
                response[:300],
            )

        return entities

    # -----------------------------------------------------------------------
    # RE — извлечение отношений
    # -----------------------------------------------------------------------

    async def extract_relations(
        self,
        text: str,
        entities: list[PredictedEntity],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        """Извлечь отношения между заданными сущностями.

        Args:
            text: Входной текст для RE.
            entities: Список сущностей, между которыми ищутся отношения.
            relation_types: Список типов отношений для извлечения.

        Returns:
            Список предсказанных отношений (PredictedRelation).
        """
        if not self._re_prompt:
            logger.warning("RE prompt is empty, cannot extract relations")
            return []

        # 1. Построение entities_list: строки NAME|TYPE через \n
        entities_list = "\n".join(
            f"{entity.name}|{entity.type}" for entity in entities
        )

        # 2. Форматирование RE-промпта
        prompt = self._re_prompt.format(
            input_text=text,
            entities_list=entities_list,
            relation_types=",".join(relation_types),
        )

        # 3. Вызов LLM
        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )

        # 4. llm.generate() вернул None — пустой результат
        if response is None:
            logger.warning(
                "llm.generate() returned None for RE request (model=%s)", self.model
            )
            return []

        # 5. Парсинг ответа
        return self._parse_re_response(response)

    def _parse_re_response(self, response: str) -> list[PredictedRelation]:
        """Распарсить RE-ответ модели в список PredictedRelation.

        Формат ответа: строки '("relationship"<|>SRC<|>TGT<|>TYPE)', разделённые
        '##', завершается '<|COMPLETE|>'.

        Args:
            response: Сырой текст ответа модели.

        Returns:
            Список PredictedRelation — только валидные строки.
        """
        relations: list[PredictedRelation] = []
        lines = response.split(_SEPARATOR)
        has_complete = _COMPLETE_MARKER in response
        any_valid = False

        for line in lines:
            line = line.strip()

            # Проверка терминатора — дальше не парсим
            if line == _COMPLETE_MARKER:
                break

            match = _RE_LINE_RE.search(line)
            if match:
                any_valid = True
                head = match.group(1).strip()
                tail = match.group(2).strip()
                rel_type = match.group(3).strip()
                relations.append(
                    PredictedRelation(head=head, tail=tail, type=rel_type)
                )
            elif line:
                # Непустая строка, но не соответствует формату — пропускаем
                logger.warning(
                    "Skipping unparseable RE line: %r (model=%s)",
                    line[:200],
                    self.model,
                )

        if not any_valid and not has_complete:
            logger.warning(
                "RE response has no valid relations and no <|COMPLETE|> terminator "
                "(model=%s, response_preview=%r)",
                self.model,
                response[:300],
            )

        return relations
