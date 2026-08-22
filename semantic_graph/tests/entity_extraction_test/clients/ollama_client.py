"""OllamaClient — клиент моделей через нативный Ollama API (POST /api/chat).

Использует httpx.AsyncClient для сетевых вызовов.
Реализует BaseModelClient: extract_entities (NER) и extract_relations (RE).
Поддерживает combined-режим: extract_entities_and_relations (NER+RE одним вызовом).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import httpx

# ── sys.path для импорта normalize_type (тот же паттерн, что в QwenClient) ──
_ENTITY_DIR = str(Path(__file__).resolve().parent.parent)
if _ENTITY_DIR not in sys.path:
    sys.path.insert(0, _ENTITY_DIR)

from metrics.metrics import normalize_type  # noqa: E402
from metrics.entity_mappings import get_dataset_entity_types  # noqa: E402
from testdata.base_loader import BaseModelClient, PredictedEntity, PredictedRelation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Регулярки для парсинга ответа модели (идентичны QwenClient)
# ---------------------------------------------------------------------------

_NER_LINE_RE = re.compile(r'\("entity"<\|>([^<]+)<\|>([^<]+)\)')
_RE_LINE_RE = re.compile(r'\("relationship"<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)\)')
_COMPLETE_MARKER = "<|COMPLETE|>"
_SEPARATOR = "##"


class OllamaClient(BaseModelClient):
    """Клиент моделей Ollama через нативный REST API.

    Поддерживает раздельные вызовы NER (extract_entities) и RE (extract_relations),
    а также combined-режим (extract_entities_and_relations) — NER+RE одним вызовом.

    Транспорт: httpx.AsyncClient → POST /api/chat (нативный Ollama API).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        think: bool | str | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        ner_prompt_path: str = "",
        re_prompt_path: str = "",
        combined_prompt_path: str = "",
        keep_alive: str | None = None,
        entity_types_source: str = "ontonotes5",
    ) -> None:
        """Инициализация OllamaClient.

        Параметры конструктора читаются из переменных окружения, если не переданы явно.

        Args:
            base_url: URL Ollama-сервера (env: OLLAMA_BASE_URL).
            model: Имя модели в Ollama (env: OLLAMA_MODEL).
            think: Режим рассуждений: False, True, "low", "medium", "high", "max"
                   (env: OLLAMA_THINK).
            num_predict: Макс. число токенов генерации (env: OLLAMA_NUM_PREDICT).
            num_ctx: Размер контекстного окна (env: OLLAMA_NUM_CTX).
            ner_prompt_path: Путь к файлу NER-промпта (prompts/ner_prompt.md).
            re_prompt_path: Путь к файлу RE-промпта (prompts/re_prompt.md).
            combined_prompt_path: Путь к файлу combined-промпта (prompts/combined_prompt.md).
            keep_alive: Время удержания модели в памяти (env: OLLAMA_KEEP_ALIVE).
        """
        # ── Параметры с env-фоллбеками ──
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://192.168.55.242:7869"
        )
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")
        self.keep_alive = keep_alive or os.environ.get("OLLAMA_KEEP_ALIVE", "5m")

        # think: парсинг из env или параметра конструктора
        think_raw = str(think) if think is not None else os.environ.get("OLLAMA_THINK", "False")
        if think_raw.lower() == "false":
            self.think: bool | str = False
        elif think_raw.lower() == "true":
            self.think = True
        else:
            self.think = think_raw  # "low", "medium", "high", "max"

        # num_predict: парсинг из env, если None
        if num_predict is None:
            raw = os.environ.get("OLLAMA_NUM_PREDICT", "256")
            self.num_predict = int(raw)
        else:
            self.num_predict = num_predict

        # num_ctx: парсинг из env, если None
        if num_ctx is None:
            raw = os.environ.get("OLLAMA_NUM_CTX", "4096")
            self.num_ctx = int(raw)
        else:
            self.num_ctx = num_ctx

        # ── Транспорт: httpx.AsyncClient ──
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        # ── Загрузка промптов из файлов (Constitution P7) ──
        if ner_prompt_path:
            self._ner_prompt = Path(ner_prompt_path).read_text()
        else:
            self._ner_prompt = ""
        if re_prompt_path:
            self._re_prompt = Path(re_prompt_path).read_text()
        else:
            self._re_prompt = ""
        if combined_prompt_path:
            self._combined_prompt = Path(combined_prompt_path).read_text()
        else:
            self._combined_prompt = ""

        self.source_dataset = entity_types_source
        self.entity_types = get_dataset_entity_types(entity_types_source)

    # -----------------------------------------------------------------------
    # _chat — базовый метод отправки запроса к Ollama /api/chat
    # -----------------------------------------------------------------------

    async def _chat(self, messages: list[dict]) -> str | None:
        """Отправить запрос к Ollama /api/chat и вернуть content ответа.

        Args:
            messages: Список сообщений в формате [{"role": ..., "content": ...}].

        Returns:
            Текст ответа модели (message.content) или None, если ключ отсутствует.

        Raises:
            ConnectionError: при сетевых ошибках или HTTP-ошибках (4xx, 5xx).
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }

        # think — только если не False
        if self.think is not False:
            payload["think"] = self.think

        # options: num_predict, num_ctx
        options: dict = {}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/chat", json=payload)
            if response.is_error:
                logger.error(
                    "Ollama HTTP error %s: %s",
                    response.status_code,
                    response.text[:500],
                )
                raise ConnectionError(
                    f"Ollama returned HTTP {response.status_code}"
                )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama HTTP error %s for /api/chat (model=%s): %s",
                exc.response.status_code,
                self.model,
                exc,
            )
            raise ConnectionError(
                f"Ollama /api/chat returned HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
            logger.error(
                "Ollama connection error for /api/chat (model=%s, url=%s): %s",
                self.model,
                self.base_url,
                exc,
            )
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}"
            ) from exc

        data = response.json()
        print(data)
        content = data.get("message", {}).get("content")
        if content is None:
            logger.warning(
                "Ollama response missing message.content (model=%s, keys=%s)",
                self.model,
                list(data.keys()),
            )
            return None
        return content

    # -----------------------------------------------------------------------
    # NER — извлечение сущностей
    # -----------------------------------------------------------------------

    async def extract_entities(
        self, text: str, entity_types: list[str] | None = None
    ) -> list[PredictedEntity]:
        """Извлечь и классифицировать сущности из текста.

        Args:
            text: Входной текст для NER.
            entity_types: Список типов сущностей для извлечения.

        Returns:
            Список предсказанных сущностей (PredictedEntity).
        """
        types = entity_types if entity_types else self.entity_types

        if not self._ner_prompt:
            logger.warning("NER prompt is empty, cannot extract entities")
            return []

        # 1. Форматирование NER-промпта
        prompt = self._ner_prompt.format(
            input_text=text,
            entity_types=",".join(types),
        )

        # 2. Вызов LLM через Ollama /api/chat
        response = await self._chat(
            messages=[{"role": "user", "content": prompt}]
        )

        # 3. _chat вернул None — пустой результат
        if response is None:
            logger.warning(
                "_chat() returned None for NER request (model=%s)", self.model
            )
            return []

        # 4. Парсинг ответа
        return self._parse_ner_response(response, types)

    def _parse_ner_response(
        self, response: str, entity_types: list[str]
    ) -> list[PredictedEntity]:
        """Распарсить NER-ответ модели в список PredictedEntity.

        Формат ответа: строки '("entity"<|>NAME<|>TYPE)', разделённые '##',
        завершается '<|COMPLETE|>'.

        Args:
            response: Сырой текст ответа модели.
            entity_types: Список канонических типов сущностей для валидации.

        Returns:
            Список PredictedEntity — только валидные строки, прошедшие
            валидацию типа через normalize_type.
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
                name = match.group(1).strip()
                ent_type = match.group(2).strip()
                # Валидация типа через normalize_type
                canonical_type = normalize_type(ent_type, entity_types)
                if canonical_type is not None:
                    any_valid = True
                    entities.append(
                        PredictedEntity(name=name, type=canonical_type)
                    )
                else:
                    logger.warning(
                        "Skipping entity %r with unknown type %r (allowed: %s)",
                        name,
                        ent_type,
                        entity_types,
                    )
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
        relation_type_descriptions: str | None = None,
        allowed_relation_types: str | None = None,
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
            relation_types=relation_type_descriptions or ",".join(relation_types),
            allowed_relation_types=allowed_relation_types or str(relation_types + ["None"]),
        )

        # 3. Вызов LLM через Ollama /api/chat
        response = await self._chat(
            messages=[{"role": "user", "content": prompt}]
        )

        # 4. _chat вернул None — пустой результат
        if response is None:
            logger.warning(
                "_chat() returned None for RE request (model=%s)", self.model
            )
            return []

        # 5. Парсинг ответа
        return self._parse_re_response(response, relation_types)

    def _parse_re_response(
        self, response: str, relation_types: list[str]
    ) -> list[PredictedRelation]:
        """Распарсить RE-ответ модели в список PredictedRelation.

        Формат ответа: строки '("relationship"<|>SRC<|>TGT<|>TYPE)', разделённые
        '##', завершается '<|COMPLETE|>'.

        Args:
            response: Сырой текст ответа модели.
            relation_types: Список канонических типов отношений для валидации.

        Returns:
            Список PredictedRelation — только валидные строки, прошедшие
            валидацию типа через normalize_type.
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
                head = match.group(1).strip()
                tail = match.group(2).strip()
                rel_type = match.group(3).strip()
                # Валидация типа через normalize_type
                canonical_type = normalize_type(rel_type, relation_types)
                if canonical_type is not None:
                    any_valid = True
                    relations.append(
                        PredictedRelation(
                            head=head, tail=tail, type=canonical_type
                        )
                    )
                else:
                    logger.warning(
                        "Skipping relation (%r, %r) with unknown type %r (allowed: %s)",
                        head,
                        tail,
                        rel_type,
                        relation_types,
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

    # -----------------------------------------------------------------------
    # Combined (E3) — извлечение сущностей и отношений одним вызовом
    # -----------------------------------------------------------------------

    async def extract_entities_and_relations(
        self, text: str, entity_types: list[str] | None = None, relation_types: list[str] | None = None,
        relation_type_descriptions: str | None = None,
        allowed_relation_types: str | None = None,
    ) -> tuple[list[PredictedEntity], list[PredictedRelation]]:
        """Извлечь сущности и отношения ОДНИМ вызовом LLM (E3 combined_single_call).

        Returns:
            Кортеж (entities, relations).
        """
        types = entity_types if entity_types else self.entity_types
        relation_types = relation_types or []

        if not self._combined_prompt:
            logger.warning(
                "Combined prompt is empty, cannot extract entities and relations"
            )
            return [], []

        # 1. Форматирование combined-промпта
        prompt = self._combined_prompt.format(
            input_text=text,
            entity_types=",".join(types),
            relation_types=relation_type_descriptions or ",".join(relation_types),
            allowed_relation_types=allowed_relation_types or str(relation_types + ["None"]),
        )

        # 2. Вызов LLM — ОДИН раз
        response = await self._chat(
            messages=[{"role": "user", "content": prompt}]
        )

        # 3. _chat вернул None
        if response is None:
            logger.warning(
                "_chat() returned None for combined request (model=%s)",
                self.model,
            )
            return [], []

        # 4. Парсинг combined-ответа
        return self._parse_combined_response(response, types, relation_types)

    def _parse_combined_response(
        self,
        response: str,
        entity_types: list[str],
        relation_types: list[str],
    ) -> tuple[list[PredictedEntity], list[PredictedRelation]]:
        """Распарсить combined-ответ: сущности И отношения из одного текста.

        Формат: строки ("entity"<|>NAME<|>TYPE) и ("relationship"<|>SRC<|>TGT<|>TYPE),
        разделённые ##, с единым терминатором <|COMPLETE|>.
        """
        entities: list[PredictedEntity] = []
        relations: list[PredictedRelation] = []
        lines = response.split(_SEPARATOR)
        has_complete = _COMPLETE_MARKER in response

        for line in lines:
            line = line.strip()
            if line == _COMPLETE_MARKER:
                break

            # Пробуем NER
            ner_match = _NER_LINE_RE.search(line)
            if ner_match:
                name = ner_match.group(1).strip()
                ent_type = ner_match.group(2).strip()
                # Валидация типа
                canonical_type = normalize_type(ent_type, entity_types)
                if canonical_type is not None:
                    entities.append(
                        PredictedEntity(name=name, type=canonical_type)
                    )
                else:
                    logger.warning(
                        "Combined response: skipping entity %r with unknown type %r (allowed: %s)",
                        name,
                        ent_type,
                        entity_types,
                    )
                continue

            # Пробуем RE
            re_match = _RE_LINE_RE.search(line)
            if re_match:
                head = re_match.group(1).strip()
                tail = re_match.group(2).strip()
                rel_type = re_match.group(3).strip()
                # Валидация типа
                canonical_type = normalize_type(rel_type, relation_types)
                if canonical_type is not None:
                    relations.append(
                        PredictedRelation(
                            head=head, tail=tail, type=canonical_type
                        )
                    )
                else:
                    logger.warning(
                        "Combined response: skipping relation (%r, %r) with unknown type %r (allowed: %s)",
                        head,
                        tail,
                        rel_type,
                        relation_types,
                    )
                continue

            # Непустая строка без совпадений
            if line:
                logger.warning("Skipping unparseable combined line: %r", line[:200])

        if not entities and not relations and not has_complete:
            logger.warning(
                "Combined response has no valid content and no <|COMPLETE|>"
            )

        return entities, relations
