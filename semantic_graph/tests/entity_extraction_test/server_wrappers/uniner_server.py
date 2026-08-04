"""FastAPI-сервер для модели UniNer (Universal-NER/UniNER-7B-all).

Загружает модель один раз в lifespan и обслуживает запросы NER через POST /extract.
Совместим с клиентом UniNerClient (clients/uniner_client.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.environ.get("UNINER_MODEL_NAME", "Universal-NER/UniNER-7B-all")
MAX_NEW_TOKENS: int = 512
DEVICE_MAP: str = "auto"

# ---------------------------------------------------------------------------
# Pydantic-модели
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    """Тело запроса к /extract."""

    text: str = Field(..., description="Исходный текст для NER")
    entity_types: list[str] = Field(..., min_length=1, description="Типы сущностей для извлечения")


class Entity(BaseModel):
    """Одна извлечённая сущность."""

    name: str
    type: str


class ExtractResponse(BaseModel):
    """Ответ эндпоинта /extract."""

    entities: list[Entity]


# ---------------------------------------------------------------------------
# Глобальные переменные модели (заполняются в lifespan)
# ---------------------------------------------------------------------------

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None


# ---------------------------------------------------------------------------
# Формирование промпта
# ---------------------------------------------------------------------------


def _build_prompt(text: str, entity_types: list[str]) -> str:
    """Собрать UniNer-промпт в формате Llama-2-chat.

    Используется шаблон, ожидаемый моделью Universal-NER/UniNER-7B-all.
    """
    joined_types = ", ".join(entity_types)
    return (
        f"[INST] <<SYS>>\n"
        f"You are a Named Entity Recognition model. Given a text and a set of "
        f'entity types, extract all entities of the specified types. List each '
        f'entity as a JSON object with fields "name" (entity text) and "type" '
        f"(entity type). Output only the JSON objects, one per line.\n"
        f"<</SYS>>\n\n"
        f"Entity types: {joined_types}\n"
        f"Text: {text} [/INST]"
    )


# ---------------------------------------------------------------------------
# Парсинг вывода модели
# ---------------------------------------------------------------------------


def _parse_entities(output_text: str) -> list[Entity]:
    """Извлечь список Entity из сырого вывода модели.

    Модель может вернуть JSON-объекты в разных форматах:
      - {"name": "Apple", "type": "Organization"}
      - {"entity": "Steve Jobs", "type": "Person"}
      - вперемешку с лишним текстом

    Парсим построчно, пытаясь декодировать JSON.
    Поддерживаются ключи 'name' и 'entity' для имени сущности.
    """
    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()  # дедупликация (name, type)

    for line in output_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Ищем JSON-объект в строке (модель может добавить префиксный текст)
        json_match = re.search(r"\{[^{}]*\}", stripped)
        if not json_match:
            continue

        try:
            obj = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.debug("Не удалось распарсить строку как JSON: %s", stripped)
            continue

        # Извлекаем имя сущности: пробуем ключи 'name', затем 'entity'
        name = obj.get("name") or obj.get("entity")
        ent_type = obj.get("type")

        if not name or not ent_type:
            logger.debug("Пропущен объект без name/entity или type: %s", obj)
            continue

        key = (str(name), str(ent_type))
        if key in seen:
            continue
        seen.add(key)

        entities.append(Entity(name=str(name), type=str(ent_type)))

    return entities


def _extract_assistant_response(raw_output: str) -> str:
    """Вырезать только ответ ассистента — всё после маркера [/INST]."""
    marker = "[/INST]"
    idx = raw_output.rfind(marker)
    if idx != -1:
        return raw_output[idx + len(marker):].strip()
    # Если маркер не найден, возвращаем как есть (модель могла не включить промпт)
    logger.debug("Маркер [/INST] не найден в выводе, используется полный ответ")
    return raw_output.strip()


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------


async def _run_inference(text: str, entity_types: list[str]) -> list[Entity]:
    """Выполнить полный пайплайн: промпт → инференс → парсинг.

    Raises:
        HTTPException: При ошибках модели (500).
    """
    assert _tokenizer is not None, "Tokenizer не инициализирован"
    assert _model is not None, "Модель не инициализирована"

    prompt = _build_prompt(text, entity_types)
    logger.debug("UniNer prompt (len=%d): %.200s...", len(prompt), prompt)

    try:
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        raw_output = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        logger.error("Ошибка инференса UniNer: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    assistant_output = _extract_assistant_response(raw_output)
    logger.debug("UniNer assistant output: %.300s", assistant_output)

    entities = _parse_entities(assistant_output)

    if not entities:
        logger.warning(
            "UniNer вернул пустой список сущностей. Вывод модели: %.200s",
            assistant_output,
        )

    return entities


# ---------------------------------------------------------------------------
# Lifespan — загрузка модели при старте
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузить токенизатор и модель один раз при старте сервера."""
    global _tokenizer, _model

    logger.info("Загрузка токенизатора UniNer: %s", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info(
        "Загрузка модели UniNer: %s (dtype=bfloat16, device_map=%s)",
        MODEL_NAME,
        DEVICE_MAP,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE_MAP,
    )
    _model.eval()
    logger.info("Модель UniNer загружена на устройство: %s", _model.device)

    yield  # Сервер работает

    # Очистка (на практике почти никогда не вызывается, но держим для порядка)
    logger.info("Выгрузка модели UniNer")
    del _model
    del _tokenizer
    _model = None
    _tokenizer = None


# ---------------------------------------------------------------------------
# FastAPI-приложение
# ---------------------------------------------------------------------------

app = FastAPI(title="UniNer Server", version="0.1.0", lifespan=lifespan)


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    """Извлечь именованные сущности из текста.

    Принимает текст и список типов сущностей, возвращает найденные сущности
    в формате {"entities": [{"name": "...", "type": "..."}]}.

    Args:
        request: Тело запроса с полями text и entity_types.

    Returns:
        ExtractResponse со списком извлечённых сущностей.

    Raises:
        HTTPException 500: При ошибках инференса модели.
    """
    entities = await _run_inference(request.text, request.entity_types)
    logger.info(
        "UniNer /extract: text_len=%d, types=%s → %d entities",
        len(request.text),
        request.entity_types,
        len(entities),
    )
    return ExtractResponse(entities=entities)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9597)