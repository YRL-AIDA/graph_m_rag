"""Gleaner FastAPI-сервер: OpenAI-совместимый endpoint для GLiNER NER-модели.

Загружает модель ``urchade/gliner_large-v2.1`` один раз при старте через lifespan,
предоставляет ``POST /v1/chat/completions``, совместимый с GleanerClient.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic-модели OpenAI Chat Completion
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Сообщение чата."""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Тело запроса ``POST /v1/chat/completions``."""
    messages: list[ChatMessage]
    model: str = "urchade/gliner_large-v2.1"

class Choice(BaseModel):
    """Вариант ответа."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    """Оценка токенов."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

# ---------------------------------------------------------------------------
# Глобальное состояние модели
# ---------------------------------------------------------------------------

_model: Any = None
_model_name: str = ""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_entity_types(system_content: str) -> list[str]:
    """Извлечь список типов сущностей из system prompt.

    Ожидаемый формат: ``"Entity types: Type1, Type2, Type3"``.
    Пустой system prompt → пустой список типов.
    """
    if not system_content:
        return []
    prefix = "Entity types:"
    if prefix.lower() in system_content.lower():
        idx = system_content.lower().index(prefix.lower()) + len(prefix)
        types_str = system_content[idx:].strip()
    else:
        types_str = system_content.strip()
    return [t.strip() for t in types_str.split(",") if t.strip()]


def _build_response(model: str, entities: list[dict[str, str]]) -> ChatCompletionResponse:
    """Собрать OpenAI Chat Completion response."""
    content = json.dumps(entities, ensure_ascii=False)
    tokens = max(len(content) // 4, 1)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[Choice(message=ChatMessage(role="assistant", content=content))],
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2),
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель при старте, освобождает ресурсы при остановке."""
    global _model, _model_name
    _model_name = os.getenv("GLINER_MODEL_NAME", "urchade/gliner_large-v2.1")
    logger.info("Загрузка модели GLiNER: %s", _model_name)
    try:
        from gliner import GLiNER  # noqa: PLC0415 — отложенный импорт
        _model = GLiNER.from_pretrained(_model_name)
        logger.info("Модель %s успешно загружена", _model_name)
    except Exception as exc:
        logger.critical("Ошибка загрузки модели %s: %s", _model_name, exc)
        raise RuntimeError(f"Failed to load model {_model_name}: {exc}") from exc
    yield
    _model = None
    logger.info("Модель %s выгружена", _model_name)


# ---------------------------------------------------------------------------
# FastAPI-приложение
# ---------------------------------------------------------------------------

app = FastAPI(title="Gleaner NER Server", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Извлечь сущности через GLiNER, вернуть в формате OpenAI Chat Completion.

    System message: ``"Entity types: Type1, Type2, ..."``.
    User message: текст для NER.
    """
    system_content = next((m.content for m in request.messages if m.role == "system"), "")
    user_content = next((m.content for m in request.messages if m.role == "user"), "")

    entity_types = _parse_entity_types(system_content)

    if not user_content:
        return _build_response(request.model, [])

    try:
        gliner_results = _model.predict_entities(user_content, labels=entity_types)
    except Exception as exc:
        logger.error("Ошибка инференса GLiNER: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    # GLiNER {"text", "label"} → {"name", "type"}
    entities = [{"name": r["text"], "type": r["label"]} for r in gliner_results]
    return _build_response(request.model, entities)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9596)
