# Data Model: Embeddings API Models

## Purpose
Модели данных для взаимодействия с внешним сервисом эмбеддингов (Qwen3-Emb) через HTTP-клиент `qwen3_emb_client.py`. Используются для сериализации запросов на получение эмбеддингов и десериализации ответов. Поддерживают текстовые, графические и смешанные (image+text) эмбеддинги.

## Schema

```python
from typing import Annotated, List, Union
from pydantic import BaseModel, ConfigDict, SkipValidation


class Message(BaseModel):
    """Элемент контента для эмбеддирования: текст, изображение или URL изображения."""
    model_config = ConfigDict(extra='ignore')
    type: str                            # Тип контента: "text", "image", "image/text", "image_url"
    text: Union[str, None] = None        # Текстовое содержимое (для type="text" или "image/text")
    image: Union[str, None] = None       # Base64-изображение (для type="image" или "image/text")
    image_url: Union[str, None] = None   # URL изображения (для type="image_url")


class EmbedRequest(BaseModel):
    """Запрос на получение эмбеддингов для списка сообщений."""
    messages: List[Message]              # Список элементов для эмбеддирования


class MessageEmbedding(BaseModel):
    """Результат эмбеддирования одного сообщения."""
    message_id: int                      # Индекс сообщения в исходном запросе (0-based)
    embedding: List[float]               # Вектор эмбеддинга


class EmbedSuccessResponse(BaseModel):
    """Успешный ответ от сервиса эмбеддингов — содержит эмбеддинги для всех сообщений."""
    messages: List[MessageEmbedding]     # Список результатов эмбеддирования по сообщениям


class EmbedResponse(BaseModel):
    """Альтернативная модель ответа — эмбеддинг одного сообщения
       (используется как внутренний контракт, не напрямую от API сервиса)."""
    message_id: int                      # Индекс сообщения
    embedding: List[float]               # Вектор эмбеддинга
```

## Storage
- **Хранение**: in-memory / HTTP transport only (сериализация запроса → JSON body → HTTP POST → десериализация ответа)
- Модели не персистятся в БД. Эмбеддинги после получения сохраняются в MinIO (`embeddings/{file_hash}/`) и загружаются в Qdrant payload через `qdrant_client_api.py`.

## Relationships
- **`Message`** → используется в `EmbeddingClient` (`app/src/qwen3_emb_client.py`) для построения тела запроса
- **`EmbedRequest`** → сериализуется (`model_dump()`) в JSON-body HTTP POST на `{base_url}/embed`
- **`EmbedSuccessResponse`** → десериализуется из JSON-ответа сервиса эмбеддингов
- **`EmbedResponse`** → декларирована в `schemas/embeddings.py`, но не импортируется напрямую из других модулей (фактически используется только `EmbedSuccessResponse`)
- **Конфигурация**: `EmbeddingClient` получает `base_url` и `timeout` из `EmbeddingSettings` (`app/config/settings.py`) через глобальный экземпляр `settings`

## Constraints
- **`Message.type`** — обязательное строковое поле, не имеет валидации допустимых значений на уровне модели (принимает любое строковое значение); фактически в коде клиента используются значения: `"text"`, `"image"`, `"image/text"`, `"image_url"`
- **`Message.model_config`** — `extra='ignore'`: лишние поля из JSON игнорируются при создании модели
- **`Message.text`, `Message.image`, `Message.image_url`** — все три опциональны (`None` по умолчанию), без валидации на взаимоисключение (можно задать несколько полей одновременно)
- **`EmbedRequest.messages`** — обязательный список, может быть пустым (пустой запрос приведёт к ошибке на стороне сервиса)
- **`MessageEmbedding.embedding`** — список float произвольной длины (не зафиксирована размерность вектора; фактически 2048-dim согласно настройкам Qdrant)
- **`EmbedSuccessResponse.messages`** — длина списка должна соответствовать длине `EmbedRequest.messages`
- **Отсутствует валидация order**: `message_id` в ответе должен соответствовать порядку сообщений в запросе, но это не проверяется на уровне Pydantic

### Граничные случаи
- Пустой `EmbedRequest.messages` → запрос отправляется, ошибка на стороне сервиса эмбеддингов (HTTP 4xx/5xx → `response.raise_for_status()`)
- Несоответствие длины ответа запросу → не детектируется Pydantic, клиент берёт `messages[0]`
- Отсутствие поля `type` → ошибка валидации Pydantic (поле обязательно)
- Невалидный JSON в ответе → `ValueError` при `model_validate()`

## Examples

### Валидный запрос на текстовый эмбеддинг
```python
message = Message(type="text", text="Привет, мир!")
request = EmbedRequest(messages=[message])
# → {"messages": [{"type": "text", "text": "Привет, мир!", "image": null, "image_url": null}]}
```

### Валидный запрос на эмбеддинг изображения (base64)
```python
message = Message(type="image", image="data:image/jpeg;base64,/9j/4AAQ...")
request = EmbedRequest(messages=[message])
```

### Валидный запрос на смешанный эмбеддинг (изображение + текст)
```python
message = Message(type="image/text", text="Диаграмма продаж", image="data:image/jpeg;base64,...")
request = EmbedRequest(messages=[message])
```

### Валидный ответ сервиса
```python
response_json = {
    "messages": [
        {"message_id": 0, "embedding": [0.123, -0.456, 0.789, ...]}
    ]
}
parsed = EmbedSuccessResponse.model_validate(response_json)
# → EmbedSuccessResponse(messages=[MessageEmbedding(message_id=0, embedding=[0.123, -0.456, ...])])
```

### Невалидный Message (отсутствует type)
```python
# Вызовет ValidationError:
Message(text="Привет")
# → type: Field required
```

### Невалидный EmbedRequest (messages не список)
```python
# Вызовет ValidationError:
EmbedRequest(messages="not_a_list")
```

## Exceptions
- **Отклонение от Constitution P6 (размещение моделей)**: Модели расположены в `app/src/schemas/embeddings.py`, а не в `dtype/` как предписано структурой N2. Каталог `dtype/` отсутствует в сервисе `app`.
- **Отклонение от Constitution P5 (синхронный HTTP)**: `EmbeddingClient` использует синхронную библиотеку `requests` вместо `httpx`/`aiohttp`, методы клиента синхронные, вызываются из `async def` эндпоинтов. Это блокирует event loop.
- **Неиспользуемая модель `EmbedResponse`**: Модель `EmbedResponse` декларирована в файле, но не импортируется ни одним модулем (ни клиентом, ни API). Фактически все методы клиента используют `EmbedSuccessResponse`.
- **Модель `Message` конфликтует по имени с `reranker.Message`**: Два разных pydantic-класса с именем `Message` существуют в `schemas/embeddings.py` и `schemas/reranker.py` с разной структурой полей.
