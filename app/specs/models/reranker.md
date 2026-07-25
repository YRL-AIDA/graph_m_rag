# Data Model: Reranker API Models

## Purpose
Модели данных для взаимодействия с внешним сервисом реранкера через HTTP-клиент `reranker_client.py`. Используются для отправки запросов на переранжирование результатов поиска по релевантности пользовательскому запросу и получения ранжированного списка с оценками.

## Schema

```python
import base64
from typing import List, Union
from pydantic import BaseModel, ConfigDict, model_validator


class Message(BaseModel):
    """Элемент контента для реранжирования. Отличается от embeddings.Message
       — отсутствует model_config и поле type опционально."""
    type: Union[str, None] = None         # Тип контента: "text", "image", "image_url" (опционально)
    text: Union[str, None] = None         # Текстовое содержимое
    image: Union[str, None] = None        # Base64-изображение
    image_url: Union[str, None] = None    # URL изображения

    def add_text_content(self, text: str) -> 'Message':
        """Установить тип 'text' и текстовое содержимое."""
        self.type = 'text'
        self.text = text
        return self

    def set_type(self, type: str) -> 'Message':
        """Установить тип сообщения."""
        self.type = type
        return self

    def add_img_content(self, source: str = 'image_url',
                        path_to_img: str = None,
                        url: str = None) -> 'Message':
        """Добавить изображение из файла или URL. При path_to_img читает файл и
           кодирует в base64 JPEG."""
        match source:
            case 'image_url':
                if path_to_img is not None:
                    with open(path_to_img, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode()
                    self.type = 'image'
                    self.image = f"data:image/jpeg;base64,{base64_image}"
                elif url is not None:
                    self.type = 'image_url',
                    self.image_url = url
        return self

    def add_img_content_base64(self, base64_image: str = None) -> 'Message':
        """Добавить изображение из готовой base64-строки."""
        self.type = 'image'
        self.image = f"data:image/jpeg;base64,{base64_image}"
        return self


class RerankRequest(BaseModel):
    """Запрос на переранжирование списка сообщений."""
    instruction: str                          # Инструкция для модели реранкера
    query: dict[str, str]                     # Поисковый запрос (ключ — тип, значение — текст запроса)
    fps: float = 1.0                          # Frames per second (для видео-контента, обычно 1.0)
    messages: List[Message]                   # Список элементов для переранжирования


class ResponseMessage(BaseModel):
    """Результат переранжирования одного сообщения."""
    message_id: int                           # Индекс сообщения в исходном запросе (0-based)
    score: float                              # Оценка релевантности (чем выше, тем релевантнее)


class RerankResponse(BaseModel):
    """Ответ сервиса реранкера — отсортированный список результатов."""
    messages: List[ResponseMessage]           # Список результатов, отсортированный по убыванию score
```

## Storage
- **Хранение**: in-memory / HTTP transport only (сериализация запроса → JSON body → HTTP POST → десериализация ответа)
- Модели не персистятся. Результаты реранкинга используются непосредственно в `/ask-document` для переупорядочивания выдачи.

## Relationships
- **`Message`** → используется в `RerankerClient` (`app/src/reranker_client.py`) и в `api.py` для подготовки кандидатов на реранкинг
- **`RerankRequest`** → сериализуется (`model_dump()`) в JSON-body HTTP POST на `{base_url}/rerank`
- **`RerankResponse`** → десериализуется из JSON-ответа сервиса реранкера методом `model_validate()`
- **`ResponseMessage`** → вложенная модель в `RerankResponse.messages`
- **Конфигурация**: `RerankerClient` получает `base_url` и `timeout` из `RerankerSettings` (`app/config/settings.py`) через глобальный экземпляр `settings`
- **Связь с API**: `api.py` импортирует `Message` из `schemas.reranker` для подготовки списка сообщений (чанков) к реранкингу; результаты реранкера используются для переупорядочивания выдачи в эндпоинте `/ask-document`
- **Отличие от `embeddings.Message`**: несмотря на одинаковое имя класса, это **разные модели** — у reranker.Message `type` опционально, нет `model_config(extra='ignore')`, есть методы-хелперы для наполнения полей

## Constraints
- **`Message.type`** — опциональное поле (`Union[str, None]`), без валидации допустимых значений; методы-хелперы устанавливают значения `"text"`, `"image"`, `"image_url"` (в последнем случае — с запятой из-за бага в `add_img_content`: `self.type = 'image_url',`)
- **`Message.add_img_content`** — содержит баг: в ветке URL-источника присваивает кортеж `self.type = 'image_url',` (trailing comma), что приводит к `type = ('image_url',)` вместо строки
- **`RerankRequest.query`** — `dict[str, str]`, обычно содержит `{'text': 'текст вопроса'}`; тип ключа не валидируется
- **`RerankRequest.instruction`** — обязательное поле, значение по умолчанию в клиенте: `"Retrieve images or text relevant to the user's query."`
- **`RerankRequest.fps`** — всегда `1.0` в текущем использовании, предназначено для видео-реранкинга
- **`RerankResponse.messages`** — список должен быть отсортирован по убыванию `score` сервисом реранкера (не валидируется Pydantic)
- **Модель не имеет `model_config`**: в отличие от `embeddings.Message`, здесь не задан режим `extra='ignore'` или `frozen`

### Граничные случаи
- Пустой `RerankRequest.messages` → запрос отправляется, результат зависит от сервиса реранкера
- `RerankRequest.query` с ключом, отличным от `"text"` → поведение зависит от сервиса реранкера
- `message_id` в ответе не соответствует индексам из запроса → клиент не проверяет соответствие, потенциально неверная привязка скоров
- Файл изображения не найден в `add_img_content(path_to_img=...)` → `FileNotFoundError` (не обрабатывается)
- Ответ сервиса с невалидным JSON → `ValueError` при `model_validate()`

## Examples

### Валидный запрос на реранкинг
```python
messages = [
    Message(type="text", text="Пассаж 1: Экономика выросла на 3%"),
    Message(type="text", text="Пассаж 2: Погода была солнечной"),
]
request = RerankRequest(
    instruction="Retrieve images or text relevant to the user's query.",
    query={"text": "рост экономики"},
    messages=messages
)
# → {"instruction": "...", "query": {"text": "рост экономики"}, "fps": 1.0, "messages": [...]}
```

### Валидный ответ сервиса
```python
response_json = {
    "messages": [
        {"message_id": 0, "score": 0.95},
        {"message_id": 1, "score": 0.12}
    ]
}
parsed = RerankResponse.model_validate(response_json)
# → RerankResponse(messages=[ResponseMessage(message_id=0, score=0.95), ResponseMessage(message_id=1, score=0.12)])
```

### Использование методов-хелперов для Message
```python
msg = Message()
msg.add_text_content("Текст документа")
# → Message(type="text", text="Текст документа", image=None, image_url=None)

msg2 = Message()
msg2.add_img_content_base64("iVBORw0KGgo...")
# → Message(type="image", image="data:image/jpeg;base64,iVBORw0KGgo...", text=None, image_url=None)
```

### Невалидный RerankRequest (отсутствует instruction)
```python
# Вызовет ValidationError:
RerankRequest(query={"text": "запрос"}, messages=[])
# → instruction: Field required
```

### Баг: add_img_content с URL устанавливает кортеж
```python
msg = Message()
msg.add_img_content(source="image_url", url="http://example.com/img.jpg")
# msg.type → ('image_url',)  # кортеж вместо строки!
```

## Exceptions
- **Отклонение от Constitution P6 (размещение моделей)**: Модели расположены в `app/src/schemas/reranker.py`, а не в `dtype/` как предписано структурой N2. Каталог `dtype/` отсутствует в сервисе `app`.
- **Отклонение от Constitution P5 (синхронный HTTP)**: `RerankerClient` использует синхронную библиотеку `requests`, методы клиента синхронные, вызываются из `async def` эндпоинтов. Блокирует event loop.
- **Баг в `add_img_content`**: присвоение `self.type = 'image_url',` (с trailing comma) приводит к кортежу вместо строки. Дефект в байт-коде Python — интерпретируется как tuple packing.
- **Отсутствие model_config**: в отличие от `embeddings.Message`, reranker-модели не защищены от extra fields при десериализации.
- **Дублирование имени `Message`**: в том же сервисе существует `embeddings.Message` с другой структурой. Может приводить к путанице при импорте, хотя импорты всегда явные.
