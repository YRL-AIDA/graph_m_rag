# Data Model: Qdrant Search/Upsert Models

## Purpose
Модели данных для операций поиска (`GET /search`) и вставки (`POST /upsert_points`) векторов в Qdrant. Используются на границе API-слоя `qdrant/api.py` и передаются между клиентом и Qdrant-сервером.

## Schema

В текущей реализации **отсутствуют Pydantic-модели** — используются сырые Python-типы (`list[dict]`, `dict`) и структуры из библиотеки `qdrant_client`.

### Входные данные (Upsert)

```python
# POST /upsert_points — body: list[dict]
# Эндпоинт принимает list[dict] напрямую, без Pydantic-валидации
# Каждый dict преобразуется в models.PointStruct из qdrant_client

from qdrant_client.models import PointStruct

# PointStruct (из библиотеки qdrant_client):
#   id: Union[int, str]       # Уникальный идентификатор точки
#   vector: List[float]       # Вектор (размерность задаётся при создании коллекции)
#   payload: Dict[str, Any]   # Произвольные метаданные
```

### Выходные данные (Search)

```python
# GET /search — возвращает dict вида:
# {
#     "results": [
#         {
#             "id": Union[int, str],
#             "score": float,
#             "payload": Dict[str, Any]
#         },
#         ...
#     ]
# }

# ScoredPoint (из библиотеки qdrant_client) — неявно используется:
#   id: Union[int, str]
#   version: int
#   score: float
#   payload: Dict[str, Any] | None
#   vector: List[float] | None
```

### Структура точки в коллекции (логическая схема)

```python
# Ожидаемая структура payload точки в коллекции "my_collection":
{
    "id": Union[int, str],           # ID точки (может быть region_id)
    "vector": List[float],           # Эмбеддинг (размерность 4 в примере, реально 2048)
    "payload": {
        "file_hash": str,            # P2: MD5-хеш PDF
        "region_id": str,            # P3: композитный ID региона
        "element_type": str,         # Тип элемента: "text", "image", "table"
        "original_element": dict,    # Исходный элемент документа
        # ... произвольные метаданные
    }
}
```

## Storage
- **Qdrant payload** — метаданные точек хранятся как JSON в коллекции Qdrant (`my_collection`)
- **in-memory** — результаты поиска и ответы upsert передаются через HTTP как JSON, не персистируются в сервисе

## Relationships
- `POST /upsert_points` — создаёт/обновляет точки в коллекции Qdrant. Модель данных точки должна соответствовать контракту I3 (Constitution): коллекция `documents`, payload с `file_hash`, `region_id`, `element_type`, `original_element`.
- `GET /search` — возвращает результат поиска `qdrant_client.ScoredPoint[]`, преобразованный в `dict`.
- Внешний контракт: `app` → Qdrant (Constitution I3) — `app` обращается к Qdrant напрямую через драйвер, но также может использовать `qdrant/api.py`.
- `PointStruct` из `qdrant_client.models` — зависимость от внутреннего типа библиотеки, а не от собственной модели данных.

## Constraints

| Поле | Ограничение |
|------|-------------|
| `PointStruct.id` | `Union[int, str]`, уникален в пределах коллекции |
| `PointStruct.vector` | Длина должна соответствовать размерности коллекции (4 в примере, 2048 в реальном использовании) |
| `PointStruct.payload` | Опциональный `dict`, должен содержать `file_hash` и `region_id` согласно P2/P3 |
| `query_vector` (search) | Comma-separated floats, строго 4 элемента в текущей реализации |
| `limit` (search) | Жёстко задан как 5 в коде |
| `SearchResult.score` | `float`, чем выше — тем релевантнее |
| `UpsertResponse.status` | Всегда `"upserted"` |
| `UpsertResponse.count` | `int`, количество вставленных/обновлённых точек |

## Examples

### Upsert — валидный запрос
```json
[
    {
        "id": "abc123_region_0",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "payload": {
            "file_hash": "d41d8cd98f00b204e9800998ecf8427e",
            "region_id": "region_0",
            "element_type": "text",
            "original_element": {"text": "Пример текста", "page": 1}
        }
    },
    {
        "id": "abc123_region_1",
        "vector": [0.5, 0.6, 0.7, 0.8],
        "payload": {
            "file_hash": "d41d8cd98f00b204e9800998ecf8427e",
            "region_id": "region_1",
            "element_type": "image"
        }
    }
]
```

### Upsert — ответ
```json
{
    "status": "upserted",
    "count": 2
}
```

### Search — запрос
```
GET /search?query_vector=0.1,0.2,0.3,0.4
```

### Search — ответ
```json
{
    "results": [
        {
            "id": "abc123_region_0",
            "score": 0.98,
            "payload": {
                "file_hash": "d41d8cd98f00b204e9800998ecf8427e",
                "region_id": "region_0",
                "element_type": "text",
                "original_element": {"text": "Пример текста", "page": 1}
            }
        },
        {
            "id": "xyz789_region_5",
            "score": 0.76,
            "payload": {
                "file_hash": "e99a18c428cb38d5f260853678922e03",
                "region_id": "region_5",
                "element_type": "text"
            }
        }
    ]
}
```

### Upsert — невалидный запрос (отсутствует вектор)
```json
[
    {
        "id": "abc123_region_0",
        "payload": {"file_hash": "d41d8cd9"}
    }
]
```
Ответ: `500 Internal Server Error` — `KeyError: 'vector'` при попытке доступа `p['vector']`.

### Upsert — невалидный запрос (id не уникален в пределах одного запроса)
```json
[
    {"id": "same_id", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {}},
    {"id": "same_id", "vector": [0.5, 0.6, 0.7, 0.8], "payload": {}}
]
```
Qdrant upsert обработает обе точки, последняя перезапишет первую (поведение upsert).

### Search — невалидный запрос (неверный формат вектора)
```
GET /search?query_vector=0.1,abc,0.3,0.4
```
Ответ: `400 Bad Request` — `"Invalid vector format. Use comma-separated floats."`

### Search — невалидный запрос (неверная размерность)
```
GET /search?query_vector=0.1,0.2,0.3
```
Ответ: `400 Bad Request` — `"Vector size must be 4."`

## Exceptions
- **Модели не определены как Pydantic** — нарушение P6 (Constitution). Эндпоинты `search_vectors` и `upsert_points` используют сырые `list[dict]` и `dict` вместо Pydantic-моделей. Отсутствие строгой типизации означает: нет валидации на уровне FastAPI/Pydantic, ошибки валидации всплывают только во время выполнения внутри `qdrant_client`, нет автодокументации через OpenAPI (FastAPI не может сгенерировать корректную схему для `list[dict]`).
- **Модели определены в `api.py` вместо `dtype/`** — нарушение N2 (Constitution). Все модели данных должны находиться в `dtype/` сервиса. Логика формирования структур данных (преобразование `list[dict]` → `PointStruct`, преобразование `ScoredPoint` → `dict`) находится непосредственно в обработчиках эндпоинтов.
- **`PointStruct` из `qdrant_client` — внешняя зависимость как контракт** — внутренние типы библиотеки `qdrant_client` (`PointStruct`, `ScoredPoint`) используются напрямую как контракт API, вместо собственных Pydantic-моделей, которые бы абстрагировали клиента от деталей реализации библиотеки.
- **Жёстко закодированные параметры** — размерность вектора (4), название коллекции (`"my_collection"`), `limit=5` захардкожены в коде, а не вынесены в конфигурацию (`config/settings.py`).
- **Payload имеет тип `Dict[str, Any]`** — даже внутри `PointStruct` payload не типизирован, что позволяет передавать произвольные данные без проверки на наличие обязательных полей (`file_hash`, `region_id` согласно P2/P3).
- **Отсутствует модель для ответа upsert** — `{"status": "upserted", "count": len(points)}` формируется как сырой dict, без Pydantic-модели.
- **Отсутствует модель для ответа search** — `{"results": [...]}` также сырой dict, без схемы.

## Рекомендации по приведению к Constitution
Для соответствия P6 и N2 рекомендуется:
1. Создать `qdrant/dtype/` с Pydantic-моделями: `UpsertPoint`, `UpsertRequest`, `UpsertResponse`, `SearchResult`, `SearchResponse`
2. Вынести размерность вектора и имя коллекции в `config/settings.py`
3. Использовать `response_model` в декораторах эндпоинтов для автогенерации OpenAPI-схемы
