# Feature: Upsert Points

## Motivation
Массовая вставка или обновление (upsert) векторных точек в коллекцию Qdrant. FastAPI-обёртка над `qdrant_client.upsert()`, предоставляющая HTTP-интерфейс для загрузки эмбеддингов с метаданными. Используется для индексации чанков документа после извлечения эмбеддингов.

## Behaviour

### Input

**POST /upsert_points** — `application/json`.

Тело запроса: `list[dict]`, где каждый элемент — точка со следующими полями:

| Поле | Тип | Обязательное | Описание |
|------|-----|-------------|----------|
| `id` | `int` или `str` | Да | Уникальный идентификатор точки в коллекции |
| `vector` | `list[float]` | Да | Вектор-эмбеддинг. Размерность должна соответствовать размерности коллекции (4 в текущей реализации) |
| `payload` | `dict` | Нет (по умолчанию `{}`) | Произвольные метаданные точки. По Constitution P2/P3 должен содержать `file_hash` и `region_id` |

**Валидация**:
- На уровне FastAPI/Pydantic: **отсутствует**. Параметр объявлен как `points: list[dict]` — FastAPI не генерирует схему для `list[dict]`, валидация структуры не производится до момента обращения к полям.
- `p['id']`, `p['vector']` — обязательный доступ в коде. Отсутствие поля → `KeyError` → `500 Internal Server Error`.
- `p.get('payload', {})` — опционально, по умолчанию `{}`.

### Processing

Пошаговый алгоритм (`api.py:upsert_points`):

1. **Преобразование входных данных** — каждый dict из `points` преобразуется в `models.PointStruct`:
   ```python
   point = models.PointStruct(
       id=p['id'],
       vector=p['vector'],
       payload=p.get('payload', {})
   )
   ```
   При отсутствии ключа `'id'` или `'vector'` → `KeyError` → `500`.

2. **Upsert в Qdrant** — вызов `client.upsert()`:
   - `collection_name=COLLECTION_NAME` → `"my_collection"`
   - `points=points_structs` → список `PointStruct`
   - Операция upsert: если точка с таким `id` существует — обновляется; если нет — создаётся.

3. **Формирование ответа**:
   ```python
   {"status": "upserted", "count": len(points)}
   ```
   Ответ — сырой dict, без Pydantic-модели.

### Output

**200 OK** (успешный upsert):
```json
{
    "status": "upserted",
    "count": 2
}
```

**Коды ответа**:

| Код | Условие | Тело ответа |
|-----|---------|-------------|
| `200` | Успешный upsert всех точек | `{"status": "upserted", "count": N}` |
| `500` | `KeyError` при отсутствии `id` или `vector` в точке | `{"detail": "'id'"}` или `{"detail": "'vector'"}` |
| `500` | Размерность вектора не совпадает с коллекцией (ошибка от Qdrant) | `{"detail": "<сообщение от Qdrant>"}` |
| `500` | Qdrant Server недоступен | `{"detail": "<сообщение исключения>"}` |
| `500` | Коллекция не существует | `{"detail": "<сообщение от Qdrant>"}` (lifespan создаёт коллекцию при старте, но теоретически возможна race condition) |

**Поведение при upsert**:
- **Новая точка** (id не существует в коллекции) → создаётся
- **Существующая точка** (id уже есть) → вектор и payload полностью заменяются
- **Дубликаты id в одном запросе** → обрабатываются последовательно, последняя точка перезаписывает предыдущую (idempotent upsert)

**Граничные случаи**:
- **Пустой список**: `POST /upsert_points` с `[]` → `200`, `{"status": "upserted", "count": 0}`
- **Большой payload**: Qdrant имеет ограничения на размер payload (по умолчанию нет жёсткого лимита, но очень большой payload может вызвать ошибку при записи)
- **Вектор неверной размерности**: Если размерность вектора ≠ 4, Qdrant отклонит запрос → `500` с ошибкой размерности

## API Contract

```openapi
post: /upsert_points
summary: Массовая вставка/обновление (upsert) векторных точек в коллекцию Qdrant
requestBody:
  required: true
  content:
    application/json:
      schema:
        type: array
        items:
          type: object
          properties:
            id:
              oneOf:
                - type: integer
                - type: string
              description: Уникальный идентификатор точки
            vector:
              type: array
              items:
                type: number
                format: float
              description: Вектор-эмбеддинг (размерность должна соответствовать коллекции)
            payload:
              type: object
              additionalProperties: true
              description: Произвольные метаданные точки (должен содержать file_hash и region_id согласно P2/P3)
          required:
            - id
            - vector
responses:
  '200':
    description: Точки успешно вставлены/обновлены
    content:
      application/json:
        schema:
          type: object
          properties:
            status:
              type: string
              enum: [upserted]
            count:
              type: integer
              description: Количество обработанных точек
  '500':
    description: Ошибка валидации, Qdrant или внутренняя ошибка сервера
```

**Примечание**: в текущей реализации FastAPI **не генерирует** корректную OpenAPI-схему для этого эндпоинта из-за использования `list[dict]` вместо Pydantic-модели. Приведённый OpenAPI-фрагмент описывает семантический контракт, но не будет совпадать с автосгенерированной схемой.

## Data Flow

```
Client
  │  POST /upsert_points
  │  body: [{"id": ..., "vector": [...], "payload": {...}}, ...]
  ▼
api.py: upsert_points()
  ├─ 1. Итерация по points:
  │     ├─ p['id'] (KeyError → 500)
  │     ├─ p['vector'] (KeyError → 500)
  │     └─ p.get('payload', {})
  ├─ 2. Преобразование в models.PointStruct
  ├─ 3. QdrantClient.upsert(
  │       collection_name="my_collection",
  │       points=points_structs
  │   )
  │   └─ Qdrant Server (REST/gRPC) → запись векторов + payload
  ▼
  {"status": "upserted", "count": N}
```

Хранилища, затронутые в процессе:
- **Qdrant Server** (`my_collection`): запись — upsert точек (векторы + payload в on-disk хранилище Qdrant)

## LLM Interactions

Отсутствуют. Сервис `qdrant` не взаимодействует с LLM — Constitution P7 не применим.

## LLM Model Requirements

Не применимо.

## Error Handling

| Failure mode | Обработка | Код | Детали |
|-------------|-----------|-----|--------|
| В точке отсутствует `id` | `p['id']` → `KeyError` → `500` | `500` | `detail="'id'"` — неинформативно, клиент не поймёт какая именно точка проблемная |
| В точке отсутствует `vector` | `p['vector']` → `KeyError` → `500` | `500` | `detail="'vector'"` |
| Размерность вектора ≠ 4 | Qdrant отклоняет upsert → исключение → `500` | `500` | `detail=str(e)` — сообщение от Qdrant |
| Qdrant Server недоступен | `QdrantClient.upsert()` выбрасывает исключение → `500` | `500` | `detail=str(e)` |
| Коллекция не существует | `QdrantClient.upsert()` выбрасывает исключение → `500` | `500` | `detail=str(e)` |
| Тип `id` не int и не str (например, float 1.0) | `PointStruct` принимает `Union[int, str]`, float может быть неявно принят или вызвать ошибку Qdrant | `500` или `200` | Зависит от поведения `qdrant_client` |
| Пустой список точек | `len(points) == 0` → `client.upsert(points=[])` → Qdrant обрабатывает без ошибок | `200` | `{"status": "upserted", "count": 0}` |
| Дубликаты id в одном запросе | Upsert идемпотентен — последняя точка перезаписывает предыдущую с тем же id | `200` | `count` = общее количество точек в запросе (включая дубликаты) |

## Testing

### Unit-тесты (с моком QdrantClient)

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 1 | Валидный запрос с 2 точками | `200`, `{"status": "upserted", "count": 2}`, `client.upsert()` вызван с правильными `points` |
| 2 | Запрос с 1 точкой без payload | `200`, `{"status": "upserted", "count": 1}`, payload = `{}` |
| 3 | Запрос с точкой, у которой `id` — int | `200`, PointStruct создан с `id=int_value` |
| 4 | Запрос с точкой, у которой `id` — str | `200`, PointStruct создан с `id=str_value` |
| 5 | Запрос с точкой без поля `id` | `500`, `KeyError: 'id'` |
| 6 | Запрос с точкой без поля `vector` | `500`, `KeyError: 'vector'` |
| 7 | Пустой список: `[]` | `200`, `{"status": "upserted", "count": 0}` |
| 8 | Qdrant выбрасывает исключение | `500`, `detail=str(e)` |

### Интеграционные тесты

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 9 | Upsert 3 точек с разными id, затем поиск по одному из векторов | Поиск возвращает upsert-нутую точку с максимальным score |
| 10 | Upsert точки с существующим id (обновление) | Поиск возвращает обновлённый вектор/payload |
| 11 | Upsert с вектором размерности ≠ 4 | `500`, ошибка от Qdrant |
| 12 | Qdrant Server недоступен | `500`, connection error |

### Граничные случаи

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 13 | Upsert большого количества точек (1000+) | `200`, `count=1000+` (проверка производительности) |
| 14 | Payload с бинарными данными (base64-строка) | `200`, данные сохранены как строка в payload |
| 15 | Вложенный payload с глубокой структурой | `200`, Qdrant сохраняет как JSON |

## Dependencies

| Зависимость | Тип | Назначение |
|------------|-----|-----------|
| `qdrant_client.QdrantClient` | Библиотека | Python-драйвер для взаимодействия с Qdrant Server |
| `qdrant_client.models.PointStruct` | Библиотека | Структура данных точки для upsert |
| Qdrant Server | Внешний сервис | Векторная БД (порт 6333 REST, 6334 gRPC) |
| `config.settings.Settings` | Модуль | Конфигурация host/port/api_key из `.env` |

## Exceptions

1. **Отсутствие Pydantic-моделей** — эндпоинт принимает `list[dict]` и возвращает сырой `dict`, не использует `response_model`. Нарушение Constitution P6 (`Pydantic — источник истины для моделей данных`). **Влияние**:
   - Нет валидации на уровне FastAPI — ошибки (`KeyError`) всплывают только во время выполнения
   - Нет автогенерации OpenAPI-схемы для тела запроса (FastAPI не может вывести схему из `list[dict]`)
   - Swagger UI показывает бессмысленную схему
   - Клиенты не имеют формального контракта (типы полей, обязательность)

2. **Модели определены inline в `api.py`** — нарушение Constitution N2 (структура модулей: модели должны быть в `dtype/`). **Причина**: переходный период.

3. **Жёстко закодированное имя коллекции = `"my_collection"`** — нарушение интеграционного контракта I3 (Constitution): ожидается коллекция `"documents"`. Имя коллекции не вынесено в `config/settings.py`.

4. **Жёстко закодированная размерность = 4** — размерность вектора неявно зафиксирована размерностью коллекции при её создании в lifespan (`size=4`), но не вынесена в конфигурацию. Не соответствует I3 (2048-dim).

5. **Нет валидации `file_hash`/`region_id` в payload** — согласно Constitution P2 (`Единый идентификатор документа — file_hash`) и P3 (`Сквозной идентификатор региона`), payload каждой точки должен содержать `file_hash` и `region_id`. Текущий код принимает любые payload без проверки обязательных полей. **Причина**: отсутствие Pydantic-модели для payload.

6. **Отсутствует `manager.py`** — бизнес-логика (вызов `client.upsert`) находится непосредственно в обработчике эндпоинта. Тривиальность операции делает вынос в менеджер избыточным на текущем этапе, но отклоняется от Constitution N2.

7. **Неинформативные ошибки валидации** — при отсутствии `id` или `vector` клиент получает `500` с `KeyError` вместо `400`/`422` с человекочитаемым описанием проблемы и указанием на конкретную точку. `500` некорректен, так как это ошибка клиента (невалидный запрос), а не сервера.

8. **`QDRANT_URL` формируется из `settings.host:settings.port`** (`0.0.0.0:8000`), что совпадает с адресом FastAPI-обёртки и не является портом Qdrant Server (по умолчанию 6333). Ожидается, что Qdrant и обёртка работают на одном хосте и порту, что не соответствует docker-образу (см. исключение в `SERVICE.md`).
