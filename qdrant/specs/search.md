# Feature: Vector Search

## Motivation
Поиск ближайших векторов в коллекции Qdrant по заданному query-вектору. FastAPI-обёртка над `qdrant_client.search()`, предоставляющая HTTP-интерфейс для векторного поиска. Используется сервисом `app` (и потенциально другими потребителями) для семантического поиска чанков документа.

## Behaviour

### Input

**GET /search** — query-параметр.

| Параметр | Тип | Часть | По умолчанию | Описание |
|----------|-----|-------|-------------|----------|
| `query_vector` | `str` | Query | — (обязательный) | Comma-separated floats, представляющие вектор запроса. Конвертируется в `list[float]`. |

Код валидирует:
- Формат: `[float(x) for x in query_vector.split(',')]`. Некорректный формат → `400` (ValueError)
- Размерность: ровно 4 элемента. Иная размерность → `400`

### Processing

Пошаговый алгоритм (`api.py:search_vectors`):

1. **Парсинг query-вектора** — строка `query_vector` разбивается по запятой, каждый элемент конвертируется в `float`.
   - При `ValueError` → `400 Bad Request`, `"Invalid vector format. Use comma-separated floats."`
2. **Валидация размерности** — `len(vector) != 4` → `400 Bad Request`, `"Vector size must be 4."`
3. **Поиск в Qdrant** — вызов `client.search()`:
   - `collection_name=COLLECTION_NAME` → `"my_collection"`
   - `query_vector=vector` → распарсенный вектор
   - `limit=5` → жёстко закодировано, всегда top-5
4. **Форматирование результата** — каждый `ScoredPoint` из ответа Qdrant преобразуется в dict:
   ```python
   {"id": res.id, "score": res.score, "payload": res.payload}
   ```
5. **Возврат ответа**:
   ```json
   {"results": [{"id": ..., "score": ..., "payload": ...}, ...]}
   ```

### Output

**200 OK** (успешный поиск):
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

**Коды ответа**:

| Код | Условие | Тело ответа |
|-----|---------|-------------|
| `200` | Успешный поиск (включая пустой результат — 0 hits) | `{"results": [...]}` |
| `400` | Неверный формат вектора (`ValueError` при `float()`) | `{"detail": "Invalid vector format. Use comma-separated floats."}` |
| `400` | Размерность вектора не равна 4 | `{"detail": "Vector size must be 4."}` |
| `422` | Отсутствует параметр `query_vector` | Стандартный FastAPI Unprocessable Entity |
| `500` | Ошибка Qdrant (нет соединения, ошибка поиска) | `{"detail": "<сообщение исключения>"}` |
| `500` | Прочие исключения | `{"detail": "<str(e)>"}` |

**Граничные случаи**:
- **Пустая коллекция** (нет точек): `200`, `results: []`
- **Нет релевантных результатов** (все score близки к 0): `200`, возвращается до 5 точек с низким score
- **Большое значение float**: `float("1e308")` — парсится корректно
- **Пробелы вокруг запятых**: `"0.1, 0.2 ,0.3,0.4"` — **упадёт с ValueError**, потому что `float(" 0.2 ")` даст ошибку. Стриппинг не применяется.

## API Contract

```openapi
get: /search
summary: Поиск ближайших векторов в коллекции Qdrant
parameters:
  - name: query_vector
    in: query
    required: true
    schema:
      type: string
    description: Comma-separated floats — вектор запроса. Пример: "0.1,0.2,0.3,0.4"
    example: "0.1,0.2,0.3,0.4"
responses:
  '200':
    description: Результаты поиска (до 5 ближайших точек)
    content:
      application/json:
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  id:
                    oneOf:
                      - type: integer
                      - type: string
                  score:
                    type: number
                    format: float
                  payload:
                    type: object
                    additionalProperties: true
  '400':
    description: Неверный формат или размерность вектора
  '500':
    description: Ошибка Qdrant или внутренняя ошибка сервера
```

## Data Flow

```
Client
  │  GET /search?query_vector=0.1,0.2,0.3,0.4
  ▼
api.py: search_vectors()
  ├─ 1. Парсинг строки в list[float]
  ├─ 2. Валидация: len(vector) == 4
  ├─ 3. QdrantClient.search(
  │       collection_name="my_collection",
  │       query_vector=vector,
  │       limit=5
  │   )
  │   └─ Qdrant Server (REST/gRPC) → поиск ANN (HNSW)
  ▼
  ├─ 4. Преобразование ScoredPoint[] → list[dict]
  ▼
  {"results": [...]}
```

Хранилища, затронутые в процессе:
- **Qdrant Server** (`my_collection`): чтение — поиск векторов (ANN), payload загружается из хранилища Qdrant

## LLM Interactions

Отсутствуют. Сервис `qdrant` не взаимодействует с LLM — Constitution P7 не применим.

## LLM Model Requirements

Не применимо.

## Error Handling

| Failure mode | Обработка | Код | Детали |
|-------------|-----------|-----|--------|
| `query_vector` содержит нечисловые символы | `float()` выбрасывает `ValueError` → `400` | `400` | `"Invalid vector format. Use comma-separated floats."` |
| `query_vector` после парсинга имеет размерность ≠ 4 | Ручная проверка `len(vector) != 4` → `400` | `400` | `"Vector size must be 4."` |
| `query_vector` не передан в запросе | FastAPI-валидация параметра | `422` | Стандартный ответ FastAPI |
| Qdrant Server недоступен | `QdrantClient.search()` выбрасывает исключение → `500` | `500` | `detail=str(e)` |
| Коллекция не существует | `QdrantClient.search()` выбрасывает исключение → `500` | `500` | `detail=str(e)` |
| Пробелы вокруг запятых в векторе | `float(" 0.2 ")` → `ValueError` → `400` | `400` | `"Invalid vector format..."` (стриппинг не применяется) |

## Testing

### Unit-тесты (с моком QdrantClient)

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 1 | Валидный запрос: `query_vector=0.1,0.2,0.3,0.4` | `200`, `results` — список из мокнутых точек, `client.search()` вызван с `limit=5` |
| 2 | Невалидный формат: `query_vector=0.1,abc,0.3,0.4` | `400`, `"Invalid vector format. Use comma-separated floats."` |
| 3 | Неверная размерность: `query_vector=0.1,0.2,0.3` (3 элемента) | `400`, `"Vector size must be 4."` |
| 4 | Неверная размерность: `query_vector=0.1,0.2,0.3,0.4,0.5` (5 элементов) | `400`, `"Vector size must be 4."` |
| 5 | Отсутствует параметр `query_vector` | `422 Unprocessable Entity` |
| 6 | Пустая коллекция (Qdrant возвращает `[]`) | `200`, `{"results": []}` |
| 7 | Пробелы в векторе: `query_vector=0.1, 0.2, 0.3, 0.4` | `400`, `"Invalid vector format..."` (баг — ожидалось бы `200` с триммингом) |
| 8 | Qdrant возвращает точки с разными типами id (`int` и `str`) | `200`, результаты корректно сериализованы |

### Интеграционные тесты

| # | Сценарий | Ожидаемый результат |
|---|----------|-------------------|
| 9 | Предварительно вставлены 3 точки, поиск по одному из векторов | `200`, 3 результата, наивысший score у ближайшей точки |
| 10 | Qdrant Server недоступен | `500`, `detail` содержит сообщение о connection error |
| 11 | Запрос с вектором, размерность которого не совпадает с коллекцией | `400`, проверка до вызова `client.search()` |

## Dependencies

| Зависимость | Тип | Назначение |
|------------|-----|-----------|
| `qdrant_client.QdrantClient` | Библиотека | Python-драйвер для взаимодействия с Qdrant Server (REST/gRPC) |
| `qdrant_client.models` | Библиотека | `ScoredPoint` — тип результата поиска |
| Qdrant Server | Внешний сервис | Собственно векторная БД (порт 6333 REST, 6334 gRPC) |
| `config.settings.Settings` | Модуль | Конфигурация host/port/api_key из `.env` |

## Exceptions

1. **Отсутствие Pydantic-моделей** — эндпоинт возвращает сырой `dict`, не использует `response_model`. Нарушение Constitution P6. **Влияние**: нет валидации ответа, нет автогенерации OpenAPI-схемы для выходных данных, клиенты не имеют формального контракта.

2. **Модели определены inline в `api.py`** — структуры данных формируются непосредственно в обработчике, а не вынесены в `dtype/`. Нарушение Constitution N2 (структура модулей). **Причина**: переходный период.

3. **Жёстко закодированная размерность = 4** — размерность вектора захардкожена в коде (`len(vector) != 4`), не вынесена в `config/settings.py`. Не соответствует интеграционному контракту I3 (Constitution), где указана размерность 2048. **Причина**: код использует демонстрационные значения, не соответствующие реальному использованию.

4. **Жёстко закодированное имя коллекции = `"my_collection"`** — коллекция захардкожена в `COLLECTION_NAME`, не вынесена в конфигурацию. Не соответствует I3 (`"documents"`).

5. **Жёстко закодированный `limit=5`** — параметр не конфигурируем, клиент не может запросить другое количество результатов.

6. **Нет валидации `file_hash`/`region_id` в payload** — эндпоинт поиска не проверяет содержимое payload возвращаемых точек, но и не требует валидации, так как только читает данные. Однако отсутствие гарантии, что точки содержат `file_hash` и `region_id` (согласно P2/P3), означает, что потребитель должен самостоятельно проверять payload.

7. **Отсутствует `manager.py`** — бизнес-логика поиска (вызов `client.search`) находится непосредственно в обработчике эндпоинта. Тривиальность операции делает вынос в менеджер избыточным на текущем этапе, но отклоняется от Constitution N2.

8. **Пробелы в query_vector вызывают ошибку** — `float(" 0.2 ")` выбрасывает `ValueError`, хотя интуитивно ожидается, что пробелы допустимы. Клиент обязан передавать вектор без пробелов. Это не задокументированное ограничение.
