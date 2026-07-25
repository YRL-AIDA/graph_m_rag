# Data Model: Process/Status Response

## Purpose
Модели ответа API MinerU для эндпоинтов `/process` и `/status/{task_id}`. `ProcessResponse` возвращается синхронно при загрузке документа на обработку (POST /process). `StatusResponse` используется для асинхронного опроса статуса задачи (GET /status/{task_id}).

## Schema

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ProcessResponse(BaseModel):
    """Ответ на POST /process — результат обработки документа."""
    status: str                              # "completed" | "failed"
    message: str                             # Человекочитаемое сообщение
    results: Optional[Dict[str, Any]] = None # {"result": {...}} при успехе, None при ошибке


class StatusResponse(BaseModel):
    """Ответ на GET /status/{task_id} — статус асинхронной задачи."""
    task_id: str                             # UUID задачи
    status: str                              # "pending" | "processing" | "completed" | "failed"
    results: Optional[Dict[str, Any]] = None # Результат обработки при статусе "completed"
    error: Optional[str] = None              # Сообщение об ошибке при статусе "failed"
```

### Связанные модели (не Pydantic)

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessingConfig:
    """Конфигурация обработки документа (manager.py)."""
    backend: str = "pipeline"               # "pipeline" | "vlm"
    method: str = "auto"                    # "auto" | "txt" | "ocr"
    lang: str = "ru"                        # Язык документа
    formula_enable: bool = True             # Включить обработку формул
    table_enable: bool = True               # Включить обработку таблиц
    start_page_id: int = 0                  # Начальная страница (0-indexed)
    end_page_id: Optional[int] = None       # Конечная страница (None = до конца)
```

## Storage
in-memory / HTTP transport (JSON). Модели сериализуются в JSON при ответе FastAPI и десериализуются клиентом. Данные не персистируются между перезапусками сервиса — словарь `tasks` в `api.py` хранится в памяти процесса.

## Relationships
- `ProcessResponse` — используется в `mineru/api.py` для ответа на `POST /process` (response_model)
- `StatusResponse` — используется в `mineru/api.py` для ответа на `GET /status/{task_id}` (response_model)
- `ProcessingConfig` — используется в `mineru/manager.py` как входной параметр для `MinerUManager.process_pdf()`
- Внешний контракт: `app` → `mineru` (см. Constitution I2), `app` ожидает `ProcessResponse` / `StatusResponse` при вызове API mineru

## Constraints

| Поле | Ограничение |
|------|-------------|
| `ProcessResponse.status` | Только `"completed"` или `"failed"` |
| `ProcessResponse.results` | `None`, если `status == "failed"`; `{"result": {...}}`, если `status == "completed"` |
| `StatusResponse.task_id` | UUID в строковом представлении |
| `StatusResponse.status` | `"pending"`, `"processing"`, `"completed"`, `"failed"` |
| `StatusResponse.results` | `None` для всех статусов кроме `"completed"` |
| `StatusResponse.error` | `None` для всех статусов кроме `"failed"` |
| `ProcessingConfig.backend` | Только `"pipeline"` или `"vlm"` |
| `ProcessingConfig.method` | Только `"auto"`, `"txt"` или `"ocr"` |
| `ProcessingConfig.start_page_id` | `>= 0` |
| `ProcessingConfig.end_page_id` | `>= start_page_id` если задан |

## Examples

### ProcessResponse — успешная обработка
```json
{
    "status": "completed",
    "message": "Документ обработан",
    "results": {
        "result": {
            "success": true,
            "content": [
                {"type": "text", "text": "Глава 1. Введение", "page_idx": 0},
                {"type": "image", "image_path": "/tmp/abc_img_0.jpg", "page_idx": 0}
            ],
            "output_dir": "/tmp/mineru_task123_/"
        }
    }
}
```

### ProcessResponse — ошибка обработки
```json
{
    "status": "failed",
    "message": "Ошибка при обработке документа",
    "results": null
}
```

### StatusResponse — задача в обработке
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "results": null,
    "error": null
}
```

### StatusResponse — задача завершена с ошибкой
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "failed",
    "results": null,
    "error": "Unsupported file format: .docx"
}
```

### ProcessResponse — невалидный статус (должен отвергаться валидацией)
```json
{
    "status": "unknown",
    "message": "...",
    "results": null
}
```
Pydantic пропустит (нет enum-ограничения), но логика `api.py` такую ситуацию не порождает.

### ProcessingConfig — типичное использование
```python
ProcessingConfig(
    backend="vlm",
    method="auto",
    lang="ru",
    formula_enable=True,
    table_enable=True,
    start_page_id=0,
    end_page_id=None
)
```

## Exceptions
- **Модели определены в `api.py` вместо `dtype/`** — нарушение N2 (Constitution). По Constitution, Pydantic-модели должны находиться в `dtype/` директории сервиса. `ProcessResponse` и `StatusResponse` определены непосредственно в `mineru/api.py`.
- **`ProcessingConfig` — dataclass, не Pydantic** — нарушение P6 (Constitution). Конфигурация обработки определена как `@dataclass` вместо Pydantic `BaseModel`, хотя передаётся между `api.py` и `manager.py` как контракт. Поля `Dict[str, Any]` в `ProcessResponse.results` и `StatusResponse.results` ослабляют типизацию — содержимое `results` не имеет строгой схемы.
- **Статусы не используют `Literal` или `Enum`** — поля `status` в обеих моделях объявлены как `str`, что не даёт проверки на уровне Pydantic на допустимые значения (`"completed"`, `"failed"` и т.д.).
- **`StatusResponse` объявлен, но `/status/{task_id}` эндпоинт не реализован** — модель присутствует в коде, однако соответствующий route в `api.py` отсутствует. Модель задекларирована «на будущее».
