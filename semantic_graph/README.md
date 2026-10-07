# semantic_graph

Модуль построения семантического графа знаний (GraphRAG) поверх чанков документов из Qdrant. Извлекает сущности и связи с помощью LLM, сохраняет их в Neo4j, кластеризует граф (алгоритм Leiden) и генерирует отчёты по сообществам.

## Архитектура

```
Qdrant (чанки) → graphrag (извлечение + суммаризация) → Neo4j (Entity / RELATED)
                                                          ↓
                                              clasterization (Leiden) → Community
                                                          ↓
                                              create_community_report (LLM) → отчёты
```

Точка входа — FastAPI-сервис `semantic_index.py` (порт по умолчанию `9595`).

## Требования

| Компонент | Назначение |
|-----------|------------|
| **Neo4j** | Хранение сущностей, связей, сообществ и отчётов |
| **Qdrant** | Источник текстовых чанков документов |
| **LLM API** | OpenAI-совместимый endpoint (`LLM_URL` в `config.py`) |
| **Tokenizer API** | HTTP-сервис подсчёта токенов (`TOKENIZER_URL`) |
| **Python 3.10+** | Асинхронные пайплайны, Pydantic v2 |

Дополнительные Python-зависимости, не указанные в `semantic_graph/requirements.txt`, но используемые кодом:

- `neo4j`, `openai`, `aiohttp`, `python-dotenv` — из корневого `requirements.txt` репозитория
- `graspologic_native` — кластеризация Leiden
- `numpy` — обработка графа

## Установка 

### 1. Neo4j (Если сервис запускается без контейнеров основного проекта)

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j123 \
  neo4j
```

> Обновлено: Функция `Manager.add_entities_batch` больше не требует плагин **APOC**, так как используется стандартная функциональность Cypher для объединения списков без дубликатов.

### 2. Переменные окружения

```bash
cd semantic_graph
cp .env.example .env
```

| Переменная | Описание | Пример |
|------------|----------|--------|
| `USER_NEO4J` | Пользователь Neo4j | `neo4j` |
| `PASSWORD` | Пароль Neo4j | `neo4j123` |
| `URL` | Хост:порт Bolt (без схемы) | `localhost:7687` |
| `NAME_DB` | Имя базы данных | `neo4j` |

### 3. Python-зависимости

```bash
# из корня репозитория (рекомендуется — полный набор зависимостей)
pip install -r requirements.txt

# или минимальный набор для API
pip install -r semantic_graph/requirements.txt
pip install neo4j openai aiohttp python-dotenv graspologic_native numpy
```

### 4. Конфигурация сервисов

Параметры в `config.py` (при необходимости отредактировать перед запуском):

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `QDRANT_URL` | `http://localhost:6333/` | URL Qdrant |
| `LLM_URL` | `http://localhost:9886/v1` | OpenAI-совместимый LLM endpoint |
| `TOKENIZER_URL` | `http://localhost:9886/tokenize` | Сервис токенизации |
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Модель для извлечения и отчётов |
| `ENTITY_TYPES` | `ORGANIZATION, PERSON, GEO, EVENT` | Типы сущностей для промпта |
| `MAX_CLUSTER_SIZE` | `10` | Макс. размер кластера Leiden |
| `USE_LCC` | `False` | Ограничить граф наибольшей связной компонентой |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `9595` | Адрес FastAPI |

* LLM endpoint и Сервис токенизации по сути единый vllm сервер 
## Запуск

### API-сервис

```bash
cd semantic_graph
python semantic_index.py
```

Или через uvicorn:

```bash
uvicorn semantic_index:app --host 0.0.0.0 --port 9595
```

Документация Swagger: `http://localhost:9595/docs`

### Типовой пайплайн (последовательность вызовов)

```bash
# 1. Извлечь сущности и связи из чанков документа
curl -X POST http://localhost:9595/process-document \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<file_hash>"}'

# 2. Кластеризовать граф (по всем связям в Neo4j)
curl http://localhost:9595/clastrize_graph

# 3. Сгенерировать отчёты по сообществам
curl http://localhost:9595/create_community_report
```



---

## Функционал

### HTTP API (`semantic_index.py`)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/process-document` | Загрузка чанков из Qdrant → извлечение графа → сохранение в Neo4j |
| `GET` | `/clastrize_graph` | Иерархическая кластеризация связей → создание узлов `Community` |
| `GET` | `/create_community_report` | Генерация LLM-отчётов по сообществам → запись в Neo4j |

**Тело запроса `/process-document`:**

```json
{"document_id": "<file_hash>"}
```

**Формат ответа (все эндпоинты):**

```json
{
  "document_id": "...",
  "statistics": {
    "total_chunks": 42,
    "processing_time_ms": 15000,
    "status": "completed",
    "nodes_created": 10,
    "nodes_updated": 5,
    "relationships_added": 8
  }
}
```

Возможные значения `status`: `completed`, `completed_without_entities`.

### Основные модули

| Модуль | Назначение |
|--------|------------|
| `graphrag.py` | Извлечение сущностей/связей из текста, суммаризация описаний |
| `manager.py` | CRUD-операции с Neo4j: сущности, связи, сообщества, отчёты |
| `clasterization.py` | Иерархическая кластеризация Leiden |
| `create_community_report.py` | Построение контекста и генерация отчётов по сообществам |
| `Qdrant_extractor/` | Чтение чанков из Qdrant, экспорт в GraphRAG CSV |
| `neo4j_service.py` | Обёртка для индексации документов MinerU (отдельный сценарий) | 
| `dtype/` | Pydantic-модели запросов/ответов |

* neo4j_service - неиспользуется в скриптах построения семантического графа
---

## Сигнатуры основных функций

### API-обработчики

```python
async def process_document(request: DocumentRequest) -> Dict[str, Any]
async def clastrize_graph() -> Dict[str, Any]
async def create_community_report() -> Dict[str, Any]
```

### Пайплайны

```python
async def run_extraction_pipeline_async(
    text_units: pd.DataFrame,          # колонки: id, text
    extraction_model: str,
    summarization_model: str,
    entity_types: List[str],
    max_gleanings: int = 1,
    max_summary_length: int = 500,
    max_input_tokens: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]  # (entities, relationships)

async def create_communities(
    relationships: pd.DataFrame,       # source, target, weight, text_unit_ids
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> list[dict[str, Any]]

async def run_community_reports_pipeline_async(
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    model: str,
    prompt: str = COMMUNITY_REPORT_PROMPT,
    max_input_length: int = 16000,
    max_report_length: int = 2000,
    max_concurrent: int = 4,
) -> pd.DataFrame
```

### Manager (Neo4j)

```python
class Manager:
    def add_entities_batch(self, req: EntitiesRequest) -> EntitiesResponse
    def get_entities(self) -> pd.DataFrame
    def get_entity_relationships(self) -> pd.DataFrame
    def get_community(self) -> pd.DataFrame
    def insert_communities_to_neo4j(
        self, communities_rows: List[Dict[str, Any]], batch_size: int = 1000
    ) -> Dict[str, int]
    def update_community_reports(
        self, community_reports: pd.DataFrame, batch_size: int = 500
    ) -> Dict[str, int]
```

### Pydantic-модели (`dtype/entity.py`)

```python
class DocumentRequest(BaseModel):
    document_id: str

class EntityCreate(BaseModel):
    title: str
    type: str
    text_unit_ids: Optional[List[str]] = []
    frequency: Optional[int] = 0
    description: Optional[str] = None
    degree: Optional[int] = 0

class RelationshipCreate(BaseModel):
    source: str          # формат "TITLE|TYPE"
    target: str
    text_unit_ids: Optional[List[str]] = []
    weight: Optional[float] = 1.0
    description: Optional[str] = None
    combined_degree: Optional[int] = 0

class EntitiesRequest(BaseModel):
    entities: List[EntityCreate]
    relationships: List[RelationshipCreate]

class EntitiesResponse(BaseModel):
    nodes_created: int
    nodes_updated: int
    relationships_added: int
```

### Qdrant

```python
def build_chunks_dataframe(
    adapter,
    doc_id_field: str = "file_hash",
    doc_id: Optional[str] = None,
) -> pd.DataFrame
# колонки: document_id, chunk_id, collection, text, page, element_index, created_at
```

### LLM-клиент

```python
class AsyncLLMClient:
    async def generate(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> Optional[str]

    async def count_tokens(self, text: str, model: str) -> int

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response_model: Type[T],
        **kwargs,
    ) -> Optional[T]
```

---

## Модель данных Neo4j

| Узел / связь | Метки / тип | Ключевые свойства |
|--------------|-------------|-------------------|
| Сущность | `:Entity` | `title`, `type` (уникальная пара), `description`, `degree`, `text_unit_ids` |
| Связь | `:RELATED` | `source` → `target`, `weight`, `description`, `combined_degree` |
| Сообщество | `:Community` | `id`, `community`, `level`, `parent`, `size`, `title`, `summary`, `findings` |
| Иерархия | `:IS_CHILD_OF`, `:IS_PARENT_OF` | между `Community` |
| Принадлежность | `:CONSISTS_OF` | `Community` → `Entity` |

Идентификатор сущности в DataFrame: `"TITLE|TYPE"` (например, `"ACME CORP|ORGANIZATION"`).

---

## Нюансы и ограничения

### Qdrant

- Чанки фильтруются по полю `file_hash` (`DOCUMENT_ID_FIELD` в `config.py`).
- Берутся только точки, у которых `payload.original_element.type == "text"`.
- Текст читается из `payload.original_element.text`, а не из верхнеуровневого поля.
- `chunk_id` формируется как `{file_hash}|{region_id}`.

### Извлечение графа (LLM)

- Требуется доступный OpenAI-совместимый endpoint и сервис токенизации.
- Парсинг ответа LLM зависит от разделителей `<|>`, `##`, `<|COMPLETE|>` — формат задан в `GRAPH_EXTRACTION_PROMPT`.
- При `max_gleanings=0` (как в API) повторные проходы извлечения отключены.
- Если сущности не извлечены — возвращается `completed_without_entities`, а не ошибка.
- Имена сущностей нормализуются в **UPPER CASE**.
- Связи без существующих сущностей отфильтровываются (`filter_orphan_relationships`).

### Neo4j

- Дедупликация сущностей по паре `(title, type)`; повторная загрузка тех же `text_unit_ids` пропускается.
- Для merge `text_unit_ids` и обновления описаний нужен плагин **APOC**.
- `clastrize_graph` и `create_community_report` работают **со всем графом** в базе, не с отдельным документом.
- Повторный вызов `clastrize_graph` создаёт/обновляет сообщества через `MERGE`, но не очищает старые данные.

### Кластеризация

- Алгоритм: иерархический Leiden (`graspologic_native`).
- `USE_LCC=True` — только наибольшая связная компонента графа.
- `seed` обеспечивает воспроизводимость (`CLUSTERIZATION_SEED=256`).
- Пустой граф связей → пустой результат, без ошибки.

### Отчёты по сообществам

- Требуют предварительно выполненных шагов: извлечение сущностей + `clastrize_graph`.
- LLM возвращает JSON; парсинг через `generate_structured` (Pydantic `CommunityReportResponse`).
- Контекст обрезается по лимиту токенов (`max_input_length`).
- Параллелизм ограничен `max_concurrent=4`.
- Отчёты записываются в существующие узлы `Community` по полю `id`.

### Порядок выполнения

```
process-document  →  clastrize_graph  →  create_community_report
```

Каждый следующий шаг зависит от результатов предыдущего.

### Производительность

- Извлечение графа — один LLM-запрос на чанк (параллельно через `asyncio.gather`).
- Большие документы с сотнями чанков создают высокую нагрузку на LLM.
- `build_chunks_dataframe` — синхронный HTTP-скроллинг Qdrant; при больших коллекциях может занять значительное время.

---

## Структура каталога

```
semantic_graph/
├── semantic_index.py      # FastAPI-сервис
├── graphrag.py            # Извлечение и суммаризация графа
├── manager.py             # Neo4j Manager
├── clasterization.py      # Leiden-кластеризация
├── create_community_report.py
├── config.py              # Конфигурация и промпты
├── neo4j_service.py       # Сервис индексации MinerU-документов
├── bootstrap.py           # Добавление корня в sys.path
├── dtype/                 # Pydantic-модели
├── Qdrant_extractor/      # Адаптер Qdrant + экспорт CSV
├── tests/                 # Unit-тесты
├── requirements.txt
└── .env.example
```
