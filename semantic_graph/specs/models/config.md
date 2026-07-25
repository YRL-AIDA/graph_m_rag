# Data Model: Configuration (config.py)

## Purpose

Централизованная конфигурация сервиса `semantic_graph`, определённая как модульные константы в `semantic_graph/config.py`. Включает: параметры подключения к LLM, Qdrant, Neo4j, токенизатору; настройки алгоритма кластеризации Leiden; константы делимитеров и промптов для LLM-взаимодействий; схемы колонок pandas DataFrame для промежуточных данных пайплайна.

**Важно**: конфигурация определена как plain Python-константы, а не Pydantic `BaseSettings` (нарушение N3) — см. [Exceptions](#exceptions).

## Schema

### LLM / Model

```python
MODEL_NAME: str = 'Qwen/Qwen3-4B-Instruct-2507'
"""Модель LLM для извлечения сущностей и генерации отчётов."""

N4G_URL: str = 'http://192.168.19.148:9998'
"""Дополнительный URL (не используется в основном API)."""

LLM_API_KEY: str = 'EMPTY'
"""API-ключ для LLM; 'EMPTY' для локальных моделей."""

LLM_URL: str = 'http://localhost:9886/v1'
"""OpenAI-совместимый endpoint LLM."""
```

### Tokenizer

```python
TOKENIZER_URL: str = 'http://localhost:9886/tokenize'
"""Endpoint сервиса токенизации (POST, возвращает число токенов)."""
```

### Qdrant

```python
QDRANT_URL: str = "http://localhost:6333/"
"""URL Qdrant (REST API для scroll-запросов)."""

QDRANT_API_KEY: Optional[str] = None
"""API-ключ Qdrant (опционально)."""

OUTPUT_DIR: str = "data/processed"
"""Директория для выходных данных (parquet-файлы)."""

DOCUMENT_ID_FIELD: str = "file_hash"
"""Поле в payload Qdrant для идентификации документа (P2)."""
```

### Graph Processing (Clustering)

```python
MAX_CLUSTER_SIZE: int = 10
"""Максимальный размер кластера в алгоритме Leiden."""

USE_LCC: bool = False
"""Использовать только наибольшую связную компоненту (largest connected component)."""

CLUSTERIZATION_SEED: int = 256
"""Seed для воспроизводимости кластеризации."""
```

### Entity Types

```python
ENTITY_TYPES: List[str] = ['ORGANIZATION', 'PERSON', 'GEO', 'EVENT']
"""Допустимые типы сущностей, передаваемые в промпт LLM."""
```

### API Server

```python
API_HOST: str = "0.0.0.0"
"""Хост FastAPI-сервера."""

API_PORT: int = 9595
"""Порт FastAPI-сервера."""
```

### Community Report Pipeline

```python
INPUT_TEXT_KEY: str = "input_text"
"""Ключ для подстановки текста контекста сообщества в промпт."""

MAX_LENGTH_KEY: str = "max_report_length"
"""Ключ для подстановки максимальной длины отчёта в промпт."""
```

### Document Content

```python
CONTENT_LABELS: List[str] = ['text', 'table', 'image']
"""Типы регионов, считающиеся контентом (не заголовками) для построения иерархии."""
```

---

### Delimiters (формат ответа LLM)

```python
TUPLE_DELIMITER: str = "<|>"
"""Разделитель полей внутри одной записи (entity или relationship) в ответе LLM."""

RECORD_DELIMITER: str = "##"
"""Разделитель записей (entity/relationship) в ответе LLM."""

COMPLETION_DELIMITER: str = "<|COMPLETE|>"
"""Маркер завершения ответа LLM."""
```

---

### DataFrame Column Schemas

Константы-строки, используемые как имена колонок в pandas DataFrame при обработке результатов извлечения, кластеризации и генерации отчётов.

#### Базовые поля

```python
ID: str = "id"
SHORT_ID: str = "human_readable_id"
TITLE: str = "title"
DESCRIPTION: str = "description"
TYPE: str = "type"
```

#### Node Table Schema

```python
NODE_DEGREE: str = "degree"           # Степень узла-сущности
NODE_FREQUENCY: str = "frequency"      # Частота упоминания сущности
NODE_DETAILS: str = "node_details"     # Детали узла (составное поле)
```

#### Edge Table Schema

```python
EDGE_SOURCE: str = "source"           # Исходная сущность связи
EDGE_TARGET: str = "target"           # Целевая сущность связи
EDGE_DEGREE: str = "combined_degree"  # Суммарная степень обеих сущностей
EDGE_DETAILS: str = "edge_details"    # Детали связи (составное поле)
EDGE_WEIGHT: str = "weight"           # Вес связи
```

#### Community Hierarchy Table Schema

```python
SUB_COMMUNITY: str = "sub_community"  # Идентификатор дочернего сообщества
```

#### Community Context Table Schema

```python
ALL_CONTEXT: str = "all_context"             # Полный контекст сообщества
CONTEXT_STRING: str = "context_string"        # Строковое представление контекста
CONTEXT_SIZE: str = "context_size"            # Размер контекста
CONTEXT_EXCEED_FLAG: str = "context_exceed_limit"  # Флаг превышения лимита контекста
```

#### Community Report Table Schema

```python
COMMUNITY_ID: str = "community"       # ID сообщества
COMMUNITY_LEVEL: str = "level"        # Уровень в иерархии Leiden
COMMUNITY_PARENT: str = "parent"      # ID родительского сообщества
COMMUNITY_CHILDREN: str = "children"  # ID дочерних сообществ
SUMMARY: str = "summary"              # Краткое резюме отчёта
FINDINGS: str = "findings"            # Список ключевых находок
RATING: str = "rank"                  # Рейтинг важности (0-10)
EXPLANATION: str = "rating_explanation"  # Объяснение рейтинга
FULL_CONTENT: str = "full_content"    # Полный текст отчёта
FULL_CONTENT_JSON: str = "full_content_json"  # JSON-версия отчёта
```

#### Общие поля

```python
ENTITY_IDS: str = "entity_ids"            # Список ID сущностей
RELATIONSHIP_IDS: str = "relationship_ids" # Список ID связей
TEXT_UNIT_IDS: str = "text_unit_ids"       # Список ID текстовых чанков
COVARIATE_IDS: str = "covariate_ids"       # Список ID ковариат
DOCUMENT_ID: str = "document_id"           # ID документа
DEGREE: str = "degree"                     # Степень
PERIOD: str = "period"                     # Временной период
SIZE: str = "size"                         # Размер сообщества
```

#### Text Units

```python
ENTITY_DEGREE: str = "entity_degree"  # Степень сущности в текстовом юните
ALL_DETAILS: str = "all_details"      # Все детали
TEXT: str = "text"                    # Текст
N_TOKENS: str = "n_tokens"            # Число токенов
```

#### Documents

```python
CREATION_DATE: str = "creation_date"  # Дата создания документа
RAW_DATA: str = "raw_data"            # Сырые данные
```

---

### Final Column Schemas (порядок колонок в parquet-выходах)

Эти списки определяют итоговый состав и порядок колонок при экспорте данных в parquet-файлы.

#### Entities Final Columns

```python
ENTITIES_FINAL_COLUMNS = [
    ID, SHORT_ID, TITLE, TYPE, DESCRIPTION,
    TEXT_UNIT_IDS, NODE_FREQUENCY, NODE_DEGREE,
]
```

#### Relationships Final Columns

```python
RELATIONSHIPS_FINAL_COLUMNS = [
    ID, SHORT_ID, EDGE_SOURCE, EDGE_TARGET, DESCRIPTION,
    EDGE_WEIGHT, EDGE_DEGREE, TEXT_UNIT_IDS,
]
```

#### Communities Final Columns

```python
COMMUNITIES_FINAL_COLUMNS = [
    ID, SHORT_ID, COMMUNITY_ID, COMMUNITY_LEVEL,
    COMMUNITY_PARENT, COMMUNITY_CHILDREN, TITLE,
    ENTITY_IDS, RELATIONSHIP_IDS, TEXT_UNIT_IDS,
    PERIOD, SIZE,
]
```

#### Community Reports Final Columns

```python
COMMUNITY_REPORTS_FINAL_COLUMNS = [
    ID, SHORT_ID, COMMUNITY_ID, COMMUNITY_LEVEL,
    COMMUNITY_PARENT, COMMUNITY_CHILDREN, TITLE,
    SUMMARY, FULL_CONTENT, RATING, EXPLANATION,
    FINDINGS, FULL_CONTENT_JSON, PERIOD, SIZE,
]
```

#### Text Units Final Columns

```python
TEXT_UNITS_FINAL_COLUMNS = [
    ID, SHORT_ID, TEXT, N_TOKENS, DOCUMENT_ID,
    ENTITY_IDS, RELATIONSHIP_IDS,
]
```

#### Documents Final Columns

```python
DOCUMENTS_FINAL_COLUMNS = [
    ID, SHORT_ID, TITLE, TEXT, TEXT_UNIT_IDS,
    CREATION_DATE, RAW_DATA,
]
```

---

### LLM Prompts (инлайн в config.py — нарушение P7)

Промпты хранятся как строковые константы непосредственно в `config.py` (строки 179–468), а не в отдельных файлах `prompts/*.md`. Здесь документируется их наличие и назначение. Полный текст промптов см. в коде.

#### GRAPH_EXTRACTION_PROMPT

```python
GRAPH_EXTRACTION_PROMPT: str = """
-Goal- ... -Steps- ... -Examples- ... -Real Data- ...
"""
```

**Назначение**: Извлечение сущностей и связей из текста чанка.
**Параметры подстановки**: `{entity_types}` (строка из `ENTITY_TYPES`), `{input_text}` (текст чанка).
**Формат ожидаемого выхода**:
```
("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
##...
("relationship"<|>SRC<|>TGT<|>DESC<|>WEIGHT)
##...
<|COMPLETE|>
```
**Размер**: ~3000 символов, включает 3 примера (few-shot).
**Должен быть в**: `prompts/graph-extraction.md` (P7).

#### SUMMARIZE_PROMPT

```python
SUMMARIZE_PROMPT: str = """
You are a helpful assistant responsible for generating a comprehensive summary...
"""
```

**Назначение**: Суммаризация повторяющихся описаний сущностей и связей при дедупликации.
**Параметры подстановки**: `{entity_name}`, `{description_list}`, `{max_length}`.
**Должен быть в**: `prompts/summarize.md` (P7).

#### COMMUNITY_REPORT_PROMPT

```python
COMMUNITY_REPORT_PROMPT: str = """
You are an AI assistant that helps a human analyst to perform general information discovery...
"""
```

**Назначение**: Генерация аналитического отчёта по сообществу.
**Параметры подстановки**: `{input_text}` (контекст сообщества), `{max_report_length}`.
**Формат ожидаемого выхода**: JSON с полями `title`, `summary`, `rating`, `rating_explanation`, `findings`.
**Размер**: ~5000 символов, включает пример (few-shot).
**Должен быть в**: `prompts/community-report.md` (P7).

#### CONTINUE_PROMPT

```python
CONTINUE_PROMPT: str = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
```

**Назначение**: Промпт для продолжения извлечения (gleaning) — запрос дополнительных сущностей после первого прохода.
**Должен быть в**: `prompts/continue-extraction.md` (P7).

#### LOOP_PROMPT

```python
LOOP_PROMPT: str = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
```

**Назначение**: Промпт для проверки необходимости повторного прохода извлечения.
**Должен быть в**: `prompts/loop-check.md` (P7).

---

### Neo4j Connection (из переменных окружения)

Не определены в `config.py` как константы, а читаются напрямую из `os.environ` в `semantic_index.py`:

| Переменная окружения | Использование |
|----------------------|---------------|
| `USER_NEO4J` | Имя пользователя Neo4j |
| `PASSWORD` | Пароль Neo4j |
| `URL` | Neo4j Bolt-хост:порт (без схемы, например `localhost:7687`) |
| `NAME_DB` | Имя базы данных Neo4j |

Фактический URI собирается как `f"bolt://{url}"` в `semantic_index.py`.

## Storage

- **Файл**: `semantic_graph/config.py` — единый модуль Python.
- **Не версионируются**: API-ключи, пароли Neo4j (читаются из `.env` через `os.environ`, но не через Pydantic Settings).
- **in-memory**: Загружаются при импорте модуля, живут в памяти процесса.

## Relationships

| Конфигурационная группа | Используется в |
|--------------------------|---------------|
| `MODEL_NAME`, `LLM_URL`, `LLM_API_KEY` | `graphrag.py` (AsyncOpenAI-клиент), `create_community_report.py` |
| `TOKENIZER_URL` | `graphrag.py` (подсчёт токенов для обрезки контекста) |
| `QDRANT_URL`, `DOCUMENT_ID_FIELD` | `Qdrant_extractor/QdrantReader.py` (scroll чанков) |
| `MAX_CLUSTER_SIZE`, `USE_LCC`, `CLUSTERIZATION_SEED` | `clasterization.py` (Leiden-кластеризация) |
| `ENTITY_TYPES` | `graphrag.py` (подстановка в GRAPH_EXTRACTION_PROMPT) |
| `TUPLE_DELIMITER`, `RECORD_DELIMITER`, `COMPLETION_DELIMITER` | `graphrag.py` (парсинг ответа LLM) |
| `GRAPH_EXTRACTION_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT` | `graphrag.py` (извлечение + gleaning) |
| `SUMMARIZE_PROMPT` | `graphrag.py` (суммаризация описаний) |
| `COMMUNITY_REPORT_PROMPT` | `create_community_report.py` (генерация отчётов) |
| `*_FINAL_COLUMNS` | Все pipeline-модули (экспорт parquet) |
| Column name константы | `graphrag.py`, `clasterization.py`, `create_community_report.py` |
| `CONTENT_LABELS` | `region.py` (Region.is_content) |
| `API_HOST`, `API_PORT` | `api.py` (uvicorn.run) |

## Constraints

| Ограничение | Детали |
|-------------|--------|
| Типы конфигурации | Все константы — модульного уровня, не Pydantic Settings (N3) |
| `MODEL_NAME` | Должен соответствовать имени модели на LLM-сервере |
| `ENTITY_TYPES` | Список строк, используется в `.format(entity_types=...)`. Не должен содержать запятых внутри элементов (разделитель в подстановке) |
| `TOKENIZER_URL` | Дублируется: определено дважды (строки 170, 172), второе значение перезаписывает первое |
| `LLM_URL` | Дублируется: определено дважды (строки 171, 173), второе значение перезаписывает первое. Закомментированные альтернативные URL (строки 174-175) |
| Prompts | Хранятся инлайн, не в `prompts/*.md` (P7). Формат: `.format()` с именованными placeholder'ами |
| `*_FINAL_COLUMNS` | Жёстко заданный порядок колонок. Изменение требует согласования со всем pipeline |
| Neo4j credentials | Не в `config.py`, читаются из `os.environ` в `semantic_index.py` |
| `CONTENT_LABELS` | Дублируется: определено в `config.py` и захардкожено в `documet_index/dtype/region.py`. Источник истины — `config.py` для `semantic_graph` |

## Examples

### Подстановка в GRAPH_EXTRACTION_PROMPT

```python
from semantic_graph.config import GRAPH_EXTRACTION_PROMPT, ENTITY_TYPES

prompt = GRAPH_EXTRACTION_PROMPT.format(
    entity_types=",".join(ENTITY_TYPES),
    input_text="The quick brown fox jumps over the lazy dog."
)
# Результат: полный текст промпта с entity_types="ORGANIZATION,PERSON,GEO,EVENT"
# и input_text="The quick brown fox jumps over the lazy dog."
```

### Использование делимитеров при парсинге ответа LLM

```python
from semantic_graph.config import TUPLE_DELIMITER, RECORD_DELIMITER, COMPLETION_DELIMITER

response = '("entity"<|>ACME<|>ORGANIZATION<|>...)##("relationship"<|>...)##<|COMPLETE|>'
records = response.split(RECORD_DELIMITER)  # ['("entity"<|>ACME<|>...', '("relationship"<|>...', '<|COMPLETE|>']
for record in records:
    if COMPLETION_DELIMITER in record:
        break
    fields = record.strip("()").split(TUPLE_DELIMITER)
    # fields = ['"entity"', 'ACME', 'ORGANIZATION', '...']
```

### Использование column schemas в DataFrame

```python
from semantic_graph.config import ENTITIES_FINAL_COLUMNS
import pandas as pd

df = pd.DataFrame(columns=ENTITIES_FINAL_COLUMNS)
# df.columns → ['id', 'human_readable_id', 'title', 'type', 'description',
#               'text_unit_ids', 'frequency', 'degree']
```

## Exceptions

| Отклонение | Обоснование |
|------------|-------------|
| **N3 — Не Pydantic Settings** | Конфигурация определена как модульные константы, а не через Pydantic `BaseSettings` в `config/settings.py`. Часть параметров (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NAME_DB) читается из `os.environ` напрямую, а не централизованно. Требуется рефакторинг на Pydantic Settings. |
| **P7 — Промпты инлайн в config.py** | Все 5 промптов (`GRAPH_EXTRACTION_PROMPT`, `SUMMARIZE_PROMPT`, `COMMUNITY_REPORT_PROMPT`, `CONTINUE_PROMPT`, `LOOP_PROMPT`) хранятся как строковые константы в `config.py`, а не в отдельных файлах `prompts/*.md`. Требуется вынос в файлы. |
| **Дублирование URL** | `TOKENIZER_URL` и `LLM_URL` определены дважды (с разными значениями), второе определение перезаписывает первое. Закомментированные альтернативные URL остались как технический долг. |
| **Отсутствие валидации** | Ни одна константа не проходит валидацию при старте (например, `LLM_URL` не проверяется на доступность). Переход на Pydantic Settings решит эту проблему через `model_post_init` или field validators. |
| **Промпты на английском, спецификация на русском** | Промпты LLM написаны на английском (целевой язык модели), документация — на русском (N1). Это соответствует Constitution. |
