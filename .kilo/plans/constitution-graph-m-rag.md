# Constitution — graph_m_rag

> **Назначение**: Свод неизменяемых принципов, архитектурных инвариантов и правил Spec-Driven Development для проекта `graph_m_rag`.  
> **Язык**: Русский для документации и спецификаций. Английский для идентификаторов кода.  
> **Версия**: 1.0.0  
> **Статус**: Принято

---

## 1. Идентичность проекта

**graph_m_rag** — система графового мультимодального Retrieval-Augmented Generation (GraphRAG) для обработки PDF-документов:

- Извлечение структурированного контента из PDF (текст, изображения, таблицы, формулы)
- Векторный поиск по чанкам документа (Qdrant)
- Построение графа структуры документа (Neo4j) — порядок чтения и иерархия заголовков
- Извлечение семантического графа знаний: сущности, связи, сообщества (LLM + Neo4j)
- Генерация иерархических отчётов по сообществам
- Вопросно-ответный интерфейс с мультимодальным контекстом (текст + изображения)

---

## 2. Неизменяемые принципы

### P1. Сервисные границы священны

Каждый сервис проекта владеет своей зоной ответственности. Никакой сервис не обращается к хранилищу другого сервиса напрямую — только через API или выделенный клиентский модуль.

| Сервис | Зона ответственности | Хранилище | Порт |
|--------|---------------------|-----------|------|
| `app` | Оркестрация, ingestion pipeline, Q&A, PDF-рендеринг | MinIO (S3) | 8000 |
| `semantic_graph` | Извлечение сущностей, кластеризация, отчёты сообществ | Neo4j (semantic graph) | 9595 |
| `documet_index` | Граф структуры документа (ORDER/PARENT) | Neo4j (document graph) | — (библиотека) |
| `mineru` | PDF → structured content (OCR, VLM) | Файловая система (кэш моделей) | 8000 (внутр.) |
| `qdrant` | Векторная БД | Собственное хранилище | 6333 |

**Правило**: При добавлении нового сервиса он получает собственное хранилище или явно определённый интерфейс доступа к существующему.

### P2. Единый идентификатор документа — `file_hash`

MD5-хеш содержимого PDF-файла используется как первичный ключ документа во **всех** хранилищах:

- MinIO: `pdfs/{file_hash}_*/`
- Qdrant: поле `file_hash` в payload каждой точки
- Neo4j (document graph): `Document.name`
- Neo4j (semantic graph): фильтрация по `text_unit_ids` (производный от `file_hash`)

**Правило**: Любая новая сущность, связанная с документом, обязана использовать `file_hash` для cross-store идентификации.

### P3. Сквозной идентификатор региона — `file_hash|region_id`

Каждый элемент документа (region) имеет уникальный `region_id`, который совпадает в:
- Qdrant point (в payload как `region_id`)
- Neo4j Document Graph (`Region` node)

Как композитный уникальный идентификатор региона в Neo4j используется связка `file_hash|region_id`.

**Правило**: При добавлении нового элемента в Qdrant или Neo4j для сохранения связи с источником обязана использоваться комбинация `file_hash|region_id`, если источником выступает регион документа.

### P4. Два графа Neo4j — два набора данных

В Neo4j сосуществуют два независимых графа:

| Граф | Типы узлов | Типы связей | Модуль |
|------|-----------|-------------|--------|
| Document Structure | `Document`, `Region:*` | `ORDER`, `PARENT` | `documet_index` |
| Semantic Knowledge | `Entity`, `Community` | `RELATED`, `CONSISTS_OF`, `IS_CHILD_OF`, `IS_PARENT_OF` | `semantic_graph` |

**Правило**: Запросы к одному графу не должны затрагивать узлы/связи другого. Если нужна связь между графами — она должна быть явно задокументирована в спецификации.

### P5. Асинхронность для I/O-bound операций

Все операции, включающие сетевые вызовы (LLM, БД, HTTP), обязаны быть асинхронными (`async/await`, `asyncio.gather` для параллелизации). Синхронный код допустим только для CPU-bound вычислений (хэширование, рендеринг PDF) и должен запускаться через `run_in_executor`.

### P6. Pydantic — источник истины для моделей данных

Все модели данных, передаваемые между компонентами системы, определяются как Pydantic-модели. Никакие `dict` или `tuple` не должны использоваться как контракты между модулями.

**Исключение**: pandas DataFrames допустимы для внутренних вычислений в pipeline (извлечение, кластеризация), но на вход/выход модулей должны преобразовываться в Pydantic-модели.

### P7. Промпты вынесены из кода в файлы

LLM-промпты определяют поведение системы. Промпт хранится в отдельном `.md` файле в директории `prompts/` сервиса. Код загружает промпт из файла, а не из строковой константы. Спецификация ссылается на файл промпта и описывает требования к нему (вход/выход/ограничения).

**Расположение**: `{service}/prompts/{prompt-name}.md`

**Правило**: Изменение текста промпта = изменение поведения системы = требует обновления спецификации.

### P8. Новая функциональность — только через спецификацию

Любое изменение, добавляющее новое поведение, новый эндпоинт, новую модель данных или новый пайплайн, обязано начинаться со спецификации. Исключения требуют явного обоснования в самом spec-файле (поле `Exceptions`).

### P9. Семантический граф глобален

Пайплайны кластеризации (`clastrize_graph`) и генерации отчётов сообществ (`create_community_report`) в `semantic_graph` работают со всем графом Neo4j, а не в разрезе отдельного документа. Это осознанное архитектурное решение — сообщества строятся на всём корпусе загруженных документов для кросс-документных инсайтов.

**Следствия**:
- Добавление нового документа может изменить состав сообществ и отчёты для всего графа
- Удаление документа требует перестроения сообществ
- `process-document` (извлечение сущностей) — per-document операция
- `clastrize_graph` + `create_community_report` — always-full-graph операции, запускаются после накопления изменений

### P10. Мягкое удаление из семантического графа

При удалении документа его сущности (`Entity`) и связи (`RELATED`) не удаляются физически из Neo4j, а помечаются как `archived`. Archived-сущности исключаются из кластеризации и отчётов, но сохраняются для возможности восстановления документа без повторного извлечения сущностей.

**Правило**: Любая операция, скрывающая сущность из активного графа, должна использовать флаг `archived`, а не физическое удаление.

---

## 3. Конфигурация и управление средой

### C1. Единый стандарт: Pydantic Settings для всех сервисов

Конфигурация любого сервиса управляется через `pydantic-settings` с классами настроек и префиксами. Переменные окружения — единственный способ переопределения.

**Текущее состояние**: `app/config/settings.py` уже использует Pydantic Settings. `semantic_graph/config.py` — plain Python константы, требует миграции (см. Этап 7 плана внедрения).

### C2. Чувствительные данные — только через .env

Пароли, API-ключи, токены никогда не хардкодятся. Всегда через `.env` / переменные окружения. Файл `.env` в `.gitignore`.

---

## 4. Структура спецификаций

### 4.1 Расположение

Спецификации хранятся рядом с кодом сервиса, к которому относятся:

```
app/specs/                    # Спецификации главного приложения
semantic_graph/specs/         # Спецификации semantic graph сервиса
documet_index/specs/          # Спецификации документного графа
mineru/specs/                 # Спецификации MinerU сервиса
```

### 4.2 Типы спецификаций

| Тип | Назначение | Имя файла |
|-----|-----------|-----------|
| **Service Spec** | Описывает сервис целиком: зона ответственности, API, зависимости | `SERVICE.md` |
| **Feature Spec** | Описывает одну фичу/эндпоинт/пайплайн. Для API-эндпоинтов включает OpenAPI-фрагмент. | `{feature-name}.md` |
| **Data Model Spec** | Описывает модель данных, схему БД, форматы обмена. Каждая Pydantic-модель из `dtype/` обязана иметь spec. | `models/{entity-name}.md` |
| **Pipeline Spec** | Описывает многоэтапный пайплайн обработки | `pipelines/{pipeline-name}.md` |
| **Integration Spec** | Описывает контракт между двумя сервисами | `integrations/{from}-{to}.md` |

### 4.3 Шаблон Service Spec

```markdown
# Service: {Имя сервиса}

## Identity
- Зона ответственности: ...
- Порт: {port}
- Хранилище: {database / file system / none}

## Dependencies
| Сервис | Протокол | Назначение |
|--------|----------|------------|
| ...    | REST/gRPC/driver | ... |

## API
| Method | Path | Feature Spec | Purpose |
|--------|------|-------------|---------|
| POST   | /upload-pdf | [document-ingestion.md](document-ingestion.md) | Загрузка PDF |

## Data Model
- Ссылки на Data Model Specs: ...

## Pipelines
- Ссылки на Pipeline Specs: ...

## Configuration
- Переменные окружения: ...

## Invariants
- Список инвариантов, которые этот сервис гарантирует

## Exceptions
- Обоснованные отклонения от Constitution (если есть)
```

### 4.4 Шаблон Feature Spec

```markdown
# Feature: {Название фичи}

## Motivation
- Какую проблему решает, почему это нужно

## Behaviour
### Input
- Что принимает на вход, формат, валидация

### Processing
- Пошаговое описание логики обработки

### Output
- Что возвращает, формат, коды ошибок

## API Contract
<!-- Только для эндпоинтов. Валидируется CI против сгенерированного OpenAPI. -->

```openapi
{method}: {path}
summary: ...
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/{RequestModel}'
responses:
  '200':
    description: ...
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/{ResponseModel}'
  '400':
    description: Invalid request
  '500':
    description: Internal processing error
```

## Data Flow
- Какие хранилища затрагивает, в каком порядке
- Диаграмма (mermaid или описание текстом)

## LLM Interactions
- Ссылки на файлы промптов: `prompts/{prompt-name}.md`
- Требования к промпту (вход/выход/ограничения)

## LLM Model Requirements
- **Тип модели**: text-only / multimodal (text+image) / reasoning
- **Минимальный размер контекста**: {N} токенов
- **Язык выхода**: русский / английский / multilingual
- **Требования к формату выхода**: JSON / свободный текст / структурированный список
- Конкретная модель задаётся в конфигурации сервиса (Pydantic Settings)

## Error Handling
- Какие ошибки возможны, как обрабатываются

## Testing
- Ключевые тест-кейсы (сценарии, граничные случаи)

## Dependencies
- Какие сервисы/модули использует

## Exceptions
- Обоснованные отклонения от Constitution (если есть)
```

### 4.5 Шаблон Data Model Spec

```markdown
# Data Model: {Имя модели}

## Purpose
- Для чего используется эта модель

## Schema
```python
class {ModelName}(BaseModel):
    field: type  # описание
```

## Storage
- Где хранится: Qdrant payload / Neo4j node / MinIO object / in-memory

## Relationships
- Связи с другими моделями

## Constraints
- Уникальность, обязательность полей, валидация

## Examples
- Примеры валидных и невалидных данных
```

### 4.6 Шаблон Pipeline Spec

```markdown
# Pipeline: {Название пайплайна}

## Purpose
- Что делает пайплайн, какой результат производит

## Stages
| Stage | Input | Processing | Output | Concurrency |
|-------|-------|------------|--------|-------------|
| 1. ... | ... | ... | ... | sequential/parallel |

## Data Flow Diagram
- Визуализация этапов (mermaid или текстом)

## LLM Interactions
- Ссылки на файлы промптов: `prompts/{prompt-name}.md`
- Требования к промптам (не точный текст)

## Performance Constraints
- Ограничения по токенам, таймауты, параллелизм

## Error Recovery
- Что происходит при сбое на каждом этапе
```

### 4.7 Формат OpenAPI-фрагментов в Feature Spec

Для каждого API-эндпоинта Feature Spec обязан содержать секцию `## API Contract` с OpenAPI-фрагментом в формате YAML внутри блока `openapi`. Фрагмент описывает:

- HTTP method и path
- `summary` — краткое описание
- `requestBody` — схема запроса (ссылка на Pydantic-модель из `dtype/`)
- `responses` — минимум `200`, `400`, `500`

CI проверяет, что:
1. Сгенерированный из FastAPI OpenAPI содержит эндпоинт с такими же method, path и статус-кодами
2. Все эндпоинты из кода имеют соответствующий Feature Spec с OpenAPI-фрагментом

### 4.8 Формат файлов промптов

Промпты хранятся в `{service}/prompts/{prompt-name}.md`:

```markdown
# Prompt: {Название промпта}

## Purpose
- Какую задачу решает промпт, в каком пайплайне используется

## Input Variables
| Переменная | Тип | Описание |
|------------|-----|----------|
| `{var}` | str | ... |

## Output Format
- Что ожидается от LLM (JSON, CSV, список кортежей, свободный текст)

## Constraints
- Ограничения: max токенов, обязательные поля, валидация выхода

## Version
- Версия промпта: {semver}

---

{текст промпта}
```

Код загружает промпт из файла:

```python
from pathlib import Path

def load_prompt(name: str) -> str:
    prompt_file = Path(__file__).parent / "prompts" / f"{name}.md"
    content = prompt_file.read_text(encoding="utf-8")
    # Извлекаем текст после разделителя `---`
    return content.split("---", 2)[-1].strip()
```

---

## 5. SDD Workflow

### 5.1 Жизненный цикл спецификации

```
Draft → Review → Approved → Implementation → Verification → Done
  ↑                                                    |
  └────────────────── Rework ←─────────────────────────┘
```

| Стадия | Ответственный | Действие |
|--------|--------------|----------|
| **Draft** | Разработчик | Создание spec-файла по шаблону, заполнение всех секций |
| **Review** | Технический лидер / второй разработчик | Проверка на соответствие Constitution, полноту, непротиворечивость |
| **Approved** | Технический лидер | Spec принят, можно начинать реализацию |
| **Implementation** | Разработчик | Код соответствует спецификации |
| **Verification** | Разработчик + Reviewer | Код ревью сверяется со spec, тесты проходят, CI проверки зелёные |
| **Done** | — | Spec и код в main-ветке |

### 5.2 Правила работы со спецификациями

1. **Spec-first**: Спецификация пишется до кода. Без Approved-спецификации нельзя начинать реализацию.
2. **Spec — контракт**: Код, не соответствующий спецификации — баг.
3. **Spec меняется с кодом**: Если в процессе реализации выясняется необходимость изменений, обновляется spec и проходит повторный Review.
4. **Spec в том же PR**: Изменения кода и соответствующий spec-файл идут в одном PR.
5. **Exception документируется**: Любое отклонение от Constitution фиксируется в секции `Exceptions` спецификации с обоснованием.

### 5.3 Исключения из spec-first

Допустимо начинать без спецификации в случаях:
- **Bug fix**: исправление поведения, которое уже должно работать по spec (или по поведению системы «как есть» в переходный период)
- **Refactoring**: изменение структуры кода без изменения поведения
- **Experiments/Spikes**: исследовательский код в отдельной ветке (перед мержем обязана появиться спецификация)

### 5.4 Переходный период (до полного покрытия spec-ами)

На старте spec-ы отсутствуют для всего существующего кода. Правила переходного периода:

| Ситуация | Правило |
|----------|---------|
| **Новая фича / эндпоинт / модель** | Строгий spec-first. Без Approved spec — код не пишется. |
| **Изменение существующего поведения** | Сначала пишется spec «как есть» (документирование текущего поведения), затем spec обновляется под желаемое поведение, затем правится код. |
| **Bug fix в существующем коде** | Можно править без spec. После фикса рекомендуется добавить spec на исправленный компонент. |
| **Документирование существующего (Этапы 2-5 плана)** | Spec пишется по текущему поведению кода. Не требует изменения кода. |

### 5.5 Автоматическая валидация (CI)

CI выполняет следующие проверки при каждом PR:

#### API Contracts (автоматически)
1. Генерируется OpenAPI spec из FastAPI (`/openapi.json`)
2. Для каждого эндпоинта в коде проверяется наличие Feature Spec с OpenAPI-фрагментом
3. method, path и status codes в OpenAPI-фрагменте сверяются со сгенерированным spec
4. Расхождение = CI failure

#### Структурная проверка spec-файлов (автоматически)
1. Каждая Pydantic-модель из `dtype/` должна иметь Data Model Spec в `specs/models/`
2. Каждый spec-файл должен содержать все обязательные секции шаблона
3. Отсутствие обязательной секции = CI warning (на переходный период) → CI failure (после полного покрытия)

#### Pipeline/Behaviour проверка (ручной review)
- Pipeline Specs и Data Model Specs проверяются вручную при code review
- Reviewer обязан сверить описанное поведение с реализацией

### 5.6 Критерии готовности спецификации

Спецификация считается готовой (Approved), когда:

1. **Все обязательные секции шаблона заполнены** — нет placeholder'ов `...` или `TODO`
2. **Достаточно для реализации** — другой разработчик может реализовать фичу, руководствуясь только спецификацией, не заглядывая в существующий код
3. **Определены граничные случаи** — секция Error Handling описывает все известные failure modes
4. **Определены тест-кейсы** — секция Testing содержит конкретные сценарии для проверки
5. **Нет противоречий с Constitution** — секция Exceptions пуста или содержит обоснованные отклонения

Для документационных spec-ов (существующий код) дополнительно:
6. **Соответствует фактическому поведению** — reviewer проверил, что описанное поведение совпадает с кодом

---

## 6. Тестирование

### T1. Уровни тестирования

| Уровень | Что тестирует | Инструменты |
|---------|--------------|-------------|
| Unit | Отдельные функции, Pydantic-модели, чистые преобразования | pytest |
| Integration | Взаимодействие с БД, внешними сервисами | pytest + testcontainers / docker-compose |
| Pipeline | Сквозные пайплайны на тестовых данных | pytest |
| API | HTTP-эндпоинты, контракты | pytest + httpx |

### T2. Тестовые данные

- Фикстуры лежат в `tests/fixtures/` сервиса
- Тестовые PDF — минимального размера, покрывают все типы контента
- Для LLM-зависимых тестов — моки или предзаписанные ответы

### T3. LLM-тестирование

Избегать вызовов LLM в тестах (недетерминированность, стоимость, скорость). Вместо этого:
- Мокать `AsyncLLMClient` с предзаписанными ответами
- Снэпшот-тесты для парсинга ответов LLM (`_parse_result`, `_parse_tuple`)

Исключение: для интеграционного тестирования API к LLM-моделям допустимы реальные вызовы с малыми моделями.

---

## 7. Именование и стиль кода

### N1. Язык

- **Комментарии и документация**: русский
- **Идентификаторы** (функции, классы, переменные): английский
- **Промпты**: русский (целевой язык системы)
- **Сообщения коммитов**: русский или английский (на выбор автора, но консистентно в PR)

### N2. Структура модулей

```
{service}/
├── specs/               # Спецификации сервиса
│   ├── SERVICE.md
│   ├── models/
│   ├── pipelines/
│   └── integrations/
├── prompts/             # LLM-промпты (вынесены из кода)
│   ├── graph-extraction.md
│   ├── summarize.md
│   └── community-report.md
├── src/ или корень      # Код
│   ├── api.py           # HTTP-эндпоинты (FastAPI)
│   ├── manager.py       # Бизнес-логика / оркестрация
│   ├── config/          # Конфигурация (Pydantic Settings)
│   │   └── settings.py
│   ├── dtype/           # Модели данных (Pydantic)
│   └── tests/           # Тесты
```

### N3. Конфигурация

- Настройки, специфичные для сервиса — в `config/settings.py` (Pydantic Settings)
- Промпты — в `prompts/*.md` (не в коде)
- Константы доменной логики (ENTITY_TYPES, MAX_CLUSTER_SIZE) — в `config/settings.py` как часть Settings-класса

### N4. Правила коммитов

Важно! Новая фича — новая ветка. Название ветки — краткое название фичи.

#### Формат

В период выполнения плана внедрения (Этапы 2-11) коммиты маркируются привязкой к треку и этапу:

```
[Трек-Этап] краткое описание

spec: specs/{service}/{file}.md   # ссылка на спецификацию (если есть)
```

| Тип изменения | Формат | Пример |
|--------------|--------|--------|
| Документирование (трек A) | `[A-{N}] описание` | `[A-2] SERVICE.md — semantic_graph сервис` |
| Рефакторинг (трек B) | `[B-{N}] описание` | `[B-7] миграция config.py на Pydantic Settings` |
| Новая фича (трек C) | `[C-{N}] описание` | `[C-11] spec на мягкое удаление сущностей` |

После завершения плана внедрения — стандартный формат:

```
type(scope): описание          # англ. или рус., консистентно в PR

spec: specs/{service}/{file}.md
```

| Тип | Назначение |
|-----|-----------|
| `feat` | Новая функциональность (обязана иметь spec) |
| `fix` | Исправление бага |
| `refactor` | Изменение структуры без изменения поведения |
| `docs` | Только изменения в spec-файлах или документации |
| `test` | Добавление или изменение тестов |
| `chore` | Инфраструктура, зависимости, конфигурация |

Примеры:
```
feat(semantic_graph): мягкое удаление сущностей
spec: specs/semantic_graph/soft-delete.md

fix(app): утечка соединений в mineru_client
refactor(documet_index): выделение BaseNeo4jManager
docs(app): SERVICE.md — все эндпоинты и конфигурация
```

#### Правила

1. **Один коммит — одно логическое изменение.** Не смешивать рефакторинг и новую фичу, документирование и исправление.
2. **Spec в том же коммите.** Если изменение затрагивает спецификацию, spec-файл коммитится вместе с кодом.
3. **Ссылка на spec в теле.** Если изменение связано со спецификацией, в теле коммита указывается `spec: {путь}`.
4. **Язык.** Русский или английский. Заголовок коммита на том же языке, что и описание PR.
5. **Merge-коммиты.** Стандартные сообщения git/GitHub, формат не применяется.
6. **Squash.** При squash-merge в main заголовок коммита = заголовок PR. Тело = описание PR + ссылки на spec-файлы.

---

## 8. Интеграционные контракты

### I1. app → semantic_graph

```
POST http://semantic_graph:9595/process-document
Body: {"document_id": "{file_hash}"}
Response: {"status": "success", "entities_created": N, ...}

GET http://semantic_graph:9595/clastrize_graph
GET http://semantic_graph:9595/create_community_report
```

### I2. app → mineru

```
POST http://mineru:8000/process
Body: multipart/form-data {file: PDF, backend, method, lang, ...}
Response: {task_id, ...} → async → content_list JSON
```

### I3. app → Qdrant (direct driver)

Коллекция `documents`, вектор 2048-dim, payload с `file_hash`, `region_id`, `element_type`, `original_element`.

### I4. app → Neo4j (direct driver, через documet_index)

Создание `Document` и `Region:*` узлов через `Manager`.

### I5. semantic_graph → Neo4j (direct driver)

Создание `Entity`, `Community` узлов и связей через `Manager`.

---

## 9. План внедрения

План состоит из трёх треков:
- **Трек A**: документирование существующей системы (пишем specs «как есть», без изменения кода)
- **Трек B**: рефакторинг через specs (устраняем техдолг, код меняется, поведение — нет)
- **Трек C**: новые фичи (требуют spec-first, добавляют новое поведение)

### Этап 1: Принять Constitution

- [ ] Constitution согласован и размещён в `CONSTITUTION.md` в корне репозитория

### Этап 2: Трек A — Service Specs для всех сервисов

- [ ] `app/specs/SERVICE.md` — главное приложение (все эндпоинты, зависимости, конфигурация, модели)
- [ ] `semantic_graph/specs/SERVICE.md` — сервис семантического графа (включая описание утилиты `Qdrant_extractor/` как CLI-инструмента экспорта)
- [ ] `documet_index/specs/SERVICE.md` — сервис документного графа
- [ ] `mineru/specs/SERVICE.md` — сервис обработки PDF

### Этап 3: Трек A — Pipeline Specs для критических пайплайнов

- [ ] `app/specs/pipelines/document-ingestion.md` — пайплайн загрузки PDF (upload → mineru → embeddings → Qdrant → Neo4j → semantic_graph)
- [ ] `app/specs/pipelines/question-answering.md` — вопросно-ответный пайплайн (embed question → search → rerank → LLM answer, включая lazy indexing fallback)
- [ ] `app/specs/pipelines/document-deletion.md` — пайплайн удаления документа (Qdrant + Neo4j document graph + MinIO, текущее состояние)
- [ ] `semantic_graph/specs/pipelines/entity-extraction.md` — извлечение сущностей и связей (Qdrant → LLM extraction → summarization, per-document)
- [ ] `semantic_graph/specs/pipelines/community-detection.md` — кластеризация Leiden + генерация отчётов сообществ (always-full-graph, P9)

### Этап 4: Трек A — Data Model Specs

- [ ] `app/specs/models/mineru-content-list.md` — формат выдачи MinerU
- [ ] `app/specs/models/qdrant-point.md` — схема точки в Qdrant
- [ ] `app/specs/models/question-request.md` — QuestionRequest / QuestionResponse
- [ ] `semantic_graph/specs/models/entity.md` — Entity, EntityCreate, RelationshipCreate
- [ ] `semantic_graph/specs/models/community.md` — Community, CommunityReport
- [ ] `documet_index/specs/models/region.md` — Region, BBox, Style
- [ ] `documet_index/specs/models/document.md` — Document, document graph schema

### Этап 5: Трек A — Integration Specs

- [ ] `app/specs/integrations/app-semantic_graph.md` — контракт app → semantic_graph
- [ ] `app/specs/integrations/app-mineru.md` — контракт app → mineru

### Этап 6: Трек A — LLM Model Requirements

- [ ] В каждый Service Spec добавить секцию с требованиями к LLM-моделям (тип, контекст, язык)
- [ ] В каждый Pipeline/Feature Spec добавить секцию `## LLM Model Requirements`
- [ ] Привести `.env` в актуальное состояние: убрать неиспользуемые модели Ollama, добавить актуальные модели vLLM

### Этап 7: Трек B — Вынос промптов в файлы

- [ ] Создать `semantic_graph/prompts/` директорию
- [ ] Вынести `GRAPH_EXTRACTION_PROMPT` → `prompts/graph-extraction.md`
- [ ] Вынести `SUMMARIZE_PROMPT` → `prompts/entity-summarize.md`
- [ ] Вынести `CONTINUE_PROMPT` + `LOOP_PROMPT` → `prompts/graph-extraction.md` (как часть extraction)
- [ ] Вынести `COMMUNITY_REPORT_PROMPT` → `prompts/community-report.md`
- [ ] Вынести системный промпт Q&A из `app/src/api.py` → `app/prompts/qa-system.md`
- [ ] Реализовать `load_prompt()` и обновить код для загрузки промптов из файлов
- [ ] Обновить Pipeline Specs: добавить ссылки на файлы промптов

### Этап 8: Трек B — Унификация конфигурации

- [ ] Мигрировать `semantic_graph/config.py` на Pydantic Settings (`semantic_graph/config/settings.py`)
- [ ] Перенести константы доменной логики (ENTITY_TYPES, MAX_CLUSTER_SIZE, CLUSTERIZATION_SEED) в Settings
- [ ] Перенести URL-ы (LLM_URL, QDRANT_URL, TOKENIZER_URL) в Settings с env-переменными
- [ ] Удалить старый `config.py` после миграции

### Этап 9: Трек B — Унификация Neo4j Manager'ов

- [ ] Создать `documet_index/specs/pipelines/neo4j-manager-unification.md` — spec на унификацию
- [ ] Выделить общий базовый класс `BaseNeo4jManager` с connection management, query helpers
- [ ] Перевести `documet_index/manager.py` на наследование от `BaseNeo4jManager`
- [ ] Перевести `semantic_graph/manager.py` на наследование от `BaseNeo4jManager`
- [ ] Удалить дублирующийся код

### Этап 10: Трек B — Выпиливание мёртвого кода

- [ ] Удалить `semantic_graph/neo4j_service.py:create_graph_from_graphrag_result()` (метод с неопределёнными переменными)
- [ ] Проверить и удалить остальной неиспользуемый код (неиспользуемые импорты, дублирующиеся файлы типа `requirements.txt` / `req2.txt`)

### Этап 11: Трек C — Мягкое удаление из семантического графа (P10)

- [ ] Создать `semantic_graph/specs/soft-delete.md` — Feature Spec на мягкое удаление
- [ ] Добавить поле `archived: bool = False` в модель Entity
- [ ] Реализовать эндпоинт маркировки сущностей документа как archived
- [ ] Модифицировать `clastrize_graph` — исключать archived-сущности из кластеризации
- [ ] Модифицировать `create_community_report` — исключать archived-сущности из отчётов
- [ ] Интегрировать в `DELETE /documents/{file_hash}` в app — вызывать archived-маркировку

### Этап 12: Настройка CI-валидации

- [ ] Добавить CI job: генерация OpenAPI из FastAPI (`/openapi.json`)
- [ ] Добавить CI job: проверка наличия Feature Spec + OpenAPI-фрагмента для каждого эндпоинта
- [ ] Добавить CI job: сверка method/path/status codes между spec и кодом
- [ ] Добавить CI job: проверка наличия Data Model Spec для каждой Pydantic-модели в `dtype/` (warning на переходный период)
- [ ] Добавить CI job: структурная проверка spec-файлов (все обязательные секции шаблона)

---

## 10. Версионирование Constitution

Constitution следует семантическому версионированию:

- **MAJOR**: изменение принципов (P1-P10) или удаление инвариантов
- **MINOR**: добавление новых секций, шаблонов, правил без изменения существующих
- **PATCH**: исправление опечаток, уточнение формулировок

Изменения Constitution проходят через тот же процесс Review, что и спецификации.

---

## 11. Практики качественного кода

### Q1. Управление ресурсами

Любой ресурс, требующий явного освобождения (соединения с БД, HTTP-сессии, файловые дескрипторы), обязан управляться через контекстный менеджер или гарантированное закрытие.

```python
# Правильно — контекстный менеджер
async with Neo4jConnection(uri, user, password) as conn:
    result = await conn.query("MATCH (n) RETURN n")

# Правильно — гарантированное закрытие
client = QdrantClient(url=url)
try:
    result = client.search(...)
finally:
    client.close()
```

**Правило**: Каждый класс, владеющий внешним ресурсом, реализует `__aenter__`/`__aexit__` (асинхронный) или `__enter__`/`__exit__` (синхронный).

### Q2. Обработка ошибок и устойчивость

#### Retry для transient-ошибок

Сетевые вызовы к внешним сервисам (LLM, базы данных) обязаны иметь retry-логику для transient-ошибок (timeout, connection reset, 429/503):

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def call_llm(messages: list[dict]) -> str:
    ...
```

#### Никаких голых except

```python
# Запрещено
try:
    ...
except:
    pass

# Правильно — ловить конкретные исключения
try:
    ...
except (ConnectionError, TimeoutError) as e:
    logger.error(f"Сетевая ошибка: {e}")
    raise
```

### Q3. Валидация на границах

Все данные, приходящие извне (HTTP-запросы, ответы внешних API, содержимое файлов), проходят валидацию через Pydantic-модели на ближайшей границе системы:

```python
# На входе — сырые данные от MinerU, сразу в Pydantic
class MinerUContentItem(BaseModel):
    type: Literal["text", "image", "table", "equation"]
    text: str | None = None
    bbox: list[float]
    page_idx: int
    ...

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    content = await mineru_client.process(file)
    items = [MinerUContentItem(**item) for item in content["content_list"]]
    # Дальше работаем ТОЛЬКО с Pydantic-моделями
```

### Q4. Идемпотентность операций

Любая операция, которая может быть повторена (повторная загрузка документа, перезапуск пайплайна), должна быть идемпотентной:

- **Neo4j**: `MERGE` вместо `CREATE` — повторная вставка не дублирует узлы
- **Qdrant**: `upsert` вместо `insert` — повторная загрузка обновляет существующие точки
- **MinIO**: проверка существования перед загрузкой — не перезаписываем без необходимости
- **LLM extraction**: дедупликация сущностей по `(title, type)` — повторный прогон не создаёт дубликатов

### Q5. Асинхронная гигиена

```python
# Запрещено — блокирующий вызов в async-контексте
async def process():
    time.sleep(5)  # БЛОКИРУЕТ event loop!

# Правильно — CPU-bound через run_in_executor
async def process():
    await asyncio.get_event_loop().run_in_executor(None, time.sleep, 5)

# Или — I/O-bound через async-библиотеку
async def process():
    await asyncio.sleep(5)
```

**Правила**:
- Никаких `time.sleep()`, `requests.get()`, синхронных драйверов БД в async-функциях
- CPU-bound операции (хэширование, рендеринг PDF) → `run_in_executor`
- Все сетевые вызовы → `aiohttp`, `httpx.AsyncClient`, асинхронные драйверы (neo4j, qdrant-client)

### Q6. Чистые функции и иммутабельность

Предпочитать чистые функции без побочных эффектов для вычислений, не требующих I/O:

```python
# Чистая функция — результат зависит только от аргументов
def parse_entities(llm_output: str, source_id: str) -> pd.DataFrame:
    records = llm_output.split(RECORD_DELIMITER)
    return _build_dataframe(records, source_id)

# Функция с побочным эффектом — явно отделена
async def save_entities_to_neo4j(entities_df: pd.DataFrame, manager: Manager):
    ...
```

Данные, прошедшие валидацию, не мутируются — преобразования создают новые объекты.

### Q7. Логирование

```python
from loguru import logger

# Структурированное логирование с контекстом
logger.info("Начало обработки документа", file_hash=file_hash, pages=len(pages))
logger.error("Ошибка извлечения сущностей", chunk_id=chunk_id, error=str(e))

# Уровни:
# DEBUG   — детали внутренней работы (парсинг промптов, построение запросов)
# INFO    — ключевые точки пайплайна (начало/конец этапа, количество результатов)
# WARNING — recoverable проблемы (retry, fallback, degraded mode)
# ERROR   — ошибки, требующие внимания (сбой этапа, недоступность сервиса)
```

Каждый запрос/пайплайн получает уникальный идентификатор трассировки, который передаётся во все логи:

```python
import uuid
trace_id = str(uuid.uuid4())[:8]
logger.bind(trace_id=trace_id).info("Запрос начат", question=question)
```

### Q8. Внедрение зависимостей

Избегать глобальных singleton'ов. Зависимости передаются явно через конструктор:

```python
# Правильно — зависимости явные
class AsyncGraphExtractor:
    def __init__(self, llm_client: AsyncLLMClient, model: str):
        self.llm_client = llm_client
        self.model = model

# Неправильно — глобальный singleton
LLM_CLIENT = AsyncLLMClient(...)  # модульный уровень

class AsyncGraphExtractor:
    def extract(self, text: str):
        return LLM_CLIENT.generate(text)  # неявная зависимость
```

### Q9. Типизация

Все публичные функции и методы обязаны иметь аннотации типов:

```python
# Правильно
async def extract(
    self,
    text: str,
    entity_types: list[str],
    source_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...

# Неправильно
async def extract(self, text, entity_types, source_id):
    ...
```

### Q10. Тестирование контрактов, а не реализации

Тесты проверяют поведение (вход → выход), а не внутреннее устройство:

```python
# Правильно — тестируем контракт
def test_entity_parsing():
    output = '("entity"<|>Alice<|>PERSON<|>Инженер)##("entity"<|>Bob<|>PERSON<|>Дизайнер)'
    entities, rels = parse_result(output, "doc-1")
    assert len(entities) == 2
    assert entities.iloc[0]["title"] == "Alice"

# Неправильно — тестируем внутренности
def test_internal_regex():
    assert ENTITY_PATTERN.match("...")  # тест зависит от детали реализации
```
