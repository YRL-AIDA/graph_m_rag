# Service: documet_index

## Identity
- **Зона ответственности**: Граф структуры документа — порядок чтения (`ORDER`) и иерархия заголовков (`PARENT`) для элементов PDF-документов, полученных из MinerU.
- **Порт**: — (библиотека, не HTTP-сервис)
- **Хранилище**: Neo4j (document graph) — узлы `Document`, `Region:*`; связи `ORDER`, `PARENT`

## Dependencies
| Сервис | Протокол | Назначение |
|--------|----------|------------|
| Neo4j  | Bolt (драйвер `neo4j`) | Хранение и запрос графа структуры документа |

## API

`documet_index` — библиотека, не HTTP-сервис. Публичный интерфейс состоит из классов и функций, экспортируемых из `__init__.py`.

### Классы

#### `DocumentIndexService` (верхнеуровневый сервисный класс)
| Метод | Назначение |
|-------|------------|
| `__init__(uri=None, user=None, password=None, name_db=None)` | Инициализация: подтягивает параметры подключения из переменных окружения (`URL`, `USER_NEO4J`, `PASSWORD`, `NAME_DB`), создаёт `Manager` |
| `create_graph_from_mineru_result(mineru_result, file_hash) -> bool` | Принимает JSON-результат MinerU и `file_hash`, строит граф в Neo4j. Возвращает `True` при успехе, `False` если документ уже существует |
| `delete_graph(file_hash) -> bool` | Удаляет граф документа по `file_hash` |
| `is_document_indexed(file_hash) -> bool` | Проверяет, проиндексирован ли документ |
| `get_status() -> Dict[str, Any]` | Возвращает статус БД (количество узлов) |
| `get_related_context(file_hash, element_type, text) -> Dict[str, Any]` | Получает связанный контекст для региона: для `image_caption`/`image_footnote` — родительское изображение; для `table_caption`/`table_footnote` — родительскую таблицу; для `image`/`table` — связанные caption/footnote |
| `delete_all_graphs() -> bool` | Удаляет все графы из БД |
| `close()` | Закрывает соединение с Neo4j |
| `__enter__` / `__exit__` | Поддержка context manager |

#### `Manager` (низкоуровневый менеджер Neo4j-операций)
| Метод | Назначение |
|-------|------------|
| `add_document(document: Document) -> bool` | Сериализует `Document.get_graph()` в Cypher-запрос и выполняет его (CREATE узлов и связей). Возвращает `True` при добавлении, `False` если документ уже существует |
| `delete_document(name: str) -> bool` | Удаляет документ и все связанные узлы/связи через MATCH/DELETE |
| `delete_all_documents() -> bool` | DETACH DELETE всех узлов в БД |
| `is_document_exist(name: str) -> bool` | Проверяет существование документа через OPTIONAL MATCH |
| `get_related_context(file_hash, element_type, text) -> Dict` | Neo4j-запросы для получения связанного контекста региона |
| `status() -> dict` | `MATCH (n) RETURN count(n)` — количество узлов в БД |
| `query(query: str) -> list` | Выполнение произвольного Cypher-запроса |
| `close()` | Закрытие Neo4j-соединения |
| `__enter__` / `__exit__` | Поддержка context manager |

#### Вспомогательные классы
| Класс | Назначение |
|-------|------------|
| `ManagerConfig` | Конфигурация: `uri`, `user`, `password`, `name_db` |
| `Neo4jConnection` | Тонкая обёртка над `neo4j.GraphDatabase.driver` — `query()` и `close()` |
| `Document` | Представление документа с парсингом MinerU-результата, свойством `regions` и методом `get_graph()` |
| `Region` | Элемент контента документа (текст, изображение, таблица…) |
| `BBox` | Координаты bounding box |
| `Style` | Стилевые атрибуты (font_size, error_rate) с оператором `<` |

### Функции
| Функция | Назначение |
|---------|------------|
| `create_neo4j_graph(mineru_result, file_hash) -> bool` | Удобная функция: создаёт `DocumentIndexService` и вызывает `create_graph_from_mineru_result`, затем закрывает соединение |
| `create_graph_from_mineru_result(mineru_result, document_name) -> List[Region]` | Пайплайн в `dtype/document.py`: парсит MinerU JSON в список `Region`-объектов (см. Pipeline Spec) |

## Data Model

- [models/document.md](models/document.md) — модель `Document`: структура, `get_graph()`, построение графа из MinerU-результата
- [models/region.md](models/region.md) — модели `Region`, `BBox`, `Style`: атрибуты, `is_content()`, оператор сравнения стилей, `to_dict()`

Схема Neo4j (Document Graph):
- **Узлы**:
  - `Document` — `{name: file_hash}`
  - `Region:{label}` — мульти-лейбл (один из: `title`, `text`, `image`, `image_caption`, `image_footnote`, `table`, `table_caption`, `table_footnote`, `equation`, `header`, или произвольный тип). Свойства: `text`, `image`, `bbox` (JSON), `style` (JSON), `order`, `element_data`
- **Связи**:
  - `(:Document) -[:ORDER]-> (:Region)` — первый элемент порядка чтения
  - `(:Region) -[:ORDER]-> (:Region)` — цепочка порядка чтения
  - `(:Document) -[:PARENT]-> (:Region:{title})` — заголовок верхнего уровня как дочерний документу
  - `(:Region:{title}) -[:PARENT]-> (:Region)` — иерархия заголовков и контента

## Pipelines

- [pipelines/document-to-graph.md](pipelines/document-to-graph.md) — пайплайн преобразования MinerU JSON → граф документа: парсинг `content_list`, создание `Region`-объектов для всех типов элементов, построение ORDER/PARENT рёбер через обход стилей заголовков.

## Configuration

| Переменная окружения | Назначение | Значение по умолчанию |
|----------------------|------------|------------------------|
| `URL`                | Neo4j хост:порт (добавляется префикс `neo4j://`) | `localhost:7687` |
| `USER_NEO4J`         | Имя пользователя Neo4j | `neo4j` |
| `PASSWORD`           | Пароль Neo4j | `''` (пустая строка) |
| `NAME_DB`            | Имя базы данных Neo4j | `neo4j` |

Конфигурация загружается через `python-dotenv` (файл `.env`). Класс `ManagerConfig` хранит параметры подключения.

## Invariants

1. **Идентификация документа**: `Document.name = file_hash` (MD5 содержимого PDF) — используется как имя узла `Document` и первичный ключ (P2).
2. **Идентификация региона**: комбинация `file_hash|region_id` связывает Region в Neo4j с точкой в Qdrant (P3). Регионы нумеруются сквозным индексом при построении графа.
3. **Изоляция графов**: запросы к document graph не затрагивают узлы/связи semantic graph (P4). `DELETE ALL` удаляет все узлы — допустимо только в dev/test, не в production с сосуществующими графами.
4. **Линейный порядок чтения**: `ORDER`-связи образуют единую цепочку от `Document` через все `Region`-узлы в порядке их следования в `content_list`.
5. **Иерархия заголовков**: `PARENT`-связи строятся на основе стилей (`Style.font_size`): заголовок с большим шрифтом становится родителем последующих заголовков с меньшим шрифтом; контентные элементы (`text`, `table`, `image`) привязываются к текущему заголовку.
6. **Мульти-лейбл регионов**: узел `Region` всегда имеет оба лейбла — `Region` и конкретный тип (например, `Region:image_caption`), что позволяет запрашивать как все регионы (`MATCH (r:Region)`), так и фильтровать по типу (`MATCH (r:Region:image_caption)`).
7. **Content labels**: `CONTENT_LABELS = ['text', 'table', 'image']` — только эти типы считаются контентными (`is_content() == True`), остальные (`title`, `image_caption`, `table_caption`, `table_footnote`, `equation`, `header`) — структурные.
8. **Дедупликация**: метод `add_document` проверяет существование документа перед созданием — повторный вызов с тем же `file_hash` возвращает `False`.

## Exceptions

1. **Нарушение P6 (Pydantic)**: `Document`, `Region`, `Style`, `BBox`, `ManagerConfig` реализованы как plain Python классы с `__init__`, а не Pydantic-модели. Валидация типов на уровне конструкторов отсутствует. **Обоснование**: переходный период — код написан до принятия Constitution. Должно быть исправлено при рефакторинге: преобразовать в Pydantic `BaseModel` с валидаторами полей.

2. **Нарушение P7 (промпты в файлах)**: отсутствует директория `prompts/`. **Обоснование**: сервис не использует LLM — промпты не требуются. Отклонение допустимо.

3. **Нарушение P5 (асинхронность)**: Neo4j-драйвер используется синхронно (`session.run()`, блокирующие вызовы). **Обоснование**: драйвер `neo4j` поддерживает и sync, и async API — текущий код использует sync. При рефакторинге должен быть переход на `AsyncGraphDatabase`.

4. **Нарушение N2 (структура модулей)**: директория `dtype/` не содержит Pydantic-моделей; отсутствует `config/settings.py` (конфигурация — plain Python классы через `.env` и `dotenv`); отсутствует `src/`. **Обоснование**: код написан до принятия Constitution. Должно быть исправлено.

5. **Нарушение N2 (отсутствие `specs/`)**: директория `specs/` создаётся данным документом — переходный период (5.3).

6. **Нарушение P3 (сквозной region_id)**: идентификатор региона в коде — порядковый индекс при обходе `content_list`, а не композитный `file_hash|region_id`. Связь с Qdrant через region_id не формализована в этом сервисе (это ответственность `app`). **Обоснование**: `documet_index` не взаимодействует с Qdrant напрямую — сквозная идентификация обеспечивается на уровне оркестратора `app`.
