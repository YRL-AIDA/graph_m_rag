# Feature: ExperimentRunner — оркестратор, метрики, конфигурация и CLI

## Motivation
Оркестратор экспериментов E1–E6: загружает датасеты, создаёт клиентов моделей, запускает NER+RE на каждой записи, вычисляет метрики (NER F1, RE F1, время), сохраняет результаты (JSON + Markdown summary). CLI на `argparse` для фильтрации по эксперименту, датасету, сплиту и лимиту. Фильтрация `--experiment` происходит ДО выполнения `run_all()`, чтобы не запускать лишние эксперименты.

## Behaviour

### Config (`runner/config.py`)

```python
class ExperimentSettings(BaseSettings):
    qwen_base_url: str = "http://192.168.19.127:8888/v1"
    qwen_api_key: str = "EMPTY"
    qwen_model: str = "Qwen/Qwen3-VL-32B-Thinking"
    uniner_base_url: str = "http://192.168.19.127:9898/v1"
    gliner_base_url: str = "http://192.168.19.127:9899/v1"
    gliner_api_key: str = "EMPTY"
    gliner_model: str = "urchade/gliner_large-v2.1"
    scierc_data_path: str = "/home/ivan/work/ooo/graph_m_rag/datasets_/scierc"
    prompts_dir: str = str(Path(__file__).parent.parent / "prompts")
    results_dir: str = str(Path(__file__).parent.parent / "results")
    datasets: list[str] = ["conll04", "scierc"]
    splits: list[str] = ["test"]
    max_samples: int | None = None

    class Config:
        env_prefix = "EXPERIMENT_"
        env_file = ".env"
```

Все поля переопределяются через env-переменные с префиксом `EXPERIMENT_`. `max_samples` — лимит записей на датасет (`None` = без ограничений). `prompts_dir` — директория с файлами промптов (`ner_prompt.md`, `re_prompt.md`, `combined_prompt.md`); используется для разрешения путей при создании `QwenClient` (см. ``run_all`` — алгоритм).

### ExperimentRunner (`runner/experiment_runner.py`)

```python
class ExperimentRunner:
    def __init__(self, settings: ExperimentSettings): ...

    async def run_experiment(
        self,
        experiment_id: str,       # "E1"–"E6"
        ner_client: BaseModelClient,
        re_client: BaseModelClient | None,
        re_mode: str,             # "combined_single_call" | "separate" | "none"
        dataset_name: str,
        split: str,
    ) -> ExperimentResult: ...

    async def run_all(self, experiment_filter: str | None = None) -> list[ExperimentResult]: ...
    def save_results(self, results: list[ExperimentResult]) -> Path: ...
```

Типы (`BaseModelClient`, `ExperimentResult`, `PredictedEntity`, `PredictedRelation`, `Entity`, `Relation`, `DatasetRecord`) — из `general_experiment.md`.

### `run_experiment` — алгоритм

Для каждого `DatasetRecord` (с учётом `max_samples`):

1. **NER**:
   - Если `re_mode == "combined_single_call"` (E3): `t0=time()`; `entities, relations = await ner_client.extract_entities_and_relations(text, entity_types, relation_types)`; `t1=time()`. Оба списка (`pred_entities`, `pred_relations`) получаются из одного API-вызова Qwen. Отдельные вызовы `extract_entities` и `extract_relations` НЕ делаются.
   - Иначе: `t0=time()`; `await ner_client.extract_entities(text, entity_types)`; `t1=time()`.

2. **RE**:
   - `"separate"` → `await re_client.extract_relations(text, entities, relation_types)`, `t2=time()`;
   - `"combined_single_call"` → уже выполнено в шаге 1 (NER);
   - `"none"` (E1, E2) → пропускается.

3. **Метрики**: `compute_ner_f1(gold, pred)` и `compute_re_f1(gold_entities, gold_relations, pred_entities, pred_relations)`.

4. **Агрегация**: средние precision/recall/F1/время по всем записям + `ner_entity_count`, `total_samples`, `failed_samples` → `ExperimentResult`.

### `run_all` — алгоритм

Создаёт клиенты моделей (`QwenClient`, `UniNerClient`, `GleanerClient`, `OllamaClient`, `HybridClient`).

Создаёт клиенты моделей (`QwenClient`, `UniNerClient`, `GleanerClient`, `HybridClient`). При создании `QwenClient` пути к файлам промптов разрешаются из `settings.prompts_dir`:

```python
prompts = Path(self.settings.prompts_dir)
qwen_client = QwenClient(
    base_url=self.settings.qwen_base_url,
    api_key=self.settings.qwen_api_key,
    model=self.settings.qwen_model,
    ner_prompt_path=str(prompts / "ner_prompt.md"),
    re_prompt_path=str(prompts / "re_prompt.md"),
    combined_prompt_path=str(prompts / "combined_prompt.md"),
)
ollama_client = OllamaClient(
    base_url=self.settings.ollama_base_url,
    model=self.settings.ollama_model,
    think=self.settings.ollama_think,
    num_predict=self.settings.ollama_num_predict,
    num_ctx=self.settings.ollama_num_ctx,
    ner_prompt_path=str(prompts / "ner_prompt.md"),
    re_prompt_path=str(prompts / "re_prompt.md"),
    combined_prompt_path=str(prompts / "combined_prompt.md"),
)
``` Формирует матрицу экспериментов:

- E1: UniNer NER, `re_mode="none"`
- E2: Gleaner NER, `re_mode="none"`
- E3: Qwen combined_single_call — `extract_entities_and_relations()` (один вызов API)
- E4: Qwen separate NER + Qwen RE, `re_mode="separate"`
- E5: Hybrid(UniNer NER + Qwen RE), `re_mode="separate"`
- E6: Hybrid(Gleaner NER + Qwen RE), `re_mode="separate"`

**Если передан `experiment_filter` — матрица фильтруется ДО выполнения**:

```python
if experiment_filter is not None:
    experiments = [(eid, ner, re, mode) for (eid, ner, re, mode) in experiments if eid == experiment_filter]
```

Клиенты моделей создаются только для экспериментов, которые реально будут запущены (после фильтрации). Затем итерация по отфильтрованной матрице × `datasets` × `splits` → `run_experiment` для каждой комбинации, затем `save_results`.

### Metrics (`metrics/metrics.py`)

**`compute_ner_f1(gold: list[Entity], pred: list[PredictedEntity]) → dict[str, float]`**

Сравнение по нормализованным `(name, type)` парам с использованием `normalize_name()` и `normalize_type()` из `general_experiment.md`:
- `gold.name` и `pred.name` нормализуются через `normalize_name()`.
- `gold.type` и `pred.type` нормализуются через `normalize_type()` с `allowed_types=entity_types` датасета.
- Совпадение — только если оба нормализованных значения равны.
- Жадный алгоритм: одна gold-сущность на одну pred-сущность.

TP = число совпадений. Precision = TP / len(pred), Recall = TP / len(gold). `pred` пуст → P=0; `gold` пуст → R=0. F1 = 2PR/(P+R) при P+R > 0, иначе 0.

**`compute_re_f1(gold_entities, gold_relations, pred_entities, pred_relations) → dict[str, float]`**

1. **Entity matching**: gold-сущности и pred-сущности сопоставляются по нормализованным `(name, type)` парам (через `normalize_name()` и `normalize_type()`), жадный алгоритм.
2. **RE TP**: gold-отношение `(head_idx, tail_idx, type)` найдено, если головная и хвостовая gold-сущности сматчены с pred-сущностями `P_head`, `P_tail` и существует pred-отношение `(head=P_head.name, tail=P_tail.name, type=gold.type)`. Сравнение типов отношений — через `normalize_type()` с `allowed_types=relation_types`.
3. Precision = TP_RE / len(pred_relations), Recall = TP_RE / len(gold_relations). F1 = 2PR/(P+R) при P+R > 0, иначе 0.

**`compute_avg_response_time(timings: list[float]) → float`**: среднее в секундах.

**`ner_entity_count`**: среднее количество предсказанных NER-сущностей на запись (после фильтрации типов через `normalize_type`). Вычисляется как `sum(len(pred_entities) for each record) / total_samples`.

### ExperimentResult.metrics

Поля `metrics: dict[str, float]`, заполняемые раннером:

| Поле | Описание |
|------|----------|
| `precision_ner` | NER precision |
| `recall_ner` | NER recall |
| `f1_ner` | NER F1 |
| `precision_re` | RE precision |
| `recall_re` | RE recall |
| `f1_re` | RE F1 |
| `ner_entity_count` | Среднее количество предсказанных сущностей на запись |
| `avg_response_time_sec` | Среднее время ответа модели (сек) |
| `total_samples` | Количество обработанных записей |
| `failed_samples` | Количество неудачных записей (timeout / connection error) |

### Save results
- `results/<timestamp>_experiments.json` — список `ExperimentResult`.
- `results/<timestamp>_summary.md` — таблица: F1 и время на эксперимент/датасет.

### CLI
```bash
python -m semantic_graph.tests.entity_extraction_test.runner.experiment_runner
python -m ... --experiment E1 --dataset conll04 --split test --max-samples 50
```

**`--experiment` фильтрация**: аргумент `--experiment` передаётся в `run_all(experiment_filter="E1")`. Внутри `run_all()` матрица экспериментов фильтруется до выполнения — запускаются только указанные эксперименты, а не все E1–E6 с постфактум-фильтрацией. Клиенты моделей создаются только для экспериментов, прошедших фильтр.

Приоритет: CLI-аргумент `--experiment` управляет фильтрацией напрямую, env-переменные для experiment_filter не используются. Остальные настройки — env-переменные (`EXPERIMENT_*`) приоритетнее CLI-аргументов. `argparse`.

## Data Flow
См. `general_experiment.md` §Data Flow. Раннер — точка входа: `DatasetLoader.load()` → `BaseModelClient.extract_*()` → `compute_*_f1()` → `save_results()`.

## LLM Interactions
Раннер напрямую с LLM не взаимодействует. Клиенты используют `prompts/ner_prompt.md` и `prompts/re_prompt.md`. Для E3 (`combined_single_call`) QwenClient использует `prompts/combined_prompt.md`. Требования — в `general_experiment.md` §LLM Interactions.

## LLM Model Requirements
Требования к моделям — в `general_experiment.md`. Раннер дополнительных ограничений не накладывает.

## Error Handling

| Сценарий | Поведение |
|----------|-----------|
| Датасет не загружен (файл отсутствует / HF offline) | `RuntimeError`, лог, датасет пропускается |
| Model timeout / connection error | `ConnectionError`, запись неудачна, инкремент `failed_samples`, метрики по успешным |
| Unparseable model response | Пустой `[]`, warning |
| Пустой датасет после фильтрации | Warning, комбинация пропускается |
| `max_samples > len(dataset)` | `min(max_samples, len(dataset))` |

## Testing

- **`test_experiment_config.py`**: env-var `EXPERIMENT_QWEN_BASE_URL=http://x:1/v1` → `settings.qwen_base_url == "http://x:1/v1"`; defaults корректны.
- **`test_metrics.py`**: `compute_ner_f1` — идеальный match → P=R=F1=1.0; промах → 1.0/0.5/0.6667; пустой gold+FP → 0/0/0; type mismatch → P=0.0. `compute_re_f1` — overlap ≥0.5 → P=1.0; overlap <0.5 → F1=0.0; пустые отношения → F1=0.0. Нормализация имён и типов через `normalize_name()`/`normalize_type()`.
- **`test_runner_mock.py`**: полный цикл E1–E6 с замоканными клиентами → валидный `ExperimentResult` + валидный JSON. Проверка, что `--experiment E1` запускает только E1, а не все эксперименты.

Все LLM-вызовы мокаются (Constitution T3). CI: `pytest semantic_graph/tests/entity_extraction_test/tests/`.

## Dependencies
- **Внутренние**: `general_experiment.md` (типы); `conll04_loader.py`, `scierc_loader.py`; `qwen_client.py`, `uniner_client.py`, `gliner_client.py`, `hybrid_client.py`.
- **Внешние**: `pydantic>=2.0`, `pydantic-settings>=2.0`, `openai>=1.0`, `pytest>=7.0`, `pytest-asyncio>=0.21`.

## Exceptions
Отклонения от Constitution — см. `general_experiment.md` §Exceptions. Данный файл не вводит дополнительных отклонений.
