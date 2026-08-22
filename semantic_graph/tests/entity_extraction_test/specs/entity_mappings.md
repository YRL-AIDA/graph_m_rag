# Универсальный маппинг типов сущностей (entity_mappings)

## Цель
Кросс-датасетный перевод типов сущностей в evaluation-фреймворке entity_extraction_test через универсальный набор OntoNotes5.

## Универсальный набор (18 типов)
CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART

## Цепочка перевода
типы модели (= типы source-датасета) -> универсальные (DATASET_TO_UNIVERSAL[source]) -> типы оценочного датасета (UNIVERSAL_TO_DATASET[eval]) -> метрики. Gold не переводится.

## Маппинги
- ontonotes5: identity обе стороны.
- conll04: forward {Peop->PERSON, Org->ORG, Loc->LOC, Other->None}; reverse {PERSON->Peop, ORG->Org, LOC->Loc, GPE->Loc}, остальные -> None.
- scierc: forward {ORGANIZATION->ORG, PERSON->PERSON, ORGANIZATION|PERSON->None, Task->None, Method->None, Material->PRODUCT, Metric->QUANTITY, Generic->None, OtherScientificTerm->None}; reverse {ORG->ORGANIZATION, PERSON->PERSON, PRODUCT->Material, QUANTITY->Metric}, остальные -> None.
None = тип не отображается, сущность исключается из перевода/оценки.

## Новый модуль metrics/entity_mappings.py
Константы и функции:
- UNIVERSAL_ENTITY_TYPES: list[str] (18 типов).
- DATASET_TO_UNIVERSAL: dict[str, dict[str, str | None]] — forward по имени датасета.
- UNIVERSAL_TO_DATASET: dict[str, dict[str, str | None]] — reverse по имени датасета.
- get_dataset_entity_types(dataset_name: str) -> list[str] — возвращает список типов сущностей датасета, реиспользуя онтологии загрузчиков (conll04/scierc/ontonotes5).
- translate_model_to_universal(types, source_dataset) -> list[str | None] — перевод типов модели в универсальные через DATASET_TO_UNIVERSAL[source].
- translate_universal_to_dataset(types, eval_dataset) -> list[str | None] — перевод универсальных в типы оценочного датасета через UNIVERSAL_TO_DATASET[eval].
Значения None отбрасываются вместе с соответствующей сущностью.

## Конфигурация (runner/config.py, ExperimentSettings)
Добавить поля (по умолчанию "ontonotes5"):
- uniner_entity_types_source: str = "ontonotes5"
- gliner_entity_types_source: str = "ontonotes5"
- qwen_entity_types_source: str = "ontonotes5"
- ollama_entity_types_source: str = "ontonotes5"
Имя указывает датасет, из которого берутся типы сущностей модели и маппинг source->universal.

## Клиенты (clients/*.py)
При инициализации клиент получает имя source-датасета, загружает список типов (для промпта) через get_dataset_entity_types и DATASET_TO_UNIVERSAL[source]. HybridClient: NER-часть использует source NER-модели (uniner/gliner), RE-часть — source RE-модели (qwen).

## Раннер (runner/experiment_runner.py)
Проброс source-датасетов клиентам при создании; при вычислении метрик для оценочного датасета — перевод pred: source->universal->eval.

## Метрики (metrics/metrics.py)
Интеграция перевода типов в compute_ner_f1/compute_re_f1. normalize_type/TYPE_SYNONYMS обобщить на универсальное пространство: person/people/per->PERSON, org/company/organisation->ORG, loc/place->LOC, gpe/geo->GPE, misc/other->None и т.п.

## Согласование RE-ограничений
Типы сущностей в RE-описаниях ограничений (relation descriptions) берутся из того же source-датасета, что и NER-типы, во избежание рассинхрона.

## Тесты
Обновить tests/test_metrics.py, tests/test_runner_mock.py, tests/test_experiment_config.py; добавить тесты маппингов (forward/reverse/None), get_dataset_entity_types, translate_model_to_universal/translate_universal_to_dataset.
