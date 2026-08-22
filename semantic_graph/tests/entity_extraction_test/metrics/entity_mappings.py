"""Universal entity type mapping for cross-dataset NER/RE evaluation."""
from __future__ import annotations

from typing import Optional

UNIVERSAL_ENTITY_TYPES: list[str] = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC",
    "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT",
    "QUANTITY", "TIME", "WORK_OF_ART",
]

DATASET_TO_UNIVERSAL: dict[str, dict[str, Optional[str]]] = {
    "ontonotes5": {t: t for t in UNIVERSAL_ENTITY_TYPES},
    "conll04": {"Peop": "PERSON", "Org": "ORG", "Loc": "LOC", "Other": None},
    "scierc": {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "ORGANIZATION|PERSON": None,
        "Task": None,
        "Method": None,
        "Material": "PRODUCT",
        "Metric": "QUANTITY",
        "Generic": None,
        "OtherScientificTerm": None,
    },
}

UNIVERSAL_TO_DATASET: dict[str, dict[str, Optional[str]]] = {
    "ontonotes5": {t: t for t in UNIVERSAL_ENTITY_TYPES},
    "conll04": {"PERSON": "Peop", "ORG": "Org", "LOC": "Loc", "GPE": "Loc"},
    "scierc": {"ORG": "ORGANIZATION", "PERSON": "PERSON", "PRODUCT": "Material", "QUANTITY": "Metric"},
}

_DATASET_ENTITY_TYPES: dict[str, list[str]] = {
    "ontonotes5": list(UNIVERSAL_ENTITY_TYPES),
    "conll04": ["Peop", "Loc", "Org", "Other"],
    "scierc": [
        "Task", "Method", "Material", "Metric", "Generic",
        "OtherScientificTerm", "ORGANIZATION", "PERSON", "ORGANIZATION|PERSON",
    ],
}


def get_dataset_entity_types(dataset_name: str) -> list[str]:
    """Return the canonical entity type list for a dataset name."""
    try:
        return list(_DATASET_ENTITY_TYPES[dataset_name])
    except KeyError as exc:
        raise ValueError(f"Неизвестный датасет: {dataset_name}") from exc


def _lookup(mapping: dict[str, Optional[str]], typ: str) -> Optional[str]:
    """Case-insensitive lookup with exact-match fallback."""
    if typ is None:
        return None
    key = typ.strip()
    for k, v in mapping.items():
        if k.lower() == key.lower():
            return v
    return mapping.get(key)


def translate_model_to_universal(types: list[str], source_dataset: str) -> list[Optional[str]]:
    """Translate model output types (source-dataset vocabulary) to universal types."""
    mapping = DATASET_TO_UNIVERSAL[source_dataset]
    return [_lookup(mapping, t) for t in types]


def translate_universal_to_dataset(types: list[Optional[str]], eval_dataset: str) -> list[Optional[str]]:
    """Translate universal types to the evaluation dataset vocabulary."""
    mapping = UNIVERSAL_TO_DATASET[eval_dataset]
    return [None if t is None else _lookup(mapping, t) for t in types]
