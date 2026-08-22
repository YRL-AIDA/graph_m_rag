"""Tests for metrics.entity_mappings."""
import pytest

from metrics.entity_mappings import (
    UNIVERSAL_ENTITY_TYPES,
    DATASET_TO_UNIVERSAL,
    UNIVERSAL_TO_DATASET,
    get_dataset_entity_types,
    translate_model_to_universal,
    translate_universal_to_dataset,
)


def test_universal_entity_types_has_18():
    assert len(UNIVERSAL_ENTITY_TYPES) == 18
    assert "PERSON" in UNIVERSAL_ENTITY_TYPES
    assert "GPE" in UNIVERSAL_ENTITY_TYPES


def test_ontonotes5_identity():
    assert DATASET_TO_UNIVERSAL["ontonotes5"]["PERSON"] == "PERSON"
    assert UNIVERSAL_TO_DATASET["ontonotes5"]["PERSON"] == "PERSON"


def test_conll04_forward():
    f = DATASET_TO_UNIVERSAL["conll04"]
    assert f["Peop"] == "PERSON"
    assert f["Org"] == "ORG"
    assert f["Loc"] == "LOC"
    assert f["Other"] is None


def test_conll04_reverse():
    r = UNIVERSAL_TO_DATASET["conll04"]
    assert r["PERSON"] == "Peop"
    assert r["ORG"] == "Org"
    assert r["LOC"] == "Loc"
    assert r["GPE"] == "Loc"


def test_scierc_forward():
    f = DATASET_TO_UNIVERSAL["scierc"]
    assert f["ORGANIZATION"] == "ORG"
    assert f["PERSON"] == "PERSON"
    assert f["Material"] == "PRODUCT"
    assert f["Metric"] == "QUANTITY"
    assert f["Task"] is None
    assert f["ORGANIZATION|PERSON"] is None


def test_get_dataset_entity_types():
    assert get_dataset_entity_types("conll04") == ["Peop", "Loc", "Org", "Other"]
    assert len(get_dataset_entity_types("ontonotes5")) == 18
    with pytest.raises(ValueError):
        get_dataset_entity_types("unknown")


def test_translate_model_to_universal():
    assert translate_model_to_universal(["Peop", "Loc", "Other"], "conll04") == ["PERSON", "LOC", None]
    assert translate_model_to_universal(["person", "GPE"], "ontonotes5") == ["PERSON", "GPE"]


def test_translate_universal_to_dataset():
    assert translate_universal_to_dataset(["PERSON", "GPE", None], "conll04") == ["Peop", "Loc", None]
    assert translate_universal_to_dataset(["ORG", "PERSON"], "scierc") == ["ORGANIZATION", "PERSON"]
