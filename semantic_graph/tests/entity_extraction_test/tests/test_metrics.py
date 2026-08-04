"""Тесты для metrics/metrics.py — чистые функции, без LLM-вызовов (Constitution T3)."""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем entity_extraction_test/ в sys.path для импорта testdata и metrics
sys.path.insert(0, str(Path(__file__).parent.parent))

from testdata.base_loader import Entity, PredictedEntity, PredictedRelation, Relation
from metrics.metrics import compute_avg_response_time, compute_ner_f1, compute_re_f1


# ═══════════════════════════════════════════════════════════════════════════
# compute_ner_f1
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_ner_f1_perfect_match() -> None:
    """Gold=[Apple/ORG], Pred=[Apple/ORG] → P=1.0, R=1.0, F1=1.0."""
    gold = [Entity(name="Apple", type="ORG", start=0, end=4)]
    pred = [PredictedEntity(name="Apple", type="ORG")]

    result = compute_ner_f1(gold, pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_compute_ner_f1_partial_match() -> None:
    """2 gold, 1 pred matching one → P=1.0, R=0.5, F1≈0.6667."""
    gold = [
        Entity(name="Apple", type="ORG", start=0, end=4),
        Entity(name="Google", type="ORG", start=10, end=15),
    ]
    pred = [PredictedEntity(name="Apple", type="ORG")]

    result = compute_ner_f1(gold, pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert abs(result["f1"] - 2.0 / 3.0) < 0.001  # ≈0.6667


def test_compute_ner_f1_empty_pred() -> None:
    """Gold has entities, pred empty → P=0.0, R=0.0, F1=0.0."""
    gold = [Entity(name="Apple", type="ORG", start=0, end=4)]
    pred: list[PredictedEntity] = []

    result = compute_ner_f1(gold, pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_compute_ner_f1_empty_gold() -> None:
    """Gold empty, pred has entities → P=0.0, R=0.0, F1=0.0."""
    gold: list[Entity] = []
    pred = [PredictedEntity(name="Apple", type="ORG")]

    result = compute_ner_f1(gold, pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_compute_ner_f1_type_mismatch() -> None:
    """Gold=[Apple/ORG], Pred=[Apple/PERSON] → P=0.0, R=0.0, F1=0.0."""
    gold = [Entity(name="Apple", type="ORG", start=0, end=4)]
    pred = [PredictedEntity(name="Apple", type="PERSON")]

    result = compute_ner_f1(gold, pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_re_f1
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_re_f1_overlap_match() -> None:
    """2 gold entities with a relation, pred entities match by name, pred relations match → P=1.0, R=1.0, F1=1.0."""
    gold_entities = [
        Entity(name="Apple", type="ORG", start=0, end=4),
        Entity(name="Cupertino", type="LOC", start=20, end=28),
    ]
    gold_relations = [
        Relation(
            head_idx=0, tail_idx=1,
            head_start=0, head_end=4,
            tail_start=20, tail_end=28,
            type="located_in",
        ),
    ]
    pred_entities = [
        PredictedEntity(name="Apple", type="ORG"),
        PredictedEntity(name="Cupertino", type="LOC"),
    ]
    pred_relations = [
        PredictedRelation(head="Apple", tail="Cupertino", type="located_in"),
    ]

    result = compute_re_f1(
        gold_entities, gold_relations, pred_entities, pred_relations,
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_compute_re_f1_overlap_below_threshold() -> None:
    """Pred entity name doesn't match gold → overlap=0.0 (<0.5) → entities don't match → F1=0.0."""
    gold_entities = [
        Entity(name="Apple", type="ORG", start=0, end=4),
    ]
    gold_relations = [
        Relation(
            head_idx=0, tail_idx=0,
            head_start=0, head_end=4,
            tail_start=0, tail_end=4,
            type="related_to",
        ),
    ]
    pred_entities = [
        PredictedEntity(name="Microsoft", type="ORG"),  # different name
    ]
    pred_relations = [
        PredictedRelation(head="Microsoft", tail="Microsoft", type="related_to"),
    ]

    result = compute_re_f1(
        gold_entities, gold_relations, pred_entities, pred_relations,
    )
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_compute_re_f1_empty_relations() -> None:
    """Gold has relations but pred_relations empty → P=0.0, R=0.0, F1=0.0."""
    gold_entities = [
        Entity(name="Apple", type="ORG", start=0, end=4),
        Entity(name="Cupertino", type="LOC", start=20, end=28),
    ]
    gold_relations = [
        Relation(
            head_idx=0, tail_idx=1,
            head_start=0, head_end=4,
            tail_start=20, tail_end=28,
            type="located_in",
        ),
    ]
    pred_entities = [
        PredictedEntity(name="Apple", type="ORG"),
        PredictedEntity(name="Cupertino", type="LOC"),
    ]
    pred_relations: list[PredictedRelation] = []

    result = compute_re_f1(
        gold_entities, gold_relations, pred_entities, pred_relations,
    )
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_avg_response_time
# ═══════════════════════════════════════════════════════════════════════════


def test_compute_avg_response_time() -> None:
    """[1.0, 2.0, 3.0] → 2.0."""
    timings = [1.0, 2.0, 3.0]
    result = compute_avg_response_time(timings)
    assert result == 2.0
