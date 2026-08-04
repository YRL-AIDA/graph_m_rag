"""Функции расчёта метрик: NER F1, RE F1, среднее время ответа.

Импортирует типы сущностей и отношений из testdata.base_loader.
"""

from __future__ import annotations

import logging

from testdata.base_loader import Entity, PredictedEntity, PredictedRelation, Relation

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Нормализовать имя сущности: lower(), strip(), схлопнуть множественные пробелы в один."""
    return " ".join(name.lower().strip().split())


# Mapping от вариантов написания (lowercase) к каноническим типам CoNLL04
TYPE_SYNONYMS: dict[str, str] = {
    # CoNLL-04 entity types
    "person": "Peop",
    "people": "Peop",
    "human": "Peop",
    "location": "Loc",
    "place": "Loc",
    "loc": "Loc",
    "organization": "Org",
    "organisation": "Org",
    "company": "Org",
    "corporation": "Org",
    "corp": "Org",
    "other": "Other",
    "miscellaneous": "Other",
    "misc": "Other",
}


def normalize_type(typ: str, allowed_types: list[str]) -> str | None:
    """Сопоставить тип, возвращённый моделью, с каноническим типом из датасета.

    Алгоритм:
    1. Точное совпадение (case-insensitive) с одним из allowed_types → вернуть канонический тип.
    2. Поиск в TYPE_SYNONYMS: если тип (lowercase) есть в словаре синонимов → вернуть канонический тип.
    3. Иначе → None (тип не распознан).

    Args:
        typ: Тип, возвращённый моделью.
        allowed_types: Список канонических типов из датасета.

    Returns:
        Канонический тип из allowed_types или None.
    """
    typ_lower = typ.lower().strip()

    # 1. Точное совпадение с allowed_types (case-insensitive)
    for allowed in allowed_types:
        if allowed.lower() == typ_lower:
            return allowed  # возвращаем канонический тип из датасета

    # 2. Поиск в TYPE_SYNONYMS
    if typ_lower in TYPE_SYNONYMS:
        canonical = TYPE_SYNONYMS[typ_lower]
        # Проверяем что канонический тип есть в allowed_types
        if canonical in allowed_types:
            return canonical

    # 3. Не распознан
    return None


def compute_ner_f1(
    gold: list[Entity], pred: list[PredictedEntity], entity_types: list[str]
) -> dict[str, float]:
    """Вычислить точность, полноту и F1 для NER по нормализованным (name, type).

    TP = количество предсказанных сущностей, для которых существует золотая
    сущность с теми же нормализованными (name, type). Жадный алгоритм: одна
    gold-сущность может быть сопоставлена не более чем с одной pred-сущностью.

    Если pred пуст → precision = 0.0. Если gold пуст → recall = 0.0.
    F1 = 2*P*R/(P+R) при P+R > 0, иначе 0.0.

    Args:
        gold: Список золотых сущностей (Entity).
        pred: Список предсказанных сущностей (PredictedEntity).
        entity_types: Список канонических типов сущностей датасета.

    Returns:
        Словарь с ключами precision, recall, f1.
    """
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Множество несматченных gold-сущностей: (normalized_name, normalized_type)
    unmatched_gold: set[tuple[str, str]] = set()
    for g in gold:
        norm_name = normalize_name(g.name)
        norm_type = normalize_type(g.type, entity_types)
        if norm_type is not None:
            unmatched_gold.add((norm_name, norm_type))

    tp = 0

    for p in pred:
        norm_name = normalize_name(p.name)
        norm_type = normalize_type(p.type, entity_types)
        if norm_type is not None:
            key = (norm_name, norm_type)
            if key in unmatched_gold:
                tp += 1
                unmatched_gold.discard(key)

    precision = tp / len(pred)
    recall = tp / len(gold)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_re_f1(
    gold_entities: list[Entity],
    gold_relations: list[Relation],
    pred_entities: list[PredictedEntity],
    pred_relations: list[PredictedRelation],
    entity_types: list[str],
    relation_types: list[str],
) -> dict[str, float]:
    """Вычислить точность, полноту и F1 для RE.

    Алгоритм (по спецификации):

    1. **Entity matching**: gold и pred сущности сопоставляются по
       нормализованным (name, type). Используются normalize_name() и
       normalize_type(). Жадный алгоритм: одна gold-сущность на одну
       pred-сущность, первый match — лучший.

    2. **RE TP**: gold-отношение (head_idx, tail_idx, type) найдено,
       если head- и tail-gold-сущности сматчены с pred-сущностями
       P_head, P_tail и существует pred-отношение
       (head=normalized_name, tail=normalized_name, type=normalized_type).

    3. Precision = TP_RE / len(pred_relations),
       Recall = TP_RE / len(gold_relations).
       F1 = 2*P*R/(P+R) при P+R > 0, иначе 0.0.

    Args:
        gold_entities: Золотые сущности.
        gold_relations: Золотые отношения.
        pred_entities: Предсказанные сущности.
        pred_relations: Предсказанные отношения.
        entity_types: Список канонических типов сущностей датасета.
        relation_types: Список канонических типов отношений датасета.

    Returns:
        Словарь с ключами precision, recall, f1.
    """
    # --- Entity matching (жадный, normalized name+type) ---
    gold_matches: dict[int, int | None] = {}
    unmatched_pred: set[int] = set(range(len(pred_entities)))

    for g_idx, g in enumerate(gold_entities):
        best_p_idx: int | None = None

        for p_idx in unmatched_pred:
            p = pred_entities[p_idx]
            # Compare normalized names
            if normalize_name(p.name) == normalize_name(g.name):
                # Both types must be normalizable and match
                g_type = normalize_type(g.type, entity_types)
                p_type = normalize_type(p.type, entity_types)
                if g_type is not None and p_type is not None and g_type == p_type:
                    best_p_idx = p_idx
                    break  # First match wins (greedy)

        if best_p_idx is not None:
            gold_matches[g_idx] = best_p_idx
            unmatched_pred.discard(best_p_idx)
        else:
            gold_matches[g_idx] = None

    # --- RE TP ---
    if not pred_relations:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gold_relations:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp_re = 0
    for rel in gold_relations:
        head_match = gold_matches.get(rel.head_idx)
        tail_match = gold_matches.get(rel.tail_idx)

        if head_match is None or tail_match is None:
            continue

        head_pred = pred_entities[head_match]
        tail_pred = pred_entities[tail_match]

        # Normalize relation types too
        g_rel_type = normalize_type(rel.type, relation_types)
        if g_rel_type is None:
            continue

        found = any(
            normalize_name(pr.head) == normalize_name(head_pred.name)
            and normalize_name(pr.tail) == normalize_name(tail_pred.name)
            and normalize_type(pr.type, relation_types) == g_rel_type
            for pr in pred_relations
        )
        if found:
            tp_re += 1

    precision = tp_re / len(pred_relations)
    recall = tp_re / len(gold_relations)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_avg_response_time(timings: list[float]) -> float:
    """Вычислить среднее время ответа в секундах.

    Args:
        timings: Список замеров времени (секунды) для каждого запроса.

    Returns:
        Среднее время в секундах. 0.0, если список пуст.
    """
    if not timings:
        return 0.0
    return sum(timings) / len(timings)
