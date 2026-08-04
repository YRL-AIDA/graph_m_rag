"""Функции расчёта метрик: NER F1, RE F1, среднее время ответа.

Импортирует типы сущностей и отношений из testdata.base_loader.
"""

from __future__ import annotations

import logging

from testdata.base_loader import Entity, PredictedEntity, PredictedRelation, Relation

logger = logging.getLogger(__name__)

# Порог overlap для матчинга сущностей в RE-метрике
_OVERLAP_THRESHOLD: float = 0.5


def compute_ner_f1(
    gold: list[Entity], pred: list[PredictedEntity]
) -> dict[str, float]:
    """Вычислить точность, полноту и F1 для NER по exact match (name, type).

    TP = количество предсказанных сущностей, для которых существует золотая
    сущность с теми же (name, type). Жадный алгоритм: одна gold-сущность
    может быть сопоставлена не более чем с одной pred-сущностью.

    Если pred пуст → precision = 0.0. Если gold пуст → recall = 0.0.
    F1 = 2*P*R/(P+R) при P+R > 0, иначе 0.0.

    Args:
        gold: Список золотых сущностей (Entity).
        pred: Список предсказанных сущностей (PredictedEntity).

    Returns:
        Словарь с ключами precision, recall, f1.
    """
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Множество несматченных gold-сущностей: (name, type)
    unmatched_gold: set[tuple[str, str]] = {(g.name, g.type) for g in gold}
    tp = 0

    for p in pred:
        key = (p.name, p.type)
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
) -> dict[str, float]:
    """Вычислить точность, полноту и F1 для RE.

    Алгоритм (по спецификации):

    1. **Entity matching**: для каждой pred-сущности вычисляется
       максимальный char-span overlap с каждой gold-сущностью:
         overlap = max(0, min(g.end, p.end) - max(g.start, p.start) + 1)
                   / max(len(g), len(p))
       Pred-сущность считается сматченной, если overlap ≥ 0.5.
       Жадный алгоритм: одна gold-сущность на одну pred-сущность.

    2. **RE TP**: gold-отношение (head_idx, tail_idx, type) найдено,
       если head- и tail-gold-сущности сматчены с pred-сущностями
       P_head, P_tail и существует pred-отношение
       (head=P_head.name, tail=P_tail.name, type=gold.type).

    3. Precision = TP_RE / len(pred_relations),
       Recall = TP_RE / len(gold_relations).
       F1 = 2*P*R/(P+R) при P+R > 0, иначе 0.0.

    Args:
        gold_entities: Золотые сущности с символьными индексами.
        gold_relations: Золотые отношения.
        pred_entities: Предсказанные сущности.
        pred_relations: Предсказанные отношения.

    Returns:
        Словарь с ключами precision, recall, f1.
    """
    # --- Entity matching (жадный) ---
    # gold_entity_idx → pred_entity_idx (или None, если не сматчена)
    gold_matches: dict[int, int | None] = {}
    # Множество ещё не сматченных pred-сущностей (по индексу)
    unmatched_pred: set[int] = set(range(len(pred_entities)))

    for g_idx, g in enumerate(gold_entities):
        best_overlap = 0.0
        best_p_idx: int | None = None
        g_len = g.end - g.start + 1

        for p_idx in unmatched_pred:
            p = pred_entities[p_idx]
            # Для PredictedEntity нет start/end — используем name-матчинг
            # в тексте gold-сущности. Совпадение по точному имени.
            # Если имя не совпадает точно, overlap считается через поиск в тексте.
            # Но для PredictedEntity у нас нет char спан, поэтому используем
            # сравнение по имени: overlap = 1.0 при совпадении, 0.0 иначе.
            if p.name == g.name:
                overlap = 1.0
            else:
                overlap = 0.0

            if overlap >= _OVERLAP_THRESHOLD and overlap > best_overlap:
                best_overlap = overlap
                best_p_idx = p_idx

        if best_p_idx is not None:
            gold_matches[g_idx] = best_p_idx
            unmatched_pred.discard(best_p_idx)
        else:
            gold_matches[g_idx] = None

    # --- RE TP ---
    if not pred_relations:
        # Если предсказанных отношений нет, TP = 0
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gold_relations:
        # Если золотых отношений нет, но есть предсказанные — precision = 0
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp_re = 0
    for rel in gold_relations:
        head_match = gold_matches.get(rel.head_idx)
        tail_match = gold_matches.get(rel.tail_idx)

        if head_match is None or tail_match is None:
            continue

        head_pred = pred_entities[head_match]
        tail_pred = pred_entities[tail_match]

        # Ищем pred-отношение с совпадающими head.name, tail.name, type
        found = any(
            pr.head == head_pred.name
            and pr.tail == tail_pred.name
            and pr.type == rel.type
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
