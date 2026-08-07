"""
MMR (Maximal Marginal Relevance) Reranker for context diversity.

Algorithm: Selects k context blocks that maximize relevance to the query
while minimizing redundancy (cosine similarity to already-selected blocks).

MMR = λ * relevance(block, query) - (1-λ) * max_{selected} similarity(block, selected)

where:
  λ ∈ [0, 1] — relevance vs diversity tradeoff (higher = more relevance, less diversity)
  relevance — cosine similarity between block embedding and query embedding
  similarity — cosine similarity between two block embeddings
"""

import logging
from typing import List, Tuple, Optional, Sequence

logger = logging.getLogger(__name__)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai * ai for ai in a) ** 0.5
    norm_b = sum(bi * bi for bi in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    items: List[str],
    item_embeddings: List[List[float]],
    query_embedding: List[float],
    lambda_param: float = 0.7,
    top_k: int = 20,
) -> Tuple[List[str], List[float]]:
    """
    Rerank items using MMR to balance relevance and diversity.

    Args:
        items: List of context block texts.
        item_embeddings: Corresponding embeddings for each item.
        query_embedding: Embedding of the user's question.
        lambda_param: Relevance-vs-diversity tradeoff (0-1).
                      1.0 = pure relevance ranking.
                      0.0 = pure diversity ranking.
        top_k: Maximum number of items to return.

    Returns:
        Tuple of (reranked_items, mmr_scores) in MMR score descending order.
        Items not selected are excluded.
    """
    n = len(items)
    if n == 0:
        return [], []

    if n != len(item_embeddings):
        raise ValueError(
            f"Mismatch: {len(items)} items vs {len(item_embeddings)} embeddings"
        )

    if top_k <= 0:
        # If top_k <= 0, return all items in relevance order
        top_k = n

    # Handle trivial cases
    if n == 1:
        return items[:], [cosine_similarity(item_embeddings[0], query_embedding)]

    # Stage 1: Compute relevance of each item to the query
    relevance_scores = [
        cosine_similarity(emb, query_embedding) for emb in item_embeddings
    ]

    # Stage 2: Greedy MMR selection
    selected_indices: List[int] = []
    mmr_scores: List[float] = []
    remaining = set(range(n))

    while remaining and len(selected_indices) < min(top_k, n):
        best_mmr = -float("inf")
        best_idx = -1

        for idx in remaining:
            relevance = relevance_scores[idx]

            if not selected_indices:
                mmr = relevance
            else:
                # Max similarity to already-selected items
                max_sim = max(
                    cosine_similarity(
                        item_embeddings[idx], item_embeddings[sel_idx]
                    )
                    for sel_idx in selected_indices
                )
                mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        mmr_scores.append(best_mmr)
        remaining.remove(best_idx)

    reranked_items = [items[i] for i in selected_indices]
    return reranked_items, mmr_scores


def mmr_rerank_with_threshold(
    items: List[str],
    item_embeddings: List[List[float]],
    query_embedding: List[float],
    lambda_param: float = 0.7,
    top_k: int = 20,
    min_relevance: float = 0.0,
    min_mmr: float = 0.0,
) -> Tuple[List[str], List[float]]:
    """
    MMR rerank with relevance and MMR score thresholds.

    Items below min_relevance at selection time OR below min_mmr
    after selection are excluded from results.

    Args:
        items, item_embeddings, query_embedding, lambda_param, top_k:
            Same as mmr_rerank.
        min_relevance: Minimum relevance score for an item to be
            considered for selection (applied during greedy loop).
        min_mmr: Minimum MMR score for an item to be included in
            the final result (applied after selection).

    Returns:
        Tuple of (reranked_items, mmr_scores) after filtering.
    """
    n = len(items)
    if n == 0:
        return [], []

    if n != len(item_embeddings):
        raise ValueError(
            f"Mismatch: {len(items)} items vs {len(item_embeddings)} embeddings"
        )

    if top_k <= 0:
        top_k = n

    # Compute relevance scores
    relevance_scores = [
        cosine_similarity(emb, query_embedding) for emb in item_embeddings
    ]

    # Greedy MMR selection with min_relevance threshold
    selected_indices: List[int] = []
    remaining = set(range(n))

    while remaining and len(selected_indices) < min(top_k, n):
        best_mmr = -float("inf")
        best_idx = -1

        for idx in remaining:
            relevance = relevance_scores[idx]

            # Skip items below minimum relevance
            if relevance < min_relevance:
                continue

            if not selected_indices:
                mmr = relevance
            else:
                max_sim = max(
                    cosine_similarity(
                        item_embeddings[idx], item_embeddings[sel_idx]
                    )
                    for sel_idx in selected_indices
                )
                mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    # Apply min_mmr threshold
    final_items = []
    final_scores = []
    for i, idx in enumerate(selected_indices):
        mmr_val = 0.0
        if i == 0:
            mmr_val = relevance_scores[idx]
        else:
            max_sim = max(
                cosine_similarity(
                    item_embeddings[idx], item_embeddings[sel_idx]
                )
                for sel_idx in selected_indices[:i]
            )
            mmr_val = lambda_param * relevance_scores[idx] - (1.0 - lambda_param) * max_sim

        if mmr_val >= min_mmr:
            final_items.append(items[idx])
            final_scores.append(mmr_val)

    return final_items, final_scores
