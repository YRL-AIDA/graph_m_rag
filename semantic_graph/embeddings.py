"""
Entity and Community embedding computation service.

Stores embeddings in Qdrant collections:
- {ENTITY_EMBEDDINGS_COLLECTION}: entity_id -> vector
- {COMMUNITY_EMBEDDINGS_COLLECTION}: community_id -> vector

Provides an abstraction layer that:
1. Calls the external embedding service via aiohttp with concurrency control.
2. Upserts computed vectors into Qdrant via AsyncQdrantClient.
3. Ensures collections exist before upsert.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    COMMUNITY_EMBEDDINGS_BATCH_SIZE,
    COMMUNITY_EMBEDDINGS_COLLECTION,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_CONCURRENCY,
    EMBEDDING_RETRY_BASE_DELAY,
    EMBEDDING_RETRY_COUNT,
    EMBEDDING_TIMEOUT,
    ENTITY_EMBEDDINGS_BATCH_SIZE,
    ENTITY_EMBEDDINGS_COLLECTION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosine similarity (public – imported by semantic_index.py)
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))

# Backward-compatible alias
_cosine_similarity = cosine_similarity

# ---------------------------------------------------------------------------
# Low-level embedding API call
# ---------------------------------------------------------------------------

async def compute_embeddings(
    texts: List[str],
    session: Optional[aiohttp.ClientSession] = None,
) -> List[List[float]]:
    """Compute embedding vectors for a list of texts via the external service.

    Uses the same `/embed` endpoint and message format as
    :class:`app.src.qwen3_emb_client.EmbeddingClient`.

    Parameters
    ----------
    texts:
        One or more text strings to embed.
    session:
        Optional shared ``aiohttp.ClientSession``.  If *None* a temporary
        session is created and closed.

    Returns
    -------
    list[list[float]]
        Embedding vectors in the same order as *texts*.  Failed texts yield a
        zero-vector whose dimension is inferred from the first successful
        response (or 2048 if nothing succeeds).
    """
    if not texts:
        return []

    semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)
    vector_size: Optional[int] = None

    async def _embed_one(text: str) -> Optional[List[float]]:
        nonlocal vector_size
        async with semaphore:
            last_exc: Optional[Exception] = None
            for attempt in range(EMBEDDING_RETRY_COUNT):
                try:
                    payload = {"messages": [{"type": "text", "text": text}]}
                    async with session_or_temp.post(
                        f"{EMBEDDING_BASE_URL}/embed",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=EMBEDDING_TIMEOUT),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        embedding = data["messages"][0]["embedding"]
                        if vector_size is None:
                            vector_size = len(embedding)
                        return embedding
                except Exception as exc:
                    last_exc = exc
                    if attempt < EMBEDDING_RETRY_COUNT - 1:
                        delay = EMBEDDING_RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Embedding attempt %d/%d failed for %.80r, "
                            "retrying in %.1fs",
                            attempt + 1,
                            EMBEDDING_RETRY_COUNT,
                            text,
                            delay,
                        )
                        await asyncio.sleep(delay)
            logger.warning(
                "Failed to compute embedding for text %.80r after %d attempts",
                text,
                EMBEDDING_RETRY_COUNT,
                exc_info=last_exc,
            )
            return None

    close_on_exit = False
    if session is None:
        session = aiohttp.ClientSession()
        close_on_exit = True
    session_or_temp = session

    try:
        tasks = [_embed_one(t) for t in texts]
        results = await asyncio.gather(*tasks)
    finally:
        if close_on_exit:
            await session_or_temp.close()

    fallback_dim = vector_size or (EMBEDDING_DIMENSION or 2048)
    return [r if r is not None else [0.0] * fallback_dim for r in results]


async def compute_embedding_single(
    text: str,
    session: aiohttp.ClientSession,
) -> Optional[List[float]]:
    """Compute a single embedding vector via the external embedding service.

    Lightweight wrapper around the same `/embed` endpoint used by
    :func:`compute_embeddings`.  Returns *None* on failure so callers can
    decide on fallback behaviour.

    Parameters
    ----------
    text:
        The text string to embed.
    session:
        A shared ``aiohttp.ClientSession`` (required – no temporary session is
        created here).

    Returns
    -------
    Optional[list[float]]
        The embedding vector, or *None* if the service call failed.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(EMBEDDING_RETRY_COUNT):
        try:
            payload = {"messages": [{"type": "text", "text": text}]}
            async with session.post(
                f"{EMBEDDING_BASE_URL}/embed",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=EMBEDDING_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["messages"][0]["embedding"]
        except Exception as exc:
            last_exc = exc
            if attempt < EMBEDDING_RETRY_COUNT - 1:
                delay = EMBEDDING_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Single embedding attempt %d/%d failed for %.80r, "
                    "retrying in %.1fs",
                    attempt + 1,
                    EMBEDDING_RETRY_COUNT,
                    text,
                    delay,
                )
                await asyncio.sleep(delay)
    logger.warning(
        "Failed to compute single embedding for text %.80r after %d attempts",
        text,
        EMBEDDING_RETRY_COUNT,
        exc_info=last_exc,
    )
    return None


async def disambiguate_entities(
    entities: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Disambiguate entities with duplicate (title, type) pairs using embeddings.

    For groups of entities sharing the same *title* and *type*, compute
    embedding vectors and compare via cosine similarity.  Entities whose
    embeddings are similar enough (``>= threshold``) are treated as the same
    concept; those below the threshold are renamed with a short context suffix
    to keep them distinct.

    Parameters
    ----------
    entities:
        List of entity dicts.  Each dict must have at least ``"title"``,
        ``"type"``, and ``"description"`` keys.
    session:
        Shared ``aiohttp.ClientSession`` used to call the embedding service.
    threshold:
        Cosine-similarity threshold.  Defaults to 0.85.

    Returns
    -------
    list[dict]
        The (possibly modified) list of entity dicts.  Ambiguous duplicates
        that fell below the threshold have had their ``"title"`` renamed.
        Non-duplicated entities are returned unchanged.

    Notes
    -----
    - This function is **non-blocking on failure**: if the embedding service
      is unavailable or any call fails, a warning is logged and the original
      entities are returned unchanged.
    """
    if not entities:
        return entities

    try:
        # 1. Group by (title, type)
        groups: Dict[tuple, List[int]] = {}
        for idx, ent in enumerate(entities):
            key = (ent.get("title", ""), ent.get("type", ""))
            groups.setdefault(key, []).append(idx)

        renamed_count = 0

        for key, indices in groups.items():
            if len(indices) <= 1:
                continue

            # 2. Build embedding texts and compute embeddings
            texts = [
                _entity_embedding_text(
                    entities[i].get("title", ""),
                    entities[i].get("type", ""),
                    entities[i].get("description", ""),
                )
                for i in indices
            ]
            embeddings = await asyncio.gather(
                *[compute_embedding_single(t, session) for t in texts]
            )

            # Fall back if any embedding failed
            if any(e is None for e in embeddings):
                logger.warning(
                    "Skipping disambiguation for '%s' (type=%s): "
                    "one or more embedding calls failed.",
                    key[0], key[1],
                )
                continue

            # 3. Pairwise comparison with union-find style clustering
            n = len(indices)
            parent = list(range(n))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def _union(x: int, y: int) -> None:
                rx, ry = _find(x), _find(y)
                if rx != ry:
                    parent[ry] = rx

            for i in range(n):
                for j in range(i + 1, n):
                    vec_i = embeddings[i]
                    vec_j = embeddings[j]
                    if vec_i is None or vec_j is None:
                        continue
                    sim = _cosine_similarity(vec_i, vec_j)
                    if sim >= threshold:
                        _union(i, j)

            # 4. Build clusters (groups that should merge)
            clusters: Dict[int, List[int]] = {}
            for i in range(n):
                root = _find(i)
                clusters.setdefault(root, []).append(i)

            if len(clusters) <= 1:
                continue  # All are similar enough to merge naturally

            # 5. Rename each cluster with a context suffix
            for cluster_idx, cluster_members in enumerate(clusters.values()):
                if cluster_idx == 0:
                    continue  # First cluster keeps the original title

                for member in cluster_members:
                    entity_index = indices[member]
                    orig_title = entities[entity_index].get("title", "")
                    description = entities[entity_index].get("description", "")
                    # Take first 5 words of description as context
                    context_words = description.split()[:5]
                    context = " ".join(context_words) if context_words else f"context{cluster_idx + 1}"
                    new_title = f"{orig_title} ({context})"
                    logger.info(
                        "Disambiguated '%s': renamed to '%s' (similarity below threshold %.3f)",
                        orig_title, new_title, threshold,
                    )
                    entities[entity_index]["title"] = new_title
                    renamed_count += 1

        if renamed_count > 0:
            logger.info(
                "Disambiguation complete: %d entity(ies) renamed across %d duplicate group(s).",
                renamed_count,
                sum(1 for v in groups.values() if len(v) > 1),
            )

    except Exception:
        logger.warning(
            "Disambiguation failed, returning original entities unchanged.",
            exc_info=True,
        )

    return entities


# ---------------------------------------------------------------------------
# Qdrant collection helpers
# ---------------------------------------------------------------------------

async def ensure_collection(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Check that a Qdrant collection exists, creating it if necessary.

    The collection is created with ``Cosine`` distance and the given
    *vector_size*.
    """
    try:
        exists = await qdrant_client.collection_exists(collection_name)
    except Exception:
        logger.info(
            "Could not check existence for collection '%s', attempting creation.",
            collection_name,
        )
        exists = False

    if not exists:
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' with size=%d", collection_name, vector_size)


# ---------------------------------------------------------------------------
# Entity embeddings
# ---------------------------------------------------------------------------

async def embed_entities(
    entity_data: List[Dict[str, Any]],
    qdrant_client: AsyncQdrantClient,
    session: Optional[aiohttp.ClientSession] = None,
) -> int:
    """Compute and store embeddings for entity nodes.

    Parameters
    ----------
    entity_data:
        List of dicts, each with at least ``"id"``, ``"title"``, ``"type"``
        and ``"description"``.  The ``"id"`` value is the *entity_id* used as
        the Qdrant point id.  ``"title"`` and ``"type"`` are stored in the
        payload.
    qdrant_client:
        Initialised :class:`AsyncQdrantClient`.
    session:
        Optional shared ``aiohttp.ClientSession``.

    Returns
    -------
    int
        Number of entities successfully embedded and upserted.
    """
    if not entity_data:
        return 0

    texts = [
        _entity_embedding_text(e.get("title", ""), e.get("type", ""), e.get("description", ""))
        for e in entity_data
    ]
    embeddings = await compute_embeddings(texts, session=session)

    await ensure_collection(qdrant_client, ENTITY_EMBEDDINGS_COLLECTION, len(embeddings[0]))

    points: List[PointStruct] = []
    for entity, vector in zip(entity_data, embeddings):
        entity_id = entity["id"]
        points.append(
            PointStruct(
                id=entity_id,
                vector=vector,
                payload={
                    "entity_title": entity.get("title", ""),
                    "entity_type": entity.get("type", ""),
                    "entity_id": entity_id,
                    "description": entity.get("description", ""),
                },
            )
        )

    batch_size = ENTITY_EMBEDDINGS_BATCH_SIZE
    upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await qdrant_client.upsert(
            collection_name=ENTITY_EMBEDDINGS_COLLECTION,
            points=batch,
            wait=True,
        )
        upserted += len(batch)
        logger.debug("Upserted entity batch %d-%d", i, i + len(batch))

    logger.info("Embedded and stored %d entities in '%s'", upserted, ENTITY_EMBEDDINGS_COLLECTION)
    return upserted


def _entity_embedding_text(title: str, entity_type: str, description: str) -> str:
    """Build the text string that will be embedded for an entity."""
    if description:
        return f"{title} ({entity_type}): {description}"
    return title


# ---------------------------------------------------------------------------
# Community embeddings
# ---------------------------------------------------------------------------

async def embed_communities(
    community_data: List[Dict[str, Any]],
    qdrant_client: AsyncQdrantClient,
    session: Optional[aiohttp.ClientSession] = None,
) -> int:
    """Compute and store embeddings for community nodes.

    Parameters
    ----------
    community_data:
        List of dicts, each with at least ``"id"``, ``"title"``, ``"summary"``
        and ``"report"``.
    qdrant_client:
        Initialised :class:`AsyncQdrantClient`.
    session:
        Optional shared ``aiohttp.ClientSession``.

    Returns
    -------
    int
        Number of communities successfully embedded and upserted.
    """
    if not community_data:
        return 0

    texts = [
        _community_embedding_text(c.get("title", ""), c.get("summary", ""), c.get("report", ""))
        for c in community_data
    ]
    embeddings = await compute_embeddings(texts, session=session)

    await ensure_collection(qdrant_client, COMMUNITY_EMBEDDINGS_COLLECTION, len(embeddings[0]))

    points: List[PointStruct] = []
    for community, vector in zip(community_data, embeddings):
        comm_id = community["id"]
        points.append(
            PointStruct(
                id=comm_id,
                vector=vector,
                payload={
                    "community_id": comm_id,
                    "title": community.get("title", ""),
                    "summary": community.get("summary", ""),
                    "report": community.get("report", ""),
                },
            )
        )

    batch_size = COMMUNITY_EMBEDDINGS_BATCH_SIZE
    upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await qdrant_client.upsert(
            collection_name=COMMUNITY_EMBEDDINGS_COLLECTION,
            points=batch,
            wait=True,
        )
        upserted += len(batch)
        logger.debug("Upserted community batch %d-%d", i, i + len(batch))

    logger.info(
        "Embedded and stored %d communities in '%s'", upserted, COMMUNITY_EMBEDDINGS_COLLECTION
    )
    return upserted


def _community_embedding_text(title: str, summary: str, report: str) -> str:
    """Build the text string that will be embedded for a community."""
    if summary:
        return f"{title}: {summary}"
    if report:
        return f"{title}: {report}"
    return title
