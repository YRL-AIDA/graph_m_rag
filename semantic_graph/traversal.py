"""
Unified Graph Traversal Engine — Strategy E.

Provides a priority-queue BFS crawler that traverses the unified
structural + semantic Neo4j graph, allocating a token budget and
collecting context from whichever path yields the most relevant
information.

Usage::

    from semantic_graph.traversal import TraversalConfig, UnifiedGraphCrawler

    config = TraversalConfig()
    crawler = UnifiedGraphCrawler(manager, config)
    context_chunks = crawler.crawl(seed_regions, question_embedding, token_budget=4000)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words for keyword-level text relevance scoring
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "we", "they", "them", "their", "he", "she", "his", "her",
    "as", "about", "into", "over", "after", "before", "between",
    "under", "again", "further", "once", "here", "there", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "just", "up", "down", "out",
    "off", "now", "also", "how", "what", "which", "who", "whom",
    "when", "where", "why", "any", "does",
})

# ---------------------------------------------------------------------------
# Scoring weights (tunable)
# ---------------------------------------------------------------------------
# Direct Qdrant match gets cosine_similarity (0.5-1.0).
SCORE_ORDER_NEIGHBOR: float = 0.6
SCORE_DIRECT_ENTITY: float = 0.8
SCORE_RELATED_1HOP: float = 0.5
SCORE_RELATED_2HOP: float = 0.3
SCORE_COMMUNITY_SIBLING: float = 0.4
SCORE_CROSS_GRAPH_REGION: float = 0.3
SCORE_COMMUNITY_NODE: float = 0.4
# Entity → Region back-expansion: structural context around a semantic entity
SCORE_ENTITY_LINKED_REGION: float = 0.55
# Entity → TextUnit → ORDER-adjacent Regions (2-hop from extraction source)
SCORE_ENTITY_TEXTUNIT_REGION: float = 0.45
# Community → Region via DESCRIBES_STRUCTURE (direct community-to-structure)
SCORE_COMMUNITY_LINKED_REGION: float = 0.35

# Estimated character count per token (rough heuristic)
CHARS_PER_TOKEN: int = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _QueueItem:
    """Priority-queue entry; lower priority value = higher priority."""

    priority: float
    node_id: str = field(compare=False)
    node_type: str = field(compare=False)  # 'region' | 'entity' | 'community'
    text: str = field(compare=False)
    metadata: Dict[str, Any] = field(compare=False)


@dataclass
class TraversalConfig:
    """Tunable parameters for the unified BFS crawler."""

    # Structural walk
    order_window_size: int = 3

    # Entity expansion
    entity_max_degree_2hop: int = 20
    community_max_siblings: int = 10

    # Cross-graph bridge
    bridge_max_regions: int = 15

    # Budget
    default_token_budget: int = 4000

    # Weights (can override per use)
    weight_order_neighbor: float = SCORE_ORDER_NEIGHBOR
    weight_direct_entity: float = SCORE_DIRECT_ENTITY
    weight_related_1hop: float = SCORE_RELATED_1HOP
    weight_related_2hop: float = SCORE_RELATED_2HOP
    weight_community_sibling: float = SCORE_COMMUNITY_SIBLING
    weight_cross_graph_region: float = SCORE_CROSS_GRAPH_REGION
    weight_community_node: float = SCORE_COMMUNITY_NODE
    weight_entity_linked_region: float = SCORE_ENTITY_LINKED_REGION
    weight_entity_textunit_region: float = SCORE_ENTITY_TEXTUNIT_REGION
    weight_community_linked_region: float = SCORE_COMMUNITY_LINKED_REGION

    # Context-aware bridge walking (C4)
    context_aware_bridge: bool = True
    bridge_relevance_boost: float = 0.15
    bridge_irrelevant_penalty: float = 0.2
    bridge_relevance_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class UnifiedGraphCrawler:
    """Priority-queue BFS over the unified structural + semantic graph.

    Parameters
    ----------
    manager:
        Instance of :class:`semantic_graph.manager.Manager`.
    neo4j_service:
        Instance of :class:`documet_index.neo4j_service.DocumentIndexService`
        (for structural ORDER walk and cross-graph bridge). Optional — if
        omitted, structural strategies are skipped.
    config:
        Tuning parameters for the traversal.
    """

    def __init__(
        self,
        manager: Any,
        neo4j_service: Any = None,
        config: Optional[TraversalConfig] = None,
    ) -> None:
        self._manager = manager
        self._neo4j = neo4j_service
        self._cfg = config or TraversalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(
        self,
        seed_regions: List[Dict[str, Any]],
        question_embedding: Optional[List[float]] = None,
        token_budget: Optional[int] = None,
        question_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the unified BFS crawl.

        Parameters
        ----------
        seed_regions:
            List of dicts from Qdrant with keys:
            ``region_id``, ``text`` (required), plus optional ``embedding``
            for cosine scoring.
        question_embedding:
            Dense vector of the user question. Used to score seed regions
            via cosine similarity. If omitted, seeds receive a default
            score of 0.7.
        token_budget:
            Approximate maximum tokens to collect. Defaults to
            :attr:`TraversalConfig.default_token_budget`.
        question_text:
            Raw text of the user question. Used for context-aware bridge
            walking to boost entities relevant to the question.

        Returns
        -------
        List of dicts with keys ``source``, ``text``, ``score``, ``node_type``.
        """
        budget = token_budget or self._cfg.default_token_budget
        ch_budget = budget * CHARS_PER_TOKEN
        self._question_text = question_text  # store for context-aware scoring

        heap: List[_QueueItem] = []
        visited: Set[str] = set()
        collected: List[Dict[str, Any]] = []
        used_chars = 0

        # --- Seed the queue ---
        for region in seed_regions:
            rid = region.get("region_id", "")
            rtext = region.get("text", "")
            if not rid or not rtext:
                continue
            emb = region.get("embedding")
            if emb is not None and question_embedding is not None:
                score = self._cosine(emb, question_embedding)
            else:
                score = 0.7
            heapq.heappush(
                heap,
                _QueueItem(
                    priority=-float(score),  # negate for max-heap via min-heap
                    node_id=rid,
                    node_type="region",
                    text=rtext,
                    metadata={"source": "qdrant_match"},
                ),
            )

        # --- Main BFS loop ---
        while heap and used_chars < ch_budget:
            item = heapq.heappop(heap)
            if item.node_id in visited:
                continue
            visited.add(item.node_id)

            if used_chars + len(item.text) > ch_budget:
                # Truncate to fit budget
                remaining = ch_budget - used_chars
                if remaining < 20:  # not worth adding a tiny snippet
                    continue
                item.text = item.text[:remaining]
            used_chars += len(item.text)

            collected.append(
                {
                    "source": item.metadata.get("source", "unknown"),
                    "text": item.text,
                    "score": -item.priority,
                    "node_type": item.node_type,
                }
            )

            # --- Expand neighbors based on node type ---
            if item.node_type == "region":
                self._expand_region(item, heap, visited, item.metadata.get("region_id", item.node_id))
            elif item.node_type == "entity":
                self._expand_entity(item, heap, visited)
            elif item.node_type == "community":
                self._expand_community(item, heap, visited)

        logger.info(
            "BFS crawl finished: %d nodes visited, ~%d chars collected (budget %d)",
            len(visited),
            used_chars,
            ch_budget,
        )
        return collected

    # ------------------------------------------------------------------
    # Node-type expansion helpers
    # ------------------------------------------------------------------

    def _expand_region(
        self,
        item: _QueueItem,
        heap: List[_QueueItem],
        visited: Set[str],
        region_id: str,
    ) -> None:
        """Expand a Region node: ORDER neighbours + directly linked Entities."""
        cfg = self._cfg

        # -- ORDER neighbours (structural walk, Strategy A) --
        if self._neo4j is not None:
            try:
                neighbours = self._neo4j.get_order_neighbors(
                    [region_id], window_size=cfg.order_window_size
                )
                for nb in neighbours:
                    nid = nb.get("region_id", "")
                    if nid in visited:
                        continue
                    heapq.heappush(
                        heap,
                        _QueueItem(
                            priority=-cfg.weight_order_neighbor,
                            node_id=nid,
                            node_type="region",
                            text=nb.get("text", ""),
                            metadata={"source": "order_neighbor"},
                        ),
                    )
            except Exception:
                logger.debug("Order-neighbour expansion failed for %s", region_id, exc_info=True)

        # -- Directly linked Entities (via bridge connections) --
        # C4: Context-aware scoring — entities relevant to the question get boosted
        try:
            entities = self._manager.get_entities_linked_to_region(region_id)
            for ent in entities:
                eid = ent.get("id", ent.get("title", ""))
                if eid in visited:
                    continue
                # Compute degree-based boost
                degree = float(ent.get("degree", 0) or 0)
                # Use bridge edge weight as base, fall back to config default
                edge_weight = float(ent.get("weight", cfg.weight_direct_entity))
                score = min(edge_weight, 0.95)
                if degree > 0:
                    score = min(score + 0.1 * (degree / max(degree, 1)), 0.95)
                # Context-aware relevance adjustment
                if cfg.context_aware_bridge and self._question_text:
                    relevance = self._text_relevance(
                        self._question_text,
                        ent.get("description", ""),
                        ent.get("title", ""),
                        ent.get("type", ""),
                    )
                    if relevance >= cfg.bridge_relevance_threshold:
                        score = min(score + cfg.bridge_relevance_boost, 0.95)
                    else:
                        score = max(score - cfg.bridge_irrelevant_penalty, 0.15)
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-score,
                        node_id=eid,
                        node_type="entity",
                        text=f"[{ent.get('title', '')}] ({ent.get('type', '')}): {ent.get('description', '')}",
                        metadata={
                            "source": "direct_entity",
                            "entity_title": ent.get("title", ""),
                            "entity_type": ent.get("type", ""),
                        },
                    ),
                )
        except Exception:
            logger.debug("Direct-entity expansion failed for %s", region_id, exc_info=True)

    def _expand_entity(
        self,
        item: _QueueItem,
        heap: List[_QueueItem],
        visited: Set[str],
    ) -> None:
        """Expand an Entity node: RELATED (1-hop and 2-hop), community siblings."""
        cfg = self._cfg
        entity_title = item.metadata.get("entity_title", "")
        entity_type = item.metadata.get("entity_type", "")

        if not entity_title:
            return

        # -- 1-hop + 2-hop RELATION (Strategy B) --
        try:
            related = self._manager.get_related_entities_2hop(
                entity_title, entity_type, max_degree=cfg.entity_max_degree_2hop
            )
            for rel in related:
                rid = rel.get("id", rel.get("title", ""))
                if rid in visited:
                    continue
                hop = rel.get("hop_depth", 1)
                score = (
                    cfg.weight_related_1hop if hop == 1 else cfg.weight_related_2hop
                )
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-score,
                        node_id=rid,
                        node_type="entity",
                        text=f"[{rel.get('title', '')}] ({rel.get('type', '')}): {rel.get('description', '')}",
                        metadata={
                            "source": f"related_{hop}hop",
                            "entity_title": rel.get("title", ""),
                            "entity_type": rel.get("type", ""),
                        },
                    ),
                )
        except Exception:
            logger.debug("2-hop expansion failed for %s", entity_title, exc_info=True)

        # -- Community siblings (Strategy C) --
        try:
            siblings = self._manager.get_community_siblings(
                entity_title, entity_type, max_siblings=cfg.community_max_siblings
            )
            for sib in siblings:
                sid = sib.get("id", sib.get("title", ""))
                if sid in visited:
                    continue
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-cfg.weight_community_sibling,
                        node_id=sid,
                        node_type="entity",
                        text=f"[{sib.get('title', '')}] ({sib.get('type', '')}): {sib.get('description', '')}",
                        metadata={
                            "source": "community_sibling",
                            "entity_title": sib.get("title", ""),
                            "entity_type": sib.get("type", ""),
                            "community_id": sib.get("community_id", ""),
                        },
                    ),
                )
        except Exception:
            logger.debug("Community-sibling expansion failed for %s", entity_title, exc_info=True)

        # -- Entity → Region back-expansion (bridge in reverse) --
        try:
            linked_regions = self._manager.get_regions_linked_to_entity(
                entity_title, entity_type, limit=10,
            )
            for reg in linked_regions:
                rid = reg.get("region_id", "")
                if rid in visited:
                    continue
                # Weight from bridge edge, fall back to config default
                edge_w = float(reg.get("weight", cfg.weight_entity_linked_region))
                score = min(edge_w, 0.95)
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-score,
                        node_id=rid,
                        node_type="region",
                        text=reg.get("text", ""),
                        metadata={
                            "source": "entity_linked_region",
                            "region_id": rid,
                        },
                    ),
                )
        except Exception:
            logger.debug(
                "Entity→Region expansion failed for %s", entity_title, exc_info=True,
            )

        # NOTE: Entity→TextUnit→Region expansion (Strategy E-sub) is
        # DISABLED because the underlying manager method queries
        # LINKED_TO_TEXTUNIT / TextUnit / PART_OF nodes that do not
        # exist in the current Neo4j schema.  Re-enable after running
        # connect_graphs.py to populate the structural→semantic bridge.
        #
        # try:
        #     textunit_regions = self._manager.get_regions_by_entity_textunits(
        #         entity_title, entity_type, window_size=2, limit=10,
        #     )
        #     ...
        # except Exception:
        #     logger.debug(...)

    def _expand_community(
        self,
        item: _QueueItem,
        heap: List[_QueueItem],
        visited: Set[str],
    ) -> None:
        """Expand a Community node: member Entities + cross-graph Regions."""
        cfg = self._cfg
        community_id = item.metadata.get("community_id", item.node_id)

        # -- Member entities --
        try:
            members = self._manager.get_community_members(community_id)
            for mem in members:
                mid = mem.get("id", mem.get("title", ""))
                if mid in visited:
                    continue
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-cfg.weight_community_node,
                        node_id=mid,
                        node_type="entity",
                        text=f"[{mem.get('title', '')}] ({mem.get('type', '')}): {mem.get('description', '')}",
                        metadata={
                            "source": "community_member",
                            "entity_title": mem.get("title", ""),
                            "entity_type": mem.get("type", ""),
                        },
                    ),
                )
        except Exception:
            logger.debug("Community-member expansion failed for %s", community_id, exc_info=True)

        # -- Community → Region via DESCRIBES_STRUCTURE (direct bridge) --
        # Falls back to 2-hop via member Entities when no direct edges exist.
        try:
            community_regions = self._manager.get_regions_linked_to_community(
                community_id, limit=10,
            )
            for reg in community_regions:
                rid = reg.get("region_id", "")
                if rid in visited:
                    continue
                src = reg.get("source", "direct")
                score = (
                    cfg.weight_community_linked_region
                    if src == "direct"
                    else cfg.weight_community_linked_region * 0.8
                )
                heapq.heappush(
                    heap,
                    _QueueItem(
                        priority=-score,
                        node_id=rid,
                        node_type="region",
                        text=reg.get("text", ""),
                        metadata={
                            "source": f"community_{src}_region",
                            "region_id": rid,
                        },
                    ),
                )
        except Exception:
            logger.debug(
                "Community→Region expansion failed for %s",
                community_id, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Cosine similarity (standalone — no external dependency required)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def _text_relevance(
        question: str,
        description: str,
        title: str = "",
        entity_type: str = "",
    ) -> float:
        """Compute keyword-level relevance between a question and entity text.

        Uses word-level Jaccard similarity as a lightweight relevance signal
        for context-aware bridge walking. Returns a float in [0.0, 1.0].
        """
        if not question:
            return 0.0
        combined = f"{title} {entity_type} {description}"
        q_tokens = set(question.lower().split()) - _STOP_WORDS
        e_tokens = set(combined.lower().split()) - _STOP_WORDS
        if not q_tokens or not e_tokens:
            return 0.0
        intersection = q_tokens & e_tokens
        # Normalize by question token count (how much of the question is covered)
        return len(intersection) / len(q_tokens)
