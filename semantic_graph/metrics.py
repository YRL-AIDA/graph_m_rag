"""
Quality Metrics Module

Computes observability metrics for the knowledge graph extraction pipeline:
 - Graph-level metrics via Cypher queries against Neo4j.
 - Per-run extraction stats from DataFrames.
 - Entity quality heuristics.
"""

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


async def compute_graph_metrics(neo4j_service) -> Dict[str, Any]:
    """Run Cypher queries against Neo4j to compute graph quality metrics.

    Args:
        neo4j_service: A :class:`Manager` instance (from ``semantic_graph.manager``)
                       that provides ``.conn`` (Neo4jConnection) and ``.name_db``.

    Returns:
        Dictionary containing all computed graph-level metrics.
    """
    metrics: Dict[str, Any] = {}

    # --- Count totals ---
    rows = await neo4j_service.run_cypher("MATCH (e:Entity) RETURN count(e) AS c")
    total_entities = rows[0]["c"] if rows else 0
    metrics["total_entities"] = total_entities

    rows = await neo4j_service.run_cypher("MATCH ()-[r:semantic_link]->() RETURN count(r) AS c")
    total_relationships = rows[0]["c"] if rows else 0
    metrics["total_relationships"] = total_relationships

    rows = await neo4j_service.run_cypher("MATCH (c:Community) RETURN count(c) AS c")
    total_communities = rows[0]["c"] if rows else 0
    metrics["total_communities"] = total_communities

    rows = await neo4j_service.run_cypher("MATCH (d:Document) RETURN count(DISTINCT d) AS c")
    total_documents = rows[0]["c"] if rows else 0
    metrics["total_documents"] = total_documents

    # --- Per-document / per-entity averages ---
    if total_documents > 0:
        metrics["entities_per_document"] = round(total_entities / total_documents, 2)
    else:
        metrics["entities_per_document"] = 0.0

    if total_entities > 0:
        metrics["relationships_per_entity"] = round(total_relationships / total_entities, 2)
    else:
        metrics["relationships_per_entity"] = 0.0

    # --- Orphan entities (no relationships) ---
    rows = await neo4j_service.run_cypher(
        "MATCH (e:Entity) WHERE NOT (e)-[:semantic_link]-() RETURN count(e) AS c",
    )
    orphan_entities = rows[0]["c"] if rows else 0
    metrics["orphan_entities"] = orphan_entities

    # --- Entity type distribution ---
    rows = await neo4j_service.run_cypher(
        "MATCH (e:Entity) RETURN e.type AS type, count(e) AS count ORDER BY count DESC",
    )
    metrics["entity_type_distribution"] = {r["type"]: r["count"] for r in rows}

    # --- Communities without report ---
    rows = await neo4j_service.run_cypher(
        "MATCH (c:Community) WHERE c.summary IS NULL RETURN count(c) AS c",
    )
    metrics["communities_without_report"] = rows[0]["c"] if rows else 0

    # --- Communities by level ---
    rows = await neo4j_service.run_cypher(
        "MATCH (c:Community) RETURN c.level AS level, count(c) AS count ORDER BY level",
    )
    metrics["communities_by_level"] = {str(r["level"]): r["count"] for r in rows}

    # --- Community size stats ---
    rows = await neo4j_service.run_cypher(
        "MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity) "
        "RETURN c.id AS community_id, count(e) AS size",
    )
    if rows:
        sizes = [r["size"] for r in rows]
        metrics["largest_community_size"] = max(sizes)
        metrics["smallest_community_size"] = min(sizes)
        metrics["avg_community_size"] = round(sum(sizes) / len(sizes), 2)
    else:
        metrics["largest_community_size"] = 0
        metrics["smallest_community_size"] = 0
        metrics["avg_community_size"] = 0.0

    # --- Modularity score placeholder (requires leiden result storage) ---
    metrics["modularity_score"] = None

    return metrics


def compute_extraction_stats(
    input_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute per-run extraction stats (no Neo4j needed — works on DataFrames).

    Args:
        input_df: DataFrame with at least a ``source_id`` or ``id`` column (text chunks).
        entities_df: DataFrame of extracted entities.
        relationships_df: DataFrame of extracted relationships.

    Returns:
        Dictionary of per-run statistics.
    """
    total_chunks = len(input_df)

    # Determine the column name for chunk IDs
    source_col = "source_id" if "source_id" in entities_df.columns else None
    if source_col and entities_df[source_col].nunique() > 0:
        chunks_with_entities = entities_df[source_col].nunique()
    else:
        chunks_with_entities = 0

    entity_coverage_ratio = round(chunks_with_entities / total_chunks, 4) if total_chunks > 0 else 0.0

    total_entities = len(entities_df)
    total_relationships = len(relationships_df)

    stats = {
        "total_chunks": total_chunks,
        "chunks_with_entities": chunks_with_entities,
        "entity_coverage_ratio": entity_coverage_ratio,
        "total_entities_extracted": total_entities,
        "total_relationships_extracted": total_relationships,
        "avg_entities_per_chunk": round(total_entities / total_chunks, 2) if total_chunks > 0 else 0.0,
        "avg_relationships_per_chunk": round(total_relationships / total_chunks, 2) if total_chunks > 0 else 0.0,
    }
    return stats


def evaluate_entity_quality(entities_df: pd.DataFrame) -> Dict[str, Any]:
    """Quick quality heuristics on extracted entities DataFrame.

    Args:
        entities_df: DataFrame with ``title`` and ``description`` columns.

    Returns:
        Dictionary of quality metrics.
    """
    # Entities without description
    if "description" in entities_df.columns:
        entities_without_description = int(
            entities_df["description"].isna() | (entities_df["description"].astype(str).str.strip() == "")
        )
    else:
        entities_without_description = len(entities_df)

    # Average title length
    if "title" in entities_df.columns:
        avg_title_length = round(entities_df["title"].astype(str).str.len().mean(), 2)
    else:
        avg_title_length = 0.0

    # Average description length
    if "description" in entities_df.columns:
        avg_description_length = round(
            entities_df["description"].astype(str).str.len().mean(), 2
        )
    else:
        avg_description_length = 0.0

    # Duplicate titles
    if "title" in entities_df.columns:
        duplicate_titles = int(entities_df.duplicated(subset=["title"]).sum())
    else:
        duplicate_titles = 0

    return {
        "entities_without_description": entities_without_description,
        "avg_title_length": avg_title_length,
        "avg_description_length": avg_description_length,
        "duplicate_titles": duplicate_titles,
    }
