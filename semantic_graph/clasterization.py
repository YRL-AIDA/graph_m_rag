

import pandas as pd
import logging
import html
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

import numpy as np
import graspologic_native as gn
from collections import defaultdict
logger = logging.getLogger(__name__)
ID = "id"
SHORT_ID = "human_readable_id"
TITLE = "title"
DESCRIPTION = "description"

TYPE = "type"



# COMMUNITY HIERARCHY TABLE SCHEMA
SUB_COMMUNITY = "sub_community"

# COMMUNITY CONTEXT TABLE SCHEMA
ALL_CONTEXT = "all_context"
CONTEXT_STRING = "context_string"
CONTEXT_SIZE = "context_size"
CONTEXT_EXCEED_FLAG = "context_exceed_limit"

# COMMUNITY REPORT TABLE SCHEMA
COMMUNITY_ID = "community"
COMMUNITY_LEVEL = "level"
COMMUNITY_PARENT = "parent"
COMMUNITY_CHILDREN = "children"
TITLE = "title"
SUMMARY = "summary"
FINDINGS = "findings"
RATING = "rank"
EXPLANATION = "rating_explanation"
FULL_CONTENT = "full_content"
FULL_CONTENT_JSON = "full_content_json"

ENTITY_IDS = "entity_ids"
RELATIONSHIP_IDS = "relationship_ids"
TEXT_UNIT_IDS = "text_unit_ids"
COVARIATE_IDS = "covariate_ids"
DOCUMENT_ID = "document_id"

PERIOD = "period"
SIZE = "size"
DEGREE = "degree"

COMMUNITIES_FINAL_COLUMNS = [
    ID,
    SHORT_ID,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    COMMUNITY_CHILDREN,
    TITLE,
    ENTITY_IDS,
    RELATIONSHIP_IDS,
    PERIOD,
    SIZE,
]
Communities = list[tuple[int, int, int, list[str]]]
def cluster_graph(
    edges: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a relationships DataFrame."""
    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        edges=edges,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result: dict[int, list[str]] = defaultdict(list)
        clusters[level] = result
        for node_id, community_id in node_id_to_community_map[level].items():
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results

def _compute_leiden_communities(
    edges: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    edge_df = edges.copy()

    # Normalize edge direction and deduplicate (undirected graph).
    # NX deduplicates reversed pairs keeping the last row's attributes,
    # so we replicate that by normalizing direction then keeping last.
    lo = edge_df[["source", "target"]].min(axis=1)
    hi = edge_df[["source", "target"]].max(axis=1)
    edge_df["source"] = lo
    edge_df["target"] = hi
    edge_df.drop_duplicates(subset=["source", "target"], keep="last", inplace=True)

#    if use_lcc:
 #       edge_df = stable_lcc(edge_df)

    weights = (
        edge_df["weight"].astype(float)
        if "weight" in edge_df.columns
        else pd.Series(1.0, index=edge_df.index)
    )
    edge_list: list[tuple[str, str, float]] = sorted(
        zip(
            edge_df["source"].astype(str),
            edge_df["target"].astype(str),
            weights,
            strict=True,
        )
    )

    community_mapping = hierarchical_leiden(
        edge_list, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy



def hierarchical_leiden(
    edges: list[tuple[str, str, float]],
    max_cluster_size: int = 10,
    random_seed: int | None = 0xDEADBEEF,
) -> list[gn.HierarchicalCluster]:
    """Run hierarchical leiden on an edge list."""
    return gn.hierarchical_leiden(
        edges=edges,
        max_cluster_size=max_cluster_size,
        seed=random_seed,
        starting_communities=None,
        resolution=1.0,
        randomness=0.001,
        use_modularity=True,
        iterations=1,
    )


def first_level_hierarchical_clustering(
    hcs: list[gn.HierarchicalCluster],
) -> dict[Any, int]:
    """Return the initial leiden clustering as a dict of node id to community id.

    Returns
    -------
    dict[Any, int]
        The initial leiden algorithm clustering results as a dictionary
        of node id to community id.
    """
    return {entry.node: entry.cluster for entry in hcs if entry.level == 0}


def final_level_hierarchical_clustering(
    hcs: list[gn.HierarchicalCluster],
) -> dict[Any, int]:
    """Return the final leiden clustering as a dict of node id to community id.

    Returns
    -------
    dict[Any, int]
        The last leiden algorithm clustering results as a dictionary
        of node id to community id.
    """
    return {entry.node: entry.cluster for entry in hcs if entry.is_final_cluster}


def stable_lcc(
    relationships: pd.DataFrame,
    source_column: str = "source",
    target_column: str = "target",
) -> pd.DataFrame:
    """Return the relationships DataFrame filtered to a stable largest connected component.

    Parameters
    ----------
    relationships : pd.DataFrame
        Edge list with at least source and target columns.
    source_column : str
        Name of the source node column.
    target_column : str
        Name of the target node column.

    Returns
    -------
    pd.DataFrame
        A copy of the input filtered to the LCC with normalized node names
        and deterministic edge ordering.
    """
    if relationships.empty:
        return relationships.copy()

    # 1. Normalize node names
    edges = relationships.copy()
    edges[source_column] = edges[source_column].apply(_normalize_name)
    edges[target_column] = edges[target_column].apply(_normalize_name)

    # 2. Filter to the largest connected component
    lcc_nodes = largest_connected_component(
        edges, source_column=source_column, target_column=target_column
    )
    edges = edges[
        edges[source_column].isin(lcc_nodes) & edges[target_column].isin(lcc_nodes)
    ]

    # 3. Stabilize edge direction: lesser node always first
    swapped = edges[source_column] > edges[target_column]
    edges.loc[swapped, [source_column, target_column]] = edges.loc[
        swapped, [target_column, source_column]
    ].to_numpy()

    # 4. Deduplicate edges that were reversed pairs in the original data
    edges = edges.drop_duplicates(subset=[source_column, target_column])

    # 5. Sort for deterministic order
    return edges.sort_values([source_column, target_column]).reset_index(drop=True)


def _normalize_name(name: str) -> str:
    """Normalize a node name: HTML unescape, uppercase, strip whitespace."""
    return html.unescape(name).upper().strip()

def connected_components(
    relationships: pd.DataFrame,
    source_column: str = "source",
    target_column: str = "target",
) -> list[set[str]]:
    """Return all connected components as a list of node-title sets.

    Uses union-find on the deduplicated edge list.

    Parameters
    ----------
    relationships : pd.DataFrame
        Edge list with at least source and target columns.
    source_column : str
        Name of the source node column.
    target_column : str
        Name of the target node column.

    Returns
    -------
    list[set[str]]
        Each element is a set of node titles belonging to one component,
        sorted by descending component size.
    """
    edges = relationships.drop_duplicates(subset=[source_column, target_column])

    # Initialize every node as its own parent
    all_nodes = pd.concat(
        [edges[source_column], edges[target_column]], ignore_index=True
    ).unique()
    parent: dict[str, str] = {node: node for node in all_nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Union each edge
    for src, tgt in zip(edges[source_column], edges[target_column], strict=True):
        union(src, tgt)

    # Group by root
    groups: dict[str, set[str]] = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, set()).add(node)

    return sorted(groups.values(), key=len, reverse=True)


def largest_connected_component(
    relationships: pd.DataFrame,
    source_column: str = "source",
    target_column: str = "target",
) -> set[str]:
    """Return the node titles belonging to the largest connected component.

    Parameters
    ----------
    relationships : pd.DataFrame
        Edge list with at least source and target columns.
    source_column : str
        Name of the source node column.
    target_column : str
        Name of the target node column.

    Returns
    -------
    set[str]
        The set of node titles in the largest connected component.
    """
    components = connected_components(
        relationships,
        source_column=source_column,
        target_column=target_column,
    )
    if not components:
        return set()
    return components[0]


async def create_communities(
    relationships: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build communities from clustered relationships and stream rows to the table.

    Args
    ----
        communities_table: Table
            Output table to write community rows to.
        entities_table: Table
            Table containing entity rows.
        relationships: pd.DataFrame
            Relationships DataFrame with source, target, weight,
            text_unit_ids columns.
        max_cluster_size: int
            Maximum cluster size for hierarchical Leiden.
        use_lcc: bool
            Whether to restrict to the largest connected component.
        seed: int | None
            Random seed for deterministic clustering.

    Returns
    -------
        list[dict[str, Any]]
            Sample of up to 5 community rows for logging.
    """
    # clusters содержит список кортежей вида (level, community, parent, [title]), где:
    # - level: уровень иерархии (int)
    # - community: идентификатор комьюнити (int)
    # - parent: идентификатор родительского комьюнити (int или None)
    # - [title]: список id сущностей (или заголовков/nodes), входящих в сообщество на этом уровне
    clusters = cluster_graph(
        relationships,
        max_cluster_size,
        use_lcc,
        seed=seed,
    )
    print(clusters)


    communities = pd.DataFrame(
        clusters, columns=pd.Index(["level", "community", "parent", "title"])
    ).explode("title")
    communities["community"] = communities["community"].astype(int)

    # aggregate entity ids for each community
    entity_map = communities[["community", "title"]].copy()
    entity_map["entity_id"] = entity_map["title"]
    # entity_ids формируется как DataFrame со столбцами:
    # - community: идентификатор комьюнити (int)
    # - entity_ids: список id сущностей, входящих в данное комьюнити (list[str])
    entity_ids = (
        entity_map
        .dropna(subset=["entity_id"])
        .groupby("community")
        .agg(entity_ids=("entity_id", list))
        .reset_index()
    )

    # aggregate relationship ids per community, limited to
    # intra-community edges (source and target in the same community).
    # Process one hierarchy level at a time to keep intermediate
    # DataFrames small, then concat the grouped results once at the end.
    level_results = []
    for level in communities["level"].unique():
        level_comms = communities[communities["level"] == level]
        with_source = relationships.merge(
            level_comms, left_on="source", right_on="title", how="inner"
        )
        with_both = with_source.merge(
            level_comms, left_on="target", right_on="title", how="inner"
        )
        intra = with_both[with_both["community_x"] == with_both["community_y"]]
        if intra.empty:
            continue
        grouped = (
            intra
            .explode("text_unit_ids")
            .groupby(["community_x", "parent_x"])
            .agg(
                relationship_ids=("id", list),
                text_unit_ids=("text_unit_ids", list),
            )
            .reset_index()
        )
        grouped["level"] = level
        level_results.append(grouped)
    # level_results накапливает результаты для каждого уровня иерархии комьюнити. 
    # Каждый элемент в level_results — это DataFrame (grouped), в котором присутствуют следующие столбцы:
    # - community_x: идентификатор комьюнити, к которому относятся связи (int)
    # - parent_x: идентификатор родительского комьюнити (int)
    # - relationship_ids: список id связей (list[str]), входящих во внутрикомьюнити-ребра на этом уровне
    # - text_unit_ids: объединённый список id всех text_unit из соответствующих связей (list[str])
    # - level: уровень иерархии (int), для которого построены эти агрегации
    #
    # После конкатенации level_results превращается в единую таблицу (all_grouped) со столбцами:
    # - community (ранее community_x): идентификатор комьюнити, к которому относятся данные
    # - parent (ранее parent_x): идентификатор родителя комьюнити
    # - relationship_ids: все id связи внутри этого комьюнити
    # - text_unit_ids: все id text_unit для этих связей
    # - level: уровень иерархии

    all_grouped = pd.concat(level_results, ignore_index=True).rename(
        columns={
            "community_x": "community",
            "parent_x": "parent",
        }
    )

    # deduplicate the lists
    all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
        lambda x: sorted(set(x))
    )
    all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
        lambda x: sorted(set(x))
    )


    # join it all up and add some new fields
    final_communities = all_grouped.merge(entity_ids, on="community", how="inner")
    final_communities["id"] = [str(uuid4()) for _ in range(len(final_communities))]
    final_communities["human_readable_id"] = final_communities["community"]
    final_communities["title"] = "Community " + final_communities["community"].astype(
        str
    )
    final_communities["parent"] = final_communities["parent"].astype(int)
    # collect the children so we have a tree going both ways
    #parent_grouped = cast(
    #    "pd.DataFrame",
    #    final_communities.groupby("parent").agg(children=("community", "unique")),
    #)
    parent_grouped = final_communities.groupby("parent").agg(children=("community", "unique"))

    final_communities = final_communities.merge(
        parent_grouped,
        left_on="community",
        right_on="parent",
        how="left",
    )
    # replace NaN children with empty list
    final_communities["children"] = final_communities["children"].apply(
        lambda x: x if isinstance(x, np.ndarray) else []  # type: ignore
    )
    # add fields for incremental update tracking
    final_communities["period"] = datetime.now(timezone.utc).date().isoformat()
    final_communities["size"] = final_communities.loc[:, "entity_ids"].apply(len)

    output = final_communities.loc[:, COMMUNITIES_FINAL_COLUMNS]
    return [_sanitize_row(row) for row in output.to_dict("records")]



def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy types to native Python types for table serialization."""
    sanitized = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized