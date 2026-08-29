#!/usr/bin/env python3
"""Print per-document Neo4j statistics.

For every ``:Document`` node the script reports how many nodes of each type are
available, attributing nodes to documents the same way the rest of the codebase
does (``inspect_delete_targets.py``):

* ``:Section``  — ``(d:Document)-[:SECTION]->(s:Section)``
* ``:Region``   — ``region_id`` prefix ``'{file_hash}|'``, broken down by region
  subtype label (text / title / image / image_caption / table / table_caption /
  equation / image_footnote / table_footnote)
* ``:Entity``   — ``text_unit_ids`` contains ``'{file_hash}|...'``
* ``:Community`` — reached via ``Community -[:CONSISTS_OF]-> Entity`` whose
  ``text_unit_ids`` reference the document (communities carry no ``document_id``)

Entities and communities are shared across documents: an entity/community that
references several documents is counted once *per* document. The per-document
tables therefore reflect "what is reachable from this document", and their
totals can exceed the global unique node counts.

Usage:
    python neo4j_stats.py
    python neo4j_stats.py --json
    python neo4j_stats.py --uri bolt://neo4j:7687 --user neo4j --password password

Connection settings are read from ``NEO4J_URI`` / ``NEO4J_USER`` /
``NEO4J_PASSWORD`` / ``NEO4J_DB`` environment variables (falling back to
``bolt://localhost:7687``, ``neo4j`` / ``password``, database ``neo4j``).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from neo4j import GraphDatabase


# Region subtype labels observed in the graph (second label on :Region nodes).
REGION_TYPES = [
    "text",
    "title",
    "image",
    "image_caption",
    "table",
    "table_caption",
    "equation",
    "image_footnote",
    "table_footnote",
]


def build_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


def query_rows(driver, cypher: str, db: Optional[str] = None, **params) -> List[Dict]:
    """Run a Cypher query and return a list of dicts (one per row)."""
    with driver.session(database=db) as session:
        result = session.run(cypher, **params)
        return [record.data() for record in result]


def collect_stats(driver, db: Optional[str] = None) -> Dict:
    # 1. Total documents + ordered list of unique document hashes.
    #    A duplicate :Document node would otherwise double-count its subtree,
    #    so we deduplicate by name and keep the duplicates for reporting.
    doc_rows = query_rows(
        driver,
        "MATCH (d:Document) RETURN d.name AS hash ORDER BY d.name",
        db=db,
    )
    all_names = [r["hash"] for r in doc_rows]
    name_counts = Counter(all_names)
    hashes = list(name_counts)  # unique, insertion (sorted) order
    duplicates = sorted(n for n, c in name_counts.items() if c > 1)

    # 2. Sections per document (unique Section nodes per document name).
    section_rows = query_rows(
        driver,
        """
        MATCH (d:Document)-[:SECTION]->(s:Section)
        RETURN d.name AS hash, count(DISTINCT s) AS sections
        """,
        db=db,
    )
    sections = {r["hash"]: r["sections"] for r in section_rows}

    # 3. Regions per document, broken down by subtype label.
    region_rows = query_rows(
        driver,
        """
        MATCH (r:Region)
        WHERE r.region_id IS NOT NULL AND r.region_id CONTAINS '|'
        WITH r,
             split(r.region_id, '|')[0] AS hash,
             coalesce([l IN labels(r) WHERE l <> 'Region'][0], 'other') AS rtype
        RETURN hash, rtype, count(*) AS c
        """,
        db=db,
    )
    regions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in region_rows:
        regions[r["hash"]][r["rtype"]] += r["c"]

    # 4. Entities per document.
    entity_rows = query_rows(
        driver,
        """
        MATCH (e:Entity)
        WHERE e.text_unit_ids IS NOT NULL
        UNWIND e.text_unit_ids AS tid
        WITH e, tid
        WHERE tid CONTAINS '|'
        WITH split(tid, '|')[0] AS hash, e
        RETURN hash, count(DISTINCT e) AS c
        """,
        db=db,
    )
    entities = {r["hash"]: r["c"] for r in entity_rows}

    # 5. Communities per document (via CONSISTS_OF -> Entity -> text_unit_ids).
    community_rows = query_rows(
        driver,
        """
        MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)
        WHERE e.text_unit_ids IS NOT NULL
        UNWIND e.text_unit_ids AS tid
        WITH c, tid
        WHERE tid CONTAINS '|'
        WITH split(tid, '|')[0] AS hash, c
        RETURN hash, count(DISTINCT c) AS c
        """,
        db=db,
    )
    communities = {r["hash"]: r["c"] for r in community_rows}

    # 6. Global unique node counts (each node counted once).
    global_rows = query_rows(
        driver,
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS c
        """,
        db=db,
    )
    global_counts = {r["label"]: r["c"] for r in global_rows}

    stats = {
        "total_documents": len(hashes),
        "total_document_nodes": len(all_names),
        "duplicate_documents": duplicates,
        "global": global_counts,
        "documents": [],
    }

    for h in hashes:
        region_breakdown = {
            t: regions.get(h, {}).get(t, 0) for t in REGION_TYPES
        }
        region_total = sum(region_breakdown.values())
        stats["documents"].append(
            {
                "hash": h,
                "sections": sections.get(h, 0),
                "regions": region_total,
                "region_types": region_breakdown,
                "entities": entities.get(h, 0),
                "communities": communities.get(h, 0),
            }
        )

    # Global totals (sum over documents).
    totals = {
        "sections": sum(d["sections"] for d in stats["documents"]),
        "regions": sum(d["regions"] for d in stats["documents"]),
        "entities": sum(d["entities"] for d in stats["documents"]),
        "communities": sum(d["communities"] for d in stats["documents"]),
    }
    totals["region_types"] = {
        t: sum(d["region_types"][t] for d in stats["documents"])
        for t in REGION_TYPES
    }
    stats["totals"] = totals

    return stats


def format_table(header: List[str], rows: List[List[str]]) -> str:
    """Render a fixed-width, space-padded table."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  ".join("-" * w for w in widths)
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
        sep,
    ]
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        )
    return "\n".join(lines)


def print_text(stats: Dict) -> None:
    print(f"Documents loaded : {stats['total_documents']}")
    if stats.get("duplicate_documents"):
        print(
            "  WARNING: duplicate :Document name(s) found — "
            + ", ".join(stats["duplicate_documents"])
        )

    g = stats.get("global", {})
    print(
        "Global nodes     : "
        + ", ".join(
            f"{k}={v}" for k, v in sorted(g.items())
        )
    )
    print()

    docs = stats["documents"]
    if not docs:
        print("No documents found.")
        return

    # Table 1: core node types.
    header = [
        "Document", "Sections", "Regions", "Entities", "Communities",
    ]
    rows = [
        [
            d["hash"],
            str(d["sections"]),
            str(d["regions"]),
            str(d["entities"]),
            str(d["communities"]),
        ]
        for d in docs
    ]
    t = stats["totals"]
    rows.append([
        "TOTAL",
        str(t["sections"]),
        str(t["regions"]),
        str(t["entities"]),
        str(t["communities"]),
    ])
    print(format_table(header, rows))
    print()

    # Table 2: region subtype breakdown.
    header2 = ["Document"] + REGION_TYPES
    rows2 = [
        [d["hash"]] + [str(d["region_types"][t]) for t in REGION_TYPES]
        for d in docs
    ]
    rows2.append(
        ["TOTAL"] + [str(t["region_types"][rt]) for rt in REGION_TYPES]
    )
    print(format_table(header2, rows2))
    print()
    print(
        "Note: Entities/Communities are shared across documents and are counted "
        "once per document, so the TOTAL row can exceed the global unique counts."
    )


def print_json(stats: Dict) -> None:
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--db", default=os.environ.get("NEO4J_DB", "neo4j"))
    parser.add_argument("--json", action="store_true", help="Output JSON instead of tables")
    args = parser.parse_args()

    driver = build_driver(args.uri, args.user, args.password)
    try:
        stats = collect_stats(driver, db=args.db)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to query Neo4j: {e}", file=sys.stderr)
        return 1
    finally:
        driver.close()

    if args.json:
        print_json(stats)
    else:
        print_text(stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
