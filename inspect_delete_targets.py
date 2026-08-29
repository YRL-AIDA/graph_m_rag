#!/usr/bin/env python3
"""
Find documents whose semantic graph was never built and, optionally, delete
every artifact related to them from **Neo4j**, **MinIO** and **Qdrant**.

Rationale
---------
A document's lifecycle spans three stores, all keyed by the same MD5 ``file_hash``:

* MinIO   — ``pdfs/{file_hash}_{filename}/{filename}`` (the source PDF) plus
            ``mineru_results/{file_hash}*`` and ``embeddings/{file_hash}/...``.
* Neo4j   — the structural graph: ``(:Document {name: file_hash})`` →
            ``(:Region {region_id: '{file_hash}|{id}'})`` and ``(:Section)``,
            linked via ``ORDER`` / ``PARENT`` / ``SECTION``.
            The semantic graph sits on top of it: ``(:Entity)`` →
            ``[:semantic_link]`` → ``(:Region)`` and
            ``(:Community {document_id: file_hash})`` → ``[:CONSISTS_OF]`` →
            ``(:Entity)``.
* Qdrant  — vector points whose payload carries ``file_hash`` (one per parsed
            element), saved by ``QdrantClientWrapper.save_embeddings``.

A document whose semantic graph was not built has a PDF in MinIO but **zero
``Entity`` nodes** referencing it in Neo4j (``Entity.text_unit_ids`` contains no
``'{file_hash}|...'`` entry). Such a document may still have a full structural
graph (Regions/Sections) and Qdrant points, so all three stores must be cleaned
up. This script finds those documents and, when run with ``--execute``, removes:

  1. from Neo4j   — the ``:Document`` node, its ``:Region``/``:Section`` subtree,
                    ``semantic_link`` edges, ``:Community`` nodes tagged with the
                    document, and any ``:Entity`` that becomes orphaned;
  2. from MinIO   — every object under ``pdfs/``, ``mineru_results/`` and
                    ``embeddings/`` for that hash;
  3. from Qdrant  — every point matching ``file_hash`` across all collections.

Note: ``images/`` objects are keyed by the image content (not by ``file_hash``)
and may be shared across documents, so they are intentionally left untouched.

Usage
-----
    python inspect_delete_targets.py             # dry-run (default): list targets only
    python inspect_delete_targets.py --execute   # actually delete from Neo4j + MinIO + Qdrant

Configuration is read from the environment / ``.env``:

* MinIO:  ``S3_URL``, ``S3_ACCESS_KEY``, ``S3_SECRET_KEY``, ``S3_BUCKET_NAME``.
* Neo4j:  ``URL`` (host:port), ``USER_NEO4J``, ``PASSWORD``, ``NAME_DB``.
* Qdrant: ``QDRANT_HOST``, ``QDRANT_PORT``, ``QDRANT_GRPC_PORT``,
          ``QDRANT_API_KEY``, ``QDRANT_COLLECTION_NAME``.
"""

import argparse
import re
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from qdrant_client.http import models

from app.src.minio_client import MinioClient
from app.src.qdrant_client_api import QdrantClientWrapper
from documet_index import DocumentIndexService

# MD5 hex digest: 32 lowercase hex characters.
HASH_RE = re.compile(r"^[0-9a-f]{32}$")

# MinIO prefixes that hold per-document artifacts, all keyed by file_hash.
PREFIXES = ("pdfs/", "mineru_results/", "embeddings/")

# Qdrant payload field that identifies the owning document.
FILE_HASH_FIELD = "file_hash"


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def minio_pdf_hashes(minio: MinioClient) -> set:
    """Return the set of file_hashes present as PDFs in MinIO."""
    hashes = set()
    for name in minio.list_objects(minio.bucket_name, prefix="pdfs/"):
        # Object layout: pdfs/{file_hash}_{filename}/{filename}
        parts = name.split("/")
        if len(parts) < 2:
            continue
        dir_name = parts[1]
        if "_" in dir_name:
            candidate = dir_name.split("_", 1)[0]
            if HASH_RE.match(candidate):
                hashes.add(candidate)
    return hashes


def objects_for_hash(minio: MinioClient, file_hash: str) -> list:
    """Return every MinIO object belonging to *file_hash*."""
    objects = []
    for prefix in PREFIXES:
        objects.extend(
            minio.list_objects(minio.bucket_name, prefix=f"{prefix}{file_hash}")
        )
    return objects


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def _hash_filter(file_hash: str) -> models.Filter:
    """Build a Qdrant filter selecting every point of *file_hash*."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key=FILE_HASH_FIELD,
                match=models.MatchValue(value=file_hash),
            )
        ]
    )


def qdrant_count_for_hash(qdrant: QdrantClientWrapper, file_hash: str) -> int:
    """Return the number of Qdrant points matching *file_hash* across all collections."""
    qfilter = _hash_filter(file_hash)
    total = 0
    for collection in qdrant.list_collections():
        cw = QdrantClientWrapper(collection_name=collection)
        try:
            result = cw.client.count(
                collection_name=collection,
                count_filter=qfilter,
                exact=True,
            )
            total += result.count
        except Exception:
            # Collection may lack the file_hash field or be unavailable.
            pass
    return total


def qdrant_delete_for_hash(qdrant: QdrantClientWrapper, file_hash: str) -> list:
    """Delete all Qdrant points matching *file_hash*; return list of failed collections."""
    failures = []
    for collection in qdrant.list_collections():
        cw = QdrantClientWrapper(collection_name=collection)
        if not cw.delete_points_by_file_hash(file_hash):
            failures.append(collection)
    return failures


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _scalar(service: DocumentIndexService, cypher: str, **params) -> int:
    """Run a Cypher query and return the single integer in its first row."""
    try:
        rows = service.manager.query(cypher, params=params)
    except Exception as exc:  # noqa: BLE001
        print(f"      WARN: Neo4j query failed: {exc}", file=sys.stderr)
        return 0
    if not rows:
        return 0
    data = rows[0].data()
    if not data:
        return 0
    return int(next(iter(data.values()), 0))


def neo4j_document_count(service: DocumentIndexService, file_hash: str) -> int:
    return _scalar(
        service,
        "MATCH (d:Document {name: $hash}) RETURN count(d) AS c",
        hash=file_hash,
    )


def neo4j_region_count(service: DocumentIndexService, file_hash: str) -> int:
    return _scalar(
        service,
        "MATCH (r:Region) WHERE r.region_id STARTS WITH $prefix RETURN count(r) AS c",
        prefix=f"{file_hash}|",
    )


def neo4j_section_count(service: DocumentIndexService, file_hash: str) -> int:
    return _scalar(
        service,
        "MATCH (s:Section) WHERE s.section_id STARTS WITH $hash RETURN count(s) AS c",
        hash=file_hash,
    )


def neo4j_entity_count(service: DocumentIndexService, file_hash: str) -> int:
    """Count Entity nodes whose ``text_unit_ids`` reference this document.

    This is the canonical per-document entity membership check used by
    ``connect_graphs.create_semantic_links`` (``tid STARTS WITH '{hash}|'``).
    """
    return _scalar(
        service,
        """
        MATCH (e:Entity)
        WHERE e.text_unit_ids IS NOT NULL
          AND ANY(tid IN e.text_unit_ids WHERE tid STARTS WITH $prefix)
        RETURN count(e) AS c
        """,
        prefix=f"{file_hash}|",
    )


def neo4j_community_count(service: DocumentIndexService, file_hash: str) -> int:
    return _scalar(
        service,
        "MATCH (c:Community) WHERE c.document_id = $hash RETURN count(c) AS c",
        hash=file_hash,
    )


def neo4j_summary(service: DocumentIndexService, file_hash: str) -> Dict[str, int]:
    """Return a snapshot of the document's footprint in Neo4j."""
    return {
        "document": neo4j_document_count(service, file_hash),
        "regions": neo4j_region_count(service, file_hash),
        "sections": neo4j_section_count(service, file_hash),
        "entities": neo4j_entity_count(service, file_hash),
        "communities": neo4j_community_count(service, file_hash),
    }


def neo4j_delete_for_hash(service: DocumentIndexService, file_hash: str) -> Dict[str, int]:
    """Delete every Neo4j node/edge belonging to *file_hash*.

    Order matters:

    1. Regions first — ``DETACH DELETE`` on a Region removes its ``ORDER``,
       ``PARENT``, ``SECTION`` and incoming ``semantic_link`` edges.
    2. Sections next (their ``SECTION`` edges to Regions are already gone).
    3. The ``:Document`` node.
    4. ``:Community`` nodes tagged with ``document_id == file_hash``.
    5. Orphaned ``:Entity`` nodes — only those whose *every* ``text_unit_ids``
       entry belongs to this document and that no longer link to any Region.
       Shared entities (``text_unit_ids`` spanning several documents) are kept.
    """
    prefix = f"{file_hash}|"
    stats: Dict[str, int] = {}

    stats["regions"] = neo4j_region_count(service, file_hash)
    service.manager.query(
        "MATCH (r:Region) WHERE r.region_id STARTS WITH $prefix DETACH DELETE r",
        params={"prefix": prefix},
    )

    stats["sections"] = neo4j_section_count(service, file_hash)
    service.manager.query(
        "MATCH (s:Section) WHERE s.section_id STARTS WITH $hash DETACH DELETE s",
        params={"hash": file_hash},
    )

    stats["document"] = neo4j_document_count(service, file_hash)
    service.manager.query(
        "MATCH (d:Document {name: $hash}) DETACH DELETE d",
        params={"hash": file_hash},
    )

    stats["communities"] = neo4j_community_count(service, file_hash)
    service.manager.query(
        "MATCH (c:Community) WHERE c.document_id = $hash DETACH DELETE c",
        params={"hash": file_hash},
    )

    stats["entities"] = neo4j_entity_count(service, file_hash)
    service.manager.query(
        """
        MATCH (e:Entity)
        WHERE e.text_unit_ids IS NOT NULL
          AND ALL(tid IN e.text_unit_ids WHERE tid STARTS WITH $prefix)
          AND NOT (e)-[:semantic_link]->(:Region)
        DETACH DELETE e
        """,
        params={"prefix": prefix},
    )

    return stats


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def find_targets(
    service: DocumentIndexService, minio_hashes: set
) -> List[tuple]:
    """Return ``(file_hash, summary)`` for every document with no Entity nodes
    (i.e. semantic graph not built), ordered by hash."""
    targets = []
    for h in sorted(minio_hashes):
        summary = neo4j_summary(service, h)
        if summary["entities"] == 0:
            targets.append((h, summary))
    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sample(objects: list, limit: int = 3) -> List[str]:
    return objects[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete targets from Neo4j, MinIO and Qdrant (default is dry-run).",
    )
    args = parser.parse_args()

    try:
        minio = MinioClient()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to connect to MinIO: {e}", file=sys.stderr)
        return 1

    try:
        service = DocumentIndexService()
    except Exception as e:  # noqa: BLE001
        print(
            f"ERROR: failed to connect to Neo4j: {e}\n"
            "Check env vars URL, USER_NEO4J, PASSWORD, NAME_DB.",
            file=sys.stderr,
        )
        return 1

    qdrant: Optional[QdrantClientWrapper] = None
    try:
        qdrant = QdrantClientWrapper()
        # Force a real connection (QdrantClient is lazy until the first call).
        qdrant.client.get_collections()
    except Exception as e:  # noqa: BLE001
        print(
            f"ERROR: failed to connect to Qdrant: {e}\n"
            "Check env vars QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY.",
            file=sys.stderr,
        )
        service.close()
        return 1

    try:
        minio_hashes = minio_pdf_hashes(minio)
        targets = find_targets(service, minio_hashes)

        print(f"MinIO PDF documents : {len(minio_hashes)}")
        print(f"Semantic-graph-less : {len(targets)}")

        if not targets:
            print("Nothing to delete.")
            return 0

        total_objects = 0
        total_points = 0
        for h, summary in targets:
            objs = objects_for_hash(minio, h)
            points = qdrant_count_for_hash(qdrant, h)
            total_objects += len(objs)
            total_points += points

            print(f"\nTARGET {h}")
            print(
                f"  Neo4j : document={summary['document']}, regions={summary['regions']}, "
                f"sections={summary['sections']}, entities={summary['entities']}, "
                f"communities={summary['communities']}"
            )
            print(f"  MinIO : {len(objs)} object(s)")
            for o in _sample(objs):
                print(f"      {o}")
            if len(objs) > 3:
                print(f"      ... and {len(objs) - 3} more")
            print(f"  Qdrant: {points} point(s)")

        if not args.execute:
            print(
                f"\nDRY-RUN: would delete everything related to {len(targets)} "
                "document(s):\n"
                f"  - Neo4j : {len(targets)} Document subtree(s) + Regions/Sections "
                f"({sum(s['regions'] for _, s in targets)} regions, "
                f"{sum(s['sections'] for _, s in targets)} sections) "
                f"+ Communities + orphaned Entities\n"
                f"  - MinIO : {total_objects} object(s)\n"
                f"  - Qdrant: {total_points} point(s)\n"
                "Re-run with --execute to actually delete."
            )
            return 0

        for h, summary in targets:
            print(f"\nDELETING {h} ...")

            try:
                nstats = neo4j_delete_for_hash(service, h)
                print(
                    f"  DELETED (Neo4j): document={nstats['document']}, "
                    f"regions={nstats['regions']}, sections={nstats['sections']}, "
                    f"communities={nstats['communities']}, entities={nstats['entities']}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED  (Neo4j): {e}", file=sys.stderr)

            for o in objects_for_hash(minio, h):
                try:
                    minio.remove_object(minio.bucket_name, o)
                    print(f"  DELETED (MinIO) {o}")
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED  (MinIO) {o}: {e}", file=sys.stderr)

            failures = qdrant_delete_for_hash(qdrant, h)
            if failures:
                print(
                    f"  FAILED  (Qdrant) {h}: could not delete from collections "
                    f"{failures}",
                    file=sys.stderr,
                )
            else:
                print(f"  DELETED (Qdrant) points for {h}")

        print(f"\nDone. Purged {len(targets)} document(s) across Neo4j, MinIO and Qdrant.")
        return 0
    finally:
        service.close()
        if qdrant is not None:
            qdrant.close()


if __name__ == "__main__":
    sys.exit(main())
