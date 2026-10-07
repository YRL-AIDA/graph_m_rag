"""
Script to connect structural and semantic graphs in Neo4j.

This script establishes connections between nodes in the structural graph
(created from document parsing) and nodes in the semantic graph
(created from LLM-extracted entities and relationships).
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_graphs_by_document(document_id: Optional[str] = None):
    """Connect semantic graph entities to structural graph elements based on document context.

    This function creates connections between:
    - Entities in the semantic graph and TextUnits in the structural graph
    - Communities in the semantic graph and Documents in the structural graph
    - Entities in the semantic graph and Document nodes directly
    - Entities in the semantic graph and structural Regions (images, tables, etc.)

    Args:
        document_id: Specific document ID to connect (if None, connects all documents)
    """
    try:
        # Import the document index service to access the Neo4j connection
        from documet_index.neo4j_service import DocumentIndexService

        # Create service instance to access the Neo4j connection
        structural_service = DocumentIndexService()

        params = {"document_id": document_id} if document_id else {}

        with structural_service.driver.session() as session:
            # Connect entities to text units that contain them
            entity_to_textunit_query = (
                "MATCH (e:Entity)\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})\n"
                "MATCH (tu)-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "WITH e, tu, d\n"
                "WHERE NOT (e)-[:LINKED_TO_TEXTUNIT]->(tu)\n"
                "MERGE (e)-[:LINKED_TO_TEXTUNIT {relationship_type: 'CONTAINED_IN', created_at: datetime(), document_id: d.name}]->(tu)\n"
                "RETURN count(e) AS relationships_created"
            )

            result1 = session.run(entity_to_textunit_query, **params)
            count1 = result1.single()["relationships_created"]
            logger.info(f"Created {count1} LINKED_TO_TEXTUNIT relationships between entities and text units for document {document_id or 'all documents'}")

            # Connect entities to documents containing them
            entity_to_document_query = (
                "MATCH (e:Entity)\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "WITH e, d\n"
                "WHERE NOT (e)-[:LINKED_TO_DOCUMENT]->(d)\n"
                "MERGE (e)-[:LINKED_TO_DOCUMENT {relationship_type: 'MENTIONED_IN', created_at: datetime(), document_id: d.name}]->(d)\n"
                "RETURN count(e) AS relationships_created"
            )

            result2 = session.run(entity_to_document_query, **params)
            count2 = result2.single()["relationships_created"]
            logger.info(f"Created {count2} LINKED_TO_DOCUMENT relationships between entities and documents for document {document_id or 'all documents'}")

            # Connect communities to documents based on contained entities
            community_to_document_query = (
                "MATCH (c:Community)\n"
                "MATCH (c)-[:CONSISTS_OF]->(e:Entity)-[:LINKED_TO_DOCUMENT]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "WITH c, d\n"
                "WHERE NOT (c)-[:CONNECTED_TO_DOCUMENT]->(d)\n"
                "MERGE (c)-[:CONNECTED_TO_DOCUMENT {relationship_type: 'REPRESENTS_CONTENT_OF', created_at: datetime(), document_id: d.name}]->(d)\n"
                "RETURN count(c) AS relationships_created"
            )

            result3 = session.run(community_to_document_query, **params)
            count3 = result3.single()["relationships_created"]
            logger.info(f"Created {count3} CONNECTED_TO_DOCUMENT relationships between communities and documents for document {document_id or 'all documents'}")

            # Connect entities to their corresponding structural elements (images, tables, etc.)
            entity_to_structure_query = (
                "MATCH (e:Entity)\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})\n"
                "MATCH (tu)-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "OPTIONAL MATCH (tu)<-[:ORDER]-(prev_region:Region)\n"
                "OPTIONAL MATCH (tu)-[:ORDER]->(next_region:Region)\n"
                "WITH e, tu, d, prev_region, next_region\n"
                "WHERE prev_region IS NOT NULL OR next_region IS NOT NULL\n"
                "FOREACH(r IN CASE WHEN prev_region IS NOT NULL THEN [prev_region] ELSE [] END |\n"
                "  MERGE (e)-[:LINKED_TO_STRUCTURE {relationship_type: 'MENTIONS_NEAR', created_at: datetime(), position: 'before', document_id: d.name}]->(r))\n"
                "FOREACH(r IN CASE WHEN next_region IS NOT NULL THEN [next_region] ELSE [] END |\n"
                "  MERGE (e)-[:LINKED_TO_STRUCTURE {relationship_type: 'MENTIONS_NEAR', created_at: datetime(), position: 'after', document_id: d.name}]->(r))\n"
                "RETURN count(e) AS relationships_processed"
            )

            result4 = session.run(entity_to_structure_query, **params)
            count4 = result4.single()["relationships_processed"]
            logger.info(f"Processed {count4} entities for LINKED_TO_STRUCTURE relationships for document {document_id or 'all documents'}")

            # Connect entities to specific structural elements like images and tables
            entity_to_images_tables_query = (
                "MATCH (e:Entity)\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})\n"
                "MATCH (tu)-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "\n"
                "// Find nearby images and tables in the structural graph\n"
                "OPTIONAL MATCH (tu)<-[:ORDER*0..5]-(region_before:Region)\n"
                "OPTIONAL MATCH (tu)-[:ORDER*0..5]->(region_after:Region)\n"
                "\n"
                "// Filter for images and tables only\n"
                "WITH e, tu, d,\n"
                "     [x IN collect(CASE WHEN region_before:Image OR region_before:Table THEN region_before END) WHERE x IS NOT NULL] +\n"
                "     [x IN collect(CASE WHEN region_after:Image OR region_after:Table THEN region_after END) WHERE x IS NOT NULL] AS filtered_regions\n"
                "UNWIND filtered_regions AS nr\n"
                "\n"
                "// Create connections to nearby images and tables\n"
                "MERGE (e)-[:MENTIONS_ELEMENT {relationship_type: 'MENTIONS_NEARBY', created_at: datetime(), document_id: d.name, distance: CASE WHEN nr.order > tu.order THEN nr.order - tu.order ELSE tu.order - nr.order END}]->(nr)\n"
                "\n"
                "RETURN count(nr) AS element_connections"
            )

            result5 = session.run(entity_to_images_tables_query, **params)
            count5 = result5.single()["element_connections"]
            logger.info(f"Created {count5} MENTIONS_ELEMENT relationships between entities and nearby structural elements for document {document_id or 'all documents'}")

            # NEW: Additional query to connect entities to ALL regions (not just images/tables) in proximity
            entity_to_all_regions_query = (
                "MATCH (e:Entity)\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})\n"
                "MATCH (tu)-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "\n"
                "// Find all nearby regions by order proximity (avoids path explosion)\n"
                "OPTIONAL MATCH (region:Region)-[:PART_OF]->(d)\n"
                "WHERE abs(region.order - tu.order) <= 3\n"
                "\n"
                "WITH e, tu, d, collect(DISTINCT region) AS all_nearby_regions\n"
                "UNWIND all_nearby_regions AS region\n"
                "WITH e, d, region\n"
                "WHERE region IS NOT NULL\n"
                "\n"
                "// Create connections to all nearby regions\n"
                "MERGE (e)-[:NEAR_REGION {relationship_type: 'PHYSICALLY_NEAR', created_at: datetime(), document_id: d.name}]->(region)\n"
                "\n"
                "RETURN count(region) AS region_connections"
            )

            result6 = session.run(entity_to_all_regions_query, **params)
            count6 = result6.single()["region_connections"]
            logger.info(f"Created {count6} NEAR_REGION relationships between entities and nearby regions for document {document_id or 'all documents'}")

        logger.info(f"Successfully connected structural and semantic graphs for document {document_id or 'all documents'}")

    except Exception as e:
        logger.error(f"Error connecting graphs: {e}")
        raise


def connect_structural_and_semantic_nodes(document_id: Optional[str] = None):
    """Enhanced function to connect structural and semantic graph nodes directly.

    This function creates more comprehensive connections between:
    - Structural regions (images, tables, captions, etc.) and semantic entities
    - Text units in structural graph and semantic entities
    - Documents in both graphs

    Args:
        document_id: Specific document ID to connect (if None, connects all documents)
    """
    try:
        # Import the document index service to access the Neo4j connection
        from documet_index.neo4j_service import DocumentIndexService

        # Create service instance to access the Neo4j connection
        structural_service = DocumentIndexService()

        params = {"document_id": document_id} if document_id else {}

        with structural_service.driver.session() as session:
            # Connect structural regions to semantic entities based on text content similarity or proximity
            structural_to_semantic_query = (
                "MATCH (sr:Region)  // Structural region (image, table, text, etc.)\n"
                "MATCH (sd:Document)  // Structural document\n"
                "WHERE ((sr)-[:PARENT]->(sd) OR (sd)-[:ORDER*]->(sr))\n"
                + ("AND sd.name = $document_id\n" if document_id else "") +
                "\n"
                "// Find semantic entities that might relate to this structural region\n"
                "// This is based on document context and potential text unit IDs\n"
                "MATCH (se:Entity)  // Semantic entity\n"
                "WHERE se.text_unit_ids IS NOT NULL\n"
                "\n"
                "// Check if any of the text units associated with the entity\n"
                "// are related to the same document as the structural region\n"
                "UNWIND se.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})-[:PART_OF]->(sd2:Document)\n"
                "WHERE sd.name = sd2.name  // Same document\n"
                "\n"
                "// Create a connection between the structural region and semantic entity\n"
                "OPTIONAL MATCH (sr)-[existing_rel:CONNECTS_TO]->(se)\n"
                "WITH sr, se, sd, existing_rel\n"
                "WHERE existing_rel IS NULL\n"
                "MERGE (sr)-[:CONNECTS_TO {relationship_type: 'STRUCTURAL_SEMANTIC_LINK', created_at: datetime(), document_id: sd.name}]->(se)\n"
                "\n"
                "RETURN count(sr) AS structural_nodes_connected"
            )

            result1 = session.run(structural_to_semantic_query, **params)
            count1 = result1.single()["structural_nodes_connected"]
            logger.info(f"Connected {count1} structural regions to semantic entities for document {document_id or 'all documents'}")

            # NEW: Directly connect entities to regions that appear in the same document and nearby text units
            entity_to_region_query = (
                "MATCH (e:Entity)  // Semantic entity\n"
                "WHERE e.text_unit_ids IS NOT NULL\n"
                "UNWIND e.text_unit_ids AS text_unit_id\n"
                "MATCH (tu:TextUnit {id: text_unit_id})-[:PART_OF]->(d:Document)\n"
                + ("WHERE d.name = $document_id\n" if document_id else "") +
                "\n"
                "// Find regions in the same document near the text unit by order proximity\n"
                "OPTIONAL MATCH (region:Region)-[:PART_OF]->(d)\n"
                "WHERE abs(region.order - tu.order) <= 3\n"
                "\n"
                "WITH e, d, collect(DISTINCT region) AS all_regions\n"
                "UNWIND all_regions AS region\n"
                "WITH e, d, region\n"
                "WHERE region IS NOT NULL\n"
                "\n"
                "// Create connection between entity and nearby regions\n"
                "OPTIONAL MATCH (e)-[existing_rel:CONNECTED_TO_REGION]->(region)\n"
                "WITH e, d, region, existing_rel\n"
                "WHERE existing_rel IS NULL\n"
                "MERGE (e)-[:CONNECTED_TO_REGION {relationship_type: 'MENTIONED_NEAR', created_at: datetime(), document_id: d.name}]->(region)\n"
                "\n"
                "RETURN count(region) AS entity_region_connections"
            )

            result_new = session.run(entity_to_region_query, **params)
            count_new = result_new.single()["entity_region_connections"]
            logger.info(f"Created {count_new} CONNECTED_TO_REGION relationships between entities and regions for document {document_id or 'all documents'}")

            # Connect semantic communities to structural regions that they summarize or represent
            community_to_structure_query = (
                "MATCH (sc:Community)  // Semantic community\n"
                "MATCH (sc)-[:CONSISTS_OF]->(se:Entity)  // Community consists of entities\n"
                "MATCH (se)-[:LINKED_TO_DOCUMENT]->(sd:Document)  // Entity linked to document\n"
                + ("WHERE sd.name = $document_id\n" if document_id else "") +
                "\n"
                "// Find structural regions in the same document\n"
                "MATCH (sr:Region)-[:PARENT|ORDER*]->(sd)  // Structural regions in the document\n"
                "\n"
                "// Connect community to related structural regions\n"
                "OPTIONAL MATCH (sc)-[existing_rel:DESCRIBES_STRUCTURE]->(sr)\n"
                "WITH sc, sr, sd, existing_rel\n"
                "WHERE existing_rel IS NULL\n"
                "MERGE (sc)-[:DESCRIBES_STRUCTURE {relationship_type: 'SEMANTIC_SUMMARIZES_STRUCTURAL', created_at: datetime(), document_id: sd.name}]->(sr)\n"
                "\n"
                "RETURN count(sc) AS community_structure_connections"
            )

            result2 = session.run(community_to_structure_query, **params)
            count2 = result2.single()["community_structure_connections"]
            logger.info(f"Created {count2} DESCRIBES_STRUCTURE relationships between communities and structural regions for document {document_id or 'all documents'}")

            # Connect documents from both graphs
            document_to_document_query = (
                "MATCH (struct_doc:Document)  // Document from structural graph\n"
                "MATCH (sem_doc:Document)  // Document from semantic graph\n"
                "WHERE struct_doc.name = sem_doc.name  // Same document based on name/hashing\n"
                + ("  AND struct_doc.name = $document_id\n" if document_id else "\n") +
                "\n"
                "// Create a connection between the two document representations\n"
                "OPTIONAL MATCH (struct_doc)-[existing_rel:SAME_CONTENT_AS]->(sem_doc)\n"
                "WITH struct_doc, sem_doc, existing_rel\n"
                "WHERE existing_rel IS NULL\n"
                "MERGE (struct_doc)-[:SAME_CONTENT_AS {relationship_type: 'TWO_PERSPECTIVES', created_at: datetime(), document_id: struct_doc.name}]->(sem_doc)\n"
                "\n"
                "RETURN count(struct_doc) AS document_connections"
            )

            result3 = session.run(document_to_document_query, **params)
            count3 = result3.single()["document_connections"]
            logger.info(f"Created {count3} SAME_CONTENT_AS relationships between structural and semantic documents for document {document_id or 'all documents'}")

        logger.info(f"Successfully enhanced connections between structural and semantic graphs for document {document_id or 'all documents'}")

    except Exception as e:
        logger.error(f"Error enhancing connections between graphs: {e}")
        raise


def create_semantic_links(document_id: Optional[str] = None):
    """Create semantic_link relationships between Entity nodes from the semantic graph
    and Region nodes that are already loaded in Neo4j.

    Args:
        document_id: Specific document ID to connect (if None, connects all documents)
    """
    try:
        # Import the document index service to access the Neo4j connection
        from documet_index.neo4j_service import DocumentIndexService

        # Create service instance to access the Neo4j connection
        structural_service = DocumentIndexService()

        params = {}
        if document_id:
            params["document_prefix"] = f"{document_id}|"

        with structural_service.driver.session() as session:
            # Create semantic_link relationships between Entity nodes and Region nodes
            # using region_id from Region and text_unit_ids from Entity
            semantic_link_query = """
            MATCH (e:Entity)  // Semantic entity from semantic graph
            WHERE e.text_unit_ids IS NOT NULL
            """
            if document_id:
                semantic_link_query += " AND ANY(tid IN e.text_unit_ids WHERE tid STARTS WITH $document_prefix)\n"
            semantic_link_query += """
            UNWIND e.text_unit_ids AS text_unit_id
            WITH e, text_unit_id
            MATCH (r:Region {region_id: text_unit_id})  // Match Region by region_id
            
            // Create semantic_link relationship between Entity and Region
            OPTIONAL MATCH (e)-[existing_rel:semantic_link]->(r)
            WITH e, r, existing_rel
            WHERE existing_rel IS NULL
            MERGE (e)-[:semantic_link {relationship_type: 'SEMANTIC_ASSOCIATION', created_at: datetime(), region_id: r.region_id}]->(r)

            RETURN count(*) AS semantic_links_created
            """

            result = session.run(semantic_link_query, **params)
            count = result.single()["semantic_links_created"]
            logger.info(f"Created {count} semantic_link relationships between Entity and Region nodes for document {document_id or 'all documents'}")

        logger.info(f"Successfully created semantic_link relationships for document {document_id or 'all documents'}")

    except Exception as e:
        logger.error(f"Error creating semantic links: {e}")
        raise


def compute_bridge_edge_weights(document_id: Optional[str] = None):
    """Assign weights to all bridge edges between structural and semantic graphs.

    Weights are based on:
    - Entity confidence (normalized 0-1 from confidence 1-10)
    - Proximity decay (ORDER distance between Region and TextUnit)

    This enables context search to prioritise stronger semantic-structural links.

    Args:
        document_id: Specific document ID (if None, computes for all documents)
    """
    try:
        # Import config for confidence multiplier
        from semantic_graph.config import (
            BRIDGE_WEIGHT_CONFIDENCE_MULTIPLIER,
            BRIDGE_WEIGHT_PROXIMITY_DECAY,
        )
    except ImportError:
        BRIDGE_WEIGHT_CONFIDENCE_MULTIPLIER = 1.0
        BRIDGE_WEIGHT_PROXIMITY_DECAY = 0.15

    try:
        from documet_index.neo4j_service import DocumentIndexService
        structural_service = DocumentIndexService()

        params = {"document_id": document_id} if document_id else {"document_id": "*"}

        with structural_service.driver.session() as session:
            # Relationship types to weight:
            # 1. Entity -> Region (CONNECTS_TO, NEAR_REGION, CONNECTED_TO_REGION)
            # 2. Entity -> TextUnit (LINKED_TO_TEXTUNIT)
            # 3. Entity -> Document (LINKED_TO_DOCUMENT)
            # 4. Entity -> Region (MENTIONS_ELEMENT)
            # 5. Entity -> Region (semantic_link)

            weight_query = """
            MATCH (e:Entity)-[r]->(target)
            WHERE type(r) IN [
                'CONNECTS_TO', 'NEAR_REGION', 'CONNECTED_TO_REGION',
                'LINKED_TO_TEXTUNIT', 'LINKED_TO_DOCUMENT',
                'MENTIONS_ELEMENT', 'semantic_link',
                'DESCRIBES_STRUCTURE', 'CONNECTED_TO_DOCUMENT'
            ]
            """
            if document_id:
                weight_query += (
                    "  AND (r.document_id = $document_id OR $document_id = '*')\n"
                )
            weight_query += """
            // Compute weight from entity confidence and proximity
            WITH e, r, target,
                 coalesce(e.confidence, 5) AS conf,
                 coalesce(r.distance, abs(coalesce(target.order, 0) - coalesce(e.degree, 0))) AS dist
            WITH e, r,
                 CASE
                     WHEN dist > 0
                     THEN (toFloat(conf) / 10.0) * $multiplier - dist * $decay
                     ELSE (toFloat(conf) / 10.0) * $multiplier
                 END AS raw_weight
            // Clamp to [0.1, 1.0] — Neo4j has no GREATEST/LEAST, use CASE
            SET r.weight = CASE
                WHEN raw_weight > 1.0 THEN 1.0
                WHEN raw_weight < 0.1 THEN 0.1
                ELSE raw_weight
            END
            RETURN count(r) AS weighted_edges
            """

            result = session.run(
                weight_query,
                {
                    "multiplier": BRIDGE_WEIGHT_CONFIDENCE_MULTIPLIER,
                    "decay": BRIDGE_WEIGHT_PROXIMITY_DECAY,
                    "document_id": document_id or "*",
                },
            )
            count = result.single()["weighted_edges"]
            logger.info(
                "Computed weights for %d bridge edges (document=%s)",
                count,
                document_id or "all documents",
            )

    except Exception as e:
        logger.error(f"Error computing bridge edge weights: {e}")
        raise


def create_aggregated_community_region_links(document_id: Optional[str] = None):
    """Pre-compute aggregated Community → Region edges with confidence-based weights.

    Aggregates bridge connections from member Entities to Regions and
    creates :rel:`AGGREGATED_REGIONS` edges on Community nodes.  The
    weight is computed as::

        avg(entity_confidence) * avg(bridge_weight)

    This gives the BFS crawler a fast 1-hop path from Community to
    structural Regions without traversing via Entities at query time.

    Args:
        document_id: Specific document ID (if None, processes all documents)
    """
    try:
        from documet_index.neo4j_service import DocumentIndexService
        structural_service = DocumentIndexService()

        params = {"document_id": document_id} if document_id else {}

        with structural_service.driver.session() as session:
            # Aggregate entity→region bridges per community, with confidence weighting
            aggregate_query = (
                "MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)\n"
                "-[bridge:CONNECTED_TO_REGION|NEAR_REGION|semantic_link]->(r:Region)\n"
            )
            if document_id:
                aggregate_query += (
                    "WHERE bridge.document_id = $document_id\n"
                )
            aggregate_query += (
                "WITH c, r,\n"
                "     avg(coalesce(e.confidence, 5) / 10.0) AS avg_conf,\n"
                "     avg(coalesce(bridge.weight, 0.5)) AS avg_bridge_weight,\n"
                "     count(DISTINCT e) AS entity_count\n"
                "WHERE entity_count >= 1\n"
                "WITH c, r, avg_conf, avg_bridge_weight, entity_count\n"
                "WHERE NOT (c)-[:AGGREGATED_REGIONS]->(r)\n"
                "MERGE (c)-[:AGGREGATED_REGIONS {\n"
                "    relationship_type: 'AGGREGATED_SEMANTIC_BRIDGE',\n"
                "    created_at: datetime(),\n"
                "    weight: avg_conf * avg_bridge_weight,\n"
                "    entity_count: entity_count,\n"
                "    avg_entity_confidence: avg_conf,\n"
                "    avg_bridge_weight: avg_bridge_weight\n"
                "}]->(r)\n"
                "RETURN count(c) AS aggregated_links"
            )

            result = session.run(aggregate_query, params)
            count = result.single()["aggregated_links"]
            logger.info(
                "Created %d AGGREGATED_REGIONS links between communities and regions "
                "for document %s",
                count,
                document_id or "all documents",
            )

            # Clean up stale aggregated links where entity membership changed
            cleanup_query = (
                "MATCH (c:Community)-[ar:AGGREGATED_REGIONS]->(r:Region)\n"
                "WHERE ar.created_at IS NOT NULL\n"
                "  AND NOT EXISTS {\n"
                "    MATCH (c)-[:CONSISTS_OF]->(:Entity)"
                "-[:CONNECTED_TO_REGION|NEAR_REGION|semantic_link]->(r)\n"
                "  }\n"
            )
            if document_id:
                cleanup_query += (
                    "  AND ar.document_id = $document_id\n"
                )
            cleanup_query += "DELETE ar\n"

            cleanup_result = session.run(cleanup_query, params)
            # Can't get count from DELETE, so just log
            logger.info(
                "Cleaned stale AGGREGATED_REGIONS links for document %s",
                document_id or "all documents",
            )

    except Exception as e:
        logger.error(f"Error creating aggregated community-region links: {e}")
        raise


def get_connection_statistics(document_id: Optional[str] = None):
    """Get statistics about connections between structural and semantic graphs.

    Args:
        document_id: Specific document ID to get stats for (if None, gets stats for all documents)

    Returns:
        Dictionary with connection statistics
    """
    try:
        # Import the document index service to access the Neo4j connection
        from documet_index.neo4j_service import DocumentIndexService

        # Create a service instance to access the Neo4j connection
        neo4j_service = DocumentIndexService()

        params = {"document_id": document_id} if document_id else {}

        with neo4j_service.driver.session() as session:
            # Count connections between entities and text units
            entity_textunit_query = (
                "MATCH (e:Entity)-[r:LINKED_TO_TEXTUNIT]->(tu:TextUnit)-[:PART_OF]->(d:Document)"
                + (" WHERE d.name = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            entity_textunit_count = session.run(entity_textunit_query, **params).single()["count"]

            # Count connections between entities and documents
            entity_doc_query = (
                "MATCH (e:Entity)-[r:LINKED_TO_DOCUMENT]->(d:Document)"
                + (" WHERE d.name = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            entity_doc_count = session.run(entity_doc_query, **params).single()["count"]

            # Count connections between communities and documents
            community_doc_query = (
                "MATCH (c:Community)-[r:CONNECTED_TO_DOCUMENT]->(d:Document)"
                + (" WHERE d.name = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            community_doc_count = session.run(community_doc_query, **params).single()["count"]

            # Count connections between entities and structural elements
            entity_structure_query = (
                "MATCH (e:Entity)-[r:MENTIONS_ELEMENT]->(sr:Region)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            entity_structure_count = session.run(entity_structure_query, **params).single()["count"]

            # Count connections between entities and all nearby regions
            entity_near_region_query = (
                "MATCH (e:Entity)-[r:NEAR_REGION]->(reg:Region)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            entity_near_region_count = session.run(entity_near_region_query, **params).single()["count"]

            # Count connections between structural regions and semantic entities
            structural_semantic_query = (
                "MATCH (sr:Region)-[r:CONNECTS_TO]->(se:Entity)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            structural_semantic_count = session.run(structural_semantic_query, **params).single()["count"]

            # Count connections between entities and regions
            entity_region_query = (
                "MATCH (e:Entity)-[r:CONNECTED_TO_REGION]->(reg:Region)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            entity_region_count = session.run(entity_region_query, **params).single()["count"]

            # Count semantic_link connections between entities and regions
            semantic_link_query = (
                "MATCH (e:Entity)-[r:semantic_link]->(reg:Region)"
                + (" WHERE r.region_id STARTS WITH $document_id + '|'" if document_id else "")
                + " RETURN count(r) AS count"
            )
            semantic_link_count = session.run(semantic_link_query, **params).single()["count"]

            # Count connections between communities and structural elements
            community_structure_query = (
                "MATCH (c:Community)-[r:DESCRIBES_STRUCTURE]->(sr:Region)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            community_structure_count = session.run(community_structure_query, **params).single()["count"]

            # Count connections between structural and semantic documents
            document_same_content_query = (
                "MATCH (struct_doc:Document)-[r:SAME_CONTENT_AS]->(sem_doc:Document)"
                + (" WHERE r.document_id = $document_id" if document_id else "")
                + " RETURN count(r) AS count"
            )
            document_same_content_count = session.run(document_same_content_query, **params).single()["count"]

            stats = {
                "entity_to_textunit_connections": entity_textunit_count,
                "entity_to_document_connections": entity_doc_count,
                "community_to_document_connections": community_doc_count,
                "entity_to_structure_connections": entity_structure_count,
                "entity_near_region_connections": entity_near_region_count,
                "structural_to_semantic_connections": structural_semantic_count,
                "entity_to_region_connections": entity_region_count,
                "semantic_link_connections": semantic_link_count,
                "community_to_structure_connections": community_structure_count,
                "document_to_document_connections": document_same_content_count
            }

            logger.info(f"Connection statistics for document {document_id or 'all documents'}: {stats}")
            return stats

    except Exception as e:
        logger.error(f"Error getting connection statistics: {e}")
        return {}


def main(document_id: Optional[str] = None):
    """Main function to run the graph connection process.

    Creates bridge relationships between structural and semantic graphs
    and computes weights for traversal.  Call with ``document_id`` to
    re-connect a single document, or omit for all documents.

    Args:
        document_id: Specific document ID (if None, connects all documents)
    """
    try:
        logger.info("Starting connection between structural and semantic graphs...")
        connect_graphs_by_document(document_id)

        logger.info("Enhancing connections between structural and semantic graphs...")
        connect_structural_and_semantic_nodes(document_id)

        logger.info("Creating semantic_link relationships between Entity and Region nodes...")
        create_semantic_links(document_id)

        logger.info("Computing bridge edge weights (confidence × proximity decay)...")
        compute_bridge_edge_weights(document_id)

        logger.info("Pre-computing aggregated Community→Region edges...")
        create_aggregated_community_region_links(document_id)

        logger.info("Getting connection statistics...")
        stats = get_connection_statistics(document_id)

        logger.info("Graph connection process completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()