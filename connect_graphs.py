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
    
    Args:
        document_id: Specific document ID to connect (if None, connects all documents)
    """
    try:
        # Import the document index service to access the Neo4j connection
        from documet_index.neo4j_service import DocumentIndexService
        
        # Create a service instance to access the Neo4j connection
        neo4j_service = DocumentIndexService()
        
        with neo4j_service.driver.session() as session:
            # Base query conditions depending on whether a specific document is targeted
            where_clause = f"WHERE d.file_hash = '{document_id}'" if document_id else ""
            
            # Connect entities to text units that contain them
            entity_to_textunit_query = f"""
            MATCH (e:Entity)
            WHERE e.text_unit_ids IS NOT NULL
            UNWIND e.text_unit_ids AS text_unit_id
            MATCH (tu:TextUnit {{id: text_unit_id}})
            MATCH (tu)-[:PART_OF]->(d:Document) 
            {where_clause}
            WITH e, tu, d
            WHERE NOT (e)-[:LINKED_TO_TEXTUNIT]->(tu)
            MERGE (e)-[:LINKED_TO_TEXTUNIT {{relationship_type: 'CONTAINED_IN', created_at: datetime(), document_id: d.file_hash}}]->(tu)
            RETURN count(e) AS relationships_created
            """
            
            result1 = session.run(entity_to_textunit_query)
            count1 = result1.single()["relationships_created"]
            logger.info(f"Created {count1} LINKED_TO_TEXTUNIT relationships between entities and text units for document {document_id or 'all documents'}")
            
            # Connect entities to documents containing them
            entity_to_document_query = f"""
            MATCH (e:Entity)
            WHERE e.text_unit_ids IS NOT NULL
            UNWIND e.text_unit_ids AS text_unit_id
            MATCH (tu:TextUnit {{id: text_unit_id}})-[:PART_OF]->(d:Document)
            {where_clause}
            WITH e, d
            WHERE NOT (e)-[:LINKED_TO_DOCUMENT]->(d)
            MERGE (e)-[:LINKED_TO_DOCUMENT {{relationship_type: 'MENTIONED_IN', created_at: datetime(), document_id: d.file_hash}}]->(d)
            RETURN count(e) AS relationships_created
            """
            
            result2 = session.run(entity_to_document_query)
            count2 = result2.single()["relationships_created"]
            logger.info(f"Created {count2} LINKED_TO_DOCUMENT relationships between entities and documents for document {document_id or 'all documents'}")
            
            # Connect communities to documents based on contained entities
            community_to_document_query = f"""
            MATCH (c:Community)
            MATCH (c)-[:CONSISTS_OF]->(e:Entity)-[:LINKED_TO_DOCUMENT]->(d:Document)
            {where_clause}
            WITH c, d
            WHERE NOT (c)-[:CONNECTED_TO_DOCUMENT]->(d)
            MERGE (c)-[:CONNECTED_TO_DOCUMENT {{relationship_type: 'REPRESENTS_CONTENT_OF', created_at: datetime(), document_id: d.file_hash}}]->(d)
            RETURN count(c) AS relationships_created
            """
            
            result3 = session.run(community_to_document_query)
            count3 = result3.single()["relationships_created"]
            logger.info(f"Created {count3} CONNECTED_TO_DOCUMENT relationships between communities and documents for document {document_id or 'all documents'}")
            
            # Connect entities to their corresponding structural elements (images, tables, etc.)
            entity_to_structure_query = f"""
            MATCH (e:Entity)
            WHERE e.text_unit_ids IS NOT NULL
            UNWIND e.text_unit_ids AS text_unit_id
            MATCH (tu:TextUnit {{id: text_unit_id}})
            MATCH (tu)-[:PART_OF]->(d:Document)
            {where_clause}
            OPTIONAL MATCH (tu)<-[:ORDER]-(prev_region:Region) 
            OPTIONAL MATCH (tu)-[:ORDER]->(next_region:Region)
            WITH e, tu, d, prev_region, next_region
            WHERE prev_region IS NOT NULL OR next_region IS NOT NULL
            FOREACH(r IN CASE WHEN prev_region IS NOT NULL THEN [prev_region] ELSE [] END |
              MERGE (e)-[:LINKED_TO_STRUCTURE {{relationship_type: 'MENTIONS_NEAR', created_at: datetime(), position: 'before', document_id: d.file_hash}}]->(r))
            FOREACH(r IN CASE WHEN next_region IS NOT NULL THEN [next_region] ELSE [] END |
              MERGE (e)-[:LINKED_TO_STRUCTURE {{relationship_type: 'MENTIONS_NEAR', created_at: datetime(), position: 'after', document_id: d.file_hash}}]->(r))
            RETURN count(e) AS relationships_processed
            """
            
            result4 = session.run(entity_to_structure_query)
            count4 = result4.single()["relationships_processed"]
            logger.info(f"Processed {count4} entities for LINKED_TO_STRUCTURE relationships for document {document_id or 'all documents'}")
            
        logger.info(f"Successfully connected structural and semantic graphs for document {document_id or 'all documents'}")
        
    except Exception as e:
        logger.error(f"Error connecting graphs: {e}")
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
        
        where_clause = f"WHERE d.file_hash = '{document_id}'" if document_id else ""
        
        with neo4j_service.driver.session() as session:
            # Count connections between entities and text units
            entity_textunit_query = f"""
            MATCH (e:Entity)-[r:LINKED_TO_TEXTUNIT]->(tu:TextUnit)-[:PART_OF]->(d:Document)
            {where_clause}
            RETURN count(r) AS count
            """
            entity_textunit_count = session.run(entity_textunit_query).single()["count"]
            
            # Count connections between entities and documents
            entity_doc_query = f"""
            MATCH (e:Entity)-[r:LINKED_TO_DOCUMENT]->(d:Document)
            {where_clause}
            RETURN count(r) AS count
            """
            entity_doc_count = session.run(entity_doc_query).single()["count"]
            
            # Count connections between communities and documents
            community_doc_query = f"""
            MATCH (c:Community)-[r:CONNECTED_TO_DOCUMENT]->(d:Document)
            {where_clause}
            RETURN count(r) AS count
            """
            community_doc_count = session.run(community_doc_query).single()["count"]
            
            stats = {
                "entity_to_textunit_connections": entity_textunit_count,
                "entity_to_document_connections": entity_doc_count,
                "community_to_document_connections": community_doc_count
            }
            
            logger.info(f"Connection statistics for document {document_id or 'all documents'}: {stats}")
            return stats
            
    except Exception as e:
        logger.error(f"Error getting connection statistics: {e}")
        return {}


def main():
    """Main function to run the graph connection process."""
    try:
        logger.info("Starting connection between structural and semantic graphs...")
        connect_graphs_by_document()
        
        logger.info("Getting connection statistics...")
        stats = get_connection_statistics()
        
        logger.info("Graph connection process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()