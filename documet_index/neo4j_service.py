"""
Neo4j Graph Service for PDF Document Indexing

This module provides functionality to create document graphs in Neo4j
from MinerU processing results.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIndexService:
    """Service for managing document graphs in Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        name_db: Optional[str] = None
    ):
        """Initialize the document index service.

        Args:
            uri: Neo4j connection URI (default from env: URL)
            user: Database username (default from env: USER_NEO4J)
            password: Database password (default from env: PASSWORD)
            name_db: Database name (default from env: NAME_DB)
        """
        # Get configuration from environment if not provided
        self.uri = uri or f"neo4j://{os.environ.get('URL', 'localhost:7687')}"
        self.user = user or os.environ.get('USER_NEO4J', 'neo4j')
        self.password = password or os.environ.get('PASSWORD', '')
        self.name_db = name_db or os.environ.get('NAME_DB', 'neo4j')

        # Import here to avoid circular imports
        from .manager import Manager, ManagerConfig

        config = ManagerConfig(
            uri=self.uri,
            user=self.user,
            password=self.password,
            name_db=self.name_db
        )

        self.manager = Manager(config)
        logger.info(f"DocumentIndexService initialized with DB: {self.name_db}")
    
    @property
    def driver(self):
        """Property to access the underlying Neo4j driver through the manager."""
        return self.manager.conn.graph

    def create_graph_from_mineru_result(
        self,
        mineru_result: Dict[str, Any],
        file_hash: str,
        region_reclassify_enabled: bool = False,
        llm_client=None,
    ) -> bool:
        """Create a graph in Neo4j from MinerU processing result.

        Args:
            mineru_result: JSON result from MinerU PDF processing
            file_hash: Unique hash identifier for the PDF file
            region_reclassify_enabled: Enable A1 region type re-classification
            llm_client: Optional LLM client for multimodal re-classification

        Returns:
            True if graph was created successfully, False if document already exists
        """
        try:
            # Import Document class
            from .dtype import Document

            # Inject region re-classification flags (A1 improvement)
            if region_reclassify_enabled:
                mineru_result.setdefault("results", {}).setdefault("result", {}).setdefault("results", {})
                mineru_result["results"]["result"]["results"]["_region_reclassify_enabled"] = True
                if llm_client is not None:
                    mineru_result["results"]["result"]["results"]["_region_llm_client"] = llm_client

            # Create Document object from MinerU result
            document = Document(
                json_data=mineru_result,
                name=file_hash,
                mode='mineru'
            )

            # Add document to Neo4j
            success = self.manager.add_document(document)

            if success:
                logger.info(f"Successfully created graph for document '{file_hash}'")
            else:
                logger.warning(f"Document '{file_hash}' already exists in graph database")

            return success

        except Exception as e:
            logger.error(f"Failed to create graph for document '{file_hash}': {e}")
            raise

    def delete_graph(self, file_hash: str) -> bool:
        """Delete a document graph from Neo4j.

        Args:
            file_hash: Unique hash identifier for the PDF file

        Returns:
            True if graph was deleted successfully, False otherwise
        """
        try:
            success = self.manager.delete_document(file_hash)
            if success:
                logger.info(f"Successfully deleted graph for document '{file_hash}'")
            else:
                logger.warning(f"Document '{file_hash}' not found in graph database")
            return success
        except Exception as e:
            logger.error(f"Failed to delete graph for document '{file_hash}': {e}")
            raise

    def is_document_indexed(self, file_hash: str) -> bool:
        """Check if a document is already indexed in Neo4j.

        Args:
            file_hash: Unique hash identifier for the PDF file

        Returns:
            True if document exists in graph database, False otherwise
        """
        return self.manager.is_document_exist(file_hash)

    def get_status(self) -> Dict[str, Any]:
        """Get database status information.

        Returns:
            Dictionary with node count and other statistics
        """
        return self.manager.status()

    def close(self):
        """Close the database connection."""
        self.manager.close()
        logger.info("DocumentIndexService connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def delete_all_graphs(self) -> bool:
        """Delete all document graphs from Neo4j.

        Returns:
            True if all graphs were deleted successfully, False otherwise
        """
        try:
            success = self.manager.delete_all_documents()
            if success:
                logger.info("Successfully deleted all graphs from database")
            else:
                logger.warning("Failed to delete all graphs from database")
            return success
        except Exception as e:
            logger.error(f"Failed to delete all graphs: {e}")
            raise

    def get_related_context(self, file_hash: str, element_type: str, text: str) -> Dict[str, Any]:
        """Get related context from Neo4j for a given element.

        For image_caption/image_footnote, returns the parent image node.
        For table_caption/table_footnote, returns the parent table node.
        For image/table, returns associated caption and footnote nodes.

        Args:
            file_hash: Document identifier
            element_type: Type of the element (image_caption, image_footnote, table_caption, table_footnote, image, table)
            text: Text content of the element to match

        Returns:
            Dictionary with related context information
        """
        return self.manager.get_related_context(file_hash, element_type, text)

    def get_order_neighbors(
        self,
        region_ids: List[str],
        window_size: int = 3,
        include_parent: bool = False,
    ) -> List[Dict[str, Any]]:
        """Walk ±K steps in ORDER from matched Regions to gather surrounding context.

        Delegates to Manager.get_order_neighbors.
        """
        return self.manager.get_order_neighbors(
            region_ids, window_size, include_parent=include_parent,
        )

    def get_sections_for_regions(
        self,
        region_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Map region_id -> its Section title from the structural graph.

        Delegates to Manager.get_sections_for_regions.
        """
        return self.manager.get_sections_for_regions(region_ids)

    def get_cross_graph_bridge(
        self,
        region_ids: List[str],
        max_regions: int = 20,
    ) -> List[Dict[str, Any]]:
        """Walk from structural Regions to Entities, expand semantically, then back.

        Strategy D — Cross-Graph Bridge Walking.
        Delegates to the semantic_graph Manager using the same Neo4j connection.
        """
        try:
            from semantic_graph.manager import Manager as SemanticManager, ManagerConfig
            config = ManagerConfig(
                uri=self.uri,
                user=self.user,
                password=self.password,
                name_db=self.name_db,
            )
            semantic_mgr = SemanticManager(config)
            return semantic_mgr.get_cross_graph_bridge(region_ids, max_regions)
        except ImportError as e:
            logger.warning("Semantic graph manager not available for cross-graph bridge: %s", e)
            return []
        except Exception as e:
            logger.error("Error in cross-graph bridge walk: %s", e)
            return []

# Convenience function for creating graph from MinerU result
def create_neo4j_graph(mineru_result: Dict[str, Any], file_hash: str) -> bool:
    """Create a Neo4j graph from MinerU result.

    This is a convenience function that creates a DocumentIndexService
    and uses it to create a graph from the MinerU result.

    Args:
        mineru_result: JSON result from MinerU PDF processing
        file_hash: Unique hash identifier for the PDF file

    Returns:
        True if graph was created successfully, False if document already exists
    """
    service = DocumentIndexService()
    try:
        return service.create_graph_from_mineru_result(mineru_result, file_hash)
    finally:
        service.close()