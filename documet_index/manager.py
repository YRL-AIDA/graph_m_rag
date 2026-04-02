from neo4j import GraphDatabase
from typing import Optional, Dict, Any
import logging

from .dtype import Document

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection wrapper."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., 'neo4j://localhost:7687')
            user: Database username
            password: Database password
        """
        self.graph = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the Neo4j connection."""
        if self.graph is not None:
            self.graph.close()
            logger.info("Neo4j connection closed")

    def query(self, query: str, db: Optional[str] = None) -> list:
        """Execute a Cypher query.

        Args:
            query: Cypher query string
            db: Optional database name

        Returns:
            List of query results
        """
        assert self.graph is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.graph.session(database=db) if db is not None else self.graph.session()
            response = list(session.run(query))
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
        finally:
            if session is not None:
                session.close()
        return response


class ManagerConfig:
    """Configuration for Document Manager."""

    def __init__(self, uri: str, user: str, password: str, name_db: str):
        """Initialize manager configuration.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
            name_db: Database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.name_db = name_db
        logger.debug(f"ManagerConfig initialized with URI: {uri}, DB: {name_db}")


class Manager:
    """Document Manager for Neo4j graph operations."""

    def __init__(self, config: ManagerConfig):
        """Initialize document manager.

        Args:
            config: Manager configuration with Neo4j connection details
        """
        self.conn = Neo4jConnection(config.uri, config.user, config.password)
        self.name_db = config.name_db

    def add_document(self, document: Document) -> bool:
        """Add a document to the graph database.

        Args:
            document: Document object to add

        Returns:
            True if document was added, False if it already exists
        """
        if not self.is_document_exist(document.name):
            graph = document.get_graph()
            query = ""
            query += f"CREATE (d:Document {{name: '{document.name}'}})\n"
            for id, reg in graph['nodes']['regions'].items():
                label = reg['label']
                text = reg['text']
                # Escape single quotes in text
                text_escaped = text.replace("'", "\\'")
                query += f"CREATE (reg{id}:Region:{label} {{text: '{text_escaped}'}})\n"

            for order in graph['edges']['order']:
                n1, n2 = order
                node1 = 'd' if n1 == -1 else f'reg{n1}'
                node2 = f'reg{n2}'
                query += f"CREATE ({node1}) -[:ORDER]-> ({node2})\n"

            for p in graph['edges']['parental']:
                n1, n2 = p
                node1 = 'd' if n1 == -1 else f'reg{n1}'
                node2 = f'reg{n2}'
                query += f"CREATE ({node1}) -[:PARENT]-> ({node2})\n"

            logger.info(f"Adding document '{document.name}' with {len(graph['nodes']['regions'])} regions")
            self.query(query)
            return True
        else:
            logger.warning(f"Document '{document.name}' already exists")
            return False

    def is_document_exist(self, name: str) -> bool:
        """Check if a document exists in the database.

        Args:
            name: Document name to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            result = self.query(f"OPTIONAL MATCH (d:Document) RETURN '{name}' in d.name as exist")
            return result[0].data().get('exist', False) if result else False
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    def delete_document(self, name: str) -> bool:
        """Delete a document and all its related nodes from the database.

        Args:
            name: Document name to delete

        Returns:
            True if document was deleted, False otherwise
        """
        try:
            query = f"""
            MATCH path = (m:Document {{name: '{name}'}}) -[:ORDER*]-> (n), () -[r2:PARENT]-> (n)
            WITH m, n, r2, relationships(path) AS order_rels
            FOREACH (rel IN order_rels | DELETE rel)
            DELETE r2, m, n
            """
            self.query(query)
            logger.info(f"Deleted document '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{name}': {e}")
            return False

    def query(self, query: str) -> list:
        """Execute a Cypher query on the database.

        Args:
            query: Cypher query string

        Returns:
            List of query results
        """
        return self.conn.query(query, self.name_db)

    def status(self) -> dict:
        """Get database status information.

        Returns:
            Dictionary with node count and other statistics
        """
        try:
            rez = self.query("MATCH (n) RETURN count(n) as count")
            node_count = rez[0].data()['count'] if rez else 0
            logger.info(f"Database status: {node_count} nodes")
            return {"node_count": node_count}
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"node_count": 0, "error": str(e)}

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()