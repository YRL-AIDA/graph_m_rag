from neo4j import GraphDatabase
from typing import Optional, Dict, Any
import logging
import json

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
                image = reg.get('image', '')
                bbox = reg.get('bbox', {})
                style = reg.get('style', {})
                order = reg.get('order', 0)
                element_data = reg.get('element_data', '')

                # Escape single quotes in text fields
                text_escaped = text.replace("'", "\\'") if text else ''
                image_escaped = image.replace("'", "\\'") if image else ''
                element_data_escaped = str(element_data).replace("'", "\\'") if element_data else ''

                # Convert bbox and style to JSON strings for storage
                bbox_json = json.dumps(bbox) if bbox else '{}'
                style_json = json.dumps(style) if style else '{}'

                query += (f"CREATE (reg{id}:Region:{label} {{text: '{text_escaped}', image: '{image_escaped}', "
                          f"bbox: '{bbox_json}', style: '{style_json}', order: {order}, element_data: "
                          f"'{element_data_escaped}'}})\n")

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

    def delete_all_documents(self) -> bool:
        """Delete all documents and all related nodes from the database.

        Returns:
            True if all documents were deleted, False otherwise
        """
        try:
            query = """
               MATCH (d:Document) -[:ORDER*]-> (n), () -[r:PARENT]-> (n)
               WITH d, n, r, relationships(path) AS order_rels
               FOREACH (rel IN order_rels | DELETE rel)
               DELETE r, d, n
               """
            # Alternative simpler approach - delete everything
            query = """
               MATCH (n)
               DETACH DELETE n
               """
            self.query(query)
            logger.info("Deleted all documents from database")
            return True
        except Exception as e:
            logger.error(f"Error deleting all documents: {e}")
            return False

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
        try:
            # Escape single quotes in text
            text_escaped = text.replace("'", "\\'")

            related_context = {
                "parent_element": None,
                "sibling_captions": [],
                "sibling_footnotes": []
            }

            # For caption/footnote elements, find the parent image/table
            if element_type in ("image_caption", "image_footnote"):
                # Find the image that this caption/footnote belongs to
                # Look for an image node that comes before this caption/footnote in ORDER
                query = f"""
                   MATCH (d:Document {{name: '{file_hash}'}}) -[:ORDER*]-> (caption:Region:{element_type} {{text: '{text_escaped}'}})
                   OPTIONAL MATCH (img:Region:image) -[:ORDER*]-> (caption)
                   WHERE img.image IS NOT NULL AND img.image <> ''
                   WITH img, caption
                   ORDER BY caption.order - img.order ASC
                   LIMIT 1
                   RETURN img.text as text, img.image as image, img.bbox as bbox, img.element_data as element_data
                   """
                result = self.query(query)
                if result and result[0].data().get('text'):
                    data = result[0].data()
                    related_context["parent_element"] = {
                        "type": "image",
                        "text": data.get('text', ''),
                        "image": data.get('image', ''),
                        "bbox": data.get('bbox', '{}'),
                        "element_data": data.get('element_data', '')
                    }

            elif element_type in ("table_caption", "table_footnote"):
                # Find the table that this caption/footnote belongs to
                query = f"""
                   MATCH (d:Document {{name: '{file_hash}'}}) -[:ORDER*]-> (caption:Region:{element_type} {{text: '{text_escaped}'}})
                   OPTIONAL MATCH (tbl:Region:table) -[:ORDER*]-> (caption)
                   WHERE tbl.image IS NOT NULL AND tbl.image <> ''
                   WITH tbl, caption
                   ORDER BY caption.order - tbl.order ASC
                   LIMIT 1
                   RETURN tbl.text as text, tbl.image as image, tbl.bbox as bbox, tbl.element_data as element_data
                   """
                result = self.query(query)
                if result and result[0].data().get('text'):
                    data = result[0].data()
                    related_context["parent_element"] = {
                        "type": "table",
                        "text": data.get('text', ''),
                        "image": data.get('image', ''),
                        "bbox": data.get('bbox', '{}'),
                        "element_data": data.get('element_data', '')
                    }

            # For image/table elements, find associated captions and footnotes
            elif element_type == "image":
                # Find image_caption and image_footnote nodes that follow this image
                query = f"""
                   MATCH (d:Document {{name: '{file_hash}'}}) -[:ORDER*]-> (img:Region:image {{text: '{text_escaped}'}})
                   OPTIONAL MATCH (img) -[:ORDER*]-> (cap:Region:image_caption)
                   OPTIONAL MATCH (img) -[:ORDER*]-> (fn:Region:image_footnote)
                   RETURN
                       collect(DISTINCT {{text: cap.text, element_data: cap.element_data}}) as captions,
                       collect(DISTINCT {{text: fn.text, element_data: fn.element_data}}) as footnotes
                   """
                result = self.query(query)
                if result:
                    data = result[0].data()
                    related_context["sibling_captions"] = [
                        {"text": c.get('text', ''), "element_data": c.get('element_data', '')}
                        for c in data.get('captions', []) if c.get('text')
                    ]
                    related_context["sibling_footnotes"] = [
                        {"text": f.get('text', ''), "element_data": f.get('element_data', '')}
                        for f in data.get('footnotes', []) if f.get('text')
                    ]

            elif element_type == "table":
                # Find table_caption and table_footnote nodes that follow this table
                query = f"""
                   MATCH (d:Document {{name: '{file_hash}'}}) -[:ORDER*]-> (tbl:Region:table {{text: '{text_escaped}'}})
                   OPTIONAL MATCH (tbl) -[:ORDER*]-> (cap:Region:table_caption)
                   OPTIONAL MATCH (tbl) -[:ORDER*]-> (fn:Region:table_footnote)
                   RETURN
                       collect(DISTINCT {{text: cap.text, element_data: cap.element_data}}) as captions,
                       collect(DISTINCT {{text: fn.text, element_data: fn.element_data}}) as footnotes
                   """
                result = self.query(query)
                if result:
                    data = result[0].data()
                    related_context["sibling_captions"] = [
                        {"text": c.get('text', ''), "element_data": c.get('element_data', '')}
                        for c in data.get('captions', []) if c.get('text')
                    ]
                    related_context["sibling_footnotes"] = [
                        {"text": f.get('text', ''), "element_data": f.get('element_data', '')}
                        for f in data.get('footnotes', []) if f.get('text')
                    ]

            return related_context

        except Exception as e:
            logger.error(f"Error getting related context for {element_type}: {e}")
            return {"parent_element": None, "sibling_captions": [], "sibling_footnotes": []}

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