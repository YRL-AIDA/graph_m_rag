from neo4j import GraphDatabase
from typing import List, Optional, Dict, Any
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

    def query(self, query: str, db: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> list:
        """Execute a Cypher query.

        Args:
            query: Cypher query string
            db: Optional database name
            params: Optional dictionary of parameters for the query

        Returns:
            List of query results
        """
        assert self.graph is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.graph.session(database=db) if db is not None else self.graph.session()
            if params:
                response = list(session.run(query, **params))
            else:
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

    def _build_sections(self, document_name: str, regions: dict) -> list:
        """Build section hierarchy from title regions (A3 improvement).

        Identifies title regions and groups subsequent regions into sections.
        Each section spans from one title to the next title.
        Regions before the first title belong to a «Preamble» section.

        Args:
            document_name: Document identifier (file_hash)
            regions: Dict of region data keyed by region id

        Returns:
            List of section dicts with keys:
                section_id, title, regions, start_order, end_order
        """
        # Collect all regions with their order for sorting
        region_list = []
        for rid, reg_data in regions.items():
            region_list.append({
                "id": rid,
                "label": reg_data.get("label", ""),
                "text": reg_data.get("text", ""),
                "order": reg_data.get("order", 0),
            })

        # Sort regions by order
        region_list.sort(key=lambda r: r["order"])

        # Identify title indices
        title_indices = [
            i for i, r in enumerate(region_list) if r["label"] == "title"
        ]

        if not title_indices:
            # No titles found — return a single default section
            return [{
                "section_id": f"{document_name}|section_default",
                "title": "Document",
                "regions": [r["id"] for r in region_list],
                "start_order": region_list[0]["order"] if region_list else 0,
                "end_order": region_list[-1]["order"] if region_list else 0,
            }]

        sections = []

        # Handle regions before the first title (preamble)
        if title_indices[0] > 0:
            preamble_regions = region_list[:title_indices[0]]
            sections.append({
                "section_id": f"{document_name}|section_preamble",
                "title": "Preamble",
                "regions": [r["id"] for r in preamble_regions],
                "start_order": preamble_regions[0]["order"],
                "end_order": preamble_regions[-1]["order"],
            })

        # Build sections from title boundaries
        for i, ti in enumerate(title_indices):
            title_region = region_list[ti]
            # Extract clean title text (remove "Title: " prefix)
            raw_text = title_region.get("text", "")
            clean_title = raw_text.replace("Title: ", "").strip() if raw_text else "Untitled"

            # Determine end of section: next title or end of list
            if i + 1 < len(title_indices):
                end_idx = title_indices[i + 1]
            else:
                end_idx = len(region_list)

            # Extract regions in this section (including the title itself)
            section_regions = region_list[ti:end_idx]

            section_id = f"{document_name}|section_{i}"
            sections.append({
                "section_id": section_id,
                "title": clean_title,
                "regions": [r["id"] for r in section_regions],
                "start_order": section_regions[0]["order"],
                "end_order": section_regions[-1]["order"],
            })

        return sections

    def add_document(self, document: Document, *, enable_sections: bool = True) -> bool:
        """Add a document to the graph database.

        Args:
            document: Document object to add
            enable_sections: Whether to create hierarchical Section nodes (A3)

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
                # Generate region_id using file_hash and region id
                region_id = f"{document.name}|{id}"

                # Escape backslashes and single quotes for Cypher string literals
                text_escaped = text.replace("\\", "\\\\").replace("'", "\\'") if text else ''
                image_escaped = image.replace("\\", "\\\\").replace("'", "\\'") if image else ''
                element_data_escaped = str(element_data).replace("\\", "\\\\").replace("'", "\\'") if element_data else ''

                # Convert bbox and style to JSON strings for storage
                bbox_json = json.dumps(bbox) if bbox else '{}'
                style_json = json.dumps(style) if style else '{}'

                query += (f"CREATE (reg{id}:Region:{label} {{region_id: '{region_id}', text: '{text_escaped}', image: '{image_escaped}', "
                          f"bbox: '{bbox_json}', style: '{style_json}', order: {order}, element_data: "
                          f"'{element_data_escaped}'}})\n")

            # A3: Build hierarchical section structure from title regions.
            # Creates Section nodes with SECTION edges: Document → Section → Region.
            if enable_sections and graph['nodes']['regions']:
                sections = self._build_sections(
                    document.name, graph['nodes']['regions']
                )
                for sec in sections:
                    sec_id_norm = sec['section_id'].replace("'", "\\'").replace("|", "_")
                    title_escaped = sec['title'].replace("'", "\\'")
                    query += (
                        f"CREATE (sec_{sec_id_norm}:Section {{section_id: "
                        f"'{sec_id_norm}', title: '{title_escaped}', "
                        f"start_order: {sec['start_order']}, "
                        f"end_order: {sec['end_order']}}})\n"
                    )
                    # Document → Section
                    query += f"CREATE (d) -[:SECTION]-> (sec_{sec_id_norm})\n"
                    # Section → Region for each region in this section
                    for rid in sec['regions']:
                        query += (
                            f"CREATE (sec_{sec_id_norm}) -[:SECTION]-> (reg{rid})\n"
                        )
                logger.info(
                    "Built %d sections for document '%s'",
                    len(sections), document.name,
                )

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
            # Escape backslashes and single quotes for Cypher string literals
            text_escaped = text.replace("\\", "\\\\").replace("'", "\\'")

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
                        OPTIONAL MATCH (d) -[:ORDER*]-> (img:Region:image) -[:ORDER*]-> (caption)
                        WITH img, caption
                        ORDER BY caption.order - img.order ASC
                        LIMIT 1
                        RETURN img.text as text, img.image as image, img.bbox as bbox, img.element_data as element_data
                        """
                result = self.query(query)
                if result:
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
                   MATCH (d:Document {{name: '{file_hash}'}}) -[:PARENT*]-> (caption:Region:{element_type} {{text: '{text_escaped}'}})
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

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        """Execute a Cypher query on the database.

        Args:
            query: Cypher query string
            params: Optional dictionary of parameters for the query

        Returns:
            List of query results
        """
        return self.conn.query(query, self.name_db, params)

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

    def add_semantic_link(self, structural_node_id: str, semantic_node_id: str, relationship_type: str = "SEMANTIC_CONNECTION"):
        """Add a link between a structural graph node and a semantic graph node.

        Args:
            structural_node_id: ID of the node in the structural graph
            semantic_node_id: ID of the node in the semantic graph
            relationship_type: Type of relationship between the nodes
        """
        try:
            query = """
            MATCH (sn)
            WHERE elementId(sn) = $structural_node_id
            MATCH (en)
            WHERE elementId(en) = $semantic_node_id
            MERGE (sn)-[:LINK {relationship_type: $relationship_type, created_at: datetime()}]->(en)
            """
            self.query(query, params={
                "structural_node_id": structural_node_id,
                "semantic_node_id": semantic_node_id,
                "relationship_type": relationship_type
            })
            logger.info(f"Added semantic link from {structural_node_id} to {semantic_node_id}")
        except Exception as e:
            logger.error(f"Error adding semantic link: {e}")
            raise

    def get_semantic_links(self, structural_node_id: str):
        """Get semantic graph nodes linked to a structural graph node.

        Args:
            structural_node_id: ID of the node in the structural graph

        Returns:
            List of linked semantic graph nodes
        """
        try:
            query = """
            MATCH (sn)
            WHERE elementId(sn) = $structural_node_id
            OPTIONAL MATCH (sn)-[r:LINK]->(en)
            RETURN elementId(en) AS semantic_node_id, r.relationship_type AS relationship_type, r.created_at AS created_at
            """
            result = self.query(query, params={"structural_node_id": structural_node_id})
            return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error getting semantic links: {e}")
            return []

    def get_order_neighbors(
        self,
        region_ids: List[str],
        window_size: int = 3,
        include_parent: bool = False,
    ) -> List[Dict[str, Any]]:
        """Walk ±K steps in ORDER from matched Regions to gather surrounding context.

        For each region_id, finds all Region nodes in the same Document
        whose ``order`` falls within ``window_size`` steps.  Returns
        text/label/order/region_id for every qualifying neighbour.

        When *include_parent* is ``True``, also walks up the ``PARENT``
        hierarchy (e.g. from ``image_caption`` to its parent ``image``
        Region), including the parent node in the results.

        Args:
            region_ids:     List of region_id strings (format: '{file_hash}|{element_index}')
            window_size:    Number of ORDER hops to walk in each direction (default 3)
            include_parent: If True, include parent Region nodes via PARENT edges

        Returns:
            List of dicts with keys: region_id, label, text, order,
            source_region_id, source (``'order'`` or ``'parent'``).
        """
        if not region_ids:
            return []

        query = """
            MATCH (r:Region)-[:PART_OF]->(d:Document)
            WHERE r.region_id IN $region_ids
            MATCH (neighbor:Region)-[:PART_OF]->(d)
            WHERE abs(neighbor.order - r.order) <= $window_size
              AND neighbor.region_id <> r.region_id
            RETURN DISTINCT
                neighbor.region_id AS region_id,
                neighbor.label    AS label,
                neighbor.text     AS text,
                neighbor.order    AS order,
                r.region_id       AS source_region_id,
                'order'           AS source
            ORDER BY neighbor.order
        """
        try:
            results = self.query(query, {
                "region_ids": region_ids,
                "window_size": window_size,
            })
        except Exception as e:
            logger.error("Error getting order neighbors: %s", e)
            results = []

        items = [
            {
                "region_id": r.get("region_id", ""),
                "label": r.get("label", ""),
                "text": r.get("text", ""),
                "order": r.get("order", 0),
                "source_region_id": r.get("source_region_id", ""),
                "source": r.get("source", "order"),
            }
            for r in (rec.data() for rec in results)
        ]

        # --- Optionally walk PARENT edges ---
        if include_parent:
            parent_query = """
                MATCH (child:Region)-[:PARENT]->(parent:Region)
                WHERE child.region_id IN $region_ids
                RETURN DISTINCT
                    parent.region_id AS region_id,
                    parent.label     AS label,
                    parent.text      AS text,
                    parent.order     AS order,
                    child.region_id  AS source_region_id,
                    'parent'         AS source
            """
            try:
                parent_results = self.query(
                    parent_query, {"region_ids": region_ids},
                )
                for rec in parent_results:
                    data = rec.data()
                    items.append({
                        "region_id": data.get("region_id", ""),
                        "label": data.get("label", ""),
                        "text": data.get("text", ""),
                        "order": data.get("order", 0),
                        "source_region_id": data.get("source_region_id", ""),
                        "source": "parent",
                    })
            except Exception as e:
                logger.error("Error getting parent neighbors: %s", e)

        return items

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()