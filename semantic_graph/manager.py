import asyncio
from dataclasses import dataclass

from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List, Set
import logging
import json
from dtype import Document,  EntityCreate, RelationshipCreate,EntitiesRequest, EntitiesResponse
import pandas as pd
import hashlib
from config import (
    COMMUNITY_CHILDREN,
    COMMUNITY_ID,
    COMMUNITY_LEVEL,
    COMMUNITY_PARENT,
    CONTENT_HASH,
    DESCRIPTION,
    DOCUMENT_ID,
    EDGE_DEGREE,
    EDGE_SOURCE,
    EDGE_TARGET,
    EDGE_WEIGHT,
    ENTITY_IDS,
    EXPLANATION,
    FINDINGS,
    FULL_CONTENT,
    FULL_CONTENT_JSON,
    ID,
    NODE_DEGREE,
    PERIOD,
    RATING,
    SHORT_ID,
    SIZE,
    SUMMARY,
    TEXT,
    TEXT_UNIT_IDS,
    TITLE,
    TYPE,
    NODE_FREQUENCY,
)
logger = logging.getLogger(__name__)
import uuid

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

    def execute_transaction(self, query, *args, db: Optional[str] = None, **kwargs):
        """Execute a function within a write transaction."""
        session = self.graph.session(database=db) if db is not None else self.graph.session()
        try:
            return session.execute_write(query, *args, **kwargs)
        finally:
            session.close()


@dataclass
class ManagerConfig:
    """Configuration for semantic graph Manager."""

    uri: str
    user: str
    password: str
    name_db: str


class Manager:
    """Document Manager for Neo4j graph operations."""

    def __init__(self, config: ManagerConfig):
        """Initialize document manager.

        Args:
            config: Manager configuration with Neo4j connection details
        """
        self.conn = Neo4jConnection(config.uri, config.user, config.password)
        self.name_db = config.name_db
        self.initialize_schema()

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

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def initialize_schema(self):
        """Initializes database constraints and vector indexes."""
        with self.conn.graph.session(database=self.name_db) as session:
            # 2. Entity Title-Type constraint
            session.run( f"CREATE CONSTRAINT entity_title_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.{TITLE}, e.{TYPE}) IS UNIQUE")



    def add_entities_batch(self, req: EntitiesRequest) -> EntitiesResponse:
        """Processes a batch of entities and relationships."""
        stats = {"nodes_created": 0, "nodes_updated": 0, "nodes_skipped": 0, "relationships_added": 0,
                 "relationships_skipped": 0}

        with self.conn.graph.session(database=self.name_db) as session:
            entity_map = {}
            # Обработка сущностей
            for entity in req.entities:
                result = session.execute_write(self._create_or_update_entity_tx, entity)
                entity_map[result[ID]] = result[ID]
                if result["action"] == "created":
                    stats["nodes_created"] += 1
                elif result["action"] == "updated":
                    stats["nodes_updated"] += 1
                else:
                    stats["nodes_skipped"] += 1

            # Обработка связей
            for rel in req.relationships:
                if rel.source in entity_map and rel.target in entity_map:
                    result = session.execute_write(self._create_relationship_tx, rel)
                    if result["action"] in ("created", "updated"):
                        stats["relationships_added"] += 1
                    else:
                        stats["relationships_skipped"] += 1

        return EntitiesResponse(
            nodes_created=stats["nodes_created"],
            nodes_updated=stats["nodes_updated"],
            relationships_added=stats["relationships_added"]
        )
    def get_entities(self) -> pd.DataFrame:
        """
        Получить все сущности Entity из Neo4j.

        Возвращает датафрейм с полями, соответствующими результату self._create_or_update_entity_tx:
        id, title, type, description, data, updated_at, created_at
        """
        query = f"""
        MATCH (e:Entity)
        RETURN 
            e.{ID} AS {ID},
            e.{TITLE} AS {TITLE},
            e.{TYPE} AS {TYPE},
            e.{DESCRIPTION} AS {DESCRIPTION},
            e.{NODE_DEGREE} AS {NODE_DEGREE},
            e.data AS data,
            e.updated_at AS updated_at,
            e.created_at AS created_at
        """
        results = self.query(query)
        # Формируем DataFrame только по этим полям
        return pd.DataFrame([{
            ID: record[TITLE]+'|'+record[TYPE],
            TITLE: record[TITLE],
            TYPE: record[TYPE],
            DESCRIPTION: record[DESCRIPTION],
            NODE_DEGREE: record[NODE_DEGREE],
            "data": record["data"],
            "updated_at": record["updated_at"],
            "created_at": record["created_at"],
        } for record in results])
 
    def get_community(self) -> pd.DataFrame:
        """
        Получить все комьюнити из Neo4j.
        """
        query = (
            f"MATCH (c:Community) RETURN c.{ID} AS {ID}, c.{TITLE} AS {TITLE}, "
            f"c.{COMMUNITY_ID} AS {COMMUNITY_ID}, c.{SHORT_ID} AS {SHORT_ID}, "
            f"c.{COMMUNITY_LEVEL} AS {COMMUNITY_LEVEL}, c.{COMMUNITY_PARENT} AS {COMMUNITY_PARENT}, "
            f"c.{SIZE} AS {SIZE}, c.{PERIOD} AS {PERIOD}"
        )
        results = self.query(query)
        # Добавляем entity_ids как список id-сущностей, связанных отношений CONSISTS_OF
        # Для каждой Community получаем связанные с ней Entity через CONSISTS_OF, формируем entity_ids = ['title|type', ...]
        communities = []
        for record in results:
            community_id = record[ID]
            entity_query = f"""
                MATCH (c:Community {{{ID}: '{community_id}'}})-[:CONSISTS_OF]->(e:Entity)
                RETURN e.{TITLE} AS {TITLE}, e.{TYPE} AS {TYPE}
            """
            children_query = f"""
                MATCH (c:Community {{{ID}: '{community_id}'}})-[:Is_PARENT_OF]->(child:Community)
                RETURN child.{COMMUNITY_ID} AS child_community
            """
            entities = self.query(entity_query)
            children = self.query(children_query)
            entity_ids = [f"{entity[TITLE]}|{entity[TYPE]}" for entity in entities]
            children_ids = [child['child_community'] for child in children]
            communities.append({
                ID: record[ID],
                TITLE: record[TITLE],
                COMMUNITY_ID: record[COMMUNITY_ID],
                SHORT_ID: record[SHORT_ID],
                COMMUNITY_LEVEL: record[COMMUNITY_LEVEL],
                COMMUNITY_PARENT: record[COMMUNITY_PARENT],
                SIZE: record[SIZE],
                PERIOD: record[PERIOD],
                ENTITY_IDS: entity_ids,
                COMMUNITY_CHILDREN: children_ids
            })
        return pd.DataFrame(communities)


    def get_entity_relationships(self) -> pd.DataFrame:
        """
        Получить все связи типа RELATED между сущностями Entity из Neo4j.

        Возвращает датафрейм с колонками:
        - source_title, source_type: идентификаторы исходной сущности
        - target_title, target_type: идентификаторы целевой сущности
        - weight, description, combined_degree, text_unit_ids, updated_at: параметры связи
        - created_at: дата создания связи (дополнительно)  

        Returns:
            pd.DataFrame: Датафрейм с результатами запроса.
                         При отсутствии результатов возвращает пустой DataFrame
                         с полным набором колонок и корректными типами данных.
        """
        # Определяем схему выходных данных для пустого результата
        columns_schema = {
            EDGE_SOURCE: "string",
            EDGE_TARGET: "string",
            EDGE_WEIGHT: "float64",
            ID: "string",
            DESCRIPTION: "string",
            EDGE_DEGREE: "int64",
            TEXT_UNIT_IDS: "object",  # list[str] или None
            "updated_at": "string",  # Neo4j datetime возвращается как строка или объект
            "created_at": "string",
        }

        cypher_query = f"""
        MATCH (source:Entity)-[r:RELATED]->(target:Entity)
        RETURN 
            source.{TITLE} AS source_title,
            source.{TYPE} AS source_type,
            target.{TITLE} AS target_title,
            target.{TYPE} AS target_type,
            r.{ID} AS {ID}, 
            r.{EDGE_WEIGHT} AS {EDGE_WEIGHT},
            r.{EDGE_DEGREE} AS {EDGE_DEGREE},
            r.{DESCRIPTION} AS {DESCRIPTION},
            r.{TEXT_UNIT_IDS} AS {TEXT_UNIT_IDS},
            r.updated_at AS updated_at,
            r.created_at AS created_at
        """

        try:
            logger.info("Executing query to fetch all RELATED relationships between Entity nodes")
            results = self.query(cypher_query)

            # Если результатов нет — возвращаем пустой DataFrame с правильной схемой
            if not results:
                logger.info("No relationships found, returning empty DataFrame with schema")
                return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in columns_schema.items()})

            # Преобразуем результаты Neo4j в список словарей
            rows = []
            for record in results:
                row = record.data()
                # Нормализуем значения: Neo4j может возвращать None для отсутствующих полей
                rows.append({
                    EDGE_SOURCE: f'{row.get("source_title")}|{row.get("source_type")}',
                    EDGE_TARGET: f'{row.get("target_title")}|{row.get("target_type")}',
                    EDGE_WEIGHT: float(row[EDGE_WEIGHT]) if row.get(EDGE_WEIGHT) is not None else None,
                    DESCRIPTION: row.get(DESCRIPTION),
                    EDGE_DEGREE: int(row[EDGE_DEGREE]) if row.get(EDGE_DEGREE) is not None else None,
                    ID: row.get(ID) or str(uuid.uuid4()),
                    TEXT_UNIT_IDS: row.get(TEXT_UNIT_IDS),  # уже list[str] или None
                    "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
                    "created_at": str(row["created_at"]) if row.get("created_at") else None,
                })

            df = pd.DataFrame(rows)

            # Приводим типы данных к ожидаемой схеме (безопасное приведение)
            for col, dtype in columns_schema.items():
                if col in df.columns:
                    if dtype == "float64":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif dtype == "string":
                        df[col] = df[col].astype("string")
                    # 'object' оставляем как есть для text_unit_ids (списки)

            logger.info(f"Successfully fetched {len(df)} relationships")
            return df

        except Exception as e:
            logger.error(f"Error fetching entity relationships: {e}")
            # При ошибке тоже возвращаем пустой DataFrame с корректной схемой
            return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in columns_schema.items()})


    @staticmethod
    def _create_or_update_entity_tx(tx, entity: EntityCreate):
        check_query = (
            f"MATCH (e:Entity {{{TITLE}: $title, {TYPE}: $type}}) "
            f"RETURN e.{TEXT_UNIT_IDS} AS existing_tuis"
        )
        record = tx.run(check_query, title=entity.title, type=entity.type).single()
        entity_id = f"{entity.title}|{entity.type}"

        new_tuis = entity.text_unit_ids or []
        if record:
            if Manager._text_unit_ids_already_exist(record["existing_tuis"], new_tuis):
                return {ID: entity_id, "action": "skipped"}

            query = f"""
                MATCH (e:Entity {{{TITLE}: $title, {TYPE}: $type}})
                SET e.{TEXT_UNIT_IDS} = CASE
                    WHEN $text_unit_ids IS NOT NULL AND e.{TEXT_UNIT_IDS} IS NOT NULL THEN e.{TEXT_UNIT_IDS} + [x IN $text_unit_ids WHERE NOT x IN e.{TEXT_UNIT_IDS}]
                    WHEN $text_unit_ids IS NOT NULL THEN $text_unit_ids ELSE e.{TEXT_UNIT_IDS} END,
                    e.{NODE_FREQUENCY} = CASE WHEN $frequency IS NOT NULL THEN COALESCE(e.{NODE_FREQUENCY}, 0) + $frequency ELSE e.{NODE_FREQUENCY} END,
                    e.{DESCRIPTION} = CASE
                    WHEN $description IS NOT NULL AND e.{DESCRIPTION} IS NOT NULL THEN e.{DESCRIPTION} + '; ' + $description
                    WHEN $description IS NOT NULL THEN $description ELSE e.{DESCRIPTION} END,
                    e.{NODE_DEGREE} = CASE WHEN $degree IS NOT NULL THEN COALESCE(e.{NODE_DEGREE}, 0) + $degree ELSE e.{NODE_DEGREE} END,
                    e.updated_at = datetime()
                RETURN e.{TITLE} AS {TITLE}, e.{TYPE} AS {TYPE}
                """
            tx.run(query, **entity.dict(exclude_unset=True))
            
            # Connect the entity to the structural graph elements (TextUnits and Documents)
            if entity.text_unit_ids:
                for tui in entity.text_unit_ids:
                    connect_query = f"""
                        MATCH (e:Entity {{{TITLE}: $title, {TYPE}: $type}})
                        MATCH (tu:TextUnit {{{ID}: $tui}})
                        MERGE (e)-[:PART_OF]->(tu)
                        
                        // Also connect to the document that contains the text unit
                        WITH e, tu
                        MATCH (tu)-[:PART_OF]->(d:Document)
                        WHERE NOT (e)-[:PART_OF_DOCUMENT]->(d)
                        MERGE (e)-[:PART_OF_DOCUMENT]->(d)
                    """
                    tx.run(connect_query, title=entity.title, type=entity.type, tui=tui)
            
            return {ID: entity_id, "action": "updated"}
        else:
            query = f"""
                CREATE (e:Entity {{{TITLE}: $title, {TYPE}: $type}})
                SET e.{TEXT_UNIT_IDS} = $text_unit_ids, e.{NODE_FREQUENCY} = $frequency,
                    e.{DESCRIPTION} = $description, e.{NODE_DEGREE} = $degree, e.created_at = datetime()
                RETURN e.{TITLE} AS {TITLE}, e.{TYPE} AS {TYPE}
                """
            result = tx.run(query, **entity.dict(exclude_unset=True)).single()
            
            # Connect the newly created entity to the structural graph elements (TextUnits and Documents)
            if entity.text_unit_ids:
                for tui in entity.text_unit_ids:
                    connect_query = f"""
                        MATCH (e:Entity {{{TITLE}: $title, {TYPE}: $type}})
                        MATCH (tu:TextUnit {{{ID}: $tui}})
                        MERGE (e)-[:PART_OF]->(tu)
                        
                        // Also connect to the document that contains the text unit
                        WITH e, tu
                        MATCH (tu)-[:PART_OF]->(d:Document)
                        WHERE NOT (e)-[:PART_OF_DOCUMENT]->(d)
                        MERGE (e)-[:PART_OF_DOCUMENT]->(d)
                    """
                    tx.run(connect_query, title=entity.title, type=entity.type, tui=tui)
            
            return {ID: f"{result[TITLE]}|{result[TYPE]}", "action": "created"}

    @staticmethod
    def _create_relationship_tx(tx, rel: RelationshipCreate):
        s_title, s_type = rel.source.split('|')
        t_title, t_type = rel.target.split('|')
        stable_id = hashlib.sha256(
            f"{rel.source}|{rel.target}|{rel.description or ''}".encode()
        ).hexdigest()[:16]
        check_query = f"""
                MATCH (s:Entity {{{TITLE}: $s_title, {TYPE}: $s_type}})-[r:RELATED]->(t:Entity {{{TITLE}: $t_title, {TYPE}: $t_type}})
                RETURN r.{TEXT_UNIT_IDS} AS existing_tuis
            """
        record = tx.run(check_query, s_title=s_title, s_type=s_type, t_title=t_title, t_type=t_type).single()

        new_tuis = rel.text_unit_ids or []
        if record and Manager._text_unit_ids_already_exist(record["existing_tuis"], new_tuis):
            return {"action": "skipped"}

        if record:
            query = f"""
                    MATCH (s:Entity {{{TITLE}: $s_title, {TYPE}: $s_type}})-[r:RELATED]->(t:Entity {{{TITLE}: $t_title, {TYPE}: $t_type}})
                    SET r.{ID} = $rel_id,
                        r.{EDGE_WEIGHT} = CASE WHEN $weight IS NOT NULL THEN COALESCE(r.{EDGE_WEIGHT}, 0) + $weight ELSE r.{EDGE_WEIGHT} END,
                        r.{DESCRIPTION} = CASE WHEN $description IS NOT NULL AND r.{DESCRIPTION} IS NOT NULL THEN r.{DESCRIPTION} + '; ' + $description
                                             WHEN $description IS NOT NULL THEN $description ELSE r.{DESCRIPTION} END,
                        r.{TEXT_UNIT_IDS} = CASE WHEN $text_unit_ids IS NOT NULL AND r.{TEXT_UNIT_IDS} IS NOT NULL THEN r.{TEXT_UNIT_IDS} + [x IN $text_unit_ids WHERE NOT x IN r.{TEXT_UNIT_IDS}]
                                               WHEN $text_unit_ids IS NOT NULL THEN $text_unit_ids ELSE r.{TEXT_UNIT_IDS} END,
                        r.{EDGE_DEGREE} = CASE WHEN $combined_degree IS NOT NULL THEN COALESCE(r.{EDGE_DEGREE}, 0) + $combined_degree ELSE r.{EDGE_DEGREE} END,
                        r.updated_at = datetime()
                """
        else:
            query = f"""
                    MATCH (s:Entity {{{TITLE}: $s_title, {TYPE}: $s_type}}), (t:Entity {{{TITLE}: $t_title, {TYPE}: $t_type}})
                    CREATE (s)-[r:RELATED]->(t)
                    SET r.{EDGE_WEIGHT} = $weight, r.{DESCRIPTION} = $description, r.{TEXT_UNIT_IDS} = $text_unit_ids, r.{EDGE_DEGREE} = $combined_degree, r.created_at = datetime()
                """
        tx.run(query, s_title=s_title, s_type=s_type, t_title=t_title, t_type=t_type,rel_id=stable_id,
               weight=rel.weight, description=rel.description, text_unit_ids=rel.text_unit_ids, combined_degree=rel.combined_degree)
        return {"action": "updated" if record else "created"}

    def insert_communities_to_neo4j(
        self,
        communities_rows: List[Dict[str, Any]],
        batch_size: int = 1000,
        document_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Двухэтапная массовая загрузка: сначала ВСЕ вершины, потом ВСЕ связи.
        Оптимизировано для минимального потребления памяти (потоковая обработка батчами).
        """
        stats = {
            "communities_created": 0,
            "parent_relations_created": 0,
            "entity_relations_created": 0
        }

        # =====================================================================
        # ЭТАП 1: ЗАГРУЗКА ВСЕХ ВЕРШИН (NODES)
        # =====================================================================
        logger.info("Этап 1: Загрузка всех вершин Community...")
        for i in range(0, len(communities_rows), batch_size):
            batch = communities_rows[i:i + batch_size]
            payload_nodes = []

            for row in batch:
                parent_id = row.get(COMMUNITY_PARENT)
                # Нормализация parent_id
                if parent_id is None or (isinstance(parent_id, float) and pd.isna(parent_id)) or parent_id == -1:
                    parent_id = -1
                
                payload_nodes.append({
                    ID: str(row[ID]),   # идентификатор комьюнити   (uuid4)
                    SHORT_ID: str(row[COMMUNITY_ID]),
                    TITLE: str(row.get(TITLE, "")),
                    COMMUNITY_ID: int(row[COMMUNITY_ID]),
                    COMMUNITY_LEVEL: int(row[COMMUNITY_LEVEL]),
                    COMMUNITY_PARENT: int(parent_id),  # Сохраняем как свойство для справки
                    SIZE: int(row.get(SIZE, 0)),
                    PERIOD: str(row.get(PERIOD, "")),
                    DOCUMENT_ID: document_id,
                })

            # Выполняем транзакцию только для узлов
            with self.conn.graph.session(database=self.name_db) as session:
                session.execute_write(self._insert_nodes_tx, payload_nodes)

            stats["communities_created"] += len(payload_nodes)
            logger.info(f"  [Этап 1] Загружено вершин: {stats['communities_created']}")

        # =====================================================================
        # ЭТАП 2: ЗАГРУЗКА ВСЕХ СВЯЗЕЙ (RELATIONSHIPS)
        # =====================================================================
        logger.info("Этап 2: Загрузка всех связей (IS_CHILD_OF, IS_PARENT_OF и CONSISTS_OF)...")

        # Мы снова проходим по communities_rows, но теперь извлекаем только данные для связей
        for i in range(0, len(communities_rows), batch_size):
            batch = communities_rows[i:i + batch_size]

            payload_parents = []
            payload_entities = []

            for row in batch:
                comm = int(row[COMMUNITY_ID])
                parent = int(row.get(COMMUNITY_PARENT))

                # 2.1. Собираем связи IS_CHILD_OF / IS_PARENT_OF       
                if parent is not None and parent != -1:
                    payload_parents.append({
                        "child": comm,    
                        COMMUNITY_PARENT: parent
                    })

                # 2.2. Собираем связи CONSISTS_OF (сплющиваем список entity_ids)
                for entity_id_str in row.get(ENTITY_IDS, []):
                    payload_entities.append({
                        COMMUNITY_ID: comm,
                        "entity_id_str": str(entity_id_str)
                    })

            # Выполняем транзакции для связей (каждая в своей сессии для чистоты)
            with self.conn.graph.session(database=self.name_db) as session:
                if payload_parents:
                    session.execute_write(self._insert_parent_relations_tx, payload_parents)
                    stats["parent_relations_created"] += len(payload_parents)

                if payload_entities:
                    session.execute_write(self._insert_entity_relations_tx, payload_entities)
                    stats["entity_relations_created"] += len(payload_entities)

            logger.info(
                f"  [Этап 2] Обработано батч связей: родителей={len(payload_parents)}, сущностей={len(payload_entities)}")

        logger.info(f"Загрузка завершена. Итоговая статистика: {stats}")
        return stats

    def save_inter_community_links(
        self,
        inter_edges: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        batch_size: int = 1000,
    ) -> Dict[str, int]:
        """Create inter-community link relationships in Neo4j.

        Each record connects two communities at the same hierarchy level
        (source_community -> target_community) and carries the shared
        entities count, aggregate weight and hierarchy level.
        """
        stats = {"links_created": 0}

        if not inter_edges:
            return stats

        for i in range(0, len(inter_edges), batch_size):
            batch = inter_edges[i:i + batch_size]
            payload: List[Dict[str, Any]] = []
            for edge in batch:
                payload.append({
                    "source": int(edge.get("source_community", -1)),
                    "target": int(edge.get("target_community", -1)),
                    "shared_entities": int(edge.get("shared_entities", 0)),
                    "weight": float(edge.get("weight", 0.0)),
                    "level": int(edge.get("level", 0)),
                    DOCUMENT_ID: document_id,
                })

            with self.conn.graph.session(database=self.name_db) as session:
                created = session.execute_write(
                    self._insert_inter_community_links_tx, payload
                )
                stats["links_created"] += created

        logger.info("Inter-community links created: %s", stats["links_created"])
        return stats

    def get_community_report_hashes(self) -> Dict[int, str]:
        """Return a mapping of community id -> content hash for incremental reports.

        Queries the stored ``content_hash`` property on each Community node.
        Used by the report pipeline to skip communities whose member list
        has not changed since the previous run.
        """
        query = f"""
        MATCH (c:Community)
        WHERE c.{CONTENT_HASH} IS NOT NULL
        RETURN c.{COMMUNITY_ID} AS {COMMUNITY_ID}, c.{CONTENT_HASH} AS {CONTENT_HASH}
        """
        try:
            results = self.query(query)
        except Exception:
            logger.exception("Failed to read community report hashes")
            return {}

        hashes: Dict[int, str] = {}
        for record in results:
            try:
                cid = int(record.get(COMMUNITY_ID))
            except (TypeError, ValueError):
                continue
            value = record.get(CONTENT_HASH)
            if value is not None:
                hashes[cid] = str(value)
        return hashes

    def update_community_reports(
        self,
        community_reports: pd.DataFrame,
        batch_size: int = 500,
    ) -> Dict[str, int]:
        """Добавляет поля отчётов к узлам Community в Neo4j по id.

        Args:
            community_reports: DataFrame — результат run_community_reports_pipeline_async
            batch_size: размер батча для UNWIND-запроса

        Returns:
            Статистика: updated, skipped (нет id), not_found (id не найден в графе)
        """
        stats = {"updated": 0, "skipped": 0, "not_found": 0}

        if community_reports is None or community_reports.empty:
            logger.info("Пустой DataFrame отчётов — обновление пропущено")
            return stats

        report_fields = (
            TITLE, SUMMARY, FULL_CONTENT, RATING,
            EXPLANATION, FINDINGS, FULL_CONTENT_JSON, CONTENT_HASH,
        )

        for i in range(0, len(community_reports), batch_size):
            batch = community_reports.iloc[i:i + batch_size]
            payload: List[Dict[str, Any]] = []

            for _, row in batch.iterrows():
                community_id = row.get(ID)
                if community_id is None or (isinstance(community_id, float) and pd.isna(community_id)):
                    stats["skipped"] += 1
                    continue

                record: Dict[str, Any] = {ID: str(community_id)}
                for field in report_fields:
                    value = row.get(field)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        record[field] = None
                    elif field == FINDINGS and hasattr(value, "tolist"):
                        record[field] = value.tolist()
                    elif field == RATING:
                        record[field] = float(value)
                    else:
                        record[field] = str(value)
                payload.append(record)

            if not payload:
                continue

            with self.conn.graph.session(database=self.name_db) as session:
                updated = session.execute_write(self._update_community_reports_tx, payload)
                stats["updated"] += updated
                stats["not_found"] += len(payload) - updated

        logger.info(
            "Обновление отчётов Community завершено: updated=%s, skipped=%s, not_found=%s",
            stats["updated"], stats["skipped"], stats["not_found"],
        )
        return stats

    # ---------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ТРАНЗАКЦИЙ (для чистоты кода)
    # ---------------------------------------------------------------------
    @staticmethod
    def _update_community_reports_tx(tx, payload: List[Dict[str, Any]]) -> int:
        query = f"""
        UNWIND $payload AS row
        MATCH (c:Community {{{ID}: row.{ID}}})
        SET c.{TITLE} = coalesce(row.{TITLE}, c.{TITLE}),
            c.{COMMUNITY_ID} = toInteger(row.{COMMUNITY_ID}),
            c.{SHORT_ID} = row.{SHORT_ID},
            c.{SUMMARY} = row.{SUMMARY},
            c.{FULL_CONTENT} = row.{FULL_CONTENT},
            c.{RATING} = row.{RATING},
            c.{EXPLANATION} = row.{EXPLANATION},
            c.{FINDINGS} = row.{FINDINGS},
            c.{FULL_CONTENT_JSON} = row.{FULL_CONTENT_JSON},
            c.{CONTENT_HASH} = row.{CONTENT_HASH},
            c.report_updated_at = datetime()
        RETURN count(c) AS updated
        """
        result = tx.run(query, payload=payload)
        record = result.single()
        return int(record["updated"]) if record else 0

    @staticmethod
    def _insert_nodes_tx(tx, payload: List[Dict[str, Any]]):
        query = f"""
        UNWIND $payload AS row
        MERGE (c:Community {{{ID}: row.{ID}}})
        SET c.{COMMUNITY_LEVEL} = toInteger(row.{COMMUNITY_LEVEL}),
            c.{TITLE} = row.{TITLE},
            c.{COMMUNITY_ID} = toInteger(row.{COMMUNITY_ID}),
            c.{SHORT_ID} = row.{SHORT_ID},
            c.{DOCUMENT_ID} = row.{DOCUMENT_ID},
            c.{COMMUNITY_PARENT} = toInteger(row.{COMMUNITY_PARENT}),
            c.{SIZE} = toInteger(row.{SIZE}),
            c.{PERIOD} = row.{PERIOD}
        """
        tx.run(query, payload=payload)

    @staticmethod
    def _insert_parent_relations_tx(tx, payload: List[Dict[str, Any]]):
        # Используем MERGE для обоих узлов на случай, если родительское комьюнити
        # еще не было создано (например, при частичной загрузке данных)
        query = f"""
        UNWIND $payload AS row
        MERGE (child:Community {{{COMMUNITY_ID}: row.child}})
        MERGE (parent:Community {{{COMMUNITY_ID}: row.{COMMUNITY_PARENT}}})
        MERGE (child)-[:IS_CHILD_OF]->(parent)
        MERGE (parent)-[:IS_PARENT_OF]->(child)
        """
        tx.run(query, payload=payload)

    @staticmethod
    def _insert_entity_relations_tx(tx, payload: List[Dict[str, Any]]):
        query = f"""
        UNWIND $payload AS row
        MATCH (c:Community {{{COMMUNITY_ID}: row.{COMMUNITY_ID}}}) // MATCH, т.к. на Этапе 1 мы гарантированно создали все Community
        WITH c, row, split(toString(row.entity_id_str), '|') AS parts
        WHERE size(parts) = 2
        MERGE (e:Entity {{{TITLE}: parts[0], {TYPE}: parts[1]}})
        MERGE (c)-[:CONSISTS_OF]->(e)
        
        // Create connection between the Community and the Document that contains the Entity
        // This links the semantic graph (Communities) to the structural graph (Documents)
        WITH c, e
        MATCH (e)-[:PART_OF]->(tu:TextUnit)-[:PART_OF]->(d:Document)
        WHERE NOT (c)-[:CONNECTED_TO_DOCUMENT]->(d)
        MERGE (c)-[:CONNECTED_TO_DOCUMENT]->(d)
        """
        tx.run(query, payload=payload)

    @staticmethod
    def _insert_inter_community_links_tx(tx, payload: List[Dict[str, Any]]) -> int:
        query = f"""
        UNWIND $payload AS row
        MATCH (source:Community {{{COMMUNITY_ID}: row.source}})
        MATCH (target:Community {{{COMMUNITY_ID}: row.target}})
        MERGE (source)-[r:INTER_COMMUNITY_LINK]->(target)
        SET r.{DOCUMENT_ID} = row.{DOCUMENT_ID},
            r.shared_entities = toInteger(row.shared_entities),
            r.{EDGE_WEIGHT} = toFloat(row.weight),
            r.{COMMUNITY_LEVEL} = toInteger(row.level)
        RETURN count(r) AS created
        """
        result = tx.run(query, payload=payload)
        record = result.single()
        return int(record["created"]) if record else 0

    # --- ВСПОМОГАТЕЛЬНЫЕ ЛОГИЧЕСКИЕ ФУНКЦИИ ---
    def add_structural_link(self, semantic_node_id: str, structural_node_id: str, relationship_type: str = "STRUCTURAL_CONNECTION"):
        """Add a link between a semantic graph node and a structural graph node.

        Args:
            semantic_node_id: ID of the node in the semantic graph
            structural_node_id: ID of the node in the structural graph
            relationship_type: Type of relationship between the nodes
        """
        try:
            query = """
            MATCH (sn)
            WHERE elementId(sn) = $semantic_node_id
            MATCH (dn)
            WHERE elementId(dn) = $structural_node_id
            MERGE (sn)-[:LINKED_TO {relationship_type: $relationship_type, created_at: datetime()}]->(dn)
            """
            with self.conn.graph.session(database=self.name_db) as session:
                session.run(query, {
                    "semantic_node_id": semantic_node_id,
                    "structural_node_id": structural_node_id,
                    "relationship_type": relationship_type
                })
            logger.info(f"Added structural link from {semantic_node_id} to {structural_node_id}")
        except Exception as e:
            logger.error(f"Error adding structural link: {e}")
            raise

    def get_structural_links(self, semantic_node_id: str):
        """Get structural graph nodes linked to a semantic graph node.

        Args:
            semantic_node_id: ID of the node in the semantic graph

        Returns:
            List of linked structural graph nodes
        """
        try:
            query = """
            MATCH (sn)
            WHERE elementId(sn) = $semantic_node_id
            OPTIONAL MATCH (sn)-[r:LINKED_TO]->(dn)
            RETURN elementId(dn) AS structural_node_id, r.relationship_type AS relationship_type, r.created_at AS created_at
            """
            with self.conn.graph.session(database=self.name_db) as session:
                result = session.run(query, {"semantic_node_id": semantic_node_id})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error getting structural links: {e}")
            return []

    @staticmethod
    def _text_unit_ids_already_exist(existing: Optional[List[str]], new: Optional[List[str]]) -> bool:
        if not new: return True
        if not existing: return False
        existing_set = set(existing)
        return all(tuid in existing_set for tuid in new)

    def close(self):
        """Close the database connection."""
        self.conn.close()

    # ---------------------------------------------------------------------
    # Cross-graph query methods (Entity ↔ Region via semantic_link bridge)
    # ---------------------------------------------------------------------

    async def run_cypher(self, cypher: str, **params):
        """Execute a Cypher query asynchronously."""
        return await asyncio.to_thread(
            self.conn.query, cypher, self.name_db, params
        )

    def get_entities_by_region_ids(
        self, region_ids: List[str], limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return Entity nodes linked to the given Region IDs via
        ``semantic_link`` edges."""
        query = f"""
        MATCH (e:Entity)-[sl:semantic_link]->(r:Region)
        WHERE r.region_id IN $region_ids
        WITH e, sl, r
        ORDER BY COALESCE(sl.weight, 0.5) DESC
        LIMIT $limit
        RETURN DISTINCT
            e.{TITLE} AS title,
            e.{TYPE} AS type,
            e.{DESCRIPTION} AS description,
            COALESCE(e.{NODE_DEGREE}, 0) AS degree,
            sl.weight AS weight,
            r.region_id AS region_id
        """
        try:
            results = self.query(query, {
                "region_ids": region_ids, "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_entities_by_region_ids failed: %s", e)
            return []

    def get_communities_by_region_ids(
        self, region_ids: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return Community nodes whose entities are linked (via CONSISTS_OF
        + semantic_link) to the given Region IDs.  Includes community
        report findings, rating, and full content."""
        query = f"""
        MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)-[sl:semantic_link]->(r:Region)
        WHERE r.region_id IN $region_ids
        WITH c, e, sl, r
        ORDER BY COALESCE(sl.weight, 0.5) DESC
        LIMIT $limit
        RETURN DISTINCT
            c.{TITLE} AS title,
            c.{SUMMARY} AS summary,
            c.{SIZE} AS size,
            c.{FINDINGS} AS findings,
            c.{RATING} AS rating,
            c.{EXPLANATION} AS rating_explanation,
            c.{FULL_CONTENT} AS full_content,
            r.region_id AS region_id
        """
        try:
            results = self.query(query, {
                "region_ids": region_ids, "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_communities_by_region_ids failed: %s", e)
            return []

    def get_cross_graph_bridge(
        self, region_ids: List[str], max_regions: int = 15
    ) -> List[Dict[str, Any]]:
        """Discover new Region nodes reachable through the Entity graph:
        Region →(semantic_link)→ Entity →(RELATED)→ Entity →(semantic_link)→ Region."""
        query = f"""
        MATCH (r1:Region)<-[sl1:semantic_link]-(e1:Entity)-[rel:RELATED]-(e2:Entity)-[sl2:semantic_link]->(r2:Region)
        WHERE r1.region_id IN $region_ids
          AND NOT r2.region_id IN $region_ids
          AND e1 <> e2
        WITH r2, sl2.weight AS bridge_weight
        ORDER BY bridge_weight DESC
        LIMIT $max_regions
        RETURN DISTINCT
            r2.region_id AS region_id,
            labels(r2) AS labels,
            r2.text AS text
        """
        try:
            results = self.query(query, {
                "region_ids": region_ids, "max_regions": max_regions,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_cross_graph_bridge failed: %s", e)
            return []

    def get_related_entity_links(
        self, entity_titles: List[str], limit: int = 20
    ) -> Dict[str, List[str]]:
        """For each entity title, return titles of directly RELATED entities.

        Only counts relationships where *both* entities appear in the
        supplied ``entity_titles`` list (i.e. entities present in the
        current context), so the caller can show cross-references.
        """
        if not entity_titles:
            return {}
        query = f"""
        MATCH (e1:Entity)-[:RELATED]-(e2:Entity)
        WHERE e1.title IN $titles
          AND e2.title IN $titles
          AND e1.title <> e2.title
        RETURN e1.title AS source, collect(DISTINCT e2.title) AS related
        LIMIT $limit
        """
        try:
            results = self.query(query, {
                "titles": entity_titles, "limit": limit,
            })
            out: Dict[str, List[str]] = {}
            for r in results:
                data = r.data()
                src = data.get("source", "")
                rel = data.get("related", [])
                if src and rel:
                    out[src] = rel
            return out
        except Exception as e:
            logger.warning("get_related_entity_links failed: %s", e)
            return {}

    def get_entities_linked_to_region(
        self, region_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return Entity nodes linked to a single Region via ``semantic_link``."""
        query = f"""
        MATCH (e:Entity)-[sl:semantic_link]->(r:Region)
        WHERE r.region_id = $region_id
        RETURN DISTINCT
            e.{TITLE} AS title,
            e.{TYPE} AS type,
            e.{DESCRIPTION} AS description,
            COALESCE(e.{NODE_DEGREE}, 0) AS degree,
            sl.weight AS weight
        ORDER BY sl.weight DESC
        LIMIT $limit
        """
        try:
            results = self.query(query, {
                "region_id": region_id, "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_entities_linked_to_region failed: %s", e)
            return []

    def get_regions_linked_to_entity(
        self, entity_title: str, entity_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return Region nodes linked to an Entity via ``semantic_link``."""
        query = f"""
        MATCH (e:Entity)-[sl:semantic_link]->(r:Region)
        WHERE e.{TITLE} = $entity_title AND e.{TYPE} = $entity_type
        RETURN DISTINCT
            r.region_id AS region_id,
            r.text AS text,
            sl.weight AS weight
        ORDER BY sl.weight DESC
        LIMIT $limit
        """
        try:
            results = self.query(query, {
                "entity_title": entity_title,
                "entity_type": entity_type,
                "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_regions_linked_to_entity failed: %s", e)
            return []

    def get_regions_linked_to_community(
        self, community_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return Region nodes reachable from a Community:
        Community →(CONSISTS_OF)→ Entity →(semantic_link)→ Region."""
        query = f"""
        MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)-[sl:semantic_link]->(r:Region)
        WHERE c.{COMMUNITY_ID} = toInteger($community_id)
           OR c.{ID} = $community_id
        RETURN DISTINCT
            r.region_id AS region_id,
            r.text AS text,
            sl.weight AS weight,
            '2hop' AS source
        ORDER BY sl.weight DESC
        LIMIT $limit
        """
        try:
            results = self.query(query, {
                "community_id": community_id,
                "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_regions_linked_to_community failed: %s", e)
            return []

    def get_community_siblings(
        self, entity_title: str, entity_type: str, max_siblings: int = 5
    ) -> List[Dict[str, Any]]:
        """Return other entities in the same communities as the given entity."""
        query = f"""
        MATCH (e:Entity {{{TITLE}: $entity_title, {TYPE}: $entity_type}})<-[:CONSISTS_OF]-(c:Community)-[:CONSISTS_OF]->(sib:Entity)
        WHERE sib.{TITLE} <> $entity_title OR sib.{TYPE} <> $entity_type
        RETURN DISTINCT
            sib.{TITLE} AS title,
            sib.{TYPE} AS type,
            sib.{DESCRIPTION} AS description,
            COALESCE(sib.{NODE_DEGREE}, 0) AS degree,
            c.{COMMUNITY_ID} AS community_id,
            c.{TITLE} AS community_title
        LIMIT $max_siblings
        """
        try:
            results = self.query(query, {
                "entity_title": entity_title,
                "entity_type": entity_type,
                "max_siblings": max_siblings,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_community_siblings failed: %s", e)
            return []

    def get_community_members(
        self, community_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return Entity members of a Community node."""
        query = f"""
        MATCH (c:Community)-[:CONSISTS_OF]->(e:Entity)
        WHERE c.{COMMUNITY_ID} = toInteger($community_id)
           OR c.{ID} = $community_id
        RETURN DISTINCT
            e.{TITLE} AS title,
            e.{TYPE} AS type,
            e.{DESCRIPTION} AS description,
            COALESCE(e.{NODE_DEGREE}, 0) AS degree
        LIMIT $limit
        """
        try:
            results = self.query(query, {
                "community_id": community_id,
                "limit": limit,
            })
            return [r.data() for r in results]
        except Exception as e:
            logger.warning("get_community_members failed: %s", e)
            return []

    def get_related_entities_2hop(
        self,
        entity_title: str,
        entity_type: str,
        max_degree: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return entities reachable from the given entity via 1-hop and
        2-hop ``RELATED`` edges."""
        query = f"""
        MATCH (e:Entity {{{TITLE}: $entity_title, {TYPE}: $entity_type}})
        OPTIONAL MATCH (e)-[r1:RELATED]-(hop1:Entity)
        OPTIONAL MATCH (hop1)-[r2:RELATED]-(hop2:Entity)
        WHERE hop2 <> e
        RETURN DISTINCT
            hop1.{TITLE} AS title_1,
            hop1.{TYPE} AS type_1,
            hop1.{DESCRIPTION} AS desc_1,
            COALESCE(r1.{EDGE_WEIGHT}, 0.5) AS weight_1,
            1 AS hop_depth,
            hop2.{TITLE} AS title_2,
            hop2.{TYPE} AS type_2,
            hop2.{DESCRIPTION} AS desc_2,
            COALESCE(r2.{EDGE_WEIGHT}, 0.3) AS weight_2,
            2 AS hop_depth_2
        LIMIT $max_degree
        """
        try:
            rows = self.query(query, {
                "entity_title": entity_title,
                "entity_type": entity_type,
                "max_degree": max_degree,
            })
            # Normalise: if a row has hop2 fields, emit a 2-hop row;
            # otherwise emit a 1-hop row.
            entities: List[Dict[str, Any]] = []
            seen: set = set()
            for record in rows:
                data = record.data()
                # 1-hop
                t1 = data.get("title_1")
                if t1:
                    key = f"{t1}|{data.get('type_1', '')}"
                    if key not in seen:
                        seen.add(key)
                        entities.append({
                            "title": t1,
                            "type": data.get("type_1", ""),
                            "description": data.get("desc_1", ""),
                            "weight": data.get("weight_1", 0.5),
                            "hop_depth": 1,
                        })
                # 2-hop
                t2 = data.get("title_2")
                if t2:
                    key = f"{t2}|{data.get('type_2', '')}"
                    if key not in seen:
                        seen.add(key)
                        entities.append({
                            "title": t2,
                            "type": data.get("type_2", ""),
                            "description": data.get("desc_2", ""),
                            "weight": data.get("weight_2", 0.3),
                            "hop_depth": 2,
                        })
            return entities
        except Exception as e:
            logger.warning("get_related_entities_2hop failed: %s", e)
            return []
