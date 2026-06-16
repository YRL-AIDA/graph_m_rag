from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List
import logging
import json
from dtype import Document,  EntityCreate, RelationshipCreate,EntitiesRequest, EntitiesResponse
import pandas as pd
import hashlib
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

    def execute_transaction(self, query, *args, db: Optional[str] = None, **kwargs):
        """Execute a function within a write transaction."""
        session = self.graph.session(database=db) if db is not None else self.graph.session()
        try:
            return session.execute_write(query, *args, **kwargs)
        finally:
            session.close()
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

    def initialize_schema(self):
        """Initializes database constraints and vector indexes."""
        with self.conn.graph.session(database=self.name_db) as session:
            # 2. Entity Title-Type constraint
            session.run(
                "CREATE CONSTRAINT entity_title_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.title, e.type) IS UNIQUE")



    def add_entities_batch(self, req: EntitiesRequest) -> EntitiesResponse:
        """Processes a batch of entities and relationships."""
        stats = {"nodes_created": 0, "nodes_updated": 0, "nodes_skipped": 0, "relationships_added": 0,
                 "relationships_skipped": 0}

        with self.conn.graph.session(database=self.name_db) as session:
            entity_map = {}
            # Обработка сущностей
            for entity in req.entities:
                result = session.execute_write(self._create_or_update_entity_tx, entity)
                entity_map[result["id"]] = result["id"]
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
        query = """
        MATCH (e:Entity)
        RETURN 
            e.id AS id,
            e.title AS title,
            e.type AS type,
            e.description AS description,
            e.degree AS degree,
            e.data AS data,
            e.updated_at AS updated_at,
            e.created_at AS created_at
        """
        results = self.query(query)
        # Формируем DataFrame только по этим полям
        return pd.DataFrame([{
            "id": record["title"]+'|'+record["type"],
            "title": record["title"],
            "type": record["type"],
            "description": record["description"],
            "degree": record["degree"],
            "data": record["data"],
            "updated_at": record["updated_at"],
            "created_at": record["created_at"],
        } for record in results])
 
    def get_community(self) -> pd.DataFrame:
        """
        Получить все комьюнити из Neo4j.
        """
        query = "MATCH (c:Community) RETURN c.id AS id, c.title AS title, c.level AS level, c.parent_id AS parent, c.size AS size, c.period AS period"
        results = self.query(query)
        # Добавляем entity_ids как список id-сущностей, связанных отношений CONSISTS_OF
        # Для каждой Community получаем связанные с ней Entity через CONSISTS_OF, формируем entity_ids = ['title|type', ...]
        communities = []
        for record in results:
            community_id = record["id"]
            entity_query = f"""
                MATCH (c:Community {{id: '{community_id}'}})-[:CONSISTS_OF]->(e:Entity)
                RETURN e.title AS title, e.type AS type
            """
            entities = self.query(entity_query)
            entity_ids = [f"{entity['title']}|{entity['type']}" for entity in entities]
            communities.append({
                "id": record["id"],
                "title": record["title"],
                "level": record["level"],
                "parent": record["parent"],
                "size": record["size"],
                "period": record["period"],
                "entity_ids": entity_ids
            })
        return pd.DataFrame(communities)


    def get_entity_relationships(self) -> pd.DataFrame:
        """
        Получить все связи типа RELATED между сущностями Entity из Neo4j.

        Возвращает датафрейм с колонками:
        - source_title, source_type: идентификаторы исходной сущности
        - target_title, target_type: идентификаторы целевой сущности
        - weight, description, text_unit_ids, updated_at: параметры связи
        - created_at: дата создания связи (дополнительно)

        Returns:
            pd.DataFrame: Датафрейм с результатами запроса.
                         При отсутствии результатов возвращает пустой DataFrame
                         с полным набором колонок и корректными типами данных.
        """
        # Определяем схему выходных данных для пустого результата
        columns_schema = {
            "source": "string",
            "target": "string",
            "weight": "float64",
            "id": "string",
            "description": "string",
            "degree": "int64",
            "text_unit_ids": "object",  # list[str] или None
            "updated_at": "string",  # Neo4j datetime возвращается как строка или объект
            "created_at": "string",
        }

        cypher_query = """
        MATCH (source:Entity)-[r:RELATED]->(target:Entity)
        RETURN 
            source.title AS source_title,
            source.type AS source_type,
            target.title AS target_title,
            target.type AS target_type,
            r.id AS id, 
            r.weight AS weight,
            r.degree AS degree,
            r.description AS description,
            r.text_unit_ids AS text_unit_ids,
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
                    "source": f'{row.get("source_title")}|{row.get("source_type")}',
                    "target": f'{row.get("target_title")}|{row.get("target_type")}',
                    "weight": float(row["weight"]) if row.get("weight") is not None else None,
                    "description": row.get("description"),
                    "degree": int(row["degree"]) if row.get("degree") is not None else None,
                    "id": row.get("id") or str(uuid.uuid4()),
                    "text_unit_ids": row.get("text_unit_ids"),  # уже list[str] или None
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
        check_query = "MATCH (e:Entity {title: $title, type: $type}) RETURN e.text_unit_ids AS existing_tuis"
        record = tx.run(check_query, title=entity.title, type=entity.type).single()
        entity_id = f"{entity.title}|{entity.type}"

        new_tuis = entity.text_unit_ids or []
        if record:
            if Manager._text_unit_ids_already_exist(record["existing_tuis"], new_tuis):
                return {"id": entity_id, "action": "skipped"}

            query = """
                MATCH (e:Entity {title: $title, type: $type})
                SET e.text_unit_ids = CASE 
                    WHEN $text_unit_ids IS NOT NULL AND e.text_unit_ids IS NOT NULL THEN apoc.coll.toSet(e.text_unit_ids + $text_unit_ids)
                    WHEN $text_unit_ids IS NOT NULL THEN $text_unit_ids ELSE e.text_unit_ids END,
                    e.frequency = CASE WHEN $frequency IS NOT NULL THEN COALESCE(e.frequency, 0) + $frequency ELSE e.frequency END,
                    e.description = CASE 
                    WHEN $description IS NOT NULL AND e.description IS NOT NULL THEN e.description + '; ' + $description
                    WHEN $description IS NOT NULL THEN $description ELSE e.description END,
                    e.degree = CASE WHEN $degree IS NOT NULL THEN COALESCE(e.degree, 0) + $degree ELSE e.degree END,    
                    e.updated_at = datetime()
                RETURN e.title AS title, e.type AS type
                """
            tx.run(query, **entity.dict(exclude_unset=True))
            return {"id": entity_id, "action": "updated"}
        else:
            query = """
                CREATE (e:Entity {title: $title, type: $type})
                SET e.text_unit_ids = $text_unit_ids, e.frequency = $frequency, 
                    e.description = $description, e.degree = $degree, e.created_at = datetime()
                RETURN e.title AS title, e.type AS type
                """
            result = tx.run(query, **entity.dict(exclude_unset=True)).single()
            return {"id": f"{result['title']}|{result['type']}", "action": "created"}

    @staticmethod
    def _create_relationship_tx(tx, rel: RelationshipCreate):
        s_title, s_type = rel.source.split('|')
        t_title, t_type = rel.target.split('|')
        stable_id = hashlib.sha256(
            f"{rel.source}|{rel.target}|{rel.description or ''}".encode()
        ).hexdigest()[:16]
        check_query = """
                MATCH (s:Entity {title: $s_title, type: $s_type})-[r:RELATED]->(t:Entity {title: $t_title, type: $t_type})
                RETURN r.text_unit_ids AS existing_tuis
            """
        record = tx.run(check_query, s_title=s_title, s_type=s_type, t_title=t_title, t_type=t_type).single()

        new_tuis = rel.text_unit_ids or []
        if record and Manager._text_unit_ids_already_exist(record["existing_tuis"], new_tuis):
            return {"action": "skipped"}

        if record:
            query = """
                    MATCH (s:Entity {title: $s_title, type: $s_type})-[r:RELATED]->(t:Entity {title: $t_title, type: $t_type})
                    SET r.id = $rel_id
                        r.weight = CASE WHEN $weight IS NOT NULL THEN COALESCE(r.weight, 0) + $weight ELSE r.weight END,
                        r.description = CASE WHEN $description IS NOT NULL AND r.description IS NOT NULL THEN r.description + '; ' + $description
                                             WHEN $description IS NOT NULL THEN $description ELSE r.description END,
                        r.text_unit_ids = CASE WHEN $text_unit_ids IS NOT NULL AND r.text_unit_ids IS NOT NULL THEN apoc.coll.toSet(r.text_unit_ids + $text_unit_ids)
                                               WHEN $text_unit_ids IS NOT NULL THEN $text_unit_ids ELSE r.text_unit_ids END,
                        r.combined_degree = CASE WHEN $combined_degree IS NOT NULL THEN COALESCE(r.combined_degree, 0) + $combined_degree ELSE r.combined_degree END,
                        r.updated_at = datetime()
                """
        else:
            query = """
                    MATCH (s:Entity {title: $s_title, type: $s_type}), (t:Entity {title: $t_title, type: $t_type})
                    CREATE (s)-[r:RELATED]->(t)
                    SET r.weight = $weight, r.description = $description, r.text_unit_ids = $text_unit_ids, r.combined_degree = $combined_degree, r.created_at = datetime()
                """
        tx.run(query, s_title=s_title, s_type=s_type, t_title=t_title, t_type=t_type,rel_id=stable_id,
               weight=rel.weight, description=rel.description, text_unit_ids=rel.text_unit_ids, combined_degree=rel.combined_degree)
        return {"action": "updated" if record else "created"}

    def insert_communities_to_neo4j(self, communities_rows: List[Dict[str, Any]], batch_size: int = 1000) -> Dict[
        str, int]:
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
                parent_id = row.get("parent")
                # Нормализация parent_id
                if parent_id is None or (isinstance(parent_id, float) and pd.isna(parent_id)) or parent_id == -1:
                    parent_id = None
                else:
                    parent_id = str(int(parent_id))

                payload_nodes.append({
                    "id": str(row["id"]),   # идентификатор комьюнити   (uuid4)
                    #"human_readable_id": str(row["human_readable_id"]),
                    "title": str(row.get("title", "")),
                    "community": str(int(row["community"])),
                    "level": int(row["level"]),
                    "parent_id": parent_id,  # Сохраняем как свойство для справки
                    "size": int(row.get("size", 0)),
                    "period": str(row.get("period", ""))
                })

            # Выполняем транзакцию только для узлов
            with self.conn.graph.session(database=self.name_db) as session:
                session.execute_write(self._insert_nodes_tx, payload_nodes)

            stats["communities_created"] += len(payload_nodes)
            logger.info(f"  [Этап 1] Загружено вершин: {stats['communities_created']}")

        # =====================================================================
        # ЭТАП 2: ЗАГРУЗКА ВСЕХ СВЯЗЕЙ (RELATIONSHIPS)
        # =====================================================================
        logger.info("Этап 2: Загрузка всех связей (IS_CHILD_OF и CONSISTS_OF)...")

        # Мы снова проходим по communities_rows, но теперь извлекаем только данные для связей
        for i in range(0, len(communities_rows), batch_size):
            batch = communities_rows[i:i + batch_size]

            payload_parents = []
            payload_entities = []

            for row in batch:
                comm_id = str(int(row["community"]))
                parent_id = row.get("parent")

                # 2.1. Собираем связи IS_CHILD_OF
                if parent_id is not None and not (
                        isinstance(parent_id, float) and pd.isna(parent_id)) and parent_id != -1:
                    payload_parents.append({
                        "child_id": comm_id,
                        "parent_id": str(int(parent_id))
                    })

                # 2.2. Собираем связи CONSISTS_OF (сплющиваем список entity_ids)
                for entity_id_str in row.get("entity_ids", []):
                    payload_entities.append({
                        "community_id": comm_id,
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
            "title", "summary", "full_content", "rank",
            "rating_explanation", "findings", "full_content_json",
        )

        for i in range(0, len(community_reports), batch_size):
            batch = community_reports.iloc[i:i + batch_size]
            payload: List[Dict[str, Any]] = []

            for _, row in batch.iterrows():
                community_id = row.get("id")
                if community_id is None or (isinstance(community_id, float) and pd.isna(community_id)):
                    stats["skipped"] += 1
                    continue

                record: Dict[str, Any] = {"id": str(community_id)}
                for field in report_fields:
                    value = row.get(field)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        record[field] = None
                    elif field == "findings" and hasattr(value, "tolist"):
                        record[field] = value.tolist()
                    elif field == "rank":
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
        query = """
        UNWIND $payload AS row
        MATCH (c:Community {id: row.id})
        SET c.title = coalesce(row.title, c.title),
            c.summary = row.summary,
            c.full_content = row.full_content,
            c.rank = row.rank,
            c.rating_explanation = row.rating_explanation,
            c.findings = row.findings,
            c.full_content_json = row.full_content_json,
            c.report_updated_at = datetime()
        RETURN count(c) AS updated
        """
        result = tx.run(query, payload=payload)
        record = result.single()
        return int(record["updated"]) if record else 0

    @staticmethod
    def _insert_nodes_tx(tx, payload: List[Dict[str, Any]]):
        query = """
        UNWIND $payload AS row
        MERGE (c:Community {id: row.id})
        SET c.level = toInteger(row.level),
            c.title = row.title,
            c.parent_id = row.parent_id,
            c.size = toInteger(row.size),
            c.period = row.period
        """
        tx.run(query, payload=payload)

    @staticmethod
    def _insert_parent_relations_tx(tx, payload: List[Dict[str, Any]]):
        # Используем MERGE для обоих узлов на случай, если родительское комьюнити
        # еще не было создано (например, при частичной загрузке данных)
        query = """
        UNWIND $payload AS row
        MERGE (child:Community {id: row.child_id})
        MERGE (parent:Community {id: row.parent_id})
        MERGE (child)-[:IS_CHILD_OF]->(parent)
        """
        tx.run(query, payload=payload)

    @staticmethod
    def _insert_entity_relations_tx(tx, payload: List[Dict[str, Any]]):
        query = """
        UNWIND $payload AS row
        MATCH (c:Community {id: row.id}) // MATCH, т.к. на Этапе 1 мы гарантированно создали все Community
        WITH c, row, split(toString(row.entity_id_str), '|') AS parts
        WHERE size(parts) = 2
        MERGE (e:Entity {title: parts[0], type: parts[1]})
        MERGE (c)-[:CONSISTS_OF]->(e)
        """
        tx.run(query, payload=payload)
    # --- ВСПОМОГАТЕЛЬНЫЕ ЛОГИЧЕСКИЕ ФУНКЦИИ ---
    @staticmethod
    def _text_unit_ids_already_exist(existing: Optional[List[str]], new: Optional[List[str]]) -> bool:
        if not new: return True
        if not existing: return False
        existing_set = set(existing)
        return all(tuid in existing_set for tuid in new)
