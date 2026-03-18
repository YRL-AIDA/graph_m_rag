from neo4j import GraphDatabase
from dtype import Document


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.graph = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.graph is not None:
            self.graph.close()

    def query(self, query, db=None):
        assert self.graph is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.graph.session(database=db) if db is not None else self.graph.session()
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

class ManagerConfig:
    def __init__(self, uri, user, password, name_db):
        self.uri = uri
        self.user = user
        self.password = password
        self.name_db = name_db
        print(self.uri, self.user, self.password, self.name_db)

class Manager:
    def __init__(self, config):
        self.conn = Neo4jConnection(config.uri, config.user, config.password)
        self.name_db = config.name_db
    
    def add_document(self, document:Document):
        if not self.is_document_exist(document.name):
            graph = document.get_graph()
            query = ""
            query += f"CREATE (d:Document {{name: '{document.name}'}})\n"
            for id, reg in graph['nodes']['regions'].items():
                label = reg['label']
                text = reg['text']
                query += f"CREATE (reg{id}:Region:{label} {{text: '{text}'}})\n"

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
            # print(query)
            self.query(query)   
    
    def is_document_exist(self, name):
        doc_exist = self.query(f"OPTIONAL MATCH (d:Document) RETURN '{name}' in d.name as exist")[0].data()['exist']
        return doc_exist

    def delete_document(self, name):
        query=f"""
        MATCH path = (m:Document {{name: '{name}'}}) -[:ORDER*]-> (n), () -[r2:PARENT]-> (n)
        WITH m, n, r2, relationships(path) AS order_rels
        FOREACH (rel IN order_rels | DELETE rel)
        DELETE r2, m, n
        """
        self.query(query)
    
    def query(self, query):
        return self.conn.query(query, None)
    

    def status(self):
        rez = self.query("MATCH (n) RETURN count(n) as count")
        print(rez)