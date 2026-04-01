from dotenv import load_dotenv
import os
from manager import Manager, ManagerConfig
from dtype import Document

load_dotenv()


config = ManagerConfig(
    uri='neo4j://'+os.environ['URL'],
    user=os.environ['USER_NEO4J'],
    password=os.environ['PASSWORD'],
    name_db=os.environ['NAME_DB']
)

import json
with open('test_file.json', 'r') as f:
    data = json.load(f)

doc_manager = Manager(config)

document1 = Document(data, mode='mineru', name='hash_name')
doc_manager.add_document(document1)
document2 = Document(data, mode='mineru', name='hash_name2')
doc_manager.add_document(document2)

doc_manager.delete_document(name='hash_name2')
doc_manager.status()