from pydantic import BaseModel
from typing import List, Optional
class EntityCreate(BaseModel):
    title: str
    type: str
    text_unit_ids: Optional[List[str]] = []
    frequency: Optional[int] = 0
    description: Optional[str] = None
    # любые дополнительные поля можно добавить через extra = "allow"

class RelationshipCreate(BaseModel):
    source: str
    target: str
    text_unit_ids: Optional[List[str]] = []
    weight: Optional[float] = 1.0
    description: Optional[str] = None

class EntitiesRequest(BaseModel):
    entities: List[EntityCreate]
    relationships: List[RelationshipCreate]

class EntitiesResponse(BaseModel):
    nodes_created: int
    nodes_updated: int
    relationships_added: int

class DocumentRequest(BaseModel):
    document_id: str