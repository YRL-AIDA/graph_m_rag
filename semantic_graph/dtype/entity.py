from pydantic import BaseModel
from typing import List, Optional
class EntityCreate(BaseModel):
    title: str
    type: str
    text_unit_ids: Optional[List[str]] = []
    frequency: Optional[int] = 0
    description: Optional[str] = None
    degree: Optional[int] = 0
    confidence: Optional[int] = 5

class RelationshipCreate(BaseModel):
    source: str
    target: str
    text_unit_ids: Optional[List[str]] = []
    weight: Optional[float] = 1.0
    description: Optional[str] = None
    combined_degree: Optional[int] = 0

class EntitiesRequest(BaseModel):
    entities: List[EntityCreate]
    relationships: List[RelationshipCreate]

class EntitiesResponse(BaseModel):
    nodes_created: int
    nodes_updated: int
    relationships_added: int

class DocumentRequest(BaseModel):
    document_id: str