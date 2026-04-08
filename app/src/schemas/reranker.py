from typing import List, Union
from pydantic import BaseModel, ConfigDict

class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')
    type: str
    text: Union[str, None] = None
    image: Union[str, None] = None
    image_url: Union[str, None] = None

class RerankRequest(BaseModel):
    instruction: str
    query: dict[str, str]
    fps: float = 1.0
    messages: List[Message]

class ResponseMessage(BaseModel):
    message_id: int
    score: float

class RerankResponse(BaseModel):
    messages: List[ResponseMessage]
