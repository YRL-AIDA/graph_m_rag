from typing import Annotated, List, Union
from pydantic import BaseModel, ConfigDict, SkipValidation

# ========== Schemas (copy from the task) ==========
class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str
    text: Union[str, None] = None
    image: Union[str, None] = None
    image_url: Union[str, None] = None

class EmbedRequest(BaseModel):
    messages: List[Message]

class EmbedResponse(BaseModel):
    message_id: int
    embedding: List[float]
