import base64
from typing import List, Union
from pydantic import BaseModel, ConfigDict, model_validator


class Message(BaseModel):

    type: Union[str, None] = None
    text: Union[str, None] = None
    image: Union[str, None] = None
    image_url: Union[str, None] = None

    def add_text_content(self, text: str)-> 'Message':
        self.type = 'text'
        self.text = text
        return self

    def set_type(self, type: str)-> 'Message':
        self.type = type
        return self


    def add_img_content(self, source: str = 'image_url', path_to_img: str = None, url: str = None)-> 'Message':
        match source:
            case 'image_url':
                if path_to_img is not None:
                    with open(path_to_img, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode()
                    self.type = 'image'
                    self.image = f"data:image/jpeg;base64,{base64_image}"
                elif url is not None:
                    self.type = 'image_url',
                    self.image_url = url
        return self

    def add_img_content_base64(self, base64_image: str = None)-> 'Message':
        self.type='image'
        self.image = f"data:image/jpeg;base64,{base64_image}"
        return self

#class Message(BaseModel):
#    model_config = ConfigDict(extra='ignore')
#    type: str
#    text: Union[str, None] = None
#    image: Union[str, None] = None
#    image_url: Union[str, None] = None

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
