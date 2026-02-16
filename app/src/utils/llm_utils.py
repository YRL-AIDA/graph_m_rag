from openai import OpenAI
import base64
from typing import List, Dict, Any, Callable
import inspect


def get_kwargs(kwargs: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    '''
        Вытаскивает аргументы из kwargs по сигнатуре функции func.
    '''
    sig = inspect.signature(func)
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


class ModelMessageDict(dict):
    '''
        Класс - словарь для удобого форматирования запроса к модели.
        Формирует словарь для передачи в клиента openia и в модель
    '''

    def __init__(self, role: str = 'user'):
        super().__init__()
        self['role'] = role
        self['content'] = []

    def add_text_content(self, content: str):
        self['content'].append({'type': 'text',
                                'text': content})

    def add_img_content(self, source: str = 'image_url', path_to_img: str = None, url: str = None):
        match source:
            case 'image_url':
                if path_to_img != None:
                    with open(path_to_img, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode()
                    self['content'].append({'type': 'image_url',
                                            'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}})
                elif url != None:
                    self['content'].append({'type': 'image_url',
                                            'image_url': {'url': url}})


def send_messasge(messages: List[ModelMessageDict], base_url: str = "http://192.168.19.127:8888/v1",
                  api_key: str = 'EMPTY',
                  model_name: str = 'Qwen/Qwen3-VL-30B-A3B-Thinking', **kwargs) -> tuple[bool, list[str | None]] | \
                                                                                   tuple[bool, None]:

    client = OpenAI(api_key=api_key, base_url=base_url, **get_kwargs(kwargs, OpenAI))

    try:
        print(f"Generating content with model: {model_name}", )

        response = client.chat.completions.create(messages=messages, model=model_name,
                                                  **get_kwargs(kwargs, client.chat.completions.create))

        return True, [answ.message.content for answ in response.choices]

    except Exception as e:
        print("Failed to call LLM: " + str(e))
        if hasattr(e, 'response'):
            error_info = e.response.json()
            code_value = error_info['error']['code']
            print(code_value)
        else:
            code_value = "context_length_exceeded"
            print(code_value)
        return False, None
