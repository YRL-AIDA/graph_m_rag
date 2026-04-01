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
                if path_to_img is not None:
                    with open(path_to_img, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode()
                    self['content'].append({'type': 'image_url',
                                            'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}})
                elif url is not None:
                    self['content'].append({'type': 'image_url',
                                            'image_url': {'url': url}})

    def add_img_content_base64(self, base64_image: str = None):
        self['content'].append({'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}})


class LLMClient:
    '''
        Класс-клиент для отправки сообщений в удаленную LLM модель.
    '''

    def __init__(self, base_url: str = "http://192.168.19.127:8888/v1",
                 api_key: str = 'EMPTY',
                 model_name: str = 'Qwen/Qwen3-VL-32B-Thinking',
                 **kwargs):
        '''
            Инициализация клиента.

            :param base_url: URL базовый для API
            :param api_key: API ключ
            :param model_name: Название модели
            :param kwargs: Дополнительные параметры для OpenAI клиента
        '''
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.client_kwargs = kwargs

    def send_message(self, messages: List[ModelMessageDict], **kwargs) -> tuple[bool, List[str] | None]:
        '''
            Отправляет сообщения в удаленную модель.

            :param messages: Список сообщений формата ModelMessageDict
            :param kwargs: Дополнительные параметры для client.chat.completions.create
            :return: Кортеж (успех: bool, результат: List[str] или None)
        '''
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=1,
            **get_kwargs(self.client_kwargs, OpenAI)
        )

        try:
            print(f"Generating content with model: {self.model_name}")

            response = client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                **get_kwargs(kwargs, client.chat.completions.create)
            )

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