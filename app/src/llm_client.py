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
        _client_opts = get_kwargs(self.client_kwargs, OpenAI)
        # Bound the request so a terminated/overloaded LLM connection cannot
        # hang a worker thread for the openai SDK's default 10 minutes (which
        # saturated the threadpool and made the whole service unresponsive).
        # 300s ≈ max demo proxy wait; long thinking generations still fit.
        _client_opts.setdefault("timeout", 300.0)
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=1,
            **_client_opts
        )

        try:
            print(f"Generating content with model: {self.model_name}")

            call_kwargs = get_kwargs(kwargs, client.chat.completions.create)
            # A caller may override the model via kwargs (e.g. question_decomposer
            # passes model=model_name). Pull it out of the spread so `model` is
            # not passed twice to create(), which would raise
            # "got multiple values for keyword argument 'model'".
            model = call_kwargs.pop("model", self.model_name)

            response = client.chat.completions.create(
                messages=messages,
                model=model,
                **call_kwargs
            )

            return True, [answ.message.content for answ in response.choices]

        except Exception as e:
            print("Failed to call LLM: " + str(e))
            if hasattr(e, 'response'):
                try:
                    error_info = e.response.json()
                    code_value = error_info['error']['code']
                    print(code_value)
                except Exception:
                    print("error_code_unknown")
            else:
                # No server response body — do NOT guess "context_length_exceeded".
                print("no_error_body")
            return False, None