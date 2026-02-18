from typing import List, Dict, Any, Optional, Union
import base64
from openai import OpenAI

from app.src.utils.llm_utils import ModelMessageDict, send_messasge


class RemoteLLMClient:
    """
    Клиент для взаимодействия с удалённой LLM через OpenAI‑совместимый API.
    Инкапсулирует функции send_messasge, ModelMessageDict и get_kwargs,
    предоставляя удобный интерфейс для отправки текстовых и мультимодальных запросов.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.19.127:10114/embedding/embed",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        **default_kwargs
    ):
        """
        Инициализирует клиент с параметрами подключения.

        :param base_url:  Адрес API, совместимого с OpenAI.
        :param api_key:   Ключ API (по умолчанию "EMPTY" для локальных серверов).
        :param model_name: Имя используемой модели.
        :param default_kwargs: Дополнительные параметры, которые будут переданы
                               во все вызовы OpenAI (например, timeout, max_retries).
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.default_kwargs = default_kwargs

    def _call_send_message(
        self,
        messages: List[ModelMessageDict],
        **kwargs
    ) -> List[str]:
        """
        Внутренний метод для отправки сообщений и обработки ошибок.

        :param messages: Список сообщений в формате ModelMessageDict.
        :param kwargs:   Параметры, переопределяющие значения по умолчанию
                         (передаются в send_messasge и client.chat.completions.create).
        :return:         Список строк — ответов модели.
        :raises RuntimeError: Если запрос не удался.
        """
        # Объединяем параметры: сначала дефолтные, затем явно переданные
        merged_kwargs = {**self.default_kwargs, **kwargs}

        success, responses = send_messasge(
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            **merged_kwargs
        )

        if not success:
            raise RuntimeError(f"Ошибка при обращении к модели: {responses}")
        return responses

    def create_message(self, role: str = "user") -> ModelMessageDict:
        """
        Создаёт новое сообщение-словарь для построения запроса.

        :param role: Роль отправителя (user, system, assistant).
        :return:     Экземпляр ModelMessageDict.
        """
        return ModelMessageDict(role=role)

    def send(self, messages: List[ModelMessageDict], **kwargs) -> List[str]:
        """
        Отправляет подготовленный список сообщений в модель.

        :param messages: Список сообщений.
        :param kwargs:   Дополнительные параметры (например, temperature, max_tokens).
        :return:         Ответы модели.
        """
        return self._call_send_message(messages, **kwargs)

    def send_text(
        self,
        text: str,
        role: str = "user",
        **kwargs
    ) -> List[str]:
        """
        Отправляет одиночное текстовое сообщение.

        :param text:  Текст запроса.
        :param role:  Роль отправителя.
        :param kwargs: Дополнительные параметры.
        :return:      Ответы модели.
        """
        msg = self.create_message(role)
        msg.add_text_content(text)
        return self.send(msg, **kwargs)

    def send_multimodal(
        self,
        text: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        image_urls: Optional[List[str]] = None,
        role: str = "user",
        **kwargs
    ) -> List[str]:
        """
        Отправляет мультимодальный запрос, содержащий текст и/или изображения.

        :param text:         Текст запроса (опционально).
        :param image_paths:  Список путей к локальным файлам изображений.
        :param image_urls:   Список URL изображений.
        :param role:         Роль отправителя.
        :param kwargs:       Дополнительные параметры.
        :return:             Ответы модели.
        """
        msg = self.create_message(role)
        if text:
            msg.add_text_content(text)

        if image_paths:
            for path in image_paths:
                msg.add_img_content(source="image_url", path_to_img=path)

        if image_urls:
            for url in image_urls:
                msg.add_img_content(source="image_url", url=url)

        return self.send([msg], **kwargs)

    # Псевдонимы для удобства
    multimodal = send_multimodal
    text = send_text


# Пример использования
if __name__ == "__main__":
    # Создаём клиент с настройками по умолчанию
    client = RemoteLLMClient()

    # Простой текстовый запрос
    try:
        answers = client.send_text("Привет, как дела?")
        print("Ответ:", answers[0])
    except RuntimeError as e:
        print(e)

    # Мультимодальный запрос: текст + локальное изображение
    try:
        answers = client.multimodal(
            text="Что изображено на картинке?",
            image_paths=["./../data/turismObjectsRewewByLLM.png"]
        )
        print("Описание:", answers[0])
    except RuntimeError as e:
        print(e)

    # Создание сложного диалога вручную
    msg1 = client.create_message("system")
    msg1.add_text_content("Ты — полезный ассистент.")

    msg2 = client.create_message("user")
    msg2.add_text_content("Расскажи о космосе.")

    response = client.send(msg1)
    print("Ответ ассистента:", response[0])


"""
Qwen3 Embedding Client for generating embeddings
"""

from typing import List, Union
import requests
from openai import OpenAI


class Qwen3EmbClient:
    """
    Client for generating embeddings using Qwen3 embedding model.
    Provides methods to generate embeddings for text content.
    """

    def __init__(self, base_url: str = "http://192.168.19.127:10014/embed", api_key: str = "EMPTY"):
        """
        Initialize the embedding client

        Args:
            base_url: Base URL for the embedding service
            api_key: API key for authentication (default is EMPTY for local servers)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text

        Args:
            text: Input text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model="BAAI/bge-m3"  # Using a suitable embedding model
            )

            # Return the embedding vector
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a dummy embedding in case of error
            return []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts to generate embeddings for

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings