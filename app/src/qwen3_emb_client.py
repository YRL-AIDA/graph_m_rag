import requests
from typing import List, Union, Optional
from pydantic import BaseModel, ConfigDict

from app.src.utils.emb_utils import EmbedResponse, Message, EmbedRequest


# ========== Client ==========
class EmbeddingClient:
    """
    A simple client for the embedding service.
    """

    def __init__(self, base_url: str, timeout: int = 300):
        """
        Args:
            base_url: The base URL of the embedding service (e.g. http://localhost:8000).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout


    def get_text_embedding(self, text: str) -> EmbedResponse:

        message = Message(type="text", text=text)
        request_payload = EmbedRequest(messages=[message]).model_dump()  # or .dict() for older Pydantic
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e

    def get_image_embedding(self, image_path: str) -> EmbedResponse:
        """
        Get embedding for an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            EmbedResponse containing the message_id and embedding vector.
        """
        # Read the image file and convert to base64
        import base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        message = Message(type="image", image=image_path, image_data=encoded_string)
        request_payload = EmbedRequest(messages=[message]).model_dump()
        url = f"{self.base_url}/embed"
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()
        try:
            return EmbedResponse.model_validate(response.json())
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e

    def get_image_embedding_base64(self, image_base64: str) -> EmbedResponse:
        image_base64 = f"'data:image/jpeg;base64,',{image_base64}"
        message = Message(type="image", image=image_base64)
        request_payload = EmbedRequest(messages=[message]).model_dump()  # or .dict() for older Pydantic
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e

    def get_image_embedding_url(self, image_url: str) -> EmbedResponse:

        message = Message(type="image", image_url=image_url)
        request_payload = EmbedRequest(messages=[message]).model_dump()  # or .dict() for older Pydantic
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e

    def get_embeddings(self, messages: List[Message]) -> EmbedResponse:
        """
        Send a list of messages to the service and retrieve the embedding.

        Args:
            messages: A list of Message objects.

        Returns:
            EmbedResponse containing the message_id and embedding vector.

        Raises:
            requests.RequestException: If the HTTP request fails.
            ValueError: If the response cannot be parsed into EmbedResponse.
        """
        # Build the request payload using the EmbedRequest model
        request_payload = EmbedRequest(messages=messages).model_dump()  # or .dict() for older Pydantic

        # Send POST request
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors

        # Parse and validate the response
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e


# ========== Example usage ==========
if __name__ == "__main__":
    # Create some messages
    messages = [
        Message(type="text", text="Hello, world!")
    ]

    # Initialize the client (point to your actual service URL)
    client = EmbeddingClient(base_url="http://192.168.19.127:10114/embedding")

    try:
        result = client.get_embeddings(messages)
        print(f"Message ID: {result.message_id}")
        print(f"Embedding (first 5 values): {result.embedding[:5]}...")
    except Exception as e:
        print(f"Error: {e}")