import requests
from typing import List, Union, Optional

from fastapi import Query
from pydantic import BaseModel, ConfigDict

from app.src.schemas.reranker import Message, RerankResponse, RerankRequest


# ========== Client ==========
class RerankerClient:
    """
    A simple client for the reranker service.
    """

    def __init__(self, base_url: str, timeout: int = 300):
        """
        Args:
            base_url: The base URL of the reranker service (e.g. http://localhost:8000).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def rerank(
        self,
        query_text: str,
        messages: List[Message],
        instruction: str = "Retrieve images or text relevant to the user's query."
    ) -> RerankResponse:
        """
        Send a reranking request to the service.

        Args:
            instruction: Instruction for the reranking task.
            query_text: The query text to rank messages against.
            messages: A list of Message objects to rerank.

        Returns:
            RerankResponse containing the ranked results with scores.

        Raises:
            requests.RequestException: If the HTTP request fails.
            ValueError: If the response cannot be parsed into RerankResponse.
        """
        # Build the request payload using the RerankRequest model

        request_payload = RerankRequest(
            instruction=instruction,
            query={'text': query_text},
            messages=messages
        ).model_dump()

        # Send POST request
        url = f"{self.base_url}/rerank"
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()

        # Parse and validate the response
        try:
            return RerankResponse.model_validate(response.json())
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e

    def rerank_from_dict(
        self,
        instruction: str,
        query_text: str,
        messages: List[dict]
    ) -> RerankResponse:
        """
        Send a reranking request to the service using dictionary messages.

        Args:
            instruction: Instruction for the reranking task.
            query_text: The query text to rank messages against.
            messages: A list of message dictionaries to rerank.

        Returns:
            RerankResponse containing the ranked results with scores.

        Raises:
            requests.RequestException: If the HTTP request fails.
            ValueError: If the response cannot be parsed into RerankResponse.
        """
        # Convert dict messages to Message objects
        message_objects = [Message(**msg) for msg in messages]
        return self.rerank(instruction, query_text, message_objects)


# ========== Example usage ==========
if __name__ == "__main__":
    # Create some messages
    messages = [
        Message(type="text", text="Hello, world!"),
        Message(type="text", text="Goodbye, world!"),
        Message(type="text", text="How are you?")
    ]

    # Initialize the client (point to your actual service URL)
    client = RerankerClient(base_url="http://192.168.19.127:10115/reranker")

    try:
        result = client.rerank(
            instruction="Retrieve images or text relevant to the user's query.",
            query_text="A woman playing with her dog on a beach at sunset.",
            messages=messages
        )
        print("Reranking results:")
        for res in result.messages:
            print(f"  Message ID: {res.message_id}, Score: {res.score}")
    except Exception as e:
        print(f"Error: {e}")