"""
MinerU Client for PDF processing
"""

import requests
import base64
import json
from typing import Dict, Any, Optional
from pathlib import Path


class MinerUClient:
    """
    Client for interacting with the MinerU service.
    Handles PDF processing and content extraction.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the MinerU client

        Args:
            base_url: Base URL for the MinerU service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _read_and_encode_file(self, file_path: str) -> str:
        """
        Read a file and encode it as base64

        Args:
            file_path: Path to the file to read

        Returns:
            Base64 encoded string of the file content
        """
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return base64.b64encode(file_content).decode('utf-8')

    def analyze_pdf(
        self,
        file_path: str,
        include_original: bool = True,
        include_previews: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a PDF file using MinerU service

        Args:
            file_path: Path to the PDF file
            include_original: Whether to include original content in the response
            include_previews: Whether to include previews in the response

        Returns:
            Dictionary containing the analysis result
        """
        file_path = str(Path(file_path).resolve())

        # Prepare the request payload
        payload = {
            "file_path": file_path,
            "include_original": include_original,
            "include_previews": include_previews
        }

        # Make the request to the MinerU service
        try:
            response = self.session.post(
                f"{self.base_url}/process",
                json=payload,
                timeout=300  # 5 minute timeout for potentially large PDFs
            )

            if response.status_code != 200:
                raise Exception(f"MinerU service returned status {response.status_code}: {response.text}")

            return response.json()

        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to MinerU service")
        except requests.exceptions.Timeout:
            raise Exception("Request to MinerU service timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to MinerU service: {str(e)}")

    def analyze_pdf_content(
        self,
        file_content: bytes,
        include_original: bool = True,
        include_previews: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze PDF content directly using MinerU service

        Args:
            file_content: Raw PDF file content as bytes
            include_original: Whether to include original content in the response
            include_previews: Whether to include previews in the response

        Returns:
            Dictionary containing the analysis result
        """
        # Encode the file content as base64
        base64_content = base64.b64encode(file_content).decode('utf-8')

        # Prepare the request payload
        payload = {
            "file_content": base64_content,
            "include_original": include_original,
            "include_previews": include_previews
        }

        # Make the request to the MinerU service
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=300  # 5 minute timeout for potentially large PDFs
            )

            if response.status_code != 200:
                raise Exception(f"MinerU service returned status {response.status_code}: {response.text}")

            return response.json()

        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to MinerU service")
        except requests.exceptions.Timeout:
            raise Exception("Request to MinerU service timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to MinerU service: {str(e)}")

    def health_check(self) -> bool:
        """
        Check if the MinerU service is healthy

        Returns:
            True if the service is accessible, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the MinerU service

        Returns:
            Dictionary containing service information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/info",
                timeout=10
            )

            if response.status_code != 200:
                raise Exception(f"MinerU service returned status {response.status_code}: {response.text}")

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting service info: {str(e)}")