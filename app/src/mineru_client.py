"""
MinerU Client for PDF processing
"""

import requests
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

    def process_document(
        self,
        file_path: str,
        backend: str = "pipeline",
        method: str = "auto",
        lang: str = "ru",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a document using MinerU service

        Args:
            file_path: Path to the document file (PDF, PNG, JPG, JPEG, TIFF, BMP)
            backend: Processing backend - 'pipeline' or 'vlm'
            method: Processing method - 'auto', 'txt', 'ocr'
            lang: Document language
            formula_enable: Enable formula processing
            table_enable: Enable table processing
            start_page: Starting page (0-indexed)
            end_page: Ending page (None for all pages after start)

        Returns:
            Dictionary containing the processing result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Prepare the request with file upload
        with open(file_path, 'rb') as f:
            files = {"file": f}

            params = {
                "backend": backend,
                "method": method,
                "lang": lang,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
                "start_page": start_page
            }

            if end_page is not None:
                params["end_page"] = end_page

            # Make the request to the MinerU service
            try:
                response = requests.post(
                    f"{self.base_url}/process",
                    files=files,
                    params=params
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

    def process_document_content(
        self,
        file_content: bytes,
        filename: str,
        backend: str = "vlm",
        method: str = "auto",
        lang: str = "ru",
        formula_enable: bool = True,
        table_enable: bool = True,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process document content directly using MinerU service

        Args:
            file_content: Raw document file content as bytes
            filename: Name of the file (with extension)
            backend: Processing backend - 'pipeline' or 'vlm'
            method: Processing method - 'auto', 'txt', 'ocr'
            lang: Document language
            formula_enable: Enable formula processing
            table_enable: Enable table processing
            start_page: Starting page (0-indexed)
            end_page: Ending page (None for all pages after start)

        Returns:
            Dictionary containing the processing result
        """
        # Prepare the request with file upload
        files = {"file": (filename, file_content)}

        params = {
            "backend": backend,
            "method": method,
            "lang": lang,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "start_page": start_page
        }

        if end_page is not None:
            params["end_page"] = end_page

        # Make the request to the MinerU service
        try:
            response = self.session.post(
                f"{self.base_url}/process",
                files=files,
                params=params,
                timeout=300  # 5 minute timeout for potentially large documents
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

    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the processing status for a specific task

        Args:
            task_id: ID of the processing task

        Returns:
            Dictionary containing the task status and results if completed
        """
        try:
            response = self.session.get(
                f"{self.base_url}/status/{task_id}",
                timeout=30
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

    def download_file(self, task_id: str, file_path: str, save_path: str) -> None:
        """
        Download a processed file

        Args:
            task_id: ID of the processing task
            file_path: Path to the file relative to the output directory
            save_path: Local path to save the downloaded file
        """
        try:
            response = self.session.get(
                f"{self.base_url}/download/{task_id}/{file_path}",
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"MinerU service returned status {response.status_code}: {response.text}")

            # Save the downloaded content
            with open(save_path, 'wb') as f:
                f.write(response.content)

        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to MinerU service")
        except requests.exceptions.Timeout:
            raise Exception("Request to MinerU service timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to MinerU service: {str(e)}")

    def cleanup_task(self, task_id: str) -> Dict[str, Any]:
        """
        Clean up resources associated with a task

        Args:
            task_id: ID of the processing task

        Returns:
            Dictionary containing the cleanup status
        """
        try:
            response = self.session.delete(
                f"{self.base_url}/cleanup/{task_id}",
                timeout=30
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
                f"{self.base_url}",
                timeout=10
            )

            if response.status_code != 200:
                raise Exception(f"MinerU service returned status {response.status_code}: {response.text}")

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting service info: {str(e)}")