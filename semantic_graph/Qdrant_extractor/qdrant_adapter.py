import requests
from typing import Dict, Iterator, Optional


class QdrantStreamAdapter:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def list_collections(self):
        url = f"{self.base_url}/collections"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        return [c["name"] for c in resp.json()["result"]["collections"]]

    def iter_points(self, collection_name: str, filter_payload: Optional[Dict] = None) -> Iterator[Dict]:
        url = f"{self.base_url}/collections/{collection_name}/points/scroll"
        offset = None

        while True:
            payload = {
                "limit": 100,
                "with_payload": True,
                "with_vector": False,
            }
            if filter_payload:
                payload["filter"] = filter_payload
            if offset is not None:  # 🔥 исправлено: offset может быть 0 или ID
                payload["offset"] = offset

            resp = requests.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            result = resp.json()["result"]

            points = result.get("points", [])
            if not points:
                break

            for p in points:
                yield p

            offset = result.get("next_page_offset")
            if offset is None:
                break

    def iter_all_points(self, filter_payload: Optional[Dict] = None):
        for col in self.list_collections():
            for p in self.iter_points(col, filter_payload=filter_payload):
                yield col, p
