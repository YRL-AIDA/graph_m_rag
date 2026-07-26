"""Tests for GET /compute_community_embeddings endpoint."""
import asyncio
import sys
import threading
import time
from pathlib import Path
import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import requests
import pytest
from aiohttp import ClientConnectorError, ClientResponseError
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from manager import Manager
from semantic_index import app, emb_client


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

VALID_EMBEDDING = [0.1] * 2048
WRONG_DIM_EMBEDDING = [0.1] * 1024

COMMUNITY_EMBEDDINGS_NAMESPACE = uuid_mod.UUID("b8e2c3d4-5e6f-7a89-bcde-f01234567890")


def _make_async_response(status=200, json_data=None, text_data="error"):
    """Create a mock aiohttp response suitable for `async with ... as resp:`."""
    resp = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data)
    resp.raise_for_status = MagicMock()
    return resp


def _qdrant_collection_ok():
    """Qdrant GET collection → 2048-dim Cosine exists."""
    return _make_async_response(200, {
        "result": {"config": {"params": {"vectors": {"size": 2048, "distance": "Cosine"}}}}
    })


def _qdrant_collection_404():
    """Qdrant collection does not exist (404)."""
    return _make_async_response(404)


def _qdrant_upsert_ok():
    """Qdrant PUT points → 200."""
    return _make_async_response(200, {"result": {"operation_id": 1, "status": "completed"}})


def _qdrant_upsert_500():
    """Qdrant PUT points → 500."""
    resp = _make_async_response(500, text_data="Internal Server Error")
    resp.raise_for_status = MagicMock(side_effect=ClientResponseError(
        request_info=MagicMock(), history=(), status=500, message="Internal Server Error"
    ))
    return resp


def _community_dict(id_="a1b2c3d4-e5f6-7890-abcd-ef1234567890", title="Community Title", level=0,
                    summary="A community summary", embedding_updated_at=None,
                    updated_at="2024-01-01T00:00:00"):
    return {
        "id": id_,
        "title": title,
        "level": level,
        "summary": summary,
        "embedding_updated_at": embedding_updated_at,
        "updated_at": updated_at,
    }


# ---------------------------------------------------------------------------
# TestGetCommunitiesNeedingEmbedding — unit tests for Manager method
# ---------------------------------------------------------------------------

class TestGetCommunitiesNeedingEmbedding:
    """Unit tests for Manager.get_communities_needing_embedding()."""

    def test_returns_empty_list_when_no_candidates(self):
        mgr = MagicMock(spec=Manager)
        mgr.query = MagicMock(return_value=[])
        result = Manager.get_communities_needing_embedding(mgr)
        assert result == []

    def test_returns_candidates_with_null_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        rec1 = MagicMock()
        rec1.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "title": "Community A", "level": 0, "summary": "Summary A",
            "embedding_updated_at": None, "updated_at": "2024-01-01"
        }[k])
        rec2 = MagicMock()
        rec2.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "a2b3c4d5-e6f7-8901-abcd-ef2345678901",
            "title": "Community B", "level": 1, "summary": "Summary B",
            "embedding_updated_at": None, "updated_at": "2024-01-02"
        }[k])
        rec3 = MagicMock()
        rec3.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "a3b4c5d6-e7f8-9012-abcd-ef3456789012",
            "title": "Community C", "level": 2, "summary": "Summary C",
            "embedding_updated_at": None, "updated_at": "2024-01-03"
        }[k])
        mgr.query = MagicMock(return_value=[rec1, rec2, rec3])
        result = Manager.get_communities_needing_embedding(mgr)
        assert len(result) == 3

    def test_returns_candidates_with_null_updated_at(self):
        mgr = MagicMock(spec=Manager)
        rec = MagicMock()
        rec.keys = MagicMock(return_value=["id", "title", "level", "summary", "embedding_updated_at", "updated_at"])
        rec.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "title": "Legacy Community", "level": 0, "summary": "Legacy Summary",
            "embedding_updated_at": "2024-01-01", "updated_at": None
        }[k])
        mgr.query = MagicMock(return_value=[rec])
        result = Manager.get_communities_needing_embedding(mgr)
        assert len(result) == 1
        assert result[0]["updated_at"] is None

    def test_returns_candidates_where_updated_at_gt_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        rec1 = MagicMock()
        rec1.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "d1e2f3a4-b5c6-7890-abcd-ef1234567890",
            "title": "Community D", "level": 1, "summary": "Summary D",
            "embedding_updated_at": "2024-01-01", "updated_at": "2024-02-01"
        }[k])
        rec2 = MagicMock()
        rec2.__getitem__ = MagicMock(side_effect=lambda k: {
            "id": "d2e3f4a5-b6c7-8901-abcd-ef2345678901",
            "title": "Community E", "level": 2, "summary": "Summary E",
            "embedding_updated_at": "2024-01-01", "updated_at": "2024-03-01"
        }[k])
        mgr.query = MagicMock(return_value=[rec1, rec2])
        result = Manager.get_communities_needing_embedding(mgr)
        assert len(result) == 2

    def test_converts_records_to_dicts(self):
        mgr = MagicMock(spec=Manager)
        rec = MagicMock()
        fields = {"id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
                  "title": "Test", "level": 0, "summary": "Desc",
                  "embedding_updated_at": None, "updated_at": "2024-01-01"}
        rec.keys = MagicMock(return_value=list(fields.keys()))
        rec.__getitem__ = MagicMock(side_effect=lambda k: fields[k])
        mgr.query = MagicMock(return_value=[rec])
        result = Manager.get_communities_needing_embedding(mgr)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        for key in ("id", "title", "level", "summary", "embedding_updated_at", "updated_at"):
            assert key in result[0]


# ---------------------------------------------------------------------------
# TestSetCommunityEmbeddingUpdatedAt — unit tests for Manager method
# ---------------------------------------------------------------------------

class TestSetCommunityEmbeddingUpdatedAt:
    """Unit tests for Manager.set_community_embedding_updated_at()."""

    def test_returns_updated_count(self):
        mgr = MagicMock(spec=Manager)
        mgr.conn = MagicMock()
        mgr.name_db = "testdb"
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.execute_write = MagicMock(return_value=5)
        mgr.conn.graph.session.return_value = mock_session
        mgr.__class__ = Manager

        communities = [{"id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
                       {"id": "a2b3c4d5-e6f7-8901-abcd-ef2345678901"}]
        result = Manager.set_community_embedding_updated_at(mgr, communities)
        assert result == 5

    def test_calls_execute_write_with_correct_params(self):
        mgr = MagicMock(spec=Manager)
        mgr.conn = MagicMock()
        mgr.name_db = "testdb"
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.execute_write = MagicMock(return_value=3)
        mgr.conn.graph.session.return_value = mock_session
        mgr.__class__ = Manager

        communities = [{"id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
                       {"id": "a2b3c4d5-e6f7-8901-abcd-ef2345678901"}]
        Manager.set_community_embedding_updated_at(mgr, communities)

        mock_session.execute_write.assert_called_once()
        call_args = mock_session.execute_write.call_args
        assert callable(call_args[0][0])
        assert call_args[0][1] == communities


# ---------------------------------------------------------------------------
# TestComputeCommunityEmbeddingsEndpoint — integration tests for the endpoint
# ---------------------------------------------------------------------------

class TestComputeCommunityEmbeddingsEndpoint:

    # ------------------------------------------------------------------
    # test 8: no candidates → 200 with all zeros
    # ------------------------------------------------------------------

    def test_no_candidates_returns_200_with_all_zeros(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 0
            assert stats["embeddings_updated"] == 0
            assert stats["embeddings_skipped"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 9: new communities computed successfully
    # ------------------------------------------------------------------

    def test_new_communities_computed_successfully(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef0{i}",
                                        embedding_updated_at=None) for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_updated"] == 0
            assert stats["embeddings_skipped"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 10: updated communities
    # ------------------------------------------------------------------

    def test_updated_communities_computed_successfully(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef1{i}",
                                        embedding_updated_at="2024-01-01",
                                        updated_at="2024-02-01") for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 0
            assert stats["embeddings_updated"] == 3
            assert stats["embeddings_skipped"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 11: mixed new and updated
    # ------------------------------------------------------------------

    def test_mixed_new_and_updated(self):
        new = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdefa{i}",
                               embedding_updated_at=None) for i in range(2)]
        updated = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdedb{i}",
                                   embedding_updated_at="2024-01-01",
                                   updated_at="2024-02-01") for i in range(3)]
        communities = new + updated

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 5

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 2
            assert stats["embeddings_updated"] == 3
            assert stats["embeddings_skipped"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 12: communities without summary not included
    # ------------------------------------------------------------------

    def test_communities_without_summary_not_included(self):
        """Only communities with summary are returned by the manager query
        (Cypher filters c.summary IS NOT NULL). Communities without summary
        are excluded at the query level, so the mock only returns valid ones."""
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef0{i}",
                                        embedding_updated_at=None) for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 13: embedding service errors increment failed
    # ------------------------------------------------------------------

    def test_embedding_service_errors_increment_failed(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef0{i}",
                                        embedding_updated_at=None) for i in range(5)]

        call_count = [0]

        def get_text_embedding_side_effect(text):
            call_count[0] += 1
            if call_count[0] <= 3:
                return VALID_EMBEDDING
            else:
                raise requests.HTTPError("500 Server Error")

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=get_text_embedding_side_effect), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_failed"] == 2

    # ------------------------------------------------------------------
    # test 14: invalid embedding response (ValueError) → failed
    # ------------------------------------------------------------------

    def test_invalid_embedding_response_increments_failed(self):
        communities = [_community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef00",
                                       embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=ValueError("Empty data array")), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 15: wrong embedding dimension → failed
    # ------------------------------------------------------------------

    def test_wrong_embedding_dimension_increments_failed(self):
        communities = [_community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef00",
                                       embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          return_value=WRONG_DIM_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 16: Qdrant unavailable → 500
    # ------------------------------------------------------------------

    def test_qdrant_unavailable_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_conn_key = MagicMock()
            mock_conn_key.ssl = None
            failing_resp = AsyncMock()
            failing_resp.__aenter__ = AsyncMock(
                side_effect=ClientConnectorError(mock_conn_key, OSError("Connection refused"))
            )
            failing_resp.__aexit__ = AsyncMock()
            mock_session.get = MagicMock(return_value=failing_resp)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "Qdrant unavailable" in detail

    # ------------------------------------------------------------------
    # test 17: Qdrant dimension mismatch → 500
    # ------------------------------------------------------------------

    def test_qdrant_dimension_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mismatch_resp = _make_async_response(200, {
                "result": {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}
            })
            mock_session.get = MagicMock(return_value=mismatch_resp)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "dimension mismatch" in detail

    # ------------------------------------------------------------------
    # test 18: Qdrant distance metric mismatch → 500
    # ------------------------------------------------------------------

    def test_qdrant_distance_metric_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mismatch_resp = _make_async_response(200, {
                "result": {"config": {"params": {"vectors": {"size": 2048, "distance": "Euclid"}}}}
            })
            mock_session.get = MagicMock(return_value=mismatch_resp)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "distance metric mismatch" in detail

    # ------------------------------------------------------------------
    # test 19: Qdrant collection created if missing
    # ------------------------------------------------------------------

    def test_qdrant_collection_created_if_missing(self):
        communities = [_community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef00",
                                       embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            # Collection GET returns 404 (not found), then PUT creates it
            mock_session.get = MagicMock(return_value=_qdrant_collection_404())
            # PUT is called twice: once for collection creation, once for point upsert
            mock_session.put = MagicMock(side_effect=[
                _make_async_response(200, {"result": True}),   # create collection
                _qdrant_upsert_ok(),                           # upsert points
            ])
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 1
            assert stats["embeddings_failed"] == 0

            # Verify collection GET was called (returned 404)
            mock_session.get.assert_called_once()
            # Verify two PUT calls: create collection + upsert points
            assert mock_session.put.call_count == 2

    # ------------------------------------------------------------------
    # test 20: embedding service fully unavailable → 200 with all failed
    # (EmbeddingClient wraps connection errors as Exception, treated per-community)
    # ------------------------------------------------------------------

    def test_embedding_service_fully_unavailable_returns_200_all_failed(self):
        communities = [_community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef00",
                                       embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=requests.ConnectionError("Connection refused")), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 21: Qdrant upsert batch failure
    # ------------------------------------------------------------------

    def test_qdrant_upsert_batch_failure(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef0{i}",
                                        embedding_updated_at=None) for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_500())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 3
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 22: respects semaphore concurrency
    # ------------------------------------------------------------------

    def test_respects_semaphore_concurrency(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef{i:02d}",
                                        embedding_updated_at=None)
                       for i in range(20)]

        # Thread-safe concurrency tracker
        lock = threading.Lock()
        state = {"concurrent": 0, "max_concurrent": 0}

        def tracking_get_text_embedding(text):
            with lock:
                state["concurrent"] += 1
                if state["concurrent"] > state["max_concurrent"]:
                    state["max_concurrent"] = state["concurrent"]
            # small sleep so other tasks can enter to_thread
            time.sleep(0.05)
            with lock:
                state["concurrent"] -= 1
            return VALID_EMBEDDING

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=tracking_get_text_embedding), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 20

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            assert state["max_concurrent"] <= config.EMBEDDING_MAX_CONCURRENCY
            assert state["max_concurrent"] > 1  # sanity: actual concurrency happened

    # ------------------------------------------------------------------
    # test 23: batching multiple Qdrant calls
    # ------------------------------------------------------------------

    def test_batching_multiple_qdrant_calls(self):
        communities = [_community_dict(id_=f"e5f67890-abcd-ef12-3456-7890abcdef{i:03d}",
                                        embedding_updated_at=None)
                       for i in range(250)]

        put_calls = []

        def capture_put(url, **kwargs):
            put_calls.append(kwargs)
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 250

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 250
            # With BATCH_SIZE=100 and 250 communities, expect 3 batch calls
            assert len(put_calls) == 3

    # ------------------------------------------------------------------
    # test 24: point ID is deterministic uuid5
    # ------------------------------------------------------------------

    def test_point_id_is_deterministic_uuid5(self):
        community_id = "e5f67890-abcd-ef12-3456-7890abcdef01"
        communities = [_community_dict(id_=community_id, title="Test Community",
                                       embedding_updated_at=None)]

        captured_points = []

        def capture_put(url, **kwargs):
            """Capture Qdrant upsert payload."""
            if "json" in kwargs:
                captured_points.extend(kwargs["json"].get("points", []))
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            assert len(captured_points) == 1
            assert captured_points[0]["id"] == str(uuid_mod.uuid5(
                COMMUNITY_EMBEDDINGS_NAMESPACE, community_id))

    # ------------------------------------------------------------------
    # test 25: payload contains required fields
    # ------------------------------------------------------------------

    def test_payload_contains_required_fields(self):
        community_id = "e5f67890-abcd-ef12-3456-7890abcdef02"
        communities = [_community_dict(id_=community_id, title="My Community", level=1,
                                       summary="A comprehensive community summary",
                                       embedding_updated_at=None)]

        captured_points = []

        def capture_put(url, **kwargs):
            if "json" in kwargs:
                captured_points.extend(kwargs["json"].get("points", []))
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            assert len(captured_points) == 1
            payload = captured_points[0]["payload"]
            assert payload["community_id"] == community_id
            assert payload["title"] == "My Community"
            assert payload["level"] == 1
            assert payload["summary"] == "A comprehensive community summary"
            assert len(captured_points[0]["vector"]) == 2048
            expected_point_id = str(uuid_mod.uuid5(
                COMMUNITY_EMBEDDINGS_NAMESPACE, community_id))
            assert captured_points[0]["id"] == expected_point_id

    # ------------------------------------------------------------------
    # test 26: idempotent second call
    # ------------------------------------------------------------------

    def test_idempotent_second_call(self):
        """After first successful call sets embedding_updated_at, second call
        should find no candidates and return all zeros."""
        community = _community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef00",
                                    embedding_updated_at=None)

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            # First call returns a community needing embedding
            # Second call returns empty (already updated)
            mock_doc_mgr.get_communities_needing_embedding.side_effect = [
                [community],
                [],
            ]
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)

            # First call — should process one community
            resp1 = client.get("/compute_community_embeddings")
            assert resp1.status_code == 200
            stats1 = resp1.json()["statistics"]
            assert stats1["embeddings_added"] == 1
            assert stats1["embeddings_failed"] == 0

            # Second call — no candidates, all zeros
            resp2 = client.get("/compute_community_embeddings")
            assert resp2.status_code == 200
            stats2 = resp2.json()["statistics"]
            assert stats2["embeddings_added"] == 0
            assert stats2["embeddings_updated"] == 0
            assert stats2["embeddings_skipped"] == 0
            assert stats2["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 27: level preserved in payload
    # ------------------------------------------------------------------

    def test_level_preserved_in_payload(self):
        communities = [_community_dict(id_="e5f67890-abcd-ef12-3456-7890abcdef03",
                                       title="Level 2 Community", level=2,
                                       summary="A level 2 summary",
                                       embedding_updated_at=None)]

        captured_points = []

        def capture_put(url, **kwargs):
            if "json" in kwargs:
                captured_points.extend(kwargs["json"].get("points", []))
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_communities_needing_embedding.return_value = communities
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_community_embeddings")

            assert resp.status_code == 200
            assert len(captured_points) == 1
            assert captured_points[0]["payload"]["level"] == 2

    # ------------------------------------------------------------------
    # test 28: two same community IDs produce same point ID
    # ------------------------------------------------------------------

    def test_two_same_community_ids_produce_same_point_id(self):
        """Two calls with the same community id should produce identical
        Qdrant point IDs (uuid5 is deterministic)."""
        community_id = "e5f67890-abcd-ef12-3456-7890abcdef04"
        community1 = _community_dict(id_=community_id, title="Same Community",
                                     embedding_updated_at=None)

        # After first call, the community has embedding_updated_at set,
        # but for the second call we simulate re-processing (e.g., updated_at changed)
        community2 = _community_dict(id_=community_id, title="Same Community",
                                     embedding_updated_at="2024-01-01",
                                     updated_at="2024-06-01")

        captured_point_ids = []

        def capture_put(url, **kwargs):
            if "json" in kwargs:
                for pt in kwargs["json"].get("points", []):
                    captured_point_ids.append(pt["id"])
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            # First call: new community
            # Second call: updated community (same id)
            mock_doc_mgr.get_communities_needing_embedding.side_effect = [
                [community1],
                [community2],
            ]
            mock_doc_mgr.set_community_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp1 = client.get("/compute_community_embeddings")
            resp2 = client.get("/compute_community_embeddings")

            assert resp1.status_code == 200
            assert resp2.status_code == 200
            assert len(captured_point_ids) == 2
            assert captured_point_ids[0] == captured_point_ids[1]
            expected_id = str(uuid_mod.uuid5(COMMUNITY_EMBEDDINGS_NAMESPACE, community_id))
            assert captured_point_ids[0] == expected_id
