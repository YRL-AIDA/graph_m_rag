"""Tests for GET /compute_entity_embeddings endpoint."""
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

# ---------- read imports ----------
import config
from manager import Manager
from semantic_index import app, emb_client


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

VALID_EMBEDDING = [0.1] * 2048
WRONG_DIM_EMBEDDING = [0.1] * 1024


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


def _entity_dict(title="Entity1", type_="PERSON", description="Some description",
                 embedding_updated_at=None, updated_at="2024-01-01T00:00:00"):
    return {
        "title": title,
        "type": type_,
        "description": description,
        "embedding_updated_at": embedding_updated_at,
        "updated_at": updated_at,
    }


# ---------------------------------------------------------------------------
# TestGetEntitiesNeedingEmbedding — unit tests for Manager method
# ---------------------------------------------------------------------------

class TestGetEntitiesNeedingEmbedding:
    """Unit tests for Manager.get_entities_needing_embedding()."""

    def test_returns_empty_list_when_no_candidates(self):
        mgr = MagicMock(spec=Manager)
        mgr.query = MagicMock(return_value=[])
        result = Manager.get_entities_needing_embedding(mgr)
        assert result == []

    def test_returns_candidates_with_null_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        rec1 = MagicMock()
        rec1.__getitem__ = MagicMock(side_effect=lambda k: {
            "title": "A", "type": "ORG", "description": "Desc A",
            "embedding_updated_at": None, "updated_at": "2024-01-01"
        }[k])
        rec2 = MagicMock()
        rec2.__getitem__ = MagicMock(side_effect=lambda k: {
            "title": "B", "type": "PERSON", "description": "Desc B",
            "embedding_updated_at": None, "updated_at": "2024-01-02"
        }[k])
        rec3 = MagicMock()
        rec3.__getitem__ = MagicMock(side_effect=lambda k: {
            "title": "C", "type": "GEO", "description": "Desc C",
            "embedding_updated_at": None, "updated_at": "2024-01-03"
        }[k])
        mgr.query = MagicMock(return_value=[rec1, rec2, rec3])
        result = Manager.get_entities_needing_embedding(mgr)
        assert len(result) == 3

    def test_returns_candidates_where_updated_at_gt_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        rec1 = MagicMock()
        rec1.__getitem__ = MagicMock(side_effect=lambda k: {
            "title": "D", "type": "ORG", "description": "Desc D",
            "embedding_updated_at": "2024-01-01", "updated_at": "2024-02-01"
        }[k])
        rec2 = MagicMock()
        rec2.__getitem__ = MagicMock(side_effect=lambda k: {
            "title": "E", "type": "PERSON", "description": "Desc E",
            "embedding_updated_at": "2024-01-01", "updated_at": "2024-03-01"
        }[k])
        mgr.query = MagicMock(return_value=[rec1, rec2])
        result = Manager.get_entities_needing_embedding(mgr)
        assert len(result) == 2

    def test_converts_records_to_dicts(self):
        mgr = MagicMock(spec=Manager)
        rec = MagicMock()
        fields = {"title": "Test", "type": "ORG", "description": "Desc",
                  "embedding_updated_at": None, "updated_at": "2024-01-01"}
        rec.keys = MagicMock(return_value=list(fields.keys()))
        rec.__getitem__ = MagicMock(side_effect=lambda k: fields[k])
        mgr.query = MagicMock(return_value=[rec])
        result = Manager.get_entities_needing_embedding(mgr)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        for key in ("title", "type", "description", "embedding_updated_at", "updated_at"):
            assert key in result[0]


# ---------------------------------------------------------------------------
# TestSetEntityEmbeddingUpdatedAt — unit tests for Manager method
# ---------------------------------------------------------------------------

class TestSetEntityEmbeddingUpdatedAt:
    """Unit tests for Manager.set_entity_embedding_updated_at()."""

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

        entities = [{"title": "E1", "type": "PERSON"}, {"title": "E2", "type": "ORG"}]
        result = Manager.set_entity_embedding_updated_at(mgr, entities)
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

        entities = [{"title": "Alice", "type": "PERSON"}, {"title": "Bob", "type": "ORG"}]
        Manager.set_entity_embedding_updated_at(mgr, entities)

        mock_session.execute_write.assert_called_once()
        call_args = mock_session.execute_write.call_args
        assert callable(call_args[0][0])
        assert call_args[0][1] == entities


# ---------------------------------------------------------------------------
# TestComputeEntityEmbeddingsEndpoint — integration tests for the endpoint
# ---------------------------------------------------------------------------

class TestComputeEntityEmbeddingsEndpoint:

    # ------------------------------------------------------------------
    # test 7: no candidates → 200 with all zeros
    # ------------------------------------------------------------------

    def test_no_candidates_returns_200_with_all_zeros(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 0
            assert stats["embeddings_updated"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 8: new entities computed successfully
    # ------------------------------------------------------------------

    def test_new_entities_computed_successfully(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None) for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_updated"] == 0
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 9: updated entities
    # ------------------------------------------------------------------

    def test_updated_entities_computed_successfully(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at="2024-01-01",
                                 updated_at="2024-02-01") for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 0
            assert stats["embeddings_updated"] == 3
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 10: mixed new and updated
    # ------------------------------------------------------------------

    def test_mixed_new_and_updated(self):
        new = [_entity_dict(title=f"N{i}", embedding_updated_at=None) for i in range(2)]
        updated = [_entity_dict(title=f"U{i}", embedding_updated_at="2024-01-01",
                                updated_at="2024-02-01") for i in range(3)]
        entities = new + updated

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 5

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 2
            assert stats["embeddings_updated"] == 3
            assert stats["embeddings_failed"] == 0

    # ------------------------------------------------------------------
    # test 11: embedding service errors increment failed
    # ------------------------------------------------------------------

    def test_embedding_service_errors_increment_failed(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None) for i in range(5)]

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
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_failed"] == 2

    # ------------------------------------------------------------------
    # test 12: invalid embedding response (ValueError) → failed
    # ------------------------------------------------------------------

    def test_invalid_embedding_response_increments_failed(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=ValueError("Empty data array")), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 13: wrong embedding dimension → failed
    # ------------------------------------------------------------------

    def test_wrong_embedding_dimension_increments_failed(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          return_value=WRONG_DIM_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 14: Qdrant unavailable → 500
    # ------------------------------------------------------------------

    def test_qdrant_unavailable_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

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
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "Qdrant unavailable" in detail

    # ------------------------------------------------------------------
    # test 15: Qdrant dimension mismatch → 500
    # ------------------------------------------------------------------

    def test_qdrant_dimension_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mismatch_resp = _make_async_response(200, {
                "result": {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}
            })
            mock_session.get = MagicMock(return_value=mismatch_resp)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "dimension mismatch" in detail

    # ------------------------------------------------------------------
    # test 16: Qdrant distance metric mismatch → 500
    # ------------------------------------------------------------------

    def test_qdrant_distance_metric_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mismatch_resp = _make_async_response(200, {
                "result": {"config": {"params": {"vectors": {"size": 2048, "distance": "Euclid"}}}}
            })
            mock_session.get = MagicMock(return_value=mismatch_resp)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "distance metric mismatch" in detail

    # ------------------------------------------------------------------
    # test 17: embedding service fully unavailable → 200 with all failed
    # (EmbeddingClient wraps connection errors as Exception, treated per-entity)
    # ------------------------------------------------------------------

    def test_embedding_service_fully_unavailable_returns_200_all_failed(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch.object(emb_client, 'get_text_embedding',
                          side_effect=requests.ConnectionError("Connection refused")), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 18: Qdrant upsert batch failure
    # ------------------------------------------------------------------

    def test_qdrant_upsert_batch_failure(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None) for i in range(3)]

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_500())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 3
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 19: respects semaphore concurrency
    # ------------------------------------------------------------------

    def test_respects_semaphore_concurrency(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None)
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
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 20

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            assert state["max_concurrent"] <= config.EMBEDDING_MAX_CONCURRENCY
            assert state["max_concurrent"] > 1  # sanity: actual concurrency happened

    # ------------------------------------------------------------------
    # test 20: point ID format is TITLE|TYPE
    # ------------------------------------------------------------------

    def test_point_id_format_is_deterministic_uuid5(self):
        entities = [_entity_dict(title="Acme Corp", type_="ORGANIZATION",
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
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            assert len(captured_points) == 1
            assert captured_points[0]["id"] == str(uuid_mod.uuid5(uuid_mod.UUID("a7f1b2c3-4d5e-6f78-9abc-def012345678"), "Acme Corp|ORGANIZATION"))

    # ------------------------------------------------------------------
    # test 21: payload contains required fields
    # ------------------------------------------------------------------

    def test_payload_contains_required_fields(self):
        entities = [_entity_dict(title="TestEntity", type_="GEO",
                                 description="A test location",
                                 embedding_updated_at=None)]

        captured_points = []

        def capture_put(url, **kwargs):
            if "json" in kwargs:
                captured_points.extend(kwargs["json"].get("points", []))
            return _qdrant_upsert_ok()

        with patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING), \
                patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(side_effect=capture_put)
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            assert len(captured_points) == 1
            payload = captured_points[0]["payload"]
            assert payload["entity_title"] == "TestEntity"
            assert payload["entity_type"] == "GEO"
            assert payload["entity_id"] == "TestEntity|GEO"
            assert payload["description"] == "A test location"
            assert len(captured_points[0]["vector"]) == 2048
            expected_point_id = str(uuid_mod.uuid5(uuid_mod.UUID("a7f1b2c3-4d5e-6f78-9abc-def012345678"), "TestEntity|GEO"))
            assert captured_points[0]["id"] == expected_point_id
