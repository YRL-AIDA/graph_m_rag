"""Tests for GET /compute_entity_embeddings endpoint."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientConnectorError, ClientResponseError
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------- read imports ----------
import config
from manager import Manager
from semantic_index import app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _qdrant_create_ok():
    """Qdrant PUT create collection → 200."""
    return _make_async_response(200, {"result": True})


def _embedding_ok():
    """Embedding service returns valid 2048-dim vector."""
    return _make_async_response(200, {"data": [{"embedding": [0.1] * 2048}]})


def _embedding_500():
    """Embedding service returns 500."""
    resp = _make_async_response(500)
    resp.raise_for_status = MagicMock(side_effect=ClientResponseError(
        request_info=MagicMock(), history=(), status=500, message="Internal Server Error"
    ))
    return resp


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
        mgr.__class__ = Manager  # allow bound-method lookup

        result = Manager.get_entities_needing_embedding(mgr)
        assert result == []

    def test_returns_candidates_with_null_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        # Simulate 3 records where embedding_updated_at is None
        records = [_entity_dict(title=f"E{i}", embedding_updated_at=None) for i in range(3)]
        mgr.query = MagicMock(return_value=records)
        mgr.__class__ = Manager

        result = Manager.get_entities_needing_embedding(mgr)
        assert len(result) == 3

    def test_returns_candidates_where_updated_at_gt_embedding_updated_at(self):
        mgr = MagicMock(spec=Manager)
        records = [
            _entity_dict(title="E0", embedding_updated_at="2024-01-01", updated_at="2024-06-01"),
            _entity_dict(title="E1", embedding_updated_at="2024-03-01", updated_at="2024-07-01"),
        ]
        mgr.query = MagicMock(return_value=records)
        mgr.__class__ = Manager

        result = Manager.get_entities_needing_embedding(mgr)
        assert len(result) == 2
        for r in result:
            # Verify updated_at > embedding_updated_at
            assert r["updated_at"] > r["embedding_updated_at"]

    def test_converts_records_to_dicts(self):
        mgr = MagicMock(spec=Manager)
        records = [
            _entity_dict(title="Alice", type_="PERSON", description="Person Alice",
                         embedding_updated_at=None, updated_at="2024-01-01"),
            _entity_dict(title="Bob", type_="PERSON", description="Person Bob",
                         embedding_updated_at="2024-01-01", updated_at="2024-06-01"),
        ]
        mgr.query = MagicMock(return_value=records)
        mgr.__class__ = Manager

        result = Manager.get_entities_needing_embedding(mgr)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, dict)
            assert set(r.keys()) == {"title", "type", "description", "embedding_updated_at", "updated_at"}


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

        # Verify execute_write called with _set_embedding_updated_at_tx and entity list
        mock_session.execute_write.assert_called_once()
        call_args = mock_session.execute_write.call_args
        # First positional arg should be a callable (the tx function)
        assert callable(call_args[0][0])
        # Second positional arg should be the entities list
        assert call_args[0][1] == entities


# ---------------------------------------------------------------------------
# TestComputeEntityEmbeddingsEndpoint — integration tests for the endpoint
# ---------------------------------------------------------------------------

class TestComputeEntityEmbeddingsEndpoint:
    """Integration tests for GET /compute_entity_embeddings."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_mocks(doc_mgr_patch, session_patch,
                     qdrant_get_resp=_qdrant_collection_ok(),
                     qdrant_create_resp=None,
                     embedding_resp=None,
                     qdrant_upsert_resp=None,
                     candidates=None):
        """Configure all mocks for an endpoint call.

        Returns (mock_doc_mgr, mock_session) for further assertions.
        """
        mock_doc_mgr = MagicMock()
        mock_doc_mgr.get_entities_needing_embedding.return_value = candidates or []
        if candidates:
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = len(candidates)
        else:
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0
        doc_mgr_patch.return_value = mock_doc_mgr

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Qdrant collection GET
        mock_session.get = MagicMock(return_value=qdrant_get_resp)

        # Qdrant collection PUT (create)
        if qdrant_create_resp is not None:
            mock_session.put = MagicMock(return_value=qdrant_create_resp)

        # Embedding POST
        if embedding_resp is not None:
            mock_session.post = MagicMock(return_value=embedding_resp)

        # Qdrant upsert PUT
        if qdrant_upsert_resp is not None:
            mock_session.put = MagicMock(return_value=qdrant_upsert_resp)

        session_patch.return_value = mock_session

        return mock_doc_mgr, mock_session

    # ------------------------------------------------------------------
    # test 7: no candidates → all zeros
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
        entities = [_entity_dict(title=f"Entity{i}", embedding_updated_at=None) for i in range(3)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            # Qdrant collection: first GET 404, then PUT create OK
            get_resp = _make_async_response(404)
            create_resp = _make_async_response(200, {"result": True})
            mock_session.get = MagicMock(return_value=get_resp)
            mock_session.post = MagicMock(return_value=_embedding_ok())
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
    # test 9: updated entities computed successfully
    # ------------------------------------------------------------------

    def test_updated_entities_computed_successfully(self):
        entities = [
            _entity_dict(title="E0", embedding_updated_at="2024-01-01", updated_at="2024-06-01"),
            _entity_dict(title="E1", embedding_updated_at="2024-02-01", updated_at="2024-07-01"),
            _entity_dict(title="E2", embedding_updated_at="2024-03-01", updated_at="2024-08-01"),
        ]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.post = MagicMock(return_value=_embedding_ok())
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
        entities = [
            _entity_dict(title="New0", embedding_updated_at=None),
            _entity_dict(title="New1", embedding_updated_at=None),
            _entity_dict(title="Upd0", embedding_updated_at="2024-01-01", updated_at="2024-06-01"),
            _entity_dict(title="Upd1", embedding_updated_at="2024-02-01", updated_at="2024-07-01"),
            _entity_dict(title="Upd2", embedding_updated_at="2024-03-01", updated_at="2024-08-01"),
        ]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 5

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.post = MagicMock(return_value=_embedding_ok())
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

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 3

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            # First 3 calls succeed, last 2 fail
            call_count = 0

            def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    return _embedding_ok()
                else:
                    return _embedding_500()

            mock_session.post = MagicMock(side_effect=post_side_effect)

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_added"] == 3
            assert stats["embeddings_failed"] == 2

    # ------------------------------------------------------------------
    # test 12: invalid embedding response increments failed
    # ------------------------------------------------------------------

    def test_invalid_embedding_response_increments_failed(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            # Embedding service returns empty data array
            invalid_resp = _make_async_response(200, {"data": []})
            mock_session.post = MagicMock(return_value=invalid_resp)

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 13: wrong embedding dimension increments failed
    # ------------------------------------------------------------------

    def test_wrong_embedding_dimension_increments_failed(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            # Returns 1024-dim vector instead of 2048
            wrong_dim_resp = _make_async_response(200, {"data": [{"embedding": [0.1] * 1024}]})
            mock_session.post = MagicMock(return_value=wrong_dim_resp)

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            assert stats["embeddings_failed"] == 1

    # ------------------------------------------------------------------
    # test 14: Qdrant unavailable returns 500
    # ------------------------------------------------------------------

    def test_qdrant_unavailable_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            # GET raises ClientConnectorError on __aenter__
            failing_resp = AsyncMock()
            mock_conn_key = MagicMock()
            mock_conn_key.ssl = None
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
    # test 15: Qdrant dimension mismatch returns 500
    # ------------------------------------------------------------------

    def test_qdrant_dimension_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            # Collection exists but with size=1024
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
    # test 16: Qdrant distance metric mismatch returns 500
    # ------------------------------------------------------------------

    def test_qdrant_distance_metric_mismatch_returns_500(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = []

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            # Collection exists but distance is Euclid
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
    # test 17: embedding service fully unavailable returns 500
    # ------------------------------------------------------------------

    def test_embedding_service_fully_unavailable_returns_500(self):
        entities = [_entity_dict(title="E0", embedding_updated_at=None)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session_cls.return_value = mock_session

            # POST raises ClientConnectorError
            failing_post = AsyncMock()
            mock_conn_key = MagicMock()
            mock_conn_key.ssl = None
            failing_post.__aenter__ = AsyncMock(
                side_effect=ClientConnectorError(mock_conn_key, OSError("Connection refused"))
            )
            failing_post.__aexit__ = AsyncMock()
            mock_session.post = MagicMock(return_value=failing_post)

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "Embedding service unavailable" in detail

    # ------------------------------------------------------------------
    # test 18: Qdrant upsert batch failure
    # ------------------------------------------------------------------

    def test_qdrant_upsert_batch_failure(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None) for i in range(3)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 0

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.post = MagicMock(return_value=_embedding_ok())
            # Qdrant PUT points fails
            mock_session.put = MagicMock(return_value=_qdrant_upsert_500())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            stats = resp.json()["statistics"]
            # All 3 in the upsert batch fail
            assert stats["embeddings_failed"] == 3
            assert stats["embeddings_added"] == 0

    # ------------------------------------------------------------------
    # test 19: respects semaphore concurrency
    # ------------------------------------------------------------------

    def test_respects_semaphore_concurrency(self):
        entities = [_entity_dict(title=f"E{i}", embedding_updated_at=None)
                    for i in range(20)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 20

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            # Track concurrency via __aenter__ on POST response
            state = {"concurrent": 0, "max_concurrent": 0}

            class TrackingPostResponse:
                """Proper async context manager for tracking concurrent calls."""
                def __init__(self):
                    self.status = 200

                async def __aenter__(self):
                    state["concurrent"] += 1
                    if state["concurrent"] > state["max_concurrent"]:
                        state["max_concurrent"] = state["concurrent"]
                    await asyncio.sleep(0)  # yield so other tasks can run
                    return self

                async def __aexit__(self, *args):
                    state["concurrent"] -= 1

                async def json(self):
                    return {"data": [{"embedding": [0.1] * 2048}]}

                def raise_for_status(self):
                    pass

            mock_session.post = MagicMock(side_effect=lambda *a, **kw: TrackingPostResponse())

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200
            # EMBEDDING_MAX_CONCURRENCY from config is 8
            assert state["max_concurrent"] <= config.EMBEDDING_MAX_CONCURRENCY
            assert state["max_concurrent"] > 1  # sanity: actual concurrency happened

    # ------------------------------------------------------------------
    # test 20: point ID format is TITLE|TYPE
    # ------------------------------------------------------------------

    def test_point_id_format_title_pipe_type(self):
        entities = [_entity_dict(title="Acme Corp", type_="ORGANIZATION",
                                 embedding_updated_at=None)]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.post = MagicMock(return_value=_embedding_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200

            # Check the PUT call args
            put_call_args = mock_session.put.call_args
            put_body = put_call_args[1]["json"]
            point_id = put_body["points"][0]["id"]
            assert point_id == "Acme Corp|ORGANIZATION"

    # ------------------------------------------------------------------
    # test 21: payload contains required fields
    # ------------------------------------------------------------------

    def test_payload_contains_required_fields(self):
        entities = [_entity_dict(
            title="Test Entity", type_="GEO",
            description="A test geographic entity",
            embedding_updated_at=None,
        )]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls:
            mock_doc_mgr.get_entities_needing_embedding.return_value = entities
            mock_doc_mgr.set_entity_embedding_updated_at.return_value = 1

            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = MagicMock(return_value=_qdrant_collection_ok())
            mock_session.post = MagicMock(return_value=_embedding_ok())
            mock_session.put = MagicMock(return_value=_qdrant_upsert_ok())
            mock_session_cls.return_value = mock_session

            client = TestClient(app)
            resp = client.get("/compute_entity_embeddings")

            assert resp.status_code == 200

            put_call_args = mock_session.put.call_args
            put_body = put_call_args[1]["json"]
            payload = put_body["points"][0]["payload"]
            assert "entity_title" in payload
            assert payload["entity_title"] == "Test Entity"
            assert "entity_type" in payload
            assert payload["entity_type"] == "GEO"
            assert "entity_id" in payload
            assert payload["entity_id"] == "Test Entity|GEO"
            assert "description" in payload
            assert payload["description"] == "A test geographic entity"
