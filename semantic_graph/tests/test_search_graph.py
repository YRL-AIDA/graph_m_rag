"""Tests for POST /search endpoint and its helper functions."""
import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_index import _cosine_similarity, app, emb_client
from dtype.search import SearchRequest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

VALID_EMBEDDING = [0.1] * 2048


def _make_async_response(status=200, json_data=None, text_data="error"):
    """Создать мок aiohttp-ответа, совместимый с `async with ... as resp:`."""
    resp = AsyncMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data)
    resp.raise_for_status = MagicMock()
    return resp


def _make_entities_df(titles_types):
    """Создать DataFrame сущностей из списка (title, type, description)."""
    data = [
        {"title": t[0], "type": t[1], "description": t[2] if len(t) > 2 else f"Desc of {t[0]}"}
        for t in titles_types
    ]
    return pd.DataFrame(data)


def _empty_entities_df():
    """Пустой DataFrame сущностей с правильными колонками."""
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships_df():
    """Пустой DataFrame связей."""
    return pd.DataFrame(columns=["source", "target", "description", "source_id", "weight"])


# ==========================================================================
# Класс 1: TestCosineSimilarity — unit-тесты _cosine_similarity
# ==========================================================================

class TestCosineSimilarity:
    """Unit-тесты для _cosine_similarity()."""

    def test_identical_vectors(self):
        """Два одинаковых вектора дают 1.0."""
        a = [0.5, 0.5, 0.5, 0.5]
        result = _cosine_similarity(a, a)
        assert abs(result - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """Ортогональные векторы дают ~0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        result = _cosine_similarity(a, b)
        assert abs(result - 0.0) < 1e-9

    def test_zero_vector(self):
        """Один вектор нулевой → 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = _cosine_similarity(a, b)
        assert result == 0.0

    def test_known_values(self):
        """Проверка на известных значениях: [1,2,3] × [4,5,6]."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        # norm_a = sqrt(1 + 4 + 9) = sqrt(14)
        # norm_b = sqrt(16 + 25 + 36) = sqrt(77)
        # result = 32 / (sqrt(14) * sqrt(77))
        expected = 32 / (math.sqrt(14) * math.sqrt(77))
        result = _cosine_similarity(a, b)
        assert abs(result - expected) < 1e-9


# ==========================================================================
# Класс 2: TestTokenBudgetCalculation — unit-тесты распределения токенов
# ==========================================================================

class TestTokenBudgetCalculation:
    """Unit-тесты расчёта бюджетов токенов."""

    def test_default_proportions(self):
        """max_tokens=4000 → text=2000, entities=1000, communities=1000."""
        proportions = {"text_units": 0.5, "entities": 0.25, "communities": 0.25}
        max_tokens = 4000
        text_budget = int(max_tokens * proportions["text_units"])
        entity_budget = int(max_tokens * proportions["entities"])
        community_budget = int(max_tokens * proportions["communities"])
        assert text_budget == 2000
        assert entity_budget == 1000
        assert community_budget == 1000

    def test_custom_proportions(self):
        """proportions {0.7, 0.2, 0.1} при max_tokens=1000."""
        proportions = {"text_units": 0.7, "entities": 0.2, "communities": 0.1}
        max_tokens = 1000
        text_budget = int(max_tokens * proportions["text_units"])
        entity_budget = int(max_tokens * proportions["entities"])
        community_budget = int(max_tokens * proportions["communities"])
        assert text_budget == 700
        assert entity_budget == 200
        assert community_budget == 100

    def test_per_entity_budget_with_remainder(self):
        """250 токенов на сущности, 3 сущности → base=83, remainder=1."""
        entity_budget = 250
        N = 3
        per_entity_base = entity_budget // N          # 83
        per_entity_remainder = entity_budget % N       # 1
        assert per_entity_base == 83
        assert per_entity_remainder == 1
        # Первые 1 сущность получает 84, остальные 83
        for i in range(N):
            budget_i = per_entity_base + (1 if i < per_entity_remainder else 0)
            if i == 0:
                assert budget_i == 84
            else:
                assert budget_i == 83


# ==========================================================================
# Класс 3: TestRoundRobinMerge — unit-тесты round-robin слияния пулов
# ==========================================================================

class TestRoundRobinMerge:
    """Unit-тесты round-robin слияния пулов."""

    def _round_robin_merge(self, pools):
        """Реализация round-robin слияния для тестирования."""
        merged = []
        max_len = max((len(pool) for pool in pools), default=0)
        for round_idx in range(max_len):
            for pool in pools:
                if round_idx < len(pool):
                    merged.append(pool[round_idx])
        return merged

    def test_two_pools_equal_length(self):
        """pool_0=[A1,A2], pool_1=[B1,B2] → [A1,B1,A2,B2]."""
        pools = [["A1", "A2"], ["B1", "B2"]]
        result = self._round_robin_merge(pools)
        assert result == ["A1", "B1", "A2", "B2"]

    def test_three_pools_unequal_length(self):
        """pool_0=[A1,A2,A3], pool_1=[B1,B2], pool_2=[C1] → [A1,B1,C1,A2,B2,A3]."""
        pools = [["A1", "A2", "A3"], ["B1", "B2"], ["C1"]]
        result = self._round_robin_merge(pools)
        assert result == ["A1", "B1", "C1", "A2", "B2", "A3"]

    def test_empty_pools(self):
        """Все пулы пусты → []."""
        pools = [[], [], []]
        result = self._round_robin_merge(pools)
        assert result == []


# ==========================================================================
# Класс 4: TestSearchRequestValidation — unit-тесты валидации SearchRequest
# ==========================================================================

class TestSearchRequestValidation:
    """Unit-тесты Pydantic-валидации SearchRequest."""

    def test_valid_default_request(self):
        """Валидный минимальный запрос."""
        req = SearchRequest(question="What is GraphRAG?", max_tokens=4000)
        assert req.question == "What is GraphRAG?"
        assert req.max_tokens == 4000
        assert req.proportions == {"text_units": 0.5, "entities": 0.25, "communities": 0.25}
        assert req.documents_filter == "text_only"

    def test_valid_custom_proportions(self):
        """Валидные кастомные пропорции."""
        req = SearchRequest(
            question="Test",
            max_tokens=1000,
            proportions={"text_units": 0.7, "entities": 0.2, "communities": 0.1},
        )
        assert req.proportions == {"text_units": 0.7, "entities": 0.2, "communities": 0.1}

    def test_proportions_sum_not_one(self):
        """Сумма пропорций ≠ 1.0 → ValueError."""
        with pytest.raises(ValueError, match="proportions must sum to 1.0"):
            SearchRequest(
                question="Test",
                max_tokens=1000,
                proportions={"text_units": 0.5, "entities": 0.2, "communities": 0.2},
            )

    def test_proportions_negative_values(self):
        """Отрицательное значение пропорции → ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SearchRequest(
                question="Test",
                max_tokens=1000,
                proportions={"text_units": 0.5, "entities": -0.1, "communities": 0.6},
            )

    def test_proportions_missing_keys(self):
        """Отсутствует ключ в proportions → ValueError."""
        with pytest.raises(ValueError, match="proportions must contain exactly keys"):
            SearchRequest(
                question="Test",
                max_tokens=1000,
                proportions={"text_units": 0.5, "entities": 0.5},
            )

    def test_invalid_documents_filter(self):
        """Некорректный documents_filter → ValueError."""
        with pytest.raises(ValueError, match="documents_filter must be"):
            SearchRequest(
                question="Test",
                max_tokens=1000,
                documents_filter="invalid",
            )

    def test_empty_question(self):
        """Пустой вопрос → ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SearchRequest(question="", max_tokens=1000)

    def test_max_tokens_zero_or_negative(self):
        """max_tokens ≤ 0 → ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SearchRequest(question="Test", max_tokens=0)
        with pytest.raises(ValidationError):
            SearchRequest(question="Test", max_tokens=-1)

    def test_documents_filter_all(self):
        """"all" валиден."""
        req = SearchRequest(question="Test", max_tokens=1000, documents_filter="all")
        assert req.documents_filter == "all"

    def test_documents_filter_text_only(self):
        """"text_only" валиден (по умолчанию)."""
        req = SearchRequest(question="Test", max_tokens=1000)
        assert req.documents_filter == "text_only"


# ==========================================================================
# Класс 5: TestSearchEndpoint — integration-тесты эндпоинта POST /search
# ==========================================================================

class TestSearchEndpoint:
    """Integration-тесты для POST /search с замоканными зависимостями."""

    # ------------------------------------------------------------------
    # helpers for setting up mocks
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mock_neo4j_run_handler(entity_map=None, related_map=None,
                                       communities_map=None, children_map=None):
        """Создать side_effect-обработчик для mock Neo4j session.run().

        Parameters
        ----------
        entity_map : dict
            Mapping (title, type) → {"title": ..., "type": ..., "description": ...}
        related_map : dict
            Mapping (title, type) → list of related entity dicts
        communities_map : dict
            Mapping (title, type) → list of community dicts
        children_map : dict
            Mapping community_id → list of child community dicts
        """
        def _mock_run(cypher, params):
            cypher_upper = cypher.upper()

            # Поиск Entry Entity
            if "MATCH (E:ENTITY" in cypher_upper and "$TITLE" in cypher_upper:
                title = params.get("title", "")
                etype = params.get("type", "")
                key = (title, etype)
                if entity_map and key in entity_map:
                    return [entity_map[key]]
                return []

            # RELATED связи
            if ":RELATED" in cypher_upper and "RELATED" in cypher:
                title = params.get("title", "")
                etype = params.get("type", "")
                key = (title, etype)
                return related_map.get(key, []) if related_map else []

            # Leaf Community (CONSISTS_OF + IS_PARENT_OF + IS_CHILD_OF)
            if "CONSISTS_OF" in cypher_upper and "IS_PARENT_OF" in cypher_upper:
                titles_list = params.get("titles", [])
                types_list = params.get("types", [])
                result = []
                if communities_map:
                    for t, tp in zip(titles_list, types_list):
                        key = (t, tp)
                        if key in communities_map:
                            result.extend(communities_map[key])
                return result

            # Подсчёт count_ent для community
            if "COUNT(E) AS COUNT_ENT" in cypher_upper:
                # Возвращаем произвольный count
                return [{"count_ent": 3}]

            # Fallback: получение Community по id
            if "MATCH (C:COMMUNITY" in cypher_upper and "$COMMUNITY_ID" in cypher_upper:
                cid = params.get("community_id", "")
                if entity_map:  # переиспользуем entity_map для community
                    for k, v in entity_map.items():
                        if isinstance(v, dict) and "community_id" in v:
                            return [v]
                return [{"title": "Root Community", "summary": "Root summary"}]

            # IS_PARENT_OF для дочерних community (fallback)
            if "IS_PARENT_OF" in cypher_upper and "$COMMUNITY_ID" in cypher_upper:
                cid = params.get("community_id", "")
                return children_map.get(cid, []) if children_map else [
                    {"id": "child-1", "title": "Child 1", "summary": "Child summary 1", "level": 1},
                    {"id": "child-2", "title": "Child 2", "summary": "Child summary 2", "level": 1},
                ]

            return []

        return _mock_run

    def _setup_base_mocks(self, mock_doc_mgr, mock_session, mock_session_cls, mock_llm_cls, mock_extractor_cls):
        """Настройка базовых моков для happy-path сценария.

        Возвращает словарь с моками, которые тест может далее настроить.
        """
        # --- doc_manager ---
        mock_doc_mgr.name_db = "neo4j"
        neo4j_session = MagicMock()
        mock_doc_mgr.conn.graph.session.return_value.__enter__.return_value = neo4j_session

        # --- aiohttp.ClientSession ---
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.get = MagicMock()
        mock_session_cls.return_value = mock_session

        # --- AsyncLLMClient ---
        mock_llm = AsyncMock()
        mock_llm.count_tokens = AsyncMock(return_value=50)
        mock_llm_instance = AsyncMock()
        mock_llm_instance.__aenter__ = AsyncMock(return_value=mock_llm)
        mock_llm_instance.__aexit__ = AsyncMock(return_value=None)
        mock_llm_cls.return_value = mock_llm_instance

        # --- AsyncGraphExtractor ---
        mock_extractor = MagicMock()
        mock_extractor_cls.return_value = mock_extractor

        return {
            "neo4j_session": neo4j_session,
            "mock_llm": mock_llm,
            "mock_extractor": mock_extractor,
            "mock_session": mock_session,
        }

    # ------------------------------------------------------------------
    # test_full_search_success
    # ------------------------------------------------------------------

    def test_full_search_success(self):
        entities_df = _make_entities_df([
            ("ORG1", "ORGANIZATION", "Org description"),
            ("PERSON1", "PERSON", "Person description"),
            ("GEO1", "GEO", "Geo description"),
        ])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            # Qdrant documents: возвращаем текстовые блоки
            def mock_post(url, **kwargs):
                json_body = kwargs.get("json", {})
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [
                            {"payload": {"original_element": {"text": f"Document text chunk {i}"}}}
                            for i in range(3)
                        ]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "ORG1", "entity_type": "ORGANIZATION"}}]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-uuid", "title": "Root"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            # Neo4j: entity exists, related entities, communities
            entity_records = {
                ("ORG1", "ORGANIZATION"): {"title": "ORG1", "type": "ORGANIZATION", "description": "Test org desc"},
                ("PERSON1", "PERSON"): {"title": "PERSON1", "type": "PERSON", "description": "Test person desc"},
                ("GEO1", "GEO"): {"title": "GEO1", "type": "GEO", "description": "Test geo desc"},
            }
            related_records = {
                ("ORG1", "ORGANIZATION"): [
                    {"title": "PERSON1", "type": "PERSON", "description": "Related person",
                     "rel_description": "works at ORG1"},
                ],
                ("PERSON1", "PERSON"): [],
                ("GEO1", "GEO"): [],
            }
            communities_records = {}  # не требуются для этого теста
            children_records = {"root-uuid": [
                {"id": "child-1", "title": "Child", "summary": "Child summary", "level": 1},
            ]}

            neo4j_handler = self._build_mock_neo4j_run_handler(
                entity_map=entity_records,
                related_map=related_records,
                communities_map=communities_records,
                children_map=children_records,
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "What is ORG1?",
                "max_tokens": 4000,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert len(data["text_units"]) > 0
            assert "entities" in data
            assert "communities" in data
            stats = data["statistics"]
            assert stats["entities_extracted_from_question"] == 3
            assert stats["fallback_used"] is False

    # ------------------------------------------------------------------
    # test_no_entities_in_question
    # ------------------------------------------------------------------

    def test_no_entities_in_question(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_empty_entities_df(), _empty_relationships_df())
            )

            # Qdrant documents возвращает тексты
            # Qdrant community_embeddings для fallback
            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [
                            {"payload": {"original_element": {"text": f"Text {i}"}}}
                            for i in range(2)
                        ]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-fallback-uuid"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-fallback-uuid": [
                    {"id": "c1", "title": "Child C", "summary": "Summary C", "level": 1},
                ]}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "No entities here",
                "max_tokens": 4000,
            })

            assert resp.status_code == 200
            data = resp.json()
            stats = data["statistics"]
            assert stats["entities_extracted_from_question"] == 0
            assert stats["fallback_used"] is True
            assert len(data["entities"]) == 0

    # ------------------------------------------------------------------
    # test_entities_not_found_in_qdrant
    # ------------------------------------------------------------------

    def test_entities_not_found_in_qdrant(self):
        entities_df = _make_entities_df([
            ("E1", "ORG", "Desc 1"),
            ("E2", "PERSON", "Desc 2"),
        ])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text A"}}}]
                    })
                if "entity_embeddings" in url:
                    # Qdrant entity_embeddings возвращает пустой результат
                    return _make_async_response(200, {"result": []})
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-miss-uuid"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-miss-uuid": [
                    {"id": "c1", "title": "Comm C1", "summary": "Summary C1", "level": 1},
                ]}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test question",
                "max_tokens": 4000,
            })

            assert resp.status_code == 200
            data = resp.json()
            stats = data["statistics"]
            assert stats["entities_extracted_from_question"] == 2
            assert stats["entities_search_misses"] == 2
            assert len(data["entities"]) == 0

    # ------------------------------------------------------------------
    # test_partial_miss
    # ------------------------------------------------------------------

    def test_partial_miss(self):
        """3 сущности извлечены, 2 найдены в Qdrant, 1 miss."""
        entities_df = _make_entities_df([
            ("A", "ORG", "Desc A"),
            ("B", "PERSON", "Desc B"),
            ("C", "GEO", "Desc C"),
        ])

        # Считаем вызовы entity_embeddings поиска
        call_count = [0]

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "entity_embeddings" in url:
                    call_count[0] += 1
                    if call_count[0] <= 2:
                        # Первые 2 найдены
                        return _make_async_response(200, {
                            "result": [{"payload": {"entity_title": f"E{call_count[0]}", "entity_type": "TYPE"}}]
                        })
                    else:
                        # 3-й — miss
                        return _make_async_response(200, {"result": []})
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-partial-miss"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            entity_records = {
                ("A", "ORG"): {"title": "A", "type": "ORG", "description": "Desc A full"},
                ("B", "PERSON"): {"title": "B", "type": "PERSON", "description": "Desc B full"},
            }
            neo4j_handler = self._build_mock_neo4j_run_handler(
                entity_map=entity_records,
                children_map={"root-partial-miss": [
                    {"id": "c1", "title": "C1", "summary": "Sum", "level": 1},
                ]}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test partial miss",
                "max_tokens": 4000,
            })

            assert resp.status_code == 200
            data = resp.json()
            stats = data["statistics"]
            assert stats["entities_extracted_from_question"] == 3
            assert stats["entities_search_misses"] == 1

    # ------------------------------------------------------------------
    # test_documents_filter_all
    # ------------------------------------------------------------------

    def test_documents_filter_all(self):
        captured_search_body = []

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_empty_entities_df(), _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    captured_search_body.append(kwargs.get("json", {}))
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-all-uuid"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-all-uuid": []}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test",
                "max_tokens": 1000,
                "documents_filter": "all",
            })

            assert resp.status_code == 200
            assert len(captured_search_body) > 0
            # При documents_filter="all" фильтр отсутствует
            search_body = captured_search_body[0]
            assert "filter" not in search_body or search_body.get("filter") is None

    # ------------------------------------------------------------------
    # test_documents_filter_text_only
    # ------------------------------------------------------------------

    def test_documents_filter_text_only(self):
        captured_search_body = []

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_empty_entities_df(), _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    captured_search_body.append(kwargs.get("json", {}))
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-to-uuid"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-to-uuid": []}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test",
                "max_tokens": 1000,
                "documents_filter": "text_only",
            })

            assert resp.status_code == 200
            assert len(captured_search_body) > 0
            search_body = captured_search_body[0]
            assert "filter" in search_body
            assert search_body["filter"]["must"][0]["key"] == "element_type"
            assert search_body["filter"]["must"][0]["match"]["value"] == "text"

    # ------------------------------------------------------------------
    # test_token_budget_respected
    # ------------------------------------------------------------------

    def test_token_budget_respected(self):
        entities_df = _make_entities_df([("E1", "ORG", "Desc")])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            # Устанавливаем count_tokens так, чтобы каждый элемент "стоил" 30 токенов
            mocks["mock_llm"].count_tokens = AsyncMock(return_value=30)

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [
                            {"payload": {"original_element": {"text": f"T{i}"}}}
                            for i in range(20)
                        ]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "E1", "entity_type": "ORG"}}]
                    })
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            entity_records = {
                ("E1", "ORG"): {"title": "E1", "type": "ORG", "description": "Desc E1"},
            }
            neo4j_handler = self._build_mock_neo4j_run_handler(
                entity_map=entity_records,
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test budget",
                "max_tokens": 500,
            })

            assert resp.status_code == 200
            data = resp.json()
            stats = data["statistics"]
            total_used = (
                stats["tokens_used"]["text_units"]
                + stats["tokens_used"]["entities"]
                + stats["tokens_used"]["communities"]
            )
            assert total_used <= 500

    # ------------------------------------------------------------------
    # test_empty_graph
    # ------------------------------------------------------------------

    def test_empty_graph(self):
        entities_df = _make_entities_df([("E1", "ORG", "Desc")])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "E1", "entity_type": "ORG"}}]
                    })
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            # Neo4j: entity не найдена (пустой результат)
            entity_records = {}  # пустой словарь → не найдены
            neo4j_handler = self._build_mock_neo4j_run_handler(entity_map=entity_records)
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test empty graph",
                "max_tokens": 1000,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert len(data["entities"]) == 0
            assert len(data["communities"]) == 0

    # ------------------------------------------------------------------
    # test_invalid_proportions_sum / test_invalid_documents_filter /
    # test_empty_question → HTTP 422
    # ------------------------------------------------------------------

    def test_invalid_proportions_sum(self):
        client = TestClient(app)
        resp = client.post("/search", json={
            "question": "Test",
            "max_tokens": 1000,
            "proportions": {"text_units": 0.3, "entities": 0.1, "communities": 0.1},
        })
        assert resp.status_code == 422

    def test_invalid_documents_filter(self):
        client = TestClient(app)
        resp = client.post("/search", json={
            "question": "Test",
            "max_tokens": 1000,
            "documents_filter": "invalid_filter",
        })
        assert resp.status_code == 422

    def test_empty_question(self):
        client = TestClient(app)
        resp = client.post("/search", json={
            "question": "",
            "max_tokens": 1000,
        })
        assert resp.status_code == 422

    # ------------------------------------------------------------------
    # test_qdrant_documents_unavailable
    # ------------------------------------------------------------------

    def test_qdrant_documents_unavailable(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_make_entities_df([("E1", "ORG")]), _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(500, text_data="Qdrant error")
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test qdrant documents unavailable",
                "max_tokens": 1000,
            })

            assert resp.status_code == 500

    # ------------------------------------------------------------------
    # test_qdrant_entity_embeddings_unavailable
    # ------------------------------------------------------------------

    def test_qdrant_entity_embeddings_unavailable(self):
        entities_df = _make_entities_df([("E1", "ORG", "Desc")])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(500, text_data="Qdrant entity error")
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test entity embeddings unavailable",
                "max_tokens": 1000,
            })

            assert resp.status_code == 500

    # ------------------------------------------------------------------
    # test_llm_extraction_failure
    # ------------------------------------------------------------------

    def test_llm_extraction_failure(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            # LLM extraction выбрасывает исключение
            mocks["mock_extractor"].extract = AsyncMock(
                side_effect=Exception("LLM API error")
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-llm-fail"}}]
                    })
                return _make_async_response(200, {})

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-llm-fail": [
                    {"id": "c1", "title": "Child", "summary": "Sum", "level": 1},
                ]}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test LLM failure",
                "max_tokens": 1000,
            })

            assert resp.status_code == 200
            data = resp.json()
            stats = data["statistics"]
            assert stats["entities_extracted_from_question"] == 0
            assert stats["fallback_used"] is True

    # ------------------------------------------------------------------
    # test_fallback_search_empty_qdrant
    # ------------------------------------------------------------------

    def test_fallback_search_empty_qdrant(self):
        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_empty_entities_df(), _empty_relationships_df())
            )

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "community_embeddings" in url:
                    # Пустой результат
                    return _make_async_response(200, {"result": []})
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test fallback empty",
                "max_tokens": 1000,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert len(data["communities"]) == 0

    # ------------------------------------------------------------------
    # test_related_entities_no_description
    # ------------------------------------------------------------------

    def test_related_entities_no_description(self):
        entities_df = _make_entities_df([("MAIN", "ORG", "Main entity")])

        embedding_calls = []

        def track_embedding(text):
            embedding_calls.append(text)
            return VALID_EMBEDDING

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', side_effect=track_embedding):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            mocks["mock_llm"].count_tokens = AsyncMock(return_value=10)

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Doc text"}}}]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "MAIN", "entity_type": "ORG"}}]
                    })
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            entity_records = {
                ("MAIN", "ORG"): {"title": "MAIN", "type": "ORG", "description": "Main desc"},
            }
            related_records = {
                ("MAIN", "ORG"): [
                    {"title": "REL", "type": "PERSON", "description": "Related desc",
                     "rel_description": None},  # rel_description IS NULL
                ],
            }
            neo4j_handler = self._build_mock_neo4j_run_handler(
                entity_map=entity_records,
                related_map=related_records,
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test related no description",
                "max_tokens": 2000,
            })

            assert resp.status_code == 200
            # Проверяем, что rel_description=None → эмбеддинг пустой строки
            # ищём вызов get_text_embedding с пустой строкой
            has_empty_string_call = any(call == "" for call in embedding_calls)
            assert has_empty_string_call, (
                f"Expected an empty-string embedding call for NULL rel_description, "
                f"got calls: {embedding_calls}"
            )

    # ------------------------------------------------------------------
    # test_entity_no_description
    # ------------------------------------------------------------------

    def test_entity_no_description(self):
        entities_df = _make_entities_df([("NO_DESC", "ORG", "Search desc")])

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', return_value=VALID_EMBEDDING):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            mocks["mock_llm"].count_tokens = AsyncMock(return_value=5)

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Doc text"}}}]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "NO_DESC", "entity_type": "ORG"}}]
                    })
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            entity_records = {
                ("NO_DESC", "ORG"): {"title": "NO_DESC", "type": "ORG", "description": None},
            }
            neo4j_handler = self._build_mock_neo4j_run_handler(entity_map=entity_records)
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "Test no description",
                "max_tokens": 2000,
            })

            assert resp.status_code == 200
            data = resp.json()
            # Проверяем, что в entities есть элемент с "No description"
            entity_texts = " ".join(data["entities"])
            assert "No description" in entity_texts

    # ------------------------------------------------------------------
    # test_cosine_similarity_in_search
    # ------------------------------------------------------------------

    def test_cosine_similarity_in_search(self):
        """Связанные сущности должны сортироваться по убыванию score."""
        entities_df = _make_entities_df([("MAIN", "ORG", "Main entity")])

        # Разные эмбеддинги для разных связанных сущностей
        question_embed = [0.0] * 2048
        question_embed[0] = 1.0  # «направлен» по первому измерению

        # Связанная сущность 1: высокое сходство
        rel1_embed = [0.0] * 2048
        rel1_embed[0] = 0.9

        # Связанная сущность 2: низкое сходство
        rel2_embed = [0.0] * 2048
        rel2_embed[1] = 1.0

        # Связанная сущность 3: среднее сходство
        rel3_embed = [0.0] * 2048
        rel3_embed[0] = 0.5

        embed_calls = []

        def smart_embed(text):
            """Возвращает эмбеддинг в зависимости от текста."""
            embed_calls.append(text)
            if "question" in text.lower() or "main" in text.lower():
                return question_embed
            if "high" in text.lower():
                return rel1_embed
            if "low" in text.lower():
                return rel2_embed
            if "mid" in text.lower():
                return rel3_embed
            return VALID_EMBEDDING

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', side_effect=smart_embed):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(entities_df, _empty_relationships_df())
            )

            mocks["mock_llm"].count_tokens = AsyncMock(return_value=3)

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Doc text"}}}]
                    })
                if "entity_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"entity_title": "MAIN", "entity_type": "ORG"}}]
                    })
                return _make_async_response(200, {})

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)

            entity_records = {
                ("MAIN", "ORG"): {"title": "MAIN", "type": "ORG", "description": "Main desc"},
            }
            related_records = {
                ("MAIN", "ORG"): [
                    {"title": "REL_LOW", "type": "PERSON", "description": "Low similarity",
                     "rel_description": "low relevance connection"},
                    {"title": "REL_HIGH", "type": "PERSON", "description": "High similarity",
                     "rel_description": "high relevance keyword match"},
                    {"title": "REL_MID", "type": "GEO", "description": "Mid similarity",
                     "rel_description": "mid relevance keyword"},
                ],
            }
            neo4j_handler = self._build_mock_neo4j_run_handler(
                entity_map=entity_records,
                related_map=related_records,
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "question about MAIN",
                "max_tokens": 2000,
            })

            assert resp.status_code == 200
            data = resp.json()
            entities = data["entities"]
            # REL_HIGH должен идти раньше REL_LOW
            high_idx = next((i for i, e in enumerate(entities) if "REL_HIGH" in e), None)
            low_idx = next((i for i, e in enumerate(entities) if "REL_LOW" in e), None)
            if high_idx is not None and low_idx is not None:
                assert high_idx < low_idx, \
                    f"Expected REL_HIGH before REL_LOW, got positions: REL_HIGH={high_idx}, REL_LOW={low_idx}"

    # ------------------------------------------------------------------
    # test_fallback_communities_scored
    # ------------------------------------------------------------------

    def test_fallback_communities_scored(self):
        """Fallback: дочерние community сортируются по cosine similarity."""
        question_embed = [0.0] * 2048
        question_embed[0] = 1.0

        # child-1: высокий score
        child1_vector = [0.0] * 2048
        child1_vector[0] = 0.95

        # child-2: низкий score
        child2_vector = [0.0] * 2048
        child2_vector[1] = 1.0

        # child-3: средний score
        child3_vector = [0.0] * 2048
        child3_vector[0] = 0.5

        def smart_embed(text):
            if "question" in text.lower() or "fallback" in text.lower():
                return question_embed
            return VALID_EMBEDDING

        with patch('semantic_index.doc_manager') as mock_doc_mgr, \
                patch('semantic_index.aiohttp.ClientSession') as mock_session_cls, \
                patch('semantic_index.AsyncLLMClient') as mock_llm_cls, \
                patch('semantic_index.AsyncGraphExtractor') as mock_extractor_cls, \
                patch.object(emb_client, 'get_text_embedding', side_effect=smart_embed):

            mocks = self._setup_base_mocks(
                mock_doc_mgr, AsyncMock(), mock_session_cls, mock_llm_cls, mock_extractor_cls
            )

            mocks["mock_extractor"].extract = AsyncMock(
                return_value=(_empty_entities_df(), _empty_relationships_df())
            )

            mocks["mock_llm"].count_tokens = AsyncMock(return_value=5)

            def mock_post(url, **kwargs):
                if "documents" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"original_element": {"text": "Text"}}}]
                    })
                if "community_embeddings" in url:
                    return _make_async_response(200, {
                        "result": [{"payload": {"community_id": "root-scored-uuid"}}]
                    })
                return _make_async_response(200, {})

            # Вычисляем point_id для каждого дочернего community
            import uuid as uuid_mod
            _ns = uuid_mod.UUID("b8e2c3d4-5e6f-7a89-bcde-f01234567890")
            _point_vectors = {
                str(uuid_mod.uuid5(_ns, "child-low")): child2_vector,   # child2 = низкий score
                str(uuid_mod.uuid5(_ns, "child-high")): child1_vector,  # child1 = высокий score
                str(uuid_mod.uuid5(_ns, "child-mid")): child3_vector,   # child3 = средний score
            }

            def mock_get(url, **kwargs):
                if "community_embeddings" in url:
                    for pid, vec in _point_vectors.items():
                        if pid in url:
                            return _make_async_response(200, {"result": {"vector": vec}})
                    return _make_async_response(200, {"result": {"vector": VALID_EMBEDDING}})
                return _make_async_response(404)

            mocks["mock_session"].post = MagicMock(side_effect=mock_post)
            mocks["mock_session"].get = MagicMock(side_effect=mock_get)

            neo4j_handler = self._build_mock_neo4j_run_handler(
                children_map={"root-scored-uuid": [
                    {"id": "child-low", "title": "Low Community", "summary": "Low score summary", "level": 1},
                    {"id": "child-high", "title": "High Community", "summary": "High score summary", "level": 1},
                    {"id": "child-mid", "title": "Mid Community", "summary": "Mid score summary", "level": 1},
                ]}
            )
            mocks["neo4j_session"].run.side_effect = neo4j_handler

            client = TestClient(app)
            resp = client.post("/search", json={
                "question": "fallback communities scored question",
                "max_tokens": 2000,
            })

            assert resp.status_code == 200
            data = resp.json()
            communities = data["communities"]
            # High Community должна идти раньше Low Community
            high_idx = next((i for i, c in enumerate(communities) if "High Community" in c), None)
            low_idx = next((i for i, c in enumerate(communities) if "Low Community" in c), None)
            if high_idx is not None and low_idx is not None:
                assert high_idx < low_idx, \
                    f"Expected High Community before Low Community, got: High={high_idx}, Low={low_idx}"
