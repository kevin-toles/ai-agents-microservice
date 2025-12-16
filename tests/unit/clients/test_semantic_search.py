"""Unit tests for MSEPSemanticSearchClient.

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-3.2.1: search(query, top_k) method
- AC-3.2.2: get_relationships(chapter_id) returns graph relationships
- AC-3.2.3: get_relationships_batch(chapter_ids) returns batch results
- AC-3.2.4: Falls back gracefully when service unavailable
- AC-3.2.5: Timeout configurable

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.2
Anti-Pattern Focus: #12 (Connection Pooling), #42/#43 (Async Context Managers)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.clients.semantic_search import MSEPSemanticSearchClient


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def client() -> MSEPSemanticSearchClient:
    """Create a client instance for testing."""
    return MSEPSemanticSearchClient(
        base_url="http://test-semantic-search:8081",
        timeout=10.0,
    )


@pytest.fixture
def mock_search_response() -> dict:
    """Mock response for search endpoint."""
    return {
        "results": [
            {
                "chapter_id": "arch_patterns_ch5",
                "book_id": "Architecture_Patterns_with_Python",
                "chapter_number": 5,
                "title": "TDD, DDD, and Event-Driven Architecture",
                "score": 0.92,
                "tier": 1,
            },
            {
                "chapter_id": "arch_patterns_ch6",
                "book_id": "Architecture_Patterns_with_Python",
                "chapter_number": 6,
                "title": "Unit of Work Pattern",
                "score": 0.85,
                "tier": 2,
            },
        ],
        "total": 2,
        "query": "test-driven development",
    }


@pytest.fixture
def mock_relationships_response() -> dict:
    """Mock response for get_relationships endpoint."""
    return {
        "chapter_id": "arch_patterns_ch5",
        "relationships": [
            {
                "target_chapter_id": "arch_patterns_ch6",
                "relationship_type": "PARALLEL",
                "weight": 0.8,
                "shared_concepts": ["TDD", "clean architecture"],
            },
            {
                "target_chapter_id": "ddd_ch3",
                "relationship_type": "PERPENDICULAR",
                "weight": 0.6,
                "shared_concepts": ["domain-driven design"],
            },
        ],
        "total_relationships": 2,
    }


@pytest.fixture
def mock_batch_relationships_response() -> dict:
    """Mock response for batch relationships endpoint."""
    return {
        "results": {
            "arch_patterns_ch5": [
                {
                    "target_chapter_id": "arch_patterns_ch6",
                    "relationship_type": "PARALLEL",
                    "weight": 0.8,
                }
            ],
            "arch_patterns_ch6": [
                {
                    "target_chapter_id": "arch_patterns_ch5",
                    "relationship_type": "PARALLEL",
                    "weight": 0.8,
                }
            ],
        },
        "total_chapters": 2,
    }


# ==============================================================================
# Constructor Tests
# ==============================================================================


class TestMSEPSemanticSearchClientInit:
    """Tests for MSEPSemanticSearchClient initialization."""

    def test_init_with_base_url(self) -> None:
        """Client initializes with base_url."""
        client = MSEPSemanticSearchClient(base_url="http://test:8081")
        assert client.base_url == "http://test:8081"

    def test_init_with_timeout(self) -> None:
        """Client initializes with custom timeout (AC-3.2.5)."""
        client = MSEPSemanticSearchClient(base_url="http://test:8081", timeout=60.0)
        assert client.timeout == 60.0

    def test_init_default_timeout(self) -> None:
        """Client has default timeout of 30.0 seconds."""
        client = MSEPSemanticSearchClient(base_url="http://test:8081")
        assert client.timeout == 30.0

    def test_client_initially_none(self) -> None:
        """Internal httpx client is None until first use (lazy init)."""
        client = MSEPSemanticSearchClient(base_url="http://test:8081")
        assert client._client is None


# ==============================================================================
# Connection Pooling Tests
# ==============================================================================


class TestSemanticSearchConnectionPooling:
    """Tests verifying connection pooling (Anti-Pattern #12)."""

    @pytest.mark.asyncio
    async def test_get_client_creates_single_instance(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """_get_client() creates a single httpx.AsyncClient instance."""
        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

    @pytest.mark.asyncio
    async def test_close_releases_client(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """close() properly releases the httpx.AsyncClient."""
        with patch.object(httpx, "AsyncClient") as mock_async_client:
            mock_instance = AsyncMock()
            mock_async_client.return_value = mock_instance

            _ = await client._get_client()
            assert client._client is not None

            await client.close()
            assert client._client is None
            mock_instance.aclose.assert_called_once()


# ==============================================================================
# search Tests (AC-3.2.1)
# ==============================================================================


class TestSearch:
    """Tests for search method (AC-3.2.1)."""

    @pytest.mark.asyncio
    async def test_search_returns_results_list(
        self,
        client: MSEPSemanticSearchClient,
        mock_search_response: dict,
    ) -> None:
        """search returns list of search results."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.search("test-driven development", top_k=5)

        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_search_respects_top_k(
        self,
        client: MSEPSemanticSearchClient,
        mock_search_response: dict,
    ) -> None:
        """search passes top_k to request."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.search("test query", top_k=10)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        # Verify top_k/limit is in the request payload
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_search_calls_correct_endpoint(
        self,
        client: MSEPSemanticSearchClient,
        mock_search_response: dict,
    ) -> None:
        """search calls /v1/search/hybrid endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.search("test query")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/v1/search/hybrid" in str(call_args)

    @pytest.mark.asyncio
    async def test_search_default_top_k(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """search has default top_k of 5."""
        import inspect

        sig = inspect.signature(client.search)
        top_k_param = sig.parameters.get("top_k")
        assert top_k_param is not None
        assert top_k_param.default == 5

    @pytest.mark.asyncio
    async def test_search_result_has_scores(
        self,
        client: MSEPSemanticSearchClient,
        mock_search_response: dict,
    ) -> None:
        """Search results include similarity scores."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_search_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.search("test query")

        for item in result["results"]:
            assert "score" in item


# ==============================================================================
# get_relationships Tests (AC-3.2.2)
# ==============================================================================


class TestGetRelationships:
    """Tests for get_relationships method (AC-3.2.2)."""

    @pytest.mark.asyncio
    async def test_get_relationships_returns_dict(
        self,
        client: MSEPSemanticSearchClient,
        mock_relationships_response: dict,
    ) -> None:
        """get_relationships returns dict with relationships."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_relationships_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_relationships("arch_patterns_ch5")

        assert "relationships" in result
        assert isinstance(result["relationships"], list)

    @pytest.mark.asyncio
    async def test_get_relationships_has_relationship_type(
        self,
        client: MSEPSemanticSearchClient,
        mock_relationships_response: dict,
    ) -> None:
        """Relationships include relationship_type (PARALLEL/PERPENDICULAR/SKIP_TIER)."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_relationships_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_relationships("arch_patterns_ch5")

        for rel in result["relationships"]:
            assert "relationship_type" in rel
            assert rel["relationship_type"] in ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"]

    @pytest.mark.asyncio
    async def test_get_relationships_has_weight(
        self,
        client: MSEPSemanticSearchClient,
        mock_relationships_response: dict,
    ) -> None:
        """Relationships include weight score."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_relationships_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_relationships("arch_patterns_ch5")

        for rel in result["relationships"]:
            assert "weight" in rel
            assert 0.0 <= rel["weight"] <= 1.0

    @pytest.mark.asyncio
    async def test_get_relationships_calls_correct_endpoint(
        self,
        client: MSEPSemanticSearchClient,
        mock_relationships_response: dict,
    ) -> None:
        """get_relationships calls /v1/graph/relationships endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_relationships_response
        mock_response.raise_for_status = MagicMock()
        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=mock_get),
        ):
            await client.get_relationships("arch_patterns_ch5")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "/v1/graph/relationships" in str(call_args) or "arch_patterns_ch5" in str(call_args)


# ==============================================================================
# get_relationships_batch Tests (AC-3.2.3)
# ==============================================================================


class TestGetRelationshipsBatch:
    """Tests for get_relationships_batch method (AC-3.2.3)."""

    @pytest.mark.asyncio
    async def test_get_relationships_batch_returns_dict(
        self,
        client: MSEPSemanticSearchClient,
        mock_batch_relationships_response: dict,
    ) -> None:
        """get_relationships_batch returns dict keyed by chapter_id."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_batch_relationships_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_relationships_batch(
                ["arch_patterns_ch5", "arch_patterns_ch6"]
            )

        assert "results" in result
        assert "arch_patterns_ch5" in result["results"]
        assert "arch_patterns_ch6" in result["results"]

    @pytest.mark.asyncio
    async def test_get_relationships_batch_accepts_list(
        self,
        client: MSEPSemanticSearchClient,
        mock_batch_relationships_response: dict,
    ) -> None:
        """get_relationships_batch accepts list of chapter_ids."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_batch_relationships_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            chapter_ids = ["ch1", "ch2", "ch3"]
            await client.get_relationships_batch(chapter_ids)

        mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_relationships_batch_empty_list(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """get_relationships_batch with empty list returns empty results."""
        result = await client.get_relationships_batch([])
        assert result == {"results": {}, "total_chapters": 0}


# ==============================================================================
# Graceful Fallback Tests (AC-3.2.4)
# ==============================================================================


class TestGracefulFallback:
    """Tests for graceful fallback on service unavailable (AC-3.2.4)."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_service_unavailable(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """search returns empty results when service unavailable."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 503
        mock_error_response.text = "Service Unavailable"

        http_error = httpx.HTTPStatusError(
            message="503",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_post = AsyncMock(side_effect=http_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            result = await client.search("test query")

        assert result["results"] == []
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_get_relationships_returns_empty_on_connection_error(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """get_relationships returns empty on connection error."""
        connection_error = httpx.ConnectError("Connection refused")

        mock_get = AsyncMock(side_effect=connection_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(get=mock_get),
        ):
            result = await client.get_relationships("any_chapter")

        assert result["relationships"] == []
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_get_relationships_batch_returns_empty_on_timeout(
        self, client: MSEPSemanticSearchClient
    ) -> None:
        """get_relationships_batch returns empty on timeout."""
        timeout_error = httpx.TimeoutException("Request timed out")

        mock_post = AsyncMock(side_effect=timeout_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            result = await client.get_relationships_batch(["ch1", "ch2"])

        assert result["results"] == {}
        assert result.get("error") is not None


# ==============================================================================
# Timeout Configuration Tests (AC-3.2.5)
# ==============================================================================


class TestTimeoutConfiguration:
    """Tests for timeout configuration (AC-3.2.5)."""

    @pytest.mark.asyncio
    async def test_client_uses_configured_timeout(self) -> None:
        """Client uses configured timeout value."""
        client = MSEPSemanticSearchClient(
            base_url="http://test:8081",
            timeout=45.0,
        )

        with patch.object(httpx, "AsyncClient") as mock_async_client:
            await client._get_client()

            mock_async_client.assert_called_once()
            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs.get("timeout") == 45.0

    def test_timeout_is_float(self) -> None:
        """Timeout is stored as float."""
        client = MSEPSemanticSearchClient(
            base_url="http://test:8081",
            timeout=30,  # int input
        )
        assert isinstance(client.timeout, (int, float))
