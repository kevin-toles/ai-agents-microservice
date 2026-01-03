"""Unit tests for CodeOrchestratorClient.

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-3.1.1: get_embeddings(texts) method
- AC-3.1.2: get_similarity_matrix(texts) returns NDArray
- AC-3.1.3: cluster_topics(corpus, chapter_index) returns topic assignments
- AC-3.1.4: extract_keywords(corpus, top_k) returns keywords per doc
- AC-3.1.5: Uses single httpx.AsyncClient (connection pooling)
- AC-3.1.6: Retries 3x on transient errors (5xx, timeout)
- AC-3.1.7: Raises ServiceUnavailableError on permanent failure

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.1
Anti-Pattern Focus: #12 (Connection Pooling), #42/#43 (Async Context Managers)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from src.agents.msep.exceptions import ServiceUnavailableError
from src.clients.code_orchestrator import CodeOrchestratorClient


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def client() -> CodeOrchestratorClient:
    """Create a client instance for testing."""
    return CodeOrchestratorClient(
        base_url="http://test-code-orchestrator:8082",
        timeout=10.0,
    )


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding/similarity tests."""
    return [
        "Test-driven development improves code quality.",
        "Domain-driven design focuses on the business domain.",
        "Event sourcing tracks all state changes.",
    ]


@pytest.fixture
def mock_embeddings_response() -> dict:
    """Mock response for embeddings endpoint."""
    return {
        "embeddings": [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        ],
        "model": "all-MiniLM-L6-v2",
        "dimension": 5,
    }


@pytest.fixture
def mock_similarity_response() -> dict:
    """Mock response for similarity matrix endpoint."""
    return {
        "similarity_matrix": [
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ],
        "texts_count": 3,
    }


@pytest.fixture
def mock_cluster_response() -> dict:
    """Mock response for cluster_topics endpoint."""
    return {
        "topic_assignments": [0, 0, 1, 1, 2],
        "topic_count": 3,
        "chapter_topic": 0,
        "topics_info": [
            {"topic_id": 0, "keywords": ["TDD", "testing"]},
            {"topic_id": 1, "keywords": ["DDD", "domain"]},
            {"topic_id": 2, "keywords": ["event", "sourcing"]},
        ],
    }


@pytest.fixture
def mock_keywords_response() -> dict:
    """Mock response for extract_keywords endpoint."""
    return {
        "keywords": [
            ["TDD", "testing", "code", "quality"],
            ["DDD", "domain", "business", "model"],
            ["event", "sourcing", "state", "changes"],
        ],
        "method": "tfidf",
    }


# ==============================================================================
# Constructor Tests
# ==============================================================================


class TestCodeOrchestratorClientInit:
    """Tests for CodeOrchestratorClient initialization."""

    def test_init_with_base_url(self) -> None:
        """Client initializes with base_url."""
        client = CodeOrchestratorClient(base_url="http://test:8082")
        assert client.base_url == "http://test:8082"

    def test_init_with_timeout(self) -> None:
        """Client initializes with custom timeout."""
        client = CodeOrchestratorClient(base_url="http://test:8082", timeout=60.0)
        assert client.timeout == pytest.approx(60.0)

    def test_init_default_timeout(self) -> None:
        """Client has default timeout of 30.0 seconds."""
        client = CodeOrchestratorClient(base_url="http://test:8082")
        assert client.timeout == pytest.approx(30.0)

    def test_init_with_max_retries(self) -> None:
        """Client initializes with custom max_retries."""
        client = CodeOrchestratorClient(
            base_url="http://test:8082", max_retries=5
        )
        assert client.max_retries == 5

    def test_init_default_max_retries(self) -> None:
        """Client has default max_retries of 3."""
        client = CodeOrchestratorClient(base_url="http://test:8082")
        assert client.max_retries == 3

    def test_client_initially_none(self) -> None:
        """Internal httpx client is None until first use (lazy init)."""
        client = CodeOrchestratorClient(base_url="http://test:8082")
        assert client._client is None


# ==============================================================================
# Connection Pooling Tests (AC-3.1.5)
# ==============================================================================


class TestConnectionPooling:
    """Tests verifying connection pooling (Anti-Pattern #12)."""

    @pytest.mark.asyncio
    async def test_get_client_creates_single_instance(
        self, client: CodeOrchestratorClient
    ) -> None:
        """_get_client() creates a single httpx.AsyncClient instance."""
        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2  # Same instance (pooled)

    @pytest.mark.asyncio
    async def test_client_reused_across_requests(
        self, client: CodeOrchestratorClient, mock_embeddings_response: dict
    ) -> None:
        """Same httpx.AsyncClient reused across multiple requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            wraps=client._get_client,
        ):
            # Mock the actual HTTP call
            with patch.object(
                httpx.AsyncClient, "post", return_value=mock_response
            ):
                # First call initializes client
                http_client1 = await client._get_client()

                # Simulate second call
                http_client2 = await client._get_client()

                assert http_client1 is http_client2

    @pytest.mark.asyncio
    async def test_close_releases_client(
        self, client: CodeOrchestratorClient
    ) -> None:
        """close() properly releases the httpx.AsyncClient."""
        # Initialize client
        with patch.object(httpx, "AsyncClient") as mock_async_client:
            mock_instance = AsyncMock()
            mock_async_client.return_value = mock_instance

            _ = await client._get_client()
            assert client._client is not None

            await client.close()
            assert client._client is None
            mock_instance.aclose.assert_called_once()


# ==============================================================================
# get_embeddings Tests (AC-3.1.1)
# ==============================================================================


class TestGetEmbeddings:
    """Tests for get_embeddings method (AC-3.1.1)."""

    @pytest.mark.asyncio
    async def test_get_embeddings_returns_ndarray(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_embeddings_response: dict,
    ) -> None:
        """get_embeddings returns numpy NDArray."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_embeddings(sample_texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 5)  # 3 texts, 5 dimensions

    @pytest.mark.asyncio
    async def test_get_embeddings_calls_correct_endpoint(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_embeddings_response: dict,
    ) -> None:
        """get_embeddings calls /v1/sbert/embeddings endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.get_embeddings(sample_texts)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/v1/embeddings" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list_returns_empty(
        self, client: CodeOrchestratorClient
    ) -> None:
        """get_embeddings with empty list returns empty array."""
        result = await client.get_embeddings([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0


# ==============================================================================
# get_similarity_matrix Tests (AC-3.1.2)
# ==============================================================================


class TestGetSimilarityMatrix:
    """Tests for get_similarity_matrix method (AC-3.1.2)."""

    @pytest.mark.asyncio
    async def test_get_similarity_matrix_returns_ndarray(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_similarity_response: dict,
    ) -> None:
        """get_similarity_matrix returns numpy NDArray."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_similarity_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_similarity_matrix(sample_texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # 3x3 similarity matrix

    @pytest.mark.asyncio
    async def test_get_similarity_matrix_diagonal_is_one(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_similarity_response: dict,
    ) -> None:
        """Similarity matrix diagonal should be 1.0 (self-similarity)."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_similarity_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.get_similarity_matrix(sample_texts)

        for i in range(result.shape[0]):
            assert result[i, i] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_get_similarity_matrix_calls_correct_endpoint(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_similarity_response: dict,
    ) -> None:
        """get_similarity_matrix calls /v1/sbert/similarity endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_similarity_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.get_similarity_matrix(sample_texts)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/v1/similarity/matrix" in str(call_args)


# ==============================================================================
# cluster_topics Tests (AC-3.1.3)
# ==============================================================================


class TestClusterTopics:
    """Tests for cluster_topics method (AC-3.1.3)."""

    @pytest.mark.asyncio
    async def test_cluster_topics_returns_dict(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_cluster_response: dict,
    ) -> None:
        """cluster_topics returns topic assignment dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_cluster_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.cluster_topics(sample_texts, chapter_index=0)

        assert "topic_assignments" in result
        assert "topic_count" in result
        assert "chapter_topic" in result

    @pytest.mark.asyncio
    async def test_cluster_topics_has_topic_assignments(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_cluster_response: dict,
    ) -> None:
        """cluster_topics returns list of topic assignments per doc."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_cluster_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.cluster_topics(sample_texts, chapter_index=0)

        assert isinstance(result["topic_assignments"], list)

    @pytest.mark.asyncio
    async def test_cluster_topics_calls_correct_endpoint(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_cluster_response: dict,
    ) -> None:
        """cluster_topics calls /v1/bertopic/cluster endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_cluster_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.cluster_topics(sample_texts, chapter_index=0)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/v1/cluster" in str(call_args)


# ==============================================================================
# extract_keywords Tests (AC-3.1.4)
# ==============================================================================


class TestExtractKeywords:
    """Tests for extract_keywords method (AC-3.1.4)."""

    @pytest.mark.asyncio
    async def test_extract_keywords_returns_list(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_keywords_response: dict,
    ) -> None:
        """extract_keywords returns list of keyword lists."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_keywords_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.extract_keywords(sample_texts, top_k=4)

        assert isinstance(result, list)
        assert len(result) == 3  # One list per text

    @pytest.mark.asyncio
    async def test_extract_keywords_respects_top_k(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_keywords_response: dict,
    ) -> None:
        """extract_keywords respects top_k parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_keywords_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await client.extract_keywords(sample_texts, top_k=4)

        for keywords in result:
            assert len(keywords) <= 4

    @pytest.mark.asyncio
    async def test_extract_keywords_calls_correct_endpoint(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_keywords_response: dict,
    ) -> None:
        """extract_keywords calls /v1/keywords/extract endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_keywords_response
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            await client.extract_keywords(sample_texts, top_k=5)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/v1/keywords" in str(call_args)

    @pytest.mark.asyncio
    async def test_extract_keywords_default_top_k(
        self, client: CodeOrchestratorClient
    ) -> None:
        """extract_keywords has default top_k of 5."""
        # Check method signature has default
        import inspect

        sig = inspect.signature(client.extract_keywords)
        top_k_param = sig.parameters.get("top_k")
        assert top_k_param is not None
        assert top_k_param.default == 5


# ==============================================================================
# Retry Logic Tests (AC-3.1.6)
# ==============================================================================


class TestRetryLogic:
    """Tests for retry logic on transient errors (AC-3.1.6)."""

    @pytest.mark.asyncio
    async def test_retries_on_500_error(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_embeddings_response: dict,
    ) -> None:
        """Client retries on 500 Internal Server Error."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_embeddings_response
        mock_success_response.raise_for_status = MagicMock()

        http_error = httpx.HTTPStatusError(
            message="500",
            request=MagicMock(),
            response=mock_error_response,
        )

        # First call fails, second succeeds
        mock_post = AsyncMock(side_effect=[http_error, mock_success_response])

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            result = await client.get_embeddings(sample_texts)

        assert isinstance(result, np.ndarray)
        assert mock_post.call_count == 2  # Retried once

    @pytest.mark.asyncio
    async def test_retries_on_503_error(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_embeddings_response: dict,
    ) -> None:
        """Client retries on 503 Service Unavailable."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 503
        mock_error_response.text = "Service Unavailable"

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_embeddings_response
        mock_success_response.raise_for_status = MagicMock()

        http_error = httpx.HTTPStatusError(
            message="503",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_post = AsyncMock(side_effect=[http_error, mock_success_response])

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            result = await client.get_embeddings(sample_texts)

        assert isinstance(result, np.ndarray)
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_timeout(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
        mock_embeddings_response: dict,
    ) -> None:
        """Client retries on timeout errors."""
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_embeddings_response
        mock_success_response.raise_for_status = MagicMock()

        timeout_error = httpx.TimeoutException("Request timed out")

        mock_post = AsyncMock(side_effect=[timeout_error, mock_success_response])

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            result = await client.get_embeddings(sample_texts)

        assert isinstance(result, np.ndarray)
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_is_three(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
    ) -> None:
        """Client retries exactly 3 times before failing."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"

        http_error = httpx.HTTPStatusError(
            message="500",
            request=MagicMock(),
            response=mock_error_response,
        )

        # Always fail
        mock_post = AsyncMock(side_effect=http_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            with pytest.raises(ServiceUnavailableError):
                await client.get_embeddings(sample_texts)

        # Initial call + 3 retries = 4 total calls
        assert mock_post.call_count == 4

    @pytest.mark.asyncio
    async def test_no_retry_on_400_error(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
    ) -> None:
        """Client does not retry on 400 Bad Request (client error)."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 400
        mock_error_response.text = "Bad Request"

        http_error = httpx.HTTPStatusError(
            message="400",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_post = AsyncMock(side_effect=http_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_embeddings(sample_texts)

        # No retry on client error
        assert mock_post.call_count == 1


# ==============================================================================
# ServiceUnavailableError Tests (AC-3.1.7)
# ==============================================================================


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError on permanent failure (AC-3.1.7)."""

    @pytest.mark.asyncio
    async def test_raises_service_unavailable_after_retries(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
    ) -> None:
        """Raises ServiceUnavailableError after exhausting retries."""
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
            with pytest.raises(ServiceUnavailableError) as exc_info:
                await client.get_embeddings(sample_texts)

        assert "code-orchestrator" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_service_unavailable_includes_cause(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
    ) -> None:
        """ServiceUnavailableError includes original exception as cause."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"

        http_error = httpx.HTTPStatusError(
            message="500",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_post = AsyncMock(side_effect=http_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            with pytest.raises(ServiceUnavailableError) as exc_info:
                await client.get_embeddings(sample_texts)

        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_connection_error_raises_service_unavailable(
        self,
        client: CodeOrchestratorClient,
        sample_texts: list[str],
    ) -> None:
        """Connection errors raise ServiceUnavailableError."""
        connection_error = httpx.ConnectError("Connection refused")

        mock_post = AsyncMock(side_effect=connection_error)

        with patch.object(
            client,
            "_get_client",
            return_value=AsyncMock(post=mock_post),
        ):
            with pytest.raises(ServiceUnavailableError):
                await client.get_embeddings(sample_texts)
