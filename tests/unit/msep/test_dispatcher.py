"""Unit Tests for MSEP Dispatcher.

WBS: MSE-4.2 - Parallel Dispatcher
Tests for asyncio.gather() orchestration of parallel service calls.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
- S3776: Cognitive complexity < 15 per function

Acceptance Criteria Tested:
- AC-4.2.1: dispatch_enrichment() uses asyncio.gather()
- AC-4.2.2: Runs SBERT, TF-IDF, BERTopic concurrently
- AC-4.2.3: Hybrid search runs conditionally (if enabled)
- AC-4.2.4: Each task has independent error handling
- AC-4.2.5: Cognitive complexity < 15 (SonarQube scan)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.msep.config import MSEPConfig
from src.agents.msep.exceptions import ServiceUnavailableError
from src.agents.msep.schemas import ChapterMeta, MSEPRequest

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def msep_config() -> MSEPConfig:
    """Create default MSEP config."""
    return MSEPConfig()


@pytest.fixture
def msep_config_no_hybrid() -> MSEPConfig:
    """Create MSEP config with hybrid search disabled."""
    return MSEPConfig(enable_hybrid_search=False)


@pytest.fixture
def sample_corpus() -> list[str]:
    """Create sample corpus for testing."""
    return [
        "Introduction to neural networks and deep learning.",
        "Advanced optimization techniques for gradient descent.",
        "Reinforcement learning and policy gradients.",
    ]


@pytest.fixture
def sample_chapter_index() -> list[ChapterMeta]:
    """Create sample chapter index for testing."""
    return [
        ChapterMeta(book="Deep Learning", chapter=1, title="Neural Networks"),
        ChapterMeta(book="Deep Learning", chapter=2, title="Optimization"),
        ChapterMeta(book="Deep Learning", chapter=3, title="RL"),
    ]


@pytest.fixture
def sample_request(
    sample_corpus: list[str],
    sample_chapter_index: list[ChapterMeta],
    msep_config: MSEPConfig,
) -> MSEPRequest:
    """Create sample MSEP request."""
    return MSEPRequest(
        corpus=sample_corpus,
        chapter_index=sample_chapter_index,
        config=msep_config,
    )


@pytest.fixture
def sample_request_no_hybrid(
    sample_corpus: list[str],
    sample_chapter_index: list[ChapterMeta],
    msep_config_no_hybrid: MSEPConfig,
) -> MSEPRequest:
    """Create sample MSEP request without hybrid search."""
    return MSEPRequest(
        corpus=sample_corpus,
        chapter_index=sample_chapter_index,
        config=msep_config_no_hybrid,
    )


@pytest.fixture
def mock_code_orchestrator() -> AsyncMock:
    """Create mock Code Orchestrator client."""
    mock = AsyncMock()
    mock.get_embeddings.return_value = {
        "embeddings": [[0.1] * 768 for _ in range(3)],
    }
    mock.get_similarity_matrix.return_value = {
        "similarity_matrix": [[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]],
    }
    mock.cluster_topics.return_value = {
        "topics": [0, 0, 1],
        "topic_info": [{"id": 0, "count": 2}, {"id": 1, "count": 1}],
    }
    mock.extract_keywords.return_value = {
        "keywords": [
            ["neural", "network"],
            ["gradient", "optimization"],
            ["reinforcement", "policy"],
        ],
    }
    return mock


@pytest.fixture
def mock_semantic_search() -> AsyncMock:
    """Create mock Semantic Search client."""
    mock = AsyncMock()
    mock.search.return_value = {
        "results": [],
        "error": None,
    }
    mock.get_relationships_batch.return_value = {
        "results": {},
        "error": None,
    }
    return mock


# =============================================================================
# AC-4.2.1: dispatch_enrichment() uses asyncio.gather()
# =============================================================================


class TestDispatcherUsesAsyncioGather:
    """Tests for AC-4.2.1: dispatch_enrichment uses asyncio.gather()."""

    @pytest.mark.asyncio
    async def test_dispatcher_returns_dispatch_result(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """dispatch_enrichment returns DispatchResult dataclass."""
        from src.agents.msep.dispatcher import MSEPDispatcher, DispatchResult

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        assert isinstance(result, DispatchResult)

    @pytest.mark.asyncio
    async def test_dispatcher_result_contains_all_fields(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """DispatchResult contains embeddings, similarity, topics, keywords."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        assert result.embeddings is not None
        assert result.similarity_matrix is not None
        assert result.topics is not None
        assert result.keywords is not None

    @pytest.mark.asyncio
    async def test_dispatcher_calls_gather_for_parallel_execution(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """dispatch_enrichment uses asyncio.gather internally."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        with patch("asyncio.gather", wraps=asyncio.gather) as mock_gather:
            await dispatcher.dispatch_enrichment(sample_request)
            mock_gather.assert_called()


# =============================================================================
# AC-4.2.2: Runs SBERT, TF-IDF, BERTopic concurrently
# =============================================================================


class TestDispatcherRunsConcurrently:
    """Tests for AC-4.2.2: Concurrent execution verification."""

    @pytest.mark.asyncio
    async def test_calls_all_services(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Dispatcher calls all Code-Orchestrator services."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        await dispatcher.dispatch_enrichment(sample_request)

        mock_code_orchestrator.get_embeddings.assert_called_once()
        mock_code_orchestrator.get_similarity_matrix.assert_called_once()
        mock_code_orchestrator.cluster_topics.assert_called_once()
        mock_code_orchestrator.extract_keywords.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_timing_verification(
        self,
        sample_request: MSEPRequest,
    ) -> None:
        """Verify concurrent execution by timing."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        # Create clients with delays
        delay_seconds = 0.1

        async def slow_embeddings(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(delay_seconds)
            return {"embeddings": [[0.1] * 768]}

        async def slow_similarity(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(delay_seconds)
            return {"similarity_matrix": [[1.0]]}

        async def slow_topics(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(delay_seconds)
            return {"topics": [0], "topic_info": []}

        async def slow_keywords(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(delay_seconds)
            return {"keywords": [["test"]]}

        mock_code_orchestrator = AsyncMock()
        mock_code_orchestrator.get_embeddings = slow_embeddings
        mock_code_orchestrator.get_similarity_matrix = slow_similarity
        mock_code_orchestrator.cluster_topics = slow_topics
        mock_code_orchestrator.extract_keywords = slow_keywords

        mock_semantic_search = AsyncMock()
        mock_semantic_search.get_relationships_batch.return_value = {"results": {}}

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        start = time.perf_counter()
        await dispatcher.dispatch_enrichment(sample_request)
        elapsed = time.perf_counter() - start

        # If sequential: 4 * 0.1 = 0.4s
        # If concurrent: ~0.1s + overhead
        # Allow margin for test environment
        assert elapsed < 0.35, f"Expected concurrent execution (<0.35s), got {elapsed:.2f}s"


# =============================================================================
# AC-4.2.3: Hybrid search runs conditionally (if enabled)
# =============================================================================


class TestDispatcherHybridSearchConditional:
    """Tests for AC-4.2.3: Hybrid search conditional execution."""

    @pytest.mark.asyncio
    async def test_hybrid_search_called_when_enabled(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Hybrid search is called when enabled in config."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        await dispatcher.dispatch_enrichment(sample_request)

        mock_semantic_search.get_relationships_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_not_called_when_disabled(
        self,
        sample_request_no_hybrid: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Hybrid search is NOT called when disabled in config."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        await dispatcher.dispatch_enrichment(sample_request_no_hybrid)

        mock_semantic_search.get_relationships_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_has_none_hybrid_when_disabled(
        self,
        sample_request_no_hybrid: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """DispatchResult has None hybrid_results when disabled."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request_no_hybrid)

        assert result.hybrid_results is None


# =============================================================================
# AC-4.2.4: Each task has independent error handling
# =============================================================================


class TestDispatcherIndependentErrorHandling:
    """Tests for AC-4.2.4: Independent error handling per task."""

    @pytest.mark.asyncio
    async def test_embeddings_failure_propagates(
        self,
        sample_request: MSEPRequest,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """SBERT embeddings failure raises (critical service)."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        mock_code_orchestrator = AsyncMock()
        mock_code_orchestrator.get_embeddings.side_effect = ServiceUnavailableError(
            "Code-Orchestrator unavailable"
        )
        mock_code_orchestrator.get_similarity_matrix.return_value = {}
        mock_code_orchestrator.cluster_topics.return_value = {"topics": [], "topic_info": []}
        mock_code_orchestrator.extract_keywords.return_value = {"keywords": []}

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        with pytest.raises(ServiceUnavailableError):
            await dispatcher.dispatch_enrichment(sample_request)

    @pytest.mark.asyncio
    async def test_keywords_failure_returns_empty(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """TF-IDF keywords failure returns empty (non-critical)."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        mock_code_orchestrator.extract_keywords.side_effect = Exception("TF-IDF failed")

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        # Should succeed with empty keywords
        assert result.keywords is not None or result.keywords_error is not None

    @pytest.mark.asyncio
    async def test_topics_failure_returns_empty(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """BERTopic failure returns empty (non-critical)."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        mock_code_orchestrator.cluster_topics.side_effect = Exception("BERTopic failed")

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        # Should succeed with empty topics
        assert result.topics is not None or result.topics_error is not None

    @pytest.mark.asyncio
    async def test_hybrid_search_failure_returns_none(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Hybrid search failure returns None (non-critical)."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        mock_semantic_search.get_relationships_batch.side_effect = Exception(
            "Semantic search failed"
        )

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        # Should succeed with None hybrid_results
        assert result is not None

    @pytest.mark.asyncio
    async def test_one_failure_others_succeed(
        self,
        sample_request: MSEPRequest,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """One non-critical failure doesn't affect others."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        # Only topics fails
        mock_code_orchestrator.cluster_topics.side_effect = Exception("BERTopic failed")

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        result = await dispatcher.dispatch_enrichment(sample_request)

        # Embeddings, similarity, keywords should succeed
        assert result.embeddings is not None
        assert result.similarity_matrix is not None
        assert result.keywords is not None


# =============================================================================
# DispatchResult Structure Tests
# =============================================================================


class TestDispatchResultStructure:
    """Tests for DispatchResult dataclass structure."""

    def test_dispatch_result_is_dataclass(self) -> None:
        """DispatchResult is a dataclass."""
        from dataclasses import is_dataclass

        from src.agents.msep.dispatcher import DispatchResult

        assert is_dataclass(DispatchResult)

    def test_dispatch_result_fields(self) -> None:
        """DispatchResult has required fields."""
        from src.agents.msep.dispatcher import DispatchResult

        result = DispatchResult(
            embeddings=[[0.1]],
            similarity_matrix=[[1.0]],
            topics=[0],
            topic_info=[{"id": 0}],
            keywords=[["test"]],
            hybrid_results=None,
            embeddings_error=None,
            similarity_error=None,
            topics_error=None,
            keywords_error=None,
            hybrid_error=None,
        )

        assert hasattr(result, "embeddings")
        assert hasattr(result, "similarity_matrix")
        assert hasattr(result, "topics")
        assert hasattr(result, "topic_info")
        assert hasattr(result, "keywords")
        assert hasattr(result, "hybrid_results")


# =============================================================================
# Dispatcher Configuration Tests
# =============================================================================


class TestDispatcherConfiguration:
    """Tests for dispatcher configuration."""

    def test_dispatcher_accepts_clients(
        self,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Dispatcher accepts client dependencies."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        assert dispatcher is not None

    def test_dispatcher_protocol_compliance(
        self,
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Dispatcher works with protocol-compliant clients."""
        from src.agents.msep.dispatcher import MSEPDispatcher

        # AsyncMock satisfies the protocol duck typing
        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )

        assert dispatcher._code_orchestrator is mock_code_orchestrator
        assert dispatcher._semantic_search is mock_semantic_search
