"""Unit Tests for MSEP Orchestrator.

WBS: MSE-4.3 - Orchestrator
Tests for main enrich_metadata() function.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
- S3776: Cognitive complexity < 15 per function

Acceptance Criteria Tested:
- AC-4.3.1: enrich_metadata() returns EnrichedMetadata
- AC-4.3.2: Calls dispatcher then merger
- AC-4.3.3: Handles partial failures gracefully
- AC-4.3.4: Records processing time
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.msep.config import MSEPConfig
from src.agents.msep.dispatcher import DispatchResult
from src.agents.msep.exceptions import ServiceUnavailableError
from src.agents.msep.orchestrator import MSEPOrchestrator
from src.agents.msep.schemas import (
    ChapterMeta,
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    MSEPRequest,
    Provenance,
)

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
def mock_dispatch_result() -> DispatchResult:
    """Create mock dispatch result."""
    return DispatchResult(
        embeddings=[[0.1] * 768 for _ in range(3)],
        similarity_matrix=[
            [1.0, 0.75, 0.65],
            [0.75, 1.0, 0.70],
            [0.65, 0.70, 1.0],
        ],
        topics=[0, 0, 1],
        topic_info=[{"id": 0, "count": 2}, {"id": 1, "count": 1}],
        keywords=[
            ["neural", "network", "deep"],
            ["gradient", "optimization", "descent"],
            ["reinforcement", "policy", "agent"],
        ],
        hybrid_results={
            "Deep Learning:ch1": [{"target": "Deep Learning:ch2", "score": 0.8}],
        },
    )


@pytest.fixture
def mock_dispatch_result_partial_failure() -> DispatchResult:
    """Create mock dispatch result with partial failure."""
    return DispatchResult(
        embeddings=[[0.1] * 768 for _ in range(3)],
        similarity_matrix=[
            [1.0, 0.75, 0.65],
            [0.75, 1.0, 0.70],
            [0.65, 0.70, 1.0],
        ],
        topics=[],  # Topics failed
        topic_info=[],
        topics_error="BERTopic service unavailable",
        keywords=[],  # Keywords failed
        keywords_error="TF-IDF service unavailable",
        hybrid_results=None,  # Hybrid failed
        hybrid_error="Semantic search timeout",
    )


@pytest.fixture
def mock_dispatcher(mock_dispatch_result: DispatchResult) -> AsyncMock:
    """Create mock dispatcher."""
    mock = AsyncMock()
    mock.dispatch_enrichment.return_value = mock_dispatch_result
    return mock


@pytest.fixture
def orchestrator_with_mock_dispatcher(mock_dispatcher: AsyncMock) -> MSEPOrchestrator:
    """Create orchestrator with mocked dispatcher."""
    mock_code_orch = AsyncMock()
    mock_semantic = AsyncMock()
    orchestrator = MSEPOrchestrator(
        code_orchestrator=mock_code_orch,
        semantic_search=mock_semantic,
    )
    orchestrator._dispatcher = mock_dispatcher
    return orchestrator


# =============================================================================
# AC-4.3.1: enrich_metadata() returns EnrichedMetadata
# =============================================================================


class TestOrchestratorReturnsEnrichedMetadata:
    """Tests for AC-4.3.1: enrich_metadata returns EnrichedMetadata."""

    @pytest.mark.asyncio
    async def test_returns_enriched_metadata_type(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """enrich_metadata returns EnrichedMetadata instance."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        assert isinstance(result, EnrichedMetadata)

    @pytest.mark.asyncio
    async def test_returns_chapters_list(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """EnrichedMetadata contains chapters list."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        assert isinstance(result.chapters, list)
        assert len(result.chapters) == len(sample_request.chapter_index)

    @pytest.mark.asyncio
    async def test_chapters_have_correct_ids(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Enriched chapters have correct chapter IDs."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        expected_ids = [ch.id for ch in sample_request.chapter_index]
        actual_ids = [ch.chapter_id for ch in result.chapters]
        assert actual_ids == expected_ids

    @pytest.mark.asyncio
    async def test_chapters_have_similar_chapters(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Enriched chapters have similar_chapters list."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        for chapter in result.chapters:
            assert isinstance(chapter.similar_chapters, list)

    @pytest.mark.asyncio
    async def test_chapters_have_keywords(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Enriched chapters have keywords."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        for chapter in result.chapters:
            assert isinstance(chapter.keywords, MergedKeywords)

    @pytest.mark.asyncio
    async def test_chapters_have_topic_id(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Enriched chapters have topic_id."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        for chapter in result.chapters:
            assert isinstance(chapter.topic_id, int)

    @pytest.mark.asyncio
    async def test_chapters_have_provenance(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Enriched chapters have provenance."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        for chapter in result.chapters:
            assert isinstance(chapter.provenance, Provenance)


# =============================================================================
# AC-4.3.2: Calls dispatcher then merger
# =============================================================================


class TestOrchestratorCallsDispatcherThenMerger:
    """Tests for AC-4.3.2: Calls dispatcher then merger."""

    @pytest.mark.asyncio
    async def test_calls_dispatcher(
        self,
        sample_request: MSEPRequest,
        mock_dispatcher: AsyncMock,
    ) -> None:
        """Orchestrator calls dispatcher.dispatch_enrichment."""
        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        await orchestrator.enrich_metadata(sample_request)

        mock_dispatcher.dispatch_enrichment.assert_called_once_with(sample_request)

    @pytest.mark.asyncio
    async def test_uses_dispatch_result_for_merging(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result: DispatchResult,
    ) -> None:
        """Orchestrator uses dispatch result for building enriched metadata."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Verify topics from dispatch result are used
        assert result.chapters[0].topic_id == mock_dispatch_result.topics[0]
        assert result.chapters[2].topic_id == mock_dispatch_result.topics[2]


# =============================================================================
# AC-4.3.3: Handles partial failures gracefully
# =============================================================================


class TestOrchestratorHandlesPartialFailures:
    """Tests for AC-4.3.3: Handles partial failures gracefully."""

    @pytest.mark.asyncio
    async def test_succeeds_with_topics_failure(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result_partial_failure: DispatchResult,
    ) -> None:
        """Orchestrator succeeds when topics fail."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result_partial_failure

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Should succeed with None topic_id when topics unavailable
        assert result is not None
        for chapter in result.chapters:
            assert chapter.topic_id is None  # None when topics unavailable

    @pytest.mark.asyncio
    async def test_succeeds_with_keywords_failure(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result_partial_failure: DispatchResult,
    ) -> None:
        """Orchestrator succeeds when keywords fail."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result_partial_failure

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Should succeed with empty keywords
        assert result is not None
        for chapter in result.chapters:
            assert chapter.keywords.tfidf == []

    @pytest.mark.asyncio
    async def test_succeeds_with_hybrid_failure(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result_partial_failure: DispatchResult,
    ) -> None:
        """Orchestrator succeeds when hybrid search fails."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result_partial_failure

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Should succeed without hybrid results
        assert result is not None

    @pytest.mark.asyncio
    async def test_raises_on_critical_failure(
        self,
        sample_request: MSEPRequest,
    ) -> None:
        """Orchestrator raises when critical services fail."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.side_effect = ServiceUnavailableError(
            "Code-Orchestrator unavailable"
        )

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        with pytest.raises(ServiceUnavailableError):
            await orchestrator.enrich_metadata(sample_request)


# =============================================================================
# AC-4.3.4: Records processing time
# =============================================================================


class TestOrchestratorRecordsProcessingTime:
    """Tests for AC-4.3.4: Records processing time."""

    @pytest.mark.asyncio
    async def test_returns_processing_time_ms(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """EnrichedMetadata includes processing_time_ms."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        assert hasattr(result, "processing_time_ms")
        assert isinstance(result.processing_time_ms, float)

    @pytest.mark.asyncio
    async def test_processing_time_is_positive(
        self,
        sample_request: MSEPRequest,
        orchestrator_with_mock_dispatcher: MSEPOrchestrator,
    ) -> None:
        """Processing time is positive."""
        result = await orchestrator_with_mock_dispatcher.enrich_metadata(sample_request)
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_processing_time_reflects_actual_time(
        self,
        sample_request: MSEPRequest,
    ) -> None:
        """Processing time approximately reflects actual execution time."""
        import asyncio

        # Create dispatcher with delay
        async def slow_dispatch(req: MSEPRequest) -> DispatchResult:
            await asyncio.sleep(0.05)  # 50ms delay
            return DispatchResult(
                embeddings=[[0.1]],
                similarity_matrix=[[1.0]],
                topics=[0],
                topic_info=[],
                keywords=[[]],
            )

        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment = slow_dispatch

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        start = time.perf_counter()
        result = await orchestrator.enrich_metadata(sample_request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Processing time should be close to actual elapsed time
        assert result.processing_time_ms >= 40  # At least 40ms
        assert result.processing_time_ms <= elapsed_ms + 10  # With margin


# =============================================================================
# Cross-Reference Building Tests
# =============================================================================


class TestOrchestratorCrossReferences:
    """Tests for cross-reference building."""

    @pytest.mark.asyncio
    async def test_cross_refs_use_similarity_scores(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result: DispatchResult,
    ) -> None:
        """Cross-references use similarity matrix scores."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Check that similar chapters exist with scores
        chapter_0 = result.chapters[0]
        assert len(chapter_0.similar_chapters) > 0
        for xref in chapter_0.similar_chapters:
            assert isinstance(xref.base_score, float)

    @pytest.mark.asyncio
    async def test_cross_refs_apply_topic_boost(
        self,
        sample_request: MSEPRequest,
        mock_dispatch_result: DispatchResult,
    ) -> None:
        """Cross-references apply topic boost for same-topic chapters."""
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(sample_request)

        # Chapters 0 and 1 have same topic (0)
        chapter_0 = result.chapters[0]
        xref_to_ch1 = next(
            (x for x in chapter_0.similar_chapters if "ch2" in x.target), None
        )
        if xref_to_ch1:
            assert xref_to_ch1.topic_boost == sample_request.config.same_topic_boost

    @pytest.mark.asyncio
    async def test_cross_refs_respect_threshold(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[ChapterMeta],
    ) -> None:
        """Cross-references respect similarity threshold."""
        # High threshold to filter most results
        config = MSEPConfig(threshold=0.9)
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=sample_chapter_index,
            config=config,
        )

        mock_dispatch_result = DispatchResult(
            embeddings=[[0.1]],
            similarity_matrix=[
                [1.0, 0.5, 0.3],  # Low similarity
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
            topics=[0, 1, 2],
            topic_info=[],
            keywords=[[], [], []],
        )

        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch_enrichment.return_value = mock_dispatch_result

        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        orchestrator._dispatcher = mock_dispatcher

        result = await orchestrator.enrich_metadata(request)

        # With 0.9 threshold and max 0.5 similarity, no similar chapters
        for chapter in result.chapters:
            assert len(chapter.similar_chapters) == 0


# =============================================================================
# Orchestrator Configuration Tests
# =============================================================================


class TestOrchestratorConfiguration:
    """Tests for orchestrator configuration."""

    def test_orchestrator_creates_dispatcher_with_clients(self) -> None:
        """Orchestrator creates dispatcher with provided clients."""
        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )
        assert orchestrator._dispatcher is not None

    def test_orchestrator_accepts_custom_clients(self) -> None:
        """Orchestrator accepts custom service clients."""
        mock_code_orch = AsyncMock()
        mock_semantic = AsyncMock()

        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orch,
            semantic_search=mock_semantic,
        )

        assert orchestrator is not None
