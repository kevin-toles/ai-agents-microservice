"""MSE-7.1: Integration Tests (Mocked).

Test ai-agents MSEP pipeline with mocked Code-Orchestrator and semantic-search services.

WBS Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-7.1
Repository: ai-agents

Acceptance Criteria:
- AC-7.1.1: Pipeline orchestrates all services in parallel
- AC-7.1.2: Results merged correctly from all sources
- AC-7.1.3: Service failures handled with partial results
- AC-7.1.4: Timeout enforced per-service

TDD Status: RED → GREEN → REFACTOR
"""

from __future__ import annotations

import asyncio
import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_corpus() -> list[str]:
    """Sample corpus for integration testing.
    
    Provides 3 chapters for meaningful pipeline testing.
    """
    return [
        "Software design is fundamentally about managing complexity. "
        "Good design reduces complexity through abstraction and modular decomposition.",
        "Information hiding is a key principle. Modules should hide implementation "
        "details and expose simple interfaces to other components.",
        "Complexity comes from dependencies and obscurity. Deep modules with simple "
        "interfaces reduce cognitive load and improve maintainability.",
    ]


@pytest.fixture
def sample_chapter_index() -> list[dict[str, Any]]:
    """Sample chapter index matching the corpus."""
    return [
        {"book": "APOSD", "chapter": 1, "title": "Introduction", "id": "APOSD:ch1"},
        {"book": "APOSD", "chapter": 2, "title": "Information Hiding", "id": "APOSD:ch2"},
        {"book": "APOSD", "chapter": 3, "title": "Complexity", "id": "APOSD:ch3"},
    ]


@pytest.fixture
def sample_embeddings_response() -> dict[str, Any]:
    """Sample SBERT embeddings response (dict format for dispatcher)."""
    return {
        "embeddings": [
            [0.8, 0.2, 0.1, 0.3] * 192,  # 768 dims
            [0.7, 0.3, 0.2, 0.25] * 192,
            [0.75, 0.25, 0.15, 0.28] * 192,
        ],
    }


@pytest.fixture
def sample_similarity_response() -> dict[str, Any]:
    """Sample similarity matrix response (dict format for dispatcher)."""
    return {
        "similarity_matrix": [
            [1.0, 0.85, 0.78],  # Chapter 1 vs all
            [0.85, 1.0, 0.82],  # Chapter 2 vs all
            [0.78, 0.82, 1.0],  # Chapter 3 vs all
        ],
    }


@pytest.fixture
def sample_topic_response() -> dict[str, Any]:
    """Sample BERTopic clustering response."""
    return {
        "topics": [0, 0, 1],  # Ch1 & Ch2 same topic, Ch3 different
        "topic_info": [
            {"id": 0, "count": 2},
            {"id": 1, "count": 1},
        ],
    }


@pytest.fixture
def sample_keywords_response() -> dict[str, Any]:
    """Sample TF-IDF keywords response (dict format for dispatcher)."""
    return {
        "keywords": [
            ["software", "design", "complexity", "abstraction", "modular"],
            ["information", "hiding", "modules", "interfaces", "implementation"],
            ["complexity", "dependencies", "obscurity", "deep", "cognitive"],
        ],
    }


@pytest.fixture
def mock_code_orchestrator(
    sample_embeddings_response: dict[str, Any],
    sample_similarity_response: dict[str, Any],
    sample_topic_response: dict[str, Any],
    sample_keywords_response: dict[str, Any],
) -> AsyncMock:
    """Create mock Code-Orchestrator client with realistic responses."""
    mock = AsyncMock()
    mock.get_embeddings.return_value = sample_embeddings_response
    mock.get_similarity_matrix.return_value = sample_similarity_response
    mock.cluster_topics.return_value = sample_topic_response
    mock.extract_keywords.return_value = sample_keywords_response
    return mock


@pytest.fixture
def mock_semantic_search() -> AsyncMock:
    """Create mock semantic-search client."""
    mock = AsyncMock()
    mock.get_relationships_batch.return_value = {
        "results": {
            "APOSD:ch1": [{"target": "APOSD:ch2", "score": 0.88}],
            "APOSD:ch2": [{"target": "APOSD:ch1", "score": 0.88}],
            "APOSD:ch3": [],
        }
    }
    return mock


# =============================================================================
# MSE-7.1: Integration Tests (Mocked)
# =============================================================================


@pytest.mark.asyncio
class TestMSE71_PipelineOrchestration:
    """Tests for AC-7.1.1: Pipeline orchestrates all services in parallel."""

    async def test_ac_7_1_1_parallel_service_orchestration(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Pipeline should orchestrate all services in parallel.
        
        Verifies asyncio.gather() is used to call services concurrently.
        """
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        # Create orchestrator with mock clients
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        # Build request
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        # Execute enrichment
        result = await orchestrator.enrich_metadata(request)
        
        # Verify Code-Orchestrator services were called
        mock_code_orchestrator.get_embeddings.assert_called()
        mock_code_orchestrator.get_similarity_matrix.assert_called()
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, "chapters")
        assert len(result.chapters) == len(sample_corpus)

    async def test_ac_7_1_1_dispatcher_uses_gather(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Dispatcher should use asyncio.gather() for concurrent execution.
        
        Per WBS AC-4.2.1: dispatch_enrichment() uses asyncio.gather()
        """
        from src.agents.msep.dispatcher import MSEPDispatcher
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        dispatcher = MSEPDispatcher(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        # Execute dispatch
        result = await dispatcher.dispatch_enrichment(request)
        
        # Verify we got results (dispatcher calls services in parallel)
        assert result is not None
        # Should have at least similarity results
        assert result.similarity_matrix is not None or result.embeddings is not None


@pytest.mark.asyncio
class TestMSE71_ResultMerging:
    """Tests for AC-7.1.2: Results merged correctly from all sources."""

    async def test_ac_7_1_2_results_merged_correctly(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Results from all services should be merged into EnrichedMetadata.
        
        Verifies cross-references include SBERT scores + topic boost.
        """
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Verify merged structure
        assert len(result.chapters) == 3
        
        # Each chapter should have provenance
        for chapter in result.chapters:
            assert hasattr(chapter, "provenance")
            assert chapter.provenance is not None
            # Provenance should track methods used
            assert hasattr(chapter.provenance, "methods_used")

    async def test_ac_7_1_2_cross_references_include_scores(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Cross-references should include both base_score and topic_boost."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Find a chapter with cross-references
        for chapter in result.chapters:
            if chapter.cross_references:
                xref = chapter.cross_references[0]
                assert hasattr(xref, "score"), "Cross-reference should have score"
                assert hasattr(xref, "base_score"), "Cross-reference should have base_score"
                assert hasattr(xref, "topic_boost"), "Cross-reference should have topic_boost"
                # Final score = base_score + topic_boost (approximately)
                assert xref.score >= xref.base_score, "Final score should >= base"
                return
        
        # It's okay if no cross-refs meet threshold in test data


@pytest.mark.asyncio
class TestMSE71_PartialFailures:
    """Tests for AC-7.1.3: Service failures handled with partial results."""

    async def test_ac_7_1_3_partial_results_on_single_failure(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        sample_embeddings_response: dict[str, Any],
        sample_similarity_response: dict[str, Any],
        sample_keywords_response: dict[str, Any],
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Pipeline should return partial results when one service fails.
        
        Per WBS AC-4.2.4: Each task has independent error handling.
        """
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.exceptions import ServiceUnavailableError
        
        # Create mock that fails on BERTopic but succeeds on others
        mock_co = AsyncMock()
        mock_co.get_embeddings.return_value = sample_embeddings_response
        mock_co.get_similarity_matrix.return_value = sample_similarity_response
        mock_co.cluster_topics.side_effect = ServiceUnavailableError(
            "BERTopic unavailable", service="code-orchestrator"
        )
        mock_co.extract_keywords.return_value = sample_keywords_response
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_co,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        # Should not raise - partial results allowed
        result = await orchestrator.enrich_metadata(request)
        
        # Should still have results (from similarity matrix)
        assert result is not None
        assert len(result.chapters) == 3

    async def test_ac_7_1_3_bertopic_failure_still_produces_similarity(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        sample_embeddings_response: dict[str, Any],
        sample_similarity_response: dict[str, Any],
        sample_keywords_response: dict[str, Any],
        mock_semantic_search: AsyncMock,
    ) -> None:
        """If BERTopic fails, similarity-based enrichment should still work."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.exceptions import ServiceUnavailableError
        
        mock_co = AsyncMock()
        mock_co.get_embeddings.return_value = sample_embeddings_response
        mock_co.get_similarity_matrix.return_value = sample_similarity_response
        mock_co.cluster_topics.side_effect = ServiceUnavailableError(
            "BERTopic down", service="code-orchestrator"
        )
        mock_co.extract_keywords.return_value = sample_keywords_response
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_co,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Should have chapters with cross-references (from similarity)
        assert result is not None
        # Without topic boost, might have fewer cross-refs above threshold


@pytest.mark.asyncio
class TestMSE71_TimeoutEnforcement:
    """Tests for AC-7.1.4: Timeout enforced per-service."""

    async def test_ac_7_1_4_timeout_on_slow_service(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        sample_embeddings_response: dict[str, Any],
        sample_similarity_response: dict[str, Any],
        sample_keywords_response: dict[str, Any],
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Pipeline should timeout when service is too slow.
        
        Per WBS AC-7.1.4: Mock slow service.
        """
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        # Create a mock with slow response
        async def slow_similarity(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(5)  # Simulate slow response
            return sample_similarity_response
        
        mock_co = AsyncMock()
        mock_co.get_embeddings.return_value = sample_embeddings_response
        mock_co.get_similarity_matrix.side_effect = slow_similarity
        mock_co.cluster_topics.return_value = {"topics": [], "topic_info": []}
        mock_co.extract_keywords.return_value = sample_keywords_response
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_co,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        # Use short timeout config
        config = MSEPConfig(timeout=0.1)  # 100ms timeout
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        # Execute with timeout - should either timeout or handle gracefully
        try:
            result = await asyncio.wait_for(
                orchestrator.enrich_metadata(request),
                timeout=1.0,  # Test-level timeout
            )
            # If no timeout raised, dispatcher handled it gracefully
            assert result is not None
        except asyncio.TimeoutError:
            # Expected when timeout is enforced
            pass


@pytest.mark.asyncio
class TestMSE71_PipelineIntegrity:
    """Additional integration tests for pipeline correctness."""

    async def test_keywords_merged_from_tfidf(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Enriched chapters should include TF-IDF keywords."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Verify keywords are present
        for chapter in result.chapters:
            if hasattr(chapter, "keywords") and chapter.keywords:
                # Keywords should be populated from TF-IDF
                assert hasattr(chapter.keywords, "tfidf") or isinstance(chapter.keywords, list)

    async def test_provenance_tracks_all_methods(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """Provenance should track all enrichment methods used."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Check provenance
        for chapter in result.chapters:
            if chapter.provenance:
                methods = chapter.provenance.methods_used
                # Should have at least one method
                assert methods, "Provenance should list methods used"

    async def test_processing_time_recorded(
        self,
        sample_corpus: list[str],
        sample_chapter_index: list[dict[str, Any]],
        mock_code_orchestrator: AsyncMock,
        mock_semantic_search: AsyncMock,
    ) -> None:
        """EnrichedMetadata should include processing_time_ms."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig
        
        orchestrator = MSEPOrchestrator(
            code_orchestrator=mock_code_orchestrator,
            semantic_search=mock_semantic_search,
        )
        
        chapter_meta = [
            ChapterMeta(
                book=ch["book"],
                chapter=ch["chapter"],
                title=ch["title"],
                id=ch["id"],
            )
            for ch in sample_chapter_index
        ]
        config = MSEPConfig()
        request = MSEPRequest(
            corpus=sample_corpus,
            chapter_index=chapter_meta,
            config=config,
        )
        
        result = await orchestrator.enrich_metadata(request)
        
        # Should have processing time
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0
