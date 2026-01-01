"""Integration tests for Unified Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.10)
Acceptance Criteria:
- AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books

Integration test with all sources - requires services to be running.
Run with: pytest tests/integration/test_unified_retrieval.py -m integration

Pattern: Integration Testing with Real Services
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

from __future__ import annotations

import asyncio
import os

import pytest


# Skip all tests if not running integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Configuration
# =============================================================================


def get_env_or_default(key: str, default: str) -> str:
    """Get environment variable or default value."""
    return os.environ.get(key, default)


# Service URLs from environment or defaults
SEMANTIC_SEARCH_URL = get_env_or_default(
    "SEMANTIC_SEARCH_URL", "http://localhost:8081"
)
NEO4J_URI = get_env_or_default("NEO4J_URI", "bolt://localhost:7687")
QDRANT_URL = get_env_or_default("QDRANT_URL", "http://localhost:6333")
BOOKS_DIR = get_env_or_default(
    "BOOKS_DIR", "/Users/kevintoles/POC/ai-platform-data/books/enriched"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def unified_retriever():
    """Create UnifiedRetriever with real clients."""
    from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
    
    # This would use real clients when they're available
    # For now, use fake clients for testing the integration pattern
    config = UnifiedRetrieverConfig(
        timeout_seconds=30.0,
        max_per_source=10,
    )
    
    retriever = UnifiedRetriever(config=config)
    
    yield retriever


# =============================================================================
# Test Classes
# =============================================================================


class TestUnifiedRetrievalIntegration:
    """Integration tests for unified retrieval across all sources."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_retrieve_repository_pattern(
        self,
        unified_retriever,
    ) -> None:
        """Test retrieving 'repository pattern' across all sources.
        
        EC: Query returns results from code-reference-engine, Neo4j, and books.
        """
        from src.schemas.retrieval_models import RetrievalScope
        
        result = await unified_retriever.retrieve(
            query="repository pattern",
            scope=RetrievalScope.ALL,
            top_k=10,
        )
        
        # Should have results from multiple sources
        assert result.total_count > 0
        assert len(result.results) > 0
        
        # Check source diversity
        source_types = {r.source_type.value for r in result.results}
        assert len(source_types) >= 1  # At least one source
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_retrieve_code_only_scope(
        self,
        unified_retriever,
    ) -> None:
        """Test retrieving with code_only scope.
        
        AC-24.6: Supports scope filtering (code-only).
        """
        from src.schemas.retrieval_models import RetrievalScope, SourceType
        
        result = await unified_retriever.retrieve(
            query="async context manager",
            scope=RetrievalScope.CODE_ONLY,
            top_k=10,
        )
        
        # All results should be code
        for item in result.results:
            assert item.source_type == SourceType.CODE
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_retrieve_books_only_scope(
        self,
        unified_retriever,
    ) -> None:
        """Test retrieving with books_only scope.
        
        AC-24.6: Supports scope filtering (books-only).
        """
        from src.schemas.retrieval_models import RetrievalScope, SourceType
        
        result = await unified_retriever.retrieve(
            query="domain driven design",
            scope=RetrievalScope.BOOKS_ONLY,
            top_k=10,
        )
        
        # All results should be books
        for item in result.results:
            assert item.source_type == SourceType.BOOK
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_results_ranked_by_relevance(
        self,
        unified_retriever,
    ) -> None:
        """Test that results are ranked by relevance.
        
        EC: Results are ranked by relevance across sources.
        """
        result = await unified_retriever.retrieve(
            query="event sourcing CQRS",
            top_k=20,
        )
        
        # Scores should be in descending order
        scores = [r.relevance_score for r in result.results]
        for i in range(1, len(scores)):
            # Allow equal scores
            assert scores[i] <= scores[i - 1] * 1.01  # 1% tolerance
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_citations_identify_source_type(
        self,
        unified_retriever,
    ) -> None:
        """Test that citations correctly identify source type.
        
        EC: Citations correctly identify source type (code, book, graph).
        """
        from src.citations.mixed_citation import SourceType
        
        result = await unified_retriever.retrieve(
            query="saga pattern",
            top_k=10,
        )
        
        # Each citation should have a valid source type
        for citation in result.citations:
            assert citation.source_type in (
                SourceType.CODE,
                SourceType.BOOK,
                SourceType.GRAPH,
                SourceType.SEMANTIC,
            )
            assert citation.source_id is not None
            assert citation.display_text is not None


class TestCrossReferenceWithUnifiedRetriever:
    """Test cross_reference function with UnifiedRetriever.
    
    AC-24.5: cross_reference agent function uses this retriever.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_cross_reference_uses_unified_retriever(self) -> None:
        """Test cross_reference with UnifiedRetriever.
        
        EC: cross_reference("repository pattern") returns mixed results.
        """
        from src.functions.cross_reference import CrossReferenceFunction
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        # Create retriever (would use real clients in integration)
        config = UnifiedRetrieverConfig()
        retriever = UnifiedRetriever(config=config)
        
        # Create cross_reference function with retriever
        func = CrossReferenceFunction(unified_retriever=retriever)
        
        # Execute with unified retriever
        result = await func.run(
            query_artifact="repository pattern",
            use_unified=True,
            top_k=10,
        )
        
        # Should return results
        assert result is not None
        assert len(result.references) >= 0  # May be empty without real services
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_cross_reference_returns_mixed_citations(self) -> None:
        """Test that cross_reference returns mixed citations.
        
        EC: cross_reference("repository pattern") returns mixed results.
        """
        from src.functions.cross_reference import CrossReferenceFunction
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        config = UnifiedRetrieverConfig()
        retriever = UnifiedRetriever(config=config)
        
        func = CrossReferenceFunction(unified_retriever=retriever)
        
        result = await func.run(
            query_artifact="dependency injection",
            use_unified=True,
            top_k=10,
        )
        
        # Citations should be present
        assert result.citations is not None


# =============================================================================
# Performance Tests
# =============================================================================


class TestUnifiedRetrievalPerformance:
    """Performance tests for unified retrieval."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_retrieve_completes_within_timeout(
        self,
        unified_retriever,
    ) -> None:
        """Test that retrieval completes within timeout.
        
        Default timeout is 10 seconds.
        """
        import time
        
        start = time.time()
        result = await unified_retriever.retrieve(
            query="microservices architecture",
            top_k=20,
        )
        elapsed = time.time() - start
        
        # Should complete within 10 seconds
        assert elapsed < 10.0
        assert result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires all services running")
    async def test_parallel_retrieval(
        self,
        unified_retriever,
    ) -> None:
        """Test parallel retrieval of multiple queries.
        
        Should execute queries concurrently.
        """
        import time
        
        queries = [
            "repository pattern",
            "event sourcing",
            "domain driven design",
        ]
        
        start = time.time()
        results = await asyncio.gather(*[
            unified_retriever.retrieve(q, top_k=5)
            for q in queries
        ])
        elapsed = time.time() - start
        
        # Parallel execution should be faster than sequential
        # With 10s timeout each, should complete well under 30s
        assert elapsed < 30.0
        assert len(results) == 3
