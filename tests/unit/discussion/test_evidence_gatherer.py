"""Unit tests for EvidenceGatherer.

WBS Reference: WBS-KB3 - Iterative Evidence Gathering
Task: KB3.8 - Unit tests for EvidenceGatherer
Acceptance Criteria:
- AC-KB3.1: evidence_gatherer.gather() takes list[InformationRequest], returns evidence
- AC-KB3.2: Gatherer calls UnifiedRetriever with request queries
- AC-KB3.5: High-priority requests processed before medium/low
- AC-KB3.6: Gatherer respects source_types filter on requests

TDD Phase: RED - These tests will fail until implementation exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.discussion.models import CrossReferenceEvidence, InformationRequest
from src.discussion.evidence_gatherer import (
    EvidenceGatherer,
    EvidenceGathererConfig,
    GatherResult,
)
from src.schemas.retrieval_models import (
    RetrievalItem,
    RetrievalResult,
    RetrievalScope,
    SourceType,
)


# =============================================================================
# Test Constants
# =============================================================================

_CODE_QUERY = "Show ParallelAgent implementation"
_BOOK_QUERY = "Explain repository pattern"
_MIXED_QUERY = "How does caching work?"


# =============================================================================
# Fake UnifiedRetriever for Testing
# =============================================================================


@dataclass
class FakeUnifiedRetriever:
    """Fake retriever for testing EvidenceGatherer.
    
    Records all calls for verification and returns predetermined results.
    """
    
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, list[RetrievalItem]] = field(default_factory=dict)
    should_fail: bool = False
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Record call and return fake results."""
        self.calls.append({
            "query": query,
            "scope": scope,
            "top_k": top_k,
        })
        
        if self.should_fail:
            return RetrievalResult(
                query=query,
                results=[],
                total_count=0,
                citations=[],
                errors=["Retrieval failed"],
                scope=scope,
            )
        
        items = self.results.get(query, [])
        return RetrievalResult(
            query=query,
            results=items,
            total_count=len(items),
            citations=[],
            errors=[],
            scope=scope,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def code_item() -> RetrievalItem:
    """Create a code retrieval item."""
    return RetrievalItem(
        source_type=SourceType.CODE,
        source_id="src/pipelines/agents.py#L135",
        content="class ParallelAgent:\n    async def run(self):\n        return await asyncio.gather(*tasks)",
        relevance_score=0.95,
        title="ParallelAgent.run",
        metadata={"language": "python"},
    )


@pytest.fixture
def book_item() -> RetrievalItem:
    """Create a book retrieval item."""
    return RetrievalItem(
        source_type=SourceType.BOOK,
        source_id="ddia/ch3",
        content="The Repository pattern mediates between the domain and data mapping layers.",
        relevance_score=0.88,
        title="Design Patterns",
        metadata={"chapter": "3", "page": "42"},
    )


@pytest.fixture
def fake_retriever(code_item: RetrievalItem, book_item: RetrievalItem) -> FakeUnifiedRetriever:
    """Create fake retriever with predetermined results."""
    return FakeUnifiedRetriever(
        results={
            _CODE_QUERY: [code_item],
            _BOOK_QUERY: [book_item],
            _MIXED_QUERY: [code_item, book_item],
        }
    )


@pytest.fixture
def config() -> EvidenceGathererConfig:
    """Create default gatherer configuration."""
    return EvidenceGathererConfig()


@pytest.fixture
def gatherer(
    config: EvidenceGathererConfig,
    fake_retriever: FakeUnifiedRetriever,
) -> EvidenceGatherer:
    """Create gatherer with fake retriever."""
    return EvidenceGatherer(config=config, retriever=fake_retriever)


# =============================================================================
# AC-KB3.1: gather() takes list[InformationRequest], returns evidence
# =============================================================================


class TestGatherBasic:
    """Test basic gather functionality (AC-KB3.1)."""

    @pytest.mark.asyncio
    async def test_gather_returns_gather_result(
        self,
        gatherer: EvidenceGatherer,
    ) -> None:
        """gather() returns GatherResult dataclass."""
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        result = await gatherer.gather(requests)
        
        assert isinstance(result, GatherResult)

    @pytest.mark.asyncio
    async def test_gather_empty_requests_returns_empty_result(
        self,
        gatherer: EvidenceGatherer,
    ) -> None:
        """gather() with empty list returns empty GatherResult."""
        result = await gatherer.gather([])
        
        assert result.evidence == []
        assert result.total_items == 0

    @pytest.mark.asyncio
    async def test_gather_single_request_returns_evidence(
        self,
        gatherer: EvidenceGatherer,
    ) -> None:
        """gather() with single request returns evidence items."""
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        result = await gatherer.gather(requests)
        
        assert len(result.evidence) >= 1
        assert all(isinstance(e, CrossReferenceEvidence) for e in result.evidence)

    @pytest.mark.asyncio
    async def test_gather_multiple_requests_returns_combined_evidence(
        self,
        gatherer: EvidenceGatherer,
        code_item: RetrievalItem,
        book_item: RetrievalItem,
    ) -> None:
        """gather() with multiple requests returns combined evidence."""
        requests = [
            InformationRequest(query=_CODE_QUERY, source_types=["code"]),
            InformationRequest(query=_BOOK_QUERY, source_types=["books"]),
        ]
        
        result = await gatherer.gather(requests)
        
        assert len(result.evidence) >= 2
        source_types = {e.source_type for e in result.evidence}
        assert "code" in source_types or "book" in source_types


# =============================================================================
# AC-KB3.2: Gatherer calls UnifiedRetriever with request queries
# =============================================================================


class TestRetrieverIntegration:
    """Test gatherer calls retriever correctly (AC-KB3.2)."""

    @pytest.mark.asyncio
    async def test_gather_calls_retriever_with_query(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """gather() calls retriever with request query."""
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        await gatherer.gather(requests)
        
        assert len(fake_retriever.calls) >= 1
        assert fake_retriever.calls[0]["query"] == _CODE_QUERY

    @pytest.mark.asyncio
    async def test_gather_calls_retriever_for_each_request(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """gather() calls retriever for each information request."""
        requests = [
            InformationRequest(query=_CODE_QUERY, source_types=["code"]),
            InformationRequest(query=_BOOK_QUERY, source_types=["books"]),
        ]
        
        await gatherer.gather(requests)
        
        queries = [call["query"] for call in fake_retriever.calls]
        assert _CODE_QUERY in queries
        assert _BOOK_QUERY in queries


# =============================================================================
# AC-KB3.5: High-priority requests processed before medium/low
# =============================================================================


class TestPriorityOrdering:
    """Test priority-based request ordering (AC-KB3.5)."""

    @pytest.mark.asyncio
    async def test_high_priority_processed_first(
        self,
        config: EvidenceGathererConfig,
    ) -> None:
        """High priority requests processed before medium/low."""
        call_order: list[str] = []
        
        class TrackingRetriever:
            async def retrieve(
                self,
                query: str,
                scope: RetrievalScope = RetrievalScope.ALL,
                top_k: int = 10,
            ) -> RetrievalResult:
                call_order.append(query)
                return RetrievalResult(
                    query=query,
                    results=[],
                    total_count=0,
                    citations=[],
                    errors=[],
                    scope=scope,
                )
        
        gatherer = EvidenceGatherer(config=config, retriever=TrackingRetriever())
        
        # Create requests in wrong order (low, high, medium)
        requests = [
            InformationRequest(query="low_query", priority="low"),
            InformationRequest(query="high_query", priority="high"),
            InformationRequest(query="medium_query", priority="medium"),
        ]
        
        await gatherer.gather(requests)
        
        # High should be first, then medium, then low
        assert call_order.index("high_query") < call_order.index("medium_query")
        assert call_order.index("medium_query") < call_order.index("low_query")

    @pytest.mark.asyncio
    async def test_same_priority_preserves_order(
        self,
        config: EvidenceGathererConfig,
    ) -> None:
        """Requests with same priority preserve original order."""
        call_order: list[str] = []
        
        class TrackingRetriever:
            async def retrieve(
                self,
                query: str,
                scope: RetrievalScope = RetrievalScope.ALL,
                top_k: int = 10,
            ) -> RetrievalResult:
                call_order.append(query)
                return RetrievalResult(
                    query=query, results=[], total_count=0,
                    citations=[], errors=[], scope=scope,
                )
        
        gatherer = EvidenceGatherer(config=config, retriever=TrackingRetriever())
        
        requests = [
            InformationRequest(query="high_1", priority="high"),
            InformationRequest(query="high_2", priority="high"),
            InformationRequest(query="high_3", priority="high"),
        ]
        
        await gatherer.gather(requests)
        
        assert call_order == ["high_1", "high_2", "high_3"]


# =============================================================================
# AC-KB3.6: Gatherer respects source_types filter on requests
# =============================================================================


class TestSourceTypeFiltering:
    """Test source_types filtering (AC-KB3.6)."""

    @pytest.mark.asyncio
    async def test_code_only_request_uses_code_scope(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """Code-only request passes CODE_ONLY scope to retriever."""
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        await gatherer.gather(requests)
        
        assert fake_retriever.calls[0]["scope"] == RetrievalScope.CODE_ONLY

    @pytest.mark.asyncio
    async def test_books_only_request_uses_books_scope(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """Books-only request passes BOOKS_ONLY scope to retriever."""
        requests = [
            InformationRequest(query=_BOOK_QUERY, source_types=["books", "textbooks"])
        ]
        
        await gatherer.gather(requests)
        
        assert fake_retriever.calls[0]["scope"] == RetrievalScope.BOOKS_ONLY

    @pytest.mark.asyncio
    async def test_mixed_sources_uses_all_scope(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """Mixed source_types request uses ALL scope."""
        requests = [
            InformationRequest(query=_MIXED_QUERY, source_types=["code", "books"])
        ]
        
        await gatherer.gather(requests)
        
        assert fake_retriever.calls[0]["scope"] == RetrievalScope.ALL

    @pytest.mark.asyncio
    async def test_graph_only_request_uses_graph_scope(
        self,
        gatherer: EvidenceGatherer,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """Graph-only request passes GRAPH_ONLY scope to retriever."""
        requests = [
            InformationRequest(query="concept relationships", source_types=["graph"])
        ]
        
        await gatherer.gather(requests)
        
        assert fake_retriever.calls[0]["scope"] == RetrievalScope.GRAPH_ONLY


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in gather()."""

    @pytest.mark.asyncio
    async def test_gather_handles_retriever_error(
        self,
        config: EvidenceGathererConfig,
    ) -> None:
        """gather() handles retriever errors gracefully."""
        failing_retriever = FakeUnifiedRetriever(should_fail=True)
        gatherer = EvidenceGatherer(config=config, retriever=failing_retriever)
        
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        result = await gatherer.gather(requests)
        
        # Should not raise, just return empty/partial results
        assert isinstance(result, GatherResult)
        assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_gather_continues_after_single_failure(
        self,
        config: EvidenceGathererConfig,
        book_item: RetrievalItem,
    ) -> None:
        """gather() continues processing after one request fails."""
        call_count = 0
        
        class MixedRetriever:
            async def retrieve(
                self,
                query: str,
                scope: RetrievalScope = RetrievalScope.ALL,
                top_k: int = 10,
            ) -> RetrievalResult:
                nonlocal call_count
                call_count += 1
                # First call fails, second succeeds
                if call_count == 1:
                    return RetrievalResult(
                        query=query, results=[], total_count=0,
                        citations=[], errors=["Failed"], scope=scope,
                    )
                return RetrievalResult(
                    query=query, results=[book_item], total_count=1,
                    citations=[], errors=[], scope=scope,
                )
        
        gatherer = EvidenceGatherer(config=config, retriever=MixedRetriever())
        
        requests = [
            InformationRequest(query="failing_query", priority="high"),
            InformationRequest(query=_BOOK_QUERY, priority="medium"),
        ]
        
        result = await gatherer.gather(requests)
        
        # Should have evidence from second request
        assert len(result.evidence) >= 1


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Test EvidenceGathererConfig."""

    def test_default_config_values(self) -> None:
        """Default config has expected values."""
        config = EvidenceGathererConfig()
        
        assert config.max_results_per_request > 0
        assert config.timeout_seconds > 0

    def test_custom_config_values(self) -> None:
        """Custom config values are applied."""
        config = EvidenceGathererConfig(
            max_results_per_request=5,
            timeout_seconds=30.0,
        )
        
        assert config.max_results_per_request == 5
        assert config.timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_max_results_limits_retrieval(
        self,
        fake_retriever: FakeUnifiedRetriever,
    ) -> None:
        """max_results_per_request is passed to retriever."""
        config = EvidenceGathererConfig(max_results_per_request=3)
        gatherer = EvidenceGatherer(config=config, retriever=fake_retriever)
        
        requests = [InformationRequest(query=_CODE_QUERY, source_types=["code"])]
        
        await gatherer.gather(requests)
        
        assert fake_retriever.calls[0]["top_k"] == 3


# =============================================================================
# GatherResult Tests
# =============================================================================


class TestGatherResult:
    """Test GatherResult dataclass."""

    def test_gather_result_serialization(self) -> None:
        """GatherResult can be converted to dict."""
        result = GatherResult(
            evidence=[
                CrossReferenceEvidence(
                    source_type="code",
                    content="test content",
                    source_id="test.py",
                )
            ],
            total_items=1,
            requests_processed=1,
            errors=[],
        )
        
        data = result.to_dict()
        
        assert data["total_items"] == 1
        assert data["requests_processed"] == 1
        assert len(data["evidence"]) == 1
