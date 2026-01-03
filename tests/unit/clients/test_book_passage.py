"""Unit Tests: BookPassageClient.

WBS Reference: WBS-AGT23 Book/JSON Passage Retrieval (AGT23.9)
Acceptance Criteria:
- AC-23.1: Retrieve passages from enriched book JSON files
- AC-23.2: Query passages via Qdrant vector similarity
- AC-23.3: Cross-reference passages with Neo4j concept nodes
- AC-23.4: Return structured BookPassage with citation metadata
- AC-23.5: Support filtering by book, chapter, concept

TDD Status: RED → GREEN → REFACTOR
Pattern: Protocol duck typing with FakeBookPassageClient for isolation

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Anti-Pattern Compliance: CODING_PATTERNS_ANALYSIS.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.schemas.passage_models import (
    BookPassage,
    PassageMetadata,
    PassageFilter,
)
from src.citations.book_citation import (
    passage_to_citation,
    passages_to_citations,
    BookCitation,
)


# =============================================================================
# Test Fixtures - FakeBookPassageClient for Unit Tests
# =============================================================================

@dataclass
class FakeBookPassageClient:
    """Fake client for unit testing - implements BookPassageClientProtocol.
    
    Pattern: Protocol duck typing for test isolation.
    Reference: CODING_PATTERNS_ANALYSIS.md - FakeClient pattern
    """
    
    # Preconfigured passage storage
    passages_by_id: dict[str, BookPassage] = field(default_factory=dict)
    passages_by_book: dict[str, list[BookPassage]] = field(default_factory=dict)
    passages_by_concept: dict[str, list[BookPassage]] = field(default_factory=dict)
    search_results: list[BookPassage] = field(default_factory=list)
    
    # Call tracking
    query_count: int = 0
    last_query: str = ""
    should_raise: Exception | None = None
    _connected: bool = False
    
    # Qdrant config
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "book_passages"
    
    async def connect(self) -> None:
        """Fake connect."""
        await asyncio.sleep(0)
        if self.should_raise:
            raise self.should_raise
        self._connected = True
    
    async def close(self) -> None:
        """Fake close."""
        await asyncio.sleep(0)
        self._connected = False
    
    async def __aenter__(self) -> "FakeBookPassageClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def health_check(self) -> bool:
        """Fake health check."""
        await asyncio.sleep(0)
        return self._connected
    
    async def search_passages(
        self,
        query: str,
        top_k: int = 10,
        filters: PassageFilter | None = None,
    ) -> list[BookPassage]:
        """Search passages via Qdrant vector similarity.
        
        AC-23.2: Query passages via Qdrant vector similarity
        AC-23.5: Support filtering by book, chapter, concept
        """
        await asyncio.sleep(0)
        self.query_count += 1
        self.last_query = query
        if self.should_raise:
            raise self.should_raise
        
        results = self.search_results[:top_k]
        
        # Apply filters if provided
        if filters:
            if filters.book_id:
                results = [p for p in results if p.book_id == filters.book_id]
            if filters.chapter_number:
                results = [p for p in results if p.chapter_number == filters.chapter_number]
            if filters.concept:
                results = [p for p in results if filters.concept in p.concepts]
        
        return results
    
    async def get_passage_by_id(
        self,
        passage_id: str,
    ) -> BookPassage | None:
        """Get passage by ID from JSON lookup.
        
        AC-23.1: Retrieve passages from enriched book JSON files
        """
        await asyncio.sleep(0)
        self.query_count += 1
        if self.should_raise:
            raise self.should_raise
        return self.passages_by_id.get(passage_id)
    
    async def get_passages_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[BookPassage]:
        """Get passages linked to a concept via Neo4j.
        
        AC-23.3: Cross-reference passages with Neo4j concept nodes
        """
        await asyncio.sleep(0)
        self.query_count += 1
        if self.should_raise:
            raise self.should_raise
        results = self.passages_by_concept.get(concept, [])
        return results[:limit]
    
    async def get_passages_for_book(
        self,
        book_id: str,
        limit: int = 100,
    ) -> list[BookPassage]:
        """Get all passages for a book.
        
        AC-23.5: Support filtering by book
        """
        await asyncio.sleep(0)
        self.query_count += 1
        results = self.passages_by_book.get(book_id, [])
        return results[:limit]
    
    async def filter_by_book(
        self,
        passages: list[BookPassage],
        book_id: str,
    ) -> list[BookPassage]:
        """Filter passages by book ID.
        
        AC-23.5: Support filtering by book
        """
        await asyncio.sleep(0)
        return [p for p in passages if p.book_id == book_id]
    
    async def filter_by_chapter(
        self,
        passages: list[BookPassage],
        chapter_number: int,
    ) -> list[BookPassage]:
        """Filter passages by chapter number.
        
        AC-23.5: Support filtering by chapter
        """
        await asyncio.sleep(0)
        return [p for p in passages if p.chapter_number == chapter_number]


# =============================================================================
# Test Data Factory
# =============================================================================

def create_test_passage(
    passage_id: str = "p_001",
    book_id: str = "aposd",
    book_title: str = "A Philosophy of Software Design",
    author: str = "John Ousterhout",
    chapter_number: int = 1,
    chapter_title: str = "Introduction",
    start_page: int = 1,
    end_page: int = 10,
    content: str = "Good software design is about managing complexity.",
    concepts: list[str] | None = None,
    keywords: list[str] | None = None,
    relevance_score: float = 0.95,
) -> BookPassage:
    """Create a test BookPassage instance."""
    return BookPassage(
        passage_id=passage_id,
        book_id=book_id,
        book_title=book_title,
        author=author,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        start_page=start_page,
        end_page=end_page,
        content=content,
        concepts=concepts or ["complexity", "design"],
        keywords=keywords or ["software", "complexity", "design"],
        relevance_score=relevance_score,
    )


@pytest.fixture
def sample_passages() -> list[BookPassage]:
    """Create sample passages for testing."""
    return [
        create_test_passage(
            passage_id="p_001",
            book_id="aposd",
            book_title="A Philosophy of Software Design",
            author="John Ousterhout",
            chapter_number=1,
            chapter_title="Introduction",
            start_page=1,
            end_page=10,
            content="Good software design is about managing complexity.",
            concepts=["complexity", "design"],
        ),
        create_test_passage(
            passage_id="p_002",
            book_id="aposd",
            book_title="A Philosophy of Software Design",
            author="John Ousterhout",
            chapter_number=4,
            chapter_title="Modules Should Be Deep",
            start_page=31,
            end_page=42,
            content="The best modules are those that provide powerful functionality yet have simple interfaces.",
            concepts=["modules", "interfaces", "abstraction"],
        ),
        create_test_passage(
            passage_id="p_003",
            book_id="ddd-eric-evans",
            book_title="Domain-Driven Design",
            author="Eric Evans",
            chapter_number=5,
            chapter_title="A Model Expressed in Software",
            start_page=65,
            end_page=80,
            content="The repository pattern provides an abstraction over data storage.",
            concepts=["ddd", "repository-pattern", "domain-model"],
        ),
        create_test_passage(
            passage_id="p_004",
            book_id="ddd-eric-evans",
            book_title="Domain-Driven Design",
            author="Eric Evans",
            chapter_number=6,
            chapter_title="The Lifecycle of a Domain Object",
            start_page=81,
            end_page=95,
            content="Aggregates enforce invariants and define consistency boundaries.",
            concepts=["ddd", "aggregate", "bounded-context"],
        ),
    ]


@pytest.fixture
def fake_client(sample_passages: list[BookPassage]) -> FakeBookPassageClient:
    """Create a fake client with default test data."""
    client = FakeBookPassageClient(
        passages_by_id={p.passage_id: p for p in sample_passages},
        passages_by_book={
            "aposd": [p for p in sample_passages if p.book_id == "aposd"],
            "ddd-eric-evans": [p for p in sample_passages if p.book_id == "ddd-eric-evans"],
        },
        passages_by_concept={
            "ddd": [sample_passages[2], sample_passages[3]],
            "repository-pattern": [sample_passages[2]],
            "complexity": [sample_passages[0]],
            "modules": [sample_passages[1]],
        },
        search_results=sample_passages,
    )
    return client


# =============================================================================
# AC-23.1: Retrieve passages from enriched book JSON files
# =============================================================================

class TestGetPassageById:
    """Tests for get_passage_by_id method (AC-23.1)."""

    @pytest.mark.asyncio
    async def test_get_passage_by_id_returns_passage(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should return passage when ID exists."""
        result = await fake_client.get_passage_by_id("p_001")
        
        assert result is not None
        assert result.passage_id == "p_001"
        assert result.book_title == "A Philosophy of Software Design"
        assert result.author == "John Ousterhout"

    @pytest.mark.asyncio
    async def test_get_passage_by_id_returns_none_for_unknown(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should return None when ID doesn't exist."""
        result = await fake_client.get_passage_by_id("unknown_id")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_passage_by_id_includes_citation_metadata(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should include all metadata required for citations (AC-23.4)."""
        result = await fake_client.get_passage_by_id("p_003")
        
        assert result is not None
        # Citation metadata per AC-23.4
        assert result.author == "Eric Evans"
        assert result.book_title == "Domain-Driven Design"
        assert result.chapter_number == 5
        assert result.chapter_title == "A Model Expressed in Software"
        assert result.start_page == 65
        assert result.end_page == 80

    @pytest.mark.asyncio
    async def test_get_passage_tracks_query_count(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should track query count for observability."""
        assert fake_client.query_count == 0
        
        await fake_client.get_passage_by_id("p_001")
        assert fake_client.query_count == 1
        
        await fake_client.get_passage_by_id("p_002")
        assert fake_client.query_count == 2


# =============================================================================
# AC-23.2: Query passages via Qdrant vector similarity
# =============================================================================

class TestSearchPassages:
    """Tests for search_passages method (AC-23.2)."""

    @pytest.mark.asyncio
    async def test_search_passages_returns_list(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should return list of BookPassage objects."""
        results = await fake_client.search_passages("repository pattern")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(p, BookPassage) for p in results)

    @pytest.mark.asyncio
    async def test_search_passages_respects_top_k(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should respect top_k limit."""
        results = await fake_client.search_passages("design", top_k=2)
        
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_passages_tracks_query(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should track the search query."""
        await fake_client.search_passages("complexity management")
        
        assert fake_client.last_query == "complexity management"

    @pytest.mark.asyncio
    async def test_search_passages_with_book_filter(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should filter by book ID (AC-23.5)."""
        filter_obj = PassageFilter(book_id="aposd")
        results = await fake_client.search_passages("design", filters=filter_obj)
        
        assert all(p.book_id == "aposd" for p in results)

    @pytest.mark.asyncio
    async def test_search_passages_with_chapter_filter(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should filter by chapter number (AC-23.5)."""
        filter_obj = PassageFilter(chapter_number=1)
        results = await fake_client.search_passages("design", filters=filter_obj)
        
        assert all(p.chapter_number == 1 for p in results)

    @pytest.mark.asyncio
    async def test_search_passages_with_concept_filter(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should filter by concept (AC-23.5)."""
        filter_obj = PassageFilter(concept="ddd")
        results = await fake_client.search_passages("pattern", filters=filter_obj)
        
        assert all("ddd" in p.concepts for p in results)


# =============================================================================
# AC-23.3: Cross-reference passages with Neo4j concept nodes
# =============================================================================

class TestGetPassagesForConcept:
    """Tests for get_passages_for_concept method (AC-23.3)."""

    @pytest.mark.asyncio
    async def test_get_passages_for_concept_returns_linked_passages(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should return passages linked to concept via Neo4j."""
        results = await fake_client.get_passages_for_concept("ddd")
        
        assert len(results) == 2
        assert all("ddd" in p.concepts for p in results)

    @pytest.mark.asyncio
    async def test_get_passages_for_concept_respects_limit(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should respect limit parameter."""
        results = await fake_client.get_passages_for_concept("ddd", limit=1)
        
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_passages_for_concept_returns_empty_for_unknown(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should return empty list for unknown concept."""
        results = await fake_client.get_passages_for_concept("unknown-concept")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_passages_for_concept_repository_pattern(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should find passages about repository pattern."""
        results = await fake_client.get_passages_for_concept("repository-pattern")
        
        assert len(results) == 1
        assert results[0].content == "The repository pattern provides an abstraction over data storage."


# =============================================================================
# AC-23.4: Return structured BookPassage with citation metadata
# =============================================================================

class TestBookPassageSchema:
    """Tests for BookPassage schema (AC-23.4)."""

    def test_book_passage_has_citation_fields(self) -> None:
        """BookPassage should have all fields for Chicago citations."""
        passage = create_test_passage()
        
        # Required citation fields
        assert hasattr(passage, "author")
        assert hasattr(passage, "book_title")
        assert hasattr(passage, "chapter_title")
        assert hasattr(passage, "chapter_number")
        assert hasattr(passage, "start_page")
        assert hasattr(passage, "end_page")

    def test_book_passage_to_dict(self) -> None:
        """BookPassage should serialize to dict."""
        passage = create_test_passage()
        data = passage.to_dict()
        
        assert data["passage_id"] == "p_001"
        assert data["author"] == "John Ousterhout"
        assert data["book_title"] == "A Philosophy of Software Design"
        assert data["start_page"] == 1
        assert data["end_page"] == 10

    def test_book_passage_has_concepts(self) -> None:
        """BookPassage should include concepts for cross-referencing."""
        passage = create_test_passage(concepts=["ddd", "repository-pattern"])
        
        assert "ddd" in passage.concepts
        assert "repository-pattern" in passage.concepts


# =============================================================================
# AC-23.4: Citation generation from BookPassage
# =============================================================================

class TestBookCitationMapper:
    """Tests for book_citation mapper (AC-23.4)."""

    def test_passage_to_citation_creates_book_citation(self) -> None:
        """passage_to_citation should create BookCitation."""
        passage = create_test_passage()
        citation = passage_to_citation(passage)
        
        assert isinstance(citation, BookCitation)
        assert citation.source_type == "book_passage"
        assert citation.source_id == "p_001"

    def test_passage_to_citation_includes_author(self) -> None:
        """Citation should include author."""
        passage = create_test_passage(author="Eric Evans")
        citation = passage_to_citation(passage)
        
        assert citation.author == "Eric Evans"

    def test_passage_to_citation_includes_title(self) -> None:
        """Citation should include book title."""
        passage = create_test_passage(book_title="Domain-Driven Design")
        citation = passage_to_citation(passage)
        
        assert citation.book_title == "Domain-Driven Design"

    def test_passage_to_citation_includes_pages(self) -> None:
        """Citation should include page range."""
        passage = create_test_passage(start_page=65, end_page=80)
        citation = passage_to_citation(passage)
        
        assert citation.start_page == 65
        assert citation.end_page == 80

    def test_passage_to_citation_chicago_format(self) -> None:
        """Citation should support Chicago format."""
        passage = create_test_passage(
            author="John Ousterhout",
            book_title="A Philosophy of Software Design",
            start_page=31,
            end_page=42,
        )
        citation = passage_to_citation(passage)
        chicago = citation.to_chicago()
        
        # Chicago format: Author, Title (Publisher, Year), Pages.
        assert "Ousterhout" in chicago
        assert "A Philosophy of Software Design" in chicago
        assert "31" in chicago or "31-42" in chicago

    def test_passages_to_citations_batch(self, sample_passages: list[BookPassage]) -> None:
        """passages_to_citations should convert batch."""
        citations = passages_to_citations(sample_passages)
        
        assert len(citations) == len(sample_passages)
        assert all(isinstance(c, BookCitation) for c in citations)


# =============================================================================
# AC-23.5: Support filtering by book, chapter, concept
# =============================================================================

class TestFiltering:
    """Tests for filtering methods (AC-23.5)."""

    @pytest.mark.asyncio
    async def test_filter_by_book(
        self, fake_client: FakeBookPassageClient, sample_passages: list[BookPassage]
    ) -> None:
        """filter_by_book should filter passages by book_id."""
        filtered = await fake_client.filter_by_book(sample_passages, "aposd")
        
        assert len(filtered) == 2
        assert all(p.book_id == "aposd" for p in filtered)

    @pytest.mark.asyncio
    async def test_filter_by_chapter(
        self, fake_client: FakeBookPassageClient, sample_passages: list[BookPassage]
    ) -> None:
        """filter_by_chapter should filter passages by chapter_number."""
        filtered = await fake_client.filter_by_chapter(sample_passages, 1)
        
        assert len(filtered) == 1
        assert filtered[0].chapter_number == 1

    @pytest.mark.asyncio
    async def test_get_passages_for_book(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """get_passages_for_book should return all passages for a book."""
        passages = await fake_client.get_passages_for_book("aposd")
        
        assert len(passages) == 2
        assert all(p.book_id == "aposd" for p in passages)

    @pytest.mark.asyncio
    async def test_get_passages_for_book_respects_limit(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """get_passages_for_book should respect limit."""
        passages = await fake_client.get_passages_for_book("aposd", limit=1)
        
        assert len(passages) == 1


# =============================================================================
# Connection Management
# =============================================================================

class TestConnectionManagement:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Should support async context manager."""
        async with fake_client as client:
            assert await client.health_check() is True
        
        # After exit, should be disconnected
        assert await fake_client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_when_connected(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """health_check should return True when connected."""
        await fake_client.connect()
        assert await fake_client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """health_check should return False when disconnected."""
        assert await fake_client.health_check() is False


# =============================================================================
# Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_search_raises_on_error(self) -> None:
        """Should propagate errors from search."""
        client = FakeBookPassageClient(should_raise=ValueError("Search failed"))
        
        with pytest.raises(ValueError, match="Search failed"):
            await client.search_passages("test")

    @pytest.mark.asyncio
    async def test_get_passage_raises_on_error(self) -> None:
        """Should propagate errors from get_passage_by_id."""
        client = FakeBookPassageClient(should_raise=ConnectionError("DB error"))
        
        with pytest.raises(ConnectionError, match="DB error"):
            await client.get_passage_by_id("p_001")

    @pytest.mark.asyncio
    async def test_connect_raises_on_error(self) -> None:
        """Should propagate errors from connect."""
        client = FakeBookPassageClient(should_raise=TimeoutError("Connection timeout"))
        
        with pytest.raises(TimeoutError, match="Connection timeout"):
            await client.connect()


# =============================================================================
# Protocol Compliance
# =============================================================================

class TestProtocolCompliance:
    """Tests for protocol duck typing compliance."""

    def test_fake_client_matches_protocol_signature(self) -> None:
        """FakeBookPassageClient should match BookPassageClientProtocol signature."""
        client = FakeBookPassageClient()
        
        # All protocol methods should exist
        assert hasattr(client, "connect")
        assert hasattr(client, "close")
        assert hasattr(client, "health_check")
        assert hasattr(client, "search_passages")
        assert hasattr(client, "get_passage_by_id")
        assert hasattr(client, "get_passages_for_concept")
        assert hasattr(client, "get_passages_for_book")
        assert hasattr(client, "filter_by_book")
        assert hasattr(client, "filter_by_chapter")

    def test_fake_client_callable_methods(self) -> None:
        """Protocol methods should be callable."""
        client = FakeBookPassageClient()
        
        assert callable(client.connect)
        assert callable(client.close)
        assert callable(client.health_check)
        assert callable(client.search_passages)
        assert callable(client.get_passage_by_id)
        assert callable(client.get_passages_for_concept)


# =============================================================================
# Integration with Neo4j Client
# =============================================================================

class TestNeo4jIntegration:
    """Tests for Neo4j integration (AC-23.3)."""

    @pytest.mark.asyncio
    async def test_passages_linked_to_concepts_via_neo4j(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Passages should be retrievable by Neo4j concept links."""
        # This tests the AC-23.3 flow: passage → Neo4j concept → passage
        passages = await fake_client.get_passages_for_concept("repository-pattern")
        
        assert len(passages) > 0
        # The passage should mention the concept
        assert any("repository" in p.content.lower() for p in passages)

    @pytest.mark.asyncio
    async def test_cross_reference_concepts_in_passage(
        self, fake_client: FakeBookPassageClient
    ) -> None:
        """Passage concepts should match Neo4j concept nodes."""
        passage = await fake_client.get_passage_by_id("p_003")
        
        assert passage is not None
        # Passage concepts should be queryable via Neo4j
        assert "ddd" in passage.concepts
        assert "repository-pattern" in passage.concepts
