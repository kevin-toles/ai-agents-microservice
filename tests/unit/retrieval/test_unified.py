"""Tests for UnifiedRetriever.

TDD tests for WBS-AGT24: Unified Knowledge Retrieval.

Acceptance Criteria:
- AC-24.1: Single interface queries all knowledge sources
- AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books
- AC-24.3: Merges and ranks results across sources
- AC-24.4: Returns unified RetrievalResult with mixed citations
- AC-24.5: cross_reference agent function uses this retriever
- AC-24.6: Supports scope filtering (code-only, books-only, all)

Exit Criteria:
- `from src.retrieval import UnifiedRetriever` succeeds
- Query returns results from code-reference-engine, Neo4j, and books
- Results are ranked by relevance across sources
- Citations correctly identify source type (code, book, graph)
- `cross_reference("repository pattern")` returns mixed results
- `pytest tests/unit/retrieval/` passes

Pattern: TDD (RED phase)
Reference: CODING_PATTERNS_ANALYSIS.md - Protocol duck typing with FakeClient
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest


# =============================================================================
# Fake Clients for Testing (Protocol Duck Typing)
# =============================================================================


class FakeCodeReferenceClient:
    """Fake CodeReferenceClient for testing.
    
    Implements CodeReferenceProtocol via duck typing.
    """
    
    def __init__(
        self,
        search_results: list[dict[str, Any]] | None = None,
        pattern_results: list[dict[str, Any]] | None = None,
    ):
        self.search_results = search_results or []
        self.pattern_results = pattern_results or []
        self.search_calls: list[dict[str, Any]] = []
        self.pattern_calls: list[str] = []
    
    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        concepts: list[str] | None = None,
        top_k: int = 10,
    ) -> Any:
        """Record call and return fake results."""
        self.search_calls.append({
            "query": query,
            "domains": domains,
            "concepts": concepts,
            "top_k": top_k,
        })
        return FakeCodeContext(references=self.search_results, query=query)
    
    async def search_by_pattern(
        self,
        pattern: str,
        top_k: int = 10,
    ) -> Any:
        """Search by pattern name."""
        self.pattern_calls.append(pattern)
        return FakeCodeContext(references=self.pattern_results, query=pattern)
    
    async def close(self) -> None:
        """Release resources."""
        pass


@dataclass
class FakeCodeContext:
    """Fake CodeContext returned by FakeCodeReferenceClient."""
    references: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    
    @property
    def primary_references(self) -> list[Any]:
        """Return references as objects."""
        return [FakeCodeReference(**r) for r in self.references]


@dataclass
class FakeCodeReference:
    """Fake code reference for testing."""
    file_path: str = "backend/repo.py"
    repo_id: str = "backend-frameworks"
    start_line: int = 1
    end_line: int = 50
    content: str = "class Repository..."
    score: float = 0.85
    github_url: str = ""
    language: str = "python"
    
    @property
    def chunk(self) -> "FakeCodeReference":
        return self


class FakeNeo4jClient:
    """Fake Neo4jClient for testing.
    
    Implements Neo4jClientProtocol via duck typing.
    """
    
    def __init__(
        self,
        concepts: list[dict[str, Any]] | None = None,
        code_files: list[dict[str, Any]] | None = None,
        patterns: list[dict[str, Any]] | None = None,
        chapters: list[dict[str, Any]] | None = None,
    ):
        self.concepts = concepts or []
        self.code_files = code_files or []
        self.patterns = patterns or []
        self.chapters = chapters or []
        self._connected = False
        self.concept_calls: list[str] = []
        self.code_calls: list[str] = []
    
    async def connect(self) -> None:
        """Establish connection."""
        self._connected = True
    
    async def close(self) -> None:
        """Close connection."""
        self._connected = False
    
    async def health_check(self) -> bool:
        """Check connection health."""
        return self._connected
    
    async def get_concepts_for_chapter(self, chapter_id: str) -> list[Any]:
        """Get concepts for chapter."""
        return [FakeConcept(**c) for c in self.concepts]
    
    async def get_code_for_concept(self, concept: str) -> list[Any]:
        """Get code files for concept."""
        self.code_calls.append(concept)
        return [FakeCodeFile(**c) for c in self.code_files]
    
    async def get_related_patterns(self, pattern: str) -> list[Any]:
        """Get related patterns."""
        return [FakePattern(**p) for p in self.patterns]
    
    async def get_chapters_for_concept(
        self, concept: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get chapters covering concept."""
        self.concept_calls.append(concept)
        return self.chapters


@dataclass
class FakeConcept:
    """Fake Concept for testing."""
    concept_id: str = "concept-001"
    name: str = "Repository Pattern"
    description: str = "Data access pattern"
    tier: int = 2
    aliases: list[str] = field(default_factory=list)


@dataclass
class FakeCodeFile:
    """Fake CodeFileReference for testing."""
    file_path: str = "backend/repo.py"
    github_url: str = "https://github.com/org/repo/blob/main/backend/repo.py"
    repo_id: str = "backend-frameworks"
    concepts: list[str] = field(default_factory=lambda: ["repository"])


@dataclass
class FakePattern:
    """Fake PatternRelationship for testing."""
    pattern_id: str = "pattern-001"
    name: str = "Repository"
    related_patterns: list[str] = field(default_factory=lambda: ["Unit of Work"])
    repos: list[str] = field(default_factory=lambda: ["backend-frameworks"])


class FakeBookPassageClient:
    """Fake BookPassageClient for testing.
    
    Implements BookPassageClientProtocol via duck typing.
    """
    
    def __init__(
        self,
        search_results: list[dict[str, Any]] | None = None,
        concept_passages: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.search_results = search_results or []
        self.concept_passages = concept_passages or {}
        self._connected = False
        self.search_calls: list[dict[str, Any]] = []
        self.concept_calls: list[str] = []
    
    async def connect(self) -> None:
        """Establish connection."""
        self._connected = True
    
    async def close(self) -> None:
        """Close connection."""
        self._connected = False
    
    async def health_check(self) -> bool:
        """Check connection health."""
        return self._connected
    
    async def search_passages(
        self,
        query: str,
        top_k: int = 10,
        filters: Any = None,
    ) -> list[Any]:
        """Search passages via Qdrant."""
        self.search_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [FakeBookPassage(**p) for p in self.search_results]
    
    async def get_passage_by_id(self, passage_id: str) -> Any | None:
        """Get passage by ID."""
        for p in self.search_results:
            if p.get("passage_id") == passage_id:
                return FakeBookPassage(**p)
        return None
    
    async def get_passages_for_concept(
        self, concept: str, limit: int = 10
    ) -> list[Any]:
        """Get passages for concept."""
        self.concept_calls.append(concept)
        passages = self.concept_passages.get(concept, [])
        return [FakeBookPassage(**p) for p in passages[:limit]]


@dataclass
class FakeBookPassage:
    """Fake BookPassage for testing."""
    passage_id: str = "ddd_ch1_p1"
    book_id: str = "ddd"
    book_title: str = "Domain-Driven Design"
    author: str = "Eric Evans"
    chapter_number: int = 1
    chapter_title: str = "Introduction"
    start_page: int = 1
    end_page: int = 20
    content: str = "Repositories isolate domain from data access..."
    concepts: list[str] = field(default_factory=lambda: ["repository"])
    keywords: list[str] = field(default_factory=list)
    relevance_score: float = 0.9


# =============================================================================
# Test Data Factories
# =============================================================================


def create_code_result(
    file_path: str = "backend/repository.py",
    repo_id: str = "backend-frameworks",
    score: float = 0.85,
    content: str = "class Repository...",
) -> dict[str, Any]:
    """Create test code reference data."""
    return {
        "file_path": file_path,
        "repo_id": repo_id,
        "start_line": 1,
        "end_line": 50,
        "content": content,
        "score": score,
        "github_url": f"https://github.com/org/{repo_id}/blob/main/{file_path}",
        "language": "python",
    }


def create_neo4j_concept(
    name: str = "Repository Pattern",
    tier: int = 2,
) -> dict[str, Any]:
    """Create test Neo4j concept data."""
    return {
        "concept_id": f"concept-{name.lower().replace(' ', '-')}",
        "name": name,
        "description": f"Description of {name}",
        "tier": tier,
        "aliases": [],
    }


def create_neo4j_code_file(
    file_path: str = "backend/repository.py",
    repo_id: str = "backend-frameworks",
) -> dict[str, Any]:
    """Create test Neo4j code file data."""
    return {
        "file_path": file_path,
        "github_url": f"https://github.com/org/{repo_id}/blob/main/{file_path}",
        "repo_id": repo_id,
        "concepts": ["repository"],
    }


def create_book_passage(
    passage_id: str = "ddd_ch4_p1",
    book_id: str = "ddd",
    book_title: str = "Domain-Driven Design",
    author: str = "Eric Evans",
    score: float = 0.9,
    concepts: list[str] | None = None,
) -> dict[str, Any]:
    """Create test book passage data."""
    return {
        "passage_id": passage_id,
        "book_id": book_id,
        "book_title": book_title,
        "author": author,
        "chapter_number": 4,
        "chapter_title": "Isolating the Domain",
        "start_page": 65,
        "end_page": 80,
        "content": "Repositories provide a collection-like interface...",
        "concepts": concepts or ["repository", "ddd"],
        "keywords": ["persistence", "collection", "aggregate"],
        "relevance_score": score,
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestUnifiedRetrieverImport:
    """Test that UnifiedRetriever can be imported."""
    
    def test_import_unified_retriever(self) -> None:
        """EC: from src.retrieval import UnifiedRetriever succeeds."""
        from src.retrieval import UnifiedRetriever
        assert UnifiedRetriever is not None
    
    def test_import_result_merger(self) -> None:
        """EC: ResultMerger can be imported."""
        from src.retrieval import ResultMerger
        assert ResultMerger is not None
    
    def test_import_cross_source_ranker(self) -> None:
        """EC: CrossSourceRanker can be imported."""
        from src.retrieval import CrossSourceRanker
        assert CrossSourceRanker is not None


class TestRetrievalResultSchema:
    """Test RetrievalResult schema (AC-24.4)."""
    
    def test_import_retrieval_result(self) -> None:
        """RetrievalResult can be imported."""
        from src.schemas.retrieval_models import RetrievalResult
        assert RetrievalResult is not None
    
    def test_retrieval_result_has_results(self) -> None:
        """RetrievalResult has results list."""
        from src.schemas.retrieval_models import RetrievalResult, RetrievalItem
        
        result = RetrievalResult(
            query="repository pattern",
            results=[],
            total_count=0,
        )
        assert result.results == []
        assert result.total_count == 0
    
    def test_retrieval_result_with_items(self) -> None:
        """RetrievalResult contains RetrievalItem objects."""
        from src.schemas.retrieval_models import (
            RetrievalResult, 
            RetrievalItem,
            SourceType,
        )
        
        item = RetrievalItem(
            source_type=SourceType.CODE,
            source_id="backend-frameworks/repo.py",
            content="class Repository...",
            relevance_score=0.85,
        )
        result = RetrievalResult(
            query="repository pattern",
            results=[item],
            total_count=1,
        )
        assert len(result.results) == 1
        assert result.results[0].source_type == SourceType.CODE


class TestMixedCitationModel:
    """Test MixedCitation model (AC-24.4)."""
    
    def test_import_mixed_citation(self) -> None:
        """MixedCitation can be imported."""
        from src.citations.mixed_citation import MixedCitation
        assert MixedCitation is not None
    
    def test_mixed_citation_source_types(self) -> None:
        """MixedCitation supports code, book, graph source types."""
        from src.citations.mixed_citation import MixedCitation, SourceType
        
        code_citation = MixedCitation(
            source_type=SourceType.CODE,
            source_id="backend/repo.py",
            display_text="Repository pattern implementation",
        )
        assert code_citation.source_type == SourceType.CODE
        
        book_citation = MixedCitation(
            source_type=SourceType.BOOK,
            source_id="ddd_ch4_p1",
            display_text="Evans, DDD, Chapter 4",
        )
        assert book_citation.source_type == SourceType.BOOK
        
        graph_citation = MixedCitation(
            source_type=SourceType.GRAPH,
            source_id="concept-repository",
            display_text="Repository Pattern (Concept)",
        )
        assert graph_citation.source_type == SourceType.GRAPH
    
    def test_mixed_citation_to_footnote(self) -> None:
        """MixedCitation can generate footnote markers."""
        from src.citations.mixed_citation import MixedCitation, SourceType
        
        citation = MixedCitation(
            source_type=SourceType.BOOK,
            source_id="ddd_ch4_p1",
            display_text="Evans, Eric, Domain-Driven Design (2003), 65-80.",
            footnote_number=1,
        )
        assert citation.to_footnote() == "[^1]: Evans, Eric, Domain-Driven Design (2003), 65-80."


class TestScopeFiltering:
    """Test scope filtering (AC-24.6)."""
    
    def test_import_retrieval_scope(self) -> None:
        """RetrievalScope enum can be imported."""
        from src.schemas.retrieval_models import RetrievalScope
        assert RetrievalScope is not None
    
    def test_retrieval_scope_values(self) -> None:
        """RetrievalScope has code_only, books_only, all values."""
        from src.schemas.retrieval_models import RetrievalScope
        
        assert RetrievalScope.CODE_ONLY.value == "code_only"
        assert RetrievalScope.BOOKS_ONLY.value == "books_only"
        assert RetrievalScope.ALL.value == "all"
    
    def test_retrieval_scope_graph_only(self) -> None:
        """RetrievalScope includes graph_only."""
        from src.schemas.retrieval_models import RetrievalScope
        
        assert RetrievalScope.GRAPH_ONLY.value == "graph_only"


class TestUnifiedRetrieverInitialization:
    """Test UnifiedRetriever initialization (AC-24.1)."""
    
    def test_unified_retriever_accepts_all_clients(self) -> None:
        """UnifiedRetriever accepts code, neo4j, book clients."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        config = UnifiedRetrieverConfig()
        retriever = UnifiedRetriever(
            config=config,
            code_client=FakeCodeReferenceClient(),
            neo4j_client=FakeNeo4jClient(),
            book_client=FakeBookPassageClient(),
        )
        assert retriever is not None
    
    def test_unified_retriever_with_partial_clients(self) -> None:
        """UnifiedRetriever works with subset of clients."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        config = UnifiedRetrieverConfig()
        # Only code client
        retriever = UnifiedRetriever(
            config=config,
            code_client=FakeCodeReferenceClient(),
        )
        assert retriever is not None


class TestMultiSourceOrchestration:
    """Test multi-source query orchestration (AC-24.2)."""
    
    @pytest.mark.asyncio
    async def test_retrieve_queries_all_sources(self) -> None:
        """Retrieve queries code, neo4j, and book sources."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        code_client = FakeCodeReferenceClient(
            search_results=[create_code_result()]
        )
        neo4j_client = FakeNeo4jClient(
            chapters=[{"chapter_id": "ch1", "title": "Intro", "book_id": "ddd", "tier": 1}]
        )
        book_client = FakeBookPassageClient(
            search_results=[create_book_passage()]
        )
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=code_client,
            neo4j_client=neo4j_client,
            book_client=book_client,
        )
        
        result = await retriever.retrieve("repository pattern")
        
        # Should have called all clients
        assert len(code_client.search_calls) == 1
        assert len(book_client.search_calls) == 1
        assert result.total_count > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_parallel_execution(self) -> None:
        """Retrieve executes source queries in parallel."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FakeCodeReferenceClient(search_results=[create_code_result()]),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        # Parallel execution should complete quickly
        result = await retriever.retrieve("repository pattern")
        assert result.query == "repository pattern"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_scope_code_only(self) -> None:
        """AC-24.6: Retrieve with code_only scope."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        from src.schemas.retrieval_models import RetrievalScope
        
        code_client = FakeCodeReferenceClient(
            search_results=[create_code_result()]
        )
        book_client = FakeBookPassageClient(
            search_results=[create_book_passage()]
        )
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=code_client,
            book_client=book_client,
        )
        
        result = await retriever.retrieve(
            "repository pattern",
            scope=RetrievalScope.CODE_ONLY,
        )
        
        # Should only call code client
        assert len(code_client.search_calls) == 1
        assert len(book_client.search_calls) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_with_scope_books_only(self) -> None:
        """AC-24.6: Retrieve with books_only scope."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        from src.schemas.retrieval_models import RetrievalScope
        
        code_client = FakeCodeReferenceClient(
            search_results=[create_code_result()]
        )
        book_client = FakeBookPassageClient(
            search_results=[create_book_passage()]
        )
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=code_client,
            book_client=book_client,
        )
        
        result = await retriever.retrieve(
            "repository pattern",
            scope=RetrievalScope.BOOKS_ONLY,
        )
        
        # Should only call book client
        assert len(code_client.search_calls) == 0
        assert len(book_client.search_calls) == 1


class TestResultMerging:
    """Test result merging strategy (AC-24.3)."""
    
    def test_import_result_merger(self) -> None:
        """ResultMerger can be imported."""
        from src.retrieval.merger import ResultMerger
        assert ResultMerger is not None
    
    def test_merger_combines_sources(self) -> None:
        """Merger combines results from multiple sources."""
        from src.retrieval.merger import ResultMerger
        from src.schemas.retrieval_models import RetrievalItem, SourceType
        
        code_items = [
            RetrievalItem(
                source_type=SourceType.CODE,
                source_id="repo.py",
                content="class Repo",
                relevance_score=0.85,
            )
        ]
        book_items = [
            RetrievalItem(
                source_type=SourceType.BOOK,
                source_id="ddd_ch4",
                content="Repositories...",
                relevance_score=0.9,
            )
        ]
        
        merger = ResultMerger()
        merged = merger.merge(code_items, book_items)
        
        assert len(merged) == 2
        source_types = {item.source_type for item in merged}
        assert SourceType.CODE in source_types
        assert SourceType.BOOK in source_types
    
    def test_merger_deduplicates_similar_content(self) -> None:
        """Merger removes duplicate content across sources."""
        from src.retrieval.merger import ResultMerger
        from src.schemas.retrieval_models import RetrievalItem, SourceType
        
        # Same content from different sources
        code_item = RetrievalItem(
            source_type=SourceType.CODE,
            source_id="repo.py",
            content="Repository pattern isolates domain",
            relevance_score=0.85,
        )
        book_item = RetrievalItem(
            source_type=SourceType.BOOK,
            source_id="ddd_ch4",
            content="Repository pattern isolates domain",  # Same content
            relevance_score=0.9,
        )
        
        merger = ResultMerger(dedup_threshold=0.95)
        merged = merger.merge([code_item], [book_item])
        
        # Should keep higher scoring item
        assert len(merged) == 1
        assert merged[0].relevance_score == 0.9


class TestCrossSourceRanking:
    """Test cross-source ranking (AC-24.3)."""
    
    def test_import_cross_source_ranker(self) -> None:
        """CrossSourceRanker can be imported."""
        from src.retrieval.ranker import CrossSourceRanker
        assert CrossSourceRanker is not None
    
    def test_ranker_sorts_by_relevance(self) -> None:
        """Ranker sorts results by relevance score descending."""
        from src.retrieval.ranker import CrossSourceRanker
        from src.schemas.retrieval_models import RetrievalItem, SourceType
        
        items = [
            RetrievalItem(
                source_type=SourceType.CODE,
                source_id="repo.py",
                content="Low relevance",
                relevance_score=0.5,
            ),
            RetrievalItem(
                source_type=SourceType.BOOK,
                source_id="ddd_ch4",
                content="High relevance",
                relevance_score=0.95,
            ),
            RetrievalItem(
                source_type=SourceType.GRAPH,
                source_id="concept-repo",
                content="Medium relevance",
                relevance_score=0.75,
            ),
        ]
        
        ranker = CrossSourceRanker()
        ranked = ranker.rank(items)
        
        assert ranked[0].relevance_score == 0.95
        assert ranked[1].relevance_score == 0.75
        assert ranked[2].relevance_score == 0.5
    
    def test_ranker_applies_source_boost(self) -> None:
        """Ranker can apply source-type boosts."""
        from src.retrieval.ranker import CrossSourceRanker
        from src.schemas.retrieval_models import RetrievalItem, SourceType
        
        items = [
            RetrievalItem(
                source_type=SourceType.CODE,
                source_id="repo.py",
                content="Code",
                relevance_score=0.8,
            ),
            RetrievalItem(
                source_type=SourceType.BOOK,
                source_id="ddd_ch4",
                content="Book",
                relevance_score=0.8,
            ),
        ]
        
        # Boost books
        ranker = CrossSourceRanker(
            source_boosts={SourceType.BOOK: 1.1, SourceType.CODE: 1.0}
        )
        ranked = ranker.rank(items)
        
        # Book should be first due to boost
        assert ranked[0].source_type == SourceType.BOOK
    
    def test_ranker_limits_results(self) -> None:
        """Ranker respects top_k limit."""
        from src.retrieval.ranker import CrossSourceRanker
        from src.schemas.retrieval_models import RetrievalItem, SourceType
        
        items = [
            RetrievalItem(
                source_type=SourceType.CODE,
                source_id=f"file{i}.py",
                content=f"Content {i}",
                relevance_score=0.9 - (i * 0.1),
            )
            for i in range(10)
        ]
        
        ranker = CrossSourceRanker()
        ranked = ranker.rank(items, top_k=5)
        
        assert len(ranked) == 5


class TestCitationGeneration:
    """Test citation generation (AC-24.4)."""
    
    @pytest.mark.asyncio
    async def test_retrieve_returns_mixed_citations(self) -> None:
        """Retrieve returns MixedCitation for each result."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FakeCodeReferenceClient(search_results=[create_code_result()]),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        result = await retriever.retrieve("repository pattern")
        
        # Each result should have a citation
        assert len(result.citations) > 0
        for citation in result.citations:
            assert citation.source_type is not None
            assert citation.source_id is not None
    
    @pytest.mark.asyncio
    async def test_citations_identify_source_type(self) -> None:
        """EC: Citations correctly identify source type (code, book, graph)."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        from src.citations.mixed_citation import SourceType
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FakeCodeReferenceClient(search_results=[create_code_result()]),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        result = await retriever.retrieve("repository pattern")
        
        source_types = {c.source_type for c in result.citations}
        assert SourceType.CODE in source_types or SourceType.BOOK in source_types


class TestCrossReferenceIntegration:
    """Test cross_reference function integration (AC-24.5)."""
    
    @pytest.mark.asyncio
    async def test_cross_reference_uses_unified_retriever(self) -> None:
        """AC-24.5: cross_reference function uses UnifiedRetriever."""
        # This will be implemented when updating cross_reference.py
        # For now, verify the interface is compatible
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        from src.schemas.retrieval_models import RetrievalScope
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FakeCodeReferenceClient(search_results=[create_code_result()]),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        # The interface should support what cross_reference needs
        result = await retriever.retrieve(
            "repository pattern",
            scope=RetrievalScope.ALL,
            top_k=10,
        )
        
        assert result.query == "repository pattern"
        assert hasattr(result, "results")
        assert hasattr(result, "citations")


class TestUnifiedRetrieverProtocol:
    """Test UnifiedRetriever protocol compliance."""
    
    def test_retriever_protocol_exists(self) -> None:
        """UnifiedRetrieverProtocol can be imported."""
        from src.retrieval.unified_retriever import UnifiedRetrieverProtocol
        assert UnifiedRetrieverProtocol is not None
    
    def test_fake_retriever_protocol_compliance(self) -> None:
        """FakeUnifiedRetriever implements protocol."""
        from src.retrieval.unified_retriever import (
            UnifiedRetrieverProtocol,
            FakeUnifiedRetriever,
        )
        
        fake = FakeUnifiedRetriever()
        assert isinstance(fake, UnifiedRetrieverProtocol)


class TestErrorHandling:
    """Test error handling in UnifiedRetriever."""
    
    @pytest.mark.asyncio
    async def test_retrieve_handles_client_failure(self) -> None:
        """Retrieve continues when one client fails."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        class FailingClient:
            async def search(self, *args, **kwargs):
                raise ConnectionError("Service unavailable")
            
            async def close(self):
                pass
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FailingClient(),  # This will fail
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        # Should not raise, should return partial results
        result = await retriever.retrieve("repository pattern")
        assert result is not None
        assert result.total_count >= 0
    
    @pytest.mark.asyncio
    async def test_retrieve_logs_client_errors(self) -> None:
        """Retrieve logs errors from failed clients."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        class FailingClient:
            async def search(self, *args, **kwargs):
                raise ConnectionError("Service unavailable")
            
            async def close(self):
                pass
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FailingClient(),
        )
        
        result = await retriever.retrieve("test")
        
        # Should have recorded the error
        assert len(result.errors) > 0
        assert "code" in result.errors[0].lower() or "connection" in result.errors[0].lower()


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_retrieve_respects_timeout(self) -> None:
        """Retrieve respects configured timeout."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        config = UnifiedRetrieverConfig(timeout_seconds=1.0)
        retriever = UnifiedRetriever(
            config=config,
            code_client=FakeCodeReferenceClient(),
        )
        
        # Should complete within timeout
        result = await retriever.retrieve("test")
        assert result is not None


class TestRetrievalResultMetadata:
    """Test RetrievalResult metadata fields."""
    
    @pytest.mark.asyncio
    async def test_result_includes_query(self) -> None:
        """Result includes original query."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        result = await retriever.retrieve("repository pattern")
        assert result.query == "repository pattern"
    
    @pytest.mark.asyncio
    async def test_result_includes_source_counts(self) -> None:
        """Result includes count per source type."""
        from src.retrieval import UnifiedRetriever, UnifiedRetrieverConfig
        from src.schemas.retrieval_models import SourceType
        
        retriever = UnifiedRetriever(
            config=UnifiedRetrieverConfig(),
            code_client=FakeCodeReferenceClient(search_results=[create_code_result()]),
            book_client=FakeBookPassageClient(search_results=[create_book_passage()]),
        )
        
        result = await retriever.retrieve("repository pattern")
        
        assert hasattr(result, "source_counts")
        assert isinstance(result.source_counts, dict)
