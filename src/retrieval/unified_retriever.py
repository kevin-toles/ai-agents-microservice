"""Unified Retriever for Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.1, AGT24.2)
Acceptance Criteria:
- AC-24.1: Single interface queries all knowledge sources
- AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books
- AC-24.3: Merges and ranks results across sources
- AC-24.4: Returns unified RetrievalResult with mixed citations
- AC-24.5: cross_reference agent function uses this retriever
- AC-24.6: Supports scope filtering (code-only, books-only, all)

Unified retrieval across multiple knowledge sources:
- Code Reference Engine (WBS-AGT21)
- Neo4j Graph (WBS-AGT22)
- Book Passages (WBS-AGT23)

Pattern: Facade pattern with async orchestration
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions
Anti-Pattern: #12 (Connection Pooling via shared clients)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.citations.mixed_citation import (
    MixedCitation,
    SourceType,
    citations_from_retrieval_items,
)
from src.retrieval.merger import ResultMerger
from src.retrieval.ranker import CrossSourceRanker
from src.schemas.retrieval_models import (
    RetrievalItem,
    RetrievalResult,
    RetrievalScope,
    SourceType as RetrievalSourceType,
)

if TYPE_CHECKING:
    from src.clients.protocols import (
        BookPassageClientProtocol,
        CodeReferenceProtocol,
        Neo4jClientProtocol,
    )


logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class UnifiedRetrieverProtocol(Protocol):
    """Protocol for UnifiedRetriever.
    
    Enables duck typing for test doubles (FakeUnifiedRetriever).
    """
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve results from knowledge sources.
        
        Args:
            query: Search query
            scope: Which sources to query
            top_k: Maximum results to return
        
        Returns:
            Unified retrieval result
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class UnifiedRetrieverConfig:
    """Configuration for UnifiedRetriever.
    
    Attributes:
        timeout_seconds: Timeout for each source query
        max_per_source: Maximum results per source (0 = unlimited)
        dedup_threshold: Similarity threshold for deduplication
        source_boosts: Relevance boost per source type
    """
    
    timeout_seconds: float = 10.0
    max_per_source: int = 20
    dedup_threshold: float = 0.95
    source_boosts: dict[RetrievalSourceType, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Set default source boosts if not provided."""
        if not self.source_boosts:
            self.source_boosts = {
                RetrievalSourceType.BOOK: 1.0,
                RetrievalSourceType.CODE: 1.0,
                RetrievalSourceType.GRAPH: 0.9,
                RetrievalSourceType.SEMANTIC: 0.8,
            }


# =============================================================================
# Unified Retriever Implementation
# =============================================================================


class UnifiedRetriever:
    """Unified retriever across all knowledge sources.
    
    WBS: AGT24.1 - Create UnifiedRetriever class
    AC-24.1: Single interface queries all knowledge sources
    AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books
    
    Provides a single interface for querying:
    - Code Reference Engine (code search)
    - Neo4j Graph (concept relationships)
    - Book Passages (book content search)
    
    Attributes:
        config: Retriever configuration
        code_client: Optional code reference client
        neo4j_client: Optional Neo4j client
        book_client: Optional book passage client
    """
    
    def __init__(
        self,
        config: UnifiedRetrieverConfig,
        code_client: Any | None = None,
        neo4j_client: Any | None = None,
        book_client: Any | None = None,
    ) -> None:
        """Initialize UnifiedRetriever.
        
        Args:
            config: Retriever configuration
            code_client: Optional CodeReferenceClient
            neo4j_client: Optional Neo4jClient
            book_client: Optional BookPassageClient
        """
        self.config = config
        self.code_client = code_client
        self.neo4j_client = neo4j_client
        self.book_client = book_client
        
        # Initialize merger and ranker
        self._merger = ResultMerger(
            dedup_threshold=config.dedup_threshold,
            max_per_source=config.max_per_source,
        )
        self._ranker = CrossSourceRanker(
            source_boosts=config.source_boosts,
        )
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve results from knowledge sources.
        
        AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books
        AC-24.6: Supports scope filtering (code-only, books-only, all)
        
        Args:
            query: Search query
            scope: Which sources to query
            top_k: Maximum results to return
        
        Returns:
            Unified retrieval result with merged, ranked results
        """
        errors: list[str] = []
        all_items: list[list[RetrievalItem]] = []
        
        # Determine which sources to query based on scope
        tasks = []
        source_names = []
        
        if self._should_query_code(scope):
            tasks.append(self._query_code(query, top_k))
            source_names.append("code")
        
        if self._should_query_books(scope):
            tasks.append(self._query_books(query, top_k))
            source_names.append("books")
        
        if self._should_query_graph(scope):
            tasks.append(self._query_graph(query, top_k))
            source_names.append("graph")
        
        # Execute queries in parallel with timeout
        if tasks:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout_seconds,
                )
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        error_msg = f"{source_names[i]}: {str(result)}"
                        errors.append(error_msg)
                        logger.warning(f"Source query failed: {error_msg}")
                    else:
                        all_items.append(result)
                        
            except asyncio.TimeoutError:
                errors.append("Query timeout exceeded")
                logger.warning(f"Unified retrieval timeout after {self.config.timeout_seconds}s")
        
        # Merge results
        merged = self._merger.merge(*all_items)
        
        # Rank results
        ranked = self._ranker.rank(merged, top_k=top_k)
        
        # Generate citations
        citations = citations_from_retrieval_items(ranked)
        
        return RetrievalResult(
            query=query,
            results=ranked,
            total_count=len(ranked),
            citations=citations,
            errors=errors,
            scope=scope,
        )
    
    def _should_query_code(self, scope: RetrievalScope) -> bool:
        """Check if code source should be queried."""
        if self.code_client is None:
            return False
        return scope in (RetrievalScope.ALL, RetrievalScope.CODE_ONLY)
    
    def _should_query_books(self, scope: RetrievalScope) -> bool:
        """Check if book source should be queried."""
        if self.book_client is None:
            return False
        return scope in (RetrievalScope.ALL, RetrievalScope.BOOKS_ONLY)
    
    def _should_query_graph(self, scope: RetrievalScope) -> bool:
        """Check if graph source should be queried."""
        if self.neo4j_client is None:
            return False
        return scope in (RetrievalScope.ALL, RetrievalScope.GRAPH_ONLY)
    
    async def _query_code(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalItem]:
        """Query code reference engine.
        
        Args:
            query: Search query
            top_k: Maximum results
        
        Returns:
            List of retrieval items from code source
        """
        if self.code_client is None:
            return []
        
        try:
            context = await self.code_client.search(query, top_k=top_k)
            
            items = []
            for ref in context.primary_references:
                chunk = ref.chunk if hasattr(ref, 'chunk') else ref
                
                item = RetrievalItem(
                    source_type=RetrievalSourceType.CODE,
                    source_id=chunk.file_path,
                    content=chunk.content[:500],  # Truncate for result
                    relevance_score=getattr(chunk, 'score', 0.0),
                    title=chunk.file_path,
                    metadata={
                        "repo_id": getattr(chunk, 'repo_id', ''),
                        "start_line": getattr(chunk, 'start_line', 1),
                        "end_line": getattr(chunk, 'end_line', 1),
                        "language": getattr(chunk, 'language', 'text'),
                        "url": getattr(chunk, 'github_url', ''),
                        "file_path": chunk.file_path,
                    },
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.warning(f"Code query failed: {e}")
            raise
    
    async def _query_books(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalItem]:
        """Query book passage client.
        
        Args:
            query: Search query
            top_k: Maximum results
        
        Returns:
            List of retrieval items from book source
        """
        if self.book_client is None:
            return []
        
        try:
            passages = await self.book_client.search_passages(query, top_k=top_k)
            
            items = []
            for passage in passages:
                item = RetrievalItem(
                    source_type=RetrievalSourceType.BOOK,
                    source_id=passage.passage_id,
                    content=passage.content[:500],  # Truncate for result
                    relevance_score=getattr(passage, 'relevance_score', 0.0),
                    title=f"{passage.book_title} - {passage.chapter_title}",
                    metadata={
                        "book_id": passage.book_id,
                        "book_title": passage.book_title,
                        "author": passage.author,
                        "chapter_number": passage.chapter_number,
                        "chapter_title": passage.chapter_title,
                        "start_page": passage.start_page,
                        "end_page": passage.end_page,
                        "concepts": list(getattr(passage, 'concepts', [])),
                    },
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.warning(f"Book query failed: {e}")
            raise
    
    async def _query_graph(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalItem]:
        """Query Neo4j graph.
        
        Uses the query to find related concepts and their code files.
        
        Args:
            query: Search query (treated as concept name)
            top_k: Maximum results
        
        Returns:
            List of retrieval items from graph source
        """
        if self.neo4j_client is None:
            return []
        
        try:
            # Query chapters related to the concept
            chapters = await self.neo4j_client.get_chapters_for_concept(
                query, limit=top_k
            )
            
            items = []
            for chapter in chapters:
                item = RetrievalItem(
                    source_type=RetrievalSourceType.GRAPH,
                    source_id=chapter.get("chapter_id", ""),
                    content=f"Chapter covering {query}",
                    relevance_score=0.75,  # Default score for graph results
                    title=chapter.get("title", ""),
                    metadata={
                        "book_id": chapter.get("book_id", ""),
                        "tier": chapter.get("tier", 0),
                        "node_type": "Chapter",
                        "name": chapter.get("title", ""),
                    },
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            logger.warning(f"Graph query failed: {e}")
            raise


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeUnifiedRetriever:
    """Fake UnifiedRetriever for testing.
    
    Implements UnifiedRetrieverProtocol via duck typing.
    """
    
    def __init__(
        self,
        results: list[RetrievalItem] | None = None,
        citations: list[MixedCitation] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Initialize with preconfigured results."""
        self._results = results or []
        self._citations = citations or []
        self._errors = errors or []
        self.retrieve_calls: list[dict[str, Any]] = []
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Return preconfigured results."""
        self.retrieve_calls.append({
            "query": query,
            "scope": scope,
            "top_k": top_k,
        })
        
        return RetrievalResult(
            query=query,
            results=self._results[:top_k],
            total_count=len(self._results),
            citations=self._citations,
            errors=self._errors,
            scope=scope,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_unified_retriever(
    code_client: Any | None = None,
    neo4j_client: Any | None = None,
    book_client: Any | None = None,
    **config_kwargs: Any,
) -> UnifiedRetriever:
    """Create a UnifiedRetriever with default configuration.
    
    Factory function for creating a retriever with common defaults.
    
    Args:
        code_client: Optional code reference client
        neo4j_client: Optional Neo4j client
        book_client: Optional book passage client
        **config_kwargs: Additional configuration options
    
    Returns:
        Configured UnifiedRetriever instance
    """
    config = UnifiedRetrieverConfig(**config_kwargs)
    return UnifiedRetriever(
        config=config,
        code_client=code_client,
        neo4j_client=neo4j_client,
        book_client=book_client,
    )
