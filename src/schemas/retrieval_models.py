"""Retrieval Models for Unified Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.5)
Acceptance Criteria:
- AC-24.4: Returns unified RetrievalResult with mixed citations
- AC-24.6: Supports scope filtering (code-only, books-only, all)

Models for representing unified retrieval results across multiple sources:
- Code Reference Engine (code)
- Neo4j Graph (graph)
- Book Passages (book)

Pattern: Value Objects (immutable data carriers)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pydantic Schemas
Anti-Pattern: No mutable default arguments (AP-1.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.citations.mixed_citation import MixedCitation


# =============================================================================
# Enums
# =============================================================================


class SourceType(str, Enum):
    """Source type for retrieval results.
    
    Identifies the origin of a retrieval result for citation generation.
    """
    
    CODE = "code"
    BOOK = "book"
    GRAPH = "graph"
    SEMANTIC = "semantic"  # Generic semantic search result


class RetrievalScope(str, Enum):
    """Scope filter for retrieval queries.
    
    AC-24.6: Supports scope filtering (code-only, books-only, all)
    """
    
    ALL = "all"
    CODE_ONLY = "code_only"
    BOOKS_ONLY = "books_only"
    GRAPH_ONLY = "graph_only"


# =============================================================================
# Retrieval Item Model
# =============================================================================


@dataclass(frozen=True)
class RetrievalItem:
    """A single retrieval result item.
    
    Represents a result from any knowledge source with normalized fields
    for cross-source ranking and citation generation.
    
    Attributes:
        source_type: Origin of the result (code, book, graph)
        source_id: Unique identifier within source
        content: Text content of the result
        relevance_score: Normalized relevance score (0.0-1.0)
        title: Optional title (chapter title, file name, concept name)
        metadata: Additional source-specific metadata
    """
    
    source_type: SourceType
    source_id: str
    content: str
    relevance_score: float = 0.0
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "title": self.title,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalItem:
        """Create from dictionary."""
        return cls(
            source_type=SourceType(data["source_type"]),
            source_id=data["source_id"],
            content=data["content"],
            relevance_score=data.get("relevance_score", 0.0),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Retrieval Result Model
# =============================================================================


@dataclass
class RetrievalResult:
    """Unified retrieval result across all sources.
    
    WBS: AGT24.5 - Create RetrievalResult schema
    AC-24.4: Returns unified RetrievalResult with mixed citations
    
    Contains results from all queried knowledge sources, merged and ranked,
    with citations for each result.
    
    Attributes:
        query: Original search query
        results: List of retrieval items, ranked by relevance
        total_count: Total number of results (before pagination)
        citations: List of mixed citations for footnote generation
        source_counts: Count of results per source type
        errors: List of error messages from failed sources
        scope: Scope filter used for the query
    """
    
    query: str
    results: list[RetrievalItem] = field(default_factory=list)
    total_count: int = 0
    citations: list[MixedCitation] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    scope: RetrievalScope = RetrievalScope.ALL
    
    def __post_init__(self) -> None:
        """Calculate source counts after initialization."""
        if not self.source_counts and self.results:
            self.source_counts = self._calculate_source_counts()
    
    def _calculate_source_counts(self) -> dict[str, int]:
        """Calculate count of results per source type."""
        counts: dict[str, int] = {}
        for item in self.results:
            key = item.source_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "citations": [c.to_dict() for c in self.citations],
            "source_counts": self.source_counts,
            "errors": self.errors,
            "scope": self.scope.value,
        }
    
    def get_results_by_source(self, source_type: SourceType) -> list[RetrievalItem]:
        """Get results filtered by source type."""
        return [r for r in self.results if r.source_type == source_type]
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during retrieval."""
        return len(self.errors) > 0


# =============================================================================
# Query Options Model
# =============================================================================


@dataclass(frozen=True)
class RetrievalOptions:
    """Options for unified retrieval query.
    
    Configures how the retrieval is executed and results are processed.
    
    Attributes:
        scope: Which sources to query (all, code_only, books_only, graph_only)
        top_k: Maximum number of results per source
        min_relevance: Minimum relevance score threshold
        include_content: Whether to include full content in results
        deduplicate: Whether to remove duplicate content
    """
    
    scope: RetrievalScope = RetrievalScope.ALL
    top_k: int = 10
    min_relevance: float = 0.0
    include_content: bool = True
    deduplicate: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope.value,
            "top_k": self.top_k,
            "min_relevance": self.min_relevance,
            "include_content": self.include_content,
            "deduplicate": self.deduplicate,
        }
