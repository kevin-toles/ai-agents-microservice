"""Passage Models for Book/JSON Retrieval.

WBS Reference: WBS-AGT23 Book/JSON Passage Retrieval (AGT23.6)
Acceptance Criteria:
- AC-23.4: Return structured BookPassage with citation metadata

Models for representing book passages with citation metadata.
Supports Chicago-style citation generation.

Pattern: Value Objects (immutable data carriers)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pydantic Schemas
Anti-Pattern: No mutable default arguments (AP-1.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Filter Models
# =============================================================================


@dataclass(frozen=True)
class PassageFilter:
    """Filter criteria for passage searches.
    
    AC-23.5: Support filtering by book, chapter, concept
    
    Attributes:
        book_id: Filter by book identifier
        chapter_number: Filter by chapter number
        concept: Filter by concept (must be in passage.concepts)
        min_relevance: Minimum relevance score threshold
    """
    
    book_id: str | None = None
    chapter_number: int | None = None
    concept: str | None = None
    min_relevance: float | None = None
    
    def is_empty(self) -> bool:
        """Check if filter has any criteria."""
        return (
            self.book_id is None
            and self.chapter_number is None
            and self.concept is None
            and self.min_relevance is None
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.book_id is not None:
            result["book_id"] = self.book_id
        if self.chapter_number is not None:
            result["chapter_number"] = self.chapter_number
        if self.concept is not None:
            result["concept"] = self.concept
        if self.min_relevance is not None:
            result["min_relevance"] = self.min_relevance
        return result


# =============================================================================
# Metadata Models
# =============================================================================


@dataclass(frozen=True)
class PassageMetadata:
    """Metadata for a book passage.
    
    Contains enrichment provenance and processing information.
    
    Attributes:
        enrichment_model: LLM model used for enrichment
        enrichment_timestamp: ISO timestamp of enrichment
        embedding_model: Model used for vector embeddings
        embedding_dimensions: Number of embedding dimensions
        neo4j_node_id: Optional Neo4j node reference
        qdrant_point_id: Optional Qdrant vector ID
    """
    
    enrichment_model: str = ""
    enrichment_timestamp: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    neo4j_node_id: str | None = None
    qdrant_point_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enrichment_model": self.enrichment_model,
            "enrichment_timestamp": self.enrichment_timestamp,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "neo4j_node_id": self.neo4j_node_id,
            "qdrant_point_id": self.qdrant_point_id,
        }


# =============================================================================
# Main Passage Model
# =============================================================================


@dataclass(frozen=True)
class BookPassage:
    """Book passage with citation metadata.
    
    WBS: AGT23.6 - Create BookPassage schema
    AC-23.4: Return structured BookPassage with citation metadata
    
    Represents a passage from an enriched book JSON file with all
    metadata required for Chicago-style citation generation.
    
    Attributes:
        passage_id: Unique identifier (e.g., "{book_id}_ch{N}_p{M}")
        book_id: Book identifier (e.g., "aposd", "ddd-eric-evans")
        book_title: Full book title
        author: Book author(s)
        chapter_number: Chapter number (1-indexed)
        chapter_title: Chapter title
        start_page: Starting page number
        end_page: Ending page number
        content: Passage text content
        concepts: List of concepts covered in passage
        keywords: Extracted keywords
        relevance_score: Vector similarity score (0.0-1.0)
        metadata: Optional enrichment metadata
    """
    
    passage_id: str
    book_id: str
    book_title: str
    author: str
    chapter_number: int
    chapter_title: str
    start_page: int
    end_page: int
    content: str
    concepts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    metadata: PassageMetadata | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passage_id": self.passage_id,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "author": self.author,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "content": self.content,
            "concepts": list(self.concepts),
            "keywords": list(self.keywords),
            "relevance_score": self.relevance_score,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }
    
    def get_page_range(self) -> str:
        """Get formatted page range string."""
        if self.start_page == self.end_page:
            return str(self.start_page)
        return f"{self.start_page}-{self.end_page}"
    
    def has_concept(self, concept: str) -> bool:
        """Check if passage covers a concept."""
        return concept.lower() in [c.lower() for c in self.concepts]
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BookPassage":
        """Create BookPassage from dictionary.
        
        Args:
            data: Dictionary with passage data
            
        Returns:
            BookPassage instance
        """
        metadata = None
        if data.get("metadata"):
            metadata = PassageMetadata(
                enrichment_model=data["metadata"].get("enrichment_model", ""),
                enrichment_timestamp=data["metadata"].get("enrichment_timestamp", ""),
                embedding_model=data["metadata"].get("embedding_model", ""),
                embedding_dimensions=data["metadata"].get("embedding_dimensions", 384),
                neo4j_node_id=data["metadata"].get("neo4j_node_id"),
                qdrant_point_id=data["metadata"].get("qdrant_point_id"),
            )
        
        return cls(
            passage_id=data["passage_id"],
            book_id=data["book_id"],
            book_title=data["book_title"],
            author=data["author"],
            chapter_number=data["chapter_number"],
            chapter_title=data["chapter_title"],
            start_page=data["start_page"],
            end_page=data["end_page"],
            content=data["content"],
            concepts=data.get("concepts", []),
            keywords=data.get("keywords", []),
            relevance_score=data.get("relevance_score", 0.0),
            metadata=metadata,
        )


# =============================================================================
# Chapter Reference (for similar chapters)
# =============================================================================


@dataclass(frozen=True)
class ChapterPassageRef:
    """Reference to a chapter containing relevant passages.
    
    Used for cross-referencing chapters in search results.
    
    Attributes:
        book_id: Book identifier
        book_title: Book title
        chapter_number: Chapter number
        chapter_title: Chapter title
        passage_count: Number of matching passages in chapter
        best_score: Highest relevance score among passages
    """
    
    book_id: str
    book_title: str
    chapter_number: int
    chapter_title: str
    passage_count: int = 0
    best_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "book_id": self.book_id,
            "book_title": self.book_title,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "passage_count": self.passage_count,
            "best_score": self.best_score,
        }


# =============================================================================
# Search Results
# =============================================================================


@dataclass
class PassageSearchResult:
    """Result from passage search operation.
    
    Contains passages with search metadata.
    
    Attributes:
        passages: List of matching passages
        total_count: Total matches (may exceed returned count)
        query: Original search query
        filter_applied: Filter criteria used
        search_time_ms: Search duration in milliseconds
    """
    
    passages: list[BookPassage] = field(default_factory=list)
    total_count: int = 0
    query: str = ""
    filter_applied: PassageFilter | None = None
    search_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passages": [p.to_dict() for p in self.passages],
            "total_count": self.total_count,
            "query": self.query,
            "filter_applied": self.filter_applied.to_dict() if self.filter_applied else None,
            "search_time_ms": self.search_time_ms,
        }
