"""Cross-Reference Agent state models.

Defines the Pydantic models for LangGraph state management.
These models represent the data flowing through the agent workflow.

Pattern: Pydantic State Models for LangGraph
Source: ARCHITECTURE.md (ai-agents), Generative AI with LangChain Ch. "MessagesState"
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Constants for Field descriptions (SonarQube S1192 - duplicated literals)
_DESC_CHAPTER_TITLE = "Chapter title"
_DESC_CHAPTER_NUMBER = "Chapter number"
_DESC_TIER_LEVEL = "Tier level"


class RelationshipType(str, Enum):
    """Tier relationship types per TIER_RELATIONSHIP_DIAGRAM.md.
    
    Spider Web Model: All relationships are BIDIRECTIONAL.
    The graph is a web, not a one-way hierarchy.
    """
    
    PARALLEL = "parallel"           # Same tier (horizontal)
    PERPENDICULAR = "perpendicular" # Adjacent tier ±1 (vertical)
    SKIP_TIER = "skip_tier"         # Non-adjacent tier ±2+ (diagonal)


class SourceChapter(BaseModel):
    """Input model representing the source chapter to cross-reference.
    
    This is the starting point for the cross-reference workflow.
    """
    
    book: str = Field(..., description="Source book title")
    chapter: int = Field(..., ge=1, description=f"{_DESC_CHAPTER_NUMBER} (1-indexed)")
    title: str = Field(..., description=_DESC_CHAPTER_TITLE)
    tier: int = Field(..., ge=1, le=3, description=f"{_DESC_TIER_LEVEL} (1=Architecture, 2=Implementation, 3=Practices)")
    content: str | None = Field(default=None, description="Chapter content (optional, can be retrieved)")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    concepts: list[str] = Field(default_factory=list, description="Key concepts")
    summary: str | None = Field(default=None, description="Chapter summary")


class TraversalConfig(BaseModel):
    """Configuration for spider web graph traversal.
    
    Controls how the agent navigates the taxonomy graph.
    """
    
    max_hops: int = Field(default=3, ge=1, le=10, description="Maximum traversal depth")
    relationship_types: list[RelationshipType] = Field(
        default_factory=lambda: [
            RelationshipType.PARALLEL,
            RelationshipType.PERPENDICULAR,
            RelationshipType.SKIP_TIER,
        ],
        description="Which relationship types to follow"
    )
    allow_cycles: bool = Field(default=True, description="Allow revisiting tiers")
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    max_results_per_tier: int = Field(default=10, ge=1, description="Max results per tier")
    include_tier1: bool = Field(default=True, description="Include Tier 1 (Architecture Spine)")
    include_tier2: bool = Field(default=True, description="Include Tier 2 (Implementation)")
    include_tier3: bool = Field(default=True, description="Include Tier 3 (Engineering Practices)")


class GraphNode(BaseModel):
    """A node in the traversal path (book + chapter)."""
    
    book: str = Field(..., description="Book title")
    chapter: int = Field(..., ge=1, description=_DESC_CHAPTER_NUMBER)
    tier: int = Field(..., ge=1, le=3, description=_DESC_TIER_LEVEL)
    title: str = Field(default="", description=_DESC_CHAPTER_TITLE)
    relationship: RelationshipType | None = Field(default=None, description="Relationship from previous node")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity to source")


class TraversalPath(BaseModel):
    """A complete traversal path through the taxonomy graph."""
    
    nodes: list[GraphNode] = Field(default_factory=list, description="Nodes in traversal order")
    total_similarity: float = Field(default=0.0, ge=0.0, description="Aggregate similarity score")
    path_type: str = Field(default="linear", description="Path type (linear, non_linear, cyclic)")


class ChapterMatch(BaseModel):
    """A matched chapter from search/traversal."""
    
    book: str = Field(..., description="Book title")
    chapter: int = Field(..., description=_DESC_CHAPTER_NUMBER)
    title: str = Field(..., description=_DESC_CHAPTER_TITLE)
    tier: int = Field(..., ge=1, le=3, description=_DESC_TIER_LEVEL)
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    keywords: list[str] = Field(default_factory=list, description="Matched keywords")
    relevance_reason: str = Field(default="", description="Why this chapter is relevant")
    content: str | None = Field(default=None, description="Retrieved content (if requested)")
    page_range: str | None = Field(default=None, description="Page range if available")


class Citation(BaseModel):
    """A Chicago-style citation for cross-reference output.
    
    Format: Author, *Book Title*, Chapter N, pp. X-Y.
    """
    
    author: str | None = Field(default=None, description="Author name(s)")
    book: str = Field(..., description="Book title (italicized in output)")
    chapter: int = Field(..., description=_DESC_CHAPTER_NUMBER)
    chapter_title: str = Field(default="", description=_DESC_CHAPTER_TITLE)
    pages: str | None = Field(default=None, description="Page range (e.g., '33-58')")
    tier: int = Field(..., ge=1, le=3, description=f"{_DESC_TIER_LEVEL} for ordering")
    
    def to_chicago_format(self, footnote_number: int) -> str:
        """Format citation in Chicago style.
        
        Args:
            footnote_number: Footnote number for the citation
            
        Returns:
            Formatted citation string
        """
        parts = []
        if self.author:
            parts.append(self.author)
        parts.append(f"*{self.book}*")
        if self.chapter_title:
            parts.append(f'"{self.chapter_title}"')
        parts.append(f"Ch. {self.chapter}")
        if self.pages:
            parts.append(f"pp. {self.pages}")
        
        citation_text = ", ".join(parts)
        return f"[^{footnote_number}]: {citation_text}."


class TierCoverage(BaseModel):
    """Coverage statistics for each tier in the cross-reference."""
    
    tier: int = Field(..., ge=1, le=3, description=_DESC_TIER_LEVEL)
    tier_name: str = Field(..., description="Tier name (e.g., 'Architecture Spine')")
    books_referenced: int = Field(default=0, ge=0, description="Number of books referenced")
    chapters_referenced: int = Field(default=0, ge=0, description="Number of chapters referenced")
    has_coverage: bool = Field(default=False, description="Whether this tier has coverage")


class CrossReferenceResult(BaseModel):
    """Output model for cross-reference agent.
    
    Contains the scholarly annotation, citations, and metadata
    about the cross-referencing process.
    """
    
    annotation: str = Field(..., description="Scholarly annotation with inline citations")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")
    traversal_paths: list[TraversalPath] = Field(default_factory=list, description="Paths followed during traversal")
    tier_coverage: list[TierCoverage] = Field(default_factory=list, description="Coverage per tier")
    matches: list[ChapterMatch] = Field(default_factory=list, description="All matched chapters")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in milliseconds")
    model_used: str = Field(default="", description="LLM model used for synthesis")


class CrossReferenceState(BaseModel):
    """LangGraph state model for Cross-Reference Agent workflow.
    
    This state flows through the workflow nodes:
    analyze_source → search_taxonomy → traverse_graph → retrieve_content → synthesize
    
    Pattern: LangGraph StateGraph state model
    Source: ARCHITECTURE.md (ai-agents), Generative AI with LangChain 2e
    """
    
    # Input
    source: SourceChapter = Field(..., description="Source chapter to cross-reference")
    config: TraversalConfig = Field(default_factory=TraversalConfig, description="Traversal configuration")
    taxonomy_id: str = Field(default="ai-ml", description="Taxonomy identifier")
    
    # Processing state (populated by workflow nodes)
    analyzed_concepts: list[str] = Field(default_factory=list, description="Concepts extracted from source")
    taxonomy_matches: list[ChapterMatch] = Field(default_factory=list, description="Matches from taxonomy search")
    traversal_paths: list[TraversalPath] = Field(default_factory=list, description="Completed traversal paths")
    retrieved_chapters: list[ChapterMatch] = Field(default_factory=list, description="Chapters with content retrieved")
    validated_matches: list[ChapterMatch] = Field(default_factory=list, description="Matches validated as relevant")
    
    # Output (populated by synthesize node)
    result: CrossReferenceResult | None = Field(default=None, description="Final result")
    
    # Metadata
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Workflow start time")
    current_node: str = Field(default="", description="Current workflow node")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during processing")
    
    class Config:
        """Pydantic config."""
        
        arbitrary_types_allowed = True
