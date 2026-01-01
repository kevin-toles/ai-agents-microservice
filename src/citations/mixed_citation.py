"""Mixed Citation Model for Unified Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.6)
Acceptance Criteria:
- AC-24.4: Returns unified RetrievalResult with mixed citations

Provides a unified citation model that can represent citations from any source:
- Code Reference Engine (code)
- Neo4j Graph (graph)
- Book Passages (book)

Supports Chicago-style footnote generation with source-type identification.

Pattern: Mapper/Transformer (CODING_PATTERNS_ANALYSIS.md)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.citations.book_citation import BookCitation
    from src.citations.code_citation import CodeCitation
    from src.citations.graph_citation import GraphCitation
    from src.schemas.retrieval_models import RetrievalItem


# =============================================================================
# Source Type Enum (Re-exported for convenience)
# =============================================================================


class SourceType(str, Enum):
    """Source type for citations.
    
    Mirrors retrieval_models.SourceType for citation context.
    """
    
    CODE = "code"
    BOOK = "book"
    GRAPH = "graph"
    SEMANTIC = "semantic"


# =============================================================================
# Mixed Citation Model
# =============================================================================


@dataclass
class MixedCitation:
    """Unified citation model for any knowledge source.
    
    WBS: AGT24.6 - Create MixedCitation model
    AC-24.4: Returns unified RetrievalResult with mixed citations
    
    Provides a consistent interface for citations regardless of source type,
    enabling uniform footnote generation and source tracking.
    
    Attributes:
        source_type: Origin of the citation (code, book, graph)
        source_id: Unique identifier within source
        display_text: Human-readable citation text
        footnote_number: Optional footnote number (1-indexed)
        url: Optional URL for the source
        metadata: Source-specific metadata
    """
    
    source_type: SourceType
    source_id: str
    display_text: str
    footnote_number: int | None = None
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_footnote(self) -> str:
        """Generate footnote marker and text.
        
        Format: [^N]: citation text
        
        Returns:
            Formatted footnote string
        """
        number = self.footnote_number or 1
        return f"[^{number}]: {self.display_text}"
    
    def to_inline_marker(self) -> str:
        """Generate inline citation marker.
        
        Returns:
            Inline marker like [^1]
        """
        number = self.footnote_number or 1
        return f"[^{number}]"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "display_text": self.display_text,
            "footnote_number": self.footnote_number,
            "url": self.url,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MixedCitation:
        """Create from dictionary."""
        return cls(
            source_type=SourceType(data["source_type"]),
            source_id=data["source_id"],
            display_text=data["display_text"],
            footnote_number=data.get("footnote_number"),
            url=data.get("url", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Conversion Functions
# =============================================================================


def from_code_citation(
    citation: CodeCitation,
    footnote_number: int | None = None,
) -> MixedCitation:
    """Convert CodeCitation to MixedCitation.
    
    Args:
        citation: Code citation to convert
        footnote_number: Optional footnote number
    
    Returns:
        MixedCitation with code source type
    """
    display = citation.to_markdown() if hasattr(citation, 'to_markdown') else str(citation)
    
    return MixedCitation(
        source_type=SourceType.CODE,
        source_id=citation.file_path,
        display_text=display,
        footnote_number=footnote_number,
        url=citation.url,
        metadata={
            "repo_id": citation.repo_id,
            "start_line": citation.start_line,
            "end_line": citation.end_line,
            "language": citation.language,
        },
    )


def from_book_citation(
    citation: BookCitation,
    footnote_number: int | None = None,
) -> MixedCitation:
    """Convert BookCitation to MixedCitation.
    
    Args:
        citation: Book citation to convert
        footnote_number: Optional footnote number
    
    Returns:
        MixedCitation with book source type
    """
    display = citation.to_chicago() if hasattr(citation, 'to_chicago') else str(citation)
    
    return MixedCitation(
        source_type=SourceType.BOOK,
        source_id=citation.source_id,
        display_text=display,
        footnote_number=footnote_number,
        url="",  # Books typically don't have URLs
        metadata={
            "author": citation.author,
            "book_title": citation.book_title,
            "chapter_number": citation.chapter_number,
            "start_page": citation.start_page,
            "end_page": citation.end_page,
        },
    )


def from_graph_citation(
    citation: GraphCitation,
    footnote_number: int | None = None,
) -> MixedCitation:
    """Convert GraphCitation to MixedCitation.
    
    Args:
        citation: Graph citation to convert
        footnote_number: Optional footnote number
    
    Returns:
        MixedCitation with graph source type
    """
    display = citation.to_markdown() if hasattr(citation, 'to_markdown') else str(citation)
    
    return MixedCitation(
        source_type=SourceType.GRAPH,
        source_id=citation.node_id,
        display_text=display,
        footnote_number=footnote_number,
        url=citation.github_url or "",
        metadata={
            "node_type": citation.node_type,
            "name": citation.name,
            "tier": citation.tier,
        },
    )


def from_retrieval_item(
    item: RetrievalItem,
    footnote_number: int | None = None,
) -> MixedCitation:
    """Convert RetrievalItem to MixedCitation.
    
    Creates a citation directly from a retrieval result item.
    
    Args:
        item: Retrieval item to convert
        footnote_number: Optional footnote number
    
    Returns:
        MixedCitation matching the item's source type
    """
    from src.schemas.retrieval_models import SourceType as RetrievalSourceType
    
    # Map retrieval source type to citation source type
    source_type_map = {
        RetrievalSourceType.CODE: SourceType.CODE,
        RetrievalSourceType.BOOK: SourceType.BOOK,
        RetrievalSourceType.GRAPH: SourceType.GRAPH,
        RetrievalSourceType.SEMANTIC: SourceType.SEMANTIC,
    }
    
    source_type = source_type_map.get(item.source_type, SourceType.SEMANTIC)
    
    # Generate display text based on source type
    if source_type == SourceType.BOOK:
        display = _format_book_display(item)
    elif source_type == SourceType.CODE:
        display = _format_code_display(item)
    elif source_type == SourceType.GRAPH:
        display = _format_graph_display(item)
    else:
        display = item.title or item.source_id
    
    return MixedCitation(
        source_type=source_type,
        source_id=item.source_id,
        display_text=display,
        footnote_number=footnote_number,
        url=item.metadata.get("url", ""),
        metadata=dict(item.metadata),
    )


def _format_book_display(item: RetrievalItem) -> str:
    """Format book citation display text."""
    author = item.metadata.get("author", "")
    title = item.metadata.get("book_title", item.title)
    pages = ""
    
    start = item.metadata.get("start_page")
    end = item.metadata.get("end_page")
    if start and end:
        pages = f", {start}-{end}" if start != end else f", {start}"
    
    if author and title:
        return f"{author}, {title}{pages}."
    return item.title or item.source_id


def _format_code_display(item: RetrievalItem) -> str:
    """Format code citation display text."""
    file_path = item.metadata.get("file_path", item.source_id)
    start = item.metadata.get("start_line", 1)
    end = item.metadata.get("end_line", start)
    
    return f"{file_path}#L{start}-L{end}"


def _format_graph_display(item: RetrievalItem) -> str:
    """Format graph citation display text."""
    name = item.metadata.get("name", item.title)
    node_type = item.metadata.get("node_type", "Concept")
    tier = item.metadata.get("tier")
    
    tier_str = f" (Tier {tier})" if tier else ""
    return f"[{node_type}: {name}{tier_str}]"


# =============================================================================
# Batch Conversion
# =============================================================================


def citations_from_retrieval_items(
    items: list[RetrievalItem],
    start_number: int = 1,
) -> list[MixedCitation]:
    """Convert list of retrieval items to citations.
    
    Assigns sequential footnote numbers starting from start_number.
    
    Args:
        items: List of retrieval items
        start_number: Starting footnote number
    
    Returns:
        List of MixedCitation objects
    """
    return [
        from_retrieval_item(item, footnote_number=start_number + i)
        for i, item in enumerate(items)
    ]
