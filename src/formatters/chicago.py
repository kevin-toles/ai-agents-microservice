"""Chicago citation formatter.

Formats citations according to Chicago Manual of Style 17th Edition.
Supports footnote format for scholarly annotations.

Pattern: Citation Formatting
Source: GRAPH_RAG_POC_PLAN WBS 5.10-5.11, TIER_RELATIONSHIP_DIAGRAM.md

Chicago Manual of Style formats:
- Footnote: Author First Last, *Book Title*, chapter info (Place: Publisher, Year), pages.
- Bibliography: Author Last, First. *Book Title*. Place: Publisher, Year.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChicagoCitation(BaseModel):
    """Chicago-style citation data model.
    
    Holds all metadata needed to format a citation in Chicago style.
    Supports both footnote and bibliography formats.
    """
    
    author: str | None = Field(default=None, description="Author name(s)")
    title: str = Field(..., min_length=1, description="Book title")
    chapter_title: str | None = Field(default=None, description="Chapter title")
    chapter_number: int = Field(..., ge=1, description="Chapter number")
    pages: str | None = Field(default=None, description="Page range (e.g., '89-112')")
    publisher: str | None = Field(default=None, description="Publisher name")
    year: int | None = Field(default=None, description="Publication year")
    tier: int = Field(..., ge=1, le=3, description="Taxonomy tier (1-3)")
    place: str | None = Field(default=None, description="Publication place")
    
    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate title is not empty."""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v
    
    @field_validator("tier")
    @classmethod
    def tier_in_range(cls, v: int) -> int:
        """Validate tier is 1, 2, or 3."""
        if v not in (1, 2, 3):
            raise ValueError("Tier must be 1, 2, or 3")
        return v


class ChicagoFormatter:
    """Formatter for Chicago Manual of Style citations.
    
    Produces footnotes, bibliography entries, and inline references
    following Chicago 17th Edition guidelines.
    """
    
    # Tier names for headers
    TIER_NAMES = {
        1: "Architecture Spine",
        2: "Implementation",
        3: "Engineering Practices",
    }
    
    def format_footnote(
        self,
        citation: ChicagoCitation,
        footnote_number: int,
    ) -> str:
        """Format a citation as a numbered footnote.
        
        Chicago footnote format:
        [^N]: Author, *Book Title*, "Chapter Title," Ch. N, pp. X-Y.
        
        Args:
            citation: Citation data
            footnote_number: Footnote number (1-indexed)
            
        Returns:
            Formatted footnote string
        """
        parts: list[str] = []
        
        # Author (if present)
        if citation.author:
            parts.append(citation.author)
        
        # Book title (italicized with asterisks for Markdown)
        parts.append(f"*{citation.title}*")
        
        # Chapter title (in quotes per Chicago style)
        if citation.chapter_title:
            parts.append(f'"{citation.chapter_title}"')
        
        # Chapter number
        parts.append(f"Ch. {citation.chapter_number}")
        
        # Pages (if present)
        if citation.pages:
            parts.append(f"pp. {citation.pages}")
        
        # Join with commas
        citation_text = ", ".join(parts)
        
        return f"[^{footnote_number}]: {citation_text}."
    
    def format_bibliography_entry(self, citation: ChicagoCitation) -> str:
        """Format a citation as a bibliography entry.
        
        Chicago bibliography format:
        Author Last, First. *Book Title*. Place: Publisher, Year.
        
        Args:
            citation: Citation data
            
        Returns:
            Formatted bibliography entry
        """
        parts: list[str] = []
        
        # Author
        if citation.author:
            parts.append(f"{citation.author}.")
        
        # Book title (italicized)
        parts.append(f"*{citation.title}*.")
        
        # Publisher info
        pub_parts: list[str] = []
        if citation.place:
            pub_parts.append(citation.place)
        if citation.publisher:
            pub_parts.append(citation.publisher)
        if citation.year:
            pub_parts.append(str(citation.year))
        
        if pub_parts:
            parts.append(", ".join(pub_parts) + ".")
        
        return " ".join(parts)
    
    def format_inline_reference(self, footnote_number: int) -> str:
        """Format an inline footnote reference marker.
        
        Args:
            footnote_number: Footnote number
            
        Returns:
            Inline reference marker (e.g., "[^5]")
        """
        return f"[^{footnote_number}]"
    
    def format_citations(
        self,
        citations: list[ChicagoCitation],
    ) -> str:
        """Format multiple citations as footnotes.
        
        Sorts citations by tier (1, 2, 3) before formatting.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted footnotes as a single string
        """
        # Sort by tier
        sorted_citations = sorted(citations, key=lambda c: c.tier)
        
        # Format each citation
        footnotes: list[str] = []
        for i, citation in enumerate(sorted_citations, start=1):
            footnotes.append(self.format_footnote(citation, i))
        
        return "\n".join(footnotes)
    
    def format_citations_by_tier(
        self,
        citations: list[ChicagoCitation],
    ) -> str:
        """Format citations grouped by tier with headers.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citations with tier group headers
        """
        # Group by tier
        tier_groups: dict[int, list[ChicagoCitation]] = {1: [], 2: [], 3: []}
        for citation in citations:
            tier_groups[citation.tier].append(citation)
        
        # Format each tier group
        result_parts: list[str] = []
        footnote_num = 1
        
        for tier in [1, 2, 3]:
            tier_citations = tier_groups[tier]
            if not tier_citations:
                continue
            
            # Add tier header
            tier_name = self.TIER_NAMES.get(tier, f"Tier {tier}")
            result_parts.append(f"### {tier_name} (Tier {tier})")
            result_parts.append("")
            
            # Add citations for this tier
            for citation in tier_citations:
                result_parts.append(self.format_footnote(citation, footnote_num))
                footnote_num += 1
            
            result_parts.append("")
        
        return "\n".join(result_parts).strip()


# Module-level convenience functions

def format_citation(citation: ChicagoCitation, footnote_number: int) -> str:
    """Format a citation as a footnote.
    
    Convenience function using default ChicagoFormatter.
    
    Args:
        citation: Citation data
        footnote_number: Footnote number
        
    Returns:
        Formatted footnote string
    """
    formatter = ChicagoFormatter()
    return formatter.format_footnote(citation, footnote_number)


def format_footnote(citation: ChicagoCitation, footnote_number: int) -> str:
    """Format a citation as a footnote.
    
    Alias for format_citation.
    
    Args:
        citation: Citation data
        footnote_number: Footnote number
        
    Returns:
        Formatted footnote string
    """
    return format_citation(citation, footnote_number)


def format_bibliography_entry(citation: ChicagoCitation) -> str:
    """Format a citation as a bibliography entry.
    
    Convenience function using default ChicagoFormatter.
    
    Args:
        citation: Citation data
        
    Returns:
        Formatted bibliography entry
    """
    formatter = ChicagoFormatter()
    return formatter.format_bibliography_entry(citation)
