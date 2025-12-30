"""Citation and provenance schemas for ai-agents.

Implements AC-4.1, AC-4.2, AC-4.3 from WBS-AGT4.

Models:
- SourceMetadata: Provenance tracking for different source types
- Citation: Individual citation with Chicago-style formatting
- CitedContent: Content with embedded [^N] markers and footnotes

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
Chicago Format Templates from architecture doc:
- Book: `[^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.`
- Code: `[^N]: \`repo/path/file.py\`, commit \`hash\`, lines X-Y.`
- Schema: `[^N]: \`repo/schemas/file.json\`, version X.Y.Z.`
- Internal Doc: `[^N]: service, *Document* (Date), §Section.`

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments (uses Field(default_factory=list))
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class SourceType(str, Enum):
    """Valid source types for citations.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Format Templates
    """
    
    BOOK = "book"
    CODE = "code"
    SCHEMA = "schema"
    INTERNAL_DOC = "internal_doc"


# =============================================================================
# AC-4.1: SourceMetadata
# =============================================================================

class SourceMetadata(BaseModel):
    """Provenance metadata for citation sources.
    
    Captures source-specific fields for different types:
    - book: author, title, publisher, year, pages
    - code: repo, file_path, line_range, commit_hash
    - schema: repo, file_path, version
    - internal_doc: service, title, section, date
    
    Attributes:
        source_type: Type of source (book, code, schema, internal_doc)
        title: Title of the source (required for all types)
        author: Author name (books)
        publisher: Publisher name (books)
        publication_city: City of publication (books)
        year: Publication year (books)
        pages: Page range (books)
        repo: Repository name (code, schema)
        file_path: Path within repository (code, schema)
        line_range: Line numbers (code)
        commit_hash: Git commit hash (code)
        version: Schema version (schema)
        service: Service name (internal_doc)
        section: Document section (internal_doc)
        date: Document date (internal_doc)
        similarity_score: Retrieval relevance score (0.0-1.0)
    
    Example:
        >>> metadata = SourceMetadata(
        ...     source_type="book",
        ...     author="Fowler, Martin",
        ...     title="Patterns of Enterprise Application Architecture",
        ...     publisher="Addison-Wesley",
        ...     year=2002,
        ...     pages="322-327",
        ... )
    """
    
    # Required field
    source_type: SourceType = Field(
        ...,
        description="Type of source: book, code, schema, or internal_doc",
    )
    
    # Common field (required for display)
    title: str | None = Field(
        default=None,
        description="Title of the source document or file",
    )
    
    # Book-specific fields
    author: str | None = Field(
        default=None,
        description="Author name in 'LastName, FirstName' format",
    )
    publisher: str | None = Field(
        default=None,
        description="Publisher name",
    )
    publication_city: str | None = Field(
        default=None,
        description="City of publication",
    )
    year: int | None = Field(
        default=None,
        description="Publication year",
        ge=1800,
        le=2100,
    )
    pages: str | None = Field(
        default=None,
        description="Page range, e.g., '322-327'",
    )
    
    # Code/Schema-specific fields
    repo: str | None = Field(
        default=None,
        description="Repository name",
    )
    file_path: str | None = Field(
        default=None,
        description="Path to file within repository",
    )
    line_range: str | None = Field(
        default=None,
        description="Line numbers, e.g., '12-45'",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Git commit hash",
    )
    
    # Schema-specific fields
    version: str | None = Field(
        default=None,
        description="Schema version, e.g., '1.2.0'",
    )
    
    # Internal doc-specific fields
    service: str | None = Field(
        default=None,
        description="Service name for internal docs",
    )
    section: str | None = Field(
        default=None,
        description="Document section reference",
    )
    date: str | None = Field(
        default=None,
        description="Document date in ISO format",
    )
    
    # Retrieval metadata
    similarity_score: float | None = Field(
        default=None,
        description="Semantic similarity score from retrieval (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_type": "book",
                    "author": "Fowler, Martin",
                    "title": "Patterns of Enterprise Application Architecture",
                    "publisher": "Addison-Wesley",
                    "publication_city": "Boston",
                    "year": 2002,
                    "pages": "322-327",
                },
                {
                    "source_type": "code",
                    "repo": "code-reference-engine",
                    "file_path": "backend/ddd/repository.py",
                    "line_range": "12-45",
                    "commit_hash": "a1b2c3d",
                },
            ]
        }
    }


# =============================================================================
# AC-4.2: Citation
# =============================================================================

class Citation(BaseModel):
    """Individual citation with Chicago-style formatting.
    
    Captures the relationship between a footnote marker ([^N]) and its
    source metadata. Provides formatting methods for different output styles.
    
    Attributes:
        marker: Footnote marker number (positive integer)
        source: Source metadata for the citation
        context: Optional context/excerpt from the source
    
    Example:
        >>> from src.schemas.citations import Citation, SourceMetadata
        >>> source = SourceMetadata(
        ...     source_type="book",
        ...     author="Fowler, Martin",
        ...     title="PEAA",
        ...     year=2002,
        ... )
        >>> citation = Citation(marker=1, source=source)
        >>> citation.chicago_format()
        'Fowler, Martin, *PEAA* (2002).'
    """
    
    marker: int = Field(
        ...,
        description="Footnote marker number (positive integer)",
        gt=0,
    )
    source: SourceMetadata = Field(
        ...,
        description="Source metadata for the citation",
    )
    context: str | None = Field(
        default=None,
        description="Context or excerpt from the source",
    )
    
    @property
    def marker_format(self) -> str:
        """Return the inline marker format [^N].
        
        Returns:
            Formatted marker string, e.g., '[^1]'
        """
        return f"[^{self.marker}]"
    
    def chicago_format(self) -> str:
        """Format citation in Chicago style.
        
        Returns Chicago-style formatted citation based on source type:
        - Book: LastName, FirstName, *Title* (City: Publisher, Year), pages.
        - Code: `repo/path/file.py`, commit `hash`, lines X-Y.
        - Schema: `repo/schemas/file.json`, version X.Y.Z.
        - Internal Doc: service, *Document* (Date), §Section.
        
        Returns:
            Chicago-style formatted citation string
        """
        src = self.source
        
        if src.source_type == SourceType.BOOK:
            return self._format_book()
        elif src.source_type == SourceType.CODE:
            return self._format_code()
        elif src.source_type == SourceType.SCHEMA:
            return self._format_schema()
        elif src.source_type == SourceType.INTERNAL_DOC:
            return self._format_internal_doc()
        else:
            # Fallback for unknown types
            return f"{src.title or 'Unknown source'}."
    
    def _format_book(self) -> str:
        """Format book citation in Chicago style."""
        src = self.source
        parts: list[str] = []
        
        # Author
        if src.author:
            parts.append(src.author)
        
        # Title (italicized with *)
        if src.title:
            parts.append(f"*{src.title}*")
        
        # Publication info: (City: Publisher, Year)
        pub_parts: list[str] = []
        if src.publication_city:
            pub_parts.append(src.publication_city)
        if src.publisher:
            if pub_parts:
                pub_parts.append(f": {src.publisher}")
            else:
                pub_parts.append(src.publisher)
        if src.year:
            if pub_parts:
                pub_parts.append(f", {src.year}")
            else:
                pub_parts.append(str(src.year))
        
        if pub_parts:
            parts.append(f"({''.join(pub_parts)})")
        
        # Pages
        if src.pages:
            parts.append(src.pages)
        
        result = ", ".join(parts)
        if result and not result.endswith("."):
            result += "."
        
        return result
    
    def _format_code(self) -> str:
        """Format code citation."""
        src = self.source
        parts: list[str] = []
        
        # File path with repo
        if src.repo and src.file_path:
            parts.append(f"`{src.repo}/{src.file_path}`")
        elif src.file_path:
            parts.append(f"`{src.file_path}`")
        
        # Commit hash
        if src.commit_hash:
            parts.append(f"commit `{src.commit_hash}`")
        
        # Line range
        if src.line_range:
            parts.append(f"lines {src.line_range}")
        
        result = ", ".join(parts)
        if result and not result.endswith("."):
            result += "."
        
        return result
    
    def _format_schema(self) -> str:
        """Format schema citation."""
        src = self.source
        parts: list[str] = []
        
        # File path with repo
        if src.repo and src.file_path:
            parts.append(f"`{src.repo}/{src.file_path}`")
        elif src.file_path:
            parts.append(f"`{src.file_path}`")
        
        # Version
        if src.version:
            parts.append(f"version {src.version}")
        
        result = ", ".join(parts)
        if result and not result.endswith("."):
            result += "."
        
        return result
    
    def _format_internal_doc(self) -> str:
        """Format internal document citation."""
        src = self.source
        parts: list[str] = []
        
        # Service name
        if src.service:
            parts.append(src.service)
        
        # Document title (italicized)
        if src.title:
            parts.append(f"*{src.title}*")
        
        # Date
        if src.date:
            parts.append(f"({src.date})")
        
        # Section
        if src.section:
            parts.append(f"§{src.section}")
        
        result = ", ".join(parts)
        if result and not result.endswith("."):
            result += "."
        
        return result
    
    @property
    def footnote_format(self) -> str:
        """Return complete footnote format [^N]: Chicago citation.
        
        Returns:
            Complete footnote string
        """
        return f"[^{self.marker}]: {self.chicago_format()}"
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "marker": 1,
                    "source": {
                        "source_type": "book",
                        "author": "Fowler, Martin",
                        "title": "Patterns of Enterprise Application Architecture",
                        "publisher": "Addison-Wesley",
                        "year": 2002,
                    },
                    "context": "Repository pattern implementation",
                }
            ]
        }
    }


# =============================================================================
# AC-4.3: CitedContent
# =============================================================================

class CitedContent(BaseModel):
    """Content with embedded [^N] citation markers.
    
    Captures generated text containing inline citation markers and
    their corresponding citations. Provides methods to render the
    content with footnotes.
    
    Attributes:
        text: Content text with [^N] markers
        citations: List of citations corresponding to markers
    
    Example:
        >>> content = CitedContent(
        ...     text="The Repository pattern[^1] provides abstraction.",
        ...     citations=[citation],
        ... )
        >>> content.render()
        'The Repository pattern[^1] provides abstraction.\\n\\n[^1]: Fowler...'
    """
    
    text: str = Field(
        ...,
        description="Content text with embedded [^N] citation markers",
    )
    # AP-1.5: Use Field(default_factory=list) instead of default=[]
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of citations corresponding to markers in text",
    )
    
    @property
    def footnotes(self) -> list[str]:
        """Return list of formatted footnotes.
        
        Returns:
            List of footnote strings in format '[^N]: citation'
        """
        return [c.footnote_format for c in sorted(self.citations, key=lambda c: c.marker)]
    
    def render(self, separator: str = "\n\n") -> str:
        """Render content with appended footnotes.
        
        Args:
            separator: String to separate text from footnotes
        
        Returns:
            Complete rendered content with footnotes
        """
        if not self.citations:
            return self.text
        
        footnotes_text = "\n".join(self.footnotes)
        return f"{self.text}{separator}{footnotes_text}"
    
    def extract_markers(self) -> list[int]:
        """Extract citation marker numbers in order of appearance.
        
        Returns:
            List of marker numbers in text order
        """
        pattern = r'\[\^(\d+)\]'
        matches = re.findall(pattern, self.text)
        return [int(m) for m in matches]
    
    def get_citation(self, marker: int) -> Citation | None:
        """Get citation by marker number.
        
        Args:
            marker: The marker number to look up
        
        Returns:
            Citation if found, None otherwise
        """
        for citation in self.citations:
            if citation.marker == marker:
                return citation
        return None
    
    def validate_markers(self) -> tuple[list[int], list[int]]:
        """Validate that markers in text match citations.
        
        Returns:
            Tuple of (missing_citations, orphan_markers) lists
        """
        text_markers = set(self.extract_markers())
        citation_markers = {c.marker for c in self.citations}
        
        missing = sorted(text_markers - citation_markers)
        orphans = sorted(citation_markers - text_markers)
        
        return missing, orphans
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "The Repository pattern[^1] provides a collection-like interface.",
                    "citations": [
                        {
                            "marker": 1,
                            "source": {
                                "source_type": "book",
                                "author": "Fowler, Martin",
                                "title": "PEAA",
                                "year": 2002,
                            },
                        }
                    ],
                }
            ]
        }
    }


# =============================================================================
# Utility Functions
# =============================================================================

def create_citation_from_retrieval(
    retrieval_result: dict[str, Any],
    marker: int,
) -> Citation:
    """Create a Citation from a semantic search retrieval result.
    
    Maps retrieval response fields to SourceMetadata and creates
    a Citation with the given marker number.
    
    Args:
        retrieval_result: Dict from semantic-search-service response
        marker: Footnote marker number to assign
    
    Returns:
        Citation with populated source metadata
    
    Example:
        >>> result = {"source_type": "book", "author": "Fowler", ...}
        >>> citation = create_citation_from_retrieval(result, marker=1)
    """
    source = SourceMetadata(
        source_type=retrieval_result.get("source_type", "book"),
        title=retrieval_result.get("title"),
        author=retrieval_result.get("author"),
        publisher=retrieval_result.get("publisher"),
        publication_city=retrieval_result.get("publication_city"),
        year=retrieval_result.get("year"),
        pages=retrieval_result.get("pages"),
        repo=retrieval_result.get("repo"),
        file_path=retrieval_result.get("file_path"),
        line_range=retrieval_result.get("line_range"),
        commit_hash=retrieval_result.get("commit_hash"),
        version=retrieval_result.get("version"),
        service=retrieval_result.get("service"),
        section=retrieval_result.get("section"),
        date=retrieval_result.get("date"),
        similarity_score=retrieval_result.get("similarity"),
    )
    
    return Citation(
        marker=marker,
        source=source,
        context=retrieval_result.get("chunk"),
    )


def merge_citations(contents: list[CitedContent]) -> CitedContent:
    """Merge multiple CitedContent instances, renumbering markers.
    
    Combines multiple content blocks into one, ensuring citation
    markers are unique and sequential across the merged content.
    
    Args:
        contents: List of CitedContent to merge
    
    Returns:
        Single CitedContent with renumbered markers
    
    Example:
        >>> merged = merge_citations([content1, content2])
        >>> # content1's [^1] stays [^1], content2's [^1] becomes [^2]
    """
    if not contents:
        return CitedContent(text="", citations=[])
    
    merged_text_parts: list[str] = []
    merged_citations: list[Citation] = []
    current_marker = 1
    
    for content in contents:
        # Build marker remapping for this content
        marker_map: dict[int, int] = {}
        for citation in content.citations:
            marker_map[citation.marker] = current_marker
            # Create new citation with updated marker
            merged_citations.append(
                Citation(
                    marker=current_marker,
                    source=citation.source,
                    context=citation.context,
                )
            )
            current_marker += 1
        
        # Remap markers in text
        remapped_text = content.text
        for old_marker, new_marker in sorted(marker_map.items(), reverse=True):
            remapped_text = remapped_text.replace(
                f"[^{old_marker}]",
                f"[^{new_marker}]"
            )
        
        merged_text_parts.append(remapped_text)
    
    return CitedContent(
        text="\n\n".join(merged_text_parts),
        citations=merged_citations,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "SourceType",
    # Models
    "SourceMetadata",
    "Citation",
    "CitedContent",
    # Utilities
    "create_citation_from_retrieval",
    "merge_citations",
]
