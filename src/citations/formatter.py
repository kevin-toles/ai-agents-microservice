"""Chicago-style citation formatter.

Implements AC-17.2 from WBS-AGT17.

The ChicagoFormatter wraps the existing Citation.chicago_format() method
and provides additional functionality for formatting multiple citations
as a footnotes section.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
Chicago Format Templates:
- Book: [^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.
- Code: [^N]: `repo/path/file.py`, commit `hash`, lines X-Y.
- Schema: [^N]: `repo/schemas/file.json`, version X.Y.Z.
- Internal Doc: [^N]: service, *Document* (Date), §Section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.schemas.citations import Citation


# =============================================================================
# AC-17.2: ChicagoFormatter
# =============================================================================

class ChicagoFormatter:
    """Chicago-style citation formatter.

    Formats citations according to Chicago Manual of Style footnote format.
    Wraps the existing Citation.chicago_format() method and adds the [^N]:
    prefix for markdown footnotes.

    Example:
        >>> from src.citations.formatter import ChicagoFormatter
        >>> from src.schemas.citations import Citation, SourceMetadata, SourceType
        >>> formatter = ChicagoFormatter()
        >>> citation = Citation(
        ...     marker=1,
        ...     source=SourceMetadata(
        ...         source_type=SourceType.BOOK,
        ...         author="Fowler, Martin",
        ...         title="PEAA",
        ...         year=2002,
        ...     ),
        ... )
        >>> formatter.format(citation)
        '[^1]: Fowler, Martin, *PEAA*, (2002).'
    """

    def __init__(self) -> None:
        """Initialize the formatter."""
        pass

    def format(self, citation: Citation) -> str:
        """Format a single citation in Chicago style with footnote marker.

        Combines the [^N]: prefix with the Citation's chicago_format() output.

        Args:
            citation: Citation object to format

        Returns:
            Formatted footnote string, e.g., "[^1]: Author, *Title* (Year)."
        """
        # Use the existing citation.footnote_format property if available
        # Otherwise construct manually
        chicago_text = citation.chicago_format()
        return f"[^{citation.marker}]: {chicago_text}"

    def format_footnotes(self, citations: list[Citation]) -> str:
        """Format multiple citations as a footnotes section.

        Creates a newline-separated list of footnotes suitable for
        appending to markdown documents.

        Args:
            citations: List of Citation objects to format

        Returns:
            Newline-separated footnotes string, or empty string if no citations
        """
        if not citations:
            return ""

        formatted = [self.format(c) for c in citations]
        return "\n".join(formatted)


__all__ = [
    "ChicagoFormatter",
]
