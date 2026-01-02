"""Chicago-Style Citation Formatter.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Task: KB5.8 - Implement Chicago formatter for all citation types

Acceptance Criteria:
- AC-KB5.6: Chicago-style footnotes generated for all citation types

Citation Types:
- book: LastName, FirstName, *Title* (City: Publisher, Year), pages.
- code: `repo/path/file.py`, commit `hash`, lines X-Y.
- schema: `repo/schemas/file.json`, version X.Y.Z.
- graph: GraphDB node_type#node_id, relationship.
- internal_doc: service, *Document* (Date), §Section.

Anti-Patterns Avoided:
- S1192: String constants at module level
- S3776: Low cognitive complexity via dispatch pattern
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_BOOK_TEMPLATE = "{author}, *{title}* ({city}: {publisher}, {year})"
_CONST_CODE_TEMPLATE = "`{file_path}`, commit `{commit_hash}`, lines {lines}"
_CONST_SCHEMA_TEMPLATE = "`{file_path}`, version {version}"
_CONST_GRAPH_TEMPLATE = "GraphDB {node_type}#{node_id}"
_CONST_INTERNAL_DOC_TEMPLATE = "{service}, *{document}*, §{section}"


# =============================================================================
# CitationType Enum
# =============================================================================


class CitationType(Enum):
    """Types of citations supported by the formatter.
    
    Each type has a specific template and required fields.
    """

    BOOK = "book"
    CODE = "code"
    SCHEMA = "schema"
    GRAPH = "graph"
    INTERNAL_DOC = "internal_doc"


# =============================================================================
# CitationData Dataclass
# =============================================================================


@dataclass(frozen=True, slots=True)
class CitationData:
    """Data for a single citation.
    
    Attributes:
        citation_type: The type of citation
        marker: The footnote marker number
        source: Raw source identifier
        metadata: Type-specific metadata for formatting
    """

    citation_type: CitationType
    marker: int
    source: str
    metadata: dict[str, Any]


# =============================================================================
# ChicagoCitationFormatter (AC-KB5.6)
# =============================================================================


class ChicagoCitationFormatter:
    """Formats citations in Chicago style.
    
    AC-KB5.6: Chicago-style footnotes generated for all citation types.
    
    Supports five citation types with specific templates:
    - book: LastName, FirstName, *Title* (City: Publisher, Year), pages.
    - code: `repo/path/file.py`, commit `hash`, lines X-Y.
    - schema: `repo/schemas/file.json`, version X.Y.Z.
    - graph: GraphDB node_type#node_id
    - internal_doc: service, *Document* (Date), §Section.
    
    Example:
        >>> formatter = ChicagoCitationFormatter()
        >>> footnote = formatter.format_book(
        ...     marker=1,
        ...     author="Fowler, Martin",
        ...     title="PEAA",
        ...     year=2002,
        ... )
        >>> print(footnote)
        [^1]: Fowler, Martin, *PEAA* (2002).
    """

    def __init__(self) -> None:
        """Initialize the formatter."""
        pass

    def format(self, citation: dict[str, Any]) -> str:
        """Format a citation based on its type.
        
        Auto-detects citation type from the 'type' field.
        
        Args:
            citation: Dictionary with type and relevant fields.
            
        Returns:
            Formatted citation string.
        """
        citation_type = citation.get("type", "book")
        marker = citation.get("marker", 0)

        if citation_type == "book":
            return self.format_book(
                marker=marker,
                author=citation.get("author", ""),
                title=citation.get("title", ""),
                year=citation.get("year", 0),
                city=citation.get("city"),
                publisher=citation.get("publisher"),
                pages=citation.get("pages"),
            )
        elif citation_type == "code":
            return self.format_code(
                marker=marker,
                file_path=citation.get("file_path", ""),
                lines=citation.get("lines"),
                commit_hash=citation.get("commit_hash"),
                repo=citation.get("repo"),
            )
        elif citation_type == "schema":
            return self.format_schema(
                marker=marker,
                file_path=citation.get("file_path", ""),
                version=citation.get("version"),
            )
        elif citation_type == "graph":
            return self.format_graph(
                marker=marker,
                node_type=citation.get("node_type", ""),
                node_id=citation.get("node_id", ""),
                relationship=citation.get("relationship"),
                related_node=citation.get("related_node"),
            )
        elif citation_type == "internal_doc":
            return self.format_internal_doc(
                marker=marker,
                service=citation.get("service", ""),
                document=citation.get("document", ""),
                section=citation.get("section"),
            )
        else:
            return f"[^{marker}]: {citation}"

    def format_citation(
        self,
        citation_type: CitationType,
        marker: int,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format a single citation in Chicago style.
        
        Args:
            citation_type: Type of citation
            marker: Footnote marker number
            source: Source identifier
            metadata: Type-specific metadata
            
        Returns:
            Formatted Chicago-style footnote string.
        """
        metadata = metadata or {}

        if citation_type == CitationType.BOOK:
            return self.format_book(
                marker=marker,
                author=metadata.get("author", ""),
                title=source,
                year=metadata.get("year", 0),
                city=metadata.get("city"),
                publisher=metadata.get("publisher"),
                pages=metadata.get("pages"),
            )
        elif citation_type == CitationType.CODE:
            return self.format_code(
                marker=marker,
                file_path=source,
                lines=metadata.get("lines"),
                commit_hash=metadata.get("commit_hash"),
                repo=metadata.get("repo"),
            )
        elif citation_type == CitationType.SCHEMA:
            return self.format_schema(
                marker=marker,
                file_path=source,
                version=metadata.get("version"),
            )
        elif citation_type == CitationType.GRAPH:
            return self.format_graph(
                marker=marker,
                node_type=metadata.get("node_type", ""),
                node_id=metadata.get("node_id", ""),
                relationship=metadata.get("relationship"),
                related_node=metadata.get("related_node"),
            )
        elif citation_type == CitationType.INTERNAL_DOC:
            return self.format_internal_doc(
                marker=marker,
                service=metadata.get("service", ""),
                document=source,
                section=metadata.get("section"),
            )
        else:
            return f"[^{marker}]: {source}"

    def format_book(
        self,
        marker: int,
        author: str,
        title: str,
        year: int | str,
        city: str | None = None,
        publisher: str | None = None,
        pages: str | None = None,
    ) -> str:
        """Format a book citation.
        
        Template: [^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.
        
        Args:
            marker: Footnote marker number
            author: Author name(s) in "LastName, FirstName" format
            title: Book title
            year: Publication year
            city: Publication city (optional)
            publisher: Publisher name (optional)
            pages: Page numbers or range (optional)
            
        Returns:
            Formatted book citation ending with a period.
        """
        parts = [f"{author}", f"*{title}*"]

        # Build the parenthetical part
        if city and publisher:
            parts.append(f"({city}: {publisher}, {year})")
        elif publisher:
            parts.append(f"({publisher}, {year})")
        else:
            parts.append(f"({year})")

        if pages:
            parts.append(f"{pages}")

        citation_text = ", ".join(parts)

        # Ensure it ends with a period
        if not citation_text.endswith("."):
            citation_text += "."

        return f"[^{marker}]: {citation_text}"

    def format_code(
        self,
        marker: int,
        file_path: str,
        lines: str | None = None,
        commit_hash: str | None = None,
        repo: str | None = None,
    ) -> str:
        """Format a code citation.
        
        Template: [^N]: `repo/path/file.py`, commit `hash`, lines X-Y.
        
        Args:
            marker: Footnote marker number
            file_path: File path
            lines: Line numbers or range (optional)
            commit_hash: Git commit hash (optional)
            repo: Repository name (optional)
            
        Returns:
            Formatted code citation.
        """
        parts = []

        # Add repo if provided
        if repo:
            parts.append(f"{repo}")

        # Add file path in backticks
        parts.append(f"`{file_path}`")

        # Add commit hash
        if commit_hash:
            parts.append(f"commit `{commit_hash}`")

        # Add line numbers
        if lines:
            parts.append(f"lines {lines}")

        citation_text = ", ".join(parts)

        # Ensure ends with period
        if not citation_text.endswith("."):
            citation_text += "."

        return f"[^{marker}]: {citation_text}"

    def format_schema(
        self,
        marker: int,
        file_path: str,
        version: str | None = None,
    ) -> str:
        """Format a schema citation.
        
        Template: [^N]: `repo/schemas/file.json`, version X.Y.Z.
        
        Args:
            marker: Footnote marker number
            file_path: Schema file path
            version: Schema version (optional)
            
        Returns:
            Formatted schema citation.
        """
        parts = [f"`{file_path}`"]

        if version:
            parts.append(f"version {version}")

        citation_text = ", ".join(parts)

        if not citation_text.endswith("."):
            citation_text += "."

        return f"[^{marker}]: {citation_text}"

    def format_graph(
        self,
        marker: int,
        node_type: str,
        node_id: str,
        relationship: str | None = None,
        related_node: str | None = None,
    ) -> str:
        """Format a graph citation.
        
        Template: GraphDB node_type#node_id
        
        Args:
            marker: Footnote marker number
            node_type: Type of graph node
            node_id: Node identifier
            relationship: Relationship type (optional)
            related_node: Related node ID (optional)
            
        Returns:
            Formatted graph citation.
        """
        parts = [f"GraphDB {node_type}#{node_id}"]

        if relationship:
            if related_node:
                parts.append(f"{relationship} → {related_node}")
            else:
                parts.append(f"{relationship}")

        citation_text = ", ".join(parts)

        if not citation_text.endswith("."):
            citation_text += "."

        return f"[^{marker}]: {citation_text}"

    def format_internal_doc(
        self,
        marker: int,
        service: str,
        document: str,
        section: str | None = None,
    ) -> str:
        """Format an internal document citation.
        
        Template: [^N]: service, *Document* (Date), §Section.
        
        Args:
            marker: Footnote marker number
            service: Service or project name
            document: Document name
            section: Section reference (optional)
            
        Returns:
            Formatted internal document citation.
        """
        parts = [f"{service}", f"*{document}*"]

        if section:
            parts.append(f"§{section}")

        citation_text = ", ".join(parts)

        if not citation_text.endswith("."):
            citation_text += "."

        return f"[^{marker}]: {citation_text}"

    def format_footnotes(
        self,
        citations: list[dict[str, Any]],
    ) -> str:
        """Format a complete footnotes section.
        
        Args:
            citations: List of citation dictionaries with type and fields.
            
        Returns:
            Complete footnotes section string, newline-separated.
        """
        if not citations:
            return ""

        footnotes = []
        for citation in citations:
            footnote = self.format(citation)
            footnotes.append(footnote)

        return "\n".join(footnotes)

    def format_footnotes_section(
        self,
        citations: list[dict[str, Any]],
    ) -> str:
        """Alias for format_footnotes for API compatibility.
        
        Args:
            citations: List of citation dictionaries.
            
        Returns:
            Complete footnotes section string.
        """
        return self.format_footnotes(citations)


__all__ = [
    "ChicagoCitationFormatter",
    "CitationData",
    "CitationType",
]
