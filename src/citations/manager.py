"""Citation manager for tracking sources through pipeline.

Implements AC-17.1 from WBS-AGT17.

The CitationManager assigns unique [^N] markers to sources and tracks
them through pipeline stages. It supports:
- Adding sources with retrieval scores and context
- Tracking pipeline stage provenance
- Recording usage contexts
- Exporting to audit records

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.audit import CitationAuditRecord
from src.schemas.citations import Citation, SourceMetadata, SourceType


# =============================================================================
# Internal Tracking Data
# =============================================================================

@dataclass
class _CitationEntry:
    """Internal tracking entry for a citation.

    Stores metadata about a citation including its retrieval score,
    pipeline stage, and usage contexts.
    """

    citation: Citation
    retrieval_score: float | None = None
    stage: str | None = None
    usage_contexts: list[str] = field(default_factory=list)


# =============================================================================
# AC-17.1: CitationManager
# =============================================================================

class CitationManager:
    """Manager for tracking citations through pipeline stages.

    Assigns unique [^N] markers to sources and tracks their usage
    throughout the pipeline execution.

    Attributes:
        next_marker: The next marker number to assign

    Example:
        >>> manager = CitationManager()
        >>> source = SourceMetadata(
        ...     source_type=SourceType.BOOK,
        ...     author="Fowler, Martin",
        ...     title="PEAA",
        ...     year=2002,
        ... )
        >>> marker = manager.add_source(source, retrieval_score=0.89)
        >>> marker
        1
        >>> manager.get_inline_marker(marker)
        '[^1]'
    """

    def __init__(self) -> None:
        """Initialize empty citation manager."""
        self._entries: dict[int, _CitationEntry] = {}
        self._next_marker: int = 1
        self._source_to_marker: dict[str, int] = {}

    @property
    def next_marker(self) -> int:
        """Return the next marker number that will be assigned.

        Returns:
            Next marker number
        """
        return self._next_marker

    def add_source(
        self,
        source: SourceMetadata,
        *,
        context: str | None = None,
        retrieval_score: float | None = None,
        stage: str | None = None,
    ) -> int:
        """Add a source and return its assigned marker number.

        Args:
            source: Source metadata to add
            context: Optional context/excerpt from the source
            retrieval_score: Semantic search relevance score
            stage: Pipeline stage that added this citation

        Returns:
            Assigned marker number
        """
        marker = self._next_marker
        self._next_marker += 1

        citation = Citation(
            marker=marker,
            source=source,
            context=context,
        )

        entry = _CitationEntry(
            citation=citation,
            retrieval_score=retrieval_score,
            stage=stage,
        )

        self._entries[marker] = entry

        # Store source hash for duplicate detection
        source_key = self._get_source_key(source)
        self._source_to_marker[source_key] = marker

        return marker

    def _get_source_key(self, source: SourceMetadata) -> str:
        """Generate a unique key for a source.

        Args:
            source: Source metadata

        Returns:
            Unique string key for the source
        """
        if source.source_type == SourceType.BOOK:
            return f"book:{source.author}:{source.title}:{source.year}"
        elif source.source_type == SourceType.CODE:
            return f"code:{source.repo}:{source.file_path}:{source.line_range}"
        elif source.source_type == SourceType.SCHEMA:
            return f"schema:{source.repo}:{source.file_path}:{source.version}"
        elif source.source_type == SourceType.INTERNAL_DOC:
            return f"internal_doc:{source.service}:{source.title}:{source.date}"
        else:
            return f"unknown:{source.title}:{source.year}"

    def get_citation(self, marker: int) -> Citation | None:
        """Get a citation by its marker number.

        Args:
            marker: The footnote marker number

        Returns:
            Citation object if found, None otherwise
        """
        entry = self._entries.get(marker)
        return entry.citation if entry else None

    def get_all_citations(self) -> list[Citation]:
        """Get all citations in marker order.

        Returns:
            List of Citation objects ordered by marker number
        """
        return [
            entry.citation
            for marker, entry in sorted(self._entries.items())
        ]

    def find_marker(self, source: SourceMetadata) -> int | None:
        """Find the marker for an existing source.

        Args:
            source: Source metadata to find

        Returns:
            Marker number if found, None otherwise
        """
        source_key = self._get_source_key(source)
        return self._source_to_marker.get(source_key)

    def get_inline_marker(self, marker: int) -> str:
        """Get the inline marker format [^N].

        Args:
            marker: The marker number

        Returns:
            Formatted inline marker string
        """
        return f"[^{marker}]"

    def get_retrieval_score(self, marker: int) -> float | None:
        """Get the retrieval score for a citation.

        Args:
            marker: The marker number

        Returns:
            Retrieval score if set, None otherwise
        """
        entry = self._entries.get(marker)
        return entry.retrieval_score if entry else None

    def get_stage(self, marker: int) -> str | None:
        """Get the pipeline stage that added a citation.

        Args:
            marker: The marker number

        Returns:
            Stage name if set, None otherwise
        """
        entry = self._entries.get(marker)
        return entry.stage if entry else None

    def record_usage(self, marker: int, usage_context: str) -> None:
        """Record a usage context for a citation.

        Args:
            marker: The marker number
            usage_context: Description of how the citation was used
        """
        entry = self._entries.get(marker)
        if entry:
            entry.usage_contexts.append(usage_context)

    def get_usage_context(self, marker: int) -> list[str]:
        """Get all usage contexts for a citation.

        Args:
            marker: The marker number

        Returns:
            List of usage context strings
        """
        entry = self._entries.get(marker)
        return entry.usage_contexts if entry else []

    def get_citations_by_stage(self, stage: str) -> list[Citation]:
        """Get all citations added by a specific pipeline stage.

        Args:
            stage: Pipeline stage name to filter by

        Returns:
            List of Citation objects from that stage
        """
        return [
            entry.citation
            for entry in self._entries.values()
            if entry.stage == stage
        ]

    def clear(self) -> None:
        """Clear all citations and reset marker counter."""
        self._entries.clear()
        self._source_to_marker.clear()
        self._next_marker = 1

    def to_audit_records(
        self,
        conversation_id: str,
        message_id: str,
    ) -> list[CitationAuditRecord]:
        """Export citations as audit records.

        Args:
            conversation_id: Conversation identifier for audit
            message_id: Message identifier for audit

        Returns:
            List of CitationAuditRecord objects
        """
        records = []

        for marker, entry in sorted(self._entries.items()):
            citation = entry.citation
            source = citation.source

            # Generate source_id
            source_id = self._get_source_key(source)

            # Combine usage contexts
            usage_context = "; ".join(entry.usage_contexts) if entry.usage_contexts else ""

            record = CitationAuditRecord(
                conversation_id=conversation_id,
                message_id=message_id,
                source_id=source_id,
                source_type=source.source_type.value,
                retrieval_score=entry.retrieval_score or 0.0,
                usage_context=usage_context,
                marker=marker,
            )
            records.append(record)

        return records


__all__ = [
    "CitationManager",
]
