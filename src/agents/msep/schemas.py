"""MSEP Schemas.

WBS: MSE-2.1 - Input Schema Dataclasses
WBS: MSE-2.2 - Output Schema Dataclasses

Defines all MSEP data structures using dataclasses.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- #2.2: Full type annotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.agents.msep.constants import CHAPTER_ID_FORMAT


if TYPE_CHECKING:
    from src.agents.msep.config import MSEPConfig


# =============================================================================
# MSE-2.1: Input Schema Dataclasses
# =============================================================================


@dataclass
class ChapterMeta:
    """Metadata for a single chapter in the corpus.

    Attributes:
        book: Book title.
        chapter: Chapter number.
        title: Chapter title.
        id: Unique chapter identifier (auto-generated if not provided).
    """

    book: str
    chapter: int
    title: str
    id: str = field(default="")

    def __post_init__(self) -> None:
        """Auto-generate id if not provided."""
        if not self.id:
            self.id = CHAPTER_ID_FORMAT.format(book=self.book, chapter=self.chapter)


@dataclass
class MSEPRequest:
    """Request payload for MSEP enrichment.

    Attributes:
        corpus: List of document texts (one per chapter).
        chapter_index: Metadata for each chapter.
        config: MSEP configuration.
    """

    corpus: list[str]
    chapter_index: list[ChapterMeta]
    config: MSEPConfig


# =============================================================================
# MSE-2.2: Output Schema Dataclasses
# =============================================================================


@dataclass
class CrossReference:
    """A cross-reference to another chapter.

    EEP-3.4 Update: Added multi-signal fusion fields (AC-3.4.2, AC-3.4.3).
    EEP-4.4 Update: Added relationship_type field (AC-4.4.2).

    Attributes:
        target: Target chapter ID (e.g., "Book:ch5").
        score: Final fused similarity score.
        base_score: Raw similarity score from SBERT.
        topic_boost: Boost applied if same topic.
        method: Enrichment method used (sbert, multi-signal, hybrid).
        concept_overlap: Concept Jaccard score (EEP-3.2). None for backward compat.
        keyword_jaccard: Keyword Jaccard score (EEP-3.3). None for backward compat.
        matched_concepts: List of matched concepts (EEP-3.2). Empty for backward compat.
        relationship_type: Graph relationship type (EEP-4.4). None for backward compat.
    """

    target: str
    score: float
    base_score: float
    topic_boost: float
    method: str
    concept_overlap: float | None = field(default=None)
    keyword_jaccard: float | None = field(default=None)
    matched_concepts: list[str] = field(default_factory=list)
    relationship_type: str | None = field(default=None)


@dataclass
class MergedKeywords:
    """Keywords from multiple extraction methods.

    Attributes:
        tfidf: Keywords extracted via TF-IDF.
        semantic: Keywords extracted via semantic methods.
        merged: Combined and deduplicated keywords.
    """

    tfidf: list[str]
    semantic: list[str]
    merged: list[str]


@dataclass
class Provenance:
    """Provenance tracking for enrichment results.

    Attributes:
        methods_used: List of enrichment methods applied.
        sbert_score: SBERT similarity score (if used).
        topic_boost: Topic boost applied (if same topic).
        timestamp: ISO 8601 timestamp of enrichment.
    """

    methods_used: list[str]
    sbert_score: float
    topic_boost: float
    timestamp: str


@dataclass
class EnrichedChapter:
    """Enriched metadata for a single chapter.

    Per MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md Schema Definitions.

    Attributes:
        book: Book title.
        chapter: Chapter number.
        title: Chapter title.
        chapter_id: Unique chapter identifier.
        cross_references: List of cross-references to other chapters.
        keywords: Merged keywords from all methods.
        topic_id: BERTopic cluster assignment (None if BERTopic unavailable).
        topic_name: Human-readable topic name (None if unavailable).
        graph_relationships: Graph relationships from hybrid search.
        provenance: Tracking info for how results were generated.
    """

    book: str
    chapter: int
    title: str
    chapter_id: str
    cross_references: list[CrossReference]
    keywords: MergedKeywords
    topic_id: int | None
    topic_name: str | None
    graph_relationships: list[str]
    provenance: Provenance


@dataclass
class EnrichedMetadata:
    """Complete enriched metadata for an MSEP request.

    Attributes:
        chapters: List of enriched chapter metadata.
        processing_time_ms: Total processing time in milliseconds.
        total_cross_references: Total count of all cross-references.
    """

    chapters: list[EnrichedChapter]
    processing_time_ms: float
    total_cross_references: int = field(default=0)

    def __post_init__(self) -> None:
        """Compute total_cross_references from chapters."""
        self.total_cross_references = sum(
            len(ch.cross_references) for ch in self.chapters
        )
