"""MSEP Result Merger.

WBS: MSE-5 - ai-agents Result Merger
Implements topic boost calculator, dynamic threshold, result aggregator, provenance builder.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-5
- MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md: Result merging

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S3776: Cognitive complexity < 15 per function (extracted helpers)
- S1192: Uses constants from constants.py (SAME_TOPIC_BOOST, etc.)
- #2.2: Full type annotations, returns typed dataclasses
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.agents.msep.constants import (
    LARGE_CORPUS_THRESHOLD,
    MAX_THRESHOLD,
    METHOD_BERTOPIC,
    METHOD_SBERT,
    METHOD_TFIDF,
    MIN_THRESHOLD,
    SAME_TOPIC_BOOST,
    SMALL_CORPUS_THRESHOLD,
    THRESHOLD_ADJUSTMENT,
)
from src.agents.msep.schemas import (
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    Provenance,
)


# =============================================================================
# MSE-5.1: Topic Boost Calculator
# =============================================================================


def apply_topic_boost(
    source_topic: int | None,
    target_topic: int | None,
) -> float:
    """Calculate topic boost for same-topic chapters.

    AC-5.1.1: Adds SAME_TOPIC_BOOST when topic_i == topic_j.
    AC-5.1.2: Returns 0.0 when topics differ.
    AC-5.1.3: Handles None topic assignments gracefully.
    AC-5.1.4: Uses SAME_TOPIC_BOOST constant (not hardcoded).

    Args:
        source_topic: BERTopic cluster ID for source chapter.
        target_topic: BERTopic cluster ID for target chapter.

    Returns:
        SAME_TOPIC_BOOST if same topic, 0.0 otherwise.
    """
    if source_topic is None or target_topic is None:
        return 0.0

    if source_topic == target_topic:
        return SAME_TOPIC_BOOST

    return 0.0


# =============================================================================
# MSE-5.2: Dynamic Threshold Calculator
# =============================================================================


def calculate_dynamic_threshold(
    corpus_size: int,
    base_threshold: float,
    use_dynamic: bool,
) -> float:
    """Calculate dynamic threshold based on corpus size.

    AC-5.2.1: Returns base - 0.1 when corpus > 500 chapters.
    AC-5.2.2: Returns base + 0.1 when corpus < 100 chapters.
    AC-5.2.3: Returns base for 100-500 chapters.
    AC-5.2.4: Clamps to [MIN_THRESHOLD, MAX_THRESHOLD] range.
    AC-5.2.5: Respects use_dynamic flag.

    Args:
        corpus_size: Number of chapters in corpus.
        base_threshold: Base similarity threshold.
        use_dynamic: Whether to apply dynamic adjustment.

    Returns:
        Adjusted threshold, clamped to [0.3, 0.6].
    """
    if not use_dynamic:
        return base_threshold

    # Calculate adjustment based on corpus size
    adjustment = _calculate_size_adjustment(corpus_size)
    adjusted = base_threshold + adjustment

    # Clamp to allowed range
    return _clamp_threshold(adjusted)


def _calculate_size_adjustment(corpus_size: int) -> float:
    """Calculate threshold adjustment based on corpus size.

    Helper to keep cognitive complexity low in main function.

    Args:
        corpus_size: Number of chapters.

    Returns:
        Adjustment value: positive, negative, or zero.
    """
    if corpus_size < SMALL_CORPUS_THRESHOLD:
        return THRESHOLD_ADJUSTMENT  # Increase threshold for small corpus
    if corpus_size > LARGE_CORPUS_THRESHOLD:
        return -THRESHOLD_ADJUSTMENT  # Decrease threshold for large corpus
    return 0.0  # No adjustment for medium corpus


def _clamp_threshold(value: float) -> float:
    """Clamp threshold to [MIN_THRESHOLD, MAX_THRESHOLD].

    Args:
        value: Threshold value to clamp.

    Returns:
        Clamped threshold.
    """
    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, value))


# =============================================================================
# MSE-TAX: Taxonomy Filter (AC-TAX-3)
# =============================================================================


def filter_by_taxonomy(
    cross_refs: list[CrossReference],
    taxonomy_books: set[str] | None,
) -> list[CrossReference]:
    """Filter cross-references to include only books in taxonomy.

    AC-TAX-3.1: Returns all cross-refs when taxonomy_books is None.
    AC-TAX-3.2: Filters to only targets whose book is in taxonomy_books.
    AC-TAX-3.3: Handles empty cross_refs gracefully.
    AC-TAX-3.4: Preserves original order of results.

    Args:
        cross_refs: List of cross-references to filter.
        taxonomy_books: Set of book names in the taxonomy, or None.

    Returns:
        Filtered list of CrossReference (or original if taxonomy_books is None).
    """
    # AC-TAX-3.1: No filtering when taxonomy_books is None
    if taxonomy_books is None:
        return cross_refs

    # AC-TAX-3.3: Handle empty cross_refs
    if not cross_refs:
        return cross_refs

    # AC-TAX-3.2 & 3.4: Filter while preserving order
    filtered: list[CrossReference] = []
    for xref in cross_refs:
        book_name = _extract_book_from_target(xref.target)
        if book_name in taxonomy_books:
            filtered.append(xref)

    return filtered


def _extract_book_from_target(target: str) -> str:
    """Extract book name from target chapter ID.

    Handles formats:
    - "Book:chN" → "Book"
    - "Book::N" → "Book"
    - Fallback: return target as-is

    Args:
        target: Target chapter identifier.

    Returns:
        Book name extracted from target.
    """
    # Handle "Book:chN" format
    if ":ch" in target:
        return target.split(":ch")[0]

    # Handle "Book::N" format
    if "::" in target:
        return target.split("::")[0]

    # Fallback: assume the whole target is the book name
    return target


# =============================================================================
# MSE-5.3: Result Aggregator
# =============================================================================


def merge_results(
    similarity_matrix: list[list[float]],
    topics: list[int] | None,
    keywords: list[list[str]] | None,
    chapter_ids: list[str],
    threshold: float,
    top_k: int,
    _taxonomy: str | None = None,
    taxonomy_books: set[str] | None = None,
) -> EnrichedMetadata:
    """Merge all enrichment results into EnrichedMetadata.

    AC-5.3.1: Returns EnrichedMetadata dataclass.
    AC-5.3.2: Combines SBERT scores + topic boosts correctly.
    AC-5.3.3: Filters results below threshold.
    AC-5.3.4: Sorts cross-references by final score (descending).
    AC-5.3.5: Limits to top_k cross-references per chapter.
    AC-5.3.6: Cognitive complexity < 15 (uses helper methods).
    AC-TAX-4: Applies taxonomy filter to cross-references when provided.

    Args:
        similarity_matrix: Pairwise similarity scores.
        topics: BERTopic cluster assignments (or None).
        keywords: TF-IDF keywords per chapter (or None).
        chapter_ids: List of chapter identifiers.
        threshold: Minimum score for cross-references.
        top_k: Maximum cross-references per chapter.
        _taxonomy: Optional taxonomy name (reserved for future provenance/logging).
        taxonomy_books: Optional set of book names to filter by.

    Returns:
        EnrichedMetadata with all chapters enriched.
    """
    start_time = time.perf_counter()

    # Handle None values
    safe_topics = topics if topics is not None else [-1] * len(chapter_ids)
    safe_keywords = keywords if keywords is not None else [[] for _ in chapter_ids]

    # Build enriched chapters (with optional taxonomy filtering)
    chapters = _build_enriched_chapters(
        similarity_matrix=similarity_matrix,
        topics=safe_topics,
        keywords=safe_keywords,
        chapter_ids=chapter_ids,
        threshold=threshold,
        top_k=top_k,
        taxonomy_books=taxonomy_books,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return EnrichedMetadata(
        chapters=chapters,
        processing_time_ms=elapsed_ms,
    )


def _build_enriched_chapters(
    similarity_matrix: list[list[float]],
    topics: list[int],
    keywords: list[list[str]],
    chapter_ids: list[str],
    threshold: float,
    top_k: int,
    taxonomy_books: set[str] | None = None,
) -> list[EnrichedChapter]:
    """Build list of EnrichedChapter from raw data.

    Helper to keep cognitive complexity manageable.

    Args:
        similarity_matrix: Pairwise similarity scores.
        topics: BERTopic cluster assignments.
        keywords: TF-IDF keywords per chapter.
        chapter_ids: List of chapter identifiers.
        threshold: Minimum score for cross-references.
        top_k: Maximum cross-references per chapter.
        taxonomy_books: Optional set of book names for filtering.

    Returns:
        List of EnrichedChapter dataclasses.
    """
    chapters: list[EnrichedChapter] = []

    for i, chapter_id in enumerate(chapter_ids):
        cross_refs = _build_cross_references(
            source_idx=i,
            similarity_matrix=similarity_matrix,
            topics=topics,
            chapter_ids=chapter_ids,
            threshold=threshold,
            top_k=top_k,
        )

        # Apply taxonomy filter if provided (AC-TAX-4)
        cross_refs = filter_by_taxonomy(cross_refs, taxonomy_books)

        merged_keywords = _build_merged_keywords(keywords[i] if i < len(keywords) else [])
        provenance = _build_chapter_provenance(cross_refs)

        # Parse chapter_id to extract book and chapter number
        # Format: "Book:chN" or "Book::N" or just "chapter_N"
        book, chapter_num, title = _parse_chapter_id(chapter_id, i)
        topic_id = topics[i] if i < len(topics) and topics[i] >= 0 else None
        topic_name = f"Topic {topic_id}" if topic_id is not None else None

        chapter = EnrichedChapter(
            book=book,
            chapter=chapter_num,
            title=title,
            chapter_id=chapter_id,
            cross_references=cross_refs,
            keywords=merged_keywords,
            topic_id=topic_id,
            topic_name=topic_name,
            graph_relationships=[],  # Merger doesn't have hybrid results
            provenance=provenance,
        )
        chapters.append(chapter)

    return chapters


def _parse_chapter_id(chapter_id: str, index: int) -> tuple[str, int, str]:
    """Parse chapter_id string to extract book, chapter number, and title.

    Handles formats:
    - "Book:chN" (e.g., "Deep Learning:ch1")
    - "Book::N" (e.g., "Deep Learning::1")
    - Fallback: "Unknown Book", index+1, chapter_id as title

    Args:
        chapter_id: Chapter identifier string.
        index: Chapter index (for fallback).

    Returns:
        Tuple of (book, chapter_number, title).
    """
    # Try "Book:chN" format
    if ":ch" in chapter_id:
        parts = chapter_id.split(":ch")
        if len(parts) == 2:
            book = parts[0]
            try:
                chapter_num = int(parts[1])
                return (book, chapter_num, f"Chapter {chapter_num}")
            except ValueError:
                pass

    # Try "Book::N" format
    if "::" in chapter_id:
        parts = chapter_id.split("::")
        if len(parts) == 2:
            book = parts[0]
            try:
                chapter_num = int(parts[1])
                return (book, chapter_num, f"Chapter {chapter_num}")
            except ValueError:
                pass

    # Fallback
    return ("Unknown Book", index + 1, chapter_id)


def _build_cross_references(
    source_idx: int,
    similarity_matrix: list[list[float]],
    topics: list[int],
    chapter_ids: list[str],
    threshold: float,
    top_k: int,
) -> list[CrossReference]:
    """Build cross-references for a single chapter.

    Filters by threshold, sorts by score, limits to top_k.

    Args:
        source_idx: Index of source chapter.
        similarity_matrix: Pairwise similarity scores.
        topics: Topic assignments for all chapters.
        chapter_ids: Chapter identifiers.
        threshold: Minimum score threshold.
        top_k: Maximum number of cross-references.

    Returns:
        Sorted, filtered list of CrossReference.
    """
    source_topic = topics[source_idx] if source_idx < len(topics) else None
    candidates: list[CrossReference] = []

    for target_idx, target_id in enumerate(chapter_ids):
        # Skip self-references
        if target_idx == source_idx:
            continue

        base_score = similarity_matrix[source_idx][target_idx]
        target_topic = topics[target_idx] if target_idx < len(topics) else None
        topic_boost = apply_topic_boost(source_topic, target_topic)
        final_score = base_score + topic_boost

        # Filter by threshold
        if final_score < threshold:
            continue

        xref = CrossReference(
            target=target_id,
            score=final_score,
            base_score=base_score,
            topic_boost=topic_boost,
            method=METHOD_SBERT,
        )
        candidates.append(xref)

    # Sort descending by score, limit to top_k
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:top_k]


def _build_merged_keywords(tfidf_keywords: list[str]) -> MergedKeywords:
    """Build MergedKeywords from TF-IDF extraction.

    Args:
        tfidf_keywords: Keywords from TF-IDF extractor.

    Returns:
        MergedKeywords dataclass.
    """
    return MergedKeywords(
        tfidf=tfidf_keywords,
        semantic=[],  # Placeholder for future semantic keywords
        merged=tfidf_keywords,  # For now, merged = tfidf
    )


def _build_chapter_provenance(cross_refs: list[CrossReference]) -> Provenance:
    """Build provenance for a chapter.

    Args:
        cross_refs: List of cross-references.

    Returns:
        Provenance dataclass.
    """
    methods: list[str] = [METHOD_SBERT]
    has_topic_boost = any(xr.topic_boost > 0 for xr in cross_refs)

    if has_topic_boost:
        methods.append(METHOD_BERTOPIC)

    # Calculate average scores if we have cross-refs
    avg_sbert = 0.0
    avg_boost = 0.0
    if cross_refs:
        avg_sbert = sum(xr.base_score for xr in cross_refs) / len(cross_refs)
        avg_boost = sum(xr.topic_boost for xr in cross_refs) / len(cross_refs)

    return build_provenance(
        sbert_score=avg_sbert,
        topic_boost=avg_boost,
        methods_used=methods,
    )


# =============================================================================
# MSE-5.4: Provenance Builder
# =============================================================================


def build_provenance(
    sbert_score: float,
    topic_boost: float,
    methods_used: list[str],
) -> Provenance:
    """Build provenance record for enrichment results.

    AC-5.4.1: Returns Provenance dataclass.
    AC-5.4.2: Tracks all methods used.
    AC-5.4.3: Records individual scores.
    AC-5.4.4: Timestamp in ISO 8601 format (UTC).

    Args:
        sbert_score: SBERT similarity score.
        topic_boost: Topic boost applied.
        methods_used: List of enrichment methods.

    Returns:
        Provenance dataclass with timestamp.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    return Provenance(
        methods_used=methods_used,
        sbert_score=sbert_score,
        topic_boost=topic_boost,
        timestamp=timestamp,
    )
