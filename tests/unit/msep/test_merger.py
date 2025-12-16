"""Unit tests for MSEP Result Merger.

WBS: MSE-5 - ai-agents Result Merger
Tests for topic boost calculator, dynamic threshold, result aggregator, provenance builder.

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Coverage:
- AC-5.1.1: apply_topic_boost() adds 0.15 when topic_i == topic_j
- AC-5.1.2: Returns 0.0 boost when topics differ
- AC-5.1.3: Handles None topic assignments gracefully
- AC-5.1.4: Uses SAME_TOPIC_BOOST constant (not hardcoded)
- AC-5.2.1: Returns base - 0.1 when corpus > 500 chapters
- AC-5.2.2: Returns base + 0.1 when corpus < 100 chapters
- AC-5.2.3: Returns base for 100-500 chapters
- AC-5.2.4: Clamps to [0.3, 0.6] range
- AC-5.2.5: Respects config.use_dynamic_threshold flag
- AC-5.3.1: merge_results() returns EnrichedMetadata
- AC-5.3.2: Combines SBERT scores + topic boosts correctly
- AC-5.3.3: Filters results below threshold
- AC-5.3.4: Sorts cross-references by final score (descending)
- AC-5.3.5: Limits to config.top_k cross-references per chapter
- AC-5.3.6: Cognitive complexity < 15
- AC-5.4.1: build_provenance() returns Provenance dataclass
- AC-5.4.2: Tracks all methods used
- AC-5.4.3: Records individual scores
- AC-5.4.4: Timestamp in ISO 8601 format (UTC)

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- S3776: Helper methods keep complexity < 15
- #2.2: Full type annotations
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from src.agents.msep.constants import (
    DEFAULT_THRESHOLD,
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

if TYPE_CHECKING:
    pass


# =============================================================================
# MSE-5.1: Topic Boost Calculator Tests
# =============================================================================


class TestTopicBoostCalculator:
    """Tests for apply_topic_boost() function (AC-5.1.1 - AC-5.1.4)."""

    def test_apply_topic_boost_same_topic_returns_boost(self) -> None:
        """AC-5.1.1: apply_topic_boost() adds 0.15 when topic_i == topic_j."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=1, target_topic=1)
        assert boost == SAME_TOPIC_BOOST

    def test_apply_topic_boost_different_topics_returns_zero(self) -> None:
        """AC-5.1.2: Returns 0.0 boost when topics differ."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=1, target_topic=2)
        assert boost == 0.0

    def test_apply_topic_boost_source_none_returns_zero(self) -> None:
        """AC-5.1.3: Handles None topic assignments gracefully."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=None, target_topic=1)
        assert boost == 0.0

    def test_apply_topic_boost_target_none_returns_zero(self) -> None:
        """AC-5.1.3: Handles None topic assignments gracefully."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=1, target_topic=None)
        assert boost == 0.0

    def test_apply_topic_boost_both_none_returns_zero(self) -> None:
        """AC-5.1.3: Both None topics return zero."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=None, target_topic=None)
        assert boost == 0.0

    def test_apply_topic_boost_uses_constant(self) -> None:
        """AC-5.1.4: Uses SAME_TOPIC_BOOST constant (not hardcoded)."""
        from src.agents.msep.merger import apply_topic_boost

        # Verify the constant value is what we expect
        assert SAME_TOPIC_BOOST == 0.15
        # And that the function returns it
        boost = apply_topic_boost(source_topic=5, target_topic=5)
        assert boost == SAME_TOPIC_BOOST

    def test_apply_topic_boost_negative_topics_work(self) -> None:
        """Negative topic IDs (outliers in BERTopic) still work."""
        from src.agents.msep.merger import apply_topic_boost

        # BERTopic assigns -1 to outliers
        boost = apply_topic_boost(source_topic=-1, target_topic=-1)
        assert boost == SAME_TOPIC_BOOST

    def test_apply_topic_boost_zero_topic_works(self) -> None:
        """Topic ID 0 is valid and should match."""
        from src.agents.msep.merger import apply_topic_boost

        boost = apply_topic_boost(source_topic=0, target_topic=0)
        assert boost == SAME_TOPIC_BOOST


# =============================================================================
# MSE-5.2: Dynamic Threshold Calculator Tests
# =============================================================================


class TestDynamicThresholdCalculator:
    """Tests for calculate_dynamic_threshold() function (AC-5.2.1 - AC-5.2.5)."""

    def test_large_corpus_decreases_threshold(self) -> None:
        """AC-5.2.1: Returns base - 0.1 when corpus > 500 chapters."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # 501 chapters (> LARGE_CORPUS_THRESHOLD)
        threshold = calculate_dynamic_threshold(
            corpus_size=501,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )
        expected = DEFAULT_THRESHOLD - THRESHOLD_ADJUSTMENT
        assert threshold == pytest.approx(expected)

    def test_small_corpus_increases_threshold(self) -> None:
        """AC-5.2.2: Returns base + 0.1 when corpus < 100 chapters."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # 99 chapters (< SMALL_CORPUS_THRESHOLD)
        threshold = calculate_dynamic_threshold(
            corpus_size=99,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )
        expected = DEFAULT_THRESHOLD + THRESHOLD_ADJUSTMENT
        assert threshold == pytest.approx(expected)

    def test_medium_corpus_keeps_base_threshold(self) -> None:
        """AC-5.2.3: Returns base for 100-500 chapters."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # 250 chapters (between thresholds)
        threshold = calculate_dynamic_threshold(
            corpus_size=250,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )
        assert threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_boundary_100_chapters_is_medium(self) -> None:
        """100 chapters is considered medium (inclusive)."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        threshold = calculate_dynamic_threshold(
            corpus_size=100,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )
        assert threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_boundary_500_chapters_is_medium(self) -> None:
        """500 chapters is considered medium (inclusive)."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        threshold = calculate_dynamic_threshold(
            corpus_size=500,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )
        assert threshold == pytest.approx(DEFAULT_THRESHOLD)

    def test_clamp_to_minimum_threshold(self) -> None:
        """AC-5.2.4: Clamps to [0.3, 0.6] range - minimum."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # Very large corpus with low base would go below MIN_THRESHOLD
        threshold = calculate_dynamic_threshold(
            corpus_size=1000,  # Large corpus, -0.1
            base_threshold=MIN_THRESHOLD,  # 0.3, result would be 0.2
            use_dynamic=True,
        )
        assert threshold == MIN_THRESHOLD

    def test_clamp_to_maximum_threshold(self) -> None:
        """AC-5.2.4: Clamps to [0.3, 0.6] range - maximum."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # Very small corpus with high base would exceed MAX_THRESHOLD
        threshold = calculate_dynamic_threshold(
            corpus_size=10,  # Small corpus, +0.1
            base_threshold=MAX_THRESHOLD,  # 0.6, result would be 0.7
            use_dynamic=True,
        )
        assert threshold == MAX_THRESHOLD

    def test_dynamic_threshold_disabled_returns_base(self) -> None:
        """AC-5.2.5: Respects config.use_dynamic_threshold flag."""
        from src.agents.msep.merger import calculate_dynamic_threshold

        # Even with large corpus, returns base when disabled
        threshold = calculate_dynamic_threshold(
            corpus_size=1000,
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=False,
        )
        assert threshold == DEFAULT_THRESHOLD

    def test_constants_have_expected_values(self) -> None:
        """Verify constants have documented values."""
        assert MIN_THRESHOLD == 0.3
        assert MAX_THRESHOLD == 0.6
        assert SMALL_CORPUS_THRESHOLD == 100
        assert LARGE_CORPUS_THRESHOLD == 500
        assert THRESHOLD_ADJUSTMENT == 0.1


# =============================================================================
# MSE-5.3: Result Aggregator Tests
# =============================================================================


class TestResultAggregator:
    """Tests for merge_results() function (AC-5.3.1 - AC-5.3.6)."""

    @pytest.fixture
    def sample_similarity_matrix(self) -> list[list[float]]:
        """Sample 3x3 similarity matrix."""
        return [
            [1.0, 0.8, 0.6],  # Chapter 0: high sim with 1, medium with 2
            [0.8, 1.0, 0.4],  # Chapter 1: high sim with 0, low with 2
            [0.6, 0.4, 1.0],  # Chapter 2: medium with 0, low with 1
        ]

    @pytest.fixture
    def sample_topics(self) -> list[int]:
        """Sample topic assignments (chapters 0,1 same topic)."""
        return [0, 0, 1]  # Chapters 0 and 1 share topic 0

    @pytest.fixture
    def sample_keywords(self) -> list[list[str]]:
        """Sample keywords per chapter."""
        return [
            ["python", "async", "await"],
            ["python", "threading", "concurrency"],
            ["java", "spring", "boot"],
        ]

    @pytest.fixture
    def sample_chapter_ids(self) -> list[str]:
        """Sample chapter IDs."""
        return ["Book:ch1", "Book:ch2", "Book:ch3"]

    def test_merge_results_returns_enriched_metadata(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """AC-5.3.1: merge_results() returns EnrichedMetadata."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.5,
            top_k=5,
        )

        assert isinstance(result, EnrichedMetadata)
        assert len(result.chapters) == 3

    def test_merge_results_combines_scores_and_boosts(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """AC-5.3.2: Combines SBERT scores + topic boosts correctly."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.0,  # No filtering to see all results
            top_k=10,
        )

        # Chapter 0's cross-ref to Chapter 1 should have topic boost
        # Base score: 0.8, boost: 0.15, final: 0.95
        ch0 = result.chapters[0]
        ch1_ref = next(
            (xr for xr in ch0.cross_references if xr.target == "Book:ch2"), None
        )
        assert ch1_ref is not None
        assert ch1_ref.base_score == pytest.approx(0.8)
        assert ch1_ref.topic_boost == pytest.approx(SAME_TOPIC_BOOST)
        assert ch1_ref.score == pytest.approx(0.8 + SAME_TOPIC_BOOST)

    def test_merge_results_filters_below_threshold(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """AC-5.3.3: Filters results below threshold."""
        from src.agents.msep.merger import merge_results

        # Set threshold at 0.7 - should filter out some cross-refs
        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.7,
            top_k=10,
        )

        # Only high-scoring cross-refs should remain
        for chapter in result.chapters:
            for xref in chapter.cross_references:
                assert xref.score >= 0.7

    def test_merge_results_sorts_by_score_descending(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """AC-5.3.4: Sorts cross-references by final score (descending)."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.0,
            top_k=10,
        )

        # Check each chapter's cross-refs are sorted descending
        for chapter in result.chapters:
            scores = [xr.score for xr in chapter.cross_references]
            assert scores == sorted(scores, reverse=True)

    def test_merge_results_limits_to_top_k(
        self,
        sample_chapter_ids: list[str],
    ) -> None:
        """AC-5.3.5: Limits to config.top_k cross-references per chapter."""
        from src.agents.msep.merger import merge_results

        # Create a larger similarity matrix (5 chapters)
        large_similarity = [
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.9, 1.0, 0.85, 0.75, 0.65],
            [0.8, 0.85, 1.0, 0.8, 0.7],
            [0.7, 0.75, 0.8, 1.0, 0.9],
            [0.6, 0.65, 0.7, 0.9, 1.0],
        ]
        topics = [0, 0, 1, 1, 2]
        keywords = [["kw"] for _ in range(5)]
        chapter_ids = [f"Book:ch{i}" for i in range(1, 6)]

        result = merge_results(
            similarity_matrix=large_similarity,
            topics=topics,
            keywords=keywords,
            chapter_ids=chapter_ids,
            threshold=0.0,
            top_k=2,  # Limit to 2 cross-refs per chapter
        )

        # Each chapter should have at most 2 cross-references
        for chapter in result.chapters:
            assert len(chapter.cross_references) <= 2

    def test_merge_results_excludes_self_references(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """Chapters should not cross-reference themselves."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.0,
            top_k=10,
        )

        # No self-references
        for chapter in result.chapters:
            for xref in chapter.cross_references:
                assert xref.target != chapter.chapter_id

    def test_merge_results_assigns_topic_ids(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """Each chapter gets its topic_id from BERTopic."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.5,
            top_k=5,
        )

        assert result.chapters[0].topic_id == 0
        assert result.chapters[1].topic_id == 0
        assert result.chapters[2].topic_id == 1

    def test_merge_results_builds_keywords(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """Keywords are assigned to each chapter."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.5,
            top_k=5,
        )

        ch0 = result.chapters[0]
        assert ch0.keywords.tfidf == ["python", "async", "await"]

    def test_merge_results_handles_none_topics(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_keywords: list[list[str]],
        sample_chapter_ids: list[str],
    ) -> None:
        """Handles None topics gracefully (no boosts applied)."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=None,  # No topic assignments
            keywords=sample_keywords,
            chapter_ids=sample_chapter_ids,
            threshold=0.0,
            top_k=10,
        )

        # Should work, with -1 as default topic and no boosts
        assert isinstance(result, EnrichedMetadata)
        for chapter in result.chapters:
            assert chapter.topic_id == -1  # Default when no topics

    def test_merge_results_handles_none_keywords(
        self,
        sample_similarity_matrix: list[list[float]],
        sample_topics: list[int],
        sample_chapter_ids: list[str],
    ) -> None:
        """Handles None keywords gracefully."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=sample_similarity_matrix,
            topics=sample_topics,
            keywords=None,  # No keywords
            chapter_ids=sample_chapter_ids,
            threshold=0.5,
            top_k=5,
        )

        # Should work with empty keywords
        assert isinstance(result, EnrichedMetadata)
        for chapter in result.chapters:
            assert chapter.keywords.tfidf == []


# =============================================================================
# MSE-5.4: Provenance Builder Tests
# =============================================================================


class TestProvenanceBuilder:
    """Tests for build_provenance() function (AC-5.4.1 - AC-5.4.4)."""

    def test_build_provenance_returns_provenance_dataclass(self) -> None:
        """AC-5.4.1: build_provenance() returns Provenance dataclass."""
        from src.agents.msep.merger import build_provenance

        result = build_provenance(
            sbert_score=0.85,
            topic_boost=0.15,
            methods_used=[METHOD_SBERT, METHOD_BERTOPIC],
        )

        assert isinstance(result, Provenance)

    def test_build_provenance_tracks_all_methods(self) -> None:
        """AC-5.4.2: Tracks all methods used."""
        from src.agents.msep.merger import build_provenance

        methods = [METHOD_SBERT, METHOD_TFIDF, METHOD_BERTOPIC]
        result = build_provenance(
            sbert_score=0.75,
            topic_boost=0.0,
            methods_used=methods,
        )

        assert result.methods_used == methods

    def test_build_provenance_records_individual_scores(self) -> None:
        """AC-5.4.3: Records individual scores."""
        from src.agents.msep.merger import build_provenance

        result = build_provenance(
            sbert_score=0.82,
            topic_boost=0.15,
            methods_used=[METHOD_SBERT],
        )

        assert result.sbert_score == pytest.approx(0.82)
        assert result.topic_boost == pytest.approx(0.15)

    def test_build_provenance_timestamp_iso8601_utc(self) -> None:
        """AC-5.4.4: Timestamp in ISO 8601 format (UTC)."""
        from src.agents.msep.merger import build_provenance

        before = datetime.now(timezone.utc)
        result = build_provenance(
            sbert_score=0.5,
            topic_boost=0.0,
            methods_used=[METHOD_SBERT],
        )
        after = datetime.now(timezone.utc)

        # Parse the timestamp
        timestamp = datetime.fromisoformat(result.timestamp)
        assert timestamp.tzinfo is not None  # Has timezone info
        assert before <= timestamp <= after

    def test_build_provenance_timestamp_format(self) -> None:
        """Timestamp string is valid ISO 8601."""
        from src.agents.msep.merger import build_provenance

        result = build_provenance(
            sbert_score=0.5,
            topic_boost=0.0,
            methods_used=[METHOD_SBERT],
        )

        # Should contain timezone indicator
        assert "+" in result.timestamp or "Z" in result.timestamp

    def test_build_provenance_empty_methods_list(self) -> None:
        """Handles empty methods list."""
        from src.agents.msep.merger import build_provenance

        result = build_provenance(
            sbert_score=0.0,
            topic_boost=0.0,
            methods_used=[],
        )

        assert result.methods_used == []

    def test_build_provenance_zero_scores(self) -> None:
        """Handles zero scores correctly."""
        from src.agents.msep.merger import build_provenance

        result = build_provenance(
            sbert_score=0.0,
            topic_boost=0.0,
            methods_used=[METHOD_TFIDF],
        )

        assert result.sbert_score == 0.0
        assert result.topic_boost == 0.0


# =============================================================================
# Integration Tests: Full Merger Flow
# =============================================================================


class TestMergerIntegration:
    """Integration tests for complete merger workflow."""

    def test_full_merge_workflow(self) -> None:
        """Test complete merge with all components."""
        from src.agents.msep.merger import (
            apply_topic_boost,
            build_provenance,
            calculate_dynamic_threshold,
            merge_results,
        )

        # Setup data
        similarity = [
            [1.0, 0.75, 0.55],
            [0.75, 1.0, 0.65],
            [0.55, 0.65, 1.0],
        ]
        topics = [0, 0, 1]
        keywords = [["k1", "k2"], ["k3"], ["k4", "k5", "k6"]]
        chapter_ids = ["A:ch1", "A:ch2", "A:ch3"]

        # Calculate threshold
        threshold = calculate_dynamic_threshold(
            corpus_size=len(chapter_ids),
            base_threshold=DEFAULT_THRESHOLD,
            use_dynamic=True,
        )

        # Small corpus should increase threshold
        assert threshold == pytest.approx(DEFAULT_THRESHOLD + THRESHOLD_ADJUSTMENT)

        # Merge results
        result = merge_results(
            similarity_matrix=similarity,
            topics=topics,
            keywords=keywords,
            chapter_ids=chapter_ids,
            threshold=threshold,
            top_k=2,
        )

        # Verify structure
        assert isinstance(result, EnrichedMetadata)
        assert len(result.chapters) == 3

        # Verify topic boost was applied correctly
        ch0 = result.chapters[0]
        if ch0.cross_references:
            # Ch0 -> Ch1 should have boost (same topic)
            ch1_ref = next(
                (xr for xr in ch0.cross_references if xr.target == "A:ch2"), None
            )
            if ch1_ref:
                assert ch1_ref.topic_boost == SAME_TOPIC_BOOST

    def test_merger_with_dispatcher_result_structure(self) -> None:
        """Test merger works with DispatchResult-like data."""
        from src.agents.msep.merger import merge_results

        # Simulate DispatchResult output
        dispatch_similarity = [[1.0, 0.8], [0.8, 1.0]]
        dispatch_topics = [0, 1]
        dispatch_keywords = [["async", "python"], ["java", "spring"]]

        result = merge_results(
            similarity_matrix=dispatch_similarity,
            topics=dispatch_topics,
            keywords=dispatch_keywords,
            chapter_ids=["Book:ch1", "Book:ch2"],
            threshold=0.5,
            top_k=5,
        )

        assert isinstance(result, EnrichedMetadata)
        assert len(result.chapters) == 2

    def test_processing_time_is_tracked(self) -> None:
        """EnrichedMetadata should have processing time."""
        from src.agents.msep.merger import merge_results

        result = merge_results(
            similarity_matrix=[[1.0, 0.5], [0.5, 1.0]],
            topics=[0, 0],
            keywords=[["kw1"], ["kw2"]],
            chapter_ids=["A:ch1", "A:ch2"],
            threshold=0.3,
            top_k=5,
        )

        # Processing time should be set (may be 0 or very small)
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0
