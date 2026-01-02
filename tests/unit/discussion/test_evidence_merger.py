"""Unit tests for evidence_merger module.

WBS Reference: WBS-KB3 - Iterative Evidence Gathering
Tasks: KB3.5, KB3.6 - Implement merge_evidence() with deduplication and provenance
Acceptance Criteria:
- AC-KB3.3: Evidence from multiple sources merged without duplicates
- AC-KB3.4: merge_evidence() combines old + new evidence, preserving provenance

TDD Phase: RED - These tests will fail until implementation exists.
"""

from __future__ import annotations

import pytest

from src.discussion.models import CrossReferenceEvidence
from src.discussion.evidence_merger import (
    EvidenceMerger,
    MergeResult,
    merge_evidence,
)


# =============================================================================
# Test Constants
# =============================================================================

_SOURCE_TYPE_CODE = "code"
_SOURCE_TYPE_BOOK = "book"
_SOURCE_TYPE_GRAPH = "graph"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def code_evidence_1() -> CrossReferenceEvidence:
    """Create first code evidence item."""
    return CrossReferenceEvidence(
        source_type=_SOURCE_TYPE_CODE,
        content="class ParallelAgent:\n    async def run(self): ...",
        source_id="src/pipelines/agents.py#L135",
    )


@pytest.fixture
def code_evidence_2() -> CrossReferenceEvidence:
    """Create second code evidence item (different file)."""
    return CrossReferenceEvidence(
        source_type=_SOURCE_TYPE_CODE,
        content="class SequentialAgent:\n    def run(self): ...",
        source_id="src/pipelines/agents.py#L200",
    )


@pytest.fixture
def code_evidence_duplicate() -> CrossReferenceEvidence:
    """Create duplicate of code_evidence_1 (same source_id)."""
    return CrossReferenceEvidence(
        source_type=_SOURCE_TYPE_CODE,
        content="class ParallelAgent:\n    async def run(self): ...",
        source_id="src/pipelines/agents.py#L135",  # Same as code_evidence_1
    )


@pytest.fixture
def book_evidence() -> CrossReferenceEvidence:
    """Create book evidence item."""
    return CrossReferenceEvidence(
        source_type=_SOURCE_TYPE_BOOK,
        content="The Repository pattern mediates between domain and data layers.",
        source_id="ddia/ch3/p42",
    )


@pytest.fixture
def graph_evidence() -> CrossReferenceEvidence:
    """Create graph evidence item."""
    return CrossReferenceEvidence(
        source_type=_SOURCE_TYPE_GRAPH,
        content="ParallelAgent -> uses -> asyncio.gather",
        source_id="neo4j://concept/ParallelAgent",
    )


@pytest.fixture
def merger() -> EvidenceMerger:
    """Create default EvidenceMerger."""
    return EvidenceMerger()


# =============================================================================
# AC-KB3.3: Evidence from multiple sources merged without duplicates
# =============================================================================


class TestDeduplication:
    """Test evidence deduplication (AC-KB3.3)."""

    def test_merge_removes_duplicates_by_source_id(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        code_evidence_duplicate: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence removes duplicates based on source_id."""
        old_evidence = [code_evidence_1]
        new_evidence = [code_evidence_duplicate]
        
        result = merger.merge(old_evidence, new_evidence)
        
        # Should only have 1 item, not 2
        assert len(result.evidence) == 1
        assert result.duplicates_removed == 1

    def test_merge_keeps_unique_evidence(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        code_evidence_2: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence keeps evidence with different source_ids."""
        old_evidence = [code_evidence_1]
        new_evidence = [code_evidence_2]
        
        result = merger.merge(old_evidence, new_evidence)
        
        assert len(result.evidence) == 2
        assert result.duplicates_removed == 0

    def test_merge_handles_multiple_duplicates(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence handles multiple duplicates of same item."""
        # Create multiple duplicates
        old_evidence = [code_evidence_1, code_evidence_1]
        new_evidence = [code_evidence_1, code_evidence_1]
        
        result = merger.merge(old_evidence, new_evidence)
        
        assert len(result.evidence) == 1
        assert result.duplicates_removed == 3

    def test_merge_deduplicates_across_source_types(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence correctly merges evidence from different source types."""
        old_evidence = [code_evidence_1]
        new_evidence = [book_evidence, code_evidence_1]  # code_evidence_1 is duplicate
        
        result = merger.merge(old_evidence, new_evidence)
        
        assert len(result.evidence) == 2
        assert result.duplicates_removed == 1
        
        source_types = {e.source_type for e in result.evidence}
        assert _SOURCE_TYPE_CODE in source_types
        assert _SOURCE_TYPE_BOOK in source_types


# =============================================================================
# AC-KB3.4: merge_evidence() combines old + new evidence, preserving provenance
# =============================================================================


class TestProvenanceTracking:
    """Test cycle provenance tracking (AC-KB3.4)."""

    def test_merge_result_tracks_source_cycle_for_old(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """Old evidence tracked as from previous cycle."""
        old_evidence = [code_evidence_1]
        new_evidence = [book_evidence]
        
        result = merger.merge(old_evidence, new_evidence, current_cycle=2)
        
        # Check provenance tracking
        old_provenance = result.provenance.get(code_evidence_1.source_id)
        assert old_provenance is not None
        assert old_provenance["cycle"] < 2  # From previous cycle

    def test_merge_result_tracks_source_cycle_for_new(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """New evidence tracked as from current cycle."""
        old_evidence = [code_evidence_1]
        new_evidence = [book_evidence]
        
        result = merger.merge(old_evidence, new_evidence, current_cycle=2)
        
        new_provenance = result.provenance.get(book_evidence.source_id)
        assert new_provenance is not None
        assert new_provenance["cycle"] == 2

    def test_merge_preserves_original_cycle_for_duplicates(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        code_evidence_duplicate: CrossReferenceEvidence,
    ) -> None:
        """When duplicate found, preserve provenance from earlier cycle."""
        old_evidence = [code_evidence_1]
        new_evidence = [code_evidence_duplicate]
        
        result = merger.merge(old_evidence, new_evidence, current_cycle=3)
        
        # Should track original cycle, not current
        provenance = result.provenance.get(code_evidence_1.source_id)
        assert provenance["cycle"] < 3

    def test_provenance_includes_source_type(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """Provenance includes source_type information."""
        result = merger.merge([], [code_evidence_1], current_cycle=1)
        
        provenance = result.provenance.get(code_evidence_1.source_id)
        assert provenance["source_type"] == _SOURCE_TYPE_CODE


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestMergeEvidenceFunction:
    """Test merge_evidence() convenience function."""

    def test_merge_evidence_function_works(
        self,
        code_evidence_1: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence() convenience function produces correct result."""
        old_evidence = [code_evidence_1]
        new_evidence = [book_evidence]
        
        result = merge_evidence(old_evidence, new_evidence)
        
        assert isinstance(result, MergeResult)
        assert len(result.evidence) == 2

    def test_merge_evidence_function_with_cycle(
        self,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence() accepts current_cycle parameter."""
        result = merge_evidence([], [code_evidence_1], current_cycle=5)
        
        provenance = result.provenance.get(code_evidence_1.source_id)
        assert provenance["cycle"] == 5


# =============================================================================
# Empty Input Tests
# =============================================================================


class TestEmptyInputs:
    """Test handling of empty inputs."""

    def test_merge_empty_old_evidence(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence handles empty old_evidence list."""
        result = merger.merge([], [code_evidence_1])
        
        assert len(result.evidence) == 1
        assert result.duplicates_removed == 0

    def test_merge_empty_new_evidence(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """merge_evidence handles empty new_evidence list."""
        result = merger.merge([code_evidence_1], [])
        
        assert len(result.evidence) == 1
        assert result.duplicates_removed == 0

    def test_merge_both_empty(
        self,
        merger: EvidenceMerger,
    ) -> None:
        """merge_evidence handles both lists empty."""
        result = merger.merge([], [])
        
        assert len(result.evidence) == 0
        assert result.duplicates_removed == 0


# =============================================================================
# MergeResult Tests
# =============================================================================


class TestMergeResult:
    """Test MergeResult dataclass."""

    def test_merge_result_to_dict(
        self,
        code_evidence_1: CrossReferenceEvidence,
    ) -> None:
        """MergeResult can be converted to dict."""
        result = MergeResult(
            evidence=[code_evidence_1],
            duplicates_removed=0,
            provenance={code_evidence_1.source_id: {"cycle": 1, "source_type": "code"}},
        )
        
        data = result.to_dict()
        
        assert data["duplicates_removed"] == 0
        assert len(data["evidence"]) == 1
        assert code_evidence_1.source_id in data["provenance"]

    def test_merge_result_total_property(
        self,
        code_evidence_1: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """MergeResult.total returns correct count."""
        result = MergeResult(
            evidence=[code_evidence_1, book_evidence],
            duplicates_removed=1,
            provenance={},
        )
        
        assert result.total == 2


# =============================================================================
# Ordering Tests
# =============================================================================


class TestMergeOrdering:
    """Test evidence ordering after merge."""

    def test_merge_preserves_order_old_before_new(
        self,
        merger: EvidenceMerger,
        code_evidence_1: CrossReferenceEvidence,
        code_evidence_2: CrossReferenceEvidence,
        book_evidence: CrossReferenceEvidence,
    ) -> None:
        """Old evidence appears before new evidence in result."""
        old_evidence = [code_evidence_1]
        new_evidence = [book_evidence, code_evidence_2]
        
        result = merger.merge(old_evidence, new_evidence)
        
        # First item should be from old evidence
        assert result.evidence[0].source_id == code_evidence_1.source_id
