"""Unit tests for ProvenanceTracker.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Tasks: KB5.1, KB5.2, KB5.9

Acceptance Criteria:
- AC-KB5.4: ProvenanceTracker logs: claim, source, participant, cycle

Exit Criteria:
- pytest tests/unit/discussion/test_provenance.py passes with 100% coverage
- Audit trail shows: "Claim X from Participant A in Cycle 2, source: agents.py#L135"

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Low cognitive complexity via focused test classes
- Frozen dataclasses for test fixtures
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Imports (Will Fail in RED Phase)
# =============================================================================


class TestProvenanceImports:
    """Test that ProvenanceTracker can be imported."""

    def test_provenance_tracker_importable(self) -> None:
        """ProvenanceTracker class should be importable."""
        from src.discussion.provenance import ProvenanceTracker

        assert ProvenanceTracker is not None

    def test_provenance_entry_importable(self) -> None:
        """ProvenanceEntry dataclass should be importable."""
        from src.discussion.provenance import ProvenanceEntry

        assert ProvenanceEntry is not None

    def test_provenance_config_importable(self) -> None:
        """ProvenanceConfig dataclass should be importable."""
        from src.discussion.provenance import ProvenanceConfig

        assert ProvenanceConfig is not None


# =============================================================================
# ProvenanceEntry Tests (AC-KB5.4)
# =============================================================================


class TestProvenanceEntry:
    """Tests for ProvenanceEntry dataclass."""

    def test_entry_has_claim_field(self) -> None:
        """ProvenanceEntry must have claim field."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="The rate limiter uses Redis.",
            source="middleware.py#L135",
            participant_id="llm-1",
            cycle_number=2,
        )
        assert entry.claim == "The rate limiter uses Redis."

    def test_entry_has_source_field(self) -> None:
        """ProvenanceEntry must have source field."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Test claim",
            source="agents.py#L42",
            participant_id="llm-1",
            cycle_number=1,
        )
        assert entry.source == "agents.py#L42"

    def test_entry_has_participant_id_field(self) -> None:
        """ProvenanceEntry must have participant_id field."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Test claim",
            source="test.py",
            participant_id="qwen-7b",
            cycle_number=1,
        )
        assert entry.participant_id == "qwen-7b"

    def test_entry_has_cycle_number_field(self) -> None:
        """ProvenanceEntry must have cycle_number field."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Test claim",
            source="test.py",
            participant_id="llm-1",
            cycle_number=3,
        )
        assert entry.cycle_number == 3

    def test_entry_has_citation_marker_field(self) -> None:
        """ProvenanceEntry should have optional citation_marker field."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Test claim [^1]",
            source="test.py",
            participant_id="llm-1",
            cycle_number=1,
            citation_marker=1,
        )
        assert entry.citation_marker == 1

    def test_entry_is_frozen(self) -> None:
        """ProvenanceEntry should be immutable (frozen)."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Test",
            source="test.py",
            participant_id="llm-1",
            cycle_number=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.claim = "Modified"  # type: ignore

    def test_entry_to_dict(self) -> None:
        """ProvenanceEntry should serialize to dict."""
        from src.discussion.provenance import ProvenanceEntry

        entry = ProvenanceEntry(
            claim="Cache uses Redis.",
            source="cache.py#L10",
            participant_id="deepseek-r1",
            cycle_number=2,
            citation_marker=3,
        )
        d = entry.to_dict()
        assert d["claim"] == "Cache uses Redis."
        assert d["source"] == "cache.py#L10"
        assert d["participant_id"] == "deepseek-r1"
        assert d["cycle_number"] == 2
        assert d["citation_marker"] == 3


# =============================================================================
# ProvenanceConfig Tests
# =============================================================================


class TestProvenanceConfig:
    """Tests for ProvenanceConfig dataclass."""

    def test_default_track_sources_enabled(self) -> None:
        """track_sources should default to True."""
        from src.discussion.provenance import ProvenanceConfig

        config = ProvenanceConfig()
        assert config.track_sources is True

    def test_default_track_participants_enabled(self) -> None:
        """track_participants should default to True."""
        from src.discussion.provenance import ProvenanceConfig

        config = ProvenanceConfig()
        assert config.track_participants is True

    def test_default_track_cycles_enabled(self) -> None:
        """track_cycles should default to True."""
        from src.discussion.provenance import ProvenanceConfig

        config = ProvenanceConfig()
        assert config.track_cycles is True

    def test_config_is_frozen(self) -> None:
        """ProvenanceConfig should be immutable."""
        from src.discussion.provenance import ProvenanceConfig

        config = ProvenanceConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.track_sources = False  # type: ignore


# =============================================================================
# ProvenanceTracker Core Tests (AC-KB5.4)
# =============================================================================


class TestProvenanceTrackerCore:
    """Core tests for ProvenanceTracker class."""

    def test_tracker_instantiation(self) -> None:
        """ProvenanceTracker should instantiate with default config."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_tracker_with_custom_config(self) -> None:
        """ProvenanceTracker should accept custom config."""
        from src.discussion.provenance import ProvenanceConfig, ProvenanceTracker

        config = ProvenanceConfig(track_sources=False)
        tracker = ProvenanceTracker(config=config)
        assert tracker.config.track_sources is False

    def test_tracker_track_claim_method_exists(self) -> None:
        """ProvenanceTracker should have track_claim method."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        assert hasattr(tracker, "track_claim")
        assert callable(tracker.track_claim)

    def test_tracker_get_entries_method_exists(self) -> None:
        """ProvenanceTracker should have get_entries method."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        assert hasattr(tracker, "get_entries")
        assert callable(tracker.get_entries)


# =============================================================================
# ProvenanceTracker.track_claim Tests (KB5.2)
# =============================================================================


class TestProvenanceTrackerTrackClaim:
    """Tests for ProvenanceTracker.track_claim method."""

    def test_track_claim_creates_entry(self) -> None:
        """track_claim should create a ProvenanceEntry."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="The rate limiter is in middleware.",
            source="middleware.py#L135",
            participant_id="llm-1",
            cycle_number=2,
        )
        entries = tracker.get_entries()
        assert len(entries) == 1

    def test_track_claim_stores_all_fields(self) -> None:
        """track_claim should store claim, source, participant, cycle."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="Test claim",
            source="test.py#L10",
            participant_id="qwen-7b",
            cycle_number=3,
        )
        entry = tracker.get_entries()[0]
        assert entry.claim == "Test claim"
        assert entry.source == "test.py#L10"
        assert entry.participant_id == "qwen-7b"
        assert entry.cycle_number == 3

    def test_track_multiple_claims(self) -> None:
        """track_claim should accumulate multiple entries."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Claim 1", "a.py", "llm-1", 1)
        tracker.track_claim("Claim 2", "b.py", "llm-2", 1)
        tracker.track_claim("Claim 3", "c.py", "llm-1", 2)
        assert len(tracker.get_entries()) == 3

    def test_track_claim_with_citation_marker(self) -> None:
        """track_claim should optionally include citation marker."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="Test claim [^1]",
            source="test.py",
            participant_id="llm-1",
            cycle_number=1,
            citation_marker=1,
        )
        entry = tracker.get_entries()[0]
        assert entry.citation_marker == 1

    def test_track_claim_returns_entry(self) -> None:
        """track_claim should return the created entry."""
        from src.discussion.provenance import ProvenanceEntry, ProvenanceTracker

        tracker = ProvenanceTracker()
        entry = tracker.track_claim(
            claim="Test",
            source="test.py",
            participant_id="llm-1",
            cycle_number=1,
        )
        assert isinstance(entry, ProvenanceEntry)
        assert entry.claim == "Test"


# =============================================================================
# ProvenanceTracker Query Methods Tests
# =============================================================================


class TestProvenanceTrackerQueries:
    """Tests for ProvenanceTracker query methods."""

    def test_get_entries_by_participant(self) -> None:
        """get_entries_by_participant filters by participant_id."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Claim A", "a.py", "llm-1", 1)
        tracker.track_claim("Claim B", "b.py", "llm-2", 1)
        tracker.track_claim("Claim C", "c.py", "llm-1", 2)

        entries = tracker.get_entries_by_participant("llm-1")
        assert len(entries) == 2
        assert all(e.participant_id == "llm-1" for e in entries)

    def test_get_entries_by_cycle(self) -> None:
        """get_entries_by_cycle filters by cycle_number."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Claim A", "a.py", "llm-1", 1)
        tracker.track_claim("Claim B", "b.py", "llm-2", 1)
        tracker.track_claim("Claim C", "c.py", "llm-1", 2)

        entries = tracker.get_entries_by_cycle(1)
        assert len(entries) == 2
        assert all(e.cycle_number == 1 for e in entries)

    def test_get_entries_by_source(self) -> None:
        """get_entries_by_source filters by source."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Claim A", "file.py#L10", "llm-1", 1)
        tracker.track_claim("Claim B", "file.py#L20", "llm-2", 1)
        tracker.track_claim("Claim C", "other.py", "llm-1", 2)

        entries = tracker.get_entries_by_source("file.py")
        assert len(entries) == 2
        assert all("file.py" in e.source for e in entries)


# =============================================================================
# ProvenanceTracker Export/Audit Trail Tests (AC-KB5.5)
# =============================================================================


class TestProvenanceTrackerExport:
    """Tests for ProvenanceTracker export methods."""

    def test_to_audit_trail(self) -> None:
        """to_audit_trail should return list of dicts."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="Rate limiter uses Redis.",
            source="middleware.py#L135",
            participant_id="llm-1",
            cycle_number=2,
        )
        trail = tracker.to_audit_trail()
        assert isinstance(trail, list)
        assert len(trail) == 1
        assert trail[0]["claim"] == "Rate limiter uses Redis."
        assert trail[0]["source"] == "middleware.py#L135"
        assert trail[0]["participant_id"] == "llm-1"
        assert trail[0]["cycle_number"] == 2

    def test_to_audit_trail_format_matches_exit_criteria(self) -> None:
        """Audit trail should match: 'Claim X from Participant A in Cycle 2, source: agents.py#L135'."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="The cache is in memory.",
            source="agents.py#L135",
            participant_id="Participant A",
            cycle_number=2,
        )
        formatted = tracker.format_audit_trail()
        assert "The cache is in memory." in formatted
        assert "Participant A" in formatted
        assert "Cycle 2" in formatted
        assert "agents.py#L135" in formatted

    def test_clear_entries(self) -> None:
        """clear should remove all entries."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Test", "test.py", "llm-1", 1)
        assert len(tracker.get_entries()) == 1
        tracker.clear()
        assert len(tracker.get_entries()) == 0


# =============================================================================
# ProvenanceTracker with DiscussionCycle Integration
# =============================================================================


class TestProvenanceTrackerIntegration:
    """Integration tests with discussion module components."""

    def test_track_from_discussion_cycle(self) -> None:
        """track_from_cycle should extract claims from analyses."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        from src.discussion.provenance import ProvenanceTracker

        analyses = [
            ParticipantAnalysis(
                participant_id="llm-1",
                model_id="qwen-7b",
                content="The cache uses Redis. [^1]",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="llm-2",
                model_id="deepseek-r1",
                content="Redis is used for caching. [^2]",
                confidence=0.85,
            ),
        ]
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=analyses,
            agreement_score=0.92,
        )

        tracker = ProvenanceTracker()
        tracker.track_from_cycle(cycle, sources=["cache.py"])
        entries = tracker.get_entries()
        assert len(entries) >= 2  # At least one claim per participant

    def test_track_consensus_claims(self) -> None:
        """track_consensus should record consensus claims with provenance."""
        from src.discussion.consensus import ConsensusResult
        from src.discussion.provenance import ProvenanceTracker

        consensus = ConsensusResult(
            content="The system uses Redis for caching.",
            claims=[
                {"text": "Redis is used.", "participants": ["llm-1", "llm-2"]},
            ],
            confidence=0.95,
            participant_contributions={"llm-1": 1, "llm-2": 1},
        )

        tracker = ProvenanceTracker()
        tracker.track_consensus(consensus, source="cache.py", cycle_number=2)
        entries = tracker.get_entries()
        assert len(entries) >= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestProvenanceEdgeCases:
    """Edge case tests for ProvenanceTracker."""

    def test_empty_tracker(self) -> None:
        """Empty tracker returns empty list."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        assert tracker.get_entries() == []
        assert tracker.to_audit_trail() == []

    def test_claim_with_no_source(self) -> None:
        """Claims with empty source should be allowed."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="Unsourced claim",
            source="",
            participant_id="llm-1",
            cycle_number=1,
        )
        assert len(tracker.get_entries()) == 1

    def test_duplicate_claims_tracked_separately(self) -> None:
        """Same claim from different participants tracked separately."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim("Same claim", "a.py", "llm-1", 1)
        tracker.track_claim("Same claim", "a.py", "llm-2", 1)
        assert len(tracker.get_entries()) == 2

    def test_special_characters_in_claim(self) -> None:
        """Claims with special characters handled correctly."""
        from src.discussion.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.track_claim(
            claim="Code uses `async/await` [^1] pattern.",
            source="test.py",
            participant_id="llm-1",
            cycle_number=1,
        )
        entry = tracker.get_entries()[0]
        assert "`async/await`" in entry.claim
        assert "[^1]" in entry.claim
