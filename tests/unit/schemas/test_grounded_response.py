"""Unit tests for GroundedResponse schema.

WBS Reference: WBS-KB6 - Cross-Reference Pipeline Orchestration
Tasks: KB6.1 - Create GroundedResponse schema

Acceptance Criteria:
- AC-KB6.4: Final output is GroundedResponse with content, citations, confidence, metadata
- AC-KB6.5: Metadata includes: cycles_used, participants, sources_consulted, processing_time

TDD Phase: RED
Exit Criteria:
- pytest tests/unit/schemas/test_grounded_response.py passes

Anti-Patterns Avoided:
- #1 (S1192): Test constants at module level
- Frozen Pydantic models for immutability
"""

from __future__ import annotations

import pytest
from typing import Any


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_TEST_CONTENT = "The repository pattern is implemented in src/repository.py"
_TEST_CONFIDENCE = 0.92
_TEST_CYCLES = 3
_TEST_PROCESSING_TIME = 2.5


# =============================================================================
# Import Tests (AC-KB6.4)
# =============================================================================


class TestGroundedResponseImports:
    """Tests for GroundedResponse module imports."""

    def test_grounded_response_importable(self) -> None:
        """GroundedResponse should be importable from schemas."""
        from src.schemas.grounded_response import GroundedResponse

        assert GroundedResponse is not None

    def test_grounded_response_metadata_importable(self) -> None:
        """GroundedResponseMetadata should be importable."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        assert GroundedResponseMetadata is not None

    def test_citation_entry_importable(self) -> None:
        """CitationEntry should be importable."""
        from src.schemas.grounded_response import CitationEntry

        assert CitationEntry is not None

    def test_termination_reason_importable(self) -> None:
        """TerminationReason enum should be importable."""
        from src.schemas.grounded_response import TerminationReason

        assert TerminationReason is not None


# =============================================================================
# TerminationReason Enum Tests (AC-KB6.3)
# =============================================================================


class TestTerminationReason:
    """Tests for TerminationReason enum."""

    def test_agreement_reached_exists(self) -> None:
        """AGREEMENT_REACHED should be a valid termination reason."""
        from src.schemas.grounded_response import TerminationReason

        assert TerminationReason.AGREEMENT_REACHED == "agreement_reached"

    def test_max_cycles_exists(self) -> None:
        """MAX_CYCLES should be a valid termination reason."""
        from src.schemas.grounded_response import TerminationReason

        assert TerminationReason.MAX_CYCLES == "max_cycles"

    def test_validation_passed_exists(self) -> None:
        """VALIDATION_PASSED should be a valid termination reason."""
        from src.schemas.grounded_response import TerminationReason

        assert TerminationReason.VALIDATION_PASSED == "validation_passed"

    def test_error_exists(self) -> None:
        """ERROR should be a valid termination reason."""
        from src.schemas.grounded_response import TerminationReason

        assert TerminationReason.ERROR == "error"


# =============================================================================
# CitationEntry Tests (AC-KB6.4)
# =============================================================================


class TestCitationEntry:
    """Tests for CitationEntry model."""

    def test_citation_entry_has_marker(self) -> None:
        """CitationEntry should have marker field."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
        )
        assert entry.marker == 1

    def test_citation_entry_has_source(self) -> None:
        """CitationEntry should have source field."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
        )
        assert entry.source == "src/repository.py"

    def test_citation_entry_has_source_type(self) -> None:
        """CitationEntry should have source_type field."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
        )
        assert entry.source_type == "code"

    def test_citation_entry_has_optional_lines(self) -> None:
        """CitationEntry should have optional lines field."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
            lines="42-56",
        )
        assert entry.lines == "42-56"

    def test_citation_entry_has_optional_participant(self) -> None:
        """CitationEntry should track which participant contributed it."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
            participant_id="qwen2.5-7b",
        )
        assert entry.participant_id == "qwen2.5-7b"

    def test_citation_entry_has_optional_cycle(self) -> None:
        """CitationEntry should track which cycle it came from."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
            cycle_number=2,
        )
        assert entry.cycle_number == 2

    def test_citation_entry_to_dict(self) -> None:
        """CitationEntry should serialize to dict."""
        from src.schemas.grounded_response import CitationEntry

        entry = CitationEntry(
            marker=1,
            source="src/repository.py",
            source_type="code",
        )
        data = entry.model_dump()
        assert data["marker"] == 1
        assert data["source"] == "src/repository.py"


# =============================================================================
# GroundedResponseMetadata Tests (AC-KB6.5)
# =============================================================================


class TestGroundedResponseMetadata:
    """Tests for GroundedResponseMetadata model."""

    def test_metadata_has_cycles_used(self) -> None:
        """Metadata should have cycles_used field."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        assert metadata.cycles_used == _TEST_CYCLES

    def test_metadata_has_participants(self) -> None:
        """Metadata should have participants list."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b", "deepseek-r1-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        assert "qwen2.5-7b" in metadata.participants
        assert "deepseek-r1-7b" in metadata.participants

    def test_metadata_has_sources_consulted(self) -> None:
        """Metadata should have sources_consulted list."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code", "books", "graph"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        assert "code" in metadata.sources_consulted
        assert "books" in metadata.sources_consulted
        assert "graph" in metadata.sources_consulted

    def test_metadata_has_processing_time(self) -> None:
        """Metadata should have processing_time_seconds field."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        assert metadata.processing_time_seconds == _TEST_PROCESSING_TIME

    def test_metadata_has_optional_termination_reason(self) -> None:
        """Metadata should have optional termination_reason field."""
        from src.schemas.grounded_response import (
            GroundedResponseMetadata,
            TerminationReason,
        )

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
            termination_reason=TerminationReason.AGREEMENT_REACHED,
        )
        assert metadata.termination_reason == TerminationReason.AGREEMENT_REACHED

    def test_metadata_has_optional_agreement_score(self) -> None:
        """Metadata should have optional agreement_score field."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
            agreement_score=0.92,
        )
        assert metadata.agreement_score == 0.92

    def test_metadata_has_optional_validation_passed(self) -> None:
        """Metadata should have optional validation_passed field."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
            validation_passed=True,
        )
        assert metadata.validation_passed is True

    def test_metadata_to_dict(self) -> None:
        """Metadata should serialize to dict with all fields."""
        from src.schemas.grounded_response import GroundedResponseMetadata

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        data = metadata.model_dump()
        assert "cycles_used" in data
        assert "participants" in data
        assert "sources_consulted" in data
        assert "processing_time_seconds" in data


# =============================================================================
# GroundedResponse Tests (AC-KB6.4, AC-KB6.5)
# =============================================================================


class TestGroundedResponseCore:
    """Core tests for GroundedResponse model."""

    def test_grounded_response_has_content(self) -> None:
        """GroundedResponse should have content field."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert response.content == _TEST_CONTENT

    def test_grounded_response_has_citations(self) -> None:
        """GroundedResponse should have citations list."""
        from src.schemas.grounded_response import (
            CitationEntry,
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[
                CitationEntry(marker=1, source="src/repo.py", source_type="code"),
            ],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert len(response.citations) == 1
        assert response.citations[0].marker == 1

    def test_grounded_response_has_confidence(self) -> None:
        """GroundedResponse should have confidence field."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert response.confidence == _TEST_CONFIDENCE

    def test_grounded_response_has_metadata(self) -> None:
        """GroundedResponse should have metadata field."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        metadata = GroundedResponseMetadata(
            cycles_used=_TEST_CYCLES,
            participants=["qwen2.5-7b"],
            sources_consulted=["code"],
            processing_time_seconds=_TEST_PROCESSING_TIME,
        )
        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=metadata,
        )
        assert response.metadata.cycles_used == _TEST_CYCLES

    def test_grounded_response_has_optional_query(self) -> None:
        """GroundedResponse should have optional query field."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
            query="Where is the repository pattern?",
        )
        assert response.query == "Where is the repository pattern?"

    def test_grounded_response_has_optional_footnotes(self) -> None:
        """GroundedResponse should have optional footnotes field."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
            footnotes="[^1]: src/repository.py, lines 42-56.",
        )
        assert "[^1]:" in response.footnotes


class TestGroundedResponseValidation:
    """Validation tests for GroundedResponse model."""

    def test_confidence_must_be_between_0_and_1(self) -> None:
        """Confidence should be validated as 0.0-1.0."""
        from pydantic import ValidationError

        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        with pytest.raises(ValidationError):
            GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=1.5,  # Invalid
                metadata=GroundedResponseMetadata(
                    cycles_used=_TEST_CYCLES,
                    participants=["qwen2.5-7b"],
                    sources_consulted=["code"],
                    processing_time_seconds=_TEST_PROCESSING_TIME,
                ),
            )

    def test_cycles_used_must_be_positive(self) -> None:
        """cycles_used should be validated as >= 1."""
        from pydantic import ValidationError

        from src.schemas.grounded_response import GroundedResponseMetadata

        with pytest.raises(ValidationError):
            GroundedResponseMetadata(
                cycles_used=0,  # Invalid
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            )

    def test_processing_time_must_be_positive(self) -> None:
        """processing_time_seconds should be validated as >= 0."""
        from pydantic import ValidationError

        from src.schemas.grounded_response import GroundedResponseMetadata

        with pytest.raises(ValidationError):
            GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=-1.0,  # Invalid
            )


class TestGroundedResponseSerialization:
    """Serialization tests for GroundedResponse model."""

    def test_grounded_response_to_dict(self) -> None:
        """GroundedResponse should serialize to dict."""
        from src.schemas.grounded_response import (
            CitationEntry,
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[
                CitationEntry(marker=1, source="src/repo.py", source_type="code"),
            ],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        data = response.model_dump()
        assert data["content"] == _TEST_CONTENT
        assert data["confidence"] == _TEST_CONFIDENCE
        assert len(data["citations"]) == 1
        assert data["metadata"]["cycles_used"] == _TEST_CYCLES

    def test_grounded_response_to_json(self) -> None:
        """GroundedResponse should serialize to JSON."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        json_str = response.model_dump_json()
        assert _TEST_CONTENT in json_str
        assert "cycles_used" in json_str


# =============================================================================
# Edge Cases
# =============================================================================


class TestGroundedResponseEdgeCases:
    """Edge case tests for GroundedResponse."""

    def test_empty_citations_allowed(self) -> None:
        """Empty citations list should be allowed."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert len(response.citations) == 0

    def test_multiple_citations(self) -> None:
        """Multiple citations should be supported."""
        from src.schemas.grounded_response import (
            CitationEntry,
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content=_TEST_CONTENT,
            citations=[
                CitationEntry(marker=1, source="src/repo.py", source_type="code"),
                CitationEntry(marker=2, source="Design Patterns", source_type="book"),
                CitationEntry(marker=3, source="Repository#abc123", source_type="graph"),
            ],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code", "books", "graph"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert len(response.citations) == 3
        assert response.citations[0].source_type == "code"
        assert response.citations[1].source_type == "book"
        assert response.citations[2].source_type == "graph"

    def test_unicode_content(self) -> None:
        """Unicode content should be supported."""
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        response = GroundedResponse(
            content="The pattern uses Müller's approach 日本語",
            citations=[],
            confidence=_TEST_CONFIDENCE,
            metadata=GroundedResponseMetadata(
                cycles_used=_TEST_CYCLES,
                participants=["qwen2.5-7b"],
                sources_consulted=["code"],
                processing_time_seconds=_TEST_PROCESSING_TIME,
            ),
        )
        assert "Müller" in response.content
        assert "日本語" in response.content
