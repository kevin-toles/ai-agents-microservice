"""Tests for CitationManager.

TDD tests for WBS-AGT17: Citation Flow & Audit.

Acceptance Criteria Coverage:
- AC-17.1: CitationManager tracks sources through pipeline

Exit Criteria:
- CitationManager assigns unique [^N] markers
- Sources are tracked through pipeline stages
- Citations can be retrieved by marker

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

import pytest
from typing import Any

from pydantic import BaseModel


# =============================================================================
# AC-17.1: CitationManager Tests
# =============================================================================

class TestCitationManagerCreation:
    """Tests for CitationManager instantiation."""

    def test_citation_manager_can_be_imported(self) -> None:
        """CitationManager can be imported from src.citations.manager."""
        from src.citations.manager import CitationManager
        
        assert isinstance(CitationManager, type)

    def test_citation_manager_can_be_instantiated(self) -> None:
        """CitationManager can be instantiated with default settings."""
        from src.citations.manager import CitationManager
        
        manager = CitationManager()
        
        assert isinstance(manager, CitationManager)

    def test_citation_manager_starts_with_empty_citations(self) -> None:
        """New CitationManager has no citations."""
        from src.citations.manager import CitationManager
        
        manager = CitationManager()
        
        assert len(manager.get_all_citations()) == 0

    def test_citation_manager_starts_with_marker_at_one(self) -> None:
        """New CitationManager starts marker numbering at 1."""
        from src.citations.manager import CitationManager
        
        manager = CitationManager()
        
        assert manager.next_marker == 1


class TestCitationManagerAddSource:
    """Tests for adding sources to CitationManager."""

    def test_add_source_returns_marker_number(self) -> None:
        """add_source returns the assigned marker number."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        )
        
        marker = manager.add_source(source)
        
        assert marker == 1

    def test_add_source_increments_marker(self) -> None:
        """Each add_source call increments the marker number."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source1 = SourceMetadata(
            source_type=SourceType.BOOK,
            title="Book 1",
            year=2020,
        )
        source2 = SourceMetadata(
            source_type=SourceType.BOOK,
            title="Book 2",
            year=2021,
        )
        
        marker1 = manager.add_source(source1)
        marker2 = manager.add_source(source2)
        
        assert marker1 == 1
        assert marker2 == 2

    def test_add_source_with_context(self) -> None:
        """add_source accepts optional context string."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.CODE,
            repo="code-reference-engine",
            file_path="backend/ddd/repository.py",
            line_range="12-45",
        )
        
        marker = manager.add_source(source, context="Repository pattern example")
        
        citation = manager.get_citation(marker)
        assert citation is not None
        assert citation.context == "Repository pattern example"

    def test_add_source_with_retrieval_score(self) -> None:
        """add_source accepts optional retrieval_score."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            title="Test Book",
            year=2020,
            similarity_score=0.89,
        )
        
        marker = manager.add_source(source, retrieval_score=0.89)
        
        assert manager.get_retrieval_score(marker) == pytest.approx(0.89)

    def test_add_source_tracks_pipeline_stage(self) -> None:
        """add_source can track which pipeline stage added the citation."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            title="Test Book",
            year=2020,
        )
        
        marker = manager.add_source(source, stage="cross_reference")
        
        assert manager.get_stage(marker) == "cross_reference"


class TestCitationManagerRetrieval:
    """Tests for retrieving citations."""

    def test_get_citation_by_marker(self) -> None:
        """Can retrieve Citation by marker number."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            author="Test Author",
            title="Test Book",
            year=2020,
        )
        marker = manager.add_source(source)
        
        citation = manager.get_citation(marker)
        
        assert citation is not None
        assert isinstance(citation, Citation)
        assert citation.marker == marker
        assert citation.source.title == "Test Book"

    def test_get_citation_returns_none_for_invalid_marker(self) -> None:
        """get_citation returns None for non-existent marker."""
        from src.citations.manager import CitationManager
        
        manager = CitationManager()
        
        citation = manager.get_citation(999)
        
        assert citation is None

    def test_get_all_citations(self) -> None:
        """get_all_citations returns list of all Citation objects."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Book 1", year=2020))
        manager.add_source(SourceMetadata(source_type=SourceType.CODE, repo="test", file_path="test.py"))
        manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Book 2", year=2021))
        
        citations = manager.get_all_citations()
        
        assert len(citations) == 3
        assert citations[0].marker == 1
        assert citations[1].marker == 2
        assert citations[2].marker == 3

    def test_get_marker_for_source_returns_existing_marker(self) -> None:
        """If same source is added twice, can find the marker."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        )
        
        marker1 = manager.add_source(source)
        # Find existing marker by source
        found_marker = manager.find_marker(source)
        
        assert found_marker == marker1


class TestCitationManagerMarkerFormat:
    """Tests for marker formatting."""

    def test_get_inline_marker(self) -> None:
        """get_inline_marker returns [^N] format."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(source_type=SourceType.BOOK, title="Test", year=2020)
        marker = manager.add_source(source)
        
        inline = manager.get_inline_marker(marker)
        
        assert inline == "[^1]"

    def test_get_inline_marker_for_multiple(self) -> None:
        """Multiple citations get sequential markers."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Book 1", year=2020))
        marker2 = manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Book 2", year=2021))
        marker3 = manager.add_source(SourceMetadata(source_type=SourceType.CODE, repo="test", file_path="test.py"))
        
        assert manager.get_inline_marker(marker2) == "[^2]"
        assert manager.get_inline_marker(marker3) == "[^3]"


class TestCitationManagerUsageTracking:
    """Tests for tracking citation usage context."""

    def test_record_usage_context(self) -> None:
        """Can record where a citation was used."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(source_type=SourceType.BOOK, title="Test", year=2020)
        marker = manager.add_source(source)
        
        manager.record_usage(marker, usage_context="Used in summary paragraph 1")
        
        usage = manager.get_usage_context(marker)
        assert "Used in summary paragraph 1" in usage

    def test_record_multiple_usages(self) -> None:
        """Can record multiple usage contexts for same citation."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        source = SourceMetadata(source_type=SourceType.BOOK, title="Test", year=2020)
        marker = manager.add_source(source)
        
        manager.record_usage(marker, usage_context="First mention")
        manager.record_usage(marker, usage_context="Second mention")
        
        usages = manager.get_usage_context(marker)
        assert len(usages) == 2


class TestCitationManagerPipelineIntegration:
    """Tests for pipeline stage integration."""

    def test_get_citations_by_stage(self) -> None:
        """Can filter citations by pipeline stage."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        manager.add_source(
            SourceMetadata(source_type=SourceType.BOOK, title="Book 1", year=2020),
            stage="cross_reference",
        )
        manager.add_source(
            SourceMetadata(source_type=SourceType.CODE, repo="test", file_path="test.py"),
            stage="cross_reference",
        )
        manager.add_source(
            SourceMetadata(source_type=SourceType.BOOK, title="Book 2", year=2021),
            stage="summarize_content",
        )
        
        xref_citations = manager.get_citations_by_stage("cross_reference")
        
        assert len(xref_citations) == 2

    def test_clear_citations(self) -> None:
        """Can clear all citations."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Test", year=2020))
        manager.add_source(SourceMetadata(source_type=SourceType.BOOK, title="Test 2", year=2021))
        
        manager.clear()
        
        assert len(manager.get_all_citations()) == 0
        assert manager.next_marker == 1

    def test_to_audit_records(self) -> None:
        """Can export citations as audit records."""
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        manager = CitationManager()
        manager.add_source(
            SourceMetadata(source_type=SourceType.BOOK, title="Test", year=2020),
            retrieval_score=0.85,
            stage="cross_reference",
        )
        manager.record_usage(1, "Used in summary")
        
        records = manager.to_audit_records(
            conversation_id="conv-123",
            message_id="msg-456",
        )
        
        assert len(records) == 1
        assert records[0].conversation_id == "conv-123"
        assert records[0].message_id == "msg-456"
        assert records[0].retrieval_score == pytest.approx(0.85)


__all__ = [
    "TestCitationManagerCreation",
    "TestCitationManagerAddSource",
    "TestCitationManagerRetrieval",
    "TestCitationManagerMarkerFormat",
    "TestCitationManagerUsageTracking",
    "TestCitationManagerPipelineIntegration",
]
