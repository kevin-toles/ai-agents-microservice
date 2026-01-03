"""Tests for ChicagoFormatter.

TDD tests for WBS-AGT17: Citation Flow & Audit.

Acceptance Criteria Coverage:
- AC-17.2: Chicago-style footnote formatting

Exit Criteria:
- Book format: [^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.
- Code format: [^N]: `repo/path/file.py`, commit `hash`, lines X-Y.
- Schema format: [^N]: `repo/schemas/file.json`, version X.Y.Z.
- Internal doc format: [^N]: service, *Document* (Date), §Section.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

import pytest
from typing import Any


# =============================================================================
# AC-17.2: ChicagoFormatter Tests
# =============================================================================

class TestChicagoFormatterCreation:
    """Tests for ChicagoFormatter instantiation."""

    def test_chicago_formatter_can_be_imported(self) -> None:
        """ChicagoFormatter can be imported from src.citations.formatter."""
        from src.citations.formatter import ChicagoFormatter
        
        # Verify import succeeded by checking it's a type
        assert isinstance(ChicagoFormatter, type)

    def test_chicago_formatter_can_be_instantiated(self) -> None:
        """ChicagoFormatter can be instantiated."""
        from src.citations.formatter import ChicagoFormatter
        
        formatter = ChicagoFormatter()
        
        # Verify instance is of correct type
        assert isinstance(formatter, ChicagoFormatter)


class TestChicagoFormatterBookFormat:
    """Tests for book citation formatting."""

    def test_format_book_citation_basic(self) -> None:
        """Formats basic book citation in Chicago style."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=1,
            source=SourceMetadata(
                source_type=SourceType.BOOK,
                author="Fowler, Martin",
                title="Patterns of Enterprise Application Architecture",
                year=2002,
            ),
        )
        
        result = formatter.format(citation)
        
        assert result.startswith("[^1]:")
        assert "Fowler, Martin" in result
        assert "*Patterns of Enterprise Application Architecture*" in result
        assert "2002" in result

    def test_format_book_with_publisher(self) -> None:
        """Formats book with publisher and city."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=1,
            source=SourceMetadata(
                source_type=SourceType.BOOK,
                author="Evans, Eric",
                title="Domain-Driven Design",
                year=2003,
                publisher="Addison-Wesley",
                publication_city="Boston",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "(Boston: Addison-Wesley, 2003)" in result

    def test_format_book_with_pages(self) -> None:
        """Formats book with page numbers."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=3,
            source=SourceMetadata(
                source_type=SourceType.BOOK,
                author="Martin, Robert C.",
                title="Clean Code",
                year=2008,
                pages="45-67",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "[^3]:" in result
        assert "45-67" in result


class TestChicagoFormatterCodeFormat:
    """Tests for code citation formatting."""

    def test_format_code_citation_basic(self) -> None:
        """Formats basic code citation."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=2,
            source=SourceMetadata(
                source_type=SourceType.CODE,
                repo="code-reference-engine",
                file_path="backend/ddd/repository.py",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "[^2]:" in result
        assert "`code-reference-engine/backend/ddd/repository.py`" in result

    def test_format_code_with_commit_hash(self) -> None:
        """Formats code citation with commit hash."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=5,
            source=SourceMetadata(
                source_type=SourceType.CODE,
                repo="ai-agents",
                file_path="src/pipelines/orchestrator.py",
                commit_hash="abc123def",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "commit `abc123def`" in result

    def test_format_code_with_line_range(self) -> None:
        """Formats code citation with line range."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=4,
            source=SourceMetadata(
                source_type=SourceType.CODE,
                repo="semantic-search-service",
                file_path="src/search.py",
                line_range="120-145",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "lines 120-145" in result


class TestChicagoFormatterSchemaFormat:
    """Tests for schema citation formatting."""

    def test_format_schema_citation_basic(self) -> None:
        """Formats basic schema citation."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=6,
            source=SourceMetadata(
                source_type=SourceType.SCHEMA,
                repo="llm-gateway",
                file_path="schemas/request.json",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "[^6]:" in result
        assert "`llm-gateway/schemas/request.json`" in result

    def test_format_schema_with_version(self) -> None:
        """Formats schema citation with version."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=7,
            source=SourceMetadata(
                source_type=SourceType.SCHEMA,
                repo="ai-models",
                file_path="config/models.yaml",
                version="1.2.3",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "version 1.2.3" in result


class TestChicagoFormatterInternalDocFormat:
    """Tests for internal document citation formatting."""

    def test_format_internal_doc_basic(self) -> None:
        """Formats basic internal document citation."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=8,
            source=SourceMetadata(
                source_type=SourceType.INTERNAL_DOC,
                service="semantic-search-service",
                title="Search API Reference",
                date="2024-01-15",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "[^8]:" in result
        assert "semantic-search-service" in result
        assert "*Search API Reference*" in result
        assert "2024-01-15" in result

    def test_format_internal_doc_with_section(self) -> None:
        """Formats internal doc with section reference."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=9,
            source=SourceMetadata(
                source_type=SourceType.INTERNAL_DOC,
                service="inference-service",
                title="Architecture Guide",
                date="2024-03-01",
                section="Conversation Store",
            ),
        )
        
        result = formatter.format(citation)
        
        assert "§Conversation Store" in result


class TestChicagoFormatterMultipleCitations:
    """Tests for formatting multiple citations."""

    def test_format_footnotes_section(self) -> None:
        """Formats all citations as footnotes section."""
        from src.citations.formatter import ChicagoFormatter
        from src.citations.manager import CitationManager
        from src.schemas.citations import SourceMetadata, SourceType
        
        formatter = ChicagoFormatter()
        manager = CitationManager()
        manager.add_source(SourceMetadata(
            source_type=SourceType.BOOK,
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        ))
        manager.add_source(SourceMetadata(
            source_type=SourceType.CODE,
            repo="test-repo",
            file_path="src/example.py",
        ))
        
        result = formatter.format_footnotes(manager.get_all_citations())
        
        assert "[^1]:" in result
        assert "[^2]:" in result
        assert result.count("\n") >= 1  # Each citation on new line

    def test_format_footnotes_empty(self) -> None:
        """Returns empty string for no citations."""
        from src.citations.formatter import ChicagoFormatter
        
        formatter = ChicagoFormatter()
        
        result = formatter.format_footnotes([])
        
        assert result == ""


class TestChicagoFormatterIntegration:
    """Integration tests with Citation class chicago_format."""

    def test_uses_citation_chicago_format_method(self) -> None:
        """ChicagoFormatter leverages existing Citation.chicago_format()."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=1,
            source=SourceMetadata(
                source_type=SourceType.BOOK,
                author="Test Author",
                title="Test Book",
                year=2020,
            ),
        )
        
        # Get format from both methods
        formatter_result = formatter.format(citation)
        direct_result = citation.chicago_format()
        
        # Formatter should produce result consistent with Citation's method
        assert "[^1]:" in formatter_result
        # The core content should match
        assert "Test Author" in formatter_result
        assert "Test Book" in formatter_result
        # Direct result should contain core content too
        assert "Test Author" in direct_result

    def test_format_adds_marker_prefix(self) -> None:
        """ChicagoFormatter adds [^N]: prefix to Citation.chicago_format() output."""
        from src.citations.formatter import ChicagoFormatter
        from src.schemas.citations import SourceMetadata, SourceType, Citation
        
        formatter = ChicagoFormatter()
        citation = Citation(
            marker=42,
            source=SourceMetadata(
                source_type=SourceType.BOOK,
                title="Test",
                year=2020,
            ),
        )
        
        result = formatter.format(citation)
        
        assert result.startswith("[^42]:")


__all__ = [
    "TestChicagoFormatterCreation",
    "TestChicagoFormatterBookFormat",
    "TestChicagoFormatterCodeFormat",
    "TestChicagoFormatterSchemaFormat",
    "TestChicagoFormatterInternalDocFormat",
    "TestChicagoFormatterMultipleCitations",
    "TestChicagoFormatterIntegration",
]
