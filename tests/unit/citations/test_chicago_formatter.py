"""Unit tests for Chicago-style citation formatter.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Tasks: KB5.8

Acceptance Criteria:
- AC-KB5.6: Chicago-style footnotes generated for all citation types

Exit Criteria:
- book, code, graph citations formatted correctly

Reference Templates (from AGENT_FUNCTIONS_ARCHITECTURE.md):
- Book: [^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.
- Code: [^N]: `repo/path/file.py`, commit `hash`, lines X-Y.
- Schema: [^N]: `repo/schemas/file.json`, version X.Y.Z.
- Internal Doc: [^N]: service, *Document* (Date), §Section.

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Low cognitive complexity via focused test classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Imports
# =============================================================================


class TestChicagoFormatterImports:
    """Test that Chicago formatter components can be imported."""

    def test_chicago_citation_formatter_importable(self) -> None:
        """ChicagoCitationFormatter class should be importable from citations module."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        assert ChicagoCitationFormatter is not None

    def test_citation_type_enum_importable(self) -> None:
        """CitationType enum should be importable."""
        from src.citations.chicago_formatter import CitationType

        assert CitationType is not None


# =============================================================================
# CitationType Enum Tests
# =============================================================================


class TestCitationType:
    """Tests for CitationType enum."""

    def test_book_type_exists(self) -> None:
        """BOOK citation type should exist."""
        from src.citations.chicago_formatter import CitationType

        assert hasattr(CitationType, "BOOK")

    def test_code_type_exists(self) -> None:
        """CODE citation type should exist."""
        from src.citations.chicago_formatter import CitationType

        assert hasattr(CitationType, "CODE")

    def test_schema_type_exists(self) -> None:
        """SCHEMA citation type should exist."""
        from src.citations.chicago_formatter import CitationType

        assert hasattr(CitationType, "SCHEMA")

    def test_graph_type_exists(self) -> None:
        """GRAPH citation type should exist."""
        from src.citations.chicago_formatter import CitationType

        assert hasattr(CitationType, "GRAPH")

    def test_internal_doc_type_exists(self) -> None:
        """INTERNAL_DOC citation type should exist."""
        from src.citations.chicago_formatter import CitationType

        assert hasattr(CitationType, "INTERNAL_DOC")


# =============================================================================
# ChicagoCitationFormatter Core Tests
# =============================================================================


class TestChicagoCitationFormatterCore:
    """Core tests for ChicagoCitationFormatter class."""

    def test_formatter_instantiation(self) -> None:
        """ChicagoCitationFormatter should instantiate."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        assert formatter is not None

    def test_format_method_exists(self) -> None:
        """format method should exist."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        assert hasattr(formatter, "format")
        assert callable(formatter.format)

    def test_format_book_method_exists(self) -> None:
        """format_book method should exist."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        assert hasattr(formatter, "format_book")
        assert callable(formatter.format_book)

    def test_format_code_method_exists(self) -> None:
        """format_code method should exist."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        assert hasattr(formatter, "format_code")
        assert callable(formatter.format_code)

    def test_format_graph_method_exists(self) -> None:
        """format_graph method should exist."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        assert hasattr(formatter, "format_graph")
        assert callable(formatter.format_graph)


# =============================================================================
# Book Citation Format Tests (AC-KB5.6)
# =============================================================================


class TestBookCitationFormat:
    """Tests for book citation formatting.
    
    Template: [^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.
    """

    def test_book_citation_has_marker(self) -> None:
        """Book citation should start with [^N]:."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Fowler, Martin",
            title="Patterns of Enterprise Application Architecture",
            city="Boston",
            publisher="Addison-Wesley",
            year=2002,
            pages="102-110",
        )
        assert result.startswith("[^1]:")

    def test_book_citation_has_author(self) -> None:
        """Book citation should include author name."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        )
        assert "Fowler, Martin" in result

    def test_book_citation_has_italicized_title(self) -> None:
        """Book citation should have italicized title with asterisks."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        )
        assert "*PEAA*" in result

    def test_book_citation_has_year(self) -> None:
        """Book citation should include publication year."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Test Author",
            title="Test Book",
            year=2023,
        )
        assert "2023" in result

    def test_book_citation_has_city_publisher(self) -> None:
        """Book citation should include city and publisher when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Gamma, Erich",
            title="Design Patterns",
            city="Boston",
            publisher="Addison-Wesley",
            year=1994,
        )
        assert "Boston" in result
        assert "Addison-Wesley" in result

    def test_book_citation_has_pages(self) -> None:
        """Book citation should include page numbers when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Test Author",
            title="Test Book",
            year=2023,
            pages="45-67",
        )
        assert "45-67" in result

    def test_book_citation_ends_with_period(self) -> None:
        """Book citation should end with a period."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Test Author",
            title="Test Book",
            year=2023,
        )
        assert result.rstrip().endswith(".")


# =============================================================================
# Code Citation Format Tests (AC-KB5.6)
# =============================================================================


class TestCodeCitationFormat:
    """Tests for code citation formatting.
    
    Template: [^N]: `repo/path/file.py`, commit `hash`, lines X-Y.
    """

    def test_code_citation_has_marker(self) -> None:
        """Code citation should start with [^N]:."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_code(
            marker=2,
            file_path="src/agents/pipeline.py",
            lines="42-56",
        )
        assert result.startswith("[^2]:")

    def test_code_citation_has_backtick_file_path(self) -> None:
        """Code citation should have file path in backticks."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_code(
            marker=1,
            file_path="src/agents/pipeline.py",
        )
        assert "`src/agents/pipeline.py`" in result

    def test_code_citation_has_commit_hash(self) -> None:
        """Code citation should include commit hash when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_code(
            marker=1,
            file_path="src/main.py",
            commit_hash="abc123",
        )
        assert "abc123" in result
        assert "commit" in result.lower()

    def test_code_citation_has_line_numbers(self) -> None:
        """Code citation should include line numbers when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_code(
            marker=1,
            file_path="src/main.py",
            lines="100-150",
        )
        assert "100-150" in result
        assert "line" in result.lower()

    def test_code_citation_has_repo_name(self) -> None:
        """Code citation should include repo name when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_code(
            marker=1,
            file_path="src/main.py",
            repo="ai-agents",
        )
        assert "ai-agents" in result


# =============================================================================
# Schema Citation Format Tests
# =============================================================================


class TestSchemaCitationFormat:
    """Tests for schema citation formatting.
    
    Template: [^N]: `repo/schemas/file.json`, version X.Y.Z.
    """

    def test_schema_citation_has_marker(self) -> None:
        """Schema citation should start with [^N]:."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_schema(
            marker=3,
            file_path="schemas/agent_state.json",
        )
        assert result.startswith("[^3]:")

    def test_schema_citation_has_backtick_path(self) -> None:
        """Schema citation should have path in backticks."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_schema(
            marker=1,
            file_path="schemas/agent_state.json",
        )
        assert "`schemas/agent_state.json`" in result

    def test_schema_citation_has_version(self) -> None:
        """Schema citation should include version when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_schema(
            marker=1,
            file_path="schemas/config.json",
            version="1.2.3",
        )
        assert "1.2.3" in result
        assert "version" in result.lower()


# =============================================================================
# Graph Citation Format Tests (AC-KB5.6)
# =============================================================================


class TestGraphCitationFormat:
    """Tests for graph/Neo4j citation formatting."""

    def test_graph_citation_has_marker(self) -> None:
        """Graph citation should start with [^N]:."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_graph(
            marker=4,
            node_type="Concept",
            node_id="rate_limiter",
        )
        assert result.startswith("[^4]:")

    def test_graph_citation_has_node_type(self) -> None:
        """Graph citation should include node type."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_graph(
            marker=1,
            node_type="Pattern",
            node_id="repository",
        )
        assert "Pattern" in result

    def test_graph_citation_has_node_id(self) -> None:
        """Graph citation should include node identifier."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_graph(
            marker=1,
            node_type="Concept",
            node_id="dependency_injection",
        )
        assert "dependency_injection" in result

    def test_graph_citation_has_relationship(self) -> None:
        """Graph citation should include relationship when provided."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_graph(
            marker=1,
            node_type="Pattern",
            node_id="repository",
            relationship="IMPLEMENTS",
            related_node="data_access",
        )
        assert "IMPLEMENTS" in result or "implements" in result.lower()


# =============================================================================
# Internal Doc Citation Format Tests
# =============================================================================


class TestInternalDocCitationFormat:
    """Tests for internal documentation citation formatting.
    
    Template: [^N]: service, *Document* (Date), §Section.
    """

    def test_internal_doc_citation_has_marker(self) -> None:
        """Internal doc citation should start with [^N]:."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_internal_doc(
            marker=5,
            service="ai-agents",
            document="ARCHITECTURE.md",
            section="Pipeline Design",
        )
        assert result.startswith("[^5]:")

    def test_internal_doc_citation_has_service(self) -> None:
        """Internal doc citation should include service name."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_internal_doc(
            marker=1,
            service="inference-service",
            document="README.md",
        )
        assert "inference-service" in result

    def test_internal_doc_citation_has_italicized_document(self) -> None:
        """Internal doc citation should have italicized document name."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_internal_doc(
            marker=1,
            service="ai-agents",
            document="ARCHITECTURE.md",
        )
        assert "*ARCHITECTURE.md*" in result

    def test_internal_doc_citation_has_section(self) -> None:
        """Internal doc citation should include section with § symbol."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_internal_doc(
            marker=1,
            service="ai-agents",
            document="README.md",
            section="Installation",
        )
        assert "§" in result or "section" in result.lower()
        assert "Installation" in result


# =============================================================================
# Format Footnotes Section Tests
# =============================================================================


class TestFormatFootnotesSection:
    """Tests for formatting multiple citations as footnotes section."""

    def test_format_footnotes_returns_string(self) -> None:
        """format_footnotes should return a string."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_footnotes([
            {"type": "book", "marker": 1, "author": "Test", "title": "Book", "year": 2023},
        ])
        assert isinstance(result, str)

    def test_format_footnotes_multiple_citations(self) -> None:
        """format_footnotes should handle multiple citations."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_footnotes([
            {"type": "book", "marker": 1, "author": "Author A", "title": "Book A", "year": 2020},
            {"type": "code", "marker": 2, "file_path": "src/main.py"},
            {"type": "graph", "marker": 3, "node_type": "Concept", "node_id": "test"},
        ])
        assert "[^1]:" in result
        assert "[^2]:" in result
        assert "[^3]:" in result

    def test_format_footnotes_newline_separated(self) -> None:
        """Multiple footnotes should be newline-separated."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_footnotes([
            {"type": "book", "marker": 1, "author": "A", "title": "B", "year": 2020},
            {"type": "book", "marker": 2, "author": "C", "title": "D", "year": 2021},
        ])
        assert "\n" in result

    def test_format_footnotes_empty_list(self) -> None:
        """Empty list should return empty string."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_footnotes([])
        assert result == ""


# =============================================================================
# Edge Cases
# =============================================================================


class TestChicagoFormatterEdgeCases:
    """Edge case tests for ChicagoCitationFormatter."""

    def test_special_characters_in_title(self) -> None:
        """Titles with special characters should be handled."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="O'Brien, John",
            title="C++ & Python: A Guide",
            year=2023,
        )
        assert "C++ & Python" in result

    def test_unicode_in_author(self) -> None:
        """Authors with unicode characters should be handled."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        result = formatter.format_book(
            marker=1,
            author="Müller, Hans",
            title="Test Book",
            year=2023,
        )
        assert "Müller" in result

    def test_missing_optional_fields(self) -> None:
        """Optional fields should not cause errors when missing."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        # Book without city, publisher, pages
        result = formatter.format_book(
            marker=1,
            author="Test Author",
            title="Test Title",
            year=2023,
        )
        assert "[^1]:" in result
        assert "Test Author" in result

    def test_auto_detect_citation_type(self) -> None:
        """format() should auto-detect citation type from data."""
        from src.citations.chicago_formatter import ChicagoCitationFormatter

        formatter = ChicagoCitationFormatter()
        # Book-like data
        result = formatter.format({
            "type": "book",
            "marker": 1,
            "author": "Test",
            "title": "Book",
            "year": 2023,
        })
        assert "[^1]:" in result
        assert "*Book*" in result  # Italicized title indicates book format
