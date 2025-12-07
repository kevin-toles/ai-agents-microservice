"""Unit tests for Chicago citation formatter.

TDD Phase: RED → GREEN
Pattern: Citation formatting with Chicago style
Source: GRAPH_RAG_POC_PLAN WBS 5.10-5.11

Chicago Manual of Style 17th Edition format for book citations:
Author Last, First. *Book Title*. Place: Publisher, Year.

For chapters in edited books:
Author Last, First. "Chapter Title." In *Book Title*, edited by Editor Name, pages. Place: Publisher, Year.
"""

import pytest

from src.formatters.chicago import (
    ChicagoFormatter,
    ChicagoCitation,
    format_citation,
    format_footnote,
    format_bibliography_entry,
)


class TestChicagoCitation:
    """Tests for ChicagoCitation data model."""
    
    def test_create_basic_citation(self) -> None:
        """Test creating a basic citation."""
        citation = ChicagoCitation(
            author="Ousterhout, John",
            title="A Philosophy of Software Design",
            chapter_title="The Nature of Complexity",
            chapter_number=2,
            pages="7-18",
            tier=1,
        )
        
        assert citation.author == "Ousterhout, John"
        assert citation.title == "A Philosophy of Software Design"
        assert citation.chapter_number == 2
        assert citation.tier == 1
    
    def test_citation_requires_title(self) -> None:
        """Test that title is required."""
        with pytest.raises(ValueError):
            ChicagoCitation(
                author="Test",
                title="",  # Empty title should fail
                chapter_number=1,
                tier=1,
            )
    
    def test_citation_tier_validation(self) -> None:
        """Test that tier must be 1, 2, or 3."""
        with pytest.raises(ValueError):
            ChicagoCitation(
                author="Test",
                title="Test Book",
                chapter_number=1,
                tier=5,  # Invalid tier
            )
    
    def test_citation_optional_fields(self) -> None:
        """Test that optional fields default properly."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_number=1,
            tier=2,
        )
        
        assert citation.chapter_title is None
        assert citation.pages is None
        assert citation.publisher is None
        assert citation.year is None


class TestChicagoFormatter:
    """Tests for ChicagoFormatter class."""
    
    @pytest.fixture
    def formatter(self) -> ChicagoFormatter:
        """Create formatter instance."""
        return ChicagoFormatter()
    
    @pytest.fixture
    def sample_citation(self) -> ChicagoCitation:
        """Create sample citation."""
        return ChicagoCitation(
            author="Newman, Sam",
            title="Building Microservices",
            chapter_title="Decomposing the Monolith",
            chapter_number=4,
            pages="89-112",
            publisher="O'Reilly Media",
            year=2021,
            tier=2,
        )
    
    def test_format_footnote_basic(
        self,
        formatter: ChicagoFormatter,
        sample_citation: ChicagoCitation,
    ) -> None:
        """Test basic footnote formatting."""
        result = formatter.format_footnote(sample_citation, footnote_number=1)
        
        # Chicago footnote format: Author, *Book Title*, chapter info.
        assert "[^1]:" in result
        assert "Newman, Sam" in result
        assert "*Building Microservices*" in result
        assert "Ch. 4" in result or "Chapter 4" in result
    
    def test_format_footnote_includes_pages(
        self,
        formatter: ChicagoFormatter,
        sample_citation: ChicagoCitation,
    ) -> None:
        """Test that footnote includes page range."""
        result = formatter.format_footnote(sample_citation, footnote_number=1)
        
        assert "89-112" in result or "pp. 89-112" in result
    
    def test_format_footnote_sequential_numbers(
        self,
        formatter: ChicagoFormatter,
        sample_citation: ChicagoCitation,
    ) -> None:
        """Test sequential footnote numbers."""
        result1 = formatter.format_footnote(sample_citation, footnote_number=1)
        result2 = formatter.format_footnote(sample_citation, footnote_number=2)
        result3 = formatter.format_footnote(sample_citation, footnote_number=3)
        
        assert "[^1]:" in result1
        assert "[^2]:" in result2
        assert "[^3]:" in result3
    
    def test_format_bibliography_entry(
        self,
        formatter: ChicagoFormatter,
        sample_citation: ChicagoCitation,
    ) -> None:
        """Test bibliography entry formatting."""
        result = formatter.format_bibliography_entry(sample_citation)
        
        # Chicago bibliography: Author Last, First. *Title*. Place: Publisher, Year.
        assert "Newman, Sam" in result
        assert "*Building Microservices*" in result
        assert "O'Reilly Media" in result
        assert "2021" in result
    
    def test_format_inline_reference(
        self,
        formatter: ChicagoFormatter,
        sample_citation: ChicagoCitation,
    ) -> None:
        """Test inline reference marker."""
        result = formatter.format_inline_reference(footnote_number=5)
        
        assert result == "[^5]"
    
    def test_format_multiple_citations(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test formatting multiple citations as a list."""
        citations = [
            ChicagoCitation(
                author="Ousterhout, John",
                title="A Philosophy of Software Design",
                chapter_number=2,
                tier=1,
            ),
            ChicagoCitation(
                author="Newman, Sam",
                title="Building Microservices",
                chapter_number=4,
                tier=2,
            ),
        ]
        
        result = formatter.format_citations(citations)
        
        assert "[^1]:" in result
        assert "[^2]:" in result
        assert "Ousterhout" in result
        assert "Newman" in result


class TestChicagoFormatterEdgeCases:
    """Tests for edge cases in Chicago formatting."""
    
    @pytest.fixture
    def formatter(self) -> ChicagoFormatter:
        """Create formatter instance."""
        return ChicagoFormatter()
    
    def test_format_citation_without_author(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test formatting when author is None."""
        citation = ChicagoCitation(
            author=None,
            title="Anonymous Work",
            chapter_number=1,
            tier=1,
        )
        
        result = formatter.format_footnote(citation, footnote_number=1)
        
        # Should handle missing author gracefully
        assert "*Anonymous Work*" in result
        assert "[^1]:" in result
    
    def test_format_citation_without_pages(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test formatting when pages are not specified."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_number=3,
            tier=2,
        )
        
        result = formatter.format_footnote(citation, footnote_number=1)
        
        # Should format without page info
        assert "pp." not in result or "pages" not in result.lower()
    
    def test_format_citation_with_chapter_title(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test formatting includes chapter title when present."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_title="Important Chapter",
            chapter_number=5,
            tier=1,
        )
        
        result = formatter.format_footnote(citation, footnote_number=1)
        
        # Chapter titles should be in quotes per Chicago style
        assert '"Important Chapter"' in result or "Important Chapter" in result


class TestModuleFunctions:
    """Tests for module-level convenience functions."""
    
    def test_format_citation_function(self) -> None:
        """Test format_citation convenience function."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_number=1,
            tier=1,
        )
        
        result = format_citation(citation, footnote_number=1)
        
        assert "[^1]:" in result
        assert "Test Author" in result
    
    def test_format_footnote_function(self) -> None:
        """Test format_footnote convenience function."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_number=1,
            tier=1,
        )
        
        result = format_footnote(citation, footnote_number=1)
        
        assert "[^1]:" in result
    
    def test_format_bibliography_entry_function(self) -> None:
        """Test format_bibliography_entry convenience function."""
        citation = ChicagoCitation(
            author="Test Author",
            title="Test Book",
            chapter_number=1,
            tier=1,
        )
        
        result = format_bibliography_entry(citation)
        
        assert "*Test Book*" in result


class TestTierOrdering:
    """Tests for tier-based citation ordering."""
    
    @pytest.fixture
    def formatter(self) -> ChicagoFormatter:
        """Create formatter instance."""
        return ChicagoFormatter()
    
    def test_citations_sorted_by_tier(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test that format_citations sorts by tier (1, 2, 3)."""
        citations = [
            ChicagoCitation(author="Tier 3", title="Book 3", chapter_number=1, tier=3),
            ChicagoCitation(author="Tier 1", title="Book 1", chapter_number=1, tier=1),
            ChicagoCitation(author="Tier 2", title="Book 2", chapter_number=1, tier=2),
        ]
        
        result = formatter.format_citations(citations)
        
        # Tier 1 should appear before Tier 2, Tier 2 before Tier 3
        tier1_pos = result.find("Tier 1")
        tier2_pos = result.find("Tier 2")
        tier3_pos = result.find("Tier 3")
        
        assert tier1_pos < tier2_pos < tier3_pos
    
    def test_format_by_tier_groups(
        self,
        formatter: ChicagoFormatter,
    ) -> None:
        """Test formatting with tier group headers."""
        citations = [
            ChicagoCitation(author="Author 1", title="Architecture Book", chapter_number=1, tier=1),
            ChicagoCitation(author="Author 2", title="Implementation Book", chapter_number=1, tier=2),
        ]
        
        result = formatter.format_citations_by_tier(citations)
        
        # Should have tier headers
        assert "Tier 1" in result or "Architecture" in result
        assert "Tier 2" in result or "Implementation" in result
