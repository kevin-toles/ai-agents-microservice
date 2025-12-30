"""Tests for citation and provenance schemas.

TDD RED Phase: Tests written before implementation.

Acceptance Criteria Coverage:
- AC-4.1: SourceMetadata for provenance tracking
- AC-4.2: Citation with Chicago-style formatting
- AC-4.3: CitedContent with embedded [^N] markers
- AC-4.5: All schemas have JSON schema export

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

import pytest
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# AC-4.1: SourceMetadata Tests
# =============================================================================

class TestSourceMetadata:
    """Tests for SourceMetadata provenance tracking."""
    
    def test_source_metadata_book_type(self) -> None:
        """SourceMetadata captures book provenance fields."""
        from src.schemas.citations import SourceMetadata
        
        metadata = SourceMetadata(
            source_type="book",
            author="Fowler, Martin",
            title="Patterns of Enterprise Application Architecture",
            publisher="Addison-Wesley",
            year=2002,
            pages="322-327",
        )
        
        assert metadata.source_type == "book"
        assert metadata.author == "Fowler, Martin"
        assert metadata.title == "Patterns of Enterprise Application Architecture"
        assert metadata.publisher == "Addison-Wesley"
        assert metadata.year == 2002
        assert metadata.pages == "322-327"
    
    def test_source_metadata_code_type(self) -> None:
        """SourceMetadata captures code provenance fields."""
        from src.schemas.citations import SourceMetadata
        
        metadata = SourceMetadata(
            source_type="code",
            repo="code-reference-engine",
            file_path="backend/ddd/repository.py",
            line_range="12-45",
            commit_hash="a1b2c3d",
        )
        
        assert metadata.source_type == "code"
        assert metadata.repo == "code-reference-engine"
        assert metadata.file_path == "backend/ddd/repository.py"
        assert metadata.line_range == "12-45"
        assert metadata.commit_hash == "a1b2c3d"
    
    def test_source_metadata_schema_type(self) -> None:
        """SourceMetadata captures schema provenance fields."""
        from src.schemas.citations import SourceMetadata
        
        metadata = SourceMetadata(
            source_type="schema",
            repo="llm-gateway",
            file_path="schemas/completion.json",
            version="1.2.0",
        )
        
        assert metadata.source_type == "schema"
        assert metadata.repo == "llm-gateway"
        assert metadata.file_path == "schemas/completion.json"
        assert metadata.version == "1.2.0"
    
    def test_source_metadata_internal_doc_type(self) -> None:
        """SourceMetadata captures internal document provenance."""
        from src.schemas.citations import SourceMetadata
        
        metadata = SourceMetadata(
            source_type="internal_doc",
            service="inference-service",
            title="ARCHITECTURE.md",
            section="Model Library",
            date="2025-12-29",
        )
        
        assert metadata.source_type == "internal_doc"
        assert metadata.service == "inference-service"
        assert metadata.title == "ARCHITECTURE.md"
        assert metadata.section == "Model Library"
        assert metadata.date == "2025-12-29"
    
    def test_source_metadata_with_similarity_score(self) -> None:
        """SourceMetadata includes retrieval similarity score."""
        from src.schemas.citations import SourceMetadata
        
        metadata = SourceMetadata(
            source_type="book",
            author="Evans, Eric",
            title="Domain-Driven Design",
            similarity_score=0.89,
        )
        
        assert metadata.similarity_score == 0.89
    
    def test_source_metadata_optional_fields(self) -> None:
        """SourceMetadata allows optional fields."""
        from src.schemas.citations import SourceMetadata
        
        # Minimal book metadata
        metadata = SourceMetadata(
            source_type="book",
            title="Test Book",
        )
        
        assert metadata.source_type == "book"
        assert metadata.title == "Test Book"
        assert metadata.author is None
        assert metadata.publisher is None
        assert metadata.year is None
    
    def test_source_metadata_source_type_validation(self) -> None:
        """SourceMetadata validates source_type enum."""
        from src.schemas.citations import SourceMetadata, SourceType
        from pydantic import ValidationError
        
        # Valid types
        for source_type in ["book", "code", "schema", "internal_doc"]:
            metadata = SourceMetadata(source_type=source_type, title="Test")
            assert metadata.source_type == source_type
        
        # Invalid type raises ValidationError
        with pytest.raises(ValidationError):
            SourceMetadata(source_type="invalid_type", title="Test")
    
    def test_source_metadata_json_schema_export(self) -> None:
        """SourceMetadata exports valid JSON Schema (AC-4.5)."""
        from src.schemas.citations import SourceMetadata
        
        schema = SourceMetadata.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "source_type" in schema["properties"]
        assert "title" in schema["properties"]


# =============================================================================
# AC-4.2: Citation Tests
# =============================================================================

class TestCitation:
    """Tests for Citation model with Chicago-style formatting."""
    
    def test_citation_creation(self) -> None:
        """Citation captures marker and source metadata."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            author="Fowler, Martin",
            title="Patterns of Enterprise Application Architecture",
            publisher="Addison-Wesley",
            year=2002,
            pages="322-327",
        )
        
        citation = Citation(
            marker=1,
            source=source,
            context="Repository pattern implementation",
        )
        
        assert citation.marker == 1
        assert citation.source.author == "Fowler, Martin"
        assert citation.context == "Repository pattern implementation"
    
    def test_citation_chicago_format_book(self) -> None:
        """Citation.chicago_format() returns Chicago-style for books."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            author="Fowler, Martin",
            title="Patterns of Enterprise Application Architecture",
            publisher="Addison-Wesley",
            publication_city="Boston",
            year=2002,
            pages="322-327",
        )
        
        citation = Citation(marker=1, source=source)
        formatted = citation.chicago_format()
        
        # Chicago format: LastName, FirstName, *Title* (City: Publisher, Year), pages.
        assert "Fowler, Martin" in formatted
        assert "Patterns of Enterprise Application Architecture" in formatted
        assert "Boston" in formatted
        assert "Addison-Wesley" in formatted
        assert "2002" in formatted
        assert "322-327" in formatted
    
    def test_citation_chicago_format_code(self) -> None:
        """Citation.chicago_format() returns Chicago-style for code."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="code",
            repo="code-reference-engine",
            file_path="backend/ddd/repository.py",
            line_range="12-45",
            commit_hash="a1b2c3d",
        )
        
        citation = Citation(marker=2, source=source)
        formatted = citation.chicago_format()
        
        # Code format: `repo/path/file.py`, commit `hash`, lines X-Y.
        assert "code-reference-engine" in formatted
        assert "backend/ddd/repository.py" in formatted
        assert "a1b2c3d" in formatted
        assert "12-45" in formatted
    
    def test_citation_chicago_format_schema(self) -> None:
        """Citation.chicago_format() returns Chicago-style for schemas."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="schema",
            repo="llm-gateway",
            file_path="schemas/completion.json",
            version="1.2.0",
        )
        
        citation = Citation(marker=3, source=source)
        formatted = citation.chicago_format()
        
        # Schema format: `repo/schemas/file.json`, version X.Y.Z.
        assert "llm-gateway" in formatted
        assert "schemas/completion.json" in formatted
        assert "1.2.0" in formatted
    
    def test_citation_chicago_format_internal_doc(self) -> None:
        """Citation.chicago_format() returns Chicago-style for internal docs."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="internal_doc",
            service="inference-service",
            title="ARCHITECTURE.md",
            section="Model Library",
            date="2025-12-29",
        )
        
        citation = Citation(marker=4, source=source)
        formatted = citation.chicago_format()
        
        # Internal doc format: service, *Document* (Date), §Section.
        assert "inference-service" in formatted
        assert "ARCHITECTURE.md" in formatted
        assert "2025-12-29" in formatted
        assert "Model Library" in formatted
    
    def test_citation_marker_format(self) -> None:
        """Citation.marker_format returns [^N] format."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(source_type="book", title="Test")
        citation = Citation(marker=5, source=source)
        
        assert citation.marker_format == "[^5]"
    
    def test_citation_footnote_format(self) -> None:
        """Citation.footnote_format returns [^N]: formatted citation."""
        from src.schemas.citations import Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            author="Test Author",
            title="Test Book",
            year=2025,
        )
        citation = Citation(marker=1, source=source)
        
        footnote = citation.footnote_format
        
        assert footnote.startswith("[^1]: ")
        assert "Test Author" in footnote
        assert "Test Book" in footnote
    
    def test_citation_marker_must_be_positive(self) -> None:
        """Citation marker must be positive integer."""
        from src.schemas.citations import Citation, SourceMetadata
        from pydantic import ValidationError
        
        source = SourceMetadata(source_type="book", title="Test")
        
        with pytest.raises(ValidationError):
            Citation(marker=0, source=source)
        
        with pytest.raises(ValidationError):
            Citation(marker=-1, source=source)
    
    def test_citation_json_schema_export(self) -> None:
        """Citation exports valid JSON Schema (AC-4.5)."""
        from src.schemas.citations import Citation
        
        schema = Citation.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "marker" in schema["properties"]
        assert "source" in schema["properties"]


# =============================================================================
# AC-4.3: CitedContent Tests
# =============================================================================

class TestCitedContent:
    """Tests for CitedContent with embedded [^N] markers."""
    
    def test_cited_content_creation(self) -> None:
        """CitedContent captures content and citations."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        source = SourceMetadata(source_type="book", author="Test", title="Test Book")
        citation = Citation(marker=1, source=source)
        
        content = CitedContent(
            text="The Repository pattern[^1] provides abstraction.",
            citations=[citation],
        )
        
        assert "[^1]" in content.text
        assert len(content.citations) == 1
        assert content.citations[0].marker == 1
    
    def test_cited_content_multiple_citations(self) -> None:
        """CitedContent handles multiple citation markers."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        source1 = SourceMetadata(source_type="book", author="Fowler", title="PEAA")
        source2 = SourceMetadata(source_type="code", repo="test", file_path="test.py")
        
        citations = [
            Citation(marker=1, source=source1),
            Citation(marker=2, source=source2),
        ]
        
        content = CitedContent(
            text="The pattern[^1] is implemented here[^2].",
            citations=citations,
        )
        
        assert "[^1]" in content.text
        assert "[^2]" in content.text
        assert len(content.citations) == 2
    
    def test_cited_content_footnotes_list(self) -> None:
        """CitedContent.footnotes returns formatted footnote list."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        source1 = SourceMetadata(
            source_type="book",
            author="Fowler, Martin",
            title="PEAA",
            year=2002,
        )
        source2 = SourceMetadata(
            source_type="code",
            repo="test-repo",
            file_path="src/test.py",
            line_range="10-20",
        )
        
        citations = [
            Citation(marker=1, source=source1),
            Citation(marker=2, source=source2),
        ]
        
        content = CitedContent(
            text="Pattern[^1] and implementation[^2].",
            citations=citations,
        )
        
        footnotes = content.footnotes
        
        assert len(footnotes) == 2
        assert footnotes[0].startswith("[^1]:")
        assert footnotes[1].startswith("[^2]:")
    
    def test_cited_content_preserves_markers(self) -> None:
        """CitedContent preserves [^N] markers in text."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        original_text = "First point[^1]. Second point[^2]. Third point[^3]."
        
        citations = [
            Citation(marker=1, source=SourceMetadata(source_type="book", title="A")),
            Citation(marker=2, source=SourceMetadata(source_type="book", title="B")),
            Citation(marker=3, source=SourceMetadata(source_type="book", title="C")),
        ]
        
        content = CitedContent(text=original_text, citations=citations)
        
        # Markers should be preserved exactly
        assert content.text == original_text
        assert content.text.count("[^") == 3
    
    def test_cited_content_render_with_footnotes(self) -> None:
        """CitedContent.render() returns text with appended footnotes."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        source = SourceMetadata(
            source_type="book",
            author="Evans, Eric",
            title="Domain-Driven Design",
            year=2003,
        )
        
        content = CitedContent(
            text="The Aggregate pattern[^1] ensures consistency.",
            citations=[Citation(marker=1, source=source)],
        )
        
        rendered = content.render()
        
        # Should contain both text and footnotes
        assert "The Aggregate pattern[^1]" in rendered
        assert "[^1]:" in rendered
        assert "Evans, Eric" in rendered
    
    def test_cited_content_extract_markers(self) -> None:
        """CitedContent.extract_markers() returns list of marker numbers."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        content = CitedContent(
            text="Point A[^1], point B[^3], point C[^2].",
            citations=[
                Citation(marker=1, source=SourceMetadata(source_type="book", title="A")),
                Citation(marker=2, source=SourceMetadata(source_type="book", title="B")),
                Citation(marker=3, source=SourceMetadata(source_type="book", title="C")),
            ],
        )
        
        markers = content.extract_markers()
        
        # Should return markers in order they appear in text
        assert markers == [1, 3, 2]
    
    def test_cited_content_empty_citations(self) -> None:
        """CitedContent allows text without citations."""
        from src.schemas.citations import CitedContent
        
        content = CitedContent(
            text="Plain text without any citations.",
            citations=[],
        )
        
        assert content.text == "Plain text without any citations."
        assert content.citations == []
        assert content.footnotes == []
    
    def test_cited_content_get_citation_by_marker(self) -> None:
        """CitedContent.get_citation(marker) returns specific citation."""
        from src.schemas.citations import CitedContent, Citation, SourceMetadata
        
        citations = [
            Citation(marker=1, source=SourceMetadata(source_type="book", title="Book A")),
            Citation(marker=2, source=SourceMetadata(source_type="book", title="Book B")),
        ]
        
        content = CitedContent(text="Test[^1] and test[^2].", citations=citations)
        
        citation = content.get_citation(2)
        
        assert citation is not None
        assert citation.source.title == "Book B"
    
    def test_cited_content_get_citation_not_found(self) -> None:
        """CitedContent.get_citation() returns None for missing marker."""
        from src.schemas.citations import CitedContent
        
        content = CitedContent(text="No citations.", citations=[])
        
        assert content.get_citation(99) is None
    
    def test_cited_content_json_schema_export(self) -> None:
        """CitedContent exports valid JSON Schema (AC-4.5)."""
        from src.schemas.citations import CitedContent
        
        schema = CitedContent.model_json_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "text" in schema["properties"]
        assert "citations" in schema["properties"]


# =============================================================================
# Additional Citation Utility Tests
# =============================================================================

class TestCitationUtilities:
    """Tests for citation utility functions."""
    
    def test_create_citation_from_retrieval(self) -> None:
        """create_citation_from_retrieval builds Citation from search result."""
        from src.schemas.citations import create_citation_from_retrieval
        
        retrieval_result = {
            "chunk": "The Repository pattern provides...",
            "source_type": "book",
            "author": "Fowler, Martin",
            "title": "Patterns of Enterprise Application Architecture",
            "publisher": "Addison-Wesley",
            "year": 2002,
            "pages": "322-327",
            "similarity": 0.89,
        }
        
        citation = create_citation_from_retrieval(retrieval_result, marker=1)
        
        assert citation.marker == 1
        assert citation.source.author == "Fowler, Martin"
        assert citation.source.similarity_score == 0.89
        assert citation.context == "The Repository pattern provides..."
    
    def test_merge_citations(self) -> None:
        """merge_citations combines citations and renumbers markers."""
        from src.schemas.citations import (
            CitedContent, Citation, SourceMetadata, merge_citations
        )
        
        content1 = CitedContent(
            text="First[^1].",
            citations=[Citation(marker=1, source=SourceMetadata(source_type="book", title="A"))],
        )
        content2 = CitedContent(
            text="Second[^1].",
            citations=[Citation(marker=1, source=SourceMetadata(source_type="book", title="B"))],
        )
        
        merged = merge_citations([content1, content2])
        
        # Citations should be renumbered
        assert "[^1]" in merged.text
        assert "[^2]" in merged.text
        assert len(merged.citations) == 2
        assert merged.citations[0].source.title == "A"
        assert merged.citations[1].source.title == "B"
