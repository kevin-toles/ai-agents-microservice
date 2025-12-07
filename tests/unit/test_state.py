"""Unit tests for state models.

TDD Phase: RED - These tests define expected behavior.
Pattern: Pydantic model testing
"""

import pytest
from pydantic import ValidationError

from src.agents.cross_reference.state import (
    SourceChapter,
    TraversalConfig,
    GraphNode,
    TraversalPath,
    ChapterMatch,
    Citation,
    TierCoverage,
    CrossReferenceResult,
    CrossReferenceState,
    RelationshipType,
)


class TestRelationshipType:
    """Tests for RelationshipType enum."""
    
    def test_relationship_type_values(self) -> None:
        """Test that all relationship types are defined."""
        assert RelationshipType.PARALLEL.value == "parallel"
        assert RelationshipType.PERPENDICULAR.value == "perpendicular"
        assert RelationshipType.SKIP_TIER.value == "skip_tier"
    
    def test_relationship_type_from_string(self) -> None:
        """Test creating RelationshipType from string."""
        assert RelationshipType("parallel") == RelationshipType.PARALLEL
        assert RelationshipType("perpendicular") == RelationshipType.PERPENDICULAR
        assert RelationshipType("skip_tier") == RelationshipType.SKIP_TIER


class TestSourceChapter:
    """Tests for SourceChapter model."""
    
    def test_source_chapter_required_fields(self) -> None:
        """Test that SourceChapter requires essential fields."""
        chapter = SourceChapter(
            book="Test Book",
            chapter=1,
            title="Test Chapter",
            tier=1,
        )
        
        assert chapter.book == "Test Book"
        assert chapter.chapter == 1
        assert chapter.title == "Test Chapter"
        assert chapter.tier == 1
    
    def test_source_chapter_optional_fields(self) -> None:
        """Test that SourceChapter has sensible optional defaults."""
        chapter = SourceChapter(
            book="Test Book",
            chapter=1,
            title="Test Chapter",
            tier=1,
        )
        
        assert chapter.keywords == []
        assert chapter.concepts == []
        assert chapter.content is None
    
    def test_source_chapter_with_metadata(self) -> None:
        """Test SourceChapter with full metadata."""
        chapter = SourceChapter(
            book="A Philosophy of Software Design",
            chapter=2,
            title="The Nature of Complexity",
            tier=1,
            content="Complexity is what makes software hard to change...",
            keywords=["complexity", "dependencies", "obscurity"],
            concepts=["complexity symptoms", "causes of complexity"],
        )
        
        assert len(chapter.keywords) == 3
        assert len(chapter.concepts) == 2
        assert chapter.tier == 1
    
    def test_source_chapter_invalid_chapter_number(self) -> None:
        """Test that chapter number must be positive."""
        with pytest.raises(ValidationError):
            SourceChapter(
                book="Test",
                chapter=0,  # Invalid - ge=1
                title="Test",
                tier=1,
            )
    
    def test_source_chapter_invalid_tier(self) -> None:
        """Test that tier must be 1-3."""
        with pytest.raises(ValidationError):
            SourceChapter(
                book="Test",
                chapter=1,
                title="Test",
                tier=4,  # Invalid - le=3
            )


class TestTraversalConfig:
    """Tests for TraversalConfig model."""
    
    def test_traversal_config_defaults(self) -> None:
        """Test TraversalConfig default values."""
        config = TraversalConfig()
        
        assert config.max_hops == 3
        assert config.allow_cycles is True
        assert config.min_similarity == 0.7
        assert config.max_results_per_tier == 10
    
    def test_traversal_config_custom_values(self) -> None:
        """Test TraversalConfig with custom values."""
        config = TraversalConfig(
            max_hops=5,
            relationship_types=[RelationshipType.PARALLEL],
            allow_cycles=False,
            min_similarity=0.8,
        )
        
        assert config.max_hops == 5
        assert config.relationship_types == [RelationshipType.PARALLEL]
        assert config.allow_cycles is False
        assert config.min_similarity == 0.8
    
    def test_traversal_config_max_hops_bounds(self) -> None:
        """Test that max_hops has reasonable bounds."""
        # Too low
        with pytest.raises(ValidationError):
            TraversalConfig(max_hops=0)
        
        # Too high (prevent runaway - le=10)
        with pytest.raises(ValidationError):
            TraversalConfig(max_hops=100)
    
    def test_traversal_config_min_similarity_bounds(self) -> None:
        """Test that min_similarity is bounded 0.0-1.0."""
        with pytest.raises(ValidationError):
            TraversalConfig(min_similarity=1.5)
        
        with pytest.raises(ValidationError):
            TraversalConfig(min_similarity=-0.1)


class TestGraphNode:
    """Tests for GraphNode model."""
    
    def test_graph_node_creation(self) -> None:
        """Test basic GraphNode creation."""
        node = GraphNode(
            book="Building Microservices",
            chapter=3,
            tier=2,
            title="How to Model Services",
        )
        
        assert node.book == "Building Microservices"
        assert node.chapter == 3
        assert node.tier == 2
        assert node.title == "How to Model Services"
    
    def test_graph_node_with_relationship(self) -> None:
        """Test GraphNode with relationship info."""
        node = GraphNode(
            book="Test",
            chapter=1,
            tier=1,
            relationship=RelationshipType.PERPENDICULAR,
            similarity_score=0.85,
        )
        
        assert node.relationship == RelationshipType.PERPENDICULAR
        assert node.similarity_score == 0.85


class TestChapterMatch:
    """Tests for ChapterMatch model."""
    
    def test_chapter_match_creation(self) -> None:
        """Test basic ChapterMatch creation."""
        match = ChapterMatch(
            book="Building Microservices",
            chapter=3,
            title="How to Model Services",
            tier=2,
            similarity=0.85,
        )
        
        assert match.book == "Building Microservices"
        assert match.chapter == 3
        assert match.tier == 2
        assert match.similarity == 0.85
    
    def test_chapter_match_similarity_bounds(self) -> None:
        """Test that similarity is bounded 0.0-1.0."""
        with pytest.raises(ValidationError):
            ChapterMatch(
                book="Test",
                chapter=1,
                title="Test",
                tier=1,
                similarity=1.5,  # Invalid
            )


class TestTraversalPath:
    """Tests for TraversalPath model."""
    
    def test_traversal_path_creation(self) -> None:
        """Test basic TraversalPath creation."""
        path = TraversalPath(
            nodes=[
                GraphNode(book="A", chapter=1, tier=1),
                GraphNode(book="B", chapter=2, tier=2),
            ],
            total_similarity=0.9,
            path_type="linear",
        )
        
        assert len(path.nodes) == 2
        assert path.total_similarity == 0.9
        assert path.path_type == "linear"


class TestCitation:
    """Tests for Citation model."""
    
    def test_citation_creation(self) -> None:
        """Test basic Citation creation."""
        citation = Citation(
            book="A Philosophy of Software Design",
            chapter=2,
            chapter_title="The Nature of Complexity",
            tier=1,
            pages="15-28",
        )
        
        assert citation.book == "A Philosophy of Software Design"
        assert citation.chapter == 2
        assert citation.pages == "15-28"
    
    def test_citation_to_chicago_format(self) -> None:
        """Test Chicago-style citation formatting."""
        citation = Citation(
            author="John Ousterhout",
            book="A Philosophy of Software Design",
            chapter=2,
            chapter_title="The Nature of Complexity",
            tier=1,
            pages="15-28",
        )
        
        formatted = citation.to_chicago_format(1)
        
        assert "[^1]:" in formatted
        assert "*A Philosophy of Software Design*" in formatted
        assert "Ch. 2" in formatted


class TestTierCoverage:
    """Tests for TierCoverage model."""
    
    def test_tier_coverage_creation(self) -> None:
        """Test TierCoverage creation."""
        coverage = TierCoverage(
            tier=1,
            tier_name="Architecture Spine",
            books_referenced=3,
            chapters_referenced=5,
            has_coverage=True,
        )
        
        assert coverage.tier == 1
        assert coverage.tier_name == "Architecture Spine"
        assert coverage.books_referenced == 3
        assert coverage.has_coverage is True


class TestCrossReferenceResult:
    """Tests for CrossReferenceResult model."""
    
    def test_result_creation(self) -> None:
        """Test CrossReferenceResult creation."""
        result = CrossReferenceResult(
            annotation="This chapter relates to complexity management...",
            citations=[],
            tier_coverage=[],
            model_used="gpt-4",
        )
        
        assert "complexity" in result.annotation.lower()
        assert result.model_used == "gpt-4"


class TestCrossReferenceState:
    """Tests for CrossReferenceState model."""
    
    def test_state_creation_minimal(self) -> None:
        """Test minimal state creation."""
        source = SourceChapter(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
        )
        
        state = CrossReferenceState(source=source)
        
        assert state.source is source
        assert state.current_node == ""
        assert state.errors == []
    
    def test_state_default_lists(self) -> None:
        """Test that state lists default to empty."""
        source = SourceChapter(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
        )
        
        state = CrossReferenceState(source=source)
        
        assert state.analyzed_concepts == []
        assert state.taxonomy_matches == []
        assert state.traversal_paths == []
        assert state.retrieved_chapters == []
    
    def test_state_immutability_warning(self) -> None:
        """Test that state fields follow immutability patterns.
        
        For LangGraph, state should be updated via reducer functions,
        not direct mutation.
        """
        source = SourceChapter(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
        )
        
        state = CrossReferenceState(source=source)
        
        # Verify we can create a copy with updates (not mutate in place)
        new_state = state.model_copy(update={"current_node": "analyze_source"})
        
        assert state.current_node == ""  # Original unchanged
        assert new_state.current_node == "analyze_source"
