"""Unit tests for Cross-Reference Agent workflow nodes.

TDD Phase: RED → GREEN
Pattern: LangGraph node function testing (dict returns)
Source: GRAPH_RAG_POC_PLAN WBS 5.4-5.5

LangGraph nodes return dicts that get merged into state.
Tests use dependency injection pattern for external clients.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from src.agents.cross_reference.state import (
    CrossReferenceState,
    SourceChapter,
    TraversalConfig,
    ChapterMatch,
    TraversalPath,
    GraphNode,
    RelationshipType,
)
from src.agents.cross_reference.nodes.analyze_source import (
    analyze_source,
    set_llm_client,
)
from src.agents.cross_reference.nodes.search_taxonomy import (
    search_taxonomy,
    set_neo4j_client,
)
from src.agents.cross_reference.nodes.traverse_graph import (
    traverse_graph,
    set_graph_client,
)
from src.agents.cross_reference.nodes.retrieve_content import (
    retrieve_content,
    set_content_client,
)
from src.agents.cross_reference.nodes.synthesize import (
    synthesize,
    set_synthesis_client,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_source() -> SourceChapter:
    """Create a sample source chapter."""
    return SourceChapter(
        book="A Philosophy of Software Design",
        chapter=2,
        title="The Nature of Complexity",
        tier=1,
        content="Complexity is anything related to the structure of a software system that makes it hard to understand and modify. Complexity manifests itself in three ways: change amplification, cognitive load, and unknown unknowns.",
        keywords=["complexity", "abstraction"],
        concepts=["complexity management"],
    )


@pytest.fixture
def sample_state(sample_source: SourceChapter) -> CrossReferenceState:
    """Create a sample initial state."""
    return CrossReferenceState(
        source=sample_source,
        config=TraversalConfig(max_hops=3),
        taxonomy_id="ai-ml",
    )


@pytest.fixture
def state_with_concepts(sample_state: CrossReferenceState) -> CrossReferenceState:
    """Create state after analyze_source."""
    return sample_state.model_copy(update={
        "analyzed_concepts": ["complexity", "modularity", "abstraction", "dependencies"],
        "current_node": "analyze_source",
    })


@pytest.fixture
def state_with_matches(state_with_concepts: CrossReferenceState) -> CrossReferenceState:
    """Create state after search_taxonomy."""
    matches = [
        ChapterMatch(
            book="Building Microservices",
            chapter=4,
            title="Decomposing the Monolith",
            tier=2,
            similarity=0.85,
            keywords=["modularity", "boundaries"],
            relevance_reason="Discusses module boundaries and decomposition",
        ),
        ChapterMatch(
            book="Architecture Patterns with Python",
            chapter=2,
            title="Repository Pattern",
            tier=2,
            similarity=0.78,
            keywords=["abstraction", "patterns"],
            relevance_reason="Covers abstraction patterns",
        ),
    ]
    return state_with_concepts.model_copy(update={
        "taxonomy_matches": matches,
        "current_node": "search_taxonomy",
    })


@pytest.fixture
def state_with_paths(state_with_matches: CrossReferenceState) -> CrossReferenceState:
    """Create state after traverse_graph."""
    paths = [
        TraversalPath(
            nodes=[
                GraphNode(book="A Philosophy of Software Design", chapter=2, tier=1),
                GraphNode(book="Building Microservices", chapter=4, tier=2,
                         relationship=RelationshipType.PERPENDICULAR, similarity_score=0.85),
            ],
            total_similarity=0.85,
            path_type="linear",
        ),
    ]
    return state_with_matches.model_copy(update={
        "traversal_paths": paths,
        "current_node": "traverse_graph",
    })


@pytest.fixture
def state_with_content(state_with_paths: CrossReferenceState) -> CrossReferenceState:
    """Create state after retrieve_content."""
    chapters = [
        ChapterMatch(
            book="Building Microservices",
            chapter=4,
            title="Decomposing the Monolith",
            tier=2,
            similarity=0.85,
            keywords=["modularity", "boundaries"],
            relevance_reason="Discusses module boundaries",
            content="When breaking apart a monolith, we need to identify seams...",
            page_range="89-112",
        ),
    ]
    return state_with_paths.model_copy(update={
        "retrieved_chapters": chapters,
        "validated_matches": chapters,
        "current_node": "retrieve_content",
    })


# ============================================================================
# analyze_source Node Tests
# ============================================================================

class TestAnalyzeSourceNode:
    """Tests for analyze_source workflow node.
    
    LangGraph nodes return dicts that get merged into state.
    """
    
    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset LLM client before each test."""
        set_llm_client(None)
    
    @pytest.mark.asyncio
    async def test_analyze_source_extracts_concepts(
        self,
        sample_state: CrossReferenceState,
    ) -> None:
        """Test that analyze_source extracts concepts from content."""
        result = await analyze_source(sample_state)
        
        assert "analyzed_concepts" in result
        assert len(result["analyzed_concepts"]) > 0
        assert result["current_node"] == "analyze_source"
    
    @pytest.mark.asyncio
    async def test_analyze_source_uses_existing_keywords(
        self,
        sample_state: CrossReferenceState,
    ) -> None:
        """Test that existing keywords are included in concepts."""
        result = await analyze_source(sample_state)
        
        # Should include keywords from source (fallback without LLM)
        for keyword in sample_state.source.keywords:
            assert keyword in result["analyzed_concepts"]
    
    @pytest.mark.asyncio
    async def test_analyze_source_extracts_from_content_with_llm(
        self,
        sample_state: CrossReferenceState,
    ) -> None:
        """Test that concepts are extracted from content via LLM."""
        mock_client = AsyncMock()
        mock_client.extract_concepts.return_value = [
            "complexity",
            "change amplification",
            "cognitive load",
        ]
        set_llm_client(mock_client)
        
        result = await analyze_source(sample_state)
        
        assert "complexity" in result["analyzed_concepts"]
        mock_client.extract_concepts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_source_handles_empty_content(self) -> None:
        """Test graceful handling of empty content."""
        source = SourceChapter(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
            content=None,
            keywords=["test"],
        )
        state = CrossReferenceState(source=source)
        
        result = await analyze_source(state)
        
        # Should return empty concepts for empty content
        assert result["analyzed_concepts"] == []
    
    @pytest.mark.asyncio
    async def test_analyze_source_handles_whitespace_content(self) -> None:
        """Test handling of whitespace-only content."""
        source = SourceChapter(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
            content="   \n\t  ",
            keywords=["test"],
        )
        state = CrossReferenceState(source=source)
        
        result = await analyze_source(state)
        
        assert result["analyzed_concepts"] == []


# ============================================================================
# search_taxonomy Node Tests
# ============================================================================

class TestSearchTaxonomyNode:
    """Tests for search_taxonomy workflow node."""
    
    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset Neo4j client before each test."""
        set_neo4j_client(None)
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_finds_matches(
        self,
        state_with_concepts: CrossReferenceState,
    ) -> None:
        """Test that search_taxonomy finds related chapters."""
        mock_client = AsyncMock()
        mock_client.search_chapters.return_value = [
            {"book": "Building Microservices", "chapter": 4, "title": "Test",
             "tier": 2, "similarity": 0.85, "keywords": ["test"]},
        ]
        set_neo4j_client(mock_client)
        
        result = await search_taxonomy(state_with_concepts)
        
        assert len(result["taxonomy_matches"]) > 0
        assert result["current_node"] == "search_taxonomy"
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_respects_tier_config(
        self,
        state_with_concepts: CrossReferenceState,
    ) -> None:
        """Test that tier config is passed to search."""
        state = state_with_concepts.model_copy(update={
            "config": TraversalConfig(include_tier1=True, include_tier2=True, include_tier3=False),
        })
        
        mock_client = AsyncMock()
        mock_client.search_chapters.return_value = []
        set_neo4j_client(mock_client)
        
        await search_taxonomy(state)
        
        # Should have called with tier filter
        call_args = mock_client.search_chapters.call_args
        assert call_args is not None
        assert call_args.kwargs.get("tiers") == [1, 2]
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_handles_no_concepts(
        self,
        sample_state: CrossReferenceState,
    ) -> None:
        """Test graceful handling when no concepts analyzed."""
        state = sample_state.model_copy(update={"analyzed_concepts": []})
        
        result = await search_taxonomy(state)
        
        assert result["taxonomy_matches"] == []
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_converts_to_chapter_match(
        self,
        state_with_concepts: CrossReferenceState,
    ) -> None:
        """Test that raw results are converted to ChapterMatch."""
        mock_client = AsyncMock()
        mock_client.search_chapters.return_value = [
            {"book": "Test Book", "chapter": 1, "title": "Chapter 1",
             "tier": 1, "similarity": 0.9, "keywords": ["test"]},
        ]
        set_neo4j_client(mock_client)
        
        result = await search_taxonomy(state_with_concepts)
        
        match = result["taxonomy_matches"][0]
        assert isinstance(match, ChapterMatch)
        assert match.book == "Test Book"
        assert match.tier == 1


# ============================================================================
# traverse_graph Node Tests
# ============================================================================

class TestTraverseGraphNode:
    """Tests for traverse_graph workflow node."""
    
    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset graph client before each test."""
        set_graph_client(None)
    
    @pytest.mark.asyncio
    async def test_traverse_graph_creates_paths(
        self,
        state_with_matches: CrossReferenceState,
    ) -> None:
        """Test that traverse_graph creates traversal paths."""
        mock_client = AsyncMock()
        mock_client.get_neighbors.return_value = [
            {"book": "Related Book", "chapter": 3, "title": "Ch 3",
             "tier": 2, "similarity": 0.8, "relationship_type": "PERPENDICULAR"},
        ]
        set_graph_client(mock_client)
        
        result = await traverse_graph(state_with_matches)
        
        assert len(result["traversal_paths"]) > 0
        assert result["current_node"] == "traverse_graph"
    
    @pytest.mark.asyncio
    async def test_traverse_graph_respects_max_hops(
        self,
        state_with_matches: CrossReferenceState,
    ) -> None:
        """Test that traversal respects max_hops config."""
        state = state_with_matches.model_copy(update={
            "config": TraversalConfig(max_hops=1),
        })
        
        # Return neighbor on each call
        call_count = 0
        async def mock_neighbors(book: str, chapter: int, **kwargs):
            nonlocal call_count
            await asyncio.sleep(0)  # Satisfy async requirement
            call_count += 1
            if call_count <= 1:
                return [{"book": f"Book{call_count}", "chapter": call_count,
                        "title": "Test", "tier": 2, "similarity": 0.7}]
            return []
        
        mock_client = AsyncMock()
        mock_client.get_neighbors = mock_neighbors
        set_graph_client(mock_client)
        
        result = await traverse_graph(state)
        
        # With max_hops=1, paths should be limited
        for path in result["traversal_paths"]:
            # Start node + max 1 hop = max 2 nodes per path
            assert len(path.nodes) <= 2
    
    @pytest.mark.asyncio
    async def test_traverse_graph_detects_cycles(
        self,
        state_with_matches: CrossReferenceState,
    ) -> None:
        """Test that cycles are detected and avoided."""
        # Return same node to create cycle
        mock_client = AsyncMock()
        mock_client.get_neighbors.return_value = [
            {"book": state_with_matches.taxonomy_matches[0].book,
             "chapter": state_with_matches.taxonomy_matches[0].chapter,
             "title": "Cycle", "tier": 2, "similarity": 0.9},
        ]
        set_graph_client(mock_client)
        
        result = await traverse_graph(state_with_matches)
        
        # Should not infinite loop - cycle detection should stop traversal
        assert "traversal_paths" in result
    
    @pytest.mark.asyncio
    async def test_traverse_graph_handles_no_matches(
        self,
        sample_state: CrossReferenceState,
    ) -> None:
        """Test handling when no taxonomy matches to traverse from."""
        state = sample_state.model_copy(update={"taxonomy_matches": []})
        
        result = await traverse_graph(state)
        
        assert result["traversal_paths"] == []


# ============================================================================
# retrieve_content Node Tests
# ============================================================================

class TestRetrieveContentNode:
    """Tests for retrieve_content workflow node."""
    
    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset content client before each test."""
        set_content_client(None)
    
    @pytest.mark.asyncio
    async def test_retrieve_content_fetches_chapters(
        self,
        state_with_paths: CrossReferenceState,
    ) -> None:
        """Test that retrieve_content fetches chapter content."""
        mock_client = AsyncMock()
        mock_client.get_chapter_content.return_value = {
            "content": "Chapter content...",
            "page_range": "1-20",
        }
        set_content_client(mock_client)
        
        result = await retrieve_content(state_with_paths)
        
        assert len(result["retrieved_chapters"]) > 0
        assert result["current_node"] == "retrieve_content"
    
    @pytest.mark.asyncio
    async def test_retrieve_content_includes_page_range(
        self,
        state_with_paths: CrossReferenceState,
    ) -> None:
        """Test that retrieved chapters include page ranges."""
        mock_client = AsyncMock()
        mock_client.get_chapter_content.return_value = {
            "content": "Chapter content...",
            "page_range": "89-112",
        }
        set_content_client(mock_client)
        
        result = await retrieve_content(state_with_paths)
        
        if result["retrieved_chapters"]:
            assert result["retrieved_chapters"][0].page_range == "89-112"
    
    @pytest.mark.asyncio
    async def test_retrieve_content_validates_matches(
        self,
        state_with_paths: CrossReferenceState,
    ) -> None:
        """Test that only found content creates validated matches."""
        mock_client = AsyncMock()
        mock_client.get_chapter_content.side_effect = [
            {"content": "Found content", "page_range": "1-10"},
            None,  # Not found
        ]
        set_content_client(mock_client)
        
        result = await retrieve_content(state_with_paths)
        
        # validated_matches should only include found content
        assert len(result["validated_matches"]) <= len(result["retrieved_chapters"])
    
    @pytest.mark.asyncio
    async def test_retrieve_content_handles_missing(
        self,
        state_with_paths: CrossReferenceState,
    ) -> None:
        """Test handling when content not found."""
        mock_client = AsyncMock()
        mock_client.get_chapter_content.return_value = None
        set_content_client(mock_client)
        
        result = await retrieve_content(state_with_paths)
        
        # Should still return valid result
        assert "retrieved_chapters" in result
        assert result["retrieved_chapters"] == []


# ============================================================================
# synthesize Node Tests
# ============================================================================

class TestSynthesizeNode:
    """Tests for synthesize workflow node."""
    
    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset synthesis client before each test."""
        set_synthesis_client(None)
    
    @pytest.mark.asyncio
    async def test_synthesize_generates_annotation(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that synthesize generates scholarly annotation."""
        result = await synthesize(state_with_content)
        
        assert result["result"] is not None
        assert result["result"].annotation != ""
        assert result["current_node"] == "synthesize"
    
    @pytest.mark.asyncio
    async def test_synthesize_includes_citations(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that synthesize includes citations in result."""
        result = await synthesize(state_with_content)
        
        assert result["result"] is not None
        assert len(result["result"].citations) > 0
    
    @pytest.mark.asyncio
    async def test_synthesize_citations_are_chicago_format(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that citations follow Chicago format."""
        result = await synthesize(state_with_content)
        
        citation = result["result"].citations[0]
        chicago_str = citation.to_chicago_format(1)
        
        # Chicago format should have book title in italics (*)
        assert "*" in chicago_str
        assert "Ch." in chicago_str
    
    @pytest.mark.asyncio
    async def test_synthesize_includes_tier_coverage(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that result includes tier coverage statistics."""
        result = await synthesize(state_with_content)
        
        assert result["result"] is not None
        assert len(result["result"].tier_coverage) == 3  # All three tiers
    
    @pytest.mark.asyncio
    async def test_synthesize_records_model_used(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that result records which LLM model was used."""
        mock_client = AsyncMock()
        mock_client.generate_annotation.return_value = "LLM annotation"
        set_synthesis_client(mock_client, model_name="claude-3-sonnet")
        
        result = await synthesize(state_with_content)
        
        assert result["result"] is not None
        assert result["result"].model_used == "claude-3-sonnet"
    
    @pytest.mark.asyncio
    async def test_synthesize_handles_empty_matches(self) -> None:
        """Test graceful handling of no validated matches."""
        source = SourceChapter(book="Test", chapter=1, title="Test", tier=1)
        state = CrossReferenceState(
            source=source,
            validated_matches=[],
            current_node="retrieve_content",
        )
        
        result = await synthesize(state)
        
        # Should still produce a result
        assert result["result"] is not None
        assert "No relevant" in result["result"].annotation
    
    @pytest.mark.asyncio
    async def test_synthesize_uses_llm_client(
        self,
        state_with_content: CrossReferenceState,
    ) -> None:
        """Test that LLM client is used when available."""
        mock_client = AsyncMock()
        mock_client.generate_annotation.return_value = "LLM-generated annotation"
        set_synthesis_client(mock_client)
        
        result = await synthesize(state_with_content)
        
        mock_client.generate_annotation.assert_called_once()
        assert "LLM-generated" in result["result"].annotation
