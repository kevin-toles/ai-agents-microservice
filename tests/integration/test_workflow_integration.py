"""Integration tests for Cross-Reference Agent workflow.

Tests the complete workflow from input to output,
with mocked external services (Neo4j, semantic-search, LLM).

Pattern: Integration testing with dependency injection
Source: GRAPH_RAG_POC_PLAN WBS 5.7
"""

import pytest
from unittest.mock import AsyncMock

from src.agents.cross_reference.agent import CrossReferenceAgent
from src.agents.cross_reference.state import (
    CrossReferenceInput,
    CrossReferenceState,
    SourceChapter,
    TraversalConfig,
    CrossReferenceResult,
)
from src.agents.cross_reference.nodes import (
    set_llm_client,
    set_neo4j_client,
    set_graph_client,
    set_content_client,
    set_synthesis_client,
)


@pytest.fixture(autouse=True)
def reset_all_clients():
    """Reset all clients before each test."""
    set_llm_client(None)
    set_neo4j_client(None)
    set_graph_client(None)
    set_content_client(None)
    set_synthesis_client(None)
    yield
    # Cleanup after test
    set_llm_client(None)
    set_neo4j_client(None)
    set_graph_client(None)
    set_content_client(None)
    set_synthesis_client(None)


class TestFullWorkflowIntegration:
    """Integration tests for complete workflow execution."""
    
    @pytest.fixture
    def mock_llm_client(self) -> AsyncMock:
        """Create mock LLM client for concept extraction."""
        mock = AsyncMock()
        mock.extract_concepts.return_value = [
            "complexity",
            "modularity",
            "abstraction",
            "dependencies",
        ]
        return mock
    
    @pytest.fixture
    def mock_neo4j_client(self) -> AsyncMock:
        """Create mock Neo4j client for taxonomy search."""
        mock = AsyncMock()
        mock.search_chapters.return_value = [
            {
                "book": "Building Microservices",
                "chapter": 4,
                "title": "Decomposing the Monolith",
                "tier": 2,
                "similarity": 0.85,
                "keywords": ["modularity", "boundaries"],
            },
            {
                "book": "Clean Architecture",
                "chapter": 22,
                "title": "The Clean Architecture",
                "tier": 1,
                "similarity": 0.82,
                "keywords": ["architecture", "layers"],
            },
        ]
        return mock
    
    @pytest.fixture
    def mock_graph_client(self) -> AsyncMock:
        """Create mock graph client for traversal."""
        mock = AsyncMock()
        mock.get_neighbors.return_value = [
            {
                "book": "Domain-Driven Design",
                "chapter": 5,
                "title": "A Model Expressed in Software",
                "tier": 2,
                "similarity": 0.78,
                "relationship_type": "PERPENDICULAR",
            },
        ]
        return mock
    
    @pytest.fixture
    def mock_content_client(self) -> AsyncMock:
        """Create mock content client for retrieval."""
        mock = AsyncMock()
        mock.get_chapter_content.return_value = {
            "content": "Chapter content about architecture patterns...",
            "page_range": "45-72",
        }
        return mock
    
    @pytest.fixture
    def mock_synthesis_client(self) -> AsyncMock:
        """Create mock synthesis client for annotation."""
        mock = AsyncMock()
        mock.generate_annotation.return_value = (
            "This chapter's treatment of complexity relates to "
            "the modular decomposition strategies in Newman's "
            "Building Microservices[^1] and the architectural "
            "principles in Clean Architecture[^2]."
        )
        return mock
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_all_mocks(
        self,
        mock_llm_client: AsyncMock,
        mock_neo4j_client: AsyncMock,
        mock_graph_client: AsyncMock,
        mock_content_client: AsyncMock,
        mock_synthesis_client: AsyncMock,
    ) -> None:
        """Test complete workflow with all mocked services."""
        # Setup clients
        set_llm_client(mock_llm_client)
        set_neo4j_client(mock_neo4j_client)
        set_graph_client(mock_graph_client)
        set_content_client(mock_content_client)
        set_synthesis_client(mock_synthesis_client, model_name="claude-3-sonnet")
        
        # Create agent and input
        agent = CrossReferenceAgent()
        input_data = CrossReferenceInput(
            book="A Philosophy of Software Design",
            chapter=2,
            title="The Nature of Complexity",
            tier=1,
            content="Complexity is anything related to software structure...",
            keywords=["complexity"],
        )
        
        # Run workflow
        result = await agent.run(input_data)
        
        # Verify result
        assert result is not None
        assert isinstance(result, CrossReferenceResult)
        assert result.annotation != ""
        assert len(result.citations) > 0
        assert len(result.tier_coverage) == 3
        assert result.model_used == "claude-3-sonnet"
        
        # Verify all mocks were called
        mock_llm_client.extract_concepts.assert_called_once()
        mock_neo4j_client.search_chapters.assert_called_once()
        assert mock_content_client.get_chapter_content.called
    
    @pytest.mark.asyncio
    async def test_workflow_without_external_services(self) -> None:
        """Test workflow runs with stub implementations."""
        agent = CrossReferenceAgent()
        input_data = CrossReferenceInput(
            book="Test Book",
            chapter=1,
            title="Test Chapter",
            tier=1,
            content="Test content",
            keywords=["test"],
        )
        
        # Run workflow (no mocks - uses stubs)
        result = await agent.run(input_data)
        
        # Should still produce result
        assert result is not None
        assert isinstance(result, CrossReferenceResult)
    
    @pytest.mark.asyncio
    async def test_workflow_tier_filtering(
        self,
        mock_llm_client: AsyncMock,
        mock_neo4j_client: AsyncMock,
        mock_graph_client: AsyncMock,
        mock_content_client: AsyncMock,
    ) -> None:
        """Test that tier filtering is applied correctly."""
        set_llm_client(mock_llm_client)
        set_neo4j_client(mock_neo4j_client)
        set_graph_client(mock_graph_client)
        set_content_client(mock_content_client)
        
        agent = CrossReferenceAgent()
        input_data = CrossReferenceInput(
            book="Test Book",
            chapter=1,
            title="Test",
            tier=1,
            content="Test",
            config=TraversalConfig(
                include_tier1=True,
                include_tier2=True,
                include_tier3=False,  # Exclude tier 3
            ),
        )
        
        await agent.run(input_data)
        
        # Verify tier filter was passed
        call_args = mock_neo4j_client.search_chapters.call_args
        assert call_args is not None
        assert call_args.kwargs.get("tiers") == [1, 2]


class TestWorkflowNodeInteraction:
    """Tests for node-to-node data flow."""
    
    @pytest.fixture
    def mock_clients(self) -> dict:
        """Create all mock clients."""
        return {
            "llm": AsyncMock(extract_concepts=AsyncMock(return_value=["concept1"])),
            "neo4j": AsyncMock(search_chapters=AsyncMock(return_value=[
                {"book": "B", "chapter": 1, "title": "T", "tier": 1, "similarity": 0.9, "keywords": []},
            ])),
            "graph": AsyncMock(get_neighbors=AsyncMock(return_value=[])),
            "content": AsyncMock(get_chapter_content=AsyncMock(return_value={
                "content": "C", "page_range": "1-10",
            })),
            "synthesis": AsyncMock(generate_annotation=AsyncMock(return_value="Annotation")),
        }
    
    @pytest.mark.asyncio
    async def test_concepts_flow_to_search(self, mock_clients: dict) -> None:
        """Test that analyzed concepts flow to taxonomy search."""
        set_llm_client(mock_clients["llm"])
        set_neo4j_client(mock_clients["neo4j"])
        set_graph_client(mock_clients["graph"])
        set_content_client(mock_clients["content"])
        
        agent = CrossReferenceAgent()
        await agent.run(CrossReferenceInput(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
            content="Test content",
        ))
        
        # Verify search was called with extracted concepts
        search_call = mock_clients["neo4j"].search_chapters.call_args
        assert search_call is not None
        assert search_call.kwargs.get("concepts") == ["concept1"]
    
    @pytest.mark.asyncio
    async def test_matches_flow_to_traversal(self, mock_clients: dict) -> None:
        """Test that taxonomy matches flow to graph traversal."""
        set_llm_client(mock_clients["llm"])
        set_neo4j_client(mock_clients["neo4j"])
        set_graph_client(mock_clients["graph"])
        set_content_client(mock_clients["content"])
        
        agent = CrossReferenceAgent()
        await agent.run(CrossReferenceInput(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
            content="Test",
        ))
        
        # Verify graph traversal was called (means matches flowed through)
        assert mock_clients["graph"].get_neighbors.called


class TestCitationGeneration:
    """Tests for citation formatting."""
    
    @pytest.mark.asyncio
    async def test_citations_are_chicago_format(self) -> None:
        """Test that citations follow Chicago style."""
        mock_neo4j = AsyncMock()
        mock_neo4j.search_chapters.return_value = [
            {
                "book": "Building Microservices",
                "chapter": 4,
                "title": "Decomposing the Monolith",
                "tier": 2,
                "similarity": 0.85,
                "keywords": [],
            },
        ]
        mock_content = AsyncMock()
        mock_content.get_chapter_content.return_value = {
            "content": "Chapter content",
            "page_range": "89-112",
        }
        
        set_neo4j_client(mock_neo4j)
        set_content_client(mock_content)
        
        # Need concepts to trigger search
        mock_llm = AsyncMock()
        mock_llm.extract_concepts.return_value = ["decomposition"]
        set_llm_client(mock_llm)
        
        agent = CrossReferenceAgent()
        result = await agent.run(CrossReferenceInput(
            book="Test",
            chapter=1,
            title="Test",
            tier=1,
            content="Test content about decomposition",
        ))
        
        if result.citations:
            citation = result.citations[0]
            chicago = citation.to_chicago_format(1)
            
            # Chicago format markers
            assert "[^1]:" in chicago
            assert "*Building Microservices*" in chicago
            assert "Ch. 4" in chicago
            assert "pp. 89-112" in chicago
