"""Unit tests for Cross-Reference Agent tools.

TDD Phase: RED → GREEN
Pattern: Tool testing with SemanticSearchClient integration
Source: GRAPH_RAG_POC_PLAN WBS 5.7

WBS 5.7: Tools call SemanticSearchClient to communicate with semantic-search-service.
Anti-Pattern References:
- #4.3: Underscore prefix for unused params
- #8.1: Real async await (not asyncio.sleep(0) stubs)
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.cross_reference.tools import (
    get_chapter_metadata,
    get_chapter_text,
    search_similar,
    search_taxonomy,
    traverse_graph,
)


class TestSearchTaxonomy:
    """Tests for search_taxonomy tool with SemanticSearchClient.
    
    WBS 5.7.1: search_taxonomy calls hybrid_search with taxonomy filtering.
    """
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_calls_client(self) -> None:
        """Test that search_taxonomy calls SemanticSearchClient.hybrid_search."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.return_value = {
            "results": [
                {"id": "book-1-ch-2", "score": 0.85, "payload": {"book": "AI Engineering", "chapter": 2}},
            ],
            "total": 1,
        }
        
        with patch("src.agents.cross_reference.tools.taxonomy.get_semantic_search_client", return_value=mock_client):
            result = await search_taxonomy("RAG pipeline", "ai-ml", source_tier=1)
        
        # Verify client was called
        mock_client.hybrid_search.assert_called_once()
        call_kwargs = mock_client.hybrid_search.call_args.kwargs
        assert call_kwargs["query"] == "RAG pipeline"
        assert call_kwargs["focus_areas"] is not None
        
        # Verify results returned
        assert result["total"] == 1
        assert len(result["results"]) == 1
        assert result["taxonomy_id"] == "ai-ml"
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_raises_when_no_client(self) -> None:
        """Test that search_taxonomy raises RuntimeError if no client injected."""
        with patch("src.agents.cross_reference.tools.taxonomy.get_semantic_search_client", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                await search_taxonomy("test", "ai-ml")
        
        assert "SemanticSearchClient not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_handles_errors(self) -> None:
        """Test that search_taxonomy returns empty on error."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.side_effect = Exception("Connection failed")
        
        with patch("src.agents.cross_reference.tools.taxonomy.get_semantic_search_client", return_value=mock_client):
            result = await search_taxonomy("test", "ai-ml")
        
        assert result["results"] == []
        assert result["total"] == 0
        assert "error" in result


class TestSearchSimilar:
    """Tests for search_similar tool (already implemented - regression tests)."""
    
    @pytest.mark.asyncio
    async def test_search_similar_calls_client(self) -> None:
        """Test that search_similar calls SemanticSearchClient.search_similar."""
        mock_client = AsyncMock()
        mock_client.search_similar.return_value = {
            "results": [{"id": "ch-1", "score": 0.9}],
            "total": 1,
        }
        
        with patch("src.agents.cross_reference.tools.similarity.get_semantic_search_client", return_value=mock_client):
            result = await search_similar("LLM chunking", top_k=5)
        
        mock_client.search_similar.assert_called_once()
        assert result["total"] == 1


class TestGetChapterMetadata:
    """Tests for get_chapter_metadata tool with SemanticSearchClient.
    
    WBS 5.7.2: get_chapter_metadata calls hybrid_search filtered by book+chapter.
    """
    
    @pytest.mark.asyncio
    async def test_get_chapter_metadata_calls_client(self) -> None:
        """Test that get_chapter_metadata calls hybrid_search with book filter."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.return_value = {
            "results": [{
                "id": "ai-eng-ch-3",
                "score": 1.0,
                "payload": {
                    "book": "AI Engineering",
                    "chapter": 3,
                    "title": "RAG Fundamentals",
                    "keywords": ["RAG", "retrieval", "embeddings"],
                    "concepts": ["vector search", "semantic similarity"],
                    "summary": "This chapter covers...",
                    "tier": 1,
                },
            }],
            "total": 1,
        }
        
        with patch("src.agents.cross_reference.tools.metadata.get_semantic_search_client", return_value=mock_client):
            result = await get_chapter_metadata("AI Engineering", 3)
        
        # Verify client was called with book filter
        mock_client.hybrid_search.assert_called_once()
        
        # Verify metadata returned
        assert result["book"] == "AI Engineering"
        assert result["chapter"] == 3
        assert result["title"] == "RAG Fundamentals"
        assert "keywords" in result
        assert len(result["keywords"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_chapter_metadata_not_found(self) -> None:
        """Test that get_chapter_metadata returns empty when chapter not found."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.return_value = {"results": [], "total": 0}
        
        with patch("src.agents.cross_reference.tools.metadata.get_semantic_search_client", return_value=mock_client):
            result = await get_chapter_metadata("Nonexistent Book", 99)
        
        assert result["book"] == "Nonexistent Book"
        assert result["chapter"] == 99
        assert result["keywords"] == []
        assert result["concepts"] == []


class TestGetChapterText:
    """Tests for get_chapter_text tool with SemanticSearchClient.
    
    WBS 5.7.3: get_chapter_text calls hybrid_search and returns content.
    """
    
    @pytest.mark.asyncio
    async def test_get_chapter_text_calls_client(self) -> None:
        """Test that get_chapter_text returns chapter content."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.return_value = {
            "results": [{
                "id": "ai-eng-ch-3",
                "score": 1.0,
                "payload": {
                    "book": "AI Engineering",
                    "chapter": 3,
                    "title": "RAG Fundamentals",
                    "content": "This chapter covers RAG fundamentals...",
                    "page_numbers": [45, 46, 47],
                    "word_count": 5000,
                },
            }],
            "total": 1,
        }
        
        with patch("src.agents.cross_reference.tools.content.get_semantic_search_client", return_value=mock_client):
            result = await get_chapter_text("AI Engineering", 3)
        
        assert result["book"] == "AI Engineering"
        assert result["chapter"] == 3
        assert result["content"] != ""
        assert result["word_count"] > 0
    
    @pytest.mark.asyncio
    async def test_get_chapter_text_not_found(self) -> None:
        """Test that get_chapter_text returns empty when not found."""
        mock_client = AsyncMock()
        mock_client.hybrid_search.return_value = {"results": [], "total": 0}
        
        with patch("src.agents.cross_reference.tools.content.get_semantic_search_client", return_value=mock_client):
            result = await get_chapter_text("Nonexistent", 1)
        
        assert result["content"] == ""
        assert result["word_count"] == 0


class TestTraverseGraph:
    """Tests for traverse_graph tool with SemanticSearchClient.
    
    WBS 5.7.4: traverse_graph calls SemanticSearchClient.traverse().
    """
    
    @pytest.mark.asyncio
    async def test_traverse_graph_calls_client(self) -> None:
        """Test that traverse_graph calls SemanticSearchClient.traverse."""
        mock_client = AsyncMock()
        mock_client.traverse.return_value = {
            "nodes": [
                {"id": "book-1-ch-2", "labels": ["Chapter"], "properties": {"title": "Ch 2"}, "depth": 1},
                {"id": "book-2-ch-5", "labels": ["Chapter"], "properties": {"title": "Ch 5"}, "depth": 2},
            ],
            "edges": [
                {"source": "book-1-ch-1", "target": "book-1-ch-2", "type": "PARALLEL"},
            ],
            "start_node": "book-1-ch-1",
            "depth": 2,
        }
        
        with patch("src.agents.cross_reference.tools.graph.get_semantic_search_client", return_value=mock_client):
            result = await traverse_graph("AI Engineering", 1, start_tier=1, max_hops=2)
        
        mock_client.traverse.assert_called_once()
        assert len(result["paths"]) > 0 or len(result.get("nodes", [])) > 0
    
    @pytest.mark.asyncio
    async def test_traverse_graph_empty_results(self) -> None:
        """Test that traverse_graph handles no paths."""
        mock_client = AsyncMock()
        mock_client.traverse.return_value = {
            "nodes": [],
            "edges": [],
            "start_node": "test",
            "depth": 0,
        }
        
        with patch("src.agents.cross_reference.tools.graph.get_semantic_search_client", return_value=mock_client):
            result = await traverse_graph("Test", 1, start_tier=1, max_hops=2)
        
        assert result.get("paths", []) == [] or result.get("nodes", []) == []
