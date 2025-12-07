"""Unit tests for Cross-Reference Agent tools.

TDD Phase: RED → GREEN
Pattern: Tool stub testing
Source: GRAPH_RAG_POC_PLAN WBS 5.4-5.5

Note: Tools are currently stubs that return empty results.
"""

import pytest

from src.agents.cross_reference.tools import (
    search_taxonomy,
    search_similar,
    get_chapter_metadata,
    get_chapter_text,
    traverse_graph,
)


class TestTaxonomyTools:
    """Tests for taxonomy search tools."""
    
    @pytest.mark.asyncio
    async def test_search_taxonomy_returns_empty_results(self) -> None:
        """Test that search_taxonomy returns empty results (stub)."""
        result = await search_taxonomy("test", "ai-ml")
        
        assert "results" in result
        assert result["results"] == []
        assert result["taxonomy_id"] == "ai-ml"
    
    @pytest.mark.asyncio
    async def test_search_similar_returns_empty_results(self) -> None:
        """Test that search_similar returns empty results (stub)."""
        result = await search_similar("test query")
        
        assert "results" in result
        assert result["results"] == []


class TestMetadataTools:
    """Tests for metadata retrieval tools."""
    
    @pytest.mark.asyncio
    async def test_get_chapter_metadata_returns_empty(self) -> None:
        """Test that get_chapter_metadata returns empty result (stub)."""
        result = await get_chapter_metadata("Test Book", 1)
        
        assert result is not None
        assert result.get("found") is False or result.get("metadata") is None


class TestContentTools:
    """Tests for content retrieval tools."""
    
    @pytest.mark.asyncio
    async def test_get_chapter_text_returns_empty(self) -> None:
        """Test that get_chapter_text returns empty result (stub)."""
        result = await get_chapter_text("Test Book", 1)
        
        assert result is not None
        assert result.get("found") is False or result.get("text") is None


class TestGraphTools:
    """Tests for graph traversal tools."""
    
    @pytest.mark.asyncio
    async def test_traverse_graph_returns_empty_paths(self) -> None:
        """Test that traverse_graph returns empty paths (stub)."""
        result = await traverse_graph("Test Book", 1, start_tier=1, max_hops=2)
        
        assert "paths" in result
        assert result["paths"] == []
