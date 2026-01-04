"""Unit tests for SemanticSearchMcpWrapper.

WBS-PI6: MCP Client & Toolbox
AC-PI6.1: SemanticSearchMcpWrapper initializes
AC-PI6.2: hybrid_search() works
AC-PI6.6: Returns None when flag disabled

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → MCP Integration Architecture
Test Pattern: TDD RED/GREEN/REFACTOR
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_flags_enabled() -> MagicMock:
    """Create mock feature flags with MCP semantic search enabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_semantic_search = True
    return flags


@pytest.fixture
def mock_flags_disabled() -> MagicMock:
    """Create mock feature flags with MCP semantic search disabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_semantic_search = False
    return flags


# -----------------------------------------------------------------------------
# AC-PI6.1: SemanticSearchMcpWrapper initializes with URL
# -----------------------------------------------------------------------------


class TestSemanticSearchMcpWrapperInit:
    """Tests for SemanticSearchMcpWrapper initialization."""
    
    def test_wrapper_creates_with_default_url(self, mock_flags_enabled: MagicMock) -> None:
        """Wrapper should create with default semantic-search-service URL."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        assert wrapper is not None
        assert wrapper.semantic_search_url == "http://localhost:8081"
        assert wrapper.timeout == 30.0
        assert wrapper.flags == mock_flags_enabled
    
    def test_wrapper_creates_with_custom_url(self, mock_flags_enabled: MagicMock) -> None:
        """Wrapper should accept custom URL and timeout."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(
            flags=mock_flags_enabled,
            semantic_search_url="http://semantic:9000",
            timeout=60.0,
        )
        
        assert wrapper.semantic_search_url == "http://semantic:9000"
        assert wrapper.timeout == 60.0
    
    def test_wrapper_client_lazy_initialized(self, mock_flags_enabled: MagicMock) -> None:
        """HTTP client should be None until first use (lazy init)."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        assert wrapper._client is None


# -----------------------------------------------------------------------------
# AC-PI6.2: hybrid_search() calls semantic-search-service
# -----------------------------------------------------------------------------


class TestHybridSearch:
    """Tests for hybrid_search() method."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(
        self, mock_flags_enabled: MagicMock
    ) -> None:
        """hybrid_search() should call service and return results."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        expected_response = {
            "results": [
                {"id": "doc1", "score": 0.95, "text": "Async patterns"},
                {"id": "doc2", "score": 0.87, "text": "Error handling"},
            ],
            "search_type": "hybrid",
        }
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        wrapper._client = mock_client
        
        result = await wrapper.hybrid_search(query="async patterns", top_k=5)
        
        assert result == expected_response
        mock_client.post.assert_called_once_with(
            "/v1/search/hybrid",
            json={"query": "async patterns", "top_k": 5, "scope": []},
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_scope(
        self, mock_flags_enabled: MagicMock
    ) -> None:
        """hybrid_search() should pass search_scope parameter."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wrapper._client = mock_client
        
        await wrapper.hybrid_search(
            query="test",
            top_k=10,
            search_scope=["books", "code-reference-engine"],
        )
        
        mock_client.post.assert_called_once_with(
            "/v1/search/hybrid",
            json={
                "query": "test",
                "top_k": 10,
                "scope": ["books", "code-reference-engine"],
            },
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_lazy_initializes_client(
        self, mock_flags_enabled: MagicMock
    ) -> None:
        """hybrid_search() should create client on first call."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        assert wrapper._client is None
        
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_async_client.return_value = mock_client
            
            await wrapper.hybrid_search(query="test")
            
            mock_async_client.assert_called_once()


# -----------------------------------------------------------------------------
# AC-PI6.6: Returns None when flag disabled
# -----------------------------------------------------------------------------


class TestHybridSearchFlagDisabled:
    """Tests for hybrid_search() when mcp_semantic_search is disabled."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_returns_none_when_disabled(
        self, mock_flags_disabled: MagicMock
    ) -> None:
        """hybrid_search() should return None when flag is disabled."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_disabled)
        
        result = await wrapper.hybrid_search(query="test", top_k=5)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_hybrid_search_does_not_call_service_when_disabled(
        self, mock_flags_disabled: MagicMock
    ) -> None:
        """hybrid_search() should not call service when flag disabled."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_disabled)
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        wrapper._client = mock_client
        
        await wrapper.hybrid_search(query="test", top_k=5)
        
        mock_client.post.assert_not_called()


# -----------------------------------------------------------------------------
# AC-PI6.8: close() cleanup
# -----------------------------------------------------------------------------


class TestClose:
    """Tests for close() method."""
    
    @pytest.mark.asyncio
    async def test_close_closes_client(self, mock_flags_enabled: MagicMock) -> None:
        """close() should close the HTTP client."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        wrapper._client = mock_client
        
        await wrapper.close()
        
        mock_client.aclose.assert_called_once()
        assert wrapper._client is None
    
    @pytest.mark.asyncio
    async def test_close_when_no_client(self, mock_flags_enabled: MagicMock) -> None:
        """close() should handle case when client never created."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        assert wrapper._client is None
        
        # Should not raise
        await wrapper.close()
        
        assert wrapper._client is None


# -----------------------------------------------------------------------------
# Connection pooling (Anti-Pattern #12)
# -----------------------------------------------------------------------------


class TestConnectionPooling:
    """Tests for connection reuse."""
    
    @pytest.mark.asyncio
    async def test_client_reused_across_calls(
        self, mock_flags_enabled: MagicMock
    ) -> None:
        """Same HTTP client should be reused across multiple calls."""
        from src.mcp.semantic_search_wrapper import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=mock_flags_enabled)
        
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_async_client.return_value = mock_client
            
            # First call
            await wrapper.hybrid_search(query="test1")
            first_client = wrapper._client
            
            # Second call - should reuse same client
            await wrapper.hybrid_search(query="test2")
            second_client = wrapper._client
            
            assert first_client is second_client
            # AsyncClient should only be constructed once
            assert mock_async_client.call_count == 1
