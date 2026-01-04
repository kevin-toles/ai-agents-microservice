"""Integration tests for MCP Client components.

WBS-PI6: MCP Client & Toolbox
AC-PI6.9: cross_reference can optionally use MCP semantic search

These tests verify the MCP client components integrate correctly
with the feature flags system and can be initialized together.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Integration Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def all_mcp_client_flags_enabled() -> MagicMock:
    """Create mock flags with all MCP client features enabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_semantic_search = True
    flags.mcp_toolbox_neo4j = True
    flags.mcp_toolbox_redis = True
    return flags


@pytest.fixture
def all_mcp_client_flags_disabled() -> MagicMock:
    """Create mock flags with all MCP client features disabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_semantic_search = False
    flags.mcp_toolbox_neo4j = False
    flags.mcp_toolbox_redis = False
    return flags


# -----------------------------------------------------------------------------
# Module Import Tests
# -----------------------------------------------------------------------------


class TestMcpModuleExports:
    """Verify MCP module exports the correct classes."""
    
    def test_semantic_search_wrapper_exported(self) -> None:
        """SemanticSearchMcpWrapper should be importable from mcp module."""
        from src.mcp import SemanticSearchMcpWrapper
        
        assert SemanticSearchMcpWrapper is not None
    
    def test_toolbox_manager_exported(self) -> None:
        """McpToolboxManager should be importable from mcp module."""
        from src.mcp import McpToolboxManager
        
        assert McpToolboxManager is not None


# -----------------------------------------------------------------------------
# Combined Client Initialization
# -----------------------------------------------------------------------------


class TestCombinedClientInit:
    """Test both MCP clients can be initialized together."""
    
    def test_both_clients_initialize_with_same_flags(
        self, all_mcp_client_flags_enabled: MagicMock
    ) -> None:
        """Both clients should initialize with the same flags instance."""
        from src.mcp import McpToolboxManager, SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=all_mcp_client_flags_enabled)
        manager = McpToolboxManager(flags=all_mcp_client_flags_enabled)
        
        assert wrapper.flags is manager.flags
    
    @pytest.mark.asyncio
    async def test_both_clients_respect_flag_state(
        self, all_mcp_client_flags_disabled: MagicMock
    ) -> None:
        """Both clients should return None when flags disabled."""
        from src.mcp import McpToolboxManager, SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=all_mcp_client_flags_disabled)
        manager = McpToolboxManager(flags=all_mcp_client_flags_disabled)
        
        # All should return None when disabled
        search_result = await wrapper.hybrid_search(query="test")
        neo4j_result = await manager.get_neo4j_toolset()
        redis_result = await manager.get_redis_toolset()
        
        assert search_result is None
        assert neo4j_result is None
        assert redis_result is None


# -----------------------------------------------------------------------------
# Cleanup Integration
# -----------------------------------------------------------------------------


class TestCombinedCleanup:
    """Test cleanup of both clients."""
    
    @pytest.mark.asyncio
    async def test_both_clients_cleanup_independently(
        self, all_mcp_client_flags_enabled: MagicMock
    ) -> None:
        """Each client should cleanup without affecting the other."""
        from src.mcp import McpToolboxManager, SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=all_mcp_client_flags_enabled)
        manager = McpToolboxManager(flags=all_mcp_client_flags_enabled)
        
        # Add mock client to wrapper
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()
        wrapper._client = mock_http_client
        
        # Add cached toolset to manager
        manager._toolsets["neo4j"] = [MagicMock()]
        
        # Cleanup wrapper
        await wrapper.close()
        assert wrapper._client is None
        assert len(manager._toolsets) == 1  # Manager unaffected
        
        # Cleanup manager
        await manager.close_all()
        assert len(manager._toolsets) == 0


# -----------------------------------------------------------------------------
# Feature Flag Integration with Real Config
# -----------------------------------------------------------------------------


class TestFeatureFlagIntegration:
    """Test MCP clients with real feature flags when possible."""
    
    @pytest.mark.asyncio
    async def test_clients_work_with_default_flags(self) -> None:
        """Clients should work with default flag values."""
        from src.config.feature_flags import ProtocolFeatureFlags
        from src.mcp import McpToolboxManager, SemanticSearchMcpWrapper
        
        # Create flags with defaults (all disabled by default)
        flags = ProtocolFeatureFlags()
        
        wrapper = SemanticSearchMcpWrapper(flags=flags)
        manager = McpToolboxManager(flags=flags)
        
        # With default flags (disabled), should return None
        result = await wrapper.hybrid_search(query="test")
        assert result is None
        
        neo4j_result = await manager.get_neo4j_toolset()
        assert neo4j_result is None
    
    @pytest.mark.asyncio
    async def test_semantic_search_respects_enabled_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SemanticSearchMcpWrapper should work when flag is enabled."""
        # Enable the flags via environment
        monkeypatch.setenv("AGENTS_MCP_ENABLED", "true")
        monkeypatch.setenv("AGENTS_MCP_SEMANTIC_SEARCH", "true")
        
        from src.config.feature_flags import ProtocolFeatureFlags
        from src.mcp import SemanticSearchMcpWrapper
        
        flags = ProtocolFeatureFlags()
        wrapper = SemanticSearchMcpWrapper(flags=flags)
        
        # Mock the HTTP client to avoid actual network call
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"score": 0.95}]}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wrapper._client = mock_client
        
        result = await wrapper.hybrid_search(query="test")
        
        assert result is not None
        assert "results" in result


# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in MCP clients."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_propagates_http_errors(
        self, all_mcp_client_flags_enabled: MagicMock
    ) -> None:
        """HTTP errors should propagate to caller."""
        from httpx import HTTPStatusError, Request, Response
        
        from src.mcp import SemanticSearchMcpWrapper
        
        wrapper = SemanticSearchMcpWrapper(flags=all_mcp_client_flags_enabled)
        
        # Mock client that raises HTTP error
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 500
        mock_response.request = MagicMock(spec=Request)
        mock_response.raise_for_status = MagicMock(
            side_effect=HTTPStatusError(
                "Server Error",
                request=mock_response.request,
                response=mock_response,
            )
        )
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        wrapper._client = mock_client
        
        with pytest.raises(HTTPStatusError):
            await wrapper.hybrid_search(query="test")
