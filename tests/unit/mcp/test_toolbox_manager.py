"""Unit tests for McpToolboxManager.

WBS-PI6: MCP Client & Toolbox
AC-PI6.3: McpToolboxManager initializes
AC-PI6.4: get_neo4j_toolset() returns toolset
AC-PI6.5: get_redis_toolset() returns toolset
AC-PI6.6: Returns None when flag disabled
AC-PI6.7: Connection pooling / singleton behavior
AC-PI6.8: close_all() cleanup

Reference: https://github.com/googleapis/genai-toolbox
Test Pattern: TDD RED/GREEN/REFACTOR

IMPORTANT: genai-toolbox does NOT support Qdrant.
For vector search, use SemanticSearchMcpWrapper.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_flags_all_enabled() -> MagicMock:
    """Create mock feature flags with all MCP toolbox flags enabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_toolbox_neo4j = True
    flags.mcp_toolbox_redis = True
    return flags


@pytest.fixture
def mock_flags_neo4j_only() -> MagicMock:
    """Create mock feature flags with only Neo4j enabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_toolbox_neo4j = True
    flags.mcp_toolbox_redis = False
    return flags


@pytest.fixture
def mock_flags_redis_only() -> MagicMock:
    """Create mock feature flags with only Redis enabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_toolbox_neo4j = False
    flags.mcp_toolbox_redis = True
    return flags


@pytest.fixture
def mock_flags_all_disabled() -> MagicMock:
    """Create mock feature flags with all MCP toolbox flags disabled."""
    flags = MagicMock()
    flags.mcp_enabled = True
    flags.mcp_client_enabled = True
    flags.mcp_toolbox_neo4j = False
    flags.mcp_toolbox_redis = False
    return flags


# -----------------------------------------------------------------------------
# AC-PI6.3: McpToolboxManager initializes
# -----------------------------------------------------------------------------


class TestMcpToolboxManagerInit:
    """Tests for McpToolboxManager initialization."""
    
    def test_manager_creates_with_default_url(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """Manager should create with default genai-toolbox URL."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        assert manager is not None
        assert manager.toolbox_url == "http://127.0.0.1:5000"
        assert manager.flags == mock_flags_all_enabled
        assert manager._toolsets == {}
    
    def test_manager_creates_with_custom_url(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """Manager should accept custom toolbox URL."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(
            flags=mock_flags_all_enabled,
            toolbox_url="http://toolbox:6000",
        )
        
        assert manager.toolbox_url == "http://toolbox:6000"
    
    def test_manager_starts_with_empty_toolsets(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """Manager should start with no cached toolsets."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        assert len(manager._toolsets) == 0


# -----------------------------------------------------------------------------
# AC-PI6.4: get_neo4j_toolset() returns toolset
# -----------------------------------------------------------------------------


class TestGetNeo4jToolset:
    """Tests for get_neo4j_toolset() method."""
    
    @pytest.mark.asyncio
    async def test_get_neo4j_toolset_returns_tools(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """get_neo4j_toolset() should return tools from genai-toolbox."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        expected_tools = [
            MagicMock(name="cypher_query"),
            MagicMock(name="schema_inspect"),
        ]
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(return_value=expected_tools)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await manager.get_neo4j_toolset()
            
            assert result == expected_tools
            mock_client.load_toolset.assert_called_once_with("neo4j_toolset")
    
    @pytest.mark.asyncio
    async def test_get_neo4j_toolset_returns_none_when_disabled(
        self, mock_flags_all_disabled: MagicMock
    ) -> None:
        """get_neo4j_toolset() should return None when flag disabled."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_disabled)
        
        result = await manager.get_neo4j_toolset()
        
        assert result is None


# -----------------------------------------------------------------------------
# AC-PI6.5: get_redis_toolset() returns toolset
# -----------------------------------------------------------------------------


class TestGetRedisToolset:
    """Tests for get_redis_toolset() method."""
    
    @pytest.mark.asyncio
    async def test_get_redis_toolset_returns_tools(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """get_redis_toolset() should return tools from genai-toolbox."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        expected_tools = [
            MagicMock(name="redis_get"),
            MagicMock(name="redis_set"),
        ]
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(return_value=expected_tools)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await manager.get_redis_toolset()
            
            assert result == expected_tools
            mock_client.load_toolset.assert_called_once_with("redis_toolset")
    
    @pytest.mark.asyncio
    async def test_get_redis_toolset_returns_none_when_disabled(
        self, mock_flags_redis_only: MagicMock
    ) -> None:
        """get_redis_toolset() should return None when flag disabled."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        # Use neo4j_only flags (redis is disabled)
        flags = MagicMock()
        flags.mcp_toolbox_redis = False
        
        manager = McpToolboxManager(flags=flags)
        
        result = await manager.get_redis_toolset()
        
        assert result is None


# -----------------------------------------------------------------------------
# AC-PI6.6: Toolsets disabled when flag is false
# -----------------------------------------------------------------------------


class TestToolsetsFlagDisabled:
    """Tests for toolset behavior when flags are disabled."""
    
    @pytest.mark.asyncio
    async def test_neo4j_disabled_does_not_call_toolbox(
        self, mock_flags_redis_only: MagicMock
    ) -> None:
        """get_neo4j_toolset() should not call toolbox when disabled."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_redis_only)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            result = await manager.get_neo4j_toolset()
            
            assert result is None
            mock_client_class.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_redis_disabled_does_not_call_toolbox(
        self, mock_flags_neo4j_only: MagicMock
    ) -> None:
        """get_redis_toolset() should not call toolbox when disabled."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_neo4j_only)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            result = await manager.get_redis_toolset()
            
            assert result is None
            mock_client_class.assert_not_called()


# -----------------------------------------------------------------------------
# AC-PI6.7: Connection pooling / singleton behavior
# -----------------------------------------------------------------------------


class TestConnectionPooling:
    """Tests for toolset caching and reuse."""
    
    @pytest.mark.asyncio
    async def test_neo4j_toolset_cached_on_subsequent_calls(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """get_neo4j_toolset() should cache and return same tools."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        expected_tools = [MagicMock(name="cypher_query")]
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(return_value=expected_tools)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # First call
            result1 = await manager.get_neo4j_toolset()
            # Second call
            result2 = await manager.get_neo4j_toolset()
            
            assert result1 is result2
            # ToolboxClient should only be called once
            assert mock_client_class.call_count == 1
    
    @pytest.mark.asyncio
    async def test_redis_toolset_cached_on_subsequent_calls(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """get_redis_toolset() should cache and return same tools."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        expected_tools = [MagicMock(name="redis_get")]
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(return_value=expected_tools)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # First call
            result1 = await manager.get_redis_toolset()
            # Second call  
            result2 = await manager.get_redis_toolset()
            
            assert result1 is result2
            assert mock_client_class.call_count == 1
    
    @pytest.mark.asyncio
    async def test_different_toolsets_cached_separately(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """Each toolset type should be cached independently."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        neo4j_tools = [MagicMock(name="cypher_query")]
        redis_tools = [MagicMock(name="redis_get")]
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(
                side_effect=[neo4j_tools, redis_tools]
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            neo4j_result = await manager.get_neo4j_toolset()
            redis_result = await manager.get_redis_toolset()
            
            assert neo4j_result == neo4j_tools
            assert redis_result == redis_tools
            assert neo4j_result is not redis_result


# -----------------------------------------------------------------------------
# AC-PI6.8: close_all() cleanup
# -----------------------------------------------------------------------------


class TestCloseAll:
    """Tests for close_all() method."""
    
    @pytest.mark.asyncio
    async def test_close_all_clears_cached_toolsets(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """close_all() should clear all cached toolsets."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        # Manually add toolsets to cache
        manager._toolsets["neo4j"] = [MagicMock()]
        manager._toolsets["redis"] = [MagicMock()]
        
        assert len(manager._toolsets) == 2
        
        await manager.close_all()
        
        assert len(manager._toolsets) == 0
    
    @pytest.mark.asyncio
    async def test_close_all_when_empty(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """close_all() should handle empty toolsets gracefully."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        assert len(manager._toolsets) == 0
        
        # Should not raise
        await manager.close_all()
        
        assert len(manager._toolsets) == 0
    
    @pytest.mark.asyncio
    async def test_close_all_allows_toolset_reload(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """After close_all(), toolsets should be reloadable."""
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        expected_tools = [MagicMock(name="cypher_query")]
        
        with patch("src.mcp.toolbox_manager.ToolboxClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.load_toolset = AsyncMock(return_value=expected_tools)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # First load
            result1 = await manager.get_neo4j_toolset()
            
            # Clear
            await manager.close_all()
            assert len(manager._toolsets) == 0
            
            # Reload should work
            result2 = await manager.get_neo4j_toolset()
            
            assert result2 == expected_tools
            # Should have called ToolboxClient twice now
            assert mock_client_class.call_count == 2


# -----------------------------------------------------------------------------
# No Qdrant Support Test
# -----------------------------------------------------------------------------


class TestNoQdrantSupport:
    """Verify that genai-toolbox Qdrant methods do NOT exist."""
    
    def test_no_get_qdrant_toolset_method(
        self, mock_flags_all_enabled: MagicMock
    ) -> None:
        """McpToolboxManager should NOT have get_qdrant_toolset() method.
        
        genai-toolbox does not support Qdrant. Use SemanticSearchMcpWrapper
        for vector search operations.
        """
        from src.mcp.toolbox_manager import McpToolboxManager
        
        manager = McpToolboxManager(flags=mock_flags_all_enabled)
        
        assert not hasattr(manager, "get_qdrant_toolset")
