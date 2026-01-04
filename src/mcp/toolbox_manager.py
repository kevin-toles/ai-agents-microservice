"""MCP Toolbox Manager for genai-toolbox integration.

WBS-PI6: MCP Client & Toolbox
AC-PI6.3, AC-PI6.4, AC-PI6.5, AC-PI6.6, AC-PI6.7, AC-PI6.8

Manages connections to external genai-toolbox MCP server for
Neo4j and Redis operations.

IMPORTANT: genai-toolbox does NOT support Qdrant.
Use SemanticSearchMcpWrapper for vector search operations.

Reference: https://github.com/googleapis/genai-toolbox
SDK: pip install toolbox-core
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.feature_flags import ProtocolFeatureFlags

# Lazy import to avoid dependency when toolbox-core not installed
ToolboxClient: type | None = None


def _get_toolbox_client() -> type:
    """Lazy import ToolboxClient to avoid hard dependency.
    
    This allows the code to run even if toolbox-core is not installed,
    which is useful for testing and when genai-toolbox is not used.
    """
    global ToolboxClient
    if ToolboxClient is None:
        from toolbox_core import ToolboxClient as TC
        ToolboxClient = TC
    return ToolboxClient


class McpToolboxManager:
    """Manage connections to external genai-toolbox MCP server.
    
    Provides access to Neo4j and Redis toolsets via genai-toolbox.
    Toolsets are cached for connection reuse.
    
    Architecture:
        - External: genai-toolbox server (Go) running on port 5000
        - This class: Python MCP client using toolbox-core SDK
        - Configure databases via tools.yaml on the genai-toolbox server
    
    IMPORTANT: genai-toolbox does NOT support Qdrant.
    For vector search, use SemanticSearchMcpWrapper which wraps
    semantic-search-service (hybrid RAG layer with score fusion).
    
    Example:
        >>> manager = McpToolboxManager(flags=flags)
        >>> tools = await manager.get_neo4j_toolset()
        >>> if tools:
        ...     cypher_tool = tools[0]
        ...     result = await cypher_tool.execute("MATCH (n) RETURN n LIMIT 5")
        >>> await manager.close_all()
    
    Attributes:
        flags: Protocol feature flags for conditional execution
        toolbox_url: URL of external genai-toolbox server
        _toolsets: Cached toolsets for connection reuse
    
    Reference:
        https://github.com/googleapis/genai-toolbox
        SDK: pip install toolbox-core
    """
    
    def __init__(
        self,
        flags: ProtocolFeatureFlags,
        toolbox_url: str = "http://127.0.0.1:5000",
    ) -> None:
        """Initialize the MCP Toolbox Manager.
        
        Args:
            flags: Protocol feature flags for conditional execution
            toolbox_url: URL of external genai-toolbox server
        """
        self.flags = flags
        self.toolbox_url = toolbox_url
        self._toolsets: dict[str, list[Any]] = {}
    
    async def get_neo4j_toolset(self) -> list[Any] | None:
        """Get Neo4j tools from genai-toolbox MCP server.
        
        Returns tools for executing Cypher queries and inspecting
        Neo4j graph schema. Tools are cached for connection reuse.
        
        Returns:
            List of Neo4j tools, or None if mcp_toolbox_neo4j is disabled
            
        Example:
            >>> tools = await manager.get_neo4j_toolset()
            >>> if tools:
            ...     for tool in tools:
            ...         print(tool.name)  # cypher_query, schema_inspect
        """
        if not self.flags.mcp_toolbox_neo4j:
            return None
        
        if "neo4j" not in self._toolsets:
            client_class = _get_toolbox_client()
            async with client_class(self.toolbox_url) as client:
                tools = await client.load_toolset("neo4j_toolset")
                self._toolsets["neo4j"] = tools
        return self._toolsets["neo4j"]
    
    async def get_redis_toolset(self) -> list[Any] | None:
        """Get Redis tools from genai-toolbox MCP server.
        
        Returns tools for Redis cache operations (get, set, etc.).
        Tools are cached for connection reuse.
        
        Returns:
            List of Redis tools, or None if mcp_toolbox_redis is disabled
            
        Example:
            >>> tools = await manager.get_redis_toolset()
            >>> if tools:
            ...     for tool in tools:
            ...         print(tool.name)  # redis_get, redis_set
        """
        if not self.flags.mcp_toolbox_redis:
            return None
        
        if "redis" not in self._toolsets:
            client_class = _get_toolbox_client()
            async with client_class(self.toolbox_url) as client:
                tools = await client.load_toolset("redis_toolset")
                self._toolsets["redis"] = tools
        return self._toolsets["redis"]
    
    async def close_all(self) -> None:
        """Clean up cached toolsets.
        
        Should be called when the manager is no longer needed to
        release any cached resources. After calling this method,
        toolsets can be reloaded by calling get_*_toolset() again.
        """
        self._toolsets.clear()
