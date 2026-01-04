"""MCP wrapper for semantic-search-service hybrid RAG layer.

WBS-PI6: MCP Client & Toolbox
AC-PI6.1, AC-PI6.2, AC-PI6.6

This wraps semantic-search-service REST API as MCP-compatible tools,
preserving the hybrid search (vector + graph) and domain taxonomy scoring
that reduces LLM hallucination.

Architecture:
    semantic-search-service provides:
    - Hybrid score fusion: α=0.7 vector + 0.3 graph
    - Domain taxonomy scoring (whitelist/blacklist books)
    - Multi-collection search (code-reference-engine + books)
    - Graceful degradation if Neo4j unavailable

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → MCP Integration Architecture
Anti-Patterns: #12 (Connection Pooling), #42/#43 (Async Context Managers)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from src.config.feature_flags import ProtocolFeatureFlags


class SemanticSearchMcpWrapper:
    """MCP wrapper for semantic-search-service hybrid RAG.
    
    Wraps the existing semantic-search-service REST API as MCP tools,
    preserving hybrid search capabilities for hallucination reduction.
    
    The semantic-search-service provides:
    - Hybrid score fusion: α=0.7 vector + 0.3 graph
    - Domain taxonomy scoring for book whitelisting/blacklisting
    - Multi-collection search across code-reference-engine and books
    - Graceful degradation if Neo4j graph is unavailable
    
    Example:
        >>> wrapper = SemanticSearchMcpWrapper(flags=flags)
        >>> result = await wrapper.hybrid_search("async patterns", top_k=5)
        >>> await wrapper.close()
    
    Attributes:
        flags: Protocol feature flags for conditional execution
        semantic_search_url: URL of semantic-search-service
        timeout: HTTP request timeout in seconds
    """
    
    def __init__(
        self,
        flags: ProtocolFeatureFlags,
        semantic_search_url: str = "http://localhost:8081",
        timeout: float = 30.0,
    ) -> None:
        """Initialize the semantic search wrapper.
        
        Args:
            flags: Protocol feature flags for conditional execution
            semantic_search_url: URL of semantic-search-service
            timeout: HTTP request timeout in seconds
        """
        self.flags = flags
        self.semantic_search_url = semantic_search_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client (connection pooling).
        
        Returns:
            The shared HTTP client instance
            
        Note:
            Implements Anti-Pattern #12 (Connection Pooling) - reuses
            the same client across calls instead of creating new ones.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.semantic_search_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        search_scope: list[str] | None = None,
    ) -> dict | None:
        """Hybrid search across knowledge bases.
        
        Performs semantic search using the hybrid RAG layer which combines:
        - Vector search (α=0.7) for semantic similarity
        - Graph search (α=0.3) for relationship context
        
        Returns None if mcp_semantic_search flag is disabled.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            search_scope: Optional list of collections to search
                         (e.g., ["books", "code-reference-engine"])
        
        Returns:
            Search results dict with "results" key, or None if disabled
            
        Example:
            >>> result = await wrapper.hybrid_search("error handling", top_k=5)
            >>> for doc in result["results"]:
            ...     print(f"{doc['score']:.2f}: {doc['text']}")
        """
        if not self.flags.mcp_semantic_search:
            return None
        
        client = await self._get_client()
        response = await client.post(
            "/v1/search/hybrid",
            json={"query": query, "top_k": top_k, "scope": search_scope or []},
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self) -> None:
        """Close HTTP client and release resources.
        
        Should be called when the wrapper is no longer needed to
        properly clean up network connections.
        
        Note:
            Implements Anti-Pattern #42/#43 (Async Context Managers) -
            ensures proper cleanup of async resources.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
