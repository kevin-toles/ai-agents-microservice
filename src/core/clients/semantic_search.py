"""Semantic Search Service HTTP client.

Provides async client for calling the semantic-search-service API.
Implements WBS 5.7 integration.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.core.config import get_settings


logger = logging.getLogger(__name__)


class SemanticSearchClient:
    """HTTP client for semantic-search-service.

    Provides methods for hybrid search, graph queries, and content retrieval.
    Uses focus_areas for domain-aware filtering.

    Attributes:
        base_url: Base URL for the semantic-search-service
        timeout: Request timeout in seconds
        focus_areas: Default focus areas for domain filtering
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        focus_areas: list[str] | None = None,
    ) -> None:
        """Initialize the semantic search client.

        Args:
            base_url: Base URL for semantic-search-service. Defaults to config.
            timeout: Request timeout in seconds.
            focus_areas: Default focus areas for domain filtering.
        """
        settings = get_settings()
        self.base_url = base_url or settings.semantic_search_url
        self.timeout = timeout
        self.focus_areas = focus_areas or ["llm_rag"]  # Default to LLM/RAG domain
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        collection: str = "documents",
        alpha: float = 0.7,
        tier_filter: list[int] | None = None,
        focus_areas: list[str] | None = None,
        focus_keywords: list[str] | None = None,
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Execute hybrid search with domain filtering.

        Args:
            query: Text query to search for
            limit: Maximum results to return
            collection: Vector collection name
            alpha: Weight for vector vs graph score (0-1)
            tier_filter: Filter to specific tiers (1, 2, 3)
            focus_areas: Domain focus areas (default: self.focus_areas)
            focus_keywords: Custom focus keywords
            min_similarity: Minimum similarity threshold

        Returns:
            Dict with results, count, and metadata
        """
        client = await self._get_client()

        # Use instance default focus_areas if not specified
        effective_focus_areas = focus_areas if focus_areas is not None else self.focus_areas

        payload = {
            "query": query,
            "limit": limit,
            "collection": collection,
            "alpha": alpha,
            "include_graph": True,
            "tier_boost": True,
        }

        # Add optional filters
        if tier_filter:
            payload["tier_filter"] = tier_filter
        if effective_focus_areas:
            payload["focus_areas"] = effective_focus_areas
        if focus_keywords:
            payload["focus_keywords"] = focus_keywords
        if min_similarity > 0:
            payload["min_term_matches"] = 1  # Enable term matching

        logger.debug(
            "Hybrid search request",
            extra={"query": query[:50], "focus_areas": effective_focus_areas, "limit": limit},
        )

        try:
            response = await client.post("/v1/search/hybrid", json=payload)
            response.raise_for_status()
            data = response.json()

            # Filter by min_similarity if specified
            results = data.get("results", [])
            if min_similarity > 0:
                results = [r for r in results if r.get("score", 0) >= min_similarity]

            return {
                "results": results,
                "total": len(results),
                "query": query,
                "focus_areas": effective_focus_areas,
            }

        except httpx.HTTPStatusError as e:
            logger.error(
                "Semantic search failed",
                extra={"status": e.response.status_code, "detail": e.response.text},
            )
            return {
                "results": [],
                "total": 0,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except httpx.RequestError as e:
            logger.error("Semantic search request failed", extra={"error": str(e)})
            return {
                "results": [],
                "total": 0,
                "error": str(e),
            }

    async def search_similar(
        self,
        query_text: str,
        top_k: int = 10,
        filter_tier: int | None = None,
        min_similarity: float = 0.7,
        focus_areas: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find semantically similar content.

        This is a convenience wrapper around hybrid_search.

        Args:
            query_text: Text to find similar content for
            top_k: Number of results to return
            filter_tier: Optional single tier to filter by
            min_similarity: Minimum similarity threshold
            focus_areas: Domain focus areas (defaults to instance value)

        Returns:
            Dict with similar results and scores
        """
        tier_filter = [filter_tier] if filter_tier else None

        return await self.hybrid_search(
            query=query_text,
            limit=top_k,
            tier_filter=tier_filter,
            min_similarity=min_similarity,
            focus_areas=focus_areas,
        )

    async def traverse(
        self,
        start_node_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Traverse the knowledge graph from a starting node.

        Calls POST /v1/graph/traverse on semantic-search-service.

        Args:
            start_node_id: ID of the starting node (e.g., "book-1-ch-2")
            relationship_types: Types of relationships to follow
            max_depth: Maximum traversal depth
            limit: Maximum nodes to return

        Returns:
            Dict with nodes, edges, start_node, depth, and latency_ms
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "limit": limit,
        }

        if relationship_types:
            payload["relationship_types"] = relationship_types

        logger.debug(
            "Graph traverse request",
            extra={
                "start_node_id": start_node_id,
                "max_depth": max_depth,
            },
        )

        try:
            response = await client.post("/v1/graph/traverse", json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "Graph traverse failed",
                extra={"status": e.response.status_code, "detail": e.response.text},
            )
            return {
                "nodes": [],
                "edges": [],
                "start_node": start_node_id,
                "depth": 0,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except httpx.RequestError as e:
            logger.error("Graph traverse request failed", extra={"error": str(e)})
            return {
                "nodes": [],
                "edges": [],
                "start_node": start_node_id,
                "depth": 0,
                "error": str(e),
            }

    async def get_chapter_content(
        self,
        book_id: str,
        chapter_number: int,
    ) -> dict[str, Any] | None:
        """Retrieve chapter content from semantic-search.

        This follows the Kitchen Brigade architecture where ai-agents
        (Expeditor) retrieves content through semantic-search (Cookbook).

        Args:
            book_id: Book identifier (e.g., "test_book_metadata")
            chapter_number: Chapter number (1-indexed)

        Returns:
            Dict with chapter content and metadata, or None if not found.
            Keys: book_id, chapter_number, title, summary, keywords,
                  concepts, page_range, found
        """
        client = await self._get_client()

        logger.debug(
            "Chapter content request",
            extra={"book_id": book_id, "chapter_number": chapter_number},
        )

        try:
            response = await client.get(
                f"/v1/chapters/{book_id}/{chapter_number}"
            )
            response.raise_for_status()
            data = response.json()

            # Check if chapter was found
            if not data.get("found", False):
                logger.info(
                    "Chapter not found: %s/%d", book_id, chapter_number
                )
                return None

            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(
                    "Chapter not found (404): %s/%d", book_id, chapter_number
                )
                return None
            logger.error(
                "Chapter content request failed",
                extra={
                    "status": e.response.status_code,
                    "detail": e.response.text,
                    "book_id": book_id,
                    "chapter_number": chapter_number,
                },
            )
            return None
        except httpx.RequestError as e:
            logger.error(
                "Chapter content request error",
                extra={"error": str(e), "book_id": book_id, "chapter_number": chapter_number},
            )
            return None


# Module-level client for dependency injection
_client: SemanticSearchClient | None = None


def get_semantic_search_client() -> SemanticSearchClient | None:
    """Get the current semantic search client."""
    return _client


def set_semantic_search_client(client: SemanticSearchClient | None) -> None:
    """Set the semantic search client for dependency injection.

    Args:
        client: SemanticSearchClient instance or None to reset
    """
    global _client
    _client = client
