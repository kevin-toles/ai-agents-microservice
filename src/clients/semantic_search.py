"""Semantic-Search-Service HTTP Client for MSEP.

Async client for calling semantic-search-service API endpoints.
Implements MSE-3.2 with connection pooling and graceful fallback.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.2
Anti-Pattern Focus: #12 (Connection Pooling), #42/#43 (Async Context Managers)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.agents.msep.constants import (
    DEFAULT_TIMEOUT,
    ENDPOINT_GRAPH_RELATIONSHIPS,
    ENDPOINT_GRAPH_RELATIONSHIPS_BATCH,
    ENDPOINT_SEARCH_HYBRID,
)


logger = logging.getLogger(__name__)


class MSEPSemanticSearchClient:
    """HTTP client for semantic-search-service (MSEP-specific).

    Provides async methods for hybrid search and graph relationship queries.
    Uses connection pooling (single httpx.AsyncClient) and graceful fallback
    when service is unavailable.

    Attributes:
        base_url: Base URL for semantic-search-service
        timeout: Request timeout in seconds

    Example:
        >>> client = MSEPSemanticSearchClient(base_url="http://localhost:8081")
        >>> results = await client.search("test-driven development", top_k=5)
        >>> await client.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the semantic search client.

        Args:
            base_url: Base URL for semantic-search-service
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization).

        Returns:
            Shared httpx.AsyncClient instance (connection pooling)
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            await asyncio.sleep(0)  # Yield to event loop on first init
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Hybrid search across chapters.

        Searches for relevant chapters using combined vector and keyword search.
        Falls back gracefully when service is unavailable (AC-3.2.4).

        Args:
            query: Search query
            top_k: Number of results to return (default: 5)

        Returns:
            Dict with results list, total count, and metadata.
            On error, returns empty results with error message.
        """
        client = await self._get_client()

        try:
            response = await client.post(
                ENDPOINT_SEARCH_HYBRID,
                json={"query": query, "limit": top_k},
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Semantic search failed: %s", str(e))
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": str(e),
            }

    async def get_relationships(self, chapter_id: str) -> dict[str, Any]:
        """Get graph relationships for a chapter.

        Retrieves PARALLEL, PERPENDICULAR, and SKIP_TIER relationships
        from the Neo4j graph via semantic-search-service.
        Falls back gracefully when service is unavailable (AC-3.2.4).

        Args:
            chapter_id: ID of the chapter

        Returns:
            Dict with chapter_id and relationships list.
            On error, returns empty relationships with error message.
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{ENDPOINT_GRAPH_RELATIONSHIPS}/{chapter_id}"
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Get relationships failed for %s: %s", chapter_id, str(e))
            return {
                "chapter_id": chapter_id,
                "relationships": [],
                "error": str(e),
            }

    async def get_relationships_batch(
        self, chapter_ids: list[str]
    ) -> dict[str, Any]:
        """Get relationships for multiple chapters.

        Batch retrieval of graph relationships for multiple chapters.
        Falls back gracefully when service is unavailable (AC-3.2.4).

        Args:
            chapter_ids: List of chapter IDs

        Returns:
            Dict with results keyed by chapter_id and total_chapters count.
            On error, returns empty results with error message.
        """
        if not chapter_ids:
            return {"results": {}, "total_chapters": 0}

        client = await self._get_client()

        try:
            response = await client.post(
                ENDPOINT_GRAPH_RELATIONSHIPS_BATCH,
                json={"chapter_ids": chapter_ids},
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Batch relationships failed: %s", str(e))
            return {
                "results": {},
                "total_chapters": 0,
                "error": str(e),
            }
