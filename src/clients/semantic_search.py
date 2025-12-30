"""Semantic-Search-Service HTTP Client for MSEP.

Async client for calling semantic-search-service API endpoints.
Implements MSE-3.2 with connection pooling and graceful fallback.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.2
Anti-Pattern Focus: #12 (Connection Pooling), #42/#43 (Async Context Managers)

Also provides:
- FakeSemanticSearchClient: Test double for unit testing (WBS-AGT13)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.clients.protocols import SemanticSearchProtocol

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


class FakeSemanticSearchClient(SemanticSearchProtocol):
    """Fake semantic search client for unit testing.

    WBS-AGT13 Exit Criteria: FakeSemanticSearchClient used in unit tests.

    Provides configurable mock responses for testing cross_reference function
    without hitting real semantic-search-service.

    Attributes:
        search_results: Configured search results to return
        relationships_results: Configured relationships to return

    Example:
        >>> client = FakeSemanticSearchClient(search_results=[
        ...     {"source": "repo/file.py", "content": "code", "score": 0.9}
        ... ])
        >>> results = await client.search("test", top_k=5)
        >>> assert len(results) == 1
    """

    def __init__(
        self,
        search_results: list[dict[str, Any]] | None = None,
        relationships_results: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the fake client with configurable responses.

        Args:
            search_results: List of search results to return.
                Each item should have: source, content, score.
                If None, returns default mock data.
            relationships_results: Relationships data to return.
                If None, returns empty relationships.
        """
        self._search_results = search_results if search_results is not None else [
            {
                "source": "code-reference-engine/backend/ddd/repository.py",
                "content": "class BaseRepository(ABC): ...",
                "score": 0.89,
                "source_type": "code",
                "line_range": "12-45",
            },
            {
                "source": "books/peaa/chapter-11.md",
                "content": "The Repository pattern provides a collection-like interface...",
                "score": 0.85,
                "source_type": "book",
            },
        ]
        self._relationships_results = relationships_results or {
            "chapter_id": "",
            "relationships": [],
        }

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return configured mock search results.

        Note: Returns list directly for cross_reference function compatibility.
        The real MSEPSemanticSearchClient returns a dict with 'results' key,
        but cross_reference extracts the results list internally.

        Args:
            query: Search query (ignored for fake)
            top_k: Maximum results to return

        Returns:
            List of mock search result dicts, limited by top_k.
        """
        # Simulate async behavior
        await asyncio.sleep(0)
        return self._search_results[:top_k]

    async def get_relationships(self, chapter_id: str) -> dict[str, Any]:
        """Return configured mock relationships.

        Args:
            chapter_id: Chapter ID (used in response)

        Returns:
            Dict with chapter_id and relationships list.
        """
        await asyncio.sleep(0)
        result = dict(self._relationships_results)
        result["chapter_id"] = chapter_id
        return result

    async def get_relationships_batch(
        self, chapter_ids: list[str]
    ) -> dict[str, Any]:
        """Return mock batch relationships.

        Args:
            chapter_ids: List of chapter IDs

        Returns:
            Dict with results keyed by chapter_id.
        """
        await asyncio.sleep(0)
        return {
            "results": {
                cid: {"chapter_id": cid, "relationships": []}
                for cid in chapter_ids
            },
            "total_chapters": len(chapter_ids),
        }

    async def close(self) -> None:
        """No-op close for fake client."""
        pass

