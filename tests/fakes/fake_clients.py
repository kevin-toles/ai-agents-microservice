"""Fake Clients for MSEP Unit Testing.

In-memory fake implementations of service clients for unit testing.
Implements the same protocols as real clients for duck typing.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.4
Pattern: FakeClient for testing (CODING_PATTERNS_ANALYSIS.md)
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from numpy.typing import NDArray


class FakeCodeOrchestratorClient:
    """Fake Code-Orchestrator client for unit testing.

    Implements CodeOrchestratorProtocol for duck typing.
    Returns configured responses and supports error injection.

    Attributes:
        call_history: List of recorded method calls for verification

    Example:
        >>> client = FakeCodeOrchestratorClient(
        ...     embeddings_response=np.array([[0.1, 0.2], [0.3, 0.4]])
        ... )
        >>> result = await client.get_embeddings(["text1", "text2"])
        >>> assert client.call_history[0]["method"] == "get_embeddings"
    """

    def __init__(
        self,
        embeddings_response: NDArray[Any] | None = None,
        similarity_response: NDArray[Any] | None = None,
        cluster_response: dict[str, Any] | None = None,
        keywords_response: list[list[str]] | None = None,
        error_on: dict[str, Exception] | None = None,
    ) -> None:
        """Initialize fake client with configured responses.

        Args:
            embeddings_response: Response for get_embeddings
            similarity_response: Response for get_similarity_matrix
            cluster_response: Response for cluster_topics
            keywords_response: Response for extract_keywords
            error_on: Dict mapping method names to exceptions to raise
        """
        self._embeddings = embeddings_response
        self._similarity = similarity_response
        self._cluster = cluster_response
        self._keywords = keywords_response
        self._error_on = error_on or {}
        self.call_history: list[dict[str, Any]] = []

    def _check_error(self, method: str) -> None:
        """Check if error should be raised for method."""
        if method in self._error_on:
            raise self._error_on[method]

    def _record_call(self, method: str, args: dict[str, Any]) -> None:
        """Record method call in history."""
        self.call_history.append({"method": method, "args": args})

    def clear_history(self) -> None:
        """Clear call history for test isolation."""
        self.call_history = []

    async def get_embeddings(self, texts: list[str]) -> NDArray[Any]:
        """Get SBERT embeddings for texts (fake implementation).

        Args:
            texts: List of texts to embed

        Returns:
            Configured embeddings response or empty array
        """
        self._check_error("get_embeddings")
        self._record_call("get_embeddings", {"texts": texts})
        await asyncio.sleep(0)  # Proper async behavior

        if self._embeddings is not None:
            return self._embeddings
        return np.array([])

    async def get_similarity_matrix(self, texts: list[str]) -> NDArray[Any]:
        """Get pairwise similarity matrix (fake implementation).

        Args:
            texts: List of texts to compare

        Returns:
            Configured similarity response or identity matrix
        """
        self._check_error("get_similarity_matrix")
        self._record_call("get_similarity_matrix", {"texts": texts})
        await asyncio.sleep(0)

        if self._similarity is not None:
            return self._similarity
        n = len(texts)
        return np.eye(n)

    async def cluster_topics(
        self, corpus: list[str], chapter_index: int
    ) -> dict[str, Any]:
        """Cluster corpus into topics (fake implementation).

        Args:
            corpus: List of documents to cluster
            chapter_index: Index of the source chapter

        Returns:
            Configured cluster response or default empty result
        """
        self._check_error("cluster_topics")
        self._record_call(
            "cluster_topics", {"corpus": corpus, "chapter_index": chapter_index}
        )
        await asyncio.sleep(0)

        if self._cluster is not None:
            return self._cluster
        return {
            "topic_assignments": [],
            "topic_count": 0,
            "chapter_topic": -1,
            "topics_info": [],
        }

    async def extract_keywords(
        self, corpus: list[str], top_k: int = 5
    ) -> list[list[str]]:
        """Extract TF-IDF keywords from corpus (fake implementation).

        Args:
            corpus: List of documents
            top_k: Number of keywords per document

        Returns:
            Configured keywords response or empty lists
        """
        self._check_error("extract_keywords")
        self._record_call("extract_keywords", {"corpus": corpus, "top_k": top_k})
        await asyncio.sleep(0)

        if self._keywords is not None:
            return self._keywords
        return [[] for _ in corpus]

    async def close(self) -> None:
        """Close client (no-op for fake)."""
        await asyncio.sleep(0)


class FakeSemanticSearchClient:
    """Fake Semantic-Search client for unit testing.

    Implements SemanticSearchProtocol for duck typing.
    Returns configured responses and supports error injection.

    Attributes:
        call_history: List of recorded method calls for verification

    Example:
        >>> client = FakeSemanticSearchClient(
        ...     search_response={"results": [{"chapter_id": "ch1"}], "total": 1}
        ... )
        >>> result = await client.search("test query")
        >>> assert result["total"] == 1
    """

    def __init__(
        self,
        search_response: dict[str, Any] | None = None,
        relationships_response: dict[str, Any] | None = None,
        batch_relationships_response: dict[str, Any] | None = None,
        error_on: dict[str, Exception] | None = None,
    ) -> None:
        """Initialize fake client with configured responses.

        Args:
            search_response: Response for search
            relationships_response: Response for get_relationships
            batch_relationships_response: Response for get_relationships_batch
            error_on: Dict mapping method names to exceptions to raise
        """
        self._search = search_response
        self._relationships = relationships_response
        self._batch_relationships = batch_relationships_response
        self._error_on = error_on or {}
        self.call_history: list[dict[str, Any]] = []

    def _check_error(self, method: str) -> None:
        """Check if error should be raised for method."""
        if method in self._error_on:
            raise self._error_on[method]

    def _record_call(self, method: str, args: dict[str, Any]) -> None:
        """Record method call in history."""
        self.call_history.append({"method": method, "args": args})

    def clear_history(self) -> None:
        """Clear call history for test isolation."""
        self.call_history = []

    async def search(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Hybrid search across chapters (fake implementation).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Configured search response or empty results
        """
        self._check_error("search")
        self._record_call("search", {"query": query, "top_k": top_k})
        await asyncio.sleep(0)

        if self._search is not None:
            return self._search
        return {"results": [], "total": 0, "query": query}

    async def get_relationships(self, chapter_id: str) -> dict[str, Any]:
        """Get graph relationships for a chapter (fake implementation).

        Args:
            chapter_id: ID of the chapter

        Returns:
            Configured relationships response or empty relationships
        """
        self._check_error("get_relationships")
        self._record_call("get_relationships", {"chapter_id": chapter_id})
        await asyncio.sleep(0)

        if self._relationships is not None:
            return self._relationships
        return {"chapter_id": chapter_id, "relationships": []}

    async def get_relationships_batch(
        self, chapter_ids: list[str]
    ) -> dict[str, Any]:
        """Get relationships for multiple chapters (fake implementation).

        Args:
            chapter_ids: List of chapter IDs

        Returns:
            Configured batch response or empty results
        """
        self._check_error("get_relationships_batch")
        self._record_call("get_relationships_batch", {"chapter_ids": chapter_ids})
        await asyncio.sleep(0)

        if self._batch_relationships is not None:
            return self._batch_relationships
        return {"results": {}, "total_chapters": 0}

    async def close(self) -> None:
        """Close client (no-op for fake)."""
        await asyncio.sleep(0)
