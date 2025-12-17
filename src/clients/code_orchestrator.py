"""Code-Orchestrator-Service HTTP Client.

Async client for calling Code-Orchestrator-Service API endpoints.
Implements MSE-3.1 with connection pooling and retry logic.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.1
Anti-Pattern Focus: #12 (Connection Pooling), #42/#43 (Async Context Managers)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from src.agents.msep.constants import (
    DEFAULT_TIMEOUT,
    ENDPOINT_BERTOPIC_CLUSTER,
    ENDPOINT_KEYWORDS_EXTRACT,
    ENDPOINT_SBERT_EMBEDDINGS,
    ENDPOINT_SBERT_SIMILARITY,
    SERVICE_CODE_ORCHESTRATOR,
)
from src.agents.msep.exceptions import ServiceUnavailableError


logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5
RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})


class CodeOrchestratorClient:
    """HTTP client for Code-Orchestrator-Service.

    Provides async methods for SBERT embeddings, similarity, BERTopic clustering,
    and TF-IDF keyword extraction. Uses connection pooling (single httpx.AsyncClient)
    and implements retry logic for transient errors.

    Attributes:
        base_url: Base URL for Code-Orchestrator-Service
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on transient errors

    Example:
        >>> client = CodeOrchestratorClient(base_url="http://localhost:8082")
        >>> embeddings = await client.get_embeddings(["text1", "text2"])
        >>> await client.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the Code-Orchestrator client.

        Args:
            base_url: Base URL for Code-Orchestrator-Service
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
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

    def _is_retryable_error(self, e: httpx.HTTPStatusError) -> bool:
        """Check if HTTP error is retryable (5xx server errors).

        Args:
            e: HTTP status error

        Returns:
            True if error is retryable (5xx), False otherwise
        """
        return e.response.status_code >= 500

    async def _log_and_backoff(
        self, method: str, endpoint: str, attempt: int, error: Exception
    ) -> None:
        """Log retry warning and apply backoff delay.

        Args:
            method: HTTP method being retried
            endpoint: API endpoint being retried
            attempt: Current attempt number (0-indexed)
            error: Exception that triggered retry
        """
        await self._backoff(attempt)
        logger.warning(
            "Retrying %s %s (attempt %d/%d): %s",
            method,
            endpoint,
            attempt + 1,
            self.max_retries,
            str(error),
        )

    async def _execute_request(
        self, client: httpx.AsyncClient, method: str, endpoint: str, json: dict[str, Any] | None
    ) -> httpx.Response:
        """Execute a single HTTP request.

        Args:
            client: HTTP client instance
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            json: Request body as JSON

        Returns:
            HTTP response
        """
        if method.upper() == "POST":
            return await client.post(endpoint, json=json)
        return await client.get(endpoint)

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request with retry logic for transient errors.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            json: Request body as JSON

        Returns:
            Parsed JSON response

        Raises:
            ServiceUnavailableError: After exhausting retries or on connection error
            httpx.HTTPStatusError: On non-retryable client errors (4xx)
        """
        client = await self._get_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._execute_request(client, method, endpoint, json)
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result

            except httpx.HTTPStatusError as e:
                if not self._is_retryable_error(e):
                    raise
                last_exception = e

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e

            if attempt < self.max_retries and last_exception:
                await self._log_and_backoff(method, endpoint, attempt, last_exception)

        raise ServiceUnavailableError(
            message=f"{SERVICE_CODE_ORCHESTRATOR} unavailable after {self.max_retries} retries",
            cause=last_exception,
        )

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retries.

        Args:
            attempt: Current attempt number (0-indexed)
        """
        delay = RETRY_BACKOFF_FACTOR * (2**attempt)
        await asyncio.sleep(delay)

    async def get_embeddings(self, texts: list[str]) -> NDArray[Any]:
        """Get SBERT embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            NDArray of embeddings, shape (n_texts, embedding_dim)

        Raises:
            ServiceUnavailableError: If service unavailable after retries
        """
        if not texts:
            return np.array([])

        data = await self._request_with_retry(
            method="POST",
            endpoint=ENDPOINT_SBERT_EMBEDDINGS,
            json={"texts": texts},
        )
        return np.array(data["embeddings"])

    async def get_similarity_matrix(self, texts: list[str]) -> NDArray[Any]:
        """Get pairwise similarity matrix.

        Args:
            texts: List of texts to compare

        Returns:
            NDArray similarity matrix, shape (n_texts, n_texts)

        Raises:
            ServiceUnavailableError: If service unavailable after retries
        """
        if not texts:
            return np.array([])

        data = await self._request_with_retry(
            method="POST",
            endpoint=ENDPOINT_SBERT_SIMILARITY,
            json={"texts": texts},
        )
        return np.array(data["similarity_matrix"])

    async def cluster_topics(
        self, corpus: list[str], chapter_index: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Cluster corpus into topics using BERTopic.

        Args:
            corpus: List of documents to cluster
            chapter_index: List of chapter metadata dicts with book, chapter, title

        Returns:
            Dict with topic_assignments, topic_count, chapter_topic, topics_info

        Raises:
            ServiceUnavailableError: If service unavailable after retries
        """
        if not corpus:
            return {
                "topic_assignments": [],
                "topic_count": 0,
                "chapter_topic": -1,
                "topics_info": [],
            }

        return await self._request_with_retry(
            method="POST",
            endpoint=ENDPOINT_BERTOPIC_CLUSTER,
            json={"corpus": corpus, "chapter_index": chapter_index},
        )

    async def extract_keywords(
        self, corpus: list[str], top_k: int = 5
    ) -> list[list[str]]:
        """Extract TF-IDF keywords from corpus.

        Args:
            corpus: List of documents to extract keywords from
            top_k: Number of keywords per document (default: 5)

        Returns:
            List of keyword lists, one per document

        Raises:
            ServiceUnavailableError: If service unavailable after retries
        """
        if not corpus:
            return []

        data = await self._request_with_retry(
            method="POST",
            endpoint=ENDPOINT_KEYWORDS_EXTRACT,
            json={"corpus": corpus, "top_k": top_k},
        )
        keywords: list[list[str]] = data["keywords"]
        return keywords
