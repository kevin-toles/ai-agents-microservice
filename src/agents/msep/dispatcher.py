"""MSEP Parallel Dispatcher.

WBS: MSE-4.2 - Parallel Dispatcher
Implements asyncio.gather() orchestration for parallel service calls.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.2
- MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md: Parallel execution

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S3776: Cognitive complexity < 15 per function (extracted helpers)
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.agents.msep.exceptions import ServiceUnavailableError


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.agents.msep.schemas import MSEPRequest
    from src.clients.protocols import CodeOrchestratorProtocol, SemanticSearchProtocol


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result from parallel dispatch operations.

    Attributes:
        embeddings: SBERT embeddings for corpus.
        similarity_matrix: Pairwise similarity scores.
        topics: BERTopic cluster assignments.
        topic_info: Topic metadata.
        keywords: TF-IDF extracted keywords.
        hybrid_results: Graph relationship results (optional).
        embeddings_error: Error from embeddings call (if any).
        similarity_error: Error from similarity call (if any).
        topics_error: Error from topics call (if any).
        keywords_error: Error from keywords call (if any).
        hybrid_error: Error from hybrid search (if any).
    """

    embeddings: list[list[float]] | None = None
    similarity_matrix: list[list[float]] | None = None
    topics: list[int] | None = None
    topic_info: list[dict[str, Any]] | None = None
    keywords: list[list[str]] | None = None
    hybrid_results: dict[str, Any] | None = None
    embeddings_error: str | None = None
    similarity_error: str | None = None
    topics_error: str | None = None
    keywords_error: str | None = None
    hybrid_error: str | None = None


class MSEPDispatcher:
    """Parallel dispatcher for MSEP service calls.

    Coordinates concurrent calls to Code-Orchestrator and semantic-search
    using asyncio.gather() for efficient parallel execution.

    Pattern: Dispatcher pattern with parallel execution
    """

    def __init__(
        self,
        code_orchestrator: CodeOrchestratorProtocol,
        semantic_search: SemanticSearchProtocol,
    ) -> None:
        """Initialize dispatcher with service clients.

        Args:
            code_orchestrator: Client for Code-Orchestrator-Service
            semantic_search: Client for semantic-search-service
        """
        self._code_orchestrator = code_orchestrator
        self._semantic_search = semantic_search

    async def dispatch_enrichment(self, request: MSEPRequest) -> DispatchResult:
        """Dispatch parallel enrichment calls to services.

        Executes SBERT, TF-IDF, and BERTopic calls concurrently.
        Optionally includes hybrid search if enabled.

        Args:
            request: MSEP request with corpus and config

        Returns:
            DispatchResult with all service responses

        Raises:
            ServiceUnavailableError: When critical services fail
        """
        # Build chapter IDs for hybrid search
        chapter_ids = [ch.id for ch in request.chapter_index]

        # Create tasks for parallel execution
        tasks = self._create_enrichment_tasks(request, chapter_ids)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with error handling
        return self._process_task_results(results, request.config.enable_hybrid_search)

    def _create_enrichment_tasks(
        self, request: MSEPRequest, chapter_ids: list[str]
    ) -> list[asyncio.Task[Any]]:
        """Create asyncio tasks for all enrichment operations.

        Args:
            request: MSEP request
            chapter_ids: List of chapter IDs

        Returns:
            List of asyncio tasks
        """
        # Convert chapter_index to list of dicts for Code-Orchestrator API
        chapter_index_dicts = [
            {"book": ch.book, "chapter": ch.chapter, "title": ch.title}
            for ch in request.chapter_index
        ]

        # Create tasks and save references to prevent garbage collection
        embeddings_task = asyncio.create_task(self._get_embeddings(request.corpus))
        similarity_task = asyncio.create_task(self._get_similarity(request.corpus))
        topics_task = asyncio.create_task(
            self._get_topics(request.corpus, chapter_index_dicts)
        )
        keywords_task = asyncio.create_task(self._get_keywords(request.corpus))

        tasks: list[asyncio.Task[Any]] = [
            embeddings_task,
            similarity_task,
            topics_task,
            keywords_task,
        ]

        # Hybrid search is conditional
        if request.config.enable_hybrid_search:
            tasks.append(
                asyncio.create_task(self._get_hybrid_relationships(chapter_ids))
            )

        return tasks

    def _process_task_results(
        self, results: list[Any], hybrid_enabled: bool
    ) -> DispatchResult:
        """Process task results and build DispatchResult.

        Args:
            results: List of results from asyncio.gather
            hybrid_enabled: Whether hybrid search was enabled

        Returns:
            DispatchResult with processed results

        Raises:
            ServiceUnavailableError: When embeddings or similarity fail
        """
        dispatch_result = DispatchResult()

        # Process embeddings (critical - index 0)
        self._process_embeddings_result(results[0], dispatch_result)

        # Process similarity (critical - index 1)
        self._process_similarity_result(results[1], dispatch_result)

        # Process topics (non-critical - index 2)
        self._process_topics_result(results[2], dispatch_result)

        # Process keywords (non-critical - index 3)
        self._process_keywords_result(results[3], dispatch_result)

        # Process hybrid (non-critical, if enabled - index 4)
        if hybrid_enabled and len(results) > 4:
            self._process_hybrid_result(results[4], dispatch_result)

        return dispatch_result

    async def _get_embeddings(self, corpus: list[str]) -> NDArray[Any]:
        """Get SBERT embeddings from Code-Orchestrator.

        Args:
            corpus: List of document texts

        Returns:
            NDArray of embeddings, shape (n_texts, embedding_dim)
        """
        return await self._code_orchestrator.get_embeddings(corpus)

    async def _get_similarity(self, corpus: list[str]) -> NDArray[Any]:
        """Get similarity matrix from Code-Orchestrator.

        Args:
            corpus: List of document texts

        Returns:
            NDArray similarity matrix, shape (n_texts, n_texts)
        """
        return await self._code_orchestrator.get_similarity_matrix(corpus)

    async def _get_topics(
        self, corpus: list[str], chapter_index: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get topic clusters from Code-Orchestrator.

        Args:
            corpus: List of document texts
            chapter_index: List of chapter metadata dicts

        Returns:
            Topics response dict
        """
        return await self._code_orchestrator.cluster_topics(corpus, chapter_index)

    async def _get_keywords(self, corpus: list[str]) -> list[list[str]]:
        """Get TF-IDF keywords from Code-Orchestrator.

        Args:
            corpus: List of document texts

        Returns:
            List of keyword lists, one per document
        """
        return await self._code_orchestrator.extract_keywords(corpus)

    async def _get_hybrid_relationships(
        self, chapter_ids: list[str]
    ) -> dict[str, Any]:
        """Get graph relationships from semantic-search.

        Args:
            chapter_ids: List of chapter IDs

        Returns:
            Relationships response dict
        """
        return await self._semantic_search.get_relationships_batch(chapter_ids)

    def _process_embeddings_result(
        self, result: Any, dispatch_result: DispatchResult
    ) -> None:
        """Process embeddings result (critical service).

        Args:
            result: Result from embeddings call (NDArray or Exception)
            dispatch_result: DispatchResult to update

        Raises:
            ServiceUnavailableError: When embeddings fail
        """
        if isinstance(result, Exception):
            logger.error(f"Embeddings failed: {result}")
            raise ServiceUnavailableError(
                f"Code-Orchestrator embeddings failed: {result}",
                service="code-orchestrator",
            )
        # Client returns numpy array directly, not a dict
        if isinstance(result, np.ndarray):
            dispatch_result.embeddings = result.tolist()
        else:
            dispatch_result.embeddings = result.get("embeddings", [])

    def _process_similarity_result(
        self, result: Any, dispatch_result: DispatchResult
    ) -> None:
        """Process similarity result (critical service).

        Args:
            result: Result from similarity call (NDArray or Exception)
            dispatch_result: DispatchResult to update

        Raises:
            ServiceUnavailableError: When similarity fails
        """
        if isinstance(result, Exception):
            logger.error(f"Similarity failed: {result}")
            raise ServiceUnavailableError(
                f"Code-Orchestrator similarity failed: {result}",
                service="code-orchestrator",
            )
        # Client returns numpy array directly, not a dict
        if isinstance(result, np.ndarray):
            dispatch_result.similarity_matrix = result.tolist()
        else:
            dispatch_result.similarity_matrix = result.get("similarity_matrix", [])

    def _process_topics_result(
        self, result: Any, dispatch_result: DispatchResult
    ) -> None:
        """Process topics result (non-critical service).

        Args:
            result: Result from topics call
            dispatch_result: DispatchResult to update
        """
        if isinstance(result, Exception):
            logger.warning(f"Topics failed (non-critical): {result}")
            dispatch_result.topics = []
            dispatch_result.topic_info = []
            dispatch_result.topics_error = str(result)
        else:
            # Extract topic_id from assignments (list of ClusterAssignment dicts)
            assignments = result.get("assignments", [])
            dispatch_result.topics = [
                a.get("topic_id", -1) if isinstance(a, dict) else -1
                for a in assignments
            ]
            dispatch_result.topic_info = result.get("topics", [])

    def _process_keywords_result(
        self, result: Any, dispatch_result: DispatchResult
    ) -> None:
        """Process keywords result (non-critical service).

        Args:
            result: Result from keywords call (list or Exception)
            dispatch_result: DispatchResult to update
        """
        if isinstance(result, Exception):
            logger.warning(f"Keywords failed (non-critical): {result}")
            dispatch_result.keywords = []
            dispatch_result.keywords_error = str(result)
        elif isinstance(result, list):
            # Client returns list directly
            dispatch_result.keywords = result
        else:
            dispatch_result.keywords = result.get("keywords", [])

    def _process_hybrid_result(
        self, result: Any, dispatch_result: DispatchResult
    ) -> None:
        """Process hybrid result (non-critical service).

        Args:
            result: Result from hybrid call
            dispatch_result: DispatchResult to update
        """
        if isinstance(result, Exception):
            logger.warning(f"Hybrid search failed (non-critical): {result}")
            dispatch_result.hybrid_results = None
            dispatch_result.hybrid_error = str(result)
        else:
            dispatch_result.hybrid_results = result.get("results", {})
