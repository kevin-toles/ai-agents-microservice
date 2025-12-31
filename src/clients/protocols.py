"""MSEP Service Client Protocols.

Duck typing protocols for service clients - enables FakeClient substitution in tests.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-3.3
Pattern: Protocol duck typing (CODING_PATTERNS_ANALYSIS.md)
Anti-Pattern Mitigation: #12 (Connection Pooling via shared client)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class CodeOrchestratorProtocol(Protocol):
    """Protocol for Code-Orchestrator-Service client.

    Defines the interface for clients interacting with Code-Orchestrator-Service.
    Enables duck typing for test doubles (FakeCodeOrchestratorClient).

    Methods:
        get_embeddings: Get SBERT embeddings for texts
        get_similarity_matrix: Get pairwise similarity matrix
        cluster_topics: Cluster corpus into topics using BERTopic
        extract_keywords: Extract TF-IDF keywords from corpus
        close: Release HTTP client resources
    """

    async def get_embeddings(self, texts: list[str]) -> NDArray[Any]:
        """Get SBERT embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            NDArray of embeddings, shape (n_texts, embedding_dim)
        """
        ...

    async def get_similarity_matrix(self, texts: list[str]) -> NDArray[Any]:
        """Get pairwise similarity matrix.

        Args:
            texts: List of texts to compare

        Returns:
            NDArray similarity matrix, shape (n_texts, n_texts)
        """
        ...

    async def cluster_topics(
        self, corpus: list[str], chapter_index: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Cluster corpus into topics using BERTopic.

        Args:
            corpus: List of documents to cluster
            chapter_index: List of chapter metadata dicts with book, chapter, title

        Returns:
            Dict with topic_assignments, topic_count, chapter_topic, topics_info
        """
        ...

    async def extract_keywords(
        self, corpus: list[str], top_k: int = 5
    ) -> list[list[str]]:
        """Extract TF-IDF keywords from corpus.

        Args:
            corpus: List of documents to extract keywords from
            top_k: Number of keywords per document

        Returns:
            List of keyword lists, one per document
        """
        ...

    async def close(self) -> None:
        """Release HTTP client resources."""
        ...


@runtime_checkable
class SemanticSearchProtocol(Protocol):
    """Protocol for Semantic-Search-Service client.

    Defines the interface for clients interacting with semantic-search-service.
    Enables duck typing for test doubles (FakeSemanticSearchClient).

    Methods:
        search: Hybrid search across chapters
        get_relationships: Get graph relationships for a chapter
        get_relationships_batch: Get relationships for multiple chapters
        close: Release HTTP client resources
    """

    async def search(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Hybrid search across chapters.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dict with results list, total count, and metadata
        """
        ...

    async def get_relationships(self, chapter_id: str) -> dict[str, Any]:
        """Get graph relationships for a chapter.

        Args:
            chapter_id: ID of the chapter

        Returns:
            Dict with chapter_id and relationships list
        """
        ...

    async def get_relationships_batch(
        self, chapter_ids: list[str]
    ) -> dict[str, Any]:
        """Get relationships for multiple chapters.

        Args:
            chapter_ids: List of chapter IDs

        Returns:
            Dict with results keyed by chapter_id
        """
        ...

    async def close(self) -> None:
        """Release HTTP client resources."""
        ...


@runtime_checkable
class AuditServiceProtocol(Protocol):
    """Protocol for Audit-Service client.

    WBS: MSE-8.1 - Audit Service Protocol
    Defines the interface for clients interacting with audit-service.
    Enables duck typing for test doubles (FakeAuditServiceClient).

    Methods:
        audit_cross_references: Audit code against reference chapters
        close: Release HTTP client resources
    """

    async def audit_cross_references(
        self,
        code: str,
        references: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Audit code against reference chapters using CodeBERT similarity.

        Args:
            code: Source code/content to audit
            references: List of reference chapter dicts with chapter_id, content
            threshold: Similarity threshold for passing audit

        Returns:
            Dict with passed, status, findings, best_similarity
        """
        ...

    async def close(self) -> None:
        """Release HTTP client resources."""
        ...
