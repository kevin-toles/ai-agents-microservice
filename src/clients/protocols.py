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


@runtime_checkable
class CodeReferenceProtocol(Protocol):
    """Protocol for Code Reference Engine client.

    WBS: WBS-AGT21 - Code Reference Engine Client
    Defines the interface for clients wrapping CodeReferenceEngine from ai-platform-data.
    Enables duck typing for test doubles (FakeCodeReferenceClient).

    Acceptance Criteria:
        AC-21.1: Client wraps CodeReferenceEngine from ai-platform-data
        AC-21.2: Async interface for search, get_metadata, fetch_file
        AC-21.3: Integration with Qdrant for semantic code search
        AC-21.4: Integration with GitHub API for on-demand file retrieval
        AC-21.5: Returns CodeContext with citations for downstream

    Methods:
        search: Semantic search across code repositories
        search_by_concept: Search by concept name (e.g., "event-driven")
        search_by_pattern: Search by design pattern (e.g., "repository")
        get_metadata: Get repository metadata by ID
        fetch_file: Fetch file content from GitHub
        close: Release HTTP client resources
    """

    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        concepts: list[str] | None = None,
        top_k: int = 10,
    ) -> Any:
        """Semantic search across code repositories.

        Uses 3-layer retrieval: Qdrant semantic → GitHub API → Neo4j graph.

        Args:
            query: Natural language search query
            domains: Optional list of domains to filter (e.g., ["backend-frameworks"])
            concepts: Optional list of concepts to filter (e.g., ["ddd", "cqrs"])
            top_k: Maximum number of results to return

        Returns:
            CodeContext with primary_references, domains_searched, citations
        """
        ...

    async def search_by_concept(
        self,
        concept: str,
        top_k: int = 10,
    ) -> Any:
        """Search by concept name.

        Args:
            concept: Concept name (e.g., "event-driven", "microservices")
            top_k: Maximum number of results to return

        Returns:
            CodeContext with matching code references
        """
        ...

    async def search_by_pattern(
        self,
        pattern: str,
        top_k: int = 10,
    ) -> Any:
        """Search by design pattern name.

        Args:
            pattern: Design pattern name (e.g., "repository", "saga", "cqrs")
            top_k: Maximum number of results to return

        Returns:
            CodeContext with matching code references
        """
        ...

    async def get_metadata(self, repo_id: str) -> dict[str, Any] | None:
        """Get repository metadata by ID.

        Args:
            repo_id: Repository identifier

        Returns:
            Dict with id, name, domain, concepts, patterns, tags, or None if not found
        """
        ...

    async def fetch_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str | None:
        """Fetch file content from GitHub.

        Args:
            file_path: Path to file within repository
            start_line: Optional start line for partial fetch
            end_line: Optional end line for partial fetch

        Returns:
            File content as string, or None if not found
        """
        ...

    async def close(self) -> None:
        """Release HTTP client resources."""
        ...


@runtime_checkable
class Neo4jClientProtocol(Protocol):
    """Protocol for Neo4j Graph Client.

    WBS: WBS-AGT22 - Neo4j Graph Integration
    Defines the interface for clients connecting to Neo4j for graph traversal.
    Enables duck typing for test doubles (FakeNeo4jClient).

    Acceptance Criteria:
        AC-22.1: Client connects to Neo4j for graph traversal
        AC-22.2: Query book → chapter → concept relationships
        AC-22.3: Query concept → code-reference-engine file mappings
        AC-22.4: Query cross-repo pattern relationships
        AC-22.5: Results include metadata for citation generation

    Reference: TIER_RELATIONSHIP_DIAGRAM.md - Spider web taxonomy structure

    Methods:
        connect: Establish connection to Neo4j
        close: Close connection and release resources
        health_check: Verify connection is healthy
        get_concepts_for_chapter: Get concepts linked to a chapter
        get_code_for_concept: Get code files implementing a concept
        get_related_patterns: Get cross-repo pattern relationships
        get_chapters_for_concept: Get chapters covering a concept
    """

    async def connect(self) -> None:
        """Establish connection to Neo4j.

        Raises:
            Neo4jConnectionError: If connection fails
        """
        ...

    async def close(self) -> None:
        """Close connection and release resources."""
        ...

    async def health_check(self) -> bool:
        """Verify connection is healthy.

        Returns:
            True if connected and healthy, False otherwise
        """
        ...

    async def get_concepts_for_chapter(
        self,
        chapter_id: str,
    ) -> list[Any]:
        """Get concepts linked to a chapter.

        AC-22.2: Query book → chapter → concept relationships

        Args:
            chapter_id: Unique chapter identifier

        Returns:
            List of Concept objects linked to the chapter
        """
        ...

    async def get_code_for_concept(
        self,
        concept: str,
    ) -> list[Any]:
        """Get code file references for a concept.

        AC-22.3: Query concept → code-reference-engine file mappings

        Args:
            concept: Concept identifier (e.g., "repository-pattern")

        Returns:
            List of CodeFileReference objects
        """
        ...

    async def get_related_patterns(
        self,
        pattern: str,
    ) -> list[Any]:
        """Get cross-repo pattern relationships.

        AC-22.4: Query cross-repo pattern relationships

        Args:
            pattern: Design pattern name (e.g., "saga")

        Returns:
            List of PatternRelationship objects
        """
        ...

    async def get_chapters_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chapters covering a concept.

        Args:
            concept: Concept name or identifier
            limit: Maximum results to return

        Returns:
            List of chapter dicts with chapter_id, title, book_id, tier
        """
        ...

    async def search_chapters(
        self,
        concepts: list[str],
        tiers: list[int] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search chapters by concepts and optional tier filter.

        PCON-4: Consolidated method for LangGraph agent compatibility.
        Searches the Neo4j graph for chapters matching the given concepts.

        Args:
            concepts: List of concepts to search for
            tiers: Optional list of tiers to filter by (1, 2, or 3)
            limit: Maximum results to return

        Returns:
            List of chapter dicts with:
                - book: Book identifier
                - chapter: Chapter number
                - title: Chapter title
                - tier: Tier level (1-3)
                - similarity: Relevance score (0-1)
                - keywords: List of keywords
                - relevance_reason: Why this chapter matched
        """
        ...


@runtime_checkable
class BookPassageClientProtocol(Protocol):
    """Protocol for Book Passage Client.

    WBS: WBS-AGT23 - Book/JSON Passage Retrieval
    Defines the interface for clients retrieving passages from enriched book JSON files.
    Enables duck typing for test doubles (FakeBookPassageClient).

    Acceptance Criteria:
        AC-23.1: Retrieve passages from enriched book JSON files
        AC-23.2: Query passages via Qdrant vector similarity
        AC-23.3: Cross-reference passages with Neo4j concept nodes
        AC-23.4: Return structured BookPassage with citation metadata
        AC-23.5: Support filtering by book, chapter, concept

    Methods:
        connect: Establish connection to Qdrant/storage
        close: Close connection and release resources
        health_check: Verify connection is healthy
        search_passages: Search passages via Qdrant vector similarity
        get_passage_by_id: Get passage by ID from JSON lookup
        get_passages_for_concept: Get passages linked via Neo4j
        get_passages_for_book: Get all passages for a book
        filter_by_book: Filter passages by book ID
        filter_by_chapter: Filter passages by chapter number
    """

    async def connect(self) -> None:
        """Establish connection to Qdrant and storage.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def close(self) -> None:
        """Close connection and release resources."""
        ...

    async def health_check(self) -> bool:
        """Verify connection is healthy.

        Returns:
            True if connected and healthy, False otherwise
        """
        ...

    async def search_passages(
        self,
        query: str,
        top_k: int = 10,
        filters: Any = None,
    ) -> list[Any]:
        """Search passages via Qdrant vector similarity.

        AC-23.2: Query passages via Qdrant vector similarity

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            filters: Optional PassageFilter for filtering

        Returns:
            List of BookPassage objects sorted by relevance
        """
        ...

    async def get_passage_by_id(
        self,
        passage_id: str,
    ) -> Any | None:
        """Get passage by ID from JSON lookup.

        AC-23.1: Retrieve passages from enriched book JSON files

        Args:
            passage_id: Unique passage identifier

        Returns:
            BookPassage if found, None otherwise
        """
        ...

    async def get_passages_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[Any]:
        """Get passages linked to a concept via Neo4j.

        AC-23.3: Cross-reference passages with Neo4j concept nodes

        Args:
            concept: Concept identifier (e.g., "ddd", "repository-pattern")
            limit: Maximum results to return

        Returns:
            List of BookPassage objects linked to the concept
        """
        ...

    async def get_passages_for_book(
        self,
        book_id: str,
        limit: int = 100,
    ) -> list[Any]:
        """Get all passages for a book.

        AC-23.5: Support filtering by book

        Args:
            book_id: Book identifier
            limit: Maximum results to return

        Returns:
            List of BookPassage objects from the book
        """
        ...

    async def filter_by_book(
        self,
        passages: list[Any],
        book_id: str,
    ) -> list[Any]:
        """Filter passages by book ID.

        AC-23.5: Support filtering by book

        Args:
            passages: List of passages to filter
            book_id: Book identifier to filter by

        Returns:
            Filtered list of passages
        """
        ...

    async def filter_by_chapter(
        self,
        passages: list[Any],
        chapter_number: int,
    ) -> list[Any]:
        """Filter passages by chapter number.

        AC-23.5: Support filtering by chapter

        Args:
            passages: List of passages to filter
            chapter_number: Chapter number to filter by

        Returns:
            Filtered list of passages
        """
        ...
