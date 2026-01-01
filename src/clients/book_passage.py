"""Book Passage Client.

WBS Reference: WBS-AGT23 Book/JSON Passage Retrieval (AGT23.2-AGT23.8)
Acceptance Criteria:
- AC-23.1: Retrieve passages from enriched book JSON files
- AC-23.2: Query passages via Qdrant vector similarity
- AC-23.3: Cross-reference passages with Neo4j concept nodes
- AC-23.4: Return structured BookPassage with citation metadata
- AC-23.5: Support filtering by book, chapter, concept

Client for retrieving book passages from enriched JSON files,
querying Qdrant for vector similarity search, and cross-referencing
with Neo4j concept nodes.

Pattern: Repository pattern with connection pooling (Anti-Pattern #12)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.schemas.passage_models import (
    BookPassage,
    PassageFilter,
    PassageMetadata,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from neo4j import Driver as Neo4jDriver


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BookPassageClientConfig:
    """Configuration for BookPassageClient.
    
    Attributes:
        qdrant_url: Qdrant server URL
        qdrant_collection: Qdrant collection name for book passages
        books_dir: Path to enriched book JSON files
        neo4j_uri: Neo4j Bolt URI (optional, for concept cross-ref)
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        embedding_model: Embedding model name for query encoding
        timeout: Request timeout in seconds
    """
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "book_passages"
    books_dir: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    timeout: int = 30


# =============================================================================
# Book Passage Client
# =============================================================================


@dataclass
class BookPassageClient:
    """Client for book passage retrieval.
    
    WBS: AGT23.2 - Implement BookPassageClient
    
    Provides access to enriched book passages via:
    - Qdrant vector similarity search (AC-23.2)
    - JSON file lookup by passage ID (AC-23.1)
    - Neo4j concept cross-reference (AC-23.3)
    
    Implements BookPassageClientProtocol for duck typing.
    
    Usage:
        config = BookPassageClientConfig(
            qdrant_url="http://localhost:6333",
            books_dir="/path/to/books/enriched"
        )
        client = BookPassageClient(config)
        async with client:
            passages = await client.search_passages("repository pattern")
    """
    
    config: BookPassageClientConfig
    
    # Internal state
    _qdrant_client: Any = field(default=None, init=False, repr=False)
    _neo4j_driver: Any = field(default=None, init=False, repr=False)
    _embedding_model: Any = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)
    _passage_cache: dict[str, BookPassage] = field(
        default_factory=dict, init=False, repr=False
    )
    
    async def connect(self) -> None:
        """Establish connections to Qdrant and Neo4j.
        
        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            return
        
        try:
            # Initialize Qdrant client
            from qdrant_client import QdrantClient
            
            self._qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.timeout,
            )
            logger.info("Connected to Qdrant: %s", self.config.qdrant_url)
            
            # Initialize embedding model
            try:
                from sentence_transformers import SentenceTransformer
                
                self._embedding_model = SentenceTransformer(
                    self.config.embedding_model
                )
                logger.info("Loaded embedding model: %s", self.config.embedding_model)
            except ImportError:
                logger.warning("sentence-transformers not available, embeddings disabled")
                self._embedding_model = None
            
            # Initialize Neo4j driver if configured
            if self.config.neo4j_password:
                try:
                    from neo4j import GraphDatabase
                    
                    self._neo4j_driver = GraphDatabase.driver(
                        self.config.neo4j_uri,
                        auth=(self.config.neo4j_user, self.config.neo4j_password),
                    )
                    logger.info("Connected to Neo4j: %s", self.config.neo4j_uri)
                except Exception as e:
                    logger.warning("Neo4j connection failed: %s", e)
                    self._neo4j_driver = None
            
            self._connected = True
            
        except Exception as e:
            logger.error("Failed to connect: %s", e)
            raise ConnectionError(f"BookPassageClient connection failed: {e}") from e
    
    async def close(self) -> None:
        """Close connections and release resources."""
        if self._qdrant_client:
            try:
                self._qdrant_client.close()
            except Exception:
                pass
            self._qdrant_client = None
        
        if self._neo4j_driver:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass
            self._neo4j_driver = None
        
        self._embedding_model = None
        self._connected = False
        logger.info("BookPassageClient closed")
    
    async def __aenter__(self) -> "BookPassageClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def health_check(self) -> bool:
        """Verify connection is healthy.
        
        Returns:
            True if connected and healthy, False otherwise
        """
        if not self._connected:
            return False
        
        try:
            # Check Qdrant
            if self._qdrant_client:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._qdrant_client.get_collections,
                )
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False
    
    # =========================================================================
    # AC-23.2: Query passages via Qdrant vector similarity
    # =========================================================================
    
    async def search_passages(
        self,
        query: str,
        top_k: int = 10,
        filters: PassageFilter | None = None,
    ) -> list[BookPassage]:
        """Search passages via Qdrant vector similarity.
        
        AC-23.2: Query passages via Qdrant vector similarity
        
        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            filters: Optional PassageFilter for filtering
            
        Returns:
            List of BookPassage objects sorted by relevance
        """
        if not self._connected:
            await self.connect()
        
        if not self._embedding_model:
            logger.warning("Embedding model not available, returning empty results")
            return []
        
        try:
            # Encode query
            query_vector = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._embedding_model.encode(query).tolist(),
            )
            
            # Build Qdrant filter
            qdrant_filter = None
            if filters and not filters.is_empty():
                qdrant_filter = self._build_qdrant_filter(filters)
            
            # Search Qdrant
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._qdrant_client.query_points(
                    collection_name=self.config.qdrant_collection,
                    query=query_vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                ),
            )
            
            # Convert to BookPassage
            passages = []
            for hit in results.points:
                passage = self._payload_to_passage(hit.payload, hit.score)
                if passage:
                    passages.append(passage)
            
            return passages
            
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []
    
    def _build_qdrant_filter(self, filters: PassageFilter) -> Any:
        """Build Qdrant filter from PassageFilter.
        
        Args:
            filters: PassageFilter with filter criteria
            
        Returns:
            Qdrant Filter object
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        
        if filters.book_id:
            conditions.append(
                FieldCondition(
                    key="book_id",
                    match=MatchValue(value=filters.book_id),
                )
            )
        
        if filters.chapter_number:
            conditions.append(
                FieldCondition(
                    key="chapter_number",
                    match=MatchValue(value=filters.chapter_number),
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def _payload_to_passage(
        self, payload: dict[str, Any], score: float
    ) -> BookPassage | None:
        """Convert Qdrant payload to BookPassage.
        
        Args:
            payload: Qdrant point payload
            score: Similarity score
            
        Returns:
            BookPassage or None if invalid
        """
        try:
            return BookPassage(
                passage_id=payload.get("passage_id", ""),
                book_id=payload.get("book_id", ""),
                book_title=payload.get("book_title", ""),
                author=payload.get("author", ""),
                chapter_number=payload.get("chapter_number", 0),
                chapter_title=payload.get("chapter_title", ""),
                start_page=payload.get("start_page", 0),
                end_page=payload.get("end_page", 0),
                content=payload.get("content", ""),
                concepts=payload.get("concepts", []),
                keywords=payload.get("keywords", []),
                relevance_score=score,
            )
        except Exception as e:
            logger.warning("Failed to parse passage payload: %s", e)
            return None
    
    # =========================================================================
    # AC-23.1: Retrieve passages from enriched book JSON files
    # =========================================================================
    
    async def get_passage_by_id(
        self,
        passage_id: str,
    ) -> BookPassage | None:
        """Get passage by ID from JSON lookup.
        
        AC-23.1: Retrieve passages from enriched book JSON files
        
        Args:
            passage_id: Unique passage identifier
            
        Returns:
            BookPassage if found, None otherwise
        """
        # Check cache first
        if passage_id in self._passage_cache:
            return self._passage_cache[passage_id]
        
        # Parse passage ID to find book
        # Format: {book_id}_ch{N}_p{M} or similar
        parts = passage_id.split("_")
        if len(parts) < 2:
            logger.warning("Invalid passage ID format: %s", passage_id)
            return None
        
        book_id = parts[0]
        
        # Load book JSON and find passage
        book_path = self._find_book_file(book_id)
        if not book_path:
            return None
        
        try:
            book_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: json.loads(book_path.read_text()),
            )
            
            # Search for passage in chapters
            for chapter in book_data.get("chapters", []):
                for passage_data in chapter.get("passages", []):
                    if passage_data.get("passage_id") == passage_id:
                        passage = BookPassage.from_dict(passage_data)
                        self._passage_cache[passage_id] = passage
                        return passage
            
            return None
            
        except Exception as e:
            logger.error("Failed to load passage from JSON: %s", e)
            return None
    
    def _find_book_file(self, book_id: str) -> Path | None:
        """Find enriched book JSON file by ID.
        
        Args:
            book_id: Book identifier
            
        Returns:
            Path to book file or None
        """
        if not self.config.books_dir:
            return None
        
        books_dir = Path(self.config.books_dir)
        if not books_dir.exists():
            return None
        
        # Try common naming patterns
        patterns = [
            f"{book_id}_metadata_enriched.json",
            f"{book_id}_enriched.json",
            f"{book_id}.json",
        ]
        
        for pattern in patterns:
            path = books_dir / pattern
            if path.exists():
                return path
        
        # Try case-insensitive search
        for path in books_dir.glob("*_enriched.json"):
            if book_id.lower() in path.stem.lower():
                return path
        
        return None
    
    # =========================================================================
    # AC-23.3: Cross-reference passages with Neo4j concept nodes
    # =========================================================================
    
    async def get_passages_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[BookPassage]:
        """Get passages linked to a concept via Neo4j.
        
        AC-23.3: Cross-reference passages with Neo4j concept nodes
        
        Uses Neo4j to find chapters covering the concept, then retrieves
        passages from those chapters.
        
        Args:
            concept: Concept identifier (e.g., "ddd", "repository-pattern")
            limit: Maximum results to return
            
        Returns:
            List of BookPassage objects linked to the concept
        """
        if not self._neo4j_driver:
            # Fallback to Qdrant search with concept filter
            logger.info("Neo4j not available, using Qdrant concept search")
            return await self.search_passages(
                concept,
                top_k=limit,
                filters=PassageFilter(concept=concept),
            )
        
        try:
            # Query Neo4j for chapters covering this concept
            cypher = """
            MATCH (c:Concept {name: $concept})<-[:COVERS]-(ch:Chapter)-[:IN_BOOK]->(b:Book)
            RETURN ch.chapter_id AS chapter_id,
                   ch.title AS chapter_title,
                   ch.chapter_number AS chapter_number,
                   b.book_id AS book_id,
                   b.title AS book_title,
                   b.author AS author
            LIMIT $limit
            """
            
            def run_query():
                with self._neo4j_driver.session() as session:
                    result = session.run(cypher, concept=concept, limit=limit)
                    return [dict(record) for record in result]
            
            chapters = await asyncio.get_event_loop().run_in_executor(
                None,
                run_query,
            )
            
            # Load passages from each chapter
            passages = []
            for chapter in chapters:
                book_path = self._find_book_file(chapter["book_id"])
                if book_path:
                    chapter_passages = await self._load_chapter_passages(
                        book_path,
                        chapter["chapter_number"],
                        concept,
                    )
                    passages.extend(chapter_passages)
                    if len(passages) >= limit:
                        break
            
            return passages[:limit]
            
        except Exception as e:
            logger.error("Neo4j concept query failed: %s", e)
            return []
    
    async def _load_chapter_passages(
        self,
        book_path: Path,
        chapter_number: int,
        concept: str,
    ) -> list[BookPassage]:
        """Load passages from a specific chapter.
        
        Args:
            book_path: Path to book JSON file
            chapter_number: Chapter number to load
            concept: Concept to filter by
            
        Returns:
            List of passages from the chapter
        """
        try:
            book_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: json.loads(book_path.read_text()),
            )
            
            passages = []
            for chapter in book_data.get("chapters", []):
                if chapter.get("chapter_number") == chapter_number:
                    for passage_data in chapter.get("passages", []):
                        # Check if passage mentions concept
                        concepts = passage_data.get("concepts", [])
                        if concept.lower() in [c.lower() for c in concepts]:
                            passage = BookPassage.from_dict(passage_data)
                            passages.append(passage)
                    break
            
            return passages
            
        except Exception as e:
            logger.warning("Failed to load chapter passages: %s", e)
            return []
    
    # =========================================================================
    # AC-23.5: Support filtering by book, chapter, concept
    # =========================================================================
    
    async def get_passages_for_book(
        self,
        book_id: str,
        limit: int = 100,
    ) -> list[BookPassage]:
        """Get all passages for a book.
        
        AC-23.5: Support filtering by book
        
        Args:
            book_id: Book identifier
            limit: Maximum results to return
            
        Returns:
            List of BookPassage objects from the book
        """
        book_path = self._find_book_file(book_id)
        if not book_path:
            return []
        
        try:
            book_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: json.loads(book_path.read_text()),
            )
            
            passages = []
            for chapter in book_data.get("chapters", []):
                for passage_data in chapter.get("passages", []):
                    passage = BookPassage.from_dict(passage_data)
                    passages.append(passage)
                    if len(passages) >= limit:
                        return passages
            
            return passages
            
        except Exception as e:
            logger.error("Failed to load book passages: %s", e)
            return []
    
    async def filter_by_book(
        self,
        passages: list[BookPassage],
        book_id: str,
    ) -> list[BookPassage]:
        """Filter passages by book ID.
        
        AC-23.5: Support filtering by book
        
        Args:
            passages: List of passages to filter
            book_id: Book identifier to filter by
            
        Returns:
            Filtered list of passages
        """
        return [p for p in passages if p.book_id == book_id]
    
    async def filter_by_chapter(
        self,
        passages: list[BookPassage],
        chapter_number: int,
    ) -> list[BookPassage]:
        """Filter passages by chapter number.
        
        AC-23.5: Support filtering by chapter
        
        Args:
            passages: List of passages to filter
            chapter_number: Chapter number to filter by
            
        Returns:
            Filtered list of passages
        """
        return [p for p in passages if p.chapter_number == chapter_number]
