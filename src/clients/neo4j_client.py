"""Neo4j Client for Graph Integration.

WBS Reference: WBS-AGT22 Neo4j Graph Integration (AGT22.2)
Acceptance Criteria:
- AC-22.1: Client connects to Neo4j for graph traversal
- AC-22.2: Query book → chapter → concept relationships
- AC-22.3: Query concept → code-reference-engine file mappings
- AC-22.4: Query cross-repo pattern relationships
- AC-22.5: Results include metadata for citation generation

Pattern: Repository pattern with async adapter
Anti-Pattern Mitigation: #12 (Connection Pooling via single driver)

Reference: TIER_RELATIONSHIP_DIAGRAM.md - Spider web taxonomy structure
Reference: ai-platform-data/docker/neo4j/init-scripts/ - Schema definitions
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Self

from src.clients.protocols import Neo4jClientProtocol
from src.schemas.graph_models import (
    CodeFileReference,
    Concept,
    PatternRelationship,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Neo4jClientConfig:
    """Configuration for Neo4jClient.
    
    Attributes:
        uri: Neo4j Bolt URI (e.g., "bolt://localhost:7687")
        user: Neo4j username
        password: Neo4j password
        database: Database name (default: "neo4j")
        max_connection_pool_size: Maximum connections in pool
        connection_timeout: Connection timeout in seconds
    """
    
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> Neo4jClientConfig:
        """Create config from environment variables.
        
        Environment Variables:
            NEO4J_URI: Bolt URI
            NEO4J_USER: Username
            NEO4J_PASSWORD: Password
            NEO4J_DATABASE: Database name
            NEO4J_MAX_POOL_SIZE: Max pool size
            NEO4J_TIMEOUT: Connection timeout
        
        Returns:
            Neo4jClientConfig instance populated from environment.
        """
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=int(os.environ.get("NEO4J_MAX_POOL_SIZE", "50")),
            connection_timeout=float(os.environ.get("NEO4J_TIMEOUT", "30.0")),
        )


# =============================================================================
# Client Implementation
# =============================================================================


class Neo4jClient:
    """Neo4j client for graph traversal.
    
    Provides async interface to Neo4j for querying book/chapter/concept
    relationships and cross-repo pattern mappings.
    
    WBS: WBS-AGT22 - Neo4j Graph Integration
    Pattern: Repository pattern with connection pooling
    
    Usage:
        config = Neo4jClientConfig.from_env()
        async with Neo4jClient(config) as client:
            concepts = await client.get_concepts_for_chapter("ch_001")
    """
    
    def __init__(self, config: Neo4jClientConfig) -> None:
        """Initialize client.
        
        Args:
            config: Neo4jClientConfig with connection settings
        """
        self._config = config
        self._driver: Any = None
    
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection to Neo4j.
        
        Creates driver instance with connection pooling.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            from neo4j import GraphDatabase
            
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
                max_connection_pool_size=self._config.max_connection_pool_size,
            )
            
            # Verify connectivity
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._driver.verify_connectivity,
            )
            
            logger.info("Connected to Neo4j: %s", self._config.uri)
            
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise ConnectionError(f"Neo4j connection failed: {e}") from e
    
    async def close(self) -> None:
        """Close connection and release resources."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    async def health_check(self) -> bool:
        """Verify connection is healthy.
        
        Returns:
            True if connected and healthy, False otherwise
        """
        if not self._driver:
            return False
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._health_check_sync,
            )
            return result
        except Exception as e:
            logger.warning("Neo4j health check failed: %s", e)
            return False
    
    def _health_check_sync(self) -> bool:
        """Synchronous health check."""
        with self._driver.session(database=self._config.database) as session:
            result = session.run("RETURN 1 as n")
            record = result.single()
            return record is not None and record["n"] == 1
    
    # =========================================================================
    # AC-22.2: book → chapter → concept relationships
    # =========================================================================
    
    async def get_concepts_for_chapter(
        self,
        chapter_id: str,
    ) -> list[Concept]:
        """Get concepts linked to a chapter.
        
        Traverses: Chapter → COVERS → Concept
        
        Args:
            chapter_id: Unique chapter identifier
        
        Returns:
            List of Concept objects linked to the chapter
        """
        cypher = """
        MATCH (ch:Chapter {chapter_id: $chapter_id})-[:COVERS]->(c:Concept)
        RETURN c.concept_id AS concept_id,
               c.name AS name,
               coalesce(c.tier, 2) AS tier,
               coalesce(c.keywords, []) AS keywords,
               coalesce(c.description, '') AS description
        """
        
        records = await self._run_query(cypher, {"chapter_id": chapter_id})
        
        return [
            Concept(
                concept_id=r["concept_id"],
                name=r["name"],
                tier=r["tier"],
                keywords=list(r["keywords"]),
                description=r["description"],
            )
            for r in records
        ]
    
    async def get_chapters_for_concept(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chapters covering a concept.
        
        Traverses: Concept ← COVERS ← Chapter
        
        Args:
            concept: Concept name or identifier
            limit: Maximum results to return
        
        Returns:
            List of chapter dicts with chapter_id, title, book_id, tier
        """
        cypher = """
        MATCH (c:Concept)-[:COVERED_BY]-(ch:Chapter)
        WHERE c.concept_id = $concept OR toLower(c.name) CONTAINS toLower($concept)
        OPTIONAL MATCH (ch)-[:PART_OF]->(b:Book)
        RETURN ch.chapter_id AS chapter_id,
               ch.title AS title,
               coalesce(b.book_id, 'unknown') AS book_id,
               coalesce(ch.tier, 2) AS tier
        LIMIT $limit
        """
        
        records = await self._run_query(cypher, {"concept": concept, "limit": limit})
        
        return [
            {
                "chapter_id": r["chapter_id"],
                "title": r["title"],
                "book_id": r["book_id"],
                "tier": r["tier"],
            }
            for r in records
        ]
    
    # =========================================================================
    # AC-22.3: concept → code-reference-engine file mappings
    # =========================================================================
    
    async def get_code_for_concept(
        self,
        concept: str,
    ) -> list[CodeFileReference]:
        """Get code file references for a concept.
        
        Traverses: Concept → IMPLEMENTED_BY → CodeFile
        
        Args:
            concept: Concept identifier (e.g., "repository-pattern")
        
        Returns:
            List of CodeFileReference objects
        """
        cypher = """
        MATCH (c:Concept {concept_id: $concept})-[:IMPLEMENTED_BY]->(f:CodeFile)
        RETURN f.file_path AS file_path,
               f.repo_id AS repo_id,
               f.start_line AS start_line,
               f.end_line AS end_line,
               coalesce(f.language, 'text') AS language,
               f.github_url AS github_url
        """
        
        records = await self._run_query(cypher, {"concept": concept})
        
        return [
            CodeFileReference(
                file_path=r["file_path"],
                repo_id=r["repo_id"],
                start_line=r["start_line"] or 1,
                end_line=r["end_line"] or 1,
                language=r["language"],
                github_url=r["github_url"] or self._build_github_url(
                    r["repo_id"], r["file_path"], r["start_line"], r["end_line"]
                ),
            )
            for r in records
        ]
    
    # =========================================================================
    # AC-22.4: cross-repo pattern relationships
    # =========================================================================
    
    async def get_related_patterns(
        self,
        pattern: str,
    ) -> list[PatternRelationship]:
        """Get cross-repo pattern relationships.
        
        Finds patterns related to the source pattern via:
        - PARALLEL: Same tier level
        - PERPENDICULAR: Adjacent tier ±1
        - SKIP_TIER: Non-adjacent tier ±2+
        
        Args:
            pattern: Design pattern name (e.g., "saga")
        
        Returns:
            List of PatternRelationship objects
        """
        cypher = """
        MATCH (p1:Pattern {name: $pattern})-[r:PARALLEL|PERPENDICULAR|SKIP_TIER]->(p2:Pattern)
        OPTIONAL MATCH (p2)-[:FOUND_IN]->(repo:Repository)
        WITH p1, p2, type(r) AS rel_type, collect(repo.repo_id) AS repos,
             coalesce(r.similarity_score, 0.5) AS similarity
        RETURN p1.name AS source_pattern,
               p2.name AS related_pattern,
               rel_type AS relationship_type,
               repos,
               similarity AS similarity_score
        """
        
        records = await self._run_query(cypher, {"pattern": pattern})
        
        return [
            PatternRelationship(
                source_pattern=r["source_pattern"],
                related_pattern=r["related_pattern"],
                relationship_type=r["relationship_type"],
                repos=[repo for repo in r["repos"] if repo],
                similarity_score=r["similarity_score"],
            )
            for r in records
        ]
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read query and return results.
        
        Args:
            cypher: Cypher query string
            parameters: Optional query parameters
        
        Returns:
            List of records as dictionaries
        """
        if not self._driver:
            raise ConnectionError("Not connected to Neo4j")
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_query_sync(cypher, parameters or {}),
            )
            return result
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
    
    def _run_query_sync(
        self,
        cypher: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronous query execution."""
        with self._driver.session(database=self._config.database) as session:
            result = session.run(cypher, parameters)
            return [dict(record) for record in result]
    
    def _build_github_url(
        self,
        repo_id: str,
        file_path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> str:
        """Build GitHub URL with line anchors.
        
        Args:
            repo_id: Repository identifier
            file_path: Path to file
            start_line: Starting line
            end_line: Ending line
        
        Returns:
            GitHub URL with #L{start}-L{end} anchor
        """
        # Default to code-reference-engine org
        base_url = f"https://github.com/code-reference-engine/{repo_id}/blob/main/{file_path}"
        
        if start_line and end_line:
            return f"{base_url}#L{start_line}-L{end_line}"
        elif start_line:
            return f"{base_url}#L{start_line}"
        return base_url


# Type alias for protocol compliance verification
_: type[Neo4jClientProtocol] = Neo4jClient  # type: ignore[assignment]
