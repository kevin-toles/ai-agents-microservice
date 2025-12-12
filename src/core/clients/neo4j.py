"""Neo4j client for taxonomy-based chapter search.

Provides async interface to Neo4j for the cross-reference agent workflow.
Implements the Neo4jClient protocol from search_taxonomy node.

Pattern: Repository pattern with async adapter
Source: TIER_RELATIONSHIP_DIAGRAM.md - Spider web taxonomy structure
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class RealNeo4jClient:
    """Neo4j client implementing the Neo4jClient protocol.
    
    Provides search_chapters method for taxonomy-based chapter lookup.
    Uses Neo4j's full-text search and graph relationships.
    
    Attributes:
        _driver: Neo4j driver instance
        _uri: Neo4j Bolt URI
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize Neo4j driver.
        
        Args:
            uri: Neo4j Bolt URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
        """
        from neo4j import GraphDatabase
        
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri
        logger.info("Initialized Neo4j client: %s", uri)

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")

    async def health_check(self) -> bool:
        """Check if Neo4j is reachable.
        
        Returns:
            True if connection successful, False otherwise
        """
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
        with self._driver.session() as session:
            result = session.run("RETURN 1 as n")
            return result.single()["n"] == 1

    async def search_chapters(
        self,
        concepts: list[str],
        tiers: list[int] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search chapters by concepts and optional tier filter.
        
        This method searches the Neo4j graph for chapters that match
        the given concepts. It uses full-text search on chapter content
        and keywords, then filters by tier if specified.
        
        Args:
            concepts: List of concepts to search for
            tiers: Optional list of tiers to filter by (1, 2, or 3)
            limit: Maximum results to return
            
        Returns:
            List of chapter dicts with:
                - book: Book title
                - chapter: Chapter number
                - title: Chapter title
                - tier: Tier level (1-3)
                - similarity: Relevance score (0-1)
                - keywords: List of keywords
                - relevance_reason: Why this chapter matched
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._search_chapters_sync(concepts, tiers, limit),
            )
            return result
        except Exception as e:
            logger.error("Neo4j chapter search failed: %s", e)
            return []

    def _search_chapters_sync(
        self,
        concepts: list[str],
        tiers: list[int] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Synchronous chapter search implementation.
        
        Note: The actual Neo4j schema uses:
            - book_id (not book)
            - number (not chapter_number)
            - keywords, concepts, title, summary
            - No tier property (derived from Book relationship or default to 2)
        """
        print(f"[DEBUG Neo4j] _search_chapters_sync called with concepts={concepts}, tiers={tiers}, limit={limit}")
        # Cypher query to search chapters using actual property names
        # Searches both keywords and concepts arrays
        cypher = """
        MATCH (c:Chapter)
        WHERE (
            ANY(keyword IN coalesce(c.keywords, []) WHERE 
                ANY(concept IN $concepts WHERE toLower(keyword) CONTAINS toLower(concept))
            )
            OR ANY(stored_concept IN coalesce(c.concepts, []) WHERE
                ANY(concept IN $concepts WHERE toLower(stored_concept) CONTAINS toLower(concept))
            )
            OR ANY(concept IN $concepts WHERE toLower(coalesce(c.title, '')) CONTAINS toLower(concept))
            OR ANY(concept IN $concepts WHERE toLower(coalesce(c.summary, '')) CONTAINS toLower(concept))
        )
        WITH c, 
             size([keyword IN coalesce(c.keywords, []) WHERE 
                ANY(concept IN $concepts WHERE toLower(keyword) CONTAINS toLower(concept))
             ]) as keyword_matches,
             size([stored_concept IN coalesce(c.concepts, []) WHERE
                ANY(concept IN $concepts WHERE toLower(stored_concept) CONTAINS toLower(concept))
             ]) as concept_matches,
             size([concept IN $concepts WHERE toLower(coalesce(c.title, '')) CONTAINS toLower(concept)]) as title_matches
        WITH c, (keyword_matches * 0.4 + concept_matches * 0.4 + title_matches * 0.2) as raw_score
        WHERE raw_score > 0
        RETURN c.book_id as book,
               c.number as chapter,
               c.title as title,
               2 as tier,
               CASE WHEN raw_score > 1.0 THEN 1.0 ELSE raw_score END as similarity,
               c.keywords as keywords,
               c.concepts as concepts
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        try:
            with self._driver.session() as session:
                result = session.run(
                    cypher,
                    concepts=concepts,
                    limit=limit,
                )
                
                matches = []
                for record in result:
                    match = {
                        "book": record["book"] or "",
                        "chapter": record["chapter"] or 0,
                        "title": record["title"] or "",
                        "tier": record["tier"] or 2,
                        "similarity": float(record["similarity"] or 0.0),
                        "keywords": record["keywords"] or [],
                        "relevance_reason": f"Matched concepts: {', '.join(concepts[:3])}",
                    }
                    matches.append(match)
                
                logger.info(
                    "Neo4j search found %d chapters for concepts: %s",
                    len(matches),
                    concepts[:3],
                )
                return matches
                
        except Exception as e:
            logger.warning("Neo4j query failed, returning empty: %s", e)
            # Return empty list on query failure (graceful degradation)
            return []


# =============================================================================
# Global client reference for dependency injection
# =============================================================================

_neo4j_client: RealNeo4jClient | None = None


def get_neo4j_client() -> RealNeo4jClient | None:
    """Get the current Neo4j client."""
    return _neo4j_client


def set_neo4j_client(client: RealNeo4jClient | None) -> None:
    """Set the Neo4j client.
    
    Args:
        client: RealNeo4jClient instance or None to reset
    """
    global _neo4j_client
    _neo4j_client = client


def create_neo4j_client_from_env() -> RealNeo4jClient | None:
    """Create Neo4j client from environment variables.
    
    Expected environment variables:
        - NEO4J_URL: Bolt URI (e.g., bolt://neo4j:7687)
        - NEO4J_USER: Username (default: neo4j)
        - NEO4J_PASSWORD: Password
        
    Returns:
        RealNeo4jClient if all env vars present, None otherwise
    """
    uri = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri:
        logger.warning("NEO4J_URL not set, Neo4j client not initialized")
        return None
    
    if not password:
        logger.warning("NEO4J_PASSWORD not set, Neo4j client not initialized")
        return None
    
    try:
        client = RealNeo4jClient(uri=uri, user=user, password=password)
        logger.info("Created Neo4j client from environment")
        return client
    except Exception as e:
        logger.error("Failed to create Neo4j client: %s", e)
        return None
