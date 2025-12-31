"""Search taxonomy node - Step 2-4 of workflow.

Searches the taxonomy for books/chapters related to source concepts.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Steps 2-4
"""

import logging
from typing import Protocol

from src.agents.cross_reference.state import ChapterMatch, CrossReferenceState


logger = logging.getLogger(__name__)


class Neo4jClient(Protocol):
    """Protocol for Neo4j client dependency injection."""

    async def search_chapters(
        self,
        concepts: list[str],
        tiers: list[int] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search chapters by concepts and optional tier filter.

        Args:
            concepts: List of concepts to search for
            tiers: Optional list of tiers to filter by
            limit: Maximum results to return

        Returns:
            List of chapter dicts with book, chapter, title, tier, similarity, keywords
        """
        ...


# Global client reference for dependency injection
_neo4j_client: Neo4jClient | None = None


def set_neo4j_client(client: Neo4jClient | None) -> None:
    """Set the Neo4j client for taxonomy search.

    Args:
        client: Neo4j client implementing search_chapters, or None to reset
    """
    global _neo4j_client
    _neo4j_client = client


def get_neo4j_client() -> Neo4jClient | None:
    """Get the current Neo4j client.

    If no client is set, attempts to create one from environment variables.
    This handles multi-worker scenarios where lifespan may not propagate.
    """
    global _neo4j_client
    if _neo4j_client is None:
        # Lazy initialization for multi-worker support
        try:
            from src.core.clients.neo4j import create_neo4j_client_from_env
            _neo4j_client = create_neo4j_client_from_env()
            if _neo4j_client:
                logger.info("Neo4j client initialized lazily in worker")
        except Exception as e:
            logger.warning("Failed to lazily initialize Neo4j client: %s", e)
    return _neo4j_client


async def search_taxonomy(state: CrossReferenceState) -> dict:
    """Search taxonomy for related content.

    This covers Steps 2-4 of the 9-step workflow:
    - Step 2: Review Taxonomy Structure
    - Step 3: Review Companion Book Metadata
    - Step 4: Cross-Reference Keywords & Match Concepts

    Uses Neo4j to search for chapters matching the analyzed concepts.
    Respects tier configuration from traversal config.

    Args:
        state: Current workflow state

    Returns:
        Dict with taxonomy_matches to merge into state
    """
    result: dict = {"current_node": "search_taxonomy"}

    # Check for concepts to search
    concepts = state.analyzed_concepts
    print(f"[DEBUG search_taxonomy] analyzed_concepts from state: {concepts}")
    logger.info("search_taxonomy called with analyzed_concepts: %s", concepts)
    if not concepts:
        result["taxonomy_matches"] = []
        return result

    # Get Neo4j client
    client = get_neo4j_client()
    print(f"[DEBUG search_taxonomy] Neo4j client: {'present' if client else 'None'}")
    logger.info("Neo4j client from get_neo4j_client(): %s", "present" if client else "None")
    if client is None:
        result["taxonomy_matches"] = []
        result["errors"] = [*state.errors, "Neo4j client not configured"]
        return result

    # Determine tier filter from config
    tiers: list[int] | None = None
    config = state.config
    enabled_tiers = []
    if config.include_tier1:
        enabled_tiers.append(1)
    if config.include_tier2:
        enabled_tiers.append(2)
    if config.include_tier3:
        enabled_tiers.append(3)

    # Only apply filter if not all tiers enabled
    if len(enabled_tiers) < 3:
        tiers = enabled_tiers if enabled_tiers else None

    try:
        # Search Neo4j
        print(f"[DEBUG search_taxonomy] Calling search_chapters with concepts={concepts}, tiers={tiers}")
        logger.info("Calling client.search_chapters with concepts=%s, tiers=%s, limit=20", concepts, tiers)
        raw_matches = await client.search_chapters(
            concepts=concepts,
            tiers=tiers,
            limit=20,  # Get more than we need for filtering
        )
        print(f"[DEBUG search_taxonomy] Raw matches from Neo4j: {len(raw_matches)}")
        logger.info("Raw matches from Neo4j: %d", len(raw_matches))

        # Convert to ChapterMatch models
        matches = []
        for raw in raw_matches:
            match = ChapterMatch(
                book=raw.get("book", ""),
                chapter=raw.get("chapter", 0),
                title=raw.get("title", ""),
                tier=raw.get("tier", 1),
                similarity=raw.get("similarity", 0.0),
                keywords=raw.get("keywords", []),
                relevance_reason=raw.get("relevance_reason", ""),
            )
            matches.append(match)

        result["taxonomy_matches"] = matches
        return result

    except Exception as e:
        result["taxonomy_matches"] = []
        result["errors"] = [*state.errors, f"Taxonomy search failed: {e}"]
        return result
