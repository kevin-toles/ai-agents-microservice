"""Search taxonomy node - Step 2-4 of workflow.

Searches the taxonomy for books/chapters related to source concepts.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Steps 2-4
"""

from typing import Protocol

from src.agents.cross_reference.state import ChapterMatch, CrossReferenceState


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


def set_neo4j_client(client: Neo4jClient) -> None:
    """Set the Neo4j client for taxonomy search.
    
    Args:
        client: Neo4j client implementing search_chapters
    """
    global _neo4j_client
    _neo4j_client = client


def get_neo4j_client() -> Neo4jClient | None:
    """Get the current Neo4j client."""
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
    if not concepts:
        result["taxonomy_matches"] = []
        return result
    
    # Get Neo4j client
    client = get_neo4j_client()
    if client is None:
        result["taxonomy_matches"] = []
        result["errors"] = state.errors + ["Neo4j client not configured"]
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
        raw_matches = await client.search_chapters(
            concepts=concepts,
            tiers=tiers,
            limit=20,  # Get more than we need for filtering
        )
        
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
        result["errors"] = state.errors + [f"Taxonomy search failed: {e}"]
        return result
