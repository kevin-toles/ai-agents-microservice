"""search_taxonomy tool - Query Neo4j for related books/tiers.

Pattern: LangChain Tool
Source: ARCHITECTURE.md tool definition

WBS 5.7.1: Implemented via semantic-search-service client.
Uses hybrid_search with focus_areas for domain-aware taxonomy search.

Anti-Pattern References (CODING_PATTERNS_ANALYSIS.md):
- #4.3: Unused params → prefix with underscore
- #8.1: Real async await (not asyncio.sleep(0) stubs)
"""

import logging
from typing import Any

from src.core.clients.semantic_search import get_semantic_search_client


logger = logging.getLogger(__name__)


async def search_taxonomy(
    query: str,
    taxonomy_id: str,
    source_tier: int | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search the taxonomy graph for books related to a query.

    Returns books with their tier levels and relationship types.
    Uses SemanticSearchClient.hybrid_search for domain-aware search.

    Args:
        query: Search query (keyword, concept, or topic)
        taxonomy_id: ID of the taxonomy to search within (used as focus_area)
        source_tier: The tier of the source book (for filtering)
        max_results: Maximum number of results to return

    Returns:
        Dict with results containing book, tier, relationship, and relevance
    """
    # Get injected client (set via set_semantic_search_client in main.py lifespan)
    client = get_semantic_search_client()
    if client is None:
        raise RuntimeError(
            "SemanticSearchClient not initialized. "
            "Ensure ai-agents service started properly (set_semantic_search_client called)."
        )

    # Map taxonomy_id to focus_areas
    focus_areas = _map_taxonomy_to_focus_areas(taxonomy_id)

    logger.debug(
        "Searching taxonomy",
        extra={
            "query": query[:50],
            "taxonomy_id": taxonomy_id,
            "focus_areas": focus_areas,
            "max_results": max_results,
        },
    )

    try:
        # Build tier filter if source_tier specified
        tier_filter = [source_tier] if source_tier else None

        result = await client.hybrid_search(
            query=query,
            limit=max_results,
            focus_areas=focus_areas,
            tier_filter=tier_filter,
        )

        return {
            "results": result.get("results", []),
            "total": result.get("total", 0),
            "taxonomy_id": taxonomy_id,
            "query": query,
        }
    except Exception as e:
        logger.error("Taxonomy search failed", extra={"error": str(e)})
        return {
            "results": [],
            "total": 0,
            "taxonomy_id": taxonomy_id,
            "error": str(e),
        }


def _map_taxonomy_to_focus_areas(taxonomy_id: str) -> list[str]:
    """Map taxonomy ID to semantic search focus areas.

    Args:
        taxonomy_id: The taxonomy identifier (e.g., "ai-ml", "python")

    Returns:
        List of focus area strings for domain filtering
    """
    # Map common taxonomy IDs to focus areas
    mapping = {
        "ai-ml": ["llm_rag", "machine_learning"],
        "python": ["python", "software_architecture"],
        "microservices": ["microservices", "distributed_systems"],
    }
    return mapping.get(taxonomy_id, ["llm_rag"])
