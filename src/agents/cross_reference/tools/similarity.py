"""search_similar tool - Find semantically similar chapters via Qdrant.

Pattern: LangChain Tool
Source: ARCHITECTURE.md tool definition

WBS 5.7: Implemented via semantic-search-service client.
Uses focus_areas=["llm_rag"] for domain-aware filtering to exclude
irrelevant C++/game programming content when searching for LLM topics.
"""

import logging
from typing import Any

from src.core.clients.semantic_search import get_semantic_search_client


logger = logging.getLogger(__name__)


async def search_similar(
    query_text: str,
    top_k: int = 10,
    filter_tier: int | None = None,
    min_similarity: float = 0.7,
    focus_areas: list[str] | None = None,
) -> dict[str, Any]:
    """Find chapters semantically similar to the source content.

    Uses vector search via semantic-search-service with domain filtering.

    Args:
        query_text: Text to find similar content for
        top_k: Number of similar results to return
        filter_tier: Optional: Only return results from this tier
        min_similarity: Minimum similarity threshold 0.0-1.0
        focus_areas: Domain focus areas for filtering (default: ["llm_rag"])

    Returns:
        Dict with similar chapters and similarity scores
    """
    # Get injected client (set via set_semantic_search_client in main.py lifespan)
    client = get_semantic_search_client()
    if client is None:
        raise RuntimeError(
            "SemanticSearchClient not initialized. "
            "Ensure ai-agents service started properly (set_semantic_search_client called)."
        )

    # Use provided focus_areas or default to llm_rag
    effective_focus_areas = focus_areas if focus_areas is not None else ["llm_rag"]

    logger.debug(
        "Searching similar content",
        extra={
            "query": query_text[:50],
            "top_k": top_k,
            "focus_areas": effective_focus_areas,
        },
    )

    try:
        result = await client.search_similar(
            query_text=query_text,
            top_k=top_k,
            filter_tier=filter_tier,
            min_similarity=min_similarity,
            focus_areas=effective_focus_areas,
        )
        return result
    except Exception as e:
        logger.error("Semantic search failed", extra={"error": str(e)})
        return {
            "results": [],
            "total": 0,
            "error": str(e),
        }
