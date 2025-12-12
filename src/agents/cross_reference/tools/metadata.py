"""get_chapter_metadata tool - Retrieve chapter metadata.

Pattern: LangChain Tool
Source: ARCHITECTURE.md tool definition

WBS 5.7.2: Implemented via semantic-search-service client.
Queries for chapter by book+chapter filter and returns metadata.

Anti-Pattern References (CODING_PATTERNS_ANALYSIS.md):
- #8.1: Real async await (not asyncio.sleep(0) stubs)
"""

import logging
from typing import Any

from src.core.clients.semantic_search import (
    SemanticSearchClient,
    get_semantic_search_client,
)


logger = logging.getLogger(__name__)


async def get_chapter_metadata(
    book: str,
    chapter: int,
) -> dict[str, Any]:
    """Retrieve metadata for a specific chapter.

    Returns keywords, concepts, summary, and page range.
    Uses SemanticSearchClient.hybrid_search filtered by book+chapter.

    Args:
        book: Book title
        chapter: Chapter number

    Returns:
        Dict with chapter metadata
    """
    # Get or create client
    client = get_semantic_search_client()
    if client is None:
        client = SemanticSearchClient(focus_areas=["llm_rag"])

    # Build a targeted query for the specific chapter
    query = f"{book} chapter {chapter}"

    logger.debug(
        "Fetching chapter metadata",
        extra={"book": book, "chapter": chapter},
    )

    try:
        result = await client.hybrid_search(
            query=query,
            limit=5,  # Fetch a few results to find the best match
            focus_areas=["llm_rag"],
        )

        # Find the matching chapter from results
        matched = _find_matching_chapter(result.get("results", []), book, chapter)

        if matched:
            payload = matched.get("payload", {})
            return {
                "book": book,
                "chapter": chapter,
                "title": payload.get("title", ""),
                "keywords": payload.get("keywords", []),
                "concepts": payload.get("concepts", []),
                "summary": payload.get("summary", ""),
                "page_range": payload.get("page_range"),
                "tier": payload.get("tier", 1),
            }

        # Not found - return empty metadata
        return {
            "book": book,
            "chapter": chapter,
            "title": "",
            "keywords": [],
            "concepts": [],
            "summary": "",
            "page_range": None,
            "tier": 1,
        }

    except Exception as e:
        logger.error("Metadata fetch failed", extra={"error": str(e)})
        return {
            "book": book,
            "chapter": chapter,
            "title": "",
            "keywords": [],
            "concepts": [],
            "summary": "",
            "page_range": None,
            "tier": 1,
            "error": str(e),
        }


def _find_matching_chapter(
    results: list[dict[str, Any]],
    book: str,
    chapter: int,
) -> dict[str, Any] | None:
    """Find the best matching chapter from search results.

    Args:
        results: Search results to filter
        book: Target book title
        chapter: Target chapter number

    Returns:
        Best matching result or None if not found
    """
    for result in results:
        payload = result.get("payload", {})
        result_book = payload.get("book", "")
        result_chapter = payload.get("chapter", 0)

        # Check for exact match (case-insensitive book title)
        if result_book.lower() == book.lower() and result_chapter == chapter:
            return result

    # If no exact match, return the first result if it looks relevant
    if results:
        first = results[0]
        payload = first.get("payload", {})
        # Return if book name is similar (contains the target)
        if book.lower() in payload.get("book", "").lower():
            return first

    return None
