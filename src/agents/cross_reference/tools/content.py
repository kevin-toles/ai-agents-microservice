"""get_chapter_text tool - Retrieve full chapter content.

Pattern: LangChain Tool
Source: ARCHITECTURE.md tool definition

WBS 5.7.3: Implemented via semantic-search-service client.
Queries for chapter by book+chapter filter and returns content.

Anti-Pattern References (CODING_PATTERNS_ANALYSIS.md):
- #4.3: Unused params → prefix with underscore
- #8.1: Real async await (not asyncio.sleep(0) stubs)
"""

import logging
from typing import Any

from src.core.clients.semantic_search import (
    SemanticSearchClient,
    get_semantic_search_client,
)


logger = logging.getLogger(__name__)


async def get_chapter_text(
    book: str,
    chapter: int,
    pages: list[int] | None = None,
) -> dict[str, Any]:
    """Retrieve the full text content of a chapter.

    Uses SemanticSearchClient.hybrid_search filtered by book+chapter.

    Args:
        book: Book title
        chapter: Chapter number
        pages: Optional: Specific pages to retrieve (currently ignored)

    Returns:
        Dict with chapter content and metadata
    """
    # Acknowledge pages param for future implementation (Anti-Pattern #4.3)
    _pages = pages  # Will be used for page-level retrieval in future

    # Get or create client
    client = get_semantic_search_client()
    if client is None:
        client = SemanticSearchClient(focus_areas=["llm_rag"])

    # Build a targeted query for the specific chapter
    query = f"{book} chapter {chapter} content"

    logger.debug(
        "Fetching chapter text",
        extra={"book": book, "chapter": chapter, "pages": _pages},
    )

    try:
        result = await client.hybrid_search(
            query=query,
            limit=5,
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
                "content": payload.get("content", ""),
                "page_numbers": payload.get("page_numbers", []),
                "word_count": payload.get("word_count", 0),
            }

        # Not found - return empty content
        return {
            "book": book,
            "chapter": chapter,
            "title": "",
            "content": "",
            "page_numbers": [],
            "word_count": 0,
        }

    except Exception as e:
        logger.error("Content fetch failed", extra={"error": str(e)})
        return {
            "book": book,
            "chapter": chapter,
            "title": "",
            "content": "",
            "page_numbers": [],
            "word_count": 0,
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
        if book.lower() in payload.get("book", "").lower():
            return first

    return None
