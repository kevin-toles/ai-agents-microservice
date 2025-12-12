"""Content client adapter for Kitchen Brigade architecture.

Adapts SemanticSearchClient to the ContentClient protocol used by
the retrieve_content node.

Pattern: Adapter Pattern (Gang of Four)
Kitchen Brigade: ai-agents (Expeditor) → semantic-search (Cookbook)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.clients.semantic_search import SemanticSearchClient


class SemanticSearchContentAdapter:
    """Adapter that wraps SemanticSearchClient to match ContentClient protocol.
    
    The ContentClient protocol expects:
        - get_chapter_content(book: str, chapter: int) -> dict | None
        
    The SemanticSearchClient provides:
        - get_chapter_content(book_id: str, chapter_number: int) -> dict | None
        
    This adapter maps the parameter names and adapts the response format.
    
    Attributes:
        client: The underlying SemanticSearchClient
    """
    
    def __init__(self, client: SemanticSearchClient) -> None:
        """Initialize the adapter with a SemanticSearchClient.
        
        Args:
            client: SemanticSearchClient instance to wrap
        """
        self._client = client
    
    async def get_chapter_content(
        self,
        book: str,
        chapter: int,
    ) -> dict | None:
        """Get content for a specific chapter via semantic-search.
        
        Adapts the ContentClient protocol call to SemanticSearchClient.
        
        Args:
            book: Book title (mapped to book_id)
            chapter: Chapter number (mapped to chapter_number)
            
        Returns:
            Dict with content data from Neo4j via semantic-search,
            or None if not found. Response includes:
            - book_id, chapter_number, title, summary, keywords,
              concepts, page_range, found
        """
        result = await self._client.get_chapter_content(
            book_id=book,
            chapter_number=chapter,
        )
        
        if result is None:
            return None
        
        # Map semantic-search response to expected format
        # The retrieve_content node expects: content, page_range
        # semantic-search returns: summary, page_range
        return {
            "content": result.get("summary", ""),
            "page_range": result.get("page_range", ""),
            # Pass through additional fields
            "title": result.get("title", ""),
            "keywords": result.get("keywords", []),
            "concepts": result.get("concepts", []),
        }
