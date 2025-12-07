"""Retrieve content node - Step 6 of workflow.

Retrieves full chapter content for matched chapters.

Pattern: LangGraph workflow node
Source: TIER_RELATIONSHIP_DIAGRAM.md Step 6
"""

from typing import Protocol

from src.agents.cross_reference.state import ChapterMatch, CrossReferenceState


class ContentClient(Protocol):
    """Protocol for content client dependency injection."""
    
    async def get_chapter_content(
        self,
        book: str,
        chapter: int,
    ) -> dict | None:
        """Get content for a specific chapter.
        
        Args:
            book: Book title
            chapter: Chapter number
            
        Returns:
            Dict with content, page_range, or None if not found
        """
        ...


# Global client reference for dependency injection
_content_client: ContentClient | None = None


def set_content_client(client: ContentClient) -> None:
    """Set the content client for content retrieval.
    
    Args:
        client: Content client implementing get_chapter_content
    """
    global _content_client
    _content_client = client


def get_content_client() -> ContentClient | None:
    """Get the current content client."""
    return _content_client


async def retrieve_content(state: CrossReferenceState) -> dict:
    """Retrieve content for matched chapters.
    
    This is Step 6 of the workflow:
    "System Retrieves Full Chapter Content"
    
    Fetches the actual text content for chapters identified
    during taxonomy search and graph traversal.
    Also validates matches by checking content exists.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with retrieved_chapters and validated_matches to merge into state
    """
    result: dict = {"current_node": "retrieve_content"}
    
    # Collect all matches from taxonomy search and traversal paths
    all_matches = list(state.taxonomy_matches)
    
    # Add nodes from traversal paths (excluding source nodes which are first)
    for path in state.traversal_paths:
        for i, node in enumerate(path.nodes):
            # Skip first node (source) - it's already in taxonomy_matches
            if i > 0:
                # Create ChapterMatch from GraphNode
                match = ChapterMatch(
                    book=node.book,
                    chapter=node.chapter,
                    title=node.title,
                    tier=node.tier,
                    similarity=0.0,  # Not available from GraphNode
                )
                all_matches.append(match)
    
    # Deduplicate by (book, chapter)
    seen: set[tuple[str, int]] = set()
    unique_matches: list[ChapterMatch] = []
    for match in all_matches:
        key = (match.book, match.chapter)
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    
    # Get content client
    client = get_content_client()
    if client is None:
        result["retrieved_chapters"] = []
        result["validated_matches"] = []
        result["errors"] = state.errors + ["Content client not configured"]
        return result
    
    retrieved_chapters: list[ChapterMatch] = []
    validated_matches: list[ChapterMatch] = []
    
    try:
        for match in unique_matches:
            content_data = await client.get_chapter_content(
                book=match.book,
                chapter=match.chapter,
            )
            
            if content_data is not None:
                # Update match with content
                updated_match = ChapterMatch(
                    book=match.book,
                    chapter=match.chapter,
                    title=match.title,
                    tier=match.tier,
                    similarity=match.similarity,
                    keywords=match.keywords,
                    relevance_reason=match.relevance_reason,
                    content=content_data.get("content"),
                    page_range=content_data.get("page_range"),
                )
                retrieved_chapters.append(updated_match)
                validated_matches.append(updated_match)
        
        result["retrieved_chapters"] = retrieved_chapters
        result["validated_matches"] = validated_matches
        return result
        
    except Exception as e:
        result["retrieved_chapters"] = []
        result["validated_matches"] = []
        result["errors"] = state.errors + [f"Content retrieval failed: {e}"]
        return result
