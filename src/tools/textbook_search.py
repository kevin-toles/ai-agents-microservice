"""Textbook Search Tool for Cross-Reference.

WBS Reference: WBS-KB8 - Textbook Search Tool
Gap: textbook_search Tool (JSON files not wired as tool)

Provides direct access to textbook JSON files for evidence retrieval.
Complements UnifiedRetriever by providing raw JSON file access.

Anti-Patterns Avoided:
- #12: Cached file loading (singleton pattern)
- S1192: String constants at module level
- S3776: Cognitive complexity < 15 via helper methods
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Module Constants (S1192 Compliance)
# =============================================================================

_DEFAULT_TEXTBOOKS_DIR = Path(__file__).parent.parent.parent.parent.parent / "textbooks" / "JSON Texts"
_DEFAULT_TOP_K = 10
_DEFAULT_MIN_SCORE = 0.3


# =============================================================================
# Data Models
# =============================================================================


class TextbookChapter(BaseModel):
    """A chapter from a textbook."""
    
    chapter_id: str = Field(..., description="Unique chapter identifier")
    chapter_number: int = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    content: str = Field(..., description="Chapter content text")
    book_title: str = Field(..., description="Source book title")
    page_range: str | None = Field(None, description="Page range if available")


class TextbookSearchResult(BaseModel):
    """Result from textbook search."""
    
    chapters: list[TextbookChapter] = Field(default_factory=list)
    query: str = Field(..., description="Original search query")
    total_matches: int = Field(0, description="Total matching chapters")
    books_searched: int = Field(0, description="Number of books searched")


@dataclass(frozen=True, slots=True)
class TextbookMetadata:
    """Metadata about a loaded textbook."""
    
    title: str
    file_path: str
    chapter_count: int
    total_chars: int


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class TextbookSearchProtocol(Protocol):
    """Protocol for textbook search tools."""
    
    async def search(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        book_filter: str | None = None,
    ) -> TextbookSearchResult:
        """Search textbooks for relevant chapters."""
        ...
    
    def get_available_books(self) -> list[TextbookMetadata]:
        """Get list of available textbooks."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TextbookSearchConfig:
    """Configuration for textbook search.
    
    Attributes:
        textbooks_dir: Path to textbook JSON files
        top_k: Default number of results to return
        min_score: Minimum relevance score threshold
        preload_books: Whether to load all books on init
    """
    
    textbooks_dir: Path = field(default_factory=lambda: _DEFAULT_TEXTBOOKS_DIR)
    top_k: int = _DEFAULT_TOP_K
    min_score: float = _DEFAULT_MIN_SCORE
    preload_books: bool = False


# =============================================================================
# TextbookSearchTool Implementation
# =============================================================================


class TextbookSearchTool:
    """Tool for searching textbook JSON files.
    
    Provides direct access to textbook content for cross-reference
    and evidence gathering in the Kitchen Brigade architecture.
    
    Example:
        >>> tool = TextbookSearchTool()
        >>> result = await tool.search("repository pattern")
        >>> for chapter in result.chapters:
        ...     print(f"{chapter.book_title}: {chapter.title}")
    """
    
    def __init__(self, config: TextbookSearchConfig | None = None) -> None:
        """Initialize the textbook search tool.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or TextbookSearchConfig()
        self._books: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, TextbookMetadata] = {}
        
        if self._config.preload_books:
            self._load_all_books()
    
    def _load_all_books(self) -> None:
        """Load all textbook JSON files into memory."""
        textbooks_dir = self._config.textbooks_dir
        
        if not textbooks_dir.exists():
            logger.warning("Textbooks directory not found: %s", textbooks_dir)
            return
        
        for json_file in textbooks_dir.glob("*.json"):
            self._load_book(json_file)
    
    def _load_book(self, file_path: Path) -> dict[str, Any] | None:
        """Load a single textbook JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Parsed JSON content or None if load failed.
        """
        file_key = file_path.stem
        
        # Check cache first
        if file_key in self._books:
            return self._books[file_key]
        
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            
            self._books[file_key] = data
            
            # Extract metadata
            chapters = data.get("chapters", [])
            total_chars = sum(len(ch.get("content", "")) for ch in chapters)
            
            self._metadata[file_key] = TextbookMetadata(
                title=data.get("title", file_key),
                file_path=str(file_path),
                chapter_count=len(chapters),
                total_chars=total_chars,
            )
            
            logger.debug("Loaded textbook: %s (%d chapters)", file_key, len(chapters))
            return data
            
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load textbook %s: %s", file_path, e)
            return None
    
    async def search(
        self,
        query: str,
        top_k: int | None = None,
        book_filter: str | None = None,
    ) -> TextbookSearchResult:
        """Search textbooks for chapters matching the query.
        
        Uses simple keyword matching. For semantic search, use
        UnifiedRetriever with Qdrant integration.
        
        Args:
            query: Search query (keywords)
            top_k: Maximum results to return
            book_filter: Optional book title filter
            
        Returns:
            TextbookSearchResult with matching chapters.
        """
        top_k = top_k or self._config.top_k
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        matches: list[tuple[float, TextbookChapter]] = []
        books_searched = 0
        
        textbooks_dir = self._config.textbooks_dir
        if not textbooks_dir.exists():
            logger.warning("Textbooks directory not found: %s", textbooks_dir)
            return TextbookSearchResult(
                chapters=[],
                query=query,
                total_matches=0,
                books_searched=0,
            )
        
        for json_file in textbooks_dir.glob("*.json"):
            # Apply book filter if specified
            if book_filter and book_filter.lower() not in json_file.stem.lower():
                continue
            
            data = self._load_book(json_file)
            if not data:
                continue
            
            books_searched += 1
            book_title = data.get("title", json_file.stem)
            
            for i, chapter in enumerate(data.get("chapters", [])):
                chapter_title = chapter.get("title", f"Chapter {i + 1}")
                content = chapter.get("content", "")
                
                # Simple keyword matching score
                content_lower = content.lower()
                title_lower = chapter_title.lower()
                
                score = self._calculate_relevance(
                    query_terms, content_lower, title_lower
                )
                
                if score >= self._config.min_score:
                    tc = TextbookChapter(
                        chapter_id=f"{json_file.stem}_ch{i + 1}",
                        chapter_number=i + 1,
                        title=chapter_title,
                        content=content[:2000] + ("..." if len(content) > 2000 else ""),
                        book_title=book_title,
                        page_range=chapter.get("page_range"),
                    )
                    matches.append((score, tc))
        
        # Sort by score and take top_k
        matches.sort(key=lambda x: x[0], reverse=True)
        top_chapters = [ch for _, ch in matches[:top_k]]
        
        return TextbookSearchResult(
            chapters=top_chapters,
            query=query,
            total_matches=len(matches),
            books_searched=books_searched,
        )
    
    def _calculate_relevance(
        self,
        query_terms: set[str],
        content: str,
        title: str,
    ) -> float:
        """Calculate relevance score based on keyword matching.
        
        Args:
            query_terms: Set of query keywords
            content: Chapter content (lowercase)
            title: Chapter title (lowercase)
            
        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not query_terms:
            return 0.0
        
        # Title matches are weighted higher
        title_matches = sum(1 for term in query_terms if term in title)
        content_matches = sum(1 for term in query_terms if term in content)
        
        title_score = title_matches / len(query_terms) * 0.6
        content_score = content_matches / len(query_terms) * 0.4
        
        return min(title_score + content_score, 1.0)
    
    def get_available_books(self) -> list[TextbookMetadata]:
        """Get metadata for all available textbooks.
        
        Returns:
            List of TextbookMetadata for each JSON file.
        """
        # Load all books if not already loaded
        if not self._metadata:
            self._load_all_books()
        
        return list(self._metadata.values())
    
    def get_chapter_by_id(self, chapter_id: str) -> TextbookChapter | None:
        """Get a specific chapter by ID.
        
        Args:
            chapter_id: Chapter ID in format "book_stem_chN"
            
        Returns:
            TextbookChapter or None if not found.
        """
        # Parse chapter_id (e.g., "Building_Microservices_ch5")
        parts = chapter_id.rsplit("_ch", 1)
        if len(parts) != 2:
            return None
        
        book_stem, ch_num_str = parts
        try:
            ch_num = int(ch_num_str) - 1  # 0-indexed
        except ValueError:
            return None
        
        # Find matching file
        textbooks_dir = self._config.textbooks_dir
        for json_file in textbooks_dir.glob("*.json"):
            if json_file.stem == book_stem:
                data = self._load_book(json_file)
                if data:
                    chapters = data.get("chapters", [])
                    if 0 <= ch_num < len(chapters):
                        chapter = chapters[ch_num]
                        return TextbookChapter(
                            chapter_id=chapter_id,
                            chapter_number=ch_num + 1,
                            title=chapter.get("title", f"Chapter {ch_num + 1}"),
                            content=chapter.get("content", ""),
                            book_title=data.get("title", book_stem),
                            page_range=chapter.get("page_range"),
                        )
        
        return None


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeTextbookSearchTool:
    """Fake implementation for testing.
    
    Allows setting predefined responses without actual file I/O.
    """
    
    def __init__(self) -> None:
        """Initialize the fake tool."""
        self._results: dict[str, TextbookSearchResult] = {}
        self._books: list[TextbookMetadata] = []
    
    def set_search_result(self, query: str, result: TextbookSearchResult) -> None:
        """Set a predefined result for a query."""
        self._results[query.lower()] = result
    
    def set_available_books(self, books: list[TextbookMetadata]) -> None:
        """Set available books metadata."""
        self._books = books
    
    async def search(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        book_filter: str | None = None,
    ) -> TextbookSearchResult:
        """Return predefined result or empty result."""
        return self._results.get(
            query.lower(),
            TextbookSearchResult(query=query, chapters=[], total_matches=0, books_searched=0),
        )
    
    def get_available_books(self) -> list[TextbookMetadata]:
        """Return predefined books."""
        return self._books


# =============================================================================
# Singleton Instance
# =============================================================================

_textbook_search_tool: TextbookSearchTool | None = None


def get_textbook_search_tool(
    config: TextbookSearchConfig | None = None,
) -> TextbookSearchTool:
    """Get or create singleton TextbookSearchTool instance.
    
    Args:
        config: Optional configuration. Only used on first call.
        
    Returns:
        Cached TextbookSearchTool instance.
    """
    global _textbook_search_tool
    if _textbook_search_tool is None:
        _textbook_search_tool = TextbookSearchTool(config)
    return _textbook_search_tool
