"""
Book Context Lookup - Search for term occurrences across book data sources.

POC Implementation: Searches local JSON files directly.
Future: Will be refactored to use AI Platform internal APIs/database.

Data Sources:
1. RAW JSONs      - Full chapter text (books/raw/)
2. METADATA JSONs - Basic book/chapter info (books/metadata/)
3. ENRICHED JSONs - LLM summaries, concepts, keywords (books/enriched/)

Usage:
    from src.context.book_context import BookContextLookup
    
    lookup = BookContextLookup()
    results = lookup.find_term_context("microservice", max_results=5)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# POC: Direct file paths (will be replaced with API calls to ai-platform-data)
AI_PLATFORM_DATA_PATH = Path(os.getenv(
    "AI_PLATFORM_DATA_PATH",
    "/Users/kevintoles/POC/ai-platform-data"
))

RAW_BOOKS_PATH = AI_PLATFORM_DATA_PATH / "books" / "raw"
METADATA_PATH = AI_PLATFORM_DATA_PATH / "books" / "metadata"
ENRICHED_PATH = AI_PLATFORM_DATA_PATH / "books" / "enriched"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TermOccurrence:
    """A single occurrence of a term in the book corpus."""
    
    term: str
    book_title: str
    chapter_number: int | None
    chapter_title: str | None
    source_type: str  # "raw_text", "summary", "concept_list", "keyword_list"
    context_snippet: str  # The surrounding text
    relevance_score: float = 1.0  # For ranking results


@dataclass 
class TermContext:
    """Full context for a disputed term."""
    
    term: str
    occurrences: list[TermOccurrence] = field(default_factory=list)
    books_found_in: int = 0
    chapters_found_in: int = 0
    appears_in_concept_lists: bool = False
    appears_in_keyword_lists: bool = False
    appears_in_summaries: bool = False
    appears_in_raw_text: bool = False


# =============================================================================
# Book Context Lookup (POC - File-based)
# =============================================================================


class BookContextLookup:
    """Search for term context across book data sources.
    
    POC Implementation: Reads JSON files directly from disk.
    
    TODO (Future AI Platform Integration):
        - Replace file reads with ai-platform-data API calls
        - Use database queries instead of in-memory search
        - Add caching layer for frequently accessed books
        - Support vector similarity search via semantic-search-service
    """
    
    def __init__(
        self,
        raw_path: Path = RAW_BOOKS_PATH,
        metadata_path: Path = METADATA_PATH,
        enriched_path: Path = ENRICHED_PATH,
        cache_books: bool = True,
    ):
        """Initialize the lookup.
        
        Args:
            raw_path: Path to raw book JSONs.
            metadata_path: Path to metadata JSONs.
            enriched_path: Path to enriched JSONs.
            cache_books: Whether to cache loaded books in memory.
        """
        self.raw_path = raw_path
        self.metadata_path = metadata_path
        self.enriched_path = enriched_path
        self.cache_books = cache_books
        
        # In-memory cache (POC only - will use Redis/DB in production)
        self._enriched_cache: dict[str, dict] = {}
        self._raw_cache: dict[str, dict] = {}
        
        # Book index for fast lookup
        self._book_index: dict[str, Path] = {}
        self._build_book_index()
    
    def _build_book_index(self) -> None:
        """Build index of available books."""
        if self.enriched_path.exists():
            for f in self.enriched_path.glob("*.json"):
                title = f.stem
                self._book_index[title.lower()] = f
        
        logger.info(f"Indexed {len(self._book_index)} books")
    
    def _load_enriched_book(self, path: Path) -> dict | None:
        """Load an enriched book JSON."""
        if self.cache_books and str(path) in self._enriched_cache:
            return self._enriched_cache[str(path)]
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            if self.cache_books:
                self._enriched_cache[str(path)] = data
            
            return data
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None
    
    def _load_raw_book(self, title: str) -> dict | None:
        """Load a raw book JSON by title."""
        # Try to find matching raw file
        raw_file = self.raw_path / f"{title}.json"
        
        if not raw_file.exists():
            return None
        
        if self.cache_books and str(raw_file) in self._raw_cache:
            return self._raw_cache[str(raw_file)]
        
        try:
            with open(raw_file) as f:
                data = json.load(f)
            
            if self.cache_books:
                self._raw_cache[str(raw_file)] = data
            
            return data
        except Exception as e:
            logger.error(f"Error loading raw book {raw_file}: {e}")
            return None
    
    def _extract_context_snippet(
        self,
        text: str,
        term: str,
        context_chars: int = 150,
    ) -> str:
        """Extract a snippet of text around the term occurrence."""
        # Case-insensitive search
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(text)
        
        if not match:
            return ""
        
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        
        snippet = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def _term_matches(self, text: str, term: str, require_word_boundary: bool = True) -> bool:
        """Check if term appears in text with optional word boundary matching.
        
        For short terms (<4 chars) or known acronyms, uses word boundaries
        to avoid false positives like "agi" matching "managing".
        
        Args:
            text: Text to search in.
            term: Term to find.
            require_word_boundary: If True, require term to be a distinct word.
        
        Returns:
            True if term found in text.
        """
        text_lower = text.lower()
        term_lower = term.lower()
        
        # For short terms or if requested, use word boundaries
        if require_word_boundary and (len(term) < 4 or term.upper() == term):
            # Use regex word boundaries for short terms and acronyms
            pattern = re.compile(r'\b' + re.escape(term_lower) + r'\b', re.IGNORECASE)
            return bool(pattern.search(text_lower))
        
        return term_lower in text_lower
    
    def find_term_context(
        self,
        term: str,
        max_results: int = 10,
        search_raw_text: bool = True,
        search_summaries: bool = True,
        search_concept_lists: bool = True,
        search_keyword_lists: bool = True,
    ) -> TermContext:
        """Find all occurrences of a term across the book corpus.
        
        Args:
            term: The term to search for.
            max_results: Maximum occurrences to return.
            search_raw_text: Whether to search full chapter text.
            search_summaries: Whether to search chapter summaries.
            search_concept_lists: Whether to search concept lists.
            search_keyword_lists: Whether to search keyword lists.
        
        Returns:
            TermContext with all found occurrences.
        """
        context = TermContext(term=term)
        books_with_term = set()
        chapters_with_term = set()
        
        term_lower = term.lower()
        # Use word boundaries for short terms to avoid false positives
        use_word_boundary = len(term) < 5 or term.upper() == term
        
        # Search enriched books
        for title, path in self._book_index.items():
            book_data = self._load_enriched_book(path)
            if not book_data:
                continue
            
            book_title = book_data.get("metadata", {}).get("title", path.stem)
            
            for chapter in book_data.get("chapters", []):
                chapter_num = chapter.get("chapter_number")
                chapter_title = chapter.get("title", "")
                
                # Search summaries
                if search_summaries:
                    summary = chapter.get("summary", "")
                    if self._term_matches(summary, term, use_word_boundary):
                        context.appears_in_summaries = True
                        books_with_term.add(book_title)
                        chapters_with_term.add(f"{book_title}:{chapter_num}")
                        
                        if len(context.occurrences) < max_results:
                            context.occurrences.append(TermOccurrence(
                                term=term,
                                book_title=book_title,
                                chapter_number=chapter_num,
                                chapter_title=chapter_title,
                                source_type="summary",
                                context_snippet=self._extract_context_snippet(summary, term),
                                relevance_score=0.9,  # Summaries are high value
                            ))
                
                # Search concept lists
                if search_concept_lists:
                    concepts = chapter.get("concepts", [])
                    if any(self._term_matches(c, term, use_word_boundary) for c in concepts):
                        context.appears_in_concept_lists = True
                        books_with_term.add(book_title)
                        chapters_with_term.add(f"{book_title}:{chapter_num}")
                        
                        matching_concepts = [c for c in concepts if self._term_matches(c, term, use_word_boundary)]
                        if len(context.occurrences) < max_results:
                            context.occurrences.append(TermOccurrence(
                                term=term,
                                book_title=book_title,
                                chapter_number=chapter_num,
                                chapter_title=chapter_title,
                                source_type="concept_list",
                                context_snippet=f"Listed as concept: {', '.join(matching_concepts)}",
                                relevance_score=1.0,  # Already identified as concept = highest
                            ))
                
                # Search keyword lists
                if search_keyword_lists:
                    keywords = chapter.get("keywords", [])
                    # enriched_keywords may be a dict with tfidf/semantic/merged
                    enriched_kw = chapter.get("enriched_keywords", {})
                    if isinstance(enriched_kw, dict):
                        keywords = keywords + enriched_kw.get("merged", []) + enriched_kw.get("tfidf", [])
                    elif isinstance(enriched_kw, list):
                        keywords = keywords + enriched_kw
                    
                    if any(self._term_matches(k, term, use_word_boundary) for k in keywords):
                        context.appears_in_keyword_lists = True
                        books_with_term.add(book_title)
                        chapters_with_term.add(f"{book_title}:{chapter_num}")
        
        # Search raw text (slower, do last)
        if search_raw_text and len(context.occurrences) < max_results:
            for title, _ in self._book_index.items():
                raw_book = self._load_raw_book(title.replace("_", " ").title())
                if not raw_book:
                    # Try exact title match
                    for f in self.raw_path.glob("*.json"):
                        if f.stem.lower() == title.lower():
                            raw_book = self._load_raw_book(f.stem)
                            break
                
                if not raw_book:
                    continue
                
                book_title = raw_book.get("metadata", {}).get("title", title)
                
                for chapter in raw_book.get("chapters", []):
                    content = chapter.get("content", "")
                    if self._term_matches(content, term, use_word_boundary):
                        context.appears_in_raw_text = True
                        books_with_term.add(book_title)
                        chapter_num = chapter.get("number")
                        chapters_with_term.add(f"{book_title}:{chapter_num}")
                        
                        if len(context.occurrences) < max_results:
                            context.occurrences.append(TermOccurrence(
                                term=term,
                                book_title=book_title,
                                chapter_number=chapter_num,
                                chapter_title=chapter.get("title", ""),
                                source_type="raw_text",
                                context_snippet=self._extract_context_snippet(content, term, 200),
                                relevance_score=0.7,  # Raw text is lower value than summaries
                            ))
                
                if len(context.occurrences) >= max_results:
                    break
        
        # Update counts
        context.books_found_in = len(books_with_term)
        context.chapters_found_in = len(chapters_with_term)
        
        # Sort by relevance
        context.occurrences.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return context
    
    def find_batch_context(
        self,
        terms: list[str],
        max_results_per_term: int = 3,
    ) -> dict[str, TermContext]:
        """Find context for multiple terms efficiently.
        
        Args:
            terms: List of terms to search for.
            max_results_per_term: Max occurrences per term.
        
        Returns:
            Dict mapping term -> TermContext.
        """
        results = {}
        
        for term in terms:
            results[term] = self.find_term_context(
                term,
                max_results=max_results_per_term,
            )
        
        return results
    
    def format_context_for_llm(
        self,
        term_context: TermContext,
        include_snippets: bool = True,
    ) -> str:
        """Format term context as text for LLM consumption.
        
        Args:
            term_context: The context to format.
            include_snippets: Whether to include text snippets.
        
        Returns:
            Formatted string suitable for LLM prompt.
        """
        lines = [
            f"Term: '{term_context.term}'",
            f"  Found in {term_context.books_found_in} books, {term_context.chapters_found_in} chapters",
        ]
        
        # Summary of where it appears
        sources = []
        if term_context.appears_in_concept_lists:
            sources.append("concept lists")
        if term_context.appears_in_summaries:
            sources.append("chapter summaries")
        if term_context.appears_in_keyword_lists:
            sources.append("keyword lists")
        if term_context.appears_in_raw_text:
            sources.append("full text")
        
        if sources:
            lines.append(f"  Appears in: {', '.join(sources)}")
        
        # Add snippets
        if include_snippets and term_context.occurrences:
            lines.append("  Evidence:")
            for occ in term_context.occurrences[:3]:
                source = f"[{occ.source_type}] {occ.book_title}"
                if occ.chapter_title:
                    source += f", Ch.{occ.chapter_number}: {occ.chapter_title}"
                lines.append(f"    - {source}")
                if occ.context_snippet:
                    # Truncate long snippets
                    snippet = occ.context_snippet[:200]
                    if len(occ.context_snippet) > 200:
                        snippet += "..."
                    lines.append(f"      \"{snippet}\"")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_context_for_disputed_terms(
    disputed_terms: list[str],
    max_per_term: int = 3,
) -> dict[str, str]:
    """Get formatted context for disputed terms.
    
    Args:
        disputed_terms: List of terms that need context.
        max_per_term: Max evidence items per term.
    
    Returns:
        Dict mapping term -> formatted context string.
    """
    lookup = BookContextLookup()
    results = {}
    
    for term in disputed_terms:
        context = lookup.find_term_context(term, max_results=max_per_term)
        results[term] = lookup.format_context_for_llm(context)
    
    return results


# =============================================================================
# CLI Test
# =============================================================================


if __name__ == "__main__":
    # Test the lookup
    lookup = BookContextLookup()
    
    test_terms = ["microservice", "acid", "bounded context", "kubernetes"]
    
    for term in test_terms:
        print(f"\n{'='*60}")
        context = lookup.find_term_context(term, max_results=5)
        print(lookup.format_context_for_llm(context))
