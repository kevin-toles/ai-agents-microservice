"""Book Citation Mapper.

WBS Reference: WBS-AGT23 Book/JSON Passage Retrieval (AGT23.7)
Acceptance Criteria:
- AC-23.4: Return structured BookPassage with citation metadata
- Citations include author, title, page numbers (Chicago format ready)

Maps BookPassage objects to Citation objects for downstream use.
Supports Chicago-style citation formatting.

Pattern: Mapper/Transformer pattern
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.passage_models import BookPassage


# =============================================================================
# Citation Model
# =============================================================================


@dataclass(frozen=True)
class BookCitation:
    """Citation for a book passage.
    
    Contains all metadata required for Chicago-style citation.
    
    Attributes:
        source_type: Always "book_passage" for book citations
        source_id: Passage ID for traceability
        author: Book author(s)
        book_title: Full book title
        chapter_title: Chapter title
        chapter_number: Chapter number
        start_page: Starting page number
        end_page: Ending page number
        publisher: Publisher name (optional)
        year: Publication year (optional)
        concepts: Related concepts for context
    """
    
    source_type: str
    source_id: str
    author: str
    book_title: str
    chapter_title: str
    chapter_number: int
    start_page: int
    end_page: int
    publisher: str = ""
    year: str = ""
    concepts: tuple[str, ...] = ()
    
    def get_page_range(self) -> str:
        """Get formatted page range string."""
        if self.start_page == self.end_page:
            return str(self.start_page)
        return f"{self.start_page}-{self.end_page}"
    
    def to_chicago(self) -> str:
        """Format citation in Chicago style.
        
        Chicago Notes-Bibliography format:
        Author Last, First, Title (Place: Publisher, Year), Pages.
        
        Example:
        Ousterhout, John, A Philosophy of Software Design (Palo Alto: Yaknyam Press, 2018), 31-42.
        
        Returns:
            Chicago-style citation string
        """
        # Extract last name for Chicago format
        author_parts = self.author.split()
        if len(author_parts) >= 2:
            # "John Ousterhout" -> "Ousterhout, John"
            last_name = author_parts[-1]
            first_names = " ".join(author_parts[:-1])
            formatted_author = f"{last_name}, {first_names}"
        else:
            formatted_author = self.author
        
        # Build citation parts
        parts = [formatted_author]
        parts.append(f'"{self.book_title}"')
        
        # Add publisher/year if available
        if self.publisher and self.year:
            parts.append(f"({self.publisher}, {self.year})")
        elif self.year:
            parts.append(f"({self.year})")
        
        # Add page range
        parts.append(self.get_page_range())
        
        return ", ".join(parts) + "."
    
    def to_short(self) -> str:
        """Format short citation for footnotes.
        
        Short format: Author, Title, Pages.
        
        Example:
        Ousterhout, A Philosophy of Software Design, 31-42.
        
        Returns:
            Short citation string
        """
        author_parts = self.author.split()
        last_name = author_parts[-1] if author_parts else self.author
        
        return f"{last_name}, {self.book_title}, {self.get_page_range()}."
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "author": self.author,
            "book_title": self.book_title,
            "chapter_title": self.chapter_title,
            "chapter_number": self.chapter_number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "publisher": self.publisher,
            "year": self.year,
            "concepts": list(self.concepts),
        }


# =============================================================================
# Mapper Functions
# =============================================================================


def passage_to_citation(passage: BookPassage) -> BookCitation:
    """Convert BookPassage to BookCitation.
    
    Args:
        passage: BookPassage to convert
        
    Returns:
        BookCitation with citation metadata
    """
    return BookCitation(
        source_type="book_passage",
        source_id=passage.passage_id,
        author=passage.author,
        book_title=passage.book_title,
        chapter_title=passage.chapter_title,
        chapter_number=passage.chapter_number,
        start_page=passage.start_page,
        end_page=passage.end_page,
        concepts=tuple(passage.concepts),
    )


def passages_to_citations(passages: list[BookPassage]) -> list[BookCitation]:
    """Convert list of BookPassages to BookCitations.
    
    Args:
        passages: List of BookPassage objects
        
    Returns:
        List of BookCitation objects
    """
    return [passage_to_citation(p) for p in passages]


def citation_from_dict(data: dict) -> BookCitation:
    """Create BookCitation from dictionary.
    
    Args:
        data: Dictionary with citation data
        
    Returns:
        BookCitation instance
    """
    return BookCitation(
        source_type=data.get("source_type", "book_passage"),
        source_id=data.get("source_id", ""),
        author=data.get("author", ""),
        book_title=data.get("book_title", ""),
        chapter_title=data.get("chapter_title", ""),
        chapter_number=data.get("chapter_number", 0),
        start_page=data.get("start_page", 0),
        end_page=data.get("end_page", 0),
        publisher=data.get("publisher", ""),
        year=data.get("year", ""),
        concepts=tuple(data.get("concepts", [])),
    )
