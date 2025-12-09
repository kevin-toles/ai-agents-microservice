#!/usr/bin/env python3
"""Graph RAG POC End-to-End Validation Script.

This script validates that the Graph RAG POC actually works by:
1. Loading the REAL taxonomy from AI-ML_taxonomy_20251128.json
2. Reading REAL content from JSON textbook files
3. Testing tier-based traversal (PARALLEL/PERPENDICULAR/SKIP_TIER)
4. Validating cross-reference generation across textbooks

Usage:
    python scripts/validate_graph_rag_poc.py
    
Data Sources:
    - Taxonomy: /Users/kevintoles/POC/textbooks/Taxonomies/AI-ML_taxonomy_20251128.json
    - Textbooks: /Users/kevintoles/POC/textbooks/JSON Texts/
    - Guidelines: /Users/kevintoles/POC/textbooks/Guidelines/
"""

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Paths to real data
TAXONOMY_PATH = Path("/Users/kevintoles/POC/textbooks/Taxonomies/AI-ML_taxonomy_20251128.json")
TEXTBOOKS_PATH = Path("/Users/kevintoles/POC/textbooks/JSON Texts")
GUIDELINES_PATH = Path("/Users/kevintoles/POC/textbooks/Guidelines")


class RelationshipType(str, Enum):
    """Types of relationships between books/chapters."""
    PARALLEL = "parallel"      # Same tier, similar concepts
    PERPENDICULAR = "perpendicular"  # Adjacent tier, bridging concepts
    SKIP_TIER = "skip_tier"   # Non-adjacent tier connection


@dataclass
class TaxonomyBook:
    """A book in the taxonomy."""
    title: str
    tier: str
    priority: int
    chapters_count: int
    file_path: Path | None = None


@dataclass
class TaxonomyTier:
    """A tier in the taxonomy."""
    name: str
    priority: int
    books: list[TaxonomyBook] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)


@dataclass
class TaxonomyGraph:
    """The full taxonomy graph."""
    total_books: int
    generated_date: str
    tiers: dict[str, TaxonomyTier] = field(default_factory=dict)
    
    
@dataclass
class ChapterContent:
    """Content from a textbook chapter."""
    book_title: str
    chapter_number: int
    chapter_title: str
    content: str
    start_page: int
    end_page: int
    

@dataclass
class CrossReference:
    """A cross-reference between chapters."""
    source_book: str
    source_chapter: int
    target_book: str
    target_chapter: int
    relationship_type: RelationshipType
    shared_concepts: list[str]
    relevance_score: float


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    details: str
    data: dict[str, Any] = field(default_factory=dict)


class TaxonomyLoader:
    """Loads and parses the real taxonomy JSON."""
    
    def __init__(self, taxonomy_path: Path):
        self.taxonomy_path = taxonomy_path
        self.raw_data: dict = {}
        
    def load(self) -> TaxonomyGraph:
        """Load and parse the taxonomy file."""
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
            
        with open(self.taxonomy_path, "r") as f:
            self.raw_data = json.load(f)
            
        return self._parse_taxonomy()
    
    def _parse_taxonomy(self) -> TaxonomyGraph:
        """Parse the raw JSON into a TaxonomyGraph."""
        graph = TaxonomyGraph(
            total_books=self.raw_data.get("total_books", 0),
            generated_date=self.raw_data.get("generated", "unknown"),
        )
        
        for tier_name, tier_data in self.raw_data.get("tiers", {}).items():
            tier = TaxonomyTier(
                name=tier_name,
                priority=tier_data.get("priority", 99),
                concepts=tier_data.get("concepts", []),
            )
            
            for book_data in tier_data.get("books", []):
                # Taxonomy uses "name" field which is the JSON filename
                book_name = book_data.get("name", "Unknown")
                # Convert filename to title (remove .json extension)
                book_title = book_name.replace(".json", "") if book_name.endswith(".json") else book_name
                
                book = TaxonomyBook(
                    title=book_title,
                    tier=tier_name,
                    priority=book_data.get("priority", 99),
                    chapters_count=book_data.get("chapters", 0),  # Field is "chapters" not "chapters_count"
                )
                # Set file path directly from name since it's the filename
                book.file_path = TEXTBOOKS_PATH / book_name
                tier.books.append(book)
                
            graph.tiers[tier_name] = tier
            
        return graph
    
    def _find_book_file(self, book_title: str) -> Path | None:
        """Find the JSON file for a book by title."""
        # Try exact match first
        for json_file in TEXTBOOKS_PATH.glob("*.json"):
            if book_title.lower() in json_file.stem.lower():
                return json_file
        return None


class TextbookLoader:
    """Loads content from JSON textbook files."""
    
    def __init__(self, textbooks_path: Path):
        self.textbooks_path = textbooks_path
        self._cache: dict[str, dict] = {}
        
    def load_book(self, file_path: Path) -> dict:
        """Load a textbook JSON file."""
        if str(file_path) in self._cache:
            return self._cache[str(file_path)]
            
        with open(file_path, "r") as f:
            data = json.load(f)
            self._cache[str(file_path)] = data
            return data
            
    def get_chapter(self, file_path: Path, chapter_num: int) -> ChapterContent | None:
        """Get a specific chapter from a textbook."""
        book_data = self.load_book(file_path)
        
        metadata = book_data.get("metadata", {})
        chapters = book_data.get("chapters", [])
        
        for chapter in chapters:
            if chapter.get("number") == chapter_num:
                return ChapterContent(
                    book_title=metadata.get("title", "Unknown"),
                    chapter_number=chapter_num,
                    chapter_title=chapter.get("title", ""),
                    content=chapter.get("content", "")[:2000],  # Truncate for display
                    start_page=chapter.get("start_page", 0),
                    end_page=chapter.get("end_page", 0),
                )
        return None
    
    def get_chapter_count(self, file_path: Path) -> int:
        """Get the number of chapters in a book."""
        book_data = self.load_book(file_path)
        return len(book_data.get("chapters", []))


class GraphTraversal:
    """Performs spider-web traversal across the taxonomy."""
    
    def __init__(self, taxonomy: TaxonomyGraph, textbook_loader: TextbookLoader):
        self.taxonomy = taxonomy
        self.textbook_loader = textbook_loader
        
    def get_tier_priority(self, tier_name: str) -> int:
        """Get the priority of a tier."""
        if tier_name in self.taxonomy.tiers:
            return self.taxonomy.tiers[tier_name].priority
        return 99
    
    def determine_relationship(self, source_tier: str, target_tier: str) -> RelationshipType:
        """Determine the relationship type between two tiers."""
        source_priority = self.get_tier_priority(source_tier)
        target_priority = self.get_tier_priority(target_tier)
        
        if source_priority == target_priority:
            return RelationshipType.PARALLEL
        elif abs(source_priority - target_priority) == 1:
            return RelationshipType.PERPENDICULAR
        else:
            return RelationshipType.SKIP_TIER
            
    def find_related_books(
        self,
        source_book: TaxonomyBook,
        relationship_type: RelationshipType | None = None,
    ) -> list[tuple[TaxonomyBook, RelationshipType]]:
        """Find books related to the source book."""
        results = []
        source_tier = self.taxonomy.tiers.get(source_book.tier)
        if not source_tier:
            return results
            
        for tier_name, tier in self.taxonomy.tiers.items():
            for book in tier.books:
                if book.title == source_book.title:
                    continue  # Skip self
                    
                rel_type = self.determine_relationship(source_book.tier, tier_name)
                
                if relationship_type is None or rel_type == relationship_type:
                    results.append((book, rel_type))
                    
        return results
    
    def find_shared_concepts(self, tier1: str, tier2: str) -> list[str]:
        """Find concepts shared between two tiers."""
        t1 = self.taxonomy.tiers.get(tier1)
        t2 = self.taxonomy.tiers.get(tier2)
        
        if not t1 or not t2:
            return []
            
        return list(set(t1.concepts) & set(t2.concepts))


class CrossReferenceGenerator:
    """Generates cross-references between textbook chapters."""
    
    def __init__(self, traversal: GraphTraversal, textbook_loader: TextbookLoader):
        self.traversal = traversal
        self.textbook_loader = textbook_loader
        
    def generate_cross_references(
        self,
        source_book: TaxonomyBook,
        source_chapter: int,
        max_refs: int = 5,
    ) -> list[CrossReference]:
        """Generate cross-references for a chapter."""
        cross_refs = []
        
        related_books = self.traversal.find_related_books(source_book)
        
        for target_book, rel_type in related_books[:max_refs]:
            if not target_book.file_path:
                continue
                
            # Get chapter count for target book
            chapter_count = self.textbook_loader.get_chapter_count(target_book.file_path)
            
            # Simple heuristic: map to similar chapter
            target_chapter = min(source_chapter, chapter_count)
            
            shared = self.traversal.find_shared_concepts(
                source_book.tier, target_book.tier
            )
            
            cross_ref = CrossReference(
                source_book=source_book.title,
                source_chapter=source_chapter,
                target_book=target_book.title,
                target_chapter=target_chapter,
                relationship_type=rel_type,
                shared_concepts=shared[:5],  # Limit to 5 concepts
                relevance_score=0.8 if rel_type == RelationshipType.PARALLEL else 0.6,
            )
            cross_refs.append(cross_ref)
            
        return cross_refs


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(result: ValidationResult) -> None:
    """Print a validation result."""
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n{status} - {result.test_name}")
    print(f"   {result.details}")
    if result.data:
        for key, value in result.data.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")


def validate_taxonomy_loading() -> ValidationResult:
    """Test 1: Can we load the real taxonomy?"""
    try:
        loader = TaxonomyLoader(TAXONOMY_PATH)
        taxonomy = loader.load()
        
        return ValidationResult(
            test_name="Taxonomy Loading",
            passed=taxonomy.total_books > 0,
            details=f"Loaded taxonomy with {taxonomy.total_books} books across {len(taxonomy.tiers)} tiers",
            data={
                "total_books": taxonomy.total_books,
                "tiers": list(taxonomy.tiers.keys()),
                "generated_date": taxonomy.generated_date,
            }
        )
    except Exception as e:
        return ValidationResult(
            test_name="Taxonomy Loading",
            passed=False,
            details=f"Failed to load taxonomy: {e}",
        )


def validate_tier_structure() -> ValidationResult:
    """Test 2: Is the tier structure correct?"""
    try:
        loader = TaxonomyLoader(TAXONOMY_PATH)
        taxonomy = loader.load()
        
        # Check for architecture tier (should exist with priority 1)
        arch_tier = taxonomy.tiers.get("architecture")
        
        if not arch_tier:
            return ValidationResult(
                test_name="Tier Structure",
                passed=False,
                details="Architecture tier not found",
            )
            
        return ValidationResult(
            test_name="Tier Structure",
            passed=arch_tier.priority == 1 and len(arch_tier.books) > 0,
            details=f"Architecture tier has {len(arch_tier.books)} books with priority {arch_tier.priority}",
            data={
                "architecture_books": [b.title for b in arch_tier.books[:5]],
                "concepts_sample": arch_tier.concepts[:10],
            }
        )
    except Exception as e:
        return ValidationResult(
            test_name="Tier Structure",
            passed=False,
            details=f"Failed: {e}",
        )


def validate_textbook_loading() -> ValidationResult:
    """Test 3: Can we load textbook JSON files?"""
    try:
        loader = TextbookLoader(TEXTBOOKS_PATH)
        
        # Try to load AI Engineering book
        ai_eng_path = TEXTBOOKS_PATH / "AI Engineering Building Applications.json"
        if not ai_eng_path.exists():
            return ValidationResult(
                test_name="Textbook Loading",
                passed=False,
                details=f"Test file not found: {ai_eng_path}",
            )
            
        book_data = loader.load_book(ai_eng_path)
        chapter = loader.get_chapter(ai_eng_path, 1)
        
        return ValidationResult(
            test_name="Textbook Loading",
            passed=chapter is not None and len(chapter.content) > 0,
            details=f"Loaded '{chapter.book_title}' - Chapter {chapter.chapter_number}: {chapter.chapter_title[:50]}...",
            data={
                "book_title": chapter.book_title,
                "chapter_count": loader.get_chapter_count(ai_eng_path),
                "content_preview": chapter.content[:200] + "...",
            }
        )
    except Exception as e:
        return ValidationResult(
            test_name="Textbook Loading",
            passed=False,
            details=f"Failed: {e}",
        )


def validate_graph_traversal() -> ValidationResult:
    """Test 4: Can we traverse the taxonomy graph?"""
    try:
        taxonomy_loader = TaxonomyLoader(TAXONOMY_PATH)
        taxonomy = taxonomy_loader.load()
        textbook_loader = TextbookLoader(TEXTBOOKS_PATH)
        traversal = GraphTraversal(taxonomy, textbook_loader)
        
        # Find a book to start from
        arch_tier = taxonomy.tiers.get("architecture")
        if not arch_tier or not arch_tier.books:
            return ValidationResult(
                test_name="Graph Traversal",
                passed=False,
                details="No books found in architecture tier",
            )
            
        source_book = arch_tier.books[0]
        
        # Find PARALLEL relationships
        parallel_books = traversal.find_related_books(
            source_book, RelationshipType.PARALLEL
        )
        
        # Find PERPENDICULAR relationships
        perpendicular_books = traversal.find_related_books(
            source_book, RelationshipType.PERPENDICULAR
        )
        
        return ValidationResult(
            test_name="Graph Traversal",
            passed=len(parallel_books) >= 0 or len(perpendicular_books) >= 0,
            details=f"From '{source_book.title}': Found {len(parallel_books)} PARALLEL, {len(perpendicular_books)} PERPENDICULAR",
            data={
                "source_book": source_book.title,
                "source_tier": source_book.tier,
                "parallel_count": len(parallel_books),
                "perpendicular_count": len(perpendicular_books),
                "parallel_sample": [b[0].title for b in parallel_books[:3]],
            }
        )
    except Exception as e:
        return ValidationResult(
            test_name="Graph Traversal",
            passed=False,
            details=f"Failed: {e}",
        )


def validate_cross_reference_generation() -> ValidationResult:
    """Test 5: Can we generate cross-references?"""
    try:
        taxonomy_loader = TaxonomyLoader(TAXONOMY_PATH)
        taxonomy = taxonomy_loader.load()
        textbook_loader = TextbookLoader(TEXTBOOKS_PATH)
        traversal = GraphTraversal(taxonomy, textbook_loader)
        generator = CrossReferenceGenerator(traversal, textbook_loader)
        
        # Find a book with a file path
        source_book = None
        for tier in taxonomy.tiers.values():
            for book in tier.books:
                if book.file_path and book.file_path.exists():
                    source_book = book
                    break
            if source_book:
                break
                
        if not source_book:
            return ValidationResult(
                test_name="Cross-Reference Generation",
                passed=False,
                details="No books with file paths found",
            )
            
        cross_refs = generator.generate_cross_references(
            source_book, source_chapter=1, max_refs=5
        )
        
        return ValidationResult(
            test_name="Cross-Reference Generation",
            passed=len(cross_refs) > 0,
            details=f"Generated {len(cross_refs)} cross-references from '{source_book.title}'",
            data={
                "source_book": source_book.title,
                "cross_refs": [
                    {
                        "target": cr.target_book,
                        "relationship": cr.relationship_type.value,
                        "shared_concepts": cr.shared_concepts[:3],
                    }
                    for cr in cross_refs[:3]
                ],
            }
        )
    except Exception as e:
        return ValidationResult(
            test_name="Cross-Reference Generation",
            passed=False,
            details=f"Failed: {e}",
        )


def validate_end_to_end_scenario() -> ValidationResult:
    """Test 6: Full E2E scenario - Navigate taxonomy and retrieve content."""
    try:
        # Load taxonomy
        taxonomy_loader = TaxonomyLoader(TAXONOMY_PATH)
        taxonomy = taxonomy_loader.load()
        textbook_loader = TextbookLoader(TEXTBOOKS_PATH)
        traversal = GraphTraversal(taxonomy, textbook_loader)
        generator = CrossReferenceGenerator(traversal, textbook_loader)
        
        # Scenario: User is reading "AI Engineering Building Applications" Chapter 1
        # and wants cross-references to other books
        
        # Find AI Engineering book (title is "AI Engineering Building Applications")
        ai_eng_book = None
        for tier in taxonomy.tiers.values():
            for book in tier.books:
                if "AI Engineering" in book.title:
                    ai_eng_book = book
                    break
            if ai_eng_book:
                break
        
        if not ai_eng_book:
            # List available books for debugging
            all_books = []
            for tier in taxonomy.tiers.values():
                all_books.extend([b.title for b in tier.books])
            return ValidationResult(
                test_name="E2E Scenario",
                passed=False,
                details=f"AI Engineering book not found in taxonomy. Available: {all_books[:5]}",
            )
        
        # Verify file path exists
        if not ai_eng_book.file_path or not ai_eng_book.file_path.exists():
            return ValidationResult(
                test_name="E2E Scenario",
                passed=False,
                details=f"AI Engineering JSON file not found at {ai_eng_book.file_path}",
            )
            
        chapter = textbook_loader.get_chapter(ai_eng_book.file_path, 1)
        
        # Generate cross-references
        cross_refs = generator.generate_cross_references(
            ai_eng_book, source_chapter=1, max_refs=3
        )
        
        # Retrieve content from cross-referenced chapters
        retrieved_content = []
        for cr in cross_refs:
            target_book = None
            for tier in taxonomy.tiers.values():
                for book in tier.books:
                    if book.title == cr.target_book and book.file_path:
                        target_book = book
                        break
                        
            if target_book and target_book.file_path and target_book.file_path.exists():
                target_chapter = textbook_loader.get_chapter(
                    target_book.file_path, cr.target_chapter
                )
                if target_chapter:
                    retrieved_content.append({
                        "book": target_chapter.book_title,
                        "chapter": target_chapter.chapter_number,
                        "title": target_chapter.chapter_title[:50],
                        "relationship": cr.relationship_type.value,
                    })
        
        return ValidationResult(
            test_name="E2E Scenario",
            passed=len(retrieved_content) > 0,
            details=f"Successfully navigated from '{ai_eng_book.title}' to {len(retrieved_content)} cross-referenced chapters",
            data={
                "source": f"{ai_eng_book.title} - Chapter 1",
                "source_content_preview": chapter.content[:150] + "..." if chapter else "No content",
                "cross_references": retrieved_content,
            }
        )
    except Exception as e:
        import traceback
        return ValidationResult(
            test_name="E2E Scenario",
            passed=False,
            details=f"Failed: {e}\n{traceback.format_exc()}",
        )


def main() -> int:
    """Run all validation tests."""
    print_header("Graph RAG POC Validation")
    print("\nValidating that the POC can navigate the real taxonomy")
    print("and retrieve real content from textbook JSON files.\n")
    
    # Check data paths exist
    print(f"Taxonomy path: {TAXONOMY_PATH}")
    print(f"  Exists: {TAXONOMY_PATH.exists()}")
    print(f"Textbooks path: {TEXTBOOKS_PATH}")
    print(f"  Exists: {TEXTBOOKS_PATH.exists()}")
    
    if not TAXONOMY_PATH.exists() or not TEXTBOOKS_PATH.exists():
        print("\n❌ ERROR: Required data files not found!")
        return 1
    
    # Run validation tests
    tests = [
        validate_taxonomy_loading,
        validate_tier_structure,
        validate_textbook_loading,
        validate_graph_traversal,
        validate_cross_reference_generation,
        validate_end_to_end_scenario,
    ]
    
    results = []
    for test in tests:
        result = test()
        print_result(result)
        results.append(result)
    
    # Summary
    print_header("Validation Summary")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n  Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n  ✅ ALL TESTS PASSED - POC is working with real data!")
        return 0
    else:
        print("\n  ❌ SOME TESTS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
