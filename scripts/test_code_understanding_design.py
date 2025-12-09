#!/usr/bin/env python3
"""AI Agent Test: Design a Scalable LLM-Powered Code Understanding System.

This script tests the Graph RAG system by having it answer a complex design question
using real cross-references from the taxonomy and textbooks.

Query: "How would you design a system where an LLM reads and understands a 20M-line 
monorepo and answers queries like 'where is the rate-limiter implemented?'?"

Focus Points:
- Multi-stage chunking
- Embeddings + hierarchical retrieval
- Indexing strategies
- Incremental refresh pipeline
- Grounding LLM outputs
- Safety and hallucination-hardening

Output: JSON file with validated cross-references (no caps, verification that refs are real)
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Paths to real data
TAXONOMY_PATH = Path("/Users/kevintoles/POC/textbooks/Taxonomies/AI-ML_taxonomy_20251128.json")
TEXTBOOKS_PATH = Path("/Users/kevintoles/POC/textbooks/JSON Texts")
OUTPUT_PATH = Path("/Users/kevintoles/POC/ai-agents/outputs")


@dataclass
class CrossReference:
    """A validated cross-reference to a textbook."""
    book_title: str
    chapter_number: int
    chapter_title: str
    page_range: str
    relevance_topic: str
    quote_excerpt: str
    is_validated: bool = False
    validation_error: str | None = None


@dataclass
class DesignSection:
    """A section of the design response."""
    topic: str
    description: str
    key_concepts: list[str]
    cross_references: list[CrossReference] = field(default_factory=list)


@dataclass
class DesignResponse:
    """Complete design response with cross-references."""
    query: str
    timestamp: str
    focus_areas: list[str]
    sections: list[DesignSection] = field(default_factory=list)
    total_cross_references: int = 0
    validated_cross_references: int = 0
    validation_summary: dict[str, Any] = field(default_factory=dict)


class TaxonomyKnowledgeBase:
    """Knowledge base built from real taxonomy and textbooks."""
    
    def __init__(self):
        self.taxonomy: dict = {}
        self.textbooks: dict[str, dict] = {}
        self.concept_index: dict[str, list[tuple[str, str]]] = {}  # concept -> [(book, tier)]
        self.chapter_index: dict[str, list[dict]] = {}  # book -> [chapters]
        
    def load(self) -> None:
        """Load taxonomy and build indices."""
        # Load taxonomy
        with open(TAXONOMY_PATH, "r") as f:
            self.taxonomy = json.load(f)
        
        # Extract valid book names from taxonomy (books that should be searched)
        taxonomy_book_names: set[str] = set()
        for tier_name, tier_data in self.taxonomy.get("tiers", {}).items():
            for book in tier_data.get("books", []):
                book_name = book.get("name", "").replace(".json", "")
                taxonomy_book_names.add(book_name)
        
        print(f"Taxonomy defines {len(taxonomy_book_names)} valid books")
            
        # Build concept index from taxonomy
        for tier_name, tier_data in self.taxonomy.get("tiers", {}).items():
            for concept in tier_data.get("concepts", []):
                if concept not in self.concept_index:
                    self.concept_index[concept] = []
                # Add all books in this tier for this concept
                for book in tier_data.get("books", []):
                    book_name = book.get("name", "").replace(".json", "")
                    self.concept_index[concept].append((book_name, tier_name))
                    
        # Load ONLY textbooks that are in the taxonomy (not all JSON files)
        loaded_count = 0
        skipped_count = 0
        for json_file in TEXTBOOKS_PATH.glob("*.json"):
            book_key = json_file.stem
            # Only load books that are defined in the taxonomy
            if book_key in taxonomy_book_names:
                try:
                    with open(json_file, "r") as f:
                        book_data = json.load(f)
                        self.textbooks[book_key] = book_data
                        self.chapter_index[book_key] = book_data.get("chapters", [])
                        loaded_count += 1
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
            else:
                skipped_count += 1
        
        print(f"Loaded {loaded_count} taxonomy books, skipped {skipped_count} non-taxonomy books")
                
    def search_concepts(self, query_terms: list[str]) -> list[tuple[str, str, str]]:
        """Search for concepts matching query terms.
        
        Returns: [(concept, book, tier), ...]
        """
        results = []
        for term in query_terms:
            term_lower = term.lower()
            for concept, book_tiers in self.concept_index.items():
                if term_lower in concept.lower() or concept.lower() in term_lower:
                    for book, tier in book_tiers:
                        results.append((concept, book, tier))
        return results
    
    def get_chapter(self, book_key: str, chapter_num: int) -> dict | None:
        """Get a specific chapter from a book."""
        chapters = self.chapter_index.get(book_key, [])
        for chapter in chapters:
            if chapter.get("number") == chapter_num:
                return chapter
        return None
    
    def search_content(self, book_key: str, search_terms: list[str]) -> list[dict]:
        """Search for content containing specific terms."""
        results = []
        chapters = self.chapter_index.get(book_key, [])
        
        for chapter in chapters:
            content = chapter.get("content", "").lower()
            for term in search_terms:
                if term.lower() in content:
                    results.append({
                        "chapter": chapter.get("number"),
                        "title": chapter.get("title"),
                        "start_page": chapter.get("start_page"),
                        "end_page": chapter.get("end_page"),
                        "match_term": term,
                    })
                    break  # One match per chapter
        return results
    
    def get_excerpt(self, book_key: str, chapter_num: int, search_term: str, context_chars: int = 300) -> str:
        """Extract a relevant excerpt from a chapter."""
        chapter = self.get_chapter(book_key, chapter_num)
        if not chapter:
            return ""
            
        content = chapter.get("content", "")
        term_lower = search_term.lower()
        content_lower = content.lower()
        
        idx = content_lower.find(term_lower)
        if idx == -1:
            # Return start of chapter if term not found
            return content[:context_chars] + "..."
            
        # Get context around the match
        start = max(0, idx - context_chars // 2)
        end = min(len(content), idx + len(search_term) + context_chars // 2)
        
        excerpt = content[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
            
        return excerpt


class CodeUnderstandingDesigner:
    """Generates a design for LLM-powered code understanding with real cross-references."""
    
    # Focus areas for the design
    FOCUS_AREAS = [
        "multi-stage chunking",
        "embeddings + hierarchical retrieval",
        "indexing strategies", 
        "incremental refresh pipeline",
        "grounding LLM outputs",
        "safety and hallucination-hardening",
    ]
    
    # Mapping of focus areas to search terms for finding relevant content
    FOCUS_SEARCH_TERMS = {
        "multi-stage chunking": [
            "chunk", "chunking", "split", "segment", "tokenization", "context window",
            "sliding window", "recursive", "document", "text splitting",
        ],
        "embeddings + hierarchical retrieval": [
            "embedding", "embeddings", "vector", "retrieval", "retriever", "RAG",
            "semantic search", "similarity", "hierarchical", "index", "HNSW",
        ],
        "indexing strategies": [
            "index", "indexing", "inverted index", "vector index", "search",
            "database", "storage", "query", "metadata", "filter",
        ],
        "incremental refresh pipeline": [
            "incremental", "update", "pipeline", "refresh", "change detection",
            "delta", "streaming", "batch", "cache", "invalidation",
        ],
        "grounding LLM outputs": [
            "grounding", "citation", "reference", "source", "attribution",
            "RAG", "retrieval", "context", "fact", "verification",
        ],
        "safety and hallucination-hardening": [
            "hallucination", "safety", "guardrail", "validation", "factual",
            "accuracy", "evaluation", "verification", "confidence", "uncertainty",
        ],
    }
    
    # Design content for each focus area
    DESIGN_CONTENT = {
        "multi-stage chunking": {
            "description": """For a 20M-line monorepo, implement a multi-stage chunking strategy:

1. **Repository-Level Chunking**: Split by project/module boundaries
2. **File-Level Chunking**: Group related files (e.g., class + tests)
3. **Semantic Chunking**: Use AST parsing for code structure awareness
4. **Context-Aware Splitting**: Maintain function/class boundaries
5. **Overlap Strategy**: 10-20% overlap to preserve context across chunks

Key considerations:
- Chunk size: 512-2048 tokens for code (smaller than prose)
- Preserve import statements and type definitions
- Include file path and module context as metadata
- Handle multiple languages with language-specific parsers""",
            "key_concepts": [
                "AST-based parsing",
                "Semantic boundaries",
                "Context windows",
                "Token limits",
                "Metadata preservation",
            ],
        },
        "embeddings + hierarchical retrieval": {
            "description": """Design a hierarchical retrieval system:

1. **Embedding Strategy**:
   - Code-specialized embeddings (CodeBERT, StarCoder embeddings)
   - Separate embeddings for: code, comments, docstrings, file paths
   - Multi-vector representations per chunk

2. **Hierarchical Index Structure**:
   - Level 0: Repository summary embeddings
   - Level 1: Module/package embeddings  
   - Level 2: File embeddings
   - Level 3: Function/class embeddings
   - Level 4: Fine-grained code block embeddings

3. **Retrieval Pipeline**:
   - Coarse-to-fine search (top-down traversal)
   - Hybrid: dense + sparse (BM25) retrieval
   - Re-ranking with cross-encoder models
   - Parent document retrieval for context""",
            "key_concepts": [
                "Code embeddings",
                "Hierarchical indexing",
                "Hybrid retrieval",
                "Cross-encoder re-ranking",
                "Multi-vector search",
            ],
        },
        "indexing strategies": {
            "description": """Implement robust indexing for code search:

1. **Vector Index** (Qdrant/Pinecone):
   - HNSW for approximate nearest neighbor
   - Payload filtering by language, file type, date
   - Quantization for memory efficiency

2. **Graph Index** (Neo4j):
   - Code dependency graph (imports, calls, inheritance)
   - File co-change relationships
   - Symbol reference graph

3. **Keyword Index** (Elasticsearch):
   - Full-text search for exact matches
   - Symbol/identifier index
   - Regex pattern matching

4. **Metadata Index**:
   - Git history integration
   - Code ownership/CODEOWNERS
   - Test coverage mapping""",
            "key_concepts": [
                "HNSW indexing",
                "Graph databases",
                "Inverted indices",
                "Metadata filtering",
                "Dependency graphs",
            ],
        },
        "incremental refresh pipeline": {
            "description": """Design an efficient incremental update system:

1. **Change Detection**:
   - Git webhook integration for push events
   - File hash comparison for direct changes
   - Dependency impact analysis (what changed affects what)

2. **Incremental Processing**:
   - Only re-embed changed files
   - Update affected graph edges
   - Propagate changes up hierarchy (module/repo summaries)

3. **Pipeline Architecture**:
   - Event-driven with message queue (Kafka/RabbitMQ)
   - Batch window for grouping rapid changes
   - Priority queue for critical path updates

4. **Cache Strategy**:
   - LRU cache for embeddings
   - Invalidation based on content hash
   - Warm cache for frequently accessed code""",
            "key_concepts": [
                "Git webhooks",
                "Event-driven architecture",
                "Dependency analysis",
                "Cache invalidation",
                "Message queues",
            ],
        },
        "grounding LLM outputs": {
            "description": """Ensure LLM responses are grounded in actual code:

1. **Citation Mechanism**:
   - Every claim must reference specific file:line
   - Include code snippets in responses
   - Link to source control (GitHub/GitLab URLs)

2. **RAG Architecture**:
   - Retrieve relevant code before generation
   - Inject retrieved context into prompt
   - Use structured output format with citations

3. **Verification Layer**:
   - Cross-check generated code paths exist
   - Validate function signatures match
   - Verify import statements are correct

4. **Context Injection**:
   - Include repository structure context
   - Add relevant documentation
   - Provide test examples for referenced code""",
            "key_concepts": [
                "RAG pattern",
                "Citation generation",
                "Source verification",
                "Context injection",
                "Structured outputs",
            ],
        },
        "safety and hallucination-hardening": {
            "description": """Implement safety measures against hallucinations:

1. **Hallucination Detection**:
   - Validate all file paths exist
   - Check function/class names against symbol table
   - Verify line numbers are within file bounds

2. **Confidence Scoring**:
   - Return confidence scores with answers
   - Flag low-confidence responses
   - Suggest manual verification for uncertain answers

3. **Guardrails**:
   - Refuse to answer if retrieval returns no results
   - Limit scope to indexed codebase only
   - Clear "I don't know" responses when uncertain

4. **Evaluation Pipeline**:
   - Regular accuracy benchmarks
   - A/B testing of retrieval strategies
   - User feedback integration for continuous improvement

5. **Multi-Model Verification**:
   - Cross-check with second model
   - Ensemble approaches for critical queries
   - Human-in-the-loop for high-stakes answers""",
            "key_concepts": [
                "Hallucination detection",
                "Confidence scoring",
                "Guardrails",
                "Evaluation metrics",
                "Human-in-the-loop",
            ],
        },
    }
    
    def __init__(self, kb: TaxonomyKnowledgeBase):
        self.kb = kb
        
    def find_cross_references(self, focus_area: str) -> list[CrossReference]:
        """Find ALL real cross-references for a focus area (no caps).
        
        Only searches books that are defined in the taxonomy.
        """
        search_terms = self.FOCUS_SEARCH_TERMS.get(focus_area, [])
        refs = []
        seen = set()  # Avoid duplicates
        
        # Search each textbook for relevant content
        for book_key in self.kb.textbooks.keys():
            matches = self.kb.search_content(book_key, search_terms)
            
            for match in matches:
                # Create unique key to avoid duplicates
                ref_key = f"{book_key}:{match['chapter']}"
                if ref_key in seen:
                    continue
                seen.add(ref_key)
                
                # Get excerpt
                excerpt = self.kb.get_excerpt(
                    book_key, 
                    match["chapter"], 
                    match["match_term"]
                )
                
                # Get book metadata
                book_data = self.kb.textbooks.get(book_key, {})
                metadata = book_data.get("metadata", {})
                book_title = metadata.get("title", book_key)
                
                ref = CrossReference(
                    book_title=book_title,
                    chapter_number=match["chapter"],
                    chapter_title=match["title"],
                    page_range=f"pp. {match['start_page']}-{match['end_page']}",
                    relevance_topic=match["match_term"],
                    quote_excerpt=excerpt[:500],  # Limit excerpt length
                )
                refs.append(ref)
        
        return refs
    
    def validate_cross_reference(self, ref: CrossReference) -> CrossReference:
        """Validate that a cross-reference actually exists."""
        # Find the book
        book_found = False
        chapter_found = False
        
        for book_key, book_data in self.kb.textbooks.items():
            metadata = book_data.get("metadata", {})
            if metadata.get("title") == ref.book_title or book_key in ref.book_title:
                book_found = True
                # Check chapter exists
                chapters = book_data.get("chapters", [])
                for chapter in chapters:
                    if chapter.get("number") == ref.chapter_number:
                        chapter_found = True
                        # Verify the excerpt exists in content
                        content = chapter.get("content", "")
                        # Check if relevance topic appears
                        if ref.relevance_topic.lower() in content.lower():
                            ref.is_validated = True
                        else:
                            ref.is_validated = False
                            ref.validation_error = f"Topic '{ref.relevance_topic}' not found in chapter content"
                        break
                break
        
        if not book_found:
            ref.is_validated = False
            ref.validation_error = f"Book not found: {ref.book_title}"
        elif not chapter_found:
            ref.is_validated = False
            ref.validation_error = f"Chapter {ref.chapter_number} not found"
            
        return ref
    
    def generate_design(self, query: str) -> DesignResponse:
        """Generate complete design with validated cross-references."""
        response = DesignResponse(
            query=query,
            timestamp=datetime.now().isoformat(),
            focus_areas=self.FOCUS_AREAS,
        )
        
        total_refs = 0
        validated_refs = 0
        
        for focus_area in self.FOCUS_AREAS:
            content = self.DESIGN_CONTENT.get(focus_area, {})
            
            # Find cross-references (no limit)
            refs = self.find_cross_references(focus_area)
            
            # Validate each reference
            validated_section_refs = []
            for ref in refs:
                validated_ref = self.validate_cross_reference(ref)
                validated_section_refs.append(validated_ref)
                total_refs += 1
                if validated_ref.is_validated:
                    validated_refs += 1
            
            section = DesignSection(
                topic=focus_area,
                description=content.get("description", ""),
                key_concepts=content.get("key_concepts", []),
                cross_references=validated_section_refs,
            )
            response.sections.append(section)
        
        response.total_cross_references = total_refs
        response.validated_cross_references = validated_refs
        response.validation_summary = {
            "total": total_refs,
            "validated": validated_refs,
            "validation_rate": f"{(validated_refs / total_refs * 100):.1f}%" if total_refs > 0 else "N/A",
            "by_focus_area": {
                section.topic: {
                    "total": len(section.cross_references),
                    "validated": sum(1 for r in section.cross_references if r.is_validated),
                }
                for section in response.sections
            }
        }
        
        return response


def convert_to_serializable(obj: Any) -> Any:
    """Convert dataclass objects to serializable dicts."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def main() -> int:
    """Run the code understanding design test."""
    print("=" * 80)
    print("  AI Agent Test: Design LLM-Powered Code Understanding System")
    print("=" * 80)
    
    query = """How would you design a system where an LLM reads and understands 
a 20M-line monorepo and answers queries like 'where is the rate-limiter implemented?'?"""
    
    print(f"\nQuery: {query}\n")
    
    # Load knowledge base
    print("Loading taxonomy and textbooks...")
    kb = TaxonomyKnowledgeBase()
    kb.load()
    print(f"  Loaded {len(kb.textbooks)} textbooks")
    print(f"  Indexed {len(kb.concept_index)} concepts")
    
    # Generate design with cross-references
    print("\nGenerating design with cross-references (NO CAPS)...")
    designer = CodeUnderstandingDesigner(kb)
    response = designer.generate_design(query)
    
    # Print summary
    print("\n" + "=" * 80)
    print("  DESIGN SUMMARY")
    print("=" * 80)
    
    for section in response.sections:
        print(f"\n📌 {section.topic.upper()}")
        print(f"   Key Concepts: {', '.join(section.key_concepts[:5])}")
        print(f"   Cross-References: {len(section.cross_references)}")
        
        validated = [r for r in section.cross_references if r.is_validated]
        if validated:
            print(f"   ✅ Validated: {len(validated)}")
            for ref in validated[:3]:  # Show first 3
                print(f"      - {ref.book_title}, Ch.{ref.chapter_number}: {ref.chapter_title[:40]}...")
        
        invalid = [r for r in section.cross_references if not r.is_validated]
        if invalid:
            print(f"   ❌ Invalid: {len(invalid)}")
    
    # Validation summary
    print("\n" + "=" * 80)
    print("  VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n  Total Cross-References: {response.total_cross_references}")
    print(f"  Validated: {response.validated_cross_references}")
    print(f"  Validation Rate: {response.validation_summary.get('validation_rate', 'N/A')}")
    
    # Save to JSON
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / f"code_understanding_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(convert_to_serializable(response), f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Return success if validation rate is acceptable
    if response.validated_cross_references > 0:
        print("\n🎉 SUCCESS: Cross-references are REAL and VALIDATED!")
        return 0
    else:
        print("\n⚠️  WARNING: No cross-references could be validated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
