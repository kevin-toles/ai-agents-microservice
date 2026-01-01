"""Graph Reference Models for Neo4j Integration.

WBS Reference: WBS-AGT22 Neo4j Graph Integration (AGT22.6)
Acceptance Criteria:
- AC-22.5: Results include metadata for citation generation

Models for representing Neo4j graph query results with citation metadata.
Per TIER_RELATIONSHIP_DIAGRAM.md, relationships are bidirectional (spider web model).

Relationship Types:
- PARALLEL: Same tier level (horizontal traversal)
- PERPENDICULAR: Adjacent tier ±1 (vertical traversal)  
- SKIP_TIER: Non-adjacent tier ±2+ (diagonal traversal)

Pattern: Value Objects (immutable data carriers)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pydantic Schemas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Node Types
# =============================================================================


@dataclass(frozen=True)
class Concept:
    """Concept node from Neo4j graph.
    
    Represents a domain concept linked to chapters and code files.
    
    Attributes:
        concept_id: Unique identifier (e.g., "ddd", "repository-pattern")
        name: Human-readable name
        tier: Tier level (1-3) per TIER_RELATIONSHIP_DIAGRAM.md
        keywords: Associated keywords for search
        description: Optional detailed description
    """
    
    concept_id: str
    name: str
    tier: int
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "tier": self.tier,
            "keywords": list(self.keywords),
            "description": self.description,
        }


@dataclass(frozen=True)
class CodeFileReference:
    """Code file reference linked via Neo4j graph.
    
    Maps concepts to actual code implementations in code-reference-engine.
    
    Attributes:
        file_path: Path within repository (e.g., "backend/ddd/repository.py")
        repo_id: Repository identifier in code-reference-engine
        start_line: Starting line number
        end_line: Ending line number
        language: Programming language
        github_url: Full GitHub URL with line anchors
    """
    
    file_path: str
    repo_id: str
    start_line: int
    end_line: int
    language: str
    github_url: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "repo_id": self.repo_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "github_url": self.github_url,
        }


@dataclass(frozen=True)
class PatternRelationship:
    """Cross-repo pattern relationship from Neo4j.
    
    Represents relationships between design patterns across repositories.
    Per TIER_RELATIONSHIP_DIAGRAM.md, relationships are typed:
    - PARALLEL: Same abstraction level
    - PERPENDICULAR: Adjacent abstraction levels
    - SKIP_TIER: Non-adjacent levels
    
    Attributes:
        source_pattern: Source pattern name (e.g., "saga")
        related_pattern: Related pattern name (e.g., "event-sourcing")
        relationship_type: Type of relationship
        repos: List of repositories where relationship exists
        similarity_score: Semantic similarity (0.0-1.0)
    """
    
    source_pattern: str
    related_pattern: str
    relationship_type: Literal["PARALLEL", "PERPENDICULAR", "SKIP_TIER"]
    repos: list[str] = field(default_factory=list)
    similarity_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_pattern": self.source_pattern,
            "related_pattern": self.related_pattern,
            "relationship_type": self.relationship_type,
            "repos": list(self.repos),
            "similarity_score": self.similarity_score,
        }


@dataclass(frozen=True)
class ChapterReference:
    """Chapter reference from Neo4j graph.
    
    Represents a book chapter with tier assignment.
    
    Attributes:
        chapter_id: Unique chapter identifier
        title: Chapter title
        book_id: Parent book identifier
        tier: Tier level (1-3)
        keywords: Chapter keywords for search
        summary: Optional chapter summary
    """
    
    chapter_id: str
    title: str
    book_id: str
    tier: int
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "book_id": self.book_id,
            "tier": self.tier,
            "keywords": list(self.keywords),
            "summary": self.summary,
        }


# =============================================================================
# Generic Graph Reference
# =============================================================================


@dataclass
class GraphReference:
    """Generic reference to any Neo4j graph node.
    
    Used for citation generation from graph query results.
    
    Attributes:
        node_type: Neo4j label (Concept, Chapter, Book, etc.)
        node_id: Unique node identifier
        name: Human-readable name
        source_query: Query that produced this reference
        metadata: Additional node properties for citation
    """
    
    node_type: str
    node_id: str
    name: str
    source_query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_citation(self) -> str:
        """Generate citation string for this reference.
        
        Returns:
            Citation string suitable for footnotes.
        """
        tier = self.metadata.get("tier", "")
        tier_str = f" (Tier {tier})" if tier else ""
        return f"[{self.node_type}: {self.name}{tier_str}]"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "name": self.name,
            "source_query": self.source_query,
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Graph Query Result Containers
# =============================================================================


@dataclass
class ConceptQueryResult:
    """Result container for concept queries.
    
    Wraps query results with metadata for citation tracking.
    """
    
    query: str
    chapter_id: str
    concepts: list[Concept] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        """Number of concepts found."""
        return len(self.concepts)
    
    def get_graph_references(self) -> list[GraphReference]:
        """Convert concepts to generic GraphReferences."""
        return [
            GraphReference(
                node_type="Concept",
                node_id=c.concept_id,
                name=c.name,
                source_query=self.query,
                metadata={"tier": c.tier, "keywords": c.keywords},
            )
            for c in self.concepts
        ]


@dataclass
class CodeQueryResult:
    """Result container for code file queries.
    
    Wraps query results with metadata for citation tracking.
    """
    
    query: str
    concept: str
    files: list[CodeFileReference] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        """Number of files found."""
        return len(self.files)
    
    def get_graph_references(self) -> list[GraphReference]:
        """Convert file references to generic GraphReferences."""
        return [
            GraphReference(
                node_type="CodeFile",
                node_id=f.file_path,
                name=f.file_path,
                source_query=self.query,
                metadata={
                    "repo_id": f.repo_id,
                    "start_line": f.start_line,
                    "end_line": f.end_line,
                    "github_url": f.github_url,
                },
            )
            for f in self.files
        ]


@dataclass
class PatternQueryResult:
    """Result container for pattern relationship queries.
    
    Wraps query results with metadata for citation tracking.
    """
    
    query: str
    source_pattern: str
    relationships: list[PatternRelationship] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        """Number of relationships found."""
        return len(self.relationships)
    
    def get_graph_references(self) -> list[GraphReference]:
        """Convert relationships to generic GraphReferences."""
        return [
            GraphReference(
                node_type="Pattern",
                node_id=f"{r.source_pattern}->{r.related_pattern}",
                name=r.related_pattern,
                source_query=self.query,
                metadata={
                    "relationship_type": r.relationship_type,
                    "repos": r.repos,
                    "similarity_score": r.similarity_score,
                },
            )
            for r in self.relationships
        ]
