"""Graph Citation Mapper.

WBS Reference: WBS-AGT22 Neo4j Graph Integration (AGT22.7)
Acceptance Criteria:
- AC-22.5: Results include metadata for citation generation

Maps Neo4j graph query results (Concept, CodeFileReference, PatternRelationship)
to citation schema used by the multi-stage enrichment pipeline.

Pattern: Mapper/Transformer (CODING_PATTERNS_ANALYSIS.md)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citations subsystem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.graph_models import (
        CodeFileReference,
        Concept,
        PatternRelationship,
    )


@dataclass
class GraphCitation:
    """Citation for a graph reference.
    
    Schema compatible with downstream pipeline citation requirements.
    
    Attributes:
        source_type: Always "graph" for graph references
        node_type: Neo4j node label (Concept, CodeFile, Pattern, etc.)
        node_id: Unique node identifier
        name: Human-readable name
        description: Citation description
        file_path: Optional file path for code citations
        github_url: Optional GitHub URL for code citations
        tier: Optional tier level for concepts
        metadata: Additional properties for citation
    """
    
    source_type: str = "graph"
    node_type: str = ""
    node_id: str = ""
    name: str = ""
    description: str = ""
    file_path: str | None = None
    github_url: str | None = None
    tier: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, str | int | None | dict[str, str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_type": self.source_type,
            "node_type": self.node_type,
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "github_url": self.github_url,
            "tier": self.tier,
            "metadata": self.metadata,
        }
    
    def to_markdown(self) -> str:
        """Format citation as markdown.
        
        Returns:
            Markdown string with link if GitHub URL available
        """
        if self.github_url:
            return f"[{self.name}]({self.github_url})"
        elif self.file_path:
            return f"`{self.file_path}`"
        else:
            tier_str = f" (Tier {self.tier})" if self.tier else ""
            return f"[{self.node_type}: {self.name}{tier_str}]"


# =============================================================================
# Conversion Functions
# =============================================================================


def concept_to_citation(concept: Concept) -> GraphCitation:
    """Convert Concept to GraphCitation.
    
    Args:
        concept: Concept from Neo4j query
    
    Returns:
        GraphCitation with concept metadata
    """
    return GraphCitation(
        source_type="graph",
        node_type="Concept",
        node_id=concept.concept_id,
        name=concept.name,
        description=f"Concept: {concept.name}",
        tier=concept.tier,
        metadata={
            "keywords": ", ".join(concept.keywords) if concept.keywords else "",
        },
    )


def code_file_to_citation(ref: CodeFileReference) -> GraphCitation:
    """Convert CodeFileReference to GraphCitation.
    
    Args:
        ref: CodeFileReference from Neo4j query
    
    Returns:
        GraphCitation with code file metadata
    """
    return GraphCitation(
        source_type="graph",
        node_type="CodeFile",
        node_id=ref.file_path,
        name=ref.file_path,
        description=f"Code: {ref.file_path}#L{ref.start_line}-L{ref.end_line}",
        file_path=ref.file_path,
        github_url=ref.github_url,
        metadata={
            "repo_id": ref.repo_id,
            "language": ref.language,
            "start_line": str(ref.start_line),
            "end_line": str(ref.end_line),
        },
    )


def pattern_to_citation(rel: PatternRelationship) -> GraphCitation:
    """Convert PatternRelationship to GraphCitation.
    
    Args:
        rel: PatternRelationship from Neo4j query
    
    Returns:
        GraphCitation with pattern relationship metadata
    """
    repos_str = ", ".join(rel.repos) if rel.repos else "unknown"
    
    return GraphCitation(
        source_type="graph",
        node_type="Pattern",
        node_id=f"{rel.source_pattern}->{rel.related_pattern}",
        name=rel.related_pattern,
        description=(
            f"{rel.source_pattern} → {rel.related_pattern} "
            f"({rel.relationship_type}, similarity: {rel.similarity_score:.2f})"
        ),
        metadata={
            "source_pattern": rel.source_pattern,
            "relationship_type": rel.relationship_type,
            "repos": repos_str,
            "similarity_score": f"{rel.similarity_score:.2f}",
        },
    )


# =============================================================================
# Batch Conversion
# =============================================================================


def concepts_to_citations(concepts: list[Concept]) -> list[GraphCitation]:
    """Convert list of Concepts to GraphCitations.
    
    Args:
        concepts: List of Concept objects
    
    Returns:
        List of GraphCitation objects
    """
    return [concept_to_citation(c) for c in concepts]


def code_files_to_citations(refs: list[CodeFileReference]) -> list[GraphCitation]:
    """Convert list of CodeFileReferences to GraphCitations.
    
    Args:
        refs: List of CodeFileReference objects
    
    Returns:
        List of GraphCitation objects
    """
    return [code_file_to_citation(r) for r in refs]


def patterns_to_citations(rels: list[PatternRelationship]) -> list[GraphCitation]:
    """Convert list of PatternRelationships to GraphCitations.
    
    Args:
        rels: List of PatternRelationship objects
    
    Returns:
        List of GraphCitation objects
    """
    return [pattern_to_citation(r) for r in rels]


# =============================================================================
# Formatting Functions
# =============================================================================


def format_graph_citations_for_prompt(citations: list[GraphCitation]) -> str:
    """Format graph citations for LLM prompt inclusion.
    
    Creates a markdown-formatted string suitable for inclusion
    in agent prompts as supporting graph references.
    
    Args:
        citations: List of GraphCitation objects
    
    Returns:
        Markdown-formatted string with citation links
    """
    if not citations:
        return ""
    
    lines = ["## Graph References\n"]
    
    # Group by node type
    by_type: dict[str, list[GraphCitation]] = {}
    for c in citations:
        by_type.setdefault(c.node_type, []).append(c)
    
    for node_type, type_citations in by_type.items():
        lines.append(f"### {node_type}s\n")
        
        for i, c in enumerate(type_citations, 1):
            lines.append(f"{i}. {c.to_markdown()}")
            if c.description:
                lines.append(f"   - {c.description}")
        
        lines.append("")
    
    return "\n".join(lines)
