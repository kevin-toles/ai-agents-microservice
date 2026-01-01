"""Code Citation Mapper.

WBS Reference: WBS-AGT21 Code Reference Engine Client (AGT21.7)
Acceptance Criteria:
- AC-21.5: Returns CodeContext with citations for downstream

Maps CodeContext from code reference search to Citation schema
used by the multi-stage enrichment pipeline.

Pattern: Mapper/Transformer (CODING_PATTERNS_ANALYSIS.md)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citations subsystem
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.code_reference import CodeContext


@dataclass
class CodeCitation:
    """Citation for a code reference.
    
    Schema compatible with downstream pipeline citation requirements.
    
    Attributes:
        source_type: Always "code" for code references
        url: GitHub URL with line anchors
        file_path: Path to file within repository
        repo_id: Repository identifier
        start_line: Starting line number
        end_line: Ending line number
        language: Programming language
        content_preview: First N characters of code content
        score: Semantic similarity score
    """
    
    source_type: str = "code"
    url: str = ""
    file_path: str = ""
    repo_id: str = ""
    start_line: int = 1
    end_line: int = 1
    language: str = "text"
    content_preview: str = ""
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, str | int | float | dict[str, str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_type": self.source_type,
            "url": self.url,
            "file_path": self.file_path,
            "repo_id": self.repo_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "content_preview": self.content_preview,
            "score": self.score,
            "metadata": self.metadata,
        }
    
    def to_markdown(self) -> str:
        """Format citation as markdown link."""
        return f"[{self.file_path}#L{self.start_line}-L{self.end_line}]({self.url})"


def code_context_to_citations(context: CodeContext) -> list[CodeCitation]:
    """Convert CodeContext to list of CodeCitation objects.
    
    Maps each CodeReference in the context to a normalized CodeCitation
    suitable for the downstream enrichment pipeline.
    
    Args:
        context: CodeContext from code reference search
    
    Returns:
        List of CodeCitation objects, empty list if no references
    
    Example:
        >>> context = await client.search("repository pattern")
        >>> citations = code_context_to_citations(context)
        >>> for c in citations:
        ...     print(c.to_markdown())
    """
    citations: list[CodeCitation] = []
    
    for ref in context.primary_references:
        chunk = ref.chunk
        
        # Create content preview (first 200 chars)
        content_preview = chunk.content[:200]
        if len(chunk.content) > 200:
            content_preview += "..."
        
        citation = CodeCitation(
            source_type="code",
            url=ref.source_url,
            file_path=chunk.file_path,
            repo_id=chunk.repo_id,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            language=chunk.language,
            content_preview=content_preview,
            score=chunk.score,
            metadata=_extract_metadata(ref.repo_metadata),
        )
        citations.append(citation)
    
    return citations


def _extract_metadata(repo_metadata: dict[str, str] | None) -> dict[str, str]:
    """Extract relevant metadata fields for citation.
    
    Args:
        repo_metadata: Optional repository metadata dict
    
    Returns:
        Filtered metadata dict with string values only
    """
    if not repo_metadata:
        return {}
    
    # Only include string values suitable for citation display
    allowed_keys = {"name", "domain", "tier", "description"}
    return {
        k: str(v)
        for k, v in repo_metadata.items()
        if k in allowed_keys and v is not None
    }


def format_citations_for_prompt(citations: list[CodeCitation]) -> str:
    """Format citations for LLM prompt inclusion.
    
    Creates a markdown-formatted string suitable for inclusion
    in agent prompts as supporting code references.
    
    Args:
        citations: List of CodeCitation objects
    
    Returns:
        Markdown-formatted string with citation links
    """
    if not citations:
        return ""
    
    lines = ["## Code References\n"]
    
    for i, c in enumerate(citations, 1):
        lines.append(f"{i}. {c.to_markdown()} ({c.language}, score: {c.score:.2f})")
        if c.content_preview:
            # Indent preview as code block
            preview_lines = c.content_preview.split("\n")[:5]  # Max 5 lines
            lines.append(f"   ```{c.language}")
            for line in preview_lines:
                lines.append(f"   {line}")
            lines.append("   ```")
    
    return "\n".join(lines)
