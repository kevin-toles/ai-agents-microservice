"""Schemas for summarize_content function.

WBS-AGT7: summarize_content Function schemas.

Acceptance Criteria:
- AC-7.1: Generates summaries with citation markers [^N]
- AC-7.2: Returns CitedContent with footnotes list
- AC-7.5: Supports detail_level parameter (brief/standard/comprehensive)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 2
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from src.schemas.citations import Citation, CitedContent, SourceMetadata


class DetailLevel(str, Enum):
    """Detail level for summarization.
    
    Controls output length and detail:
    - brief: <500 tokens, key points only
    - standard: Balanced summary
    - comprehensive: Detailed with context
    
    Reference: AC-7.5
    """
    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class SummaryStyle(str, Enum):
    """Style of summary output.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - technical: Technical prose
    - executive: Business-focused summary
    - bullets: Bullet-point format
    """
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    BULLETS = "bullets"


class SummarizeContentInput(BaseModel):
    """Input schema for summarize_content function.
    
    Reference: AC-7.1, AC-7.5
    """
    content: str = Field(
        ...,
        description="Content to summarize",
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.STANDARD,
        description="Level of detail (brief/standard/comprehensive)",
    )
    target_tokens: Optional[int] = Field(
        default=None,
        description="Target output token count (overrides detail_level if set)",
    )
    preserve: list[str] = Field(
        default_factory=list,
        description="Concepts that must be included in summary",
    )
    style: SummaryStyle = Field(
        default=SummaryStyle.TECHNICAL,
        description="Style of summary output",
    )
    sources: list[SourceMetadata] = Field(
        default_factory=list,
        description="Source metadata for citation generation",
    )


class SummarizeContentOutput(BaseModel):
    """Output schema for summarize_content function.
    
    Reference: AC-7.1, AC-7.2
    """
    summary: str = Field(
        ...,
        description="Human-readable summary with [^N] citation markers",
    )
    footnotes: list[Citation] = Field(
        default_factory=list,
        description="List of citations matching [^N] markers",
    )
    invariants: list[str] = Field(
        default_factory=list,
        description="Key facts preserved in summary (for validation)",
    )
    compression_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Output/input token ratio (can exceed 1.0 with citation markers)",
    )
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Token count of summary",
    )
    
    def to_cited_content(self) -> CitedContent:
        """Convert to CitedContent for pipeline handoff.
        
        Returns:
            CitedContent with text and citations
        """
        return CitedContent(
            text=self.summary,
            citations=self.footnotes,
        )
