"""GroundedResponse schema for Cross-Reference Pipeline.

WBS Reference: WBS-KB6 - Cross-Reference Pipeline Orchestration
Task: KB6.1 - Create GroundedResponse schema

Acceptance Criteria:
- AC-KB6.4: Final output is GroundedResponse with content, citations, confidence, metadata
- AC-KB6.5: Metadata includes: cycles_used, participants, sources_consulted, processing_time

Exit Criteria:
- pytest tests/unit/schemas/test_grounded_response.py passes

Anti-Patterns Avoided:
- S1192: String constants at module level
- Frozen Pydantic models for immutability
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_CONFIDENCE = 0.0
_CONST_MIN_CYCLES = 1
_CONST_MIN_PROCESSING_TIME = 0.0


# =============================================================================
# TerminationReason Enum (AC-KB6.3)
# =============================================================================


class TerminationReason(str, Enum):
    """Reasons why the pipeline terminated.
    
    AC-KB6.3: Pipeline terminates when:
    - agreement reached OR
    - max_cycles OR
    - validation passed
    """

    AGREEMENT_REACHED = "agreement_reached"
    MAX_CYCLES = "max_cycles"
    VALIDATION_PASSED = "validation_passed"
    ERROR = "error"


# =============================================================================
# CitationEntry (AC-KB6.4)
# =============================================================================


class CitationEntry(BaseModel):
    """A single citation entry with provenance tracking.
    
    AC-KB6.4: Citations in GroundedResponse.
    
    Attributes:
        marker: Footnote marker number (e.g., 1 for [^1])
        source: Source identifier (file path, book title, node ID)
        source_type: Type of source (code, book, graph, schema, internal_doc)
        lines: Optional line numbers for code citations
        participant_id: Optional participant who contributed this citation
        cycle_number: Optional cycle number when citation was added
    """

    model_config = ConfigDict(frozen=True)

    marker: int = Field(
        ...,
        ge=1,
        description="Footnote marker number (e.g., 1 for [^1])",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="Source identifier (file path, book title, node ID)",
    )
    source_type: str = Field(
        ...,
        description="Type of source (code, book, graph, schema, internal_doc)",
    )
    lines: str | None = Field(
        default=None,
        description="Optional line numbers for code citations (e.g., '42-56')",
    )
    participant_id: str | None = Field(
        default=None,
        description="Participant who contributed this citation",
    )
    cycle_number: int | None = Field(
        default=None,
        ge=1,
        description="Cycle number when citation was added",
    )


# =============================================================================
# GroundedResponseMetadata (AC-KB6.5)
# =============================================================================


class GroundedResponseMetadata(BaseModel):
    """Metadata for a grounded response.
    
    AC-KB6.5: Metadata includes cycles_used, participants, sources_consulted, processing_time.
    
    Attributes:
        cycles_used: Number of discussion cycles executed
        participants: List of participant IDs involved
        sources_consulted: List of source types consulted
        processing_time_seconds: Total processing time
        termination_reason: Why the pipeline terminated
        agreement_score: Final agreement score (0.0-1.0)
        validation_passed: Whether audit validation passed
    """

    model_config = ConfigDict(frozen=True)

    cycles_used: int = Field(
        ...,
        ge=_CONST_MIN_CYCLES,
        description="Number of discussion cycles executed",
    )
    participants: list[str] = Field(
        ...,
        min_length=1,
        description="List of participant IDs involved",
    )
    sources_consulted: list[str] = Field(
        ...,
        description="List of source types consulted (code, books, graph)",
    )
    processing_time_seconds: float = Field(
        ...,
        ge=_CONST_MIN_PROCESSING_TIME,
        description="Total processing time in seconds",
    )
    termination_reason: TerminationReason | None = Field(
        default=None,
        description="Why the pipeline terminated",
    )
    agreement_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Final agreement score between participants",
    )
    validation_passed: bool | None = Field(
        default=None,
        description="Whether audit validation passed",
    )


# =============================================================================
# GroundedResponse (AC-KB6.4, AC-KB6.5)
# =============================================================================


class GroundedResponse(BaseModel):
    """Final output from the Cross-Reference Pipeline.
    
    AC-KB6.4: GroundedResponse with content, citations, confidence, metadata.
    AC-KB6.5: Metadata includes cycles_used, participants, sources_consulted, processing_time.
    
    Attributes:
        content: The grounded response content with citation markers [^N]
        citations: List of citation entries with provenance
        confidence: Overall confidence score (0.0-1.0)
        metadata: Pipeline execution metadata
        query: Original query (optional)
        footnotes: Formatted Chicago-style footnotes (optional)
    
    Example:
        >>> response = GroundedResponse(
        ...     content="The repository pattern is implemented in src/repository.py [^1]",
        ...     citations=[CitationEntry(marker=1, source="src/repository.py", source_type="code")],
        ...     confidence=0.92,
        ...     metadata=GroundedResponseMetadata(
        ...         cycles_used=3,
        ...         participants=["qwen2.5-7b", "deepseek-r1-7b"],
        ...         sources_consulted=["code", "books"],
        ...         processing_time_seconds=2.5,
        ...     ),
        ... )
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(
        ...,
        description="The grounded response content with citation markers [^N]",
    )
    citations: list[CitationEntry] = Field(
        default_factory=list,
        description="List of citation entries with provenance",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0.0-1.0)",
    )
    metadata: GroundedResponseMetadata = Field(
        ...,
        description="Pipeline execution metadata",
    )
    query: str | None = Field(
        default=None,
        description="Original query",
    )
    footnotes: str | None = Field(
        default=None,
        description="Formatted Chicago-style footnotes",
    )


__all__ = [
    "CitationEntry",
    "GroundedResponse",
    "GroundedResponseMetadata",
    "TerminationReason",
]
