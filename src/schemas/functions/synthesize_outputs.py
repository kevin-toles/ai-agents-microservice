"""Schemas for synthesize_outputs function.

WBS-AGT11: synthesize_outputs Function schemas.

Acceptance Criteria:
- AC-11.1: Combines multiple outputs into coherent whole
- AC-11.2: Returns SynthesizedOutput with merged_content, source_map
- AC-11.5: Preserves citations from input sources

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 6

Anti-Pattern Compliance:
- AP-1.5: No mutable default arguments (uses Field(default_factory=list))
- S3516: Consistent return types
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.schemas.citations import Citation


# =============================================================================
# Enums
# =============================================================================

class SynthesisStrategy(str, Enum):
    """Strategy for combining multiple outputs.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - merge: Concatenate and deduplicate content
    - reconcile: Resolve conflicts between outputs
    - vote: Select most common/agreed content
    """
    MERGE = "merge"
    RECONCILE = "reconcile"
    VOTE = "vote"


class ConflictPolicy(str, Enum):
    """Policy for handling conflicts between outputs.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - first_wins: Use content from first source
    - consensus: Attempt to find common ground
    - flag: Mark conflict for human review
    """
    FIRST_WINS = "first_wins"
    CONSENSUS = "consensus"
    FLAG = "flag"


# =============================================================================
# Supporting Models
# =============================================================================

class OutputItem(BaseModel):
    """Individual output item to synthesize.
    
    Represents a single output from a prior pipeline stage
    that will be combined with others.
    
    Attributes:
        content: The text content to synthesize
        source_id: Unique identifier for provenance tracking
        citations: Optional list of citations embedded in content
        metadata: Optional additional context
    
    Example:
        >>> item = OutputItem(
        ...     content="Chapter 1 discusses DDD patterns[^1].",
        ...     source_id="ch1_summary",
        ...     citations=[Citation(marker=1, source=...)],
        ... )
    """
    content: str = Field(
        ...,
        description="Text content to synthesize",
    )
    source_id: str = Field(
        ...,
        description="Unique identifier for provenance tracking",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations embedded in the content",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (chapter, section, etc.)",
    )


class Conflict(BaseModel):
    """Represents a conflict between synthesized sources.
    
    Captures disagreements or contradictions found during synthesis
    that may require human review or additional resolution.
    
    Attributes:
        section: Identifier for the conflicting section
        source_ids: List of source IDs involved in conflict
        description: Human-readable description of the conflict
        resolution: Optional resolution applied
    
    Example:
        >>> conflict = Conflict(
        ...     section="error_handling",
        ...     source_ids=["model_a", "model_b"],
        ...     description="Conflicting approaches to error handling",
        ...     resolution="Used Result type approach per GUIDELINES",
        ... )
    """
    section: str = Field(
        ...,
        description="Identifier for the conflicting section",
    )
    source_ids: list[str] = Field(
        ...,
        description="List of source IDs involved in the conflict",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the conflict",
    )
    resolution: str | None = Field(
        default=None,
        description="Resolution applied to resolve the conflict",
    )


# =============================================================================
# Input Schema - AC-11.1
# =============================================================================

class SynthesizeOutputsInput(BaseModel):
    """Input schema for synthesize_outputs function.
    
    Accepts multiple outputs to combine into a coherent whole.
    
    Reference: AC-11.1, AGENT_FUNCTIONS_ARCHITECTURE.md
    
    Attributes:
        outputs: List of OutputItem to synthesize (min 2)
        synthesis_strategy: Strategy for combining outputs
        conflict_policy: Policy for handling conflicts
    
    Example:
        >>> input_data = SynthesizeOutputsInput(
        ...     outputs=[
        ...         OutputItem(content="Summary 1", source_id="ch1"),
        ...         OutputItem(content="Summary 2", source_id="ch2"),
        ...     ],
        ...     synthesis_strategy=SynthesisStrategy.MERGE,
        ... )
    """
    outputs: list[OutputItem] = Field(
        ...,
        description="List of outputs to synthesize (minimum 2)",
        min_length=2,
    )
    synthesis_strategy: SynthesisStrategy = Field(
        default=SynthesisStrategy.MERGE,
        description="Strategy for combining outputs: merge, reconcile, vote",
    )
    conflict_policy: ConflictPolicy = Field(
        default=ConflictPolicy.FIRST_WINS,
        description="Policy for handling conflicts: first_wins, consensus, flag",
    )

    @field_validator("outputs")
    @classmethod
    def validate_minimum_outputs(cls, v: list[OutputItem]) -> list[OutputItem]:
        """Validate at least 2 outputs for synthesis.
        
        Synthesis requires multiple inputs to combine.
        """
        if len(v) < 2:
            msg = "Synthesis requires at least 2 outputs to combine"
            raise ValueError(msg)
        return v


# =============================================================================
# Output Schema - AC-11.2, AC-11.5
# =============================================================================

class SynthesizedOutput(BaseModel):
    """Output schema for synthesize_outputs function.
    
    Contains the merged content with provenance tracking and citations.
    
    Reference: AC-11.2, AC-11.5
    
    Attributes:
        merged_content: The synthesized text combining all inputs
        source_map: Maps sections to their source IDs
        citations: Merged and renumbered citations
        agreement_score: Score indicating source consensus (0.0-1.0)
        conflicts: List of detected conflicts
    
    Example:
        >>> output = SynthesizedOutput(
        ...     merged_content="Combined summary with [^1] and [^2].",
        ...     source_map={"para_1": ["ch1"], "para_2": ["ch2"]},
        ...     citations=[...],
        ...     agreement_score=0.85,
        ... )
    """
    merged_content: str = Field(
        ...,
        description="Synthesized text combining all input outputs",
    )
    source_map: dict[str, list[str]] = Field(
        ...,
        description="Maps content sections to their source IDs for provenance",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Merged and renumbered citations from all inputs",
    )
    agreement_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score indicating consensus among sources (0.0-1.0)",
    )
    conflicts: list[Conflict] = Field(
        default_factory=list,
        description="List of detected conflicts during synthesis",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "merged_content": "The book covers domain modeling[^1] and strategic patterns[^2].",
                    "source_map": {
                        "section_1": ["ch1_summary"],
                        "section_2": ["ch2_summary"],
                    },
                    "citations": [],
                    "agreement_score": 0.92,
                    "conflicts": [],
                },
            ],
        },
    }
