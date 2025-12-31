"""Chapter Summarization Pipeline.

WBS-AGT15: Chapter Summarization Pipeline

Implements a 4-stage pipeline for producing summarized content with citations:
1. extract_structure - Parse chapter to extract keywords, concepts, outline
2. cross_reference - Find related sources via semantic search
3. summarize_content - Generate summary with [^N] citation markers
4. validate_against_spec - Verify summary against original outline

Acceptance Criteria:
- AC-15.1: 4-stage pipeline: extract → cross_ref → summarize → validate
- AC-15.2: Produces CitedContent output with footnotes
- AC-15.3: Configurable via preset (Light/Standard/High Quality)
- AC-15.4: Registers as `/v1/pipelines/chapter-summarization/run`

Preset Configuration:
- light: Uses S1 presets (fast processing)
- standard: Uses D4 presets (balanced) - DEFAULT
- high_quality: Uses D10 presets (critique mode)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: chapter-summarization
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.pipelines.orchestrator import PipelineDefinition, StageDefinition
from src.schemas.citations import Citation, SourceMetadata


# =============================================================================
# Type Aliases
# =============================================================================

PresetType = Literal["light", "standard", "high_quality"]


# =============================================================================
# Preset Configuration
# =============================================================================

# Preset to stage preset mapping
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Preset Configuration
PRESET_MAPPING: dict[PresetType, dict[str, str]] = {
    "light": {
        "extract_structure": "S1",
        "cross_reference": "S1",
        "summarize_content": "S1",
        "validate_against_spec": "S1",
    },
    "standard": {
        "extract_structure": "S5",
        "cross_reference": "S4",
        "summarize_content": "D4",
        "validate_against_spec": "D4",
    },
    "high_quality": {
        "extract_structure": "S5",
        "cross_reference": "S4",
        "summarize_content": "D10",
        "validate_against_spec": "D10",
    },
}


# =============================================================================
# Input/Output Schemas
# =============================================================================

class ChapterSummarizationInput(BaseModel):
    """Input schema for chapter summarization pipeline.

    Attributes:
        chapter_text: The full text of the chapter to summarize
        specification: Optional specification for validation (e.g., outline)
        preset: Quality preset for the summarization
    """

    chapter_text: str = Field(
        ...,
        description="Full text of the chapter to summarize",
        min_length=1,
    )
    specification: str | None = Field(
        default=None,
        description="Optional specification/outline to validate against",
    )
    preset: PresetType = Field(
        default="standard",
        description="Quality preset: light, standard, or high_quality",
    )


class ChapterSummarizationOutput(BaseModel):
    """Output schema for chapter summarization pipeline.

    Produces CitedContent-compatible output with summary and footnotes.

    Attributes:
        summary: Summarized text with [^N] citation markers
        footnotes: List of citations in Chicago format
        compression_ratio: Ratio of output to input length
        sources_used: Number of sources cited
        validation_passed: Whether validation against spec passed
    """

    summary: str = Field(
        ...,
        description="Summarized chapter text with [^N] citation markers",
    )
    footnotes: list[Citation] = Field(
        default_factory=list,
        description="List of citations for the summary",
    )
    compression_ratio: float = Field(
        ...,
        description="Ratio of summary length to original chapter length",
        ge=0.0,
        le=1.0,
    )
    sources_used: int = Field(
        ...,
        description="Number of sources cited in the summary",
        ge=0,
    )
    validation_passed: bool = Field(
        ...,
        description="Whether the summary passed validation against spec",
    )


# =============================================================================
# Citation Aggregator
# =============================================================================

class CitationAggregator:
    """Aggregates and renumbers citations from pipeline stages.

    Collects citations from cross_reference and summarize_content stages,
    renumbers them sequentially [^1], [^2], etc.

    Example:
        >>> aggregator = CitationAggregator()
        >>> aggregator.add_citation(source_metadata)
        >>> citations = aggregator.get_citations()
    """

    def __init__(self) -> None:
        """Initialize the citation aggregator."""
        self._sources: list[SourceMetadata] = []

    def add_citation(self, source: SourceMetadata) -> int:
        """Add a citation source and return its marker number.

        Args:
            source: Source metadata for the citation

        Returns:
            The marker number (1-indexed) for this citation
        """
        self._sources.append(source)
        return len(self._sources)

    def get_citations(self) -> list[Citation]:
        """Get all citations with sequential marker numbers.

        Returns:
            List of Citation objects numbered [^1], [^2], etc.
        """
        return [
            Citation(marker=i + 1, source=source)
            for i, source in enumerate(self._sources)
        ]

    def clear(self) -> None:
        """Clear all collected citations."""
        self._sources.clear()

    def __len__(self) -> int:
        """Return the number of citations collected."""
        return len(self._sources)


# =============================================================================
# Pipeline Definition
# =============================================================================

class ChapterSummarizationPipeline:
    """Chapter Summarization Pipeline.

    4-stage pipeline for producing summarized content with citations:

    Stage 1: extract_structure
        - Input: chapter_text
        - Output: {keywords, concepts, outline}
        - Preset: S1 (light) / S5 (standard/high_quality)

    Stage 2: cross_reference
        - Input: keywords, concepts from Stage 1
        - Output: {citations, related_chapters}
        - Preset: S1 (light) / S4 (standard/high_quality)

    Stage 3: summarize_content
        - Input: chapter_text + citations from Stage 2
        - Output: {summary, footnotes, invariants}
        - Preset: S1 (light) / D4 (standard) / D10 (high_quality)

    Stage 4: validate_against_spec
        - Input: summary from Stage 3, outline from Stage 1
        - Output: {valid, violations}
        - Preset: S1 (light) / D4 (standard) / D10 (high_quality)

    Attributes:
        pipeline_id: Unique identifier for this pipeline
        preset: Quality preset (light, standard, high_quality)
        api_route: API endpoint for this pipeline
    """

    pipeline_id: str = "chapter-summarization"
    api_route: str = "/v1/pipelines/chapter-summarization/run"

    def __init__(self, preset: PresetType = "standard") -> None:
        """Initialize the pipeline with the specified preset.

        Args:
            preset: Quality preset for the summarization.
                - "light": Fast processing with S1 presets
                - "standard": Balanced with D4 presets (default)
                - "high_quality": Maximum quality with D10 presets
        """
        self.preset: PresetType = preset
        self._aggregator = CitationAggregator()

    def get_definition(self) -> PipelineDefinition:
        """Get the pipeline definition with stages configured for the preset.

        Returns:
            PipelineDefinition with 4 configured stages
        """
        stage_presets = PRESET_MAPPING[self.preset]

        return PipelineDefinition(
            name=self.pipeline_id,
            description="Summarize chapter content with citations",
            stages=[
                # Stage 1: Extract Structure
                StageDefinition(
                    name="extract_structure",
                    function="extract_structure",
                    input_mapping={
                        "content": "input.chapter_text",
                    },
                    depends_on=[],
                    output_key="extract_output",
                    preset=stage_presets["extract_structure"],
                ),
                # Stage 2: Cross Reference
                StageDefinition(
                    name="cross_reference",
                    function="cross_reference",
                    input_mapping={
                        "query": "extract_output.keywords",
                        "query_artifact": "extract_output.concepts",
                    },
                    depends_on=["extract_structure"],
                    output_key="xref_output",
                    preset=stage_presets["cross_reference"],
                ),
                # Stage 3: Summarize Content
                StageDefinition(
                    name="summarize_content",
                    function="summarize_content",
                    input_mapping={
                        "content": "input.chapter_text",
                        "citations": "xref_output.citations",
                    },
                    depends_on=["cross_reference"],
                    output_key="summary_output",
                    preset=stage_presets["summarize_content"],
                ),
                # Stage 4: Validate Against Spec
                StageDefinition(
                    name="validate_against_spec",
                    function="validate_against_spec",
                    input_mapping={
                        "artifact": "summary_output.summary",
                        "specification": "extract_output.outline",
                    },
                    depends_on=["summarize_content"],
                    output_key="validate_output",
                    preset=stage_presets["validate_against_spec"],
                ),
            ],
        )

    def aggregate_citations(
        self,
        xref_citations: list[SourceMetadata] | None = None,
        summary_footnotes: list[Citation] | None = None,
    ) -> list[Citation]:
        """Aggregate citations from pipeline stages.

        Collects citations from cross_reference and summarize_content stages,
        renumbers them sequentially.

        Args:
            xref_citations: Citations from cross_reference stage
            summary_footnotes: Footnotes from summarize_content stage

        Returns:
            List of sequentially numbered citations
        """
        self._aggregator.clear()

        # Add cross-reference citations
        if xref_citations:
            for source in xref_citations:
                self._aggregator.add_citation(source)

        # Add summary footnote sources (extract source metadata)
        if summary_footnotes:
            for footnote in summary_footnotes:
                self._aggregator.add_citation(footnote.source)

        return self._aggregator.get_citations()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PRESET_MAPPING",
    "ChapterSummarizationInput",
    "ChapterSummarizationOutput",
    "ChapterSummarizationPipeline",
    "CitationAggregator",
    "PresetType",
]
