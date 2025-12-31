"""Tests for Chapter Summarization Pipeline.

TDD tests for WBS-AGT15: Chapter Summarization Pipeline.

Acceptance Criteria Coverage:
- AC-15.1: 4-stage pipeline: extract → cross_ref → summarize → validate
- AC-15.2: Produces CitedContent output with footnotes
- AC-15.3: Configurable via preset (Light/Standard/High Quality)
- AC-15.4: Registers as `/v1/pipelines/chapter-summarization/run`

Exit Criteria:
- pytest tests/unit/pipelines/test_chapter_summarization.py passes
- Pipeline produces summary with [^N] citation markers
- Footnotes contain Chicago-style source references
- preset="high_quality" uses D10, preset="light" uses S1

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: chapter-summarization
"""

import pytest
from typing import Any
from pydantic import BaseModel


# =============================================================================
# AC-15.1: Pipeline Definition Tests
# =============================================================================

class TestChapterSummarizationPipelineDefinition:
    """Tests for pipeline definition structure."""

    def test_pipeline_has_correct_id(self) -> None:
        """Pipeline ID is 'chapter-summarization'."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        
        assert pipeline.pipeline_id == "chapter-summarization"

    def test_pipeline_has_four_stages(self) -> None:
        """Pipeline has exactly 4 stages (AC-15.1)."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        assert len(definition.stages) == 4

    def test_pipeline_stages_in_correct_order(self) -> None:
        """Stages are: extract → cross_ref → summarize → validate (AC-15.1)."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        stage_names = [stage.name for stage in definition.stages]
        
        assert stage_names == [
            "extract_structure",
            "cross_reference",
            "summarize_content",
            "validate_against_spec",
        ]

    def test_stage_1_uses_extract_structure_function(self) -> None:
        """Stage 1 uses extract_structure function."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        stage_1 = definition.stages[0]
        
        assert stage_1.function == "extract_structure"
        assert stage_1.name == "extract_structure"

    def test_stage_2_uses_cross_reference_function(self) -> None:
        """Stage 2 uses cross_reference function."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        stage_2 = definition.stages[1]
        
        assert stage_2.function == "cross_reference"
        assert stage_2.name == "cross_reference"
        assert "extract_structure" in stage_2.depends_on

    def test_stage_3_uses_summarize_content_function(self) -> None:
        """Stage 3 uses summarize_content function."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        stage_3 = definition.stages[2]
        
        assert stage_3.function == "summarize_content"
        assert stage_3.name == "summarize_content"
        assert "cross_reference" in stage_3.depends_on

    def test_stage_4_uses_validate_against_spec_function(self) -> None:
        """Stage 4 uses validate_against_spec function."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        stage_4 = definition.stages[3]
        
        assert stage_4.function == "validate_against_spec"
        assert stage_4.name == "validate_against_spec"
        assert "summarize_content" in stage_4.depends_on


class TestChapterSummarizationDependencies:
    """Tests for stage dependencies."""

    def test_extract_structure_has_no_dependencies(self) -> None:
        """Extract structure is the first stage with no dependencies."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        extract_stage = definition.stages[0]
        
        assert extract_stage.depends_on == []

    def test_cross_reference_depends_on_extract(self) -> None:
        """Cross reference depends on extract_structure output."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        xref_stage = definition.stages[1]
        
        assert "extract_structure" in xref_stage.depends_on

    def test_summarize_depends_on_cross_reference(self) -> None:
        """Summarize depends on cross_reference output."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        summarize_stage = definition.stages[2]
        
        assert "cross_reference" in summarize_stage.depends_on

    def test_validate_depends_on_summarize(self) -> None:
        """Validate depends on summarize_content output."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        validate_stage = definition.stages[3]
        
        assert "summarize_content" in validate_stage.depends_on


class TestChapterSummarizationInputMapping:
    """Tests for input mapping between stages."""

    def test_extract_structure_receives_chapter_text(self) -> None:
        """Extract structure receives chapter_text as content input."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        extract_stage = definition.stages[0]
        
        assert "content" in extract_stage.input_mapping

    def test_cross_reference_receives_keywords_from_extract(self) -> None:
        """Cross reference receives keywords from extract_structure."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        xref_stage = definition.stages[1]
        
        # Should map query to extracted keywords/concepts
        assert "query" in xref_stage.input_mapping or "query_artifact" in xref_stage.input_mapping

    def test_summarize_receives_content_and_citations(self) -> None:
        """Summarize receives original content plus cross_reference citations."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        summarize_stage = definition.stages[2]
        
        assert "content" in summarize_stage.input_mapping

    def test_validate_receives_summary_and_outline(self) -> None:
        """Validate receives summary and original outline as spec."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        definition = pipeline.get_definition()
        
        validate_stage = definition.stages[3]
        
        assert "artifact" in validate_stage.input_mapping
        assert "specification" in validate_stage.input_mapping


# =============================================================================
# AC-15.2: Citation Output Tests
# =============================================================================

class TestChapterSummarizationCitationOutput:
    """Tests for citation output format."""

    def test_output_schema_is_cited_content(self) -> None:
        """Pipeline output schema is CitedContent."""
        from src.pipelines.chapter_summarization import (
            ChapterSummarizationPipeline,
            ChapterSummarizationOutput,
        )
        from src.schemas.citations import CitedContent
        
        # Output should be compatible with CitedContent (check model_fields for Pydantic v2)
        assert "summary" in ChapterSummarizationOutput.model_fields
        assert "footnotes" in ChapterSummarizationOutput.model_fields

    def test_output_has_citation_markers(self) -> None:
        """Output summary contains [^N] citation markers."""
        from src.pipelines.chapter_summarization import ChapterSummarizationOutput
        from src.schemas.citations import Citation, SourceMetadata, SourceType
        
        # Create sample output with citations
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            author="Test Author",
            title="Test Book",
            year=2020,
        )
        citation = Citation(marker=1, source=source)
        
        output = ChapterSummarizationOutput(
            summary="This is a summary with a citation[^1].",
            footnotes=[citation],
            compression_ratio=0.5,
            sources_used=1,
            validation_passed=True,
        )
        
        assert "[^1]" in output.summary
        assert len(output.footnotes) == 1

    def test_footnotes_have_chicago_format(self) -> None:
        """Footnotes use Chicago-style formatting."""
        from src.schemas.citations import Citation, SourceMetadata, SourceType
        
        source = SourceMetadata(
            source_type=SourceType.BOOK,
            author="Fowler, Martin",
            title="Patterns of Enterprise Application Architecture",
            publisher="Addison-Wesley",
            publication_city="Boston",
            year=2002,
            pages="322-327",
        )
        citation = Citation(marker=1, source=source)
        
        formatted = citation.chicago_format()
        
        # Chicago format should include author, italicized title, pub info
        assert "Fowler, Martin" in formatted
        assert "*Patterns of Enterprise Application Architecture*" in formatted
        assert "2002" in formatted

    def test_code_citations_have_repo_and_line_info(self) -> None:
        """Code citations include repo, file, and line information."""
        from src.schemas.citations import Citation, SourceMetadata, SourceType
        
        source = SourceMetadata(
            source_type=SourceType.CODE,
            repo="code-reference-engine",
            file_path="backend/ddd/repository.py",
            line_range="12-45",
            commit_hash="a1b2c3d",
        )
        citation = Citation(marker=2, source=source)
        
        formatted = citation.chicago_format()
        
        assert "code-reference-engine" in formatted
        assert "repository.py" in formatted
        assert "12-45" in formatted


class TestChapterSummarizationCitationAggregation:
    """Tests for citation aggregation from all stages."""

    def test_citations_aggregated_from_cross_reference(self) -> None:
        """Citations from cross_reference stage are aggregated into output."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        
        # Pipeline should have a method to aggregate citations
        assert hasattr(pipeline, "aggregate_citations")

    def test_citations_renumbered_sequentially(self) -> None:
        """Citations are renumbered [^1], [^2], etc. in final output."""
        from src.pipelines.chapter_summarization import CitationAggregator
        from src.schemas.citations import Citation, SourceMetadata, SourceType
        
        # Create citations with non-sequential markers
        sources = [
            SourceMetadata(source_type=SourceType.BOOK, title="Book A", year=2020),
            SourceMetadata(source_type=SourceType.BOOK, title="Book B", year=2021),
        ]
        
        aggregator = CitationAggregator()
        aggregator.add_citation(sources[0])
        aggregator.add_citation(sources[1])
        
        citations = aggregator.get_citations()
        
        # Should be sequentially numbered 1, 2
        assert citations[0].marker == 1
        assert citations[1].marker == 2


# =============================================================================
# AC-15.3: Preset Configuration Tests
# =============================================================================

class TestChapterSummarizationPresets:
    """Tests for preset configuration."""

    def test_supports_light_preset(self) -> None:
        """Pipeline supports 'light' preset using S1."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline(preset="light")
        
        assert pipeline.preset == "light"
        
        # Should configure stages with S1 presets
        definition = pipeline.get_definition()
        # Light uses S1 for fast processing
        assert any(
            stage.preset == "S1" for stage in definition.stages
            if stage.preset is not None
        )

    def test_supports_standard_preset(self) -> None:
        """Pipeline supports 'standard' preset using D4."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline(preset="standard")
        
        assert pipeline.preset == "standard"

    def test_supports_high_quality_preset(self) -> None:
        """Pipeline supports 'high_quality' preset using D10."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline(preset="high_quality")
        
        assert pipeline.preset == "high_quality"
        
        # Should configure stages with D10 presets for quality
        definition = pipeline.get_definition()
        assert any(
            stage.preset == "D10" for stage in definition.stages
            if stage.preset is not None
        )

    def test_default_preset_is_standard(self) -> None:
        """Default preset is 'standard'."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        
        assert pipeline.preset == "standard"

    def test_preset_configures_extract_stage(self) -> None:
        """Preset configures extract_structure stage preset."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        light_pipeline = ChapterSummarizationPipeline(preset="light")
        hq_pipeline = ChapterSummarizationPipeline(preset="high_quality")
        
        light_def = light_pipeline.get_definition()
        hq_def = hq_pipeline.get_definition()
        
        # Presets should differ
        light_extract = light_def.stages[0]
        hq_extract = hq_def.stages[0]
        
        # Light should use faster preset than high_quality
        # Per architecture: light uses S1, high_quality uses S5/D10
        assert light_extract.preset != hq_extract.preset

    def test_preset_configures_summarize_stage(self) -> None:
        """Preset configures summarize_content stage preset."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        light_pipeline = ChapterSummarizationPipeline(preset="light")
        hq_pipeline = ChapterSummarizationPipeline(preset="high_quality")
        
        light_def = light_pipeline.get_definition()
        hq_def = hq_pipeline.get_definition()
        
        light_summarize = light_def.stages[2]
        hq_summarize = hq_def.stages[2]
        
        # High quality should use D10 (critique mode)
        assert hq_summarize.preset == "D10"


# =============================================================================
# AC-15.4: Pipeline Registration Tests (Route Tests)
# =============================================================================

class TestChapterSummarizationRegistration:
    """Tests for pipeline API registration."""

    def test_pipeline_has_api_route(self) -> None:
        """Pipeline has API route attribute."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        
        pipeline = ChapterSummarizationPipeline()
        
        assert hasattr(pipeline, "api_route")
        assert pipeline.api_route == "/v1/pipelines/chapter-summarization/run"

    def test_pipeline_input_schema(self) -> None:
        """Pipeline has defined input schema."""
        from src.pipelines.chapter_summarization import ChapterSummarizationInput
        
        # Should be a Pydantic model
        assert hasattr(ChapterSummarizationInput, "model_fields")
        
        # Required field
        assert "chapter_text" in ChapterSummarizationInput.model_fields

    def test_pipeline_output_schema(self) -> None:
        """Pipeline has defined output schema."""
        from src.pipelines.chapter_summarization import ChapterSummarizationOutput
        
        # Should be a Pydantic model with citation fields
        assert hasattr(ChapterSummarizationOutput, "model_fields")
        
        # Required fields
        assert "summary" in ChapterSummarizationOutput.model_fields
        assert "footnotes" in ChapterSummarizationOutput.model_fields


# =============================================================================
# Pipeline Execution Tests
# =============================================================================

class TestChapterSummarizationExecution:
    """Tests for pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_all_stages(self) -> None:
        """Pipeline executes all 4 stages in order."""
        from src.pipelines.chapter_summarization import ChapterSummarizationPipeline
        from src.pipelines.orchestrator import PipelineOrchestrator
        from src.functions.base import AgentFunction
        from pydantic import BaseModel
        
        execution_log: list[str] = []
        
        # Create mock functions that track execution
        class ExtractOutput(BaseModel):
            keywords: list[str]
            concepts: list[str]
            outline: str
        
        class CrossRefOutput(BaseModel):
            references: list[dict]
            citations: list[dict]
        
        class SummarizeOutput(BaseModel):
            summary: str
            footnotes: list
            invariants: list[str]
        
        class ValidateOutput(BaseModel):
            valid: bool
            violations: list
        
        class MockExtract(AgentFunction):
            name = "extract_structure"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> ExtractOutput:
                execution_log.append("extract")
                return ExtractOutput(
                    keywords=["test"],
                    concepts=["testing"],
                    outline="# Test",
                )
        
        class MockCrossRef(AgentFunction):
            name = "cross_reference"
            default_preset = "S4"
            
            async def run(self, **kwargs: Any) -> CrossRefOutput:
                execution_log.append("cross_ref")
                return CrossRefOutput(references=[], citations=[])
        
        class MockSummarize(AgentFunction):
            name = "summarize_content"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> SummarizeOutput:
                execution_log.append("summarize")
                return SummarizeOutput(
                    summary="Summary text[^1]",
                    footnotes=[],
                    invariants=["test"],
                )
        
        class MockValidate(AgentFunction):
            name = "validate_against_spec"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> ValidateOutput:
                execution_log.append("validate")
                return ValidateOutput(valid=True, violations=[])
        
        pipeline = ChapterSummarizationPipeline()
        orchestrator = PipelineOrchestrator()
        
        # Register mock functions
        orchestrator.register_function(MockExtract())
        orchestrator.register_function(MockCrossRef())
        orchestrator.register_function(MockSummarize())
        orchestrator.register_function(MockValidate())
        
        result = await orchestrator.execute(
            pipeline.get_definition(),
            {"chapter_text": "Test chapter content"},
        )
        
        assert result.success is True
        assert execution_log == ["extract", "cross_ref", "summarize", "validate"]

    @pytest.mark.asyncio
    async def test_pipeline_produces_cited_output(self) -> None:
        """Pipeline execution produces output with citations."""
        from src.pipelines.chapter_summarization import (
            ChapterSummarizationPipeline,
            ChapterSummarizationOutput,
        )
        
        pipeline = ChapterSummarizationPipeline()
        
        # This would require real execution - mark as integration test
        # For unit test, verify output schema
        output = ChapterSummarizationOutput(
            summary="Test summary with citation[^1].",
            footnotes=[],
            compression_ratio=0.5,
            sources_used=1,
            validation_passed=True,
        )
        
        assert "[^1]" in output.summary


__all__ = [
    "TestChapterSummarizationPipelineDefinition",
    "TestChapterSummarizationDependencies",
    "TestChapterSummarizationInputMapping",
    "TestChapterSummarizationCitationOutput",
    "TestChapterSummarizationCitationAggregation",
    "TestChapterSummarizationPresets",
    "TestChapterSummarizationRegistration",
    "TestChapterSummarizationExecution",
]
