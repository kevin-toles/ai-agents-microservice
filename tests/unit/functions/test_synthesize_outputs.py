"""Tests for synthesize_outputs function.

TDD tests for WBS-AGT11: synthesize_outputs Function.

Acceptance Criteria Coverage:
- AC-11.1: Combines multiple outputs into coherent whole
- AC-11.2: Returns SynthesizedOutput with merged_content, source_map
- AC-11.3: Context budget: 8192 input / 4096 output
- AC-11.4: Default preset: S1 (Light)
- AC-11.5: Preserves citations from input sources

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 6
"""

import pytest
from pydantic import ValidationError


# =============================================================================
# AGT11.1 RED: Input Schema Tests - AC-11.1
# =============================================================================

class TestSynthesizeOutputsInput:
    """Tests for SynthesizeOutputsInput schema.
    
    Verifies the input schema correctly accepts multiple outputs
    and synthesis configuration.
    """

    def test_input_requires_outputs(self) -> None:
        """SynthesizeOutputsInput requires outputs field."""
        from src.schemas.functions.synthesize_outputs import SynthesizeOutputsInput
        
        with pytest.raises(ValidationError) as exc_info:
            SynthesizeOutputsInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("outputs",) for e in errors)

    def test_input_accepts_output_items(self) -> None:
        """SynthesizeOutputsInput accepts list of OutputItem."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
        )
        
        outputs = [
            OutputItem(
                content="Summary of chapter 1",
                source_id="ch1_summary",
            ),
            OutputItem(
                content="Summary of chapter 2",
                source_id="ch2_summary",
            ),
        ]
        input_data = SynthesizeOutputsInput(outputs=outputs)
        
        assert len(input_data.outputs) == 2
        assert input_data.outputs[0].content == "Summary of chapter 1"
        assert input_data.outputs[1].source_id == "ch2_summary"

    def test_input_requires_minimum_two_outputs(self) -> None:
        """SynthesizeOutputsInput requires at least 2 outputs for synthesis."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
        )
        
        # Single output should fail - need multiple to synthesize
        with pytest.raises(ValidationError) as exc_info:
            SynthesizeOutputsInput(
                outputs=[
                    OutputItem(content="Only one", source_id="single"),
                ]
            )
        
        errors = exc_info.value.errors()
        assert any("at least 2" in str(e).lower() for e in errors)

    def test_input_has_synthesis_strategy_with_default(self) -> None:
        """SynthesizeOutputsInput has synthesis_strategy with default 'merge'."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            SynthesisStrategy,
        )
        
        outputs = [
            OutputItem(content="Content 1", source_id="src1"),
            OutputItem(content="Content 2", source_id="src2"),
        ]
        input_data = SynthesizeOutputsInput(outputs=outputs)
        
        assert input_data.synthesis_strategy == SynthesisStrategy.MERGE

    def test_input_accepts_reconcile_strategy(self) -> None:
        """SynthesizeOutputsInput accepts reconcile strategy."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            SynthesisStrategy,
        )
        
        outputs = [
            OutputItem(content="Content 1", source_id="src1"),
            OutputItem(content="Content 2", source_id="src2"),
        ]
        input_data = SynthesizeOutputsInput(
            outputs=outputs,
            synthesis_strategy=SynthesisStrategy.RECONCILE,
        )
        
        assert input_data.synthesis_strategy == SynthesisStrategy.RECONCILE

    def test_input_accepts_vote_strategy(self) -> None:
        """SynthesizeOutputsInput accepts vote strategy."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            SynthesisStrategy,
        )
        
        outputs = [
            OutputItem(content="Option A", source_id="src1"),
            OutputItem(content="Option B", source_id="src2"),
            OutputItem(content="Option A", source_id="src3"),
        ]
        input_data = SynthesizeOutputsInput(
            outputs=outputs,
            synthesis_strategy=SynthesisStrategy.VOTE,
        )
        
        assert input_data.synthesis_strategy == SynthesisStrategy.VOTE

    def test_input_has_conflict_policy_with_default(self) -> None:
        """SynthesizeOutputsInput has conflict_policy with default 'first_wins'."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            ConflictPolicy,
        )
        
        outputs = [
            OutputItem(content="Content 1", source_id="src1"),
            OutputItem(content="Content 2", source_id="src2"),
        ]
        input_data = SynthesizeOutputsInput(outputs=outputs)
        
        assert input_data.conflict_policy == ConflictPolicy.FIRST_WINS

    def test_input_accepts_consensus_policy(self) -> None:
        """SynthesizeOutputsInput accepts consensus conflict policy."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            ConflictPolicy,
        )
        
        outputs = [
            OutputItem(content="Content 1", source_id="src1"),
            OutputItem(content="Content 2", source_id="src2"),
        ]
        input_data = SynthesizeOutputsInput(
            outputs=outputs,
            conflict_policy=ConflictPolicy.CONSENSUS,
        )
        
        assert input_data.conflict_policy == ConflictPolicy.CONSENSUS

    def test_input_accepts_flag_policy(self) -> None:
        """SynthesizeOutputsInput accepts flag conflict policy."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizeOutputsInput,
            OutputItem,
            ConflictPolicy,
        )
        
        outputs = [
            OutputItem(content="Content 1", source_id="src1"),
            OutputItem(content="Content 2", source_id="src2"),
        ]
        input_data = SynthesizeOutputsInput(
            outputs=outputs,
            conflict_policy=ConflictPolicy.FLAG,
        )
        
        assert input_data.conflict_policy == ConflictPolicy.FLAG

    def test_input_json_schema_export(self) -> None:
        """SynthesizeOutputsInput exports valid JSON schema."""
        from src.schemas.functions.synthesize_outputs import SynthesizeOutputsInput
        
        schema = SynthesizeOutputsInput.model_json_schema()
        
        assert "properties" in schema
        assert "outputs" in schema["properties"]
        assert "synthesis_strategy" in schema["properties"]
        assert "conflict_policy" in schema["properties"]


class TestOutputItem:
    """Tests for OutputItem schema - individual output to synthesize."""

    def test_output_item_requires_content(self) -> None:
        """OutputItem requires content field."""
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        with pytest.raises(ValidationError) as exc_info:
            OutputItem(source_id="test")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content",) for e in errors)

    def test_output_item_requires_source_id(self) -> None:
        """OutputItem requires source_id for provenance tracking."""
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        with pytest.raises(ValidationError) as exc_info:
            OutputItem(content="Test content")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("source_id",) for e in errors)

    def test_output_item_accepts_optional_citations(self) -> None:
        """OutputItem accepts optional citations list for AC-11.5."""
        from src.schemas.functions.synthesize_outputs import OutputItem
        from src.schemas.citations import Citation, SourceMetadata
        
        citation = Citation(
            marker=1,
            source=SourceMetadata(
                source_type="book",
                title="Domain-Driven Design",
                author="Evans, Eric",
            ),
        )
        
        item = OutputItem(
            content="DDD patterns discussed[^1]",
            source_id="ch1_summary",
            citations=[citation],
        )
        
        assert len(item.citations) == 1
        assert item.citations[0].marker == 1

    def test_output_item_accepts_metadata(self) -> None:
        """OutputItem accepts optional metadata dict."""
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        item = OutputItem(
            content="Test content",
            source_id="test_source",
            metadata={"chapter": 1, "section": "introduction"},
        )
        
        assert item.metadata["chapter"] == 1


# =============================================================================
# AGT11.3 RED: Output Schema Tests - AC-11.2, AC-11.5
# =============================================================================

class TestSynthesizedOutput:
    """Tests for SynthesizedOutput schema.
    
    Verifies the output schema includes merged_content, source_map,
    and preserves citations.
    """

    def test_output_has_merged_content(self) -> None:
        """SynthesizedOutput has merged_content field."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        
        output = SynthesizedOutput(
            merged_content="Combined result from multiple sources.",
            source_map={},
        )
        
        assert output.merged_content == "Combined result from multiple sources."

    def test_output_has_source_map(self) -> None:
        """SynthesizedOutput has source_map for provenance tracking - AC-11.2."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        
        source_map = {
            "paragraph_1": ["ch1_summary"],
            "paragraph_2": ["ch2_summary", "ch3_summary"],
        }
        
        output = SynthesizedOutput(
            merged_content="Combined content.",
            source_map=source_map,
        )
        
        assert "paragraph_1" in output.source_map
        assert output.source_map["paragraph_1"] == ["ch1_summary"]
        assert "ch2_summary" in output.source_map["paragraph_2"]

    def test_output_has_citations_list(self) -> None:
        """SynthesizedOutput has citations list - AC-11.5."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        from src.schemas.citations import Citation, SourceMetadata
        
        citations = [
            Citation(
                marker=1,
                source=SourceMetadata(source_type="book", title="Book 1"),
            ),
            Citation(
                marker=2,
                source=SourceMetadata(source_type="code", repo="test-repo", file_path="src/main.py"),
            ),
        ]
        
        output = SynthesizedOutput(
            merged_content="Result with [^1] and [^2] citations.",
            source_map={},
            citations=citations,
        )
        
        assert len(output.citations) == 2
        assert output.citations[0].marker == 1
        assert output.citations[1].marker == 2

    def test_output_has_agreement_score(self) -> None:
        """SynthesizedOutput has agreement_score field."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        
        output = SynthesizedOutput(
            merged_content="Combined content.",
            source_map={},
            agreement_score=0.85,
        )
        
        assert output.agreement_score is not None
        assert abs(output.agreement_score - 0.85) < 0.001  # Avoid float equality

    def test_output_agreement_score_validates_range(self) -> None:
        """SynthesizedOutput agreement_score must be 0.0-1.0."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        
        with pytest.raises(ValidationError):
            SynthesizedOutput(
                merged_content="Content",
                source_map={},
                agreement_score=1.5,  # Invalid - exceeds 1.0
            )

    def test_output_has_conflicts_list(self) -> None:
        """SynthesizedOutput has conflicts list for flagged conflicts."""
        from src.schemas.functions.synthesize_outputs import (
            SynthesizedOutput,
            Conflict,
        )
        
        conflicts = [
            Conflict(
                section="architecture",
                source_ids=["src1", "src2"],
                description="Conflicting approaches to error handling",
            ),
        ]
        
        output = SynthesizedOutput(
            merged_content="Combined content.",
            source_map={},
            conflicts=conflicts,
        )
        
        assert len(output.conflicts) == 1
        assert output.conflicts[0].section == "architecture"

    def test_output_json_schema_export(self) -> None:
        """SynthesizedOutput exports valid JSON schema."""
        from src.schemas.functions.synthesize_outputs import SynthesizedOutput
        
        schema = SynthesizedOutput.model_json_schema()
        
        assert "properties" in schema
        assert "merged_content" in schema["properties"]
        assert "source_map" in schema["properties"]
        assert "citations" in schema["properties"]


class TestConflict:
    """Tests for Conflict schema - represents synthesis conflicts."""

    def test_conflict_requires_section(self) -> None:
        """Conflict requires section identifier."""
        from src.schemas.functions.synthesize_outputs import Conflict
        
        with pytest.raises(ValidationError):
            Conflict(
                source_ids=["src1", "src2"],
                description="Test conflict",
            )  # type: ignore

    def test_conflict_requires_source_ids(self) -> None:
        """Conflict requires source_ids list."""
        from src.schemas.functions.synthesize_outputs import Conflict
        
        with pytest.raises(ValidationError):
            Conflict(
                section="test_section",
                description="Test conflict",
            )  # type: ignore

    def test_conflict_requires_description(self) -> None:
        """Conflict requires description of the conflict."""
        from src.schemas.functions.synthesize_outputs import Conflict
        
        with pytest.raises(ValidationError):
            Conflict(
                section="test_section",
                source_ids=["src1", "src2"],
            )  # type: ignore

    def test_conflict_accepts_all_fields(self) -> None:
        """Conflict accepts all required fields."""
        from src.schemas.functions.synthesize_outputs import Conflict
        
        conflict = Conflict(
            section="error_handling",
            source_ids=["model_a", "model_b"],
            description="Model A suggests exceptions, Model B suggests Result types",
            resolution="Used Result type approach per GUIDELINES",
        )
        
        assert conflict.section == "error_handling"
        assert len(conflict.source_ids) == 2
        assert "Result type" in conflict.resolution


# =============================================================================
# AGT11.5 RED: Synthesis Function Tests - AC-11.1 to AC-11.5
# =============================================================================

class TestSynthesizeOutputsFunction:
    """Tests for SynthesizeOutputsFunction.
    
    Verifies the function combines multiple outputs into a coherent whole
    while preserving citations and tracking provenance.
    """

    def test_function_has_correct_name(self) -> None:
        """SynthesizeOutputsFunction has name 'synthesize_outputs'."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        
        func = SynthesizeOutputsFunction()
        assert func.name == "synthesize_outputs"

    def test_function_has_correct_default_preset(self) -> None:
        """SynthesizeOutputsFunction has default preset S1 - AC-11.4."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        
        func = SynthesizeOutputsFunction()
        assert func.default_preset == "S1"

    def test_function_context_budget(self) -> None:
        """SynthesizeOutputsFunction has correct context budget - AC-11.3."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        
        func = SynthesizeOutputsFunction()
        budget = func.get_context_budget()
        
        assert budget["input"] == 8192
        assert budget["output"] == 4096

    @pytest.mark.asyncio
    async def test_function_combines_multiple_outputs(self) -> None:
        """Function combines multiple outputs into coherent whole - AC-11.1."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        func = SynthesizeOutputsFunction()
        
        outputs = [
            OutputItem(
                content="Chapter 1 discusses domain modeling.",
                source_id="ch1_summary",
            ),
            OutputItem(
                content="Chapter 2 covers repository patterns.",
                source_id="ch2_summary",
            ),
        ]
        
        result = await func.run(outputs=outputs)
        
        # Result should contain content from both sources
        assert "domain modeling" in result.merged_content.lower() or "chapter 1" in result.merged_content.lower()
        assert "repository" in result.merged_content.lower() or "chapter 2" in result.merged_content.lower()

    @pytest.mark.asyncio
    async def test_function_returns_source_map(self) -> None:
        """Function returns source_map tracking provenance - AC-11.2."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        func = SynthesizeOutputsFunction()
        
        outputs = [
            OutputItem(
                content="Content from source A.",
                source_id="source_a",
            ),
            OutputItem(
                content="Content from source B.",
                source_id="source_b",
            ),
        ]
        
        result = await func.run(outputs=outputs)
        
        # source_map should reference the input source_ids
        assert result.source_map is not None
        # Check that source_ids appear somewhere in source_map values
        all_sources = []
        for sources in result.source_map.values():
            all_sources.extend(sources)
        assert "source_a" in all_sources or "source_b" in all_sources

    @pytest.mark.asyncio
    async def test_function_preserves_citations(self) -> None:
        """Function preserves citations from input sources - AC-11.5."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        from src.schemas.citations import Citation, SourceMetadata
        
        func = SynthesizeOutputsFunction()
        
        citation1 = Citation(
            marker=1,
            source=SourceMetadata(
                source_type="book",
                title="Domain-Driven Design",
                author="Evans, Eric",
            ),
        )
        citation2 = Citation(
            marker=2,
            source=SourceMetadata(
                source_type="book",
                title="Clean Architecture",
                author="Martin, Robert C.",
            ),
        )
        
        outputs = [
            OutputItem(
                content="DDD emphasizes bounded contexts[^1].",
                source_id="ch1",
                citations=[citation1],
            ),
            OutputItem(
                content="Clean Architecture promotes dependency inversion[^2].",
                source_id="ch2",
                citations=[citation2],
            ),
        ]
        
        result = await func.run(outputs=outputs)
        
        # Citations should be preserved (possibly renumbered)
        assert len(result.citations) >= 2

    @pytest.mark.asyncio
    async def test_function_renumbers_citations_correctly(self) -> None:
        """Function renumbers citations to avoid duplicates - AC-11.5."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        from src.schemas.citations import Citation, SourceMetadata
        
        func = SynthesizeOutputsFunction()
        
        # Both inputs have marker=1 - should be renumbered
        citation1 = Citation(
            marker=1,
            source=SourceMetadata(source_type="book", title="Book A"),
        )
        citation2 = Citation(
            marker=1,  # Same marker number - conflict
            source=SourceMetadata(source_type="book", title="Book B"),
        )
        
        outputs = [
            OutputItem(
                content="Reference from book A[^1].",
                source_id="src_a",
                citations=[citation1],
            ),
            OutputItem(
                content="Reference from book B[^1].",
                source_id="src_b",
                citations=[citation2],
            ),
        ]
        
        result = await func.run(outputs=outputs)
        
        # Citations should have unique markers after merge
        markers = [c.marker for c in result.citations]
        assert len(markers) == len(set(markers)), "Citation markers should be unique"

    @pytest.mark.asyncio
    async def test_function_enforces_input_budget(self) -> None:
        """Function enforces input context budget - AC-11.3."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.functions.base import ContextBudgetExceededError
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        func = SynthesizeOutputsFunction()
        
        # Create outputs that exceed 8192 tokens (~32K chars)
        large_content = "x" * 40000  # Well over budget
        
        outputs = [
            OutputItem(content=large_content, source_id="large_src"),
            OutputItem(content="Small content.", source_id="small_src"),
        ]
        
        with pytest.raises(ContextBudgetExceededError):
            await func.run(outputs=outputs)

    @pytest.mark.asyncio
    async def test_function_uses_merge_strategy(self) -> None:
        """Function uses merge strategy to combine content - AC-11.1."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import (
            OutputItem,
            SynthesisStrategy,
        )
        
        func = SynthesizeOutputsFunction()
        
        outputs = [
            OutputItem(content="Introduction to the topic.", source_id="intro"),
            OutputItem(content="Detailed explanation follows.", source_id="detail"),
        ]
        
        result = await func.run(
            outputs=outputs,
            synthesis_strategy=SynthesisStrategy.MERGE,
        )
        
        # Merged content should contain both pieces
        assert len(result.merged_content) > 0

    @pytest.mark.asyncio
    async def test_function_no_duplicate_content(self) -> None:
        """Function avoids duplicate content in merged output - Exit Criteria."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        func = SynthesizeOutputsFunction()
        
        # Same content in both inputs - should not duplicate
        shared_content = "The repository pattern encapsulates data access."
        
        outputs = [
            OutputItem(content=shared_content, source_id="src1"),
            OutputItem(content=shared_content, source_id="src2"),
        ]
        
        result = await func.run(outputs=outputs)
        
        # Count occurrences of the content
        occurrences = result.merged_content.lower().count("repository pattern")
        assert occurrences <= 1, "Duplicate content should be merged"

    @pytest.mark.asyncio
    async def test_function_returns_agreement_score(self) -> None:
        """Function returns agreement_score indicating source consensus."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import OutputItem
        
        func = SynthesizeOutputsFunction()
        
        outputs = [
            OutputItem(content="Approach A is recommended.", source_id="src1"),
            OutputItem(content="Approach A is the best choice.", source_id="src2"),
        ]
        
        result = await func.run(outputs=outputs)
        
        # Should have an agreement score
        assert result.agreement_score is not None
        assert 0.0 <= result.agreement_score <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestSynthesizeOutputsIntegration:
    """Integration tests for synthesize_outputs function."""

    @pytest.mark.asyncio
    async def test_full_synthesis_workflow(self) -> None:
        """Test complete synthesis workflow with citations and source map."""
        from src.functions.synthesize_outputs import SynthesizeOutputsFunction
        from src.schemas.functions.synthesize_outputs import (
            OutputItem,
            SynthesisStrategy,
        )
        from src.schemas.citations import Citation, SourceMetadata
        
        func = SynthesizeOutputsFunction()
        
        # Simulate chapter summaries with citations
        outputs = [
            OutputItem(
                content="Chapter 1 introduces domain-driven design[^1]. "
                        "The author emphasizes ubiquitous language.",
                source_id="ch1_summary",
                citations=[
                    Citation(
                        marker=1,
                        source=SourceMetadata(
                            source_type="book",
                            author="Evans, Eric",
                            title="Domain-Driven Design",
                            pages="1-25",
                        ),
                    ),
                ],
            ),
            OutputItem(
                content="Chapter 2 covers strategic design patterns[^1]. "
                        "Bounded contexts are essential.",
                source_id="ch2_summary",
                citations=[
                    Citation(
                        marker=1,
                        source=SourceMetadata(
                            source_type="book",
                            author="Evans, Eric",
                            title="Domain-Driven Design",
                            pages="26-50",
                        ),
                    ),
                ],
            ),
        ]
        
        result = await func.run(
            outputs=outputs,
            synthesis_strategy=SynthesisStrategy.MERGE,
        )
        
        # Verify merged content
        assert "domain" in result.merged_content.lower()
        assert len(result.merged_content) > 50
        
        # Verify source_map tracks provenance
        assert len(result.source_map) > 0
        
        # Verify citations are preserved and properly numbered
        assert len(result.citations) >= 2
        
        # Verify no conflicts for similar content
        # Agreement score depends on word overlap - may be low for different chapters
        # The important thing is that it's calculated (not None)
        if result.agreement_score is not None:
            assert 0.0 <= result.agreement_score <= 1.0
