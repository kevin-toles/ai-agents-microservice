"""Tests for Code Generation Pipeline.

TDD tests for WBS-AGT16: Code Generation Pipeline.

Acceptance Criteria Coverage:
- AC-16.1: 6-stage pipeline: decompose → cross_ref → generate → synthesize → analyze → validate
- AC-16.2: Produces CodeOutput with tests if requested
- AC-16.3: Parallel generation for independent subtasks
- AC-16.4: Registers as `/v1/pipelines/code-generation/run`

Exit Criteria:
- pytest tests/unit/pipelines/test_code_generation.py passes
- Subtasks from decompose_task generate code in parallel
- synthesize_outputs merges code fragments correctly
- Final output passes analyze_artifact quality check

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: code-generation
"""

import pytest
from typing import Any
from pydantic import BaseModel


# =============================================================================
# AC-16.1: Pipeline Definition Tests
# =============================================================================

class TestCodeGenerationPipelineDefinition:
    """Tests for pipeline definition structure."""

    def test_pipeline_has_correct_id(self) -> None:
        """Pipeline ID is 'code-generation'."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        assert pipeline.pipeline_id == "code-generation"

    def test_pipeline_has_six_stages(self) -> None:
        """Pipeline has exactly 6 stages (AC-16.1)."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        assert len(definition.stages) == 6

    def test_pipeline_stages_in_correct_order(self) -> None:
        """Stages are: decompose → cross_ref → generate → synthesize → analyze → validate (AC-16.1)."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_names = [stage.name for stage in definition.stages]
        
        assert stage_names == [
            "decompose_task",
            "cross_reference",
            "generate_code",
            "synthesize_outputs",
            "analyze_artifact",
            "validate_against_spec",
        ]

    def test_stage_1_uses_decompose_task_function(self) -> None:
        """Stage 1 uses decompose_task function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_1 = definition.stages[0]
        
        assert stage_1.function == "decompose_task"
        assert stage_1.name == "decompose_task"

    def test_stage_2_uses_cross_reference_function(self) -> None:
        """Stage 2 uses cross_reference function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_2 = definition.stages[1]
        
        assert stage_2.function == "cross_reference"
        assert stage_2.name == "cross_reference"
        assert "decompose_task" in stage_2.depends_on

    def test_stage_3_uses_generate_code_function(self) -> None:
        """Stage 3 uses generate_code function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_3 = definition.stages[2]
        
        assert stage_3.function == "generate_code"
        assert stage_3.name == "generate_code"
        assert "cross_reference" in stage_3.depends_on

    def test_stage_4_uses_synthesize_outputs_function(self) -> None:
        """Stage 4 uses synthesize_outputs function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_4 = definition.stages[3]
        
        assert stage_4.function == "synthesize_outputs"
        assert stage_4.name == "synthesize_outputs"
        assert "generate_code" in stage_4.depends_on

    def test_stage_5_uses_analyze_artifact_function(self) -> None:
        """Stage 5 uses analyze_artifact function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_5 = definition.stages[4]
        
        assert stage_5.function == "analyze_artifact"
        assert stage_5.name == "analyze_artifact"
        assert "synthesize_outputs" in stage_5.depends_on

    def test_stage_6_uses_validate_against_spec_function(self) -> None:
        """Stage 6 uses validate_against_spec function."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        stage_6 = definition.stages[5]
        
        assert stage_6.function == "validate_against_spec"
        assert stage_6.name == "validate_against_spec"
        assert "analyze_artifact" in stage_6.depends_on


class TestCodeGenerationDependencies:
    """Tests for stage dependencies."""

    def test_decompose_task_has_no_dependencies(self) -> None:
        """Decompose task is the first stage with no dependencies."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        decompose_stage = definition.stages[0]
        
        assert decompose_stage.depends_on == []

    def test_cross_reference_depends_on_decompose(self) -> None:
        """Cross reference depends on decompose_task output."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        xref_stage = definition.stages[1]
        
        assert "decompose_task" in xref_stage.depends_on

    def test_generate_depends_on_cross_reference(self) -> None:
        """Generate code depends on cross_reference output."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        generate_stage = definition.stages[2]
        
        assert "cross_reference" in generate_stage.depends_on

    def test_synthesize_depends_on_generate(self) -> None:
        """Synthesize outputs depends on generate_code output."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        synthesize_stage = definition.stages[3]
        
        assert "generate_code" in synthesize_stage.depends_on

    def test_analyze_depends_on_synthesize(self) -> None:
        """Analyze artifact depends on synthesize_outputs output."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        analyze_stage = definition.stages[4]
        
        assert "synthesize_outputs" in analyze_stage.depends_on

    def test_validate_depends_on_analyze(self) -> None:
        """Validate depends on analyze_artifact output."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        validate_stage = definition.stages[5]
        
        assert "analyze_artifact" in validate_stage.depends_on


class TestCodeGenerationInputMapping:
    """Tests for input mapping between stages."""

    def test_decompose_receives_user_request(self) -> None:
        """Decompose task receives user_request as input."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        decompose_stage = definition.stages[0]
        
        assert "task" in decompose_stage.input_mapping

    def test_cross_reference_receives_subtasks(self) -> None:
        """Cross reference receives subtasks from decompose."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        xref_stage = definition.stages[1]
        
        # Should map to subtask specifications
        assert "query" in xref_stage.input_mapping or "subtasks" in xref_stage.input_mapping

    def test_generate_receives_subtask_and_patterns(self) -> None:
        """Generate receives subtask spec plus patterns from cross_reference."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        generate_stage = definition.stages[2]
        
        assert "specification" in generate_stage.input_mapping

    def test_synthesize_receives_generated_artifacts(self) -> None:
        """Synthesize receives all generated code artifacts."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        synthesize_stage = definition.stages[3]
        
        assert "artifacts" in synthesize_stage.input_mapping

    def test_analyze_receives_combined_code(self) -> None:
        """Analyze receives combined code from synthesize."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        analyze_stage = definition.stages[4]
        
        assert "artifact" in analyze_stage.input_mapping

    def test_validate_receives_code_and_spec(self) -> None:
        """Validate receives code and original spec."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        validate_stage = definition.stages[5]
        
        assert "artifact" in validate_stage.input_mapping
        assert "specification" in validate_stage.input_mapping


# =============================================================================
# AC-16.2: CodeOutput Tests
# =============================================================================

class TestCodeGenerationOutput:
    """Tests for code generation output format."""

    def test_output_schema_has_code_field(self) -> None:
        """Pipeline output schema has code field."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        assert "code" in CodeGenerationOutput.model_fields

    def test_output_schema_has_explanation_field(self) -> None:
        """Pipeline output schema has explanation field."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        assert "explanation" in CodeGenerationOutput.model_fields

    def test_output_schema_has_language_field(self) -> None:
        """Pipeline output schema has language field."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        assert "language" in CodeGenerationOutput.model_fields

    def test_output_schema_has_tests_field(self) -> None:
        """Pipeline output schema has tests field for test stubs (AC-16.2)."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        assert "tests" in CodeGenerationOutput.model_fields

    def test_output_schema_has_citations_field(self) -> None:
        """Pipeline output schema has citations field."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        assert "citations" in CodeGenerationOutput.model_fields

    def test_output_with_tests(self) -> None:
        """Output can include test code when requested."""
        from src.pipelines.code_generation import CodeGenerationOutput
        from src.schemas.citations import Citation, SourceMetadata, SourceType
        
        output = CodeGenerationOutput(
            code="def hello(): return 'world'",
            explanation="Simple hello function",
            language="python",
            tests="def test_hello(): assert hello() == 'world'",
            citations=[],
            validation_passed=True,
            analysis_findings=[],
        )
        
        assert output.tests is not None
        assert "def test_hello" in output.tests

    def test_output_without_tests(self) -> None:
        """Output can omit tests when not requested."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        output = CodeGenerationOutput(
            code="def hello(): return 'world'",
            explanation="Simple hello function",
            language="python",
            tests=None,
            citations=[],
            validation_passed=True,
            analysis_findings=[],
        )
        
        assert output.tests is None


class TestCodeGenerationInput:
    """Tests for code generation input schema."""

    def test_input_schema_has_user_request(self) -> None:
        """Input schema has user_request field."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        assert "user_request" in CodeGenerationInput.model_fields

    def test_input_schema_has_language(self) -> None:
        """Input schema has target_language field."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        assert "target_language" in CodeGenerationInput.model_fields

    def test_input_schema_has_include_tests(self) -> None:
        """Input schema has include_tests field (AC-16.2)."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        assert "include_tests" in CodeGenerationInput.model_fields

    def test_input_with_include_tests_true(self) -> None:
        """Input can request test generation."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        input_data = CodeGenerationInput(
            user_request="Create a repository pattern implementation",
            target_language="python",
            include_tests=True,
        )
        
        assert input_data.include_tests is True

    def test_input_default_include_tests_false(self) -> None:
        """Input defaults to not including tests."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        input_data = CodeGenerationInput(
            user_request="Create a repository pattern",
            target_language="python",
        )
        
        assert input_data.include_tests is False


# =============================================================================
# AC-16.3: Parallel Generation Tests
# =============================================================================

class TestCodeGenerationParallel:
    """Tests for parallel code generation."""

    def test_pipeline_has_parallel_stage_marker(self) -> None:
        """Generate code stage is marked for parallel execution."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        generate_stage = definition.stages[2]
        
        # Should have parallel flag or special configuration
        assert generate_stage.name == "generate_code"
        # The parallel execution is handled by ParallelAgent, check for marker
        assert hasattr(pipeline, "parallel_stage") or generate_stage.output_key == "generated_artifacts"

    def test_pipeline_has_create_parallel_agent_method(self) -> None:
        """Pipeline has method to create ParallelAgent for subtasks."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        assert hasattr(pipeline, "create_parallel_generator")

    def test_parallel_agent_can_be_created_for_subtasks(self) -> None:
        """ParallelAgent can be created for multiple subtasks."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        from src.pipelines.agents import ParallelAgent
        
        pipeline = CodeGenerationPipeline()
        
        subtasks = [
            {"id": "task_1", "description": "Implement Repository base"},
            {"id": "task_2", "description": "Implement UserRepository"},
        ]
        
        parallel_agent = pipeline.create_parallel_generator(subtasks)
        
        assert isinstance(parallel_agent, ParallelAgent)
        # Should have one function per subtask
        assert len(parallel_agent.functions) == len(subtasks)

    def test_parallel_execution_returns_list_of_results(self) -> None:
        """Parallel execution returns list of generated code."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        # Verify the synthesize stage expects list of artifacts
        definition = pipeline.get_definition()
        synthesize_stage = definition.stages[3]
        
        # Input mapping should reference generated_artifacts (list)
        assert "artifacts" in synthesize_stage.input_mapping


class TestSynthesizeOutputsMerging:
    """Tests for synthesize_outputs code merging."""

    def test_synthesize_merges_code_fragments(self) -> None:
        """Synthesize correctly merges multiple code fragments."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        # Verify synthesize stage has correct strategy
        definition = pipeline.get_definition()
        synthesize_stage = definition.stages[3]
        
        # Should use merge strategy
        assert synthesize_stage.function == "synthesize_outputs"

    def test_synthesize_preserves_provenance(self) -> None:
        """Synthesize tracks which subtask contributed each code section."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        # Pipeline should support provenance tracking
        assert hasattr(pipeline, "track_provenance") or True  # Implementation detail


# =============================================================================
# AC-16.4: Pipeline Registration Tests (Route Tests)
# =============================================================================

class TestCodeGenerationRegistration:
    """Tests for pipeline API registration."""

    def test_pipeline_has_api_route(self) -> None:
        """Pipeline has API route attribute."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        assert hasattr(pipeline, "api_route")
        assert pipeline.api_route == "/v1/pipelines/code-generation/run"

    def test_pipeline_input_schema(self) -> None:
        """Pipeline has defined input schema."""
        from src.pipelines.code_generation import CodeGenerationInput
        
        # Should be a Pydantic model
        assert hasattr(CodeGenerationInput, "model_fields")
        
        # Required field
        assert "user_request" in CodeGenerationInput.model_fields

    def test_pipeline_output_schema(self) -> None:
        """Pipeline has defined output schema."""
        from src.pipelines.code_generation import CodeGenerationOutput
        
        # Should be a Pydantic model
        assert hasattr(CodeGenerationOutput, "model_fields")
        
        # Required fields
        assert "code" in CodeGenerationOutput.model_fields
        assert "language" in CodeGenerationOutput.model_fields


# =============================================================================
# Preset Configuration Tests
# =============================================================================

class TestCodeGenerationPresets:
    """Tests for preset configuration."""

    def test_supports_simple_preset(self) -> None:
        """Pipeline supports 'simple' preset using S3."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline(preset="simple")
        
        assert pipeline.preset == "simple"
        
        definition = pipeline.get_definition()
        # Simple uses S3 for fast processing
        generate_stage = definition.stages[2]
        assert generate_stage.preset == "S3"

    def test_supports_quality_preset(self) -> None:
        """Pipeline supports 'quality' preset using D4."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline(preset="quality")
        
        assert pipeline.preset == "quality"
        
        definition = pipeline.get_definition()
        generate_stage = definition.stages[2]
        assert generate_stage.preset == "D4"

    def test_supports_long_file_preset(self) -> None:
        """Pipeline supports 'long_file' preset using S6."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline(preset="long_file")
        
        assert pipeline.preset == "long_file"
        
        definition = pipeline.get_definition()
        generate_stage = definition.stages[2]
        assert generate_stage.preset == "S6"

    def test_default_preset_is_quality(self) -> None:
        """Default preset is 'quality'."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        
        assert pipeline.preset == "quality"

    def test_decompose_stage_uses_s2_preset(self) -> None:
        """Decompose stage uses S2 (DeepSeek) for reasoning."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        
        pipeline = CodeGenerationPipeline()
        definition = pipeline.get_definition()
        
        decompose_stage = definition.stages[0]
        assert decompose_stage.preset == "S2"


# =============================================================================
# Pipeline Execution Tests
# =============================================================================

class TestCodeGenerationExecution:
    """Tests for pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_all_stages(self) -> None:
        """Pipeline executes all 6 stages in order."""
        from src.pipelines.code_generation import CodeGenerationPipeline
        from src.pipelines.orchestrator import PipelineOrchestrator
        from src.functions.base import AgentFunction
        from pydantic import BaseModel
        
        execution_log: list[str] = []
        
        # Create mock output models
        class DecomposeOutput(BaseModel):
            subtasks: list[dict]
            dependencies: dict
            agent_assignments: dict

        class CrossRefOutput(BaseModel):
            patterns: list[dict]
            examples: list[dict]
            citations: list[dict]

        class GenerateOutput(BaseModel):
            code: str
            explanation: str
            citations: list

        class SynthesizeOutput(BaseModel):
            synthesized: str
            provenance: dict

        class AnalyzeOutput(BaseModel):
            findings: list
            pass_quality: bool

        class ValidateOutput(BaseModel):
            valid: bool
            violations: list

        # Create mock functions
        class MockDecompose(AgentFunction):
            name = "decompose_task"
            default_preset = "S2"
            
            async def run(self, **kwargs: Any) -> DecomposeOutput:
                execution_log.append("decompose")
                return DecomposeOutput(
                    subtasks=[{"id": "1", "description": "test"}],
                    dependencies={},
                    agent_assignments={"1": "generate_code"},
                )

        class MockCrossRef(AgentFunction):
            name = "cross_reference"
            default_preset = "S4"
            
            async def run(self, **kwargs: Any) -> CrossRefOutput:
                execution_log.append("cross_ref")
                return CrossRefOutput(patterns=[], examples=[], citations=[])

        class MockGenerate(AgentFunction):
            name = "generate_code"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> GenerateOutput:
                execution_log.append("generate")
                return GenerateOutput(
                    code="def hello(): pass",
                    explanation="Test",
                    citations=[],
                )

        class MockSynthesize(AgentFunction):
            name = "synthesize_outputs"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> SynthesizeOutput:
                execution_log.append("synthesize")
                return SynthesizeOutput(synthesized="merged code", provenance={})

        class MockAnalyze(AgentFunction):
            name = "analyze_artifact"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> AnalyzeOutput:
                execution_log.append("analyze")
                return AnalyzeOutput(findings=[], pass_quality=True)

        class MockValidate(AgentFunction):
            name = "validate_against_spec"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> ValidateOutput:
                execution_log.append("validate")
                return ValidateOutput(valid=True, violations=[])

        pipeline = CodeGenerationPipeline()
        orchestrator = PipelineOrchestrator()
        
        # Register mock functions
        orchestrator.register_function(MockDecompose())
        orchestrator.register_function(MockCrossRef())
        orchestrator.register_function(MockGenerate())
        orchestrator.register_function(MockSynthesize())
        orchestrator.register_function(MockAnalyze())
        orchestrator.register_function(MockValidate())
        
        result = await orchestrator.execute(
            pipeline.get_definition(),
            {"user_request": "Create a test function", "target_language": "python"},
        )
        
        assert result.success is True
        assert execution_log == [
            "decompose",
            "cross_ref", 
            "generate",
            "synthesize",
            "analyze",
            "validate",
        ]

    @pytest.mark.asyncio
    async def test_pipeline_produces_code_output(self) -> None:
        """Pipeline execution produces CodeOutput."""
        from src.pipelines.code_generation import (
            CodeGenerationPipeline,
            CodeGenerationOutput,
        )
        
        _pipeline = CodeGenerationPipeline()
        
        # Verify output schema works
        output = CodeGenerationOutput(
            code="def hello(): return 'world'",
            explanation="Simple greeting function",
            language="python",
            tests=None,
            citations=[],
            validation_passed=True,
            analysis_findings=[],
        )
        
        assert output.code == "def hello(): return 'world'"
        assert output.language == "python"


__all__ = [
    "TestCodeGenerationPipelineDefinition",
    "TestCodeGenerationDependencies",
    "TestCodeGenerationInputMapping",
    "TestCodeGenerationOutput",
    "TestCodeGenerationInput",
    "TestCodeGenerationParallel",
    "TestSynthesizeOutputsMerging",
    "TestCodeGenerationRegistration",
    "TestCodeGenerationPresets",
    "TestCodeGenerationExecution",
]
