"""Code Generation Pipeline.

WBS-AGT16: Code Generation Pipeline

Implements a 6-stage pipeline for generating code from natural language:
1. decompose_task - Break request into subtasks with dependencies
2. cross_reference - Find patterns and examples for each subtask
3. generate_code - Generate code for each subtask (parallel)
4. synthesize_outputs - Merge code fragments into coherent result
5. analyze_artifact - Quality and security analysis
6. validate_against_spec - Verify output matches original request

Acceptance Criteria:
- AC-16.1: 6-stage pipeline: decompose → cross_ref → generate → synthesize → analyze → validate
- AC-16.2: Produces CodeOutput with tests if requested
- AC-16.3: Parallel generation for independent subtasks
- AC-16.4: Registers as `/v1/pipelines/code-generation/run`

Preset Configuration:
- simple: Uses S3 presets (qwen2.5-7b solo)
- quality: Uses D4 presets (critique mode) - DEFAULT
- long_file: Uses S6 presets (granite-8b-code-128k)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline DAG: code-generation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.pipelines.orchestrator import PipelineDefinition, StageDefinition


if TYPE_CHECKING:
    from src.functions.base import AgentFunction
    from src.functions.base import AgentFunction as BaseAgentFunction
    from src.pipelines.agents import ParallelAgent
    from src.schemas.citations import Citation


# =============================================================================
# Type Aliases
# =============================================================================

PresetType = Literal["simple", "quality", "long_file"]


# =============================================================================
# Preset Configuration
# =============================================================================

# Preset to stage preset mapping
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Preset Configuration
PRESET_MAPPING: dict[PresetType, dict[str, str]] = {
    "simple": {
        "decompose_task": "S2",           # DeepSeek for reasoning
        "cross_reference": "S4",          # Semantic search
        "generate_code": "S3",            # qwen2.5-7b solo (fast)
        "synthesize_outputs": "S1",       # phi-4 for merging
        "analyze_artifact": "S3",         # Quick analysis
        "validate_against_spec": "S3",    # Fast validation
    },
    "quality": {
        "decompose_task": "S2",           # DeepSeek for reasoning
        "cross_reference": "S4",          # Semantic search
        "generate_code": "D4",            # qwen + deepseek critique
        "synthesize_outputs": "D4",       # Code-aware synthesis
        "analyze_artifact": "D4",         # Think + Code critique
        "validate_against_spec": "D4",    # Quality validation
    },
    "long_file": {
        "decompose_task": "S2",           # DeepSeek for reasoning
        "cross_reference": "S4",          # Semantic search
        "generate_code": "S6",            # granite-8b-code-128k
        "synthesize_outputs": "S6",       # Long context merge
        "analyze_artifact": "D4",         # Quality analysis
        "validate_against_spec": "D4",    # Quality validation
    },
}


# =============================================================================
# Input/Output Schemas
# =============================================================================

class CodeGenerationInput(BaseModel):
    """Input schema for code generation pipeline.

    Attributes:
        user_request: Natural language description of the code to generate
        target_language: Programming language for the generated code
        include_tests: Whether to generate test stubs (AC-16.2)
        constraints: Optional constraints for the generated code
        context: Optional project/domain context
        preset: Quality preset for generation
    """

    user_request: str = Field(
        ...,
        description="Natural language description of the code to generate",
        min_length=1,
    )
    target_language: str = Field(
        default="python",
        description="Target programming language",
    )
    include_tests: bool = Field(
        default=False,
        description="Whether to generate test stubs (AC-16.2)",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Optional constraints for code generation",
    )
    context: str | None = Field(
        default=None,
        description="Optional project/domain context",
    )
    preset: PresetType = Field(
        default="quality",
        description="Quality preset: simple, quality, or long_file",
    )


class CodeGenerationOutput(BaseModel):
    """Output schema for code generation pipeline.

    Produces CodeOutput with generated code, explanation, and optional tests.

    Attributes:
        code: Generated code
        explanation: Explanation of design decisions and choices
        language: Programming language of the generated code
        tests: Optional test stubs if include_tests was True
        citations: Sources used during generation
        validation_passed: Whether validation against spec passed
        analysis_findings: Findings from code analysis stage
    """

    code: str = Field(
        ...,
        description="Generated code",
    )
    explanation: str = Field(
        ...,
        description="Explanation of design decisions",
    )
    language: str = Field(
        ...,
        description="Programming language of the generated code",
    )
    tests: str | None = Field(
        default=None,
        description="Generated test stubs (if include_tests=True)",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources used during generation",
    )
    validation_passed: bool = Field(
        ...,
        description="Whether code passed validation against original request",
    )
    analysis_findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Findings from code analysis (quality, security issues)",
    )


# =============================================================================
# Subtask Schema
# =============================================================================

class Subtask(BaseModel):
    """A subtask decomposed from the main task.

    Attributes:
        id: Unique identifier for the subtask
        description: What this subtask should accomplish
        depends_on: IDs of subtasks this depends on
        estimated_tokens: Token budget estimate for this subtask
    """

    id: str = Field(
        ...,
        description="Unique identifier for the subtask",
    )
    description: str = Field(
        ...,
        description="What this subtask should accomplish",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of subtasks this depends on",
    )
    estimated_tokens: int = Field(
        default=1000,
        description="Token budget estimate",
        ge=0,
    )


# =============================================================================
# Pipeline Definition
# =============================================================================

class CodeGenerationPipeline:
    """Code Generation Pipeline.

    6-stage pipeline for generating code from natural language:

    Stage 1: decompose_task
        - Input: user_request, constraints, context
        - Output: {subtasks, dependencies, agent_assignments}
        - Preset: S2 (DeepSeek for chain-of-thought reasoning)

    Stage 2: cross_reference
        - Input: subtask specifications
        - Output: {patterns, examples, citations}
        - Preset: S4 (semantic search)
        - Scope: code-reference-engine/, ai-platform-data/

    Stage 3: generate_code (PARALLEL - AC-16.3)
        - Input: subtask + patterns + context
        - Output: {code, explanation, citations}
        - Preset: S3 (simple) / D4 (quality) / S6 (long_file)
        - Executes in parallel for independent subtasks

    Stage 4: synthesize_outputs
        - Input: all generated code artifacts
        - Output: {combined_code, provenance}
        - Preset: S1 (simple) / D4 (quality)

    Stage 5: analyze_artifact
        - Input: combined code
        - Output: {findings, pass/fail}
        - Preset: S3 (simple) / D4 (quality)
        - Type: quality + security analysis

    Stage 6: validate_against_spec
        - Input: code + original_request
        - Output: {valid, violations}
        - Can retry Stage 3 if validation fails

    Attributes:
        pipeline_id: Unique identifier for this pipeline
        preset: Quality preset (simple, quality, long_file)
        api_route: API endpoint for this pipeline
    """

    pipeline_id: str = "code-generation"
    api_route: str = "/v1/pipelines/code-generation/run"
    parallel_stage: str = "generate_code"

    def __init__(self, preset: PresetType = "quality") -> None:
        """Initialize the pipeline with the specified preset.

        Args:
            preset: Quality preset for code generation.
                - "simple": Fast processing with S3 presets
                - "quality": Balanced with D4 critique mode (default)
                - "long_file": Long context with S6 presets
        """
        self.preset: PresetType = preset
        self._subtasks: list[Subtask] = []

    def get_definition(self) -> PipelineDefinition:
        """Get the pipeline definition with stages configured for the preset.

        Returns:
            PipelineDefinition with 6 configured stages
        """
        stage_presets = PRESET_MAPPING[self.preset]

        return PipelineDefinition(
            name=self.pipeline_id,
            description="Generate code from natural language specification",
            stages=[
                # Stage 1: Decompose Task
                StageDefinition(
                    name="decompose_task",
                    function="decompose_task",
                    input_mapping={
                        "task": "input.user_request",
                        "constraints": "input.constraints",
                        "context": "input.context",
                    },
                    depends_on=[],
                    output_key="decompose_output",
                    preset=stage_presets["decompose_task"],
                ),
                # Stage 2: Cross Reference
                StageDefinition(
                    name="cross_reference",
                    function="cross_reference",
                    input_mapping={
                        "query": "decompose_output.subtasks",
                        "subtasks": "decompose_output.subtasks",
                    },
                    depends_on=["decompose_task"],
                    output_key="xref_output",
                    preset=stage_presets["cross_reference"],
                ),
                # Stage 3: Generate Code (PARALLEL)
                StageDefinition(
                    name="generate_code",
                    function="generate_code",
                    input_mapping={
                        "specification": "decompose_output.subtasks",
                        "patterns": "xref_output.patterns",
                        "context_artifacts": "xref_output.examples",
                        "language": "input.target_language",
                    },
                    depends_on=["cross_reference"],
                    output_key="generated_artifacts",
                    preset=stage_presets["generate_code"],
                ),
                # Stage 4: Synthesize Outputs
                StageDefinition(
                    name="synthesize_outputs",
                    function="synthesize_outputs",
                    input_mapping={
                        "artifacts": "generated_artifacts",
                        "synthesis_strategy": "'merge'",
                    },
                    depends_on=["generate_code"],
                    output_key="synthesized_output",
                    preset=stage_presets["synthesize_outputs"],
                ),
                # Stage 5: Analyze Artifact
                StageDefinition(
                    name="analyze_artifact",
                    function="analyze_artifact",
                    input_mapping={
                        "artifact": "synthesized_output.synthesized",
                        "artifact_type": "'code'",
                        "analysis_type": "'quality'",
                    },
                    depends_on=["synthesize_outputs"],
                    output_key="analysis_output",
                    preset=stage_presets["analyze_artifact"],
                ),
                # Stage 6: Validate Against Spec
                StageDefinition(
                    name="validate_against_spec",
                    function="validate_against_spec",
                    input_mapping={
                        "artifact": "synthesized_output.synthesized",
                        "specification": "input.user_request",
                    },
                    depends_on=["analyze_artifact"],
                    output_key="validation_output",
                    preset=stage_presets["validate_against_spec"],
                ),
            ],
        )

    def create_parallel_generator(
        self,
        subtasks: list[dict[str, Any]],
        generate_function: AgentFunction | None = None,
    ) -> ParallelAgent:
        """Create a ParallelAgent for generating code for multiple subtasks.

        This enables AC-16.3: Parallel generation for independent subtasks.

        Args:
            subtasks: List of subtask dictionaries with id and description
            generate_function: Optional generate_code function instance

        Returns:
            ParallelAgent configured to generate code in parallel
        """
        from src.pipelines.agents import ParallelAgent

        # Create wrapper functions for each subtask
        functions: list[BaseAgentFunction] = []

        if generate_function is not None:
            # Use the provided function for each subtask
            functions = [generate_function for _ in subtasks]
        else:
            # Create placeholder functions for subtask execution
            # In production, these would be SubtaskGeneratorFunction instances
            from src.pipelines.code_generation import _create_subtask_generator
            functions = [_create_subtask_generator(subtask) for subtask in subtasks]

        return ParallelAgent(
            name="parallel_code_generation",
            functions=functions,
            return_exceptions=False,
        )

    def track_provenance(
        self,
        subtask_id: str,
        code_fragment: str,
        citations: list[Citation],
    ) -> dict[str, Any]:
        """Track which subtask contributed each code section.

        Args:
            subtask_id: ID of the subtask that generated this code
            code_fragment: The generated code fragment
            citations: Citations used in generation

        Returns:
            Provenance record for tracking
        """
        return {
            "subtask_id": subtask_id,
            "code_length": len(code_fragment),
            "citation_count": len(citations),
            "citations": [c.marker for c in citations],
        }


# =============================================================================
# Subtask Generator Factory
# =============================================================================

def _create_subtask_generator(subtask: dict[str, Any]) -> Any:
    """Create a generator function wrapper for a subtask.

    This factory creates AgentFunction-compatible objects that can be
    executed in parallel by ParallelAgent.

    Args:
        subtask: Subtask dictionary with id and description

    Returns:
        A callable function wrapper for the subtask
    """
    from pydantic import BaseModel

    from src.functions.base import AgentFunction

    class SubtaskGeneratorOutput(BaseModel):
        """Output from subtask generation."""
        code: str
        explanation: str
        subtask_id: str

    class SubtaskGeneratorFunction(AgentFunction):
        """Wrapper function for generating code for a specific subtask."""

        name = "subtask_generator"
        default_preset = "D4"

        def __init__(self, subtask_data: dict[str, Any]) -> None:
            """Initialize with subtask data."""
            self._subtask = subtask_data

        async def run(self, **kwargs: Any) -> SubtaskGeneratorOutput:
            """Generate code for this subtask.

            In production, this would call the actual generate_code function
            with the subtask specification.
            """
            return SubtaskGeneratorOutput(
                code=f"# Code for: {self._subtask.get('description', 'unknown')}",
                explanation=f"Generated for subtask {self._subtask.get('id', 'unknown')}",
                subtask_id=self._subtask.get("id", "unknown"),
            )

    return SubtaskGeneratorFunction(subtask)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PRESET_MAPPING",
    "CodeGenerationInput",
    "CodeGenerationOutput",
    "CodeGenerationPipeline",
    "PresetType",
    "Subtask",
]
