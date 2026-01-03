"""Tests for PipelineOrchestrator.

TDD tests for WBS-AGT14: Pipeline Orchestrator.

Acceptance Criteria Coverage:
- AC-14.1: PipelineOrchestrator executes function DAGs
- AC-14.3: HandoffState flows between pipeline stages
- AC-14.4: Pipeline stages can be conditional

Exit Criteria:
- pytest tests/unit/pipelines/test_orchestrator.py passes with 100% coverage
- Stages execute in dependency order
- HandoffState propagates between stages

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
"""

import pytest
from typing import Any
from pydantic import ValidationError, BaseModel


# =============================================================================
# AC-14.1: PipelineOrchestrator Core Tests
# =============================================================================

class TestPipelineDefinition:
    """Tests for PipelineDefinition schema."""

    def test_pipeline_definition_requires_name(self) -> None:
        """PipelineDefinition requires name field."""
        from src.pipelines.orchestrator import PipelineDefinition, StageDefinition
        
        stage = StageDefinition(name="s1", function="f1")
        
        with pytest.raises(ValidationError) as exc_info:
            PipelineDefinition(stages=[stage])  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_pipeline_definition_requires_stages(self) -> None:
        """PipelineDefinition requires stages list."""
        from src.pipelines.orchestrator import PipelineDefinition
        
        with pytest.raises(ValidationError) as exc_info:
            PipelineDefinition(name="test")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("stages",) for e in errors)

    def test_pipeline_definition_accepts_valid_input(self) -> None:
        """PipelineDefinition accepts valid pipeline definition."""
        from src.pipelines.orchestrator import PipelineDefinition, StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="extract_structure",
            input_mapping={"content": "raw_input"},
        )
        
        pipeline = PipelineDefinition(
            name="test_pipeline",
            stages=[stage],
        )
        
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.stages) == 1

    def test_pipeline_definition_has_description(self) -> None:
        """PipelineDefinition has optional description."""
        from src.pipelines.orchestrator import PipelineDefinition, StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="extract_structure",
            input_mapping={"content": "raw_input"},
        )
        
        pipeline = PipelineDefinition(
            name="test_pipeline",
            stages=[stage],
            description="A test pipeline for unit testing",
        )
        
        assert "test pipeline" in pipeline.description.lower()

    def test_pipeline_definition_has_version(self) -> None:
        """PipelineDefinition has optional version."""
        from src.pipelines.orchestrator import PipelineDefinition, StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="extract_structure",
            input_mapping={"content": "raw_input"},
        )
        
        pipeline = PipelineDefinition(
            name="test_pipeline",
            stages=[stage],
            version="1.0.0",
        )
        
        assert pipeline.version == "1.0.0"


class TestStageDefinition:
    """Tests for StageDefinition schema."""

    def test_stage_definition_requires_name(self) -> None:
        """StageDefinition requires name field."""
        from src.pipelines.orchestrator import StageDefinition
        
        with pytest.raises(ValidationError) as exc_info:
            StageDefinition(
                function="extract_structure",
                input_mapping={},
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_stage_definition_requires_function(self) -> None:
        """StageDefinition requires function name."""
        from src.pipelines.orchestrator import StageDefinition
        
        with pytest.raises(ValidationError) as exc_info:
            StageDefinition(
                name="stage_1",
                input_mapping={},
            )  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("function",) for e in errors)

    def test_stage_definition_has_dependencies(self) -> None:
        """StageDefinition has optional dependencies list."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="stage_2",
            function="summarize_content",
            input_mapping={"content": "stage_1.output"},
            depends_on=["stage_1"],
        )
        
        assert stage.depends_on == ["stage_1"]

    def test_stage_definition_has_input_mapping(self) -> None:
        """StageDefinition has input_mapping for parameter resolution."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="extract_structure",
            input_mapping={
                "content": "pipeline.input.raw_text",
                "artifact_type": "markdown",
            },
        )
        
        assert "content" in stage.input_mapping
        assert "artifact_type" in stage.input_mapping

    def test_stage_definition_has_output_key(self) -> None:
        """StageDefinition has output_key for handoff cache."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="extract_structure",
            input_mapping={"content": "raw_input"},
            output_key="extracted_structure",
        )
        
        assert stage.output_key == "extracted_structure"

    def test_stage_definition_has_preset_override(self) -> None:
        """StageDefinition can override function preset."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="stage_1",
            function="summarize_content",
            input_mapping={"content": "raw_input"},
            preset="D10",  # High quality override
        )
        
        assert stage.preset == "D10"

    def test_stage_definition_has_retry_config(self) -> None:
        """StageDefinition has optional retry configuration."""
        from src.pipelines.orchestrator import StageDefinition, RetryConfig
        
        retry = RetryConfig(max_retries=3, backoff_factor=2.0)
        
        stage = StageDefinition(
            name="stage_1",
            function="generate_code",
            input_mapping={"specification": "spec"},
            retry_config=retry,
        )
        
        assert stage.retry_config is not None
        assert stage.retry_config.max_retries == 3


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""

    def test_orchestrator_initialization(self) -> None:
        """PipelineOrchestrator initializes correctly."""
        from src.pipelines.orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        
        assert isinstance(orchestrator, PipelineOrchestrator)
        assert hasattr(orchestrator, "function_registry")

    def test_orchestrator_has_function_registry(self) -> None:
        """PipelineOrchestrator has function registry."""
        from src.pipelines.orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        
        # Registry should be accessible and empty
        assert hasattr(orchestrator, "function_registry")
        assert len(orchestrator.function_registry) == 0

    def test_orchestrator_register_function(self) -> None:
        """PipelineOrchestrator can register agent functions."""
        from src.pipelines.orchestrator import PipelineOrchestrator
        from src.functions.base import AgentFunction
        
        class DummyOutput(BaseModel):
            result: str
        
        class DummyFunction(AgentFunction):
            name = "dummy_function"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> DummyOutput:
                return DummyOutput(result="test")
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(DummyFunction())
        
        assert "dummy_function" in orchestrator.function_registry

    @pytest.mark.asyncio
    async def test_orchestrator_executes_single_stage(self) -> None:
        """PipelineOrchestrator executes single-stage pipeline."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        from src.functions.base import AgentFunction
        
        class DummyOutput(BaseModel):
            result: str
        
        class DummyFunction(AgentFunction):
            name = "dummy_function"
            default_preset = "S1"
            
            async def run(self, *, input_text: str = "default", **kwargs: Any) -> DummyOutput:
                return DummyOutput(result=f"processed: {input_text}")
        
        stage = StageDefinition(
            name="stage_1",
            function="dummy_function",
            input_mapping={"input_text": "text"},
        )
        
        pipeline = PipelineDefinition(
            name="test_pipeline",
            stages=[stage],
        )
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(DummyFunction())
        
        result = await orchestrator.execute(pipeline, {"text": "hello"})
        
        assert result.success is True
        assert "stage_1" in result.outputs

    @pytest.mark.asyncio
    async def test_orchestrator_executes_multi_stage_pipeline(self) -> None:
        """PipelineOrchestrator executes multi-stage pipeline in order."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        from src.functions.base import AgentFunction
        
        execution_order: list[str] = []
        
        class StageOutput(BaseModel):
            data: str
        
        class FirstFunction(AgentFunction):
            name = "first_function"
            default_preset = "S1"
            
            async def run(self, *, input_data: str = "default", **kwargs: Any) -> StageOutput:
                execution_order.append("first")
                return StageOutput(data=f"first:{input_data}")
        
        class SecondFunction(AgentFunction):
            name = "second_function"
            default_preset = "S1"
            
            async def run(self, *, input_data: str = "default", **kwargs: Any) -> StageOutput:
                execution_order.append("second")
                return StageOutput(data=f"second:{input_data}")
        
        stages = [
            StageDefinition(
                name="stage_1",
                function="first_function",
                input_mapping={"input_data": "text"},
            ),
            StageDefinition(
                name="stage_2",
                function="second_function",
                input_mapping={"input_data": "stage_1.data"},
                depends_on=["stage_1"],
            ),
        ]
        
        pipeline = PipelineDefinition(
            name="test_pipeline",
            stages=stages,
        )
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(FirstFunction())
        orchestrator.register_function(SecondFunction())
        
        result = await orchestrator.execute(pipeline, {"text": "start"})
        
        assert result.success is True
        assert execution_order == ["first", "second"]

    def test_orchestrator_builds_dag_from_dependencies(self) -> None:
        """PipelineOrchestrator builds execution DAG from dependencies."""
        from src.pipelines.orchestrator import PipelineOrchestrator, PipelineDefinition, StageDefinition
        
        # Diamond dependency pattern:
        #     stage_1
        #    /       \
        # stage_2   stage_3
        #    \       /
        #     stage_4
        
        stages = [
            StageDefinition(
                name="stage_1",
                function="func",
                input_mapping={},
            ),
            StageDefinition(
                name="stage_2",
                function="func",
                input_mapping={},
                depends_on=["stage_1"],
            ),
            StageDefinition(
                name="stage_3",
                function="func",
                input_mapping={},
                depends_on=["stage_1"],
            ),
            StageDefinition(
                name="stage_4",
                function="func",
                input_mapping={},
                depends_on=["stage_2", "stage_3"],
            ),
        ]
        
        pipeline = PipelineDefinition(
            name="diamond_pipeline",
            stages=stages,
        )
        
        orchestrator = PipelineOrchestrator()
        execution_levels = orchestrator.build_dag(pipeline)
        
        # First level: stage_1
        assert "stage_1" in execution_levels[0]
        # Second level: stage_2 and stage_3 (parallel)
        assert set(execution_levels[1]) == {"stage_2", "stage_3"}
        # Third level: stage_4
        assert "stage_4" in execution_levels[2]


# =============================================================================
# AC-14.3: HandoffState Tests
# =============================================================================

class TestHandoffState:
    """Tests for HandoffState management."""

    def test_handoff_state_get_set(self) -> None:
        """HandoffState can get and set values."""
        from src.pipelines.orchestrator import HandoffState
        
        state = HandoffState()
        state.set("key1", "value1")
        
        assert state.get("key1") == "value1"
        assert state.get("nonexistent", "default") == "default"

    def test_handoff_state_dotted_notation(self) -> None:
        """HandoffState supports dotted notation for stage outputs."""
        from src.pipelines.orchestrator import HandoffState
        
        state = HandoffState()
        state.set_stage_output("extract_stage", {"keywords": ["python", "testing"]})
        
        # Should support dotted access
        value = state.get("extract_stage.keywords")
        assert value == ["python", "testing"]

    @pytest.mark.asyncio
    async def test_handoff_state_flows_between_stages(self) -> None:
        """HandoffState flows between pipeline stages (AC-14.3)."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        from src.functions.base import AgentFunction
        
        class ExtractOutput(BaseModel):
            keywords: list[str]
        
        class SummarizeOutput(BaseModel):
            summary: str
        
        class ExtractFunction(AgentFunction):
            name = "extract"
            default_preset = "S1"
            
            async def run(self, *, content: str = "", **kwargs: Any) -> ExtractOutput:
                return ExtractOutput(keywords=["python", "testing"])
        
        class SummarizeFunction(AgentFunction):
            name = "summarize"
            default_preset = "S1"
            
            async def run(self, *, keywords: list[str] = None, **kwargs: Any) -> SummarizeOutput:
                keywords = keywords or []
                return SummarizeOutput(summary=f"Keywords: {', '.join(keywords)}")
        
        stages = [
            StageDefinition(
                name="extract_stage",
                function="extract",
                input_mapping={"content": "text"},
            ),
            StageDefinition(
                name="summarize_stage",
                function="summarize",
                input_mapping={"keywords": "extract_stage.keywords"},
                depends_on=["extract_stage"],
            ),
        ]
        
        pipeline = PipelineDefinition(name="handoff_test", stages=stages)
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(ExtractFunction())
        orchestrator.register_function(SummarizeFunction())
        
        result = await orchestrator.execute(pipeline, {"text": "Python testing guide"})
        
        assert result.success is True
        # Verify handoff worked - summary contains keywords from extract
        final_output = result.outputs.get("summarize_stage")
        assert final_output is not None
        assert "python" in final_output["summary"].lower()


# =============================================================================
# AC-14.4: Conditional Stage Tests
# =============================================================================

class TestConditionalStages:
    """Tests for conditional pipeline stages."""

    def test_stage_definition_has_condition(self) -> None:
        """StageDefinition has optional condition."""
        from src.pipelines.orchestrator import StageDefinition, StageCondition
        
        condition = StageCondition(
            expression="check_stage.passed == True",
            skip_on_false=True,
        )
        
        stage = StageDefinition(
            name="optional_stage",
            function="analyze_artifact",
            input_mapping={"artifact": "generated_code"},
            condition=condition,
        )
        
        assert stage.condition is not None
        assert stage.condition.skip_on_false is True

    @pytest.mark.asyncio
    async def test_conditional_stage_skipped_when_false(self) -> None:
        """Conditional stage is skipped when condition is false (AC-14.4)."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition, StageCondition,
            StageStatus
        )
        from src.functions.base import AgentFunction
        
        class CheckOutput(BaseModel):
            passed: bool
        
        class ProcessOutput(BaseModel):
            result: str
        
        class CheckFunction(AgentFunction):
            name = "check"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> CheckOutput:
                return CheckOutput(passed=False)  # Will cause skip
        
        class ProcessFunction(AgentFunction):
            name = "process"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> ProcessOutput:
                return ProcessOutput(result="processed")
        
        stages = [
            StageDefinition(
                name="check_stage",
                function="check",
                input_mapping={},
            ),
            StageDefinition(
                name="process_stage",
                function="process",
                input_mapping={},
                depends_on=["check_stage"],
                condition=StageCondition(
                    expression="check_stage.passed == True",
                    skip_on_false=True,
                ),
            ),
        ]
        
        pipeline = PipelineDefinition(name="conditional_test", stages=stages)
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(CheckFunction())
        orchestrator.register_function(ProcessFunction())
        
        result = await orchestrator.execute(pipeline)
        
        assert result.success is True
        assert "check_stage" in result.stage_results
        assert result.stage_results["check_stage"].status == StageStatus.COMPLETED
        assert result.stage_results["process_stage"].status == StageStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_conditional_stage_executes_when_true(self) -> None:
        """Conditional stage executes when condition is true (AC-14.4)."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition, StageCondition,
            StageStatus
        )
        from src.functions.base import AgentFunction
        
        class CheckOutput(BaseModel):
            passed: bool
        
        class ProcessOutput(BaseModel):
            result: str
        
        class CheckFunction(AgentFunction):
            name = "check"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> CheckOutput:
                return CheckOutput(passed=True)  # Will allow execution
        
        class ProcessFunction(AgentFunction):
            name = "process"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> ProcessOutput:
                return ProcessOutput(result="processed")
        
        stages = [
            StageDefinition(
                name="check_stage",
                function="check",
                input_mapping={},
            ),
            StageDefinition(
                name="process_stage",
                function="process",
                input_mapping={},
                depends_on=["check_stage"],
                condition=StageCondition(
                    expression="check_stage.passed == True",
                    skip_on_false=True,
                ),
            ),
        ]
        
        pipeline = PipelineDefinition(name="conditional_test", stages=stages)
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(CheckFunction())
        orchestrator.register_function(ProcessFunction())
        
        result = await orchestrator.execute(pipeline)
        
        assert result.success is True
        assert result.stage_results["check_stage"].status == StageStatus.COMPLETED
        assert result.stage_results["process_stage"].status == StageStatus.COMPLETED


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_stage_failure_marks_pipeline_failed(self) -> None:
        """Stage failure marks pipeline as failed."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        from src.functions.base import AgentFunction
        
        class DummyOutput(BaseModel):
            data: str
        
        class FailingFunction(AgentFunction):
            name = "failing"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> DummyOutput:
                raise RuntimeError("Stage failed!")
        
        stage = StageDefinition(
            name="failing_stage",
            function="failing",
            input_mapping={},
        )
        
        pipeline = PipelineDefinition(name="error_test", stages=[stage])
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(FailingFunction())
        
        result = await orchestrator.execute(pipeline)
        
        assert result.success is False
        assert result.error is not None
        assert "Stage failed!" in str(result.error)

    @pytest.mark.asyncio
    async def test_missing_function_raises_error(self) -> None:
        """Missing function in registry causes pipeline failure."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        
        stage = StageDefinition(
            name="stage_1",
            function="nonexistent_function",
            input_mapping={},
        )
        
        pipeline = PipelineDefinition(name="missing_func_test", stages=[stage])
        orchestrator = PipelineOrchestrator()
        
        result = await orchestrator.execute(pipeline)
        
        assert result.success is False
        assert "nonexistent_function" in str(result.error)

    def test_cyclic_dependencies_detected(self) -> None:
        """Cyclic dependencies in stages are detected."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator, PipelineDefinition, StageDefinition
        )
        
        # Create cycle: stage_1 -> stage_2 -> stage_1
        stages = [
            StageDefinition(
                name="stage_1",
                function="func",
                input_mapping={},
                depends_on=["stage_2"],
            ),
            StageDefinition(
                name="stage_2",
                function="func",
                input_mapping={},
                depends_on=["stage_1"],
            ),
        ]
        
        pipeline = PipelineDefinition(name="cyclic_test", stages=stages)
        orchestrator = PipelineOrchestrator()
        
        with pytest.raises(ValueError) as exc_info:
            orchestrator.build_dag(pipeline)
        
        assert "cycl" in str(exc_info.value).lower()


# =============================================================================
# Pipeline Result Tests
# =============================================================================

class TestPipelineResult:
    """Tests for PipelineResult schema."""

    def test_pipeline_result_has_success_flag(self) -> None:
        """PipelineResult has success flag."""
        from src.pipelines.orchestrator import PipelineResult
        
        result = PipelineResult(
            success=True,
            pipeline_name="test_pipeline",
        )
        
        assert result.success is True

    def test_pipeline_result_has_outputs(self) -> None:
        """PipelineResult has stage outputs."""
        from src.pipelines.orchestrator import PipelineResult
        
        result = PipelineResult(
            success=True,
            pipeline_name="test_pipeline",
            outputs={"stage_1": {"data": "test"}},
        )
        
        assert "stage_1" in result.outputs

    def test_pipeline_result_has_timing(self) -> None:
        """PipelineResult has timing information."""
        from src.pipelines.orchestrator import PipelineResult
        
        result = PipelineResult(
            success=True,
            pipeline_name="test_pipeline",
            total_duration_ms=150.5,
        )
        
        assert result.total_duration_ms == pytest.approx(150.5)

    def test_pipeline_result_has_error_info(self) -> None:
        """PipelineResult has error information when failed."""
        from src.pipelines.orchestrator import PipelineResult
        
        result = PipelineResult(
            success=False,
            pipeline_name="test_pipeline",
            error="Stage 1 failed: timeout",
            failed_stage="stage_1",
        )
        
        assert result.error == "Stage 1 failed: timeout"
        assert result.failed_stage == "stage_1"


__all__ = [
    "TestPipelineDefinition",
    "TestStageDefinition",
    "TestPipelineOrchestrator",
    "TestHandoffState",
    "TestConditionalStages",
    "TestPipelineErrorHandling",
    "TestPipelineResult",
]
