"""Tests for Pipeline Saga Pattern.

TDD tests for WBS-AGT14: Pipeline Orchestrator - Saga Compensation.

Acceptance Criteria Coverage:
- AC-14.5: Saga compensation on stage failure

Exit Criteria:
- PipelineSaga tracks completed stages
- Saga triggers compensation on failure
- Compensation runs in reverse order
- Partial results are preserved

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Error Handling / Saga
"""

import pytest
import asyncio
from typing import Any
from pydantic import BaseModel


# =============================================================================
# AC-14.5: PipelineSaga Tests
# =============================================================================

class TestPipelineSaga:
    """Tests for PipelineSaga pattern."""

    def test_saga_creation(self) -> None:
        """PipelineSaga can be created."""
        from src.pipelines.saga import PipelineSaga
        
        saga = PipelineSaga(name="test_saga")
        
        assert saga.name == "test_saga"
        assert saga.completed_stages == []
        assert saga.compensation_handlers == {}

    def test_saga_register_compensation(self) -> None:
        """PipelineSaga can register compensation handlers."""
        from src.pipelines.saga import PipelineSaga
        
        saga = PipelineSaga(name="test_saga")
        
        async def rollback_stage_a(context: dict) -> None:
            pass
        
        saga.register_compensation("stage_a", rollback_stage_a)
        
        assert "stage_a" in saga.compensation_handlers
        assert saga.compensation_handlers["stage_a"] == rollback_stage_a

    def test_saga_record_completion(self) -> None:
        """PipelineSaga records stage completion."""
        from src.pipelines.saga import PipelineSaga
        
        saga = PipelineSaga(name="test_saga")
        
        saga.record_completion("stage_1", {"result": "success"})
        saga.record_completion("stage_2", {"result": "also_success"})
        
        assert len(saga.completed_stages) == 2
        assert saga.completed_stages[0].stage == "stage_1"
        assert saga.completed_stages[1].stage == "stage_2"


# =============================================================================
# Saga Compensation Execution Tests
# =============================================================================

class TestSagaCompensation:
    """Tests for saga compensation execution."""

    @pytest.mark.asyncio
    async def test_saga_compensates_on_failure(self) -> None:
        """Saga triggers compensation when failure occurs (AC-14.5)."""
        from src.pipelines.saga import PipelineSaga
        
        compensation_log: list[str] = []
        
        saga = PipelineSaga(name="compensation_test")
        
        async def compensate_a(context: dict) -> None:
            compensation_log.append("rollback_a")
        
        async def compensate_b(context: dict) -> None:
            compensation_log.append("rollback_b")
        
        saga.register_compensation("stage_a", compensate_a)
        saga.register_compensation("stage_b", compensate_b)
        
        # Simulate stages completing
        saga.record_completion("stage_a", {"data": "a"})
        saga.record_completion("stage_b", {"data": "b"})
        
        # Trigger compensation
        await saga.compensate()
        
        # Compensation should have been called for both stages
        assert "rollback_a" in compensation_log
        assert "rollback_b" in compensation_log

    @pytest.mark.asyncio
    async def test_saga_compensation_reverse_order(self) -> None:
        """Saga compensation runs in reverse order of completion."""
        from src.pipelines.saga import PipelineSaga
        
        compensation_order: list[str] = []
        
        saga = PipelineSaga(name="reverse_test")
        
        async def compensate_first(context: dict) -> None:
            compensation_order.append("first")
        
        async def compensate_second(context: dict) -> None:
            compensation_order.append("second")
        
        async def compensate_third(context: dict) -> None:
            compensation_order.append("third")
        
        saga.register_compensation("first", compensate_first)
        saga.register_compensation("second", compensate_second)
        saga.register_compensation("third", compensate_third)
        
        # Complete in order: first, second, third
        saga.record_completion("first", {})
        saga.record_completion("second", {})
        saga.record_completion("third", {})
        
        # Compensate
        await saga.compensate()
        
        # Should compensate in reverse: third, second, first
        assert compensation_order == ["third", "second", "first"]

    @pytest.mark.asyncio
    async def test_saga_compensation_receives_context(self) -> None:
        """Compensation handlers receive stage context."""
        from src.pipelines.saga import PipelineSaga
        
        received_contexts: list[dict] = []
        
        saga = PipelineSaga(name="context_test")
        
        async def compensate(context: dict) -> None:
            received_contexts.append(context)
        
        saga.register_compensation("data_stage", compensate)
        saga.record_completion("data_stage", {"created_id": "12345", "temp_file": "/tmp/x"})
        
        await saga.compensate()
        
        assert len(received_contexts) == 1
        assert received_contexts[0]["created_id"] == "12345"
        assert received_contexts[0]["temp_file"] == "/tmp/x"

    @pytest.mark.asyncio
    async def test_saga_only_compensates_completed_stages(self) -> None:
        """Saga only compensates stages that actually completed."""
        from src.pipelines.saga import PipelineSaga
        
        compensation_log: list[str] = []
        
        saga = PipelineSaga(name="partial_test")
        
        async def compensate_a(context: dict) -> None:
            compensation_log.append("a")
        
        async def compensate_b(context: dict) -> None:
            compensation_log.append("b")
        
        async def compensate_c(context: dict) -> None:
            compensation_log.append("c")
        
        saga.register_compensation("stage_a", compensate_a)
        saga.register_compensation("stage_b", compensate_b)
        saga.register_compensation("stage_c", compensate_c)
        
        # Only a and b completed (c failed, never recorded)
        saga.record_completion("stage_a", {})
        saga.record_completion("stage_b", {})
        
        await saga.compensate()
        
        # Only a and b should be compensated
        assert "a" in compensation_log
        assert "b" in compensation_log
        assert "c" not in compensation_log


# =============================================================================
# Saga Error Handling Tests
# =============================================================================

class TestSagaErrorHandling:
    """Tests for saga error handling during compensation."""

    @pytest.mark.asyncio
    async def test_saga_continues_on_compensation_error(self) -> None:
        """Saga continues compensation even if one handler fails."""
        from src.pipelines.saga import PipelineSaga
        
        compensation_log: list[str] = []
        
        saga = PipelineSaga(name="error_resilient")
        
        async def compensate_good(context: dict) -> None:
            compensation_log.append("good")
        
        async def compensate_bad(context: dict) -> None:
            raise RuntimeError("Compensation failed!")
        
        async def compensate_also_good(context: dict) -> None:
            compensation_log.append("also_good")
        
        saga.register_compensation("first", compensate_also_good)
        saga.register_compensation("second", compensate_bad)
        saga.register_compensation("third", compensate_good)
        
        saga.record_completion("first", {})
        saga.record_completion("second", {})
        saga.record_completion("third", {})
        
        # Compensation should complete despite error
        result = await saga.compensate()
        
        # Both good handlers should have run
        assert "good" in compensation_log
        assert "also_good" in compensation_log
        
        # Result should include the error
        assert len(result.errors) == 1
        assert "second" in str(result.errors[0])

    @pytest.mark.asyncio
    async def test_saga_tracks_compensation_result(self) -> None:
        """Saga returns compensation result with success/failure info."""
        from src.pipelines.saga import PipelineSaga, CompensationResult
        
        saga = PipelineSaga(name="result_test")
        
        async def success_handler(context: dict) -> None:
            pass
        
        async def fail_handler(context: dict) -> None:
            raise ValueError("oops")
        
        saga.register_compensation("good_stage", success_handler)
        saga.register_compensation("bad_stage", fail_handler)
        
        saga.record_completion("good_stage", {})
        saga.record_completion("bad_stage", {})
        
        result = await saga.compensate()
        
        assert isinstance(result, CompensationResult)
        assert result.stages_compensated == 2
        assert result.stages_succeeded == 1
        assert result.stages_failed == 1


# =============================================================================
# Saga Integration with Pipeline Tests
# =============================================================================

class TestSagaPipelineIntegration:
    """Tests for saga integration with PipelineOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_saga_on_failure(self) -> None:
        """PipelineOrchestrator uses saga pattern on failure (AC-14.5)."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator,
            PipelineDefinition,
            StageDefinition,
        )
        from src.functions.base import AgentFunction
        
        cleanup_called = False
        
        class SuccessOutput(BaseModel):
            created_id: str
        
        class CreateFunc(AgentFunction):
            name = "create_resource"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> SuccessOutput:
                return SuccessOutput(created_id="resource_123")
            
            async def compensate(self, context: dict) -> None:
                nonlocal cleanup_called
                cleanup_called = True
        
        class FailOutput(BaseModel):
            result: str
        
        class FailFunc(AgentFunction):
            name = "failing_operation"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> FailOutput:
                raise RuntimeError("Operation failed!")
        
        pipeline = PipelineDefinition(
            name="saga_pipeline",
            stages=[
                StageDefinition(
                    name="create",
                    function="create_resource",
                    compensatable=True,
                ),
                StageDefinition(
                    name="fail",
                    function="failing_operation",
                    depends_on=["create"],
                ),
            ],
        )
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(CreateFunc())
        orchestrator.register_function(FailFunc())
        
        result = await orchestrator.execute(pipeline)
        
        # Pipeline should have failed
        assert not result.success
        
        # Compensation should have been called
        assert cleanup_called

    @pytest.mark.asyncio
    async def test_saga_preserves_partial_results(self) -> None:
        """Saga preserves partial results after compensation."""
        from src.pipelines.orchestrator import (
            PipelineOrchestrator,
            PipelineDefinition,
            StageDefinition,
        )
        from src.functions.base import AgentFunction
        
        class Output(BaseModel):
            value: str
        
        class SuccessFunc(AgentFunction):
            name = "success"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                return Output(value="completed")
        
        class FailFunc(AgentFunction):
            name = "fail"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> Output:
                raise RuntimeError("Failed!")
        
        pipeline = PipelineDefinition(
            name="partial_results",
            stages=[
                StageDefinition(name="first", function="success"),
                StageDefinition(name="second", function="success", depends_on=["first"]),
                StageDefinition(name="third", function="fail", depends_on=["second"]),
            ],
        )
        
        orchestrator = PipelineOrchestrator()
        orchestrator.register_function(SuccessFunc())
        orchestrator.register_function(FailFunc())
        
        result = await orchestrator.execute(pipeline)
        
        # Should have partial results from first two stages
        assert not result.success
        assert "first" in result.outputs
        assert "second" in result.outputs
        assert result.outputs["first"]["value"] == "completed"


# =============================================================================
# Compensatable Stage Definition Tests
# =============================================================================

class TestCompensatableStages:
    """Tests for compensatable stage configuration."""

    def test_stage_can_be_marked_compensatable(self) -> None:
        """StageDefinition can be marked as compensatable."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="resource_create",
            function="create_resource",
            compensatable=True,
        )
        
        assert stage.compensatable is True

    def test_stage_compensatable_default_false(self) -> None:
        """StageDefinition.compensatable defaults to False."""
        from src.pipelines.orchestrator import StageDefinition
        
        stage = StageDefinition(
            name="simple_stage",
            function="simple_func",
        )
        
        assert stage.compensatable is False


__all__ = [
    "TestPipelineSaga",
    "TestSagaCompensation",
    "TestSagaErrorHandling",
    "TestSagaPipelineIntegration",
    "TestCompensatableStages",
]
