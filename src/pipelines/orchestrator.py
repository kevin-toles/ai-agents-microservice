"""Pipeline Orchestrator for executing function DAGs.

WBS-AGT14: Pipeline Orchestrator

Implements:
- PipelineDefinition and StageDefinition schemas
- PipelineOrchestrator for DAG execution
- Conditional stage support
- Handoff state management

Acceptance Criteria:
- AC-14.1: PipelineOrchestrator executes function DAGs
- AC-14.3: HandoffState flows between pipeline stages
- AC-14.4: Conditional pipeline stages supported

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from src.functions.base import AgentFunction


# =============================================================================
# Enums
# =============================================================================

class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# =============================================================================
# Pydantic Schemas
# =============================================================================

class StageCondition(BaseModel):
    """Condition for conditional stage execution."""

    model_config = ConfigDict(frozen=True)

    expression: str = Field(
        ...,
        description="Expression to evaluate (e.g., 'previous_output.success == True')",
    )
    skip_on_false: bool = Field(
        default=True,
        description="If True, skip stage when condition is False; if False, fail pipeline",
    )


class RetryConfig(BaseModel):
    """Configuration for stage retry behavior."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier",
    )
    initial_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay in seconds before first retry",
    )


class StageDefinition(BaseModel):
    """Definition of a pipeline stage."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique name for this stage",
    )
    function: str = Field(
        ...,
        min_length=1,
        description="Name of the registered function to execute",
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map of function inputs to handoff state keys (e.g., {'query': 'previous_stage.result'})",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of stage names this stage depends on",
    )
    output_key: str | None = Field(
        default=None,
        description="Key under which to store the output in handoff state",
    )
    preset: str | None = Field(
        default=None,
        description="Preset override for this stage",
    )
    condition: StageCondition | None = Field(
        default=None,
        description="Optional condition for stage execution",
    )
    retry_config: RetryConfig | None = Field(
        default=None,
        description="Optional retry configuration",
    )
    compensatable: bool = Field(
        default=False,
        description="Whether this stage supports compensation (saga pattern)",
    )


class PipelineDefinition(BaseModel):
    """Definition of a complete pipeline."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique name for this pipeline",
    )
    stages: list[StageDefinition] = Field(
        ...,
        min_length=1,
        description="Ordered list of stages in the pipeline",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the pipeline",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the pipeline definition",
    )


class StageResult(BaseModel):
    """Result of executing a single stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage_name: str
    status: StageStatus
    output: Any | None = None
    error: str | None = None
    duration_ms: float = 0.0
    retries: int = 0


class PipelineResult(BaseModel):
    """Result of executing a complete pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    pipeline_name: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    stage_results: dict[str, StageResult] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
    error: str | None = None
    failed_stage: str | None = None


# =============================================================================
# Handoff State
# =============================================================================

@dataclass
class HandoffState:
    """State passed between pipeline stages."""

    data: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from handoff state.

        Supports dotted notation: 'stage_name.output_field'
        """
        if "." in key:
            parts = key.split(".", 1)
            stage_name, field_name = parts
            if stage_name in self.stage_outputs:
                stage_out = self.stage_outputs[stage_name]
                if isinstance(stage_out, dict):
                    return stage_out.get(field_name, default)
                if hasattr(stage_out, field_name):
                    return getattr(stage_out, field_name, default)

        # Direct lookup
        if key in self.data:
            return self.data[key]
        if key in self.stage_outputs:
            return self.stage_outputs[key]
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a value in handoff state."""
        self.data[key] = value

    def set_stage_output(self, stage_name: str, output: Any) -> None:
        """Record output from a stage."""
        self.stage_outputs[stage_name] = output
        # Also convert to dict for easier access
        if hasattr(output, "model_dump"):
            self.data[stage_name] = output.model_dump()
        elif hasattr(output, "__dict__"):
            self.data[stage_name] = vars(output)
        else:
            self.data[stage_name] = output


# =============================================================================
# Condition Evaluator
# =============================================================================

class ConditionEvaluator:
    """Evaluates stage conditions."""

    def evaluate(self, condition: StageCondition, state: HandoffState) -> bool:
        """Evaluate a condition against current handoff state."""
        # Create evaluation context
        context = {
            **state.data,
            **state.stage_outputs,
            "handoff": state,
        }

        try:
            # Simple expression evaluation
            # In production, use a safer expression parser
            result = eval(condition.expression, {"__builtins__": {}}, context)
            return bool(result)
        except Exception:
            # If evaluation fails, treat as False
            return False


# =============================================================================
# DAG Builder
# =============================================================================

class DAGBuilder:
    """Builds execution order from stage dependencies."""

    def _build_dependency_graph(
        self, stages: list[StageDefinition]
    ) -> tuple[dict[str, StageDefinition], dict[str, int], dict[str, list[str]]]:
        """Build adjacency list and in-degree count for topological sort."""
        stage_map = {s.name: s for s in stages}
        in_degree: dict[str, int] = {s.name: 0 for s in stages}
        dependents: dict[str, list[str]] = {s.name: [] for s in stages}

        for stage in stages:
            for dep in stage.depends_on:
                if dep not in stage_map:
                    raise ValueError(f"Stage '{stage.name}' depends on unknown stage '{dep}'")
                in_degree[stage.name] += 1
                dependents[dep].append(stage.name)

        return stage_map, in_degree, dependents

    def build_execution_order(
        self, stages: list[StageDefinition]
    ) -> list[list[str]]:
        """Build execution order using topological sort.

        Returns list of lists - each inner list contains stages
        that can be executed in parallel.
        """
        _, in_degree, dependents = self._build_dependency_graph(stages)

        # Kahn's algorithm for topological sort
        execution_order: list[list[str]] = []
        ready = [name for name, degree in in_degree.items() if degree == 0]

        while ready:
            execution_order.append(sorted(ready))
            next_ready: list[str] = []
            for stage_name in ready:
                for dependent in dependents[stage_name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)
            ready = next_ready

        # Check for cycles
        total_scheduled = sum(len(level) for level in execution_order)
        if total_scheduled != len(stages):
            raise ValueError("Pipeline contains cyclic dependencies")

        return execution_order


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class PipelineOrchestrator:
    """Orchestrates execution of pipeline DAGs.

    Implements AC-14.1: PipelineOrchestrator executes function DAGs
    """

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self._function_registry: dict[str, AgentFunction] = {}
        self._dag_builder = DAGBuilder()
        self._condition_evaluator = ConditionEvaluator()

    @property
    def function_registry(self) -> dict[str, AgentFunction]:
        """Get the function registry."""
        return self._function_registry

    def register_function(self, function: AgentFunction) -> None:
        """Register a function for use in pipelines."""
        self._function_registry[function.name] = function

    def build_dag(self, pipeline: PipelineDefinition) -> list[list[str]]:
        """Build DAG execution order for a pipeline."""
        return self._dag_builder.build_execution_order(pipeline.stages)

    def _create_failure_result(
        self,
        pipeline_name: str,
        start_time: float,
        error: str,
        outputs: dict[str, Any] | None = None,
        stage_results: dict[str, StageResult] | None = None,
        failed_stage: str | None = None,
    ) -> PipelineResult:
        """Create a failure PipelineResult."""
        return PipelineResult(
            success=False,
            pipeline_name=pipeline_name,
            outputs=outputs or {},
            stage_results=stage_results or {},
            total_duration_ms=(time.time() - start_time) * 1000,
            error=error,
            failed_stage=failed_stage,
        )

    def _process_stage_result(
        self,
        stage: StageDefinition,
        result: StageResult | BaseException,
        stage_results: dict[str, StageResult],
        outputs: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Process a single stage result. Returns (failed_stage, error_message) if failed."""
        if isinstance(result, BaseException):
            stage_results[stage.name] = StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=str(result),
            )
            return stage.name, str(result)

        stage_results[stage.name] = result
        if result.output is not None:
            output_dict = (
                result.output.model_dump()
                if hasattr(result.output, "model_dump")
                else result.output
            )
            outputs[stage.name] = output_dict
        return None, None

    async def _execute_level(
        self,
        level: list[str],
        stage_map: dict[str, StageDefinition],
        state: HandoffState,
        saga: Any,
    ) -> list[tuple[StageDefinition, StageResult | BaseException]]:
        """Execute all stages at a given level in parallel."""
        level_tasks = []
        level_stages = []

        for stage_name in level:
            stage = stage_map[stage_name]
            level_stages.append(stage)
            level_tasks.append(self._execute_stage(stage, state, saga))

        results = await asyncio.gather(*level_tasks, return_exceptions=True)
        return list(zip(level_stages, results, strict=False))

    async def _execute_all_levels(
        self,
        execution_order: list[list[str]],
        stage_map: dict[str, StageDefinition],
        state: HandoffState,
        saga: Any,
        stage_results: dict[str, StageResult],
        outputs: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Execute all levels of the pipeline.

        Returns:
            Tuple of (failed_stage, error_message) if any stage failed, else (None, None)
        """
        for level in execution_order:
            level_results = await self._execute_level(level, stage_map, state, saga)

            for stage, result in level_results:
                failed, _ = self._process_stage_result(stage, result, stage_results, outputs)
                if failed:
                    if isinstance(result, Exception):
                        raise result
                    raise RuntimeError(str(result)) from result  # type: ignore[arg-type]

        return None, None

    async def execute(
        self,
        pipeline: PipelineDefinition,
        initial_inputs: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute a pipeline.

        Args:
            pipeline: The pipeline definition to execute
            initial_inputs: Initial inputs to provide to the pipeline

        Returns:
            PipelineResult with success status, outputs, and timing
        """
        from src.pipelines.saga import PipelineSaga

        start_time = time.time()

        state = HandoffState()
        if initial_inputs:
            for key, value in initial_inputs.items():
                state.set(key, value)

        try:
            execution_order = self._dag_builder.build_execution_order(pipeline.stages)
        except ValueError as e:
            return self._create_failure_result(pipeline.name, start_time, str(e))

        stage_map = {s.name: s for s in pipeline.stages}
        saga = PipelineSaga(name=f"{pipeline.name}_saga")
        stage_results: dict[str, StageResult] = {}
        outputs: dict[str, Any] = {}

        try:
            await self._execute_all_levels(execution_order, stage_map, state, saga, stage_results, outputs)
        except Exception as e:
            await saga.compensate()
            return self._create_failure_result(pipeline.name, start_time, str(e), outputs, stage_results)

        return PipelineResult(
            success=True,
            pipeline_name=pipeline.name,
            outputs=outputs,
            stage_results=stage_results,
            total_duration_ms=(time.time() - start_time) * 1000,
        )

    def _check_stage_condition(
        self,
        stage: StageDefinition,
        state: HandoffState,
        start_time: float,
    ) -> StageResult | None:
        """Check stage condition. Returns StageResult if should skip, None to continue."""
        if not stage.condition:
            return None

        should_run = self._condition_evaluator.evaluate(stage.condition, state)
        if should_run:
            return None

        if stage.condition.skip_on_false:
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.SKIPPED,
                duration_ms=(time.time() - start_time) * 1000,
            )

        raise RuntimeError(f"Stage '{stage.name}' condition not met: {stage.condition.expression}")

    async def _execute_with_retry(
        self,
        function: AgentFunction,
        inputs: dict[str, Any],
        retry_config: RetryConfig,
    ) -> Any:
        """Execute function with retry logic."""
        retries = 0
        last_error: Exception | None = None

        while retries <= retry_config.max_retries:
            try:
                return await function.run(**inputs), retries
            except Exception as e:
                last_error = e
                retries += 1
                if retries <= retry_config.max_retries:
                    delay = retry_config.initial_delay * (retry_config.backoff_factor ** (retries - 1))
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError(f"Function '{function.name}' failed")

    def _record_saga_completion(
        self,
        stage: StageDefinition,
        function: AgentFunction,
        output: Any,
        saga: Any,
    ) -> None:
        """Record stage completion for saga compensation if compensatable."""
        if not stage.compensatable or not hasattr(function, "compensate"):
            return

        output_dict = output.model_dump() if hasattr(output, "model_dump") else output
        saga.register_compensation(stage.name, function.compensate)
        saga.record_completion(stage.name, output_dict)

    async def _execute_stage(
        self,
        stage: StageDefinition,
        state: HandoffState,
        saga: Any,
    ) -> StageResult:
        """Execute a single pipeline stage."""
        start_time = time.time()

        if skip_result := self._check_stage_condition(stage, state, start_time):
            return skip_result

        if stage.function not in self._function_registry:
            raise ValueError(f"Function '{stage.function}' not found in registry")

        function = self._function_registry[stage.function]
        inputs = self._build_stage_inputs(stage, state)
        retry_config = stage.retry_config or RetryConfig(max_retries=0)

        output, retries = await self._execute_with_retry(function, inputs, retry_config)

        output_key = stage.output_key or stage.name
        state.set_stage_output(output_key, output)
        self._record_saga_completion(stage, function, output, saga)

        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            output=output,
            duration_ms=(time.time() - start_time) * 1000,
            retries=retries,
        )

    def _build_stage_inputs(
        self, stage: StageDefinition, state: HandoffState
    ) -> dict[str, Any]:
        """Build inputs for a stage from handoff state."""
        inputs: dict[str, Any] = {}

        for input_name, state_key in stage.input_mapping.items():
            value = state.get(state_key)
            if value is not None:
                inputs[input_name] = value

        return inputs


__all__ = [
    "ConditionEvaluator",
    "DAGBuilder",
    "HandoffState",
    "PipelineDefinition",
    "PipelineOrchestrator",
    "PipelineResult",
    "RetryConfig",
    "StageCondition",
    "StageDefinition",
    "StageResult",
    "StageStatus",
]
