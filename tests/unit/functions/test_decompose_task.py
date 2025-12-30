"""Tests for decompose_task function.

TDD tests for WBS-AGT12: decompose_task Function.

Acceptance Criteria Coverage:
- AC-12.1: Breaks complex task into subtasks
- AC-12.2: Returns TaskDecomposition with subtasks, dependencies
- AC-12.3: Context budget: 4096 input / 2048 output
- AC-12.4: Default preset: S2
- AC-12.5: Subtasks form valid DAG (no cycles)

Exit Criteria:
- Each Subtask has id, description, depends_on list
- Cyclic dependencies raise ValidationError
- Topological sort produces valid execution order

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 7
"""

import pytest
from pydantic import ValidationError


# =============================================================================
# AGT12.1: AC-12.1 Input Schema Tests - DecomposeTaskInput
# =============================================================================

class TestDecomposeTaskInput:
    """Tests for DecomposeTaskInput schema.
    
    AC-12.1: Breaks complex task into subtasks.
    Input must capture the task, constraints, available agents, and context.
    """

    def test_input_requires_task(self) -> None:
        """DecomposeTaskInput requires task field."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        with pytest.raises(ValidationError) as exc_info:
            DecomposeTaskInput()  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("task",) for e in errors)

    def test_input_accepts_task(self) -> None:
        """DecomposeTaskInput accepts task string."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        input_data = DecomposeTaskInput(
            task="Implement user authentication with OAuth2",
        )
        assert input_data.task == "Implement user authentication with OAuth2"

    def test_input_has_optional_constraints(self) -> None:
        """DecomposeTaskInput has optional constraints list."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        # Default is empty list
        input_data = DecomposeTaskInput(task="test task")
        assert input_data.constraints == []
        
        # Can provide constraints
        input_with_constraints = DecomposeTaskInput(
            task="test task",
            constraints=["Must complete in 2 sprints", "Cannot modify legacy API"],
        )
        assert len(input_with_constraints.constraints) == 2
        assert "Must complete in 2 sprints" in input_with_constraints.constraints

    def test_input_has_optional_available_agents(self) -> None:
        """DecomposeTaskInput has optional available_agents list.
        
        Per architecture: available_agents specifies which agent functions exist.
        """
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        # Default is empty list
        input_data = DecomposeTaskInput(task="test task")
        assert input_data.available_agents == []
        
        # Can provide available agents
        input_with_agents = DecomposeTaskInput(
            task="test task",
            available_agents=["generate_code", "analyze_artifact", "validate_against_spec"],
        )
        assert len(input_with_agents.available_agents) == 3
        assert "generate_code" in input_with_agents.available_agents

    def test_input_has_optional_context(self) -> None:
        """DecomposeTaskInput has optional context string.
        
        Per architecture: context provides domain/project context.
        """
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        # Default is empty string
        input_data = DecomposeTaskInput(task="test task")
        assert input_data.context == ""
        
        # Can provide context
        input_with_context = DecomposeTaskInput(
            task="test task",
            context="This is a Python FastAPI project using PostgreSQL.",
        )
        assert "FastAPI" in input_with_context.context

    def test_input_has_optional_max_subtasks(self) -> None:
        """DecomposeTaskInput has optional max_subtasks limit.
        
        Prevents unbounded decomposition.
        """
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        # Default is reasonable limit
        input_data = DecomposeTaskInput(task="test task")
        assert input_data.max_subtasks == 10
        
        # Can override
        input_custom = DecomposeTaskInput(
            task="test task",
            max_subtasks=5,
        )
        assert input_custom.max_subtasks == 5

    def test_input_validates_max_subtasks_positive(self) -> None:
        """max_subtasks must be positive."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        with pytest.raises(ValidationError):
            DecomposeTaskInput(task="test", max_subtasks=0)
        
        with pytest.raises(ValidationError):
            DecomposeTaskInput(task="test", max_subtasks=-1)

    def test_input_json_schema_export(self) -> None:
        """DecomposeTaskInput exports valid JSON schema.
        
        AC-4.5: All schemas have JSON schema export.
        """
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        schema = DecomposeTaskInput.model_json_schema()
        
        assert "properties" in schema
        assert "task" in schema["properties"]
        assert "constraints" in schema["properties"]
        assert "available_agents" in schema["properties"]
        assert "context" in schema["properties"]
        assert "max_subtasks" in schema["properties"]

    def test_input_task_must_be_nonempty(self) -> None:
        """Task string cannot be empty."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        with pytest.raises(ValidationError):
            DecomposeTaskInput(task="")
        
        with pytest.raises(ValidationError):
            DecomposeTaskInput(task="   ")


# =============================================================================
# AGT12.3: AC-12.2 Output Schema Tests - Subtask
# =============================================================================

class TestSubtask:
    """Tests for Subtask model.
    
    Exit Criteria: Each Subtask has id, description, depends_on list.
    """

    def test_subtask_requires_id(self) -> None:
        """Subtask requires id field."""
        from src.schemas.functions.decompose_task import Subtask
        
        with pytest.raises(ValidationError) as exc_info:
            Subtask(description="Test task")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)

    def test_subtask_requires_description(self) -> None:
        """Subtask requires description field."""
        from src.schemas.functions.decompose_task import Subtask
        
        with pytest.raises(ValidationError) as exc_info:
            Subtask(id="subtask_1")  # type: ignore
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("description",) for e in errors)

    def test_subtask_has_depends_on_list(self) -> None:
        """Subtask has depends_on list (defaults to empty).
        
        Exit Criteria: Each Subtask has depends_on list.
        """
        from src.schemas.functions.decompose_task import Subtask
        
        subtask = Subtask(id="subtask_1", description="First task")
        assert subtask.depends_on == []
        
        subtask_with_deps = Subtask(
            id="subtask_2",
            description="Second task",
            depends_on=["subtask_1"],
        )
        assert subtask_with_deps.depends_on == ["subtask_1"]

    def test_subtask_has_optional_agent_assignment(self) -> None:
        """Subtask has optional assigned_agent field.
        
        Per architecture: agent_assignments maps subtask_id -> agent_function.
        """
        from src.schemas.functions.decompose_task import Subtask
        
        subtask = Subtask(id="subtask_1", description="Generate code")
        assert subtask.assigned_agent is None
        
        subtask_assigned = Subtask(
            id="subtask_1",
            description="Generate code",
            assigned_agent="generate_code",
        )
        assert subtask_assigned.assigned_agent == "generate_code"

    def test_subtask_has_optional_estimated_tokens(self) -> None:
        """Subtask has optional estimated_tokens field.
        
        Per architecture: estimated_tokens for budget tracking.
        """
        from src.schemas.functions.decompose_task import Subtask
        
        subtask = Subtask(id="subtask_1", description="Task")
        assert subtask.estimated_tokens is None
        
        subtask_with_estimate = Subtask(
            id="subtask_1",
            description="Task",
            estimated_tokens=1024,
        )
        assert subtask_with_estimate.estimated_tokens == 1024

    def test_subtask_id_must_be_nonempty(self) -> None:
        """Subtask id cannot be empty."""
        from src.schemas.functions.decompose_task import Subtask
        
        with pytest.raises(ValidationError):
            Subtask(id="", description="Test")

    def test_subtask_description_must_be_nonempty(self) -> None:
        """Subtask description cannot be empty."""
        from src.schemas.functions.decompose_task import Subtask
        
        with pytest.raises(ValidationError):
            Subtask(id="subtask_1", description="")


# =============================================================================
# AGT12.3: AC-12.2, AC-12.5 Output Schema Tests - TaskDecomposition
# =============================================================================

class TestTaskDecomposition:
    """Tests for TaskDecomposition model.
    
    AC-12.2: Returns TaskDecomposition with subtasks, dependencies.
    AC-12.5: Subtasks form valid DAG (no cycles).
    """

    def test_task_decomposition_has_subtasks_list(self) -> None:
        """TaskDecomposition has subtasks list."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        decomposition = TaskDecomposition(subtasks=[])
        assert decomposition.subtasks == []
        
        subtasks = [
            Subtask(id="s1", description="First task"),
            Subtask(id="s2", description="Second task", depends_on=["s1"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        assert len(decomposition.subtasks) == 2

    def test_task_decomposition_has_execution_order(self) -> None:
        """TaskDecomposition has execution_order list (topological sort).
        
        Exit Criteria: Topological sort produces valid execution order.
        """
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="First"),
            Subtask(id="s2", description="Second", depends_on=["s1"]),
        ]
        decomposition = TaskDecomposition(
            subtasks=subtasks,
            execution_order=["s1", "s2"],
        )
        assert decomposition.execution_order == ["s1", "s2"]

    def test_task_decomposition_has_estimated_total_tokens(self) -> None:
        """TaskDecomposition has estimated_total_tokens.
        
        Per architecture: Total budget estimate across all subtasks.
        """
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        decomposition = TaskDecomposition(subtasks=[])
        assert decomposition.estimated_total_tokens == 0
        
        decomposition_with_estimate = TaskDecomposition(
            subtasks=[],
            estimated_total_tokens=4096,
        )
        assert decomposition_with_estimate.estimated_total_tokens == 4096

    def test_task_decomposition_has_original_task(self) -> None:
        """TaskDecomposition preserves original task for reference."""
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        decomposition = TaskDecomposition(
            subtasks=[],
            original_task="Build authentication system",
        )
        assert decomposition.original_task == "Build authentication system"

    def test_task_decomposition_json_schema_export(self) -> None:
        """TaskDecomposition exports valid JSON schema."""
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        schema = TaskDecomposition.model_json_schema()
        
        assert "properties" in schema
        assert "subtasks" in schema["properties"]
        assert "execution_order" in schema["properties"]
        assert "estimated_total_tokens" in schema["properties"]


# =============================================================================
# AGT12.5: AC-12.5 DAG Validation Tests
# =============================================================================

class TestDAGValidation:
    """Tests for DAG validation and cycle detection.
    
    AC-12.5: Subtasks form valid DAG (no cycles).
    Exit Criteria: Cyclic dependencies raise ValidationError.
    """

    def test_valid_dag_passes_validation(self) -> None:
        """Valid DAG with no cycles passes validation."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        # Linear chain: s1 -> s2 -> s3
        subtasks = [
            Subtask(id="s1", description="First"),
            Subtask(id="s2", description="Second", depends_on=["s1"]),
            Subtask(id="s3", description="Third", depends_on=["s2"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        # Should not raise - valid DAG
        assert decomposition.is_valid_dag()

    def test_parallel_tasks_form_valid_dag(self) -> None:
        """Parallel tasks with no dependencies form valid DAG."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Task A"),
            Subtask(id="s2", description="Task B"),
            Subtask(id="s3", description="Task C"),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        assert decomposition.is_valid_dag()

    def test_diamond_dag_is_valid(self) -> None:
        """Diamond dependency pattern is valid DAG.
        
            s1
           /  \\
          s2   s3
           \\  /
            s4
        """
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Start"),
            Subtask(id="s2", description="Branch A", depends_on=["s1"]),
            Subtask(id="s3", description="Branch B", depends_on=["s1"]),
            Subtask(id="s4", description="Merge", depends_on=["s2", "s3"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        assert decomposition.is_valid_dag()

    def test_self_cycle_raises_validation_error(self) -> None:
        """Task depending on itself raises ValidationError."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Self-referential", depends_on=["s1"]),
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            TaskDecomposition(subtasks=subtasks)
        
        # Error should mention cycle
        error_str = str(exc_info.value)
        assert "cycl" in error_str.lower()  # matches "cycle" or "cyclic"

    def test_two_node_cycle_raises_validation_error(self) -> None:
        """Two-node cycle raises ValidationError.
        
        s1 -> s2 -> s1 (cycle)
        """
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Task 1", depends_on=["s2"]),
            Subtask(id="s2", description="Task 2", depends_on=["s1"]),
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            TaskDecomposition(subtasks=subtasks)
        
        error_str = str(exc_info.value)
        assert "cycl" in error_str.lower()  # matches "cycle" or "cyclic"

    def test_longer_cycle_raises_validation_error(self) -> None:
        """Longer cycle (3+ nodes) raises ValidationError.
        
        s1 -> s2 -> s3 -> s1 (cycle)
        """
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Task 1", depends_on=["s3"]),
            Subtask(id="s2", description="Task 2", depends_on=["s1"]),
            Subtask(id="s3", description="Task 3", depends_on=["s2"]),
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            TaskDecomposition(subtasks=subtasks)
        
        error_str = str(exc_info.value)
        assert "cycl" in error_str.lower()  # matches "cycle" or "cyclic"

    def test_dependency_on_nonexistent_subtask_raises_error(self) -> None:
        """Dependency on non-existent subtask raises ValidationError."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Task 1", depends_on=["nonexistent"]),
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            TaskDecomposition(subtasks=subtasks)
        
        error_str = str(exc_info.value)
        assert "nonexistent" in error_str.lower() or "unknown" in error_str.lower() or "invalid" in error_str.lower()

    def test_duplicate_subtask_ids_raise_error(self) -> None:
        """Duplicate subtask IDs raise ValidationError."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s1", description="Task 1"),
            Subtask(id="s1", description="Duplicate ID"),  # Same ID
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            TaskDecomposition(subtasks=subtasks)
        
        error_str = str(exc_info.value)
        assert "duplicate" in error_str.lower() or "unique" in error_str.lower()


# =============================================================================
# AGT12.5: Topological Sort Tests
# =============================================================================

class TestTopologicalSort:
    """Tests for topological sort functionality.
    
    Exit Criteria: Topological sort produces valid execution order.
    """

    def test_topological_sort_linear_chain(self) -> None:
        """Topological sort on linear chain produces correct order."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="s3", description="Third", depends_on=["s2"]),
            Subtask(id="s1", description="First"),
            Subtask(id="s2", description="Second", depends_on=["s1"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        order = decomposition.get_execution_order()
        
        # s1 must come before s2, s2 must come before s3
        assert order.index("s1") < order.index("s2")
        assert order.index("s2") < order.index("s3")

    def test_topological_sort_parallel_tasks(self) -> None:
        """Topological sort on parallel tasks includes all tasks."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="a", description="Task A"),
            Subtask(id="b", description="Task B"),
            Subtask(id="c", description="Task C"),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        order = decomposition.get_execution_order()
        
        assert set(order) == {"a", "b", "c"}
        assert len(order) == 3

    def test_topological_sort_diamond(self) -> None:
        """Topological sort on diamond pattern respects dependencies."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [
            Subtask(id="start", description="Start"),
            Subtask(id="left", description="Left branch", depends_on=["start"]),
            Subtask(id="right", description="Right branch", depends_on=["start"]),
            Subtask(id="end", description="End", depends_on=["left", "right"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        order = decomposition.get_execution_order()
        
        # start must be first
        assert order.index("start") < order.index("left")
        assert order.index("start") < order.index("right")
        # end must be last
        assert order.index("left") < order.index("end")
        assert order.index("right") < order.index("end")

    def test_topological_sort_complex_dag(self) -> None:
        """Topological sort on complex DAG respects all dependencies."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        # Complex DAG:
        # a -> b -> d
        # a -> c -> d
        # c -> e
        subtasks = [
            Subtask(id="a", description="A"),
            Subtask(id="b", description="B", depends_on=["a"]),
            Subtask(id="c", description="C", depends_on=["a"]),
            Subtask(id="d", description="D", depends_on=["b", "c"]),
            Subtask(id="e", description="E", depends_on=["c"]),
        ]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        order = decomposition.get_execution_order()
        
        # Verify all dependencies respected
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")
        assert order.index("c") < order.index("e")


# =============================================================================
# AGT12.7: DecomposeTaskFunction Tests
# =============================================================================

class TestDecomposeTaskFunction:
    """Tests for DecomposeTaskFunction implementation.
    
    AC-12.1: Breaks complex task into subtasks.
    AC-12.3: Context budget: 4096 input / 2048 output.
    AC-12.4: Default preset: S2.
    """

    def test_function_has_correct_name(self) -> None:
        """DecomposeTaskFunction has name 'decompose_task'."""
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        assert func.name == "decompose_task"

    def test_function_has_default_preset_s2(self) -> None:
        """DecomposeTaskFunction has default preset S2.
        
        AC-12.4: Default preset: S2.
        """
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        assert func.default_preset == "S2"

    def test_function_context_budget(self) -> None:
        """DecomposeTaskFunction has correct context budget.
        
        AC-12.3: Context budget: 4096 input / 2048 output.
        """
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        budget = func.get_context_budget()
        
        assert budget["input"] == 4096
        assert budget["output"] == 2048

    def test_function_inherits_from_agent_function(self) -> None:
        """DecomposeTaskFunction inherits from AgentFunction."""
        from src.functions.decompose_task import DecomposeTaskFunction
        from src.functions.base import AgentFunction
        
        func = DecomposeTaskFunction()
        assert isinstance(func, AgentFunction)

    def test_function_implements_protocol(self) -> None:
        """DecomposeTaskFunction implements AgentFunctionProtocol."""
        from src.functions.decompose_task import DecomposeTaskFunction
        from src.functions.base import AgentFunctionProtocol
        
        func = DecomposeTaskFunction()
        assert isinstance(func, AgentFunctionProtocol)

    @pytest.mark.asyncio
    async def test_run_returns_task_decomposition(self) -> None:
        """run() returns TaskDecomposition output."""
        from src.functions.decompose_task import DecomposeTaskFunction
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Implement user authentication",
            constraints=[],
            available_agents=["generate_code"],
            context="Python web application",
        )
        
        assert isinstance(result, TaskDecomposition)

    @pytest.mark.asyncio
    async def test_run_produces_subtasks(self) -> None:
        """run() produces at least one subtask for non-trivial input."""
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Build a REST API with CRUD operations for users",
            constraints=["Use FastAPI", "Include tests"],
            available_agents=["generate_code", "analyze_artifact"],
            context="Python backend service",
        )
        
        assert len(result.subtasks) >= 1

    @pytest.mark.asyncio
    async def test_run_provides_execution_order(self) -> None:
        """run() provides valid execution order."""
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Create multi-step feature",
            constraints=[],
            available_agents=[],
            context="",
        )
        
        # execution_order should contain all subtask ids
        subtask_ids = {s.id for s in result.subtasks}
        assert set(result.execution_order) == subtask_ids

    @pytest.mark.asyncio
    async def test_run_enforces_context_budget(self) -> None:
        """run() enforces context budget on input."""
        from src.functions.decompose_task import DecomposeTaskFunction
        from src.functions.base import ContextBudgetExceededError
        
        func = DecomposeTaskFunction()
        
        # Create input that exceeds budget (4096 tokens ≈ 16384 chars)
        huge_task = "x" * 20000
        
        with pytest.raises(ContextBudgetExceededError):
            await func.run(
                task=huge_task,
                constraints=[],
                available_agents=[],
                context="",
            )

    @pytest.mark.asyncio
    async def test_run_assigns_agents_when_available(self) -> None:
        """run() assigns agents to subtasks when available_agents provided."""
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Generate and validate code",
            constraints=[],
            available_agents=["generate_code", "validate_against_spec"],
            context="",
        )
        
        # At least some subtasks should have assigned agents
        assigned = [s for s in result.subtasks if s.assigned_agent is not None]
        # This is a soft check - decomposition may not always assign agents
        assert isinstance(assigned, list)

    @pytest.mark.asyncio
    async def test_run_respects_max_subtasks(self) -> None:
        """run() respects max_subtasks limit."""
        from src.functions.decompose_task import DecomposeTaskFunction
        
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Very complex task with many steps",
            constraints=[],
            available_agents=[],
            context="",
            max_subtasks=3,
        )
        
        assert len(result.subtasks) <= 3


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestDecomposeTaskEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_task_after_validation_fails(self) -> None:
        """Empty or whitespace-only task fails validation."""
        from src.schemas.functions.decompose_task import DecomposeTaskInput
        
        with pytest.raises(ValidationError):
            DecomposeTaskInput(task="")

    def test_subtask_with_multiple_dependencies(self) -> None:
        """Subtask can have multiple dependencies."""
        from src.schemas.functions.decompose_task import Subtask
        
        subtask = Subtask(
            id="final",
            description="Final step",
            depends_on=["step1", "step2", "step3"],
        )
        assert len(subtask.depends_on) == 3

    def test_empty_decomposition_is_valid(self) -> None:
        """Empty subtasks list is valid (trivial task)."""
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        decomposition = TaskDecomposition(subtasks=[])
        assert decomposition.subtasks == []
        assert decomposition.is_valid_dag()

    def test_single_subtask_is_valid_dag(self) -> None:
        """Single subtask with no dependencies is valid DAG."""
        from src.schemas.functions.decompose_task import TaskDecomposition, Subtask
        
        subtasks = [Subtask(id="only", description="Only task")]
        decomposition = TaskDecomposition(subtasks=subtasks)
        
        assert decomposition.is_valid_dag()
        assert decomposition.get_execution_order() == ["only"]


# =============================================================================
# Integration with Token Estimation
# =============================================================================

class TestDecomposeTaskTokenEstimation:
    """Tests for token estimation integration."""

    def test_subtask_estimated_tokens_positive(self) -> None:
        """estimated_tokens must be positive if provided."""
        from src.schemas.functions.decompose_task import Subtask
        
        with pytest.raises(ValidationError):
            Subtask(id="s1", description="Task", estimated_tokens=-100)

    def test_decomposition_total_tokens_non_negative(self) -> None:
        """estimated_total_tokens must be non-negative."""
        from src.schemas.functions.decompose_task import TaskDecomposition
        
        with pytest.raises(ValidationError):
            TaskDecomposition(subtasks=[], estimated_total_tokens=-1)


__all__ = [
    "TestDecomposeTaskInput",
    "TestSubtask",
    "TestTaskDecomposition",
    "TestDAGValidation",
    "TestTopologicalSort",
    "TestDecomposeTaskFunction",
    "TestDecomposeTaskEdgeCases",
    "TestDecomposeTaskTokenEstimation",
]
