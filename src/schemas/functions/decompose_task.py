"""Schemas for decompose_task function.

WBS-AGT12: decompose_task Function

This module defines the input/output schemas for the decompose_task
function, which breaks complex tasks into subtasks forming a valid DAG.

Acceptance Criteria:
- AC-12.1: Breaks complex task into subtasks
- AC-12.2: Returns TaskDecomposition with subtasks, dependencies
- AC-12.5: Subtasks form valid DAG (no cycles)

Exit Criteria:
- Each Subtask has id, description, depends_on list
- Cyclic dependencies raise ValidationError
- Topological sort produces valid execution order

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 7
"""

from collections import deque

from pydantic import BaseModel, Field, field_validator, model_validator


class DecomposeTaskInput(BaseModel):
    """Input schema for decompose_task function.

    AC-12.1: Breaks complex task into subtasks.

    Per architecture:
    - task: High-level objective
    - constraints: Time, scope, dependencies
    - available_agents: Which agent functions exist
    - context: Domain/project context

    Attributes:
        task: High-level objective to decompose
        constraints: Time, scope, dependency constraints
        available_agents: List of available agent function names
        context: Domain/project context for decomposition
        max_subtasks: Maximum number of subtasks to generate
    """

    task: str = Field(
        ...,
        description="High-level objective to decompose",
        min_length=1,
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Time, scope, dependency constraints",
    )
    available_agents: list[str] = Field(
        default_factory=list,
        description="List of available agent function names",
    )
    context: str = Field(
        default="",
        description="Domain/project context for decomposition",
    )
    max_subtasks: int = Field(
        default=10,
        gt=0,
        description="Maximum number of subtasks to generate",
    )

    @field_validator("task")
    @classmethod
    def validate_task_not_whitespace(cls, v: str) -> str:
        """Validate task is not just whitespace."""
        if not v.strip():
            raise ValueError("task cannot be empty or whitespace only")
        return v


class Subtask(BaseModel):
    """A single subtask in a task decomposition.

    Exit Criteria: Each Subtask has id, description, depends_on list.

    Attributes:
        id: Unique identifier for this subtask
        description: Human-readable description of what this subtask does
        depends_on: List of subtask IDs that must complete before this one
        assigned_agent: Optional agent function assigned to this subtask
        estimated_tokens: Optional estimated token budget for this subtask
    """

    id: str = Field(
        ...,
        description="Unique identifier for this subtask",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this subtask does",
        min_length=1,
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of subtask IDs that must complete before this one",
    )
    assigned_agent: str | None = Field(
        default=None,
        description="Optional agent function assigned to this subtask",
    )
    estimated_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Optional estimated token budget for this subtask",
    )


class TaskDecomposition(BaseModel):
    """Result of decomposing a task into subtasks.

    AC-12.2: Returns TaskDecomposition with subtasks, dependencies.
    AC-12.5: Subtasks form valid DAG (no cycles).

    Exit Criteria:
    - Cyclic dependencies raise ValidationError
    - Topological sort produces valid execution order

    Attributes:
        subtasks: List of subtasks that make up the decomposition
        execution_order: Topologically sorted list of subtask IDs
        estimated_total_tokens: Total estimated token budget
        original_task: The original task that was decomposed
    """

    subtasks: list[Subtask] = Field(
        default_factory=list,
        description="List of subtasks that make up the decomposition",
    )
    execution_order: list[str] = Field(
        default_factory=list,
        description="Topologically sorted list of subtask IDs",
    )
    estimated_total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total estimated token budget",
    )
    original_task: str = Field(
        default="",
        description="The original task that was decomposed",
    )

    @model_validator(mode="after")
    def validate_dag(self) -> "TaskDecomposition":
        """Validate that subtasks form a valid DAG with no cycles.

        AC-12.5: Subtasks form valid DAG (no cycles).
        Exit Criteria: Cyclic dependencies raise ValidationError.
        """
        if not self.subtasks:
            return self

        # Build set of valid subtask IDs
        subtask_ids = {s.id for s in self.subtasks}

        # Check for duplicate IDs
        if len(subtask_ids) != len(self.subtasks):
            seen = set()
            duplicates = []
            for s in self.subtasks:
                if s.id in seen:
                    duplicates.append(s.id)
                seen.add(s.id)
            raise ValueError(f"Duplicate subtask IDs found: {duplicates}")

        # Check for dependencies on non-existent subtasks
        for subtask in self.subtasks:
            for dep in subtask.depends_on:
                if dep not in subtask_ids:
                    raise ValueError(
                        f"Subtask '{subtask.id}' depends on unknown subtask '{dep}'"
                    )

        # Check for cycles using DFS
        if self._has_cycle():
            raise ValueError("Cyclic dependency detected in subtasks")

        # Auto-populate execution_order if not provided
        if not self.execution_order:
            self.execution_order = self._topological_sort()

        return self

    def _has_cycle(self) -> bool:
        """Detect if the dependency graph has a cycle.

        Uses DFS with color marking:
        - WHITE (0): Not visited
        - GRAY (1): Currently being visited (in stack)
        - BLACK (2): Fully processed

        Returns:
            True if a cycle exists, False otherwise.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {s.id: WHITE for s in self.subtasks}

        # Build adjacency list (subtask -> its dependents)
        # For cycle detection, we check if following dependencies leads back
        adj: dict[str, list[str]] = {s.id: [] for s in self.subtasks}
        for subtask in self.subtasks:
            for dep in subtask.depends_on:
                # Edge from dependency to this subtask (dep must finish before subtask)
                adj[dep].append(subtask.id)

        def dfs(node: str) -> bool:
            """DFS visit, returns True if cycle found."""
            color[node] = GRAY
            for neighbor in adj[node]:
                if color[neighbor] == GRAY:
                    # Back edge found - cycle!
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        # Check all nodes (handles disconnected components)
        return any(color[subtask.id] == WHITE and dfs(subtask.id) for subtask in self.subtasks)

    def _topological_sort(self) -> list[str]:
        """Perform topological sort on subtasks.

        Uses Kahn's algorithm (BFS-based).

        Returns:
            List of subtask IDs in valid execution order.

        Exit Criteria: Topological sort produces valid execution order.
        """
        if not self.subtasks:
            return []

        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {s.id: 0 for s in self.subtasks}
        adj: dict[str, list[str]] = {s.id: [] for s in self.subtasks}

        for subtask in self.subtasks:
            for dep in subtask.depends_on:
                adj[dep].append(subtask.id)
                in_degree[subtask.id] += 1

        # Initialize queue with nodes having in-degree 0
        queue: deque[str] = deque()
        for subtask_id, degree in in_degree.items():
            if degree == 0:
                queue.append(subtask_id)

        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def is_valid_dag(self) -> bool:
        """Check if subtasks form a valid DAG.

        Returns:
            True if valid DAG, False if cycle exists.
        """
        return not self._has_cycle()

    def get_execution_order(self) -> list[str]:
        """Get the topologically sorted execution order.

        Returns:
            List of subtask IDs in valid execution order.

        Exit Criteria: Topological sort produces valid execution order.
        """
        if self.execution_order:
            return self.execution_order
        return self._topological_sort()


__all__ = [
    "DecomposeTaskInput",
    "Subtask",
    "TaskDecomposition",
]
