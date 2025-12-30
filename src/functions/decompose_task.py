"""DecomposeTask Agent Function.

WBS-AGT12: decompose_task Function

This module implements the DecomposeTaskFunction which breaks complex
tasks into subtasks forming a valid DAG (Directed Acyclic Graph).

Acceptance Criteria:
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

REFACTOR Phase:
- Using shared utilities from src/functions/utils/token_utils.py (S1192)
- Context budget defined in src/core/constants.py
"""

import re
from typing import Any

from src.functions.base import AgentFunction
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.functions.decompose_task import (
    DecomposeTaskInput,
    Subtask,
    TaskDecomposition,
)


class DecomposeTaskFunction(AgentFunction):
    """Agent function to decompose complex tasks into subtasks.

    Breaks down high-level objectives into executable subtasks with
    dependencies, forming a valid DAG for pipeline execution.

    Context Budget (AC-12.3):
        - Input: 4096 tokens
        - Output: 2048 tokens

    Default Preset (AC-12.4): S2 (DeepSeek for chain-of-thought)

    Attributes:
        name: Function identifier 'decompose_task'
        default_preset: Default to S2 for chain-of-thought reasoning

    Example:
        ```python
        func = DecomposeTaskFunction()
        result = await func.run(
            task="Build user authentication with OAuth2",
            constraints=["Must support Google OAuth", "Use FastAPI"],
            available_agents=["generate_code", "analyze_artifact"],
            context="Python web application",
        )
        for subtask in result.subtasks:
            print(f"{subtask.id}: {subtask.description}")
        ```

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → decompose_task
    """

    name: str = "decompose_task"
    default_preset: str = "S2"  # DeepSeek for chain-of-thought

    async def run(
        self,
        *,
        task: str,
        constraints: list[str] | None = None,
        available_agents: list[str] | None = None,
        context: str = "",
        max_subtasks: int = 10,
        **kwargs: Any,
    ) -> TaskDecomposition:
        """Decompose a complex task into subtasks.

        AC-12.1: Breaks complex task into subtasks.

        Args:
            task: High-level objective to decompose
            constraints: Time, scope, dependency constraints
            available_agents: List of available agent function names
            context: Domain/project context for decomposition
            max_subtasks: Maximum number of subtasks to generate
            **kwargs: Additional arguments for extensibility

        Returns:
            TaskDecomposition with subtasks, dependencies, and execution order.

        Raises:
            ContextBudgetExceededError: If input exceeds 4096 token budget.
        """
        constraints = constraints or []
        available_agents = available_agents or []

        # Validate input schema
        input_data = DecomposeTaskInput(
            task=task,
            constraints=constraints,
            available_agents=available_agents,
            context=context,
            max_subtasks=max_subtasks,
        )

        # Estimate input tokens and enforce budget
        input_text = self._build_input_text(input_data)
        input_tokens = estimate_tokens(input_text)
        self.enforce_budget(input_tokens)

        # Decompose the task into subtasks
        subtasks = self._decompose(input_data)

        # Limit to max_subtasks
        subtasks = subtasks[:max_subtasks]

        # Calculate estimated total tokens
        estimated_total = sum(
            s.estimated_tokens or 0 for s in subtasks
        )

        # Build and validate the decomposition (DAG validation happens in model)
        decomposition = TaskDecomposition(
            subtasks=subtasks,
            estimated_total_tokens=estimated_total,
            original_task=task,
        )

        return decomposition

    def _build_input_text(self, input_data: DecomposeTaskInput) -> str:
        """Build combined input text for token estimation.

        Args:
            input_data: Validated input data.

        Returns:
            Combined input text string.
        """
        parts = [input_data.task]

        if input_data.constraints:
            parts.append(" ".join(input_data.constraints))

        if input_data.available_agents:
            parts.append(" ".join(input_data.available_agents))

        if input_data.context:
            parts.append(input_data.context)

        return " ".join(parts)

    def _decompose(self, input_data: DecomposeTaskInput) -> list[Subtask]:
        """Decompose the task into subtasks.

        This is a rule-based decomposition for local execution.
        In production, this would call the inference-service with
        the S2 preset for chain-of-thought reasoning.

        Args:
            input_data: Validated input data.

        Returns:
            List of Subtask objects forming a valid DAG.
        """
        task_lower = input_data.task.lower()
        subtasks: list[Subtask] = []

        # Analyze task for common patterns
        any(
            pattern in task_lower
            for pattern in [
                " and ", " with ", " then ", " after ",
                "implement", "build", "create", "develop",
            ]
        )

        # Extract key actions from task
        actions = self._extract_actions(input_data.task)

        if not actions:
            # Single atomic task - return as-is
            subtask = Subtask(
                id="task_1",
                description=input_data.task,
                depends_on=[],
                assigned_agent=self._assign_agent(
                    input_data.task, input_data.available_agents
                ),
                estimated_tokens=self._estimate_subtask_tokens(input_data.task),
            )
            return [subtask]

        # Build subtasks from actions
        prev_id: str | None = None
        for i, action in enumerate(actions, 1):
            subtask_id = f"subtask_{i}"

            # Determine dependencies based on action type
            depends_on: list[str] = []
            if self._requires_prior_step(action) and prev_id:
                depends_on = [prev_id]

            subtask = Subtask(
                id=subtask_id,
                description=action,
                depends_on=depends_on,
                assigned_agent=self._assign_agent(
                    action, input_data.available_agents
                ),
                estimated_tokens=self._estimate_subtask_tokens(action),
            )
            subtasks.append(subtask)
            prev_id = subtask_id

        return subtasks

    def _extract_actions(self, task: str) -> list[str]:
        """Extract discrete actions from a task description.

        Args:
            task: Task description string.

        Returns:
            List of action descriptions.
        """
        actions: list[str] = []

        # Split on common conjunctions
        parts = re.split(r'\s+(?:and|then|,)\s+', task, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            if part and len(part) > 5:  # Skip very short fragments
                actions.append(part)

        # If no split occurred but task is complex, create logical steps
        if len(actions) <= 1 and len(task) > 50:
            task_lower = task.lower()

            # Check for implementation keywords
            if any(kw in task_lower for kw in ["implement", "build", "create", "develop"]):
                actions = [
                    f"Design {self._extract_subject(task)}",
                    f"Implement {self._extract_subject(task)}",
                    f"Test {self._extract_subject(task)}",
                ]

        return actions if len(actions) > 1 else []

    def _extract_subject(self, task: str) -> str:
        """Extract the main subject/noun from a task.

        Args:
            task: Task description.

        Returns:
            Extracted subject or truncated task.
        """
        # Remove common action verbs
        subject = re.sub(
            r'^(implement|build|create|develop|write|design)\s+',
            '',
            task,
            flags=re.IGNORECASE,
        )

        # Take first 50 chars if still long
        if len(subject) > 50:
            subject = subject[:50] + "..."

        return subject

    def _requires_prior_step(self, action: str) -> bool:
        """Check if an action requires prior steps to complete.

        Args:
            action: Action description.

        Returns:
            True if this action depends on prior work.
        """
        action_lower = action.lower()

        # Test/validate actions typically need prior implementation
        dependent_keywords = [
            "test", "validate", "verify", "check",
            "review", "refactor", "optimize", "deploy",
        ]

        return any(kw in action_lower for kw in dependent_keywords)

    def _assign_agent(
        self, action: str, available_agents: list[str]
    ) -> str | None:
        """Assign an appropriate agent to an action.

        Args:
            action: Action description.
            available_agents: List of available agent names.

        Returns:
            Agent name if a match found, None otherwise.
        """
        if not available_agents:
            return None

        action_lower = action.lower()

        # Simple keyword matching to agents
        agent_keywords: dict[str, list[str]] = {
            "generate_code": ["implement", "create", "write", "build", "code", "develop"],
            "analyze_artifact": ["analyze", "review", "examine", "inspect", "assess"],
            "validate_against_spec": ["validate", "verify", "check", "test", "confirm"],
            "extract_structure": ["extract", "parse", "structure", "outline"],
            "summarize_content": ["summarize", "describe", "explain", "document"],
            "cross_reference": ["reference", "find", "search", "lookup", "query"],
            "synthesize_outputs": ["combine", "merge", "synthesize", "aggregate"],
        }

        for agent, keywords in agent_keywords.items():
            if agent in available_agents and any(kw in action_lower for kw in keywords):
                return agent

        # Default to first available if no match
        return None

    def _estimate_subtask_tokens(self, description: str) -> int:
        """Estimate token budget for a subtask.

        Args:
            description: Subtask description.

        Returns:
            Estimated token count (minimum 256).
        """
        base_tokens = estimate_tokens(description)

        # Multiply by complexity factor (2-4x for generation tasks)
        complexity = 3 if len(description) > 50 else 2

        return max(256, base_tokens * complexity)


__all__ = [
    "DecomposeTaskFunction",
]
