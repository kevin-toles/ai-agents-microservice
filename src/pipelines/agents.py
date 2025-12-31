"""Agent Patterns for Pipeline Composition.

WBS-AGT14: Pipeline Orchestrator - Agent Patterns

Implements:
- SequentialAgent: Executes functions in order
- ParallelAgent: Executes functions concurrently with asyncio.gather
- LoopAgent: Repeats function until condition is met

Acceptance Criteria:
- AC-14.2: Supports SequentialAgent, ParallelAgent, LoopAgent patterns

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.functions.base import AgentFunction


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Exceptions
# =============================================================================

class MaxIterationsExceededError(Exception):
    """Raised when LoopAgent exceeds max iterations."""

    def __init__(self, iterations: int, message: str | None = None) -> None:
        """Initialize the exception."""
        self.iterations = iterations
        super().__init__(message or f"Maximum iterations ({iterations}) exceeded")


# =============================================================================
# Base Agent Protocol
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for agent patterns."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent name."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the agent pattern."""
        ...


# =============================================================================
# Sequential Agent
# =============================================================================

@dataclass
class SequentialAgent(BaseAgent):
    """Executes functions sequentially, passing output to next.

    Each function's output is passed as input to the next function.
    Execution stops on error.

    Example:
        agent = SequentialAgent(
            name="process_chain",
            functions=[parse_func, transform_func, validate_func],
        )
        result = await agent.execute(raw_data="...")
    """

    _name: str
    functions: list[AgentFunction | BaseAgent]

    def __init__(
        self,
        name: str,
        functions: list[AgentFunction | BaseAgent],
    ) -> None:
        """Initialize SequentialAgent."""
        self._name = name
        self.functions = functions

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    async def execute(self, **kwargs: Any) -> Any:
        """Execute functions sequentially.

        Args:
            **kwargs: Initial inputs for the first function

        Returns:
            Output from the last function in the sequence
        """
        current_inputs = kwargs
        result: Any = None

        for func in self.functions:
            # Execute function or nested agent
            if isinstance(func, BaseAgent):
                result = await func.execute(**current_inputs)
            else:
                result = await func.run(**current_inputs)

            # Pass output to next function
            if hasattr(result, "model_dump"):
                current_inputs = result.model_dump()
            elif isinstance(result, dict):
                current_inputs = result
            elif hasattr(result, "__dict__"):
                current_inputs = vars(result)
            else:
                # For list results from ParallelAgent, wrap them
                current_inputs = {"results": result}

        return result


# =============================================================================
# Parallel Agent
# =============================================================================

@dataclass
class ParallelAgent(BaseAgent):
    """Executes functions concurrently using asyncio.gather.

    All functions receive the same inputs and execute in parallel.
    Returns list of all results.

    Example:
        agent = ParallelAgent(
            name="parallel_search",
            functions=[search_web, search_docs, search_code],
        )
        results = await agent.execute(query="...")
    """

    _name: str
    functions: list[AgentFunction]
    return_exceptions: bool = False

    def __init__(
        self,
        name: str,
        functions: list[AgentFunction],
        return_exceptions: bool = False,
    ) -> None:
        """Initialize ParallelAgent.

        Args:
            name: Agent name
            functions: Functions to execute in parallel
            return_exceptions: If True, exceptions are returned in results
                              instead of being raised
        """
        self._name = name
        self.functions = functions
        self.return_exceptions = return_exceptions

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    async def execute(self, **kwargs: Any) -> list[Any]:
        """Execute all functions in parallel.

        Args:
            **kwargs: Inputs passed to all functions

        Returns:
            List of outputs from all functions
        """
        tasks = [func.run(**kwargs) for func in self.functions]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=self.return_exceptions,
        )

        return list(results)


# =============================================================================
# Loop Agent
# =============================================================================

@dataclass
class LoopAgent(BaseAgent, Generic[T]):
    """Repeats function until condition returns False.

    Output from each iteration is passed as input to the next.
    Respects max_iterations to prevent infinite loops.

    Example:
        agent = LoopAgent(
            name="refine_until_complete",
            function=refine_func,
            condition=lambda r: r.needs_refinement,
            max_iterations=10,
        )
        result = await agent.execute(draft="...")
    """

    _name: str
    function: AgentFunction
    condition: Callable[[Any], bool]
    max_iterations: int
    keep_history: bool = False
    history: list[Any] = field(default_factory=list)

    def __init__(
        self,
        name: str,
        function: AgentFunction,
        condition: Callable[[Any], bool],
        max_iterations: int = 10,
        keep_history: bool = False,
    ) -> None:
        """Initialize LoopAgent.

        Args:
            name: Agent name
            function: Function to execute repeatedly
            condition: Callable that returns True to continue, False to stop
            max_iterations: Maximum iterations before raising error
            keep_history: If True, store all iteration results
        """
        self._name = name
        self.function = function
        self.condition = condition
        self.max_iterations = max_iterations
        self.keep_history = keep_history
        self.history = []

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    async def execute(self, **kwargs: Any) -> Any:
        """Execute function repeatedly until condition is False.

        Args:
            **kwargs: Initial inputs

        Returns:
            Final output when condition returns False

        Raises:
            MaxIterationsExceededError: If max iterations reached
        """
        self.history = []
        current_inputs = kwargs
        iteration = 0
        result: Any = None

        while iteration < self.max_iterations:
            # Execute function
            result = await self.function.run(**current_inputs)
            iteration += 1

            # Track history if enabled
            if self.keep_history:
                self.history.append(result)

            # Check condition
            if not self.condition(result):
                # Condition is False, we're done
                return result

            # Prepare inputs for next iteration
            if hasattr(result, "model_dump"):
                current_inputs = result.model_dump()
            elif isinstance(result, dict):
                current_inputs = result
            elif hasattr(result, "__dict__"):
                current_inputs = vars(result)
            else:
                current_inputs = {"value": result}

        # Max iterations exceeded
        raise MaxIterationsExceededError(self.max_iterations)


__all__ = [
    "BaseAgent",
    "LoopAgent",
    "MaxIterationsExceededError",
    "ParallelAgent",
    "SequentialAgent",
]
