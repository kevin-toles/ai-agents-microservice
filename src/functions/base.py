"""Agent Function Base Class.

Defines the abstract base class for all agent functions.
Agent functions are stateless executors over cached artifacts.

Pattern: Abstract Base Class (ABC) with Protocol duck typing
Anti-Pattern Avoided: ABC signatures without **kwargs flexibility (AP-ABC)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function Base Class

REFACTOR Phase (WBS-AGT5):
- Extracted CONTEXT_BUDGET_DEFAULTS to src/core/constants.py (S1192)
- Single source of truth for all context budgets
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from src.core.constants import (
    CONTEXT_BUDGET_DEFAULTS,
    DEFAULT_CONTEXT_BUDGET,
)


# ============================================================================
# Type Variables
# ============================================================================

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


# ============================================================================
# Protocol for Type Checking
# ============================================================================

@runtime_checkable
class AgentFunctionProtocol(Protocol):
    """Protocol for agent function type hints.

    Enables duck typing for agent functions without requiring inheritance.
    Use this for type annotations in client code.
    """

    @property
    def name(self) -> str:
        """Return the function name."""
        ...

    @property
    def default_preset(self) -> str:
        """Return the default preset for this function."""
        ...

    async def run(self, **kwargs: Any) -> BaseModel:
        """Execute the agent function."""
        ...

    def get_context_budget(self) -> dict[str, int]:
        """Return the context budget for this function."""
        ...


# ============================================================================
# Abstract Base Class
# ============================================================================

class AgentFunction(ABC):
    """Abstract base class for agent functions.

    Agent functions are stateless executors that:
    - Read from caches (temp:, user:, app: prefixes)
    - Execute a specific transformation
    - Write results back to cache
    - Produce typed Pydantic outputs

    Design Philosophy:
        "Agents do not remember, do not chat, do not accumulate context.
        They read from caches and write new state back."

    Anti-Pattern Compliance:
        - Uses **kwargs in abstract method signature (AP-ABC)
        - No mutable default arguments in dataclasses (AP-1.5)
        - Consistent return types (S3516)

    Example:
        ```python
        class ExtractStructureFunction(AgentFunction):
            name = "extract_structure"
            default_preset = "S1"

            async def run(self, *, content: str, artifact_type: str, **kwargs) -> StructuredOutput:
                # Implementation
                pass
        ```

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions
    """

    # Class attributes to be overridden by subclasses
    name: str = ""
    default_preset: str = "D4"  # Standard preset

    @abstractmethod
    async def run(self, **kwargs: Any) -> BaseModel:
        """Execute the agent function.

        Args:
            **kwargs: Function-specific arguments. Using **kwargs allows
                subclasses to define their own typed signatures while
                maintaining ABC compatibility (AP-ABC pattern).

        Returns:
            Typed Pydantic model output specific to the function.

        Raises:
            ValidationError: If input validation fails.
            ContextBudgetExceededError: If input exceeds budget.

        Note:
            Subclasses should define specific keyword arguments:

            async def run(self, *, content: str, **kwargs) -> OutputModel:
                ...
        """
        ...

    def get_context_budget(self) -> dict[str, int]:
        """Get the context budget for this function.

        Returns:
            Dict with 'input' and 'output' token limits.

        Example:
            >>> func = ExtractStructureFunction()
            >>> func.get_context_budget()
            {'input': 16384, 'output': 2048}
        """
        return CONTEXT_BUDGET_DEFAULTS.get(self.name, DEFAULT_CONTEXT_BUDGET)

    def enforce_budget(self, input_tokens: int) -> None:
        """Enforce context budget for input.

        Args:
            input_tokens: Number of tokens in input.

        Raises:
            ContextBudgetExceededError: If input exceeds budget.
        """
        budget = self.get_context_budget()
        if input_tokens > budget["input"]:
            raise ContextBudgetExceededError(
                function_name=self.name,
                actual=input_tokens,
                limit=budget["input"],
            )

    def select_preset(self, quality_hint: str | None = None) -> str:
        """Select the appropriate preset for execution.

        Args:
            quality_hint: Optional hint ('light', 'standard', 'high_quality').

        Returns:
            Preset identifier (e.g., 'S1', 'D4', 'D10').

        Example:
            >>> func = SummarizeContentFunction()
            >>> func.select_preset('high_quality')
            'D10'
        """
        preset_map = {
            "light": "S1",
            "standard": "D4",
            "high_quality": "D10",
        }
        return preset_map.get(quality_hint or "", self.default_preset)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, preset={self.default_preset!r})"


# ============================================================================
# Exceptions
# ============================================================================

class ContextBudgetExceededError(Exception):
    """Raised when input exceeds the function's context budget.

    Attributes:
        function_name: Name of the agent function.
        actual: Actual token count.
        limit: Maximum allowed tokens.
    """

    def __init__(self, function_name: str, actual: int, limit: int) -> None:
        self.function_name = function_name
        self.actual = actual
        self.limit = limit
        super().__init__(
            f"Context budget exceeded for {function_name}: "
            f"{actual} tokens > {limit} limit"
        )
