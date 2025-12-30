"""Fake Agent Function for Testing.

Implements AgentFunction ABC for use in unit tests.
Follows the FakeClient pattern for predictable test behavior.

Pattern: Test Double (Fake)
Anti-Pattern Avoided: Mocking too deep into implementation details
Reference: tests/conftest.py FakeClient pattern
"""

from typing import Any
from pydantic import BaseModel

from src.functions.base import AgentFunction


class FakeOutput(BaseModel):
    """Fake output model for testing."""
    
    result: str = "fake_result"
    metadata: dict[str, Any] | None = None


class FakeAgentFunction(AgentFunction):
    """Fake implementation of AgentFunction for testing.
    
    Use this in unit tests instead of mocking the AgentFunction ABC.
    Configure the expected output in the constructor.
    
    Example:
        ```python
        def test_pipeline_executes_function():
            fake_func = FakeAgentFunction(
                name="extract_structure",
                output=FakeOutput(result="extracted")
            )
            result = await fake_func.run(content="test")
            assert result.result == "extracted"
        ```
    
    Pattern: Test Double (Fake) - provides canned answers
    Reference: conftest.py FakeClient pattern
    """
    
    def __init__(
        self,
        name: str = "fake_function",
        default_preset: str = "D4",
        output: BaseModel | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        """Initialize fake agent function.
        
        Args:
            name: Function name to simulate.
            default_preset: Default preset to use.
            output: Canned output to return from run().
            should_raise: Exception to raise from run() (for error testing).
        """
        self.name = name
        self.default_preset = default_preset
        self._output = output or FakeOutput()
        self._should_raise = should_raise
        self._call_count = 0
        self._call_args: list[dict[str, Any]] = []
    
    async def run(self, **kwargs: Any) -> BaseModel:
        """Return canned output or raise configured exception.
        
        Records call arguments for assertion in tests.
        """
        self._call_count += 1
        self._call_args.append(kwargs)
        
        if self._should_raise:
            raise self._should_raise
        
        return self._output
    
    @property
    def call_count(self) -> int:
        """Return number of times run() was called."""
        return self._call_count
    
    @property
    def call_args(self) -> list[dict[str, Any]]:
        """Return list of kwargs passed to run() calls."""
        return self._call_args
    
    def reset(self) -> None:
        """Reset call tracking for reuse across tests."""
        self._call_count = 0
        self._call_args = []
