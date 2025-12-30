"""Tests for AgentFunction base class.

TDD tests for WBS-AGT5: Agent Function Base Class.

Acceptance Criteria Coverage:
- AC-5.1: AgentFunction ABC with run() abstract method
- AC-5.2: ABC signature uses **kwargs for flexibility (AP-ABC)
- AC-5.3: Context budget enforcement per function
- AC-5.4: Preset selection mechanism
- AC-5.5: FakeAgentFunction for testing
- AC-5.6: Protocol duck typing for type hints

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions
"""

import pytest
from typing import Any
from pydantic import BaseModel


# =============================================================================
# AC-5.1: AgentFunction ABC with run() abstract method
# =============================================================================

class TestAgentFunctionABC:
    """Tests for AgentFunction abstract base class."""
    
    def test_agent_function_is_abc(self) -> None:
        """AgentFunction is an abstract base class."""
        from abc import ABC
        from src.functions.base import AgentFunction
        
        assert issubclass(AgentFunction, ABC)
    
    def test_agent_function_cannot_be_instantiated(self) -> None:
        """AgentFunction cannot be instantiated directly."""
        from src.functions.base import AgentFunction
        
        with pytest.raises(TypeError, match="Can't instantiate"):
            AgentFunction()  # type: ignore
    
    def test_agent_function_run_is_abstract(self) -> None:
        """AgentFunction.run() is an abstract method."""
        from src.functions.base import AgentFunction
        import inspect
        
        # Get the run method from the ABC
        run_method = getattr(AgentFunction, "run")
        
        # Verify it's marked as abstract
        assert getattr(run_method, "__isabstractmethod__", False)
    
    def test_subclass_must_implement_run(self) -> None:
        """Subclasses must implement run() method."""
        from src.functions.base import AgentFunction
        
        # Subclass without run() should not be instantiable
        class IncompleteFunction(AgentFunction):
            name = "incomplete"
        
        with pytest.raises(TypeError, match="Can't instantiate"):
            IncompleteFunction()  # type: ignore
    
    def test_subclass_with_run_can_be_instantiated(self) -> None:
        """Subclass with run() can be instantiated."""
        from src.functions.base import AgentFunction
        
        class CompleteFunction(AgentFunction):
            name = "complete"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = CompleteFunction()
        assert func.name == "complete"
    
    def test_agent_function_has_name_attribute(self) -> None:
        """AgentFunction has name class attribute."""
        from src.functions.base import AgentFunction
        
        assert hasattr(AgentFunction, "name")
    
    def test_agent_function_has_default_preset_attribute(self) -> None:
        """AgentFunction has default_preset class attribute."""
        from src.functions.base import AgentFunction
        
        assert hasattr(AgentFunction, "default_preset")
        assert AgentFunction.default_preset == "D4"  # Standard preset


# =============================================================================
# AC-5.2: ABC signature uses **kwargs for flexibility (AP-ABC)
# =============================================================================

class TestABCKwargsPattern:
    """Tests for **kwargs flexibility pattern."""
    
    def test_run_signature_uses_kwargs(self) -> None:
        """AgentFunction.run() accepts **kwargs."""
        from src.functions.base import AgentFunction
        import inspect
        
        sig = inspect.signature(AgentFunction.run)
        params = list(sig.parameters.values())
        
        # Should have self and **kwargs
        param_names = [p.name for p in params]
        assert "kwargs" in param_names
        
        # kwargs should be VAR_KEYWORD
        kwargs_param = sig.parameters.get("kwargs")
        assert kwargs_param is not None
        assert kwargs_param.kind == inspect.Parameter.VAR_KEYWORD
    
    def test_subclass_can_define_specific_params(self) -> None:
        """Subclasses can define specific typed parameters."""
        from src.functions.base import AgentFunction
        
        class SpecificFunction(AgentFunction):
            name = "specific"
            
            async def run(
                self,
                *,
                content: str,
                artifact_type: str = "code",
                **kwargs: Any,
            ) -> BaseModel:
                return BaseModel()
        
        func = SpecificFunction()
        # Should be instantiable and have specific params
        assert func.name == "specific"
    
    @pytest.mark.asyncio
    async def test_subclass_receives_kwargs(self) -> None:
        """Subclass run() receives kwargs correctly."""
        from src.functions.base import AgentFunction
        
        received_kwargs: dict[str, Any] = {}
        
        class TrackingOutput(BaseModel):
            success: bool = True
        
        class TrackingFunction(AgentFunction):
            name = "tracking"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                received_kwargs.update(kwargs)
                return TrackingOutput()
        
        func = TrackingFunction()
        await func.run(content="test", extra="data")
        
        assert received_kwargs["content"] == "test"
        assert received_kwargs["extra"] == "data"


# =============================================================================
# AC-5.3: Context budget enforcement per function
# =============================================================================

class TestContextBudget:
    """Tests for context budget enforcement."""
    
    def test_context_budget_defaults_exists(self) -> None:
        """CONTEXT_BUDGET_DEFAULTS constant exists."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        assert isinstance(CONTEXT_BUDGET_DEFAULTS, dict)
    
    def test_context_budget_has_all_functions(self) -> None:
        """CONTEXT_BUDGET_DEFAULTS has entries for all agent functions."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        expected_functions = [
            "extract_structure",
            "summarize_content",
            "generate_code",
            "analyze_artifact",
            "validate_against_spec",
            "synthesize_outputs",
            "decompose_task",
            "cross_reference",
        ]
        
        for func_name in expected_functions:
            assert func_name in CONTEXT_BUDGET_DEFAULTS, f"Missing budget for {func_name}"
    
    def test_context_budget_has_input_output(self) -> None:
        """Each budget entry has input and output limits."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        for func_name, budget in CONTEXT_BUDGET_DEFAULTS.items():
            assert "input" in budget, f"Missing 'input' for {func_name}"
            assert "output" in budget, f"Missing 'output' for {func_name}"
            assert isinstance(budget["input"], int)
            assert isinstance(budget["output"], int)
    
    def test_context_budget_values_match_architecture(self) -> None:
        """Budget values match AGENT_FUNCTIONS_ARCHITECTURE.md."""
        from src.functions.base import CONTEXT_BUDGET_DEFAULTS
        
        # Values from architecture doc
        expected = {
            "extract_structure": {"input": 16384, "output": 2048},
            "summarize_content": {"input": 8192, "output": 4096},
            "generate_code": {"input": 4096, "output": 8192},
            "analyze_artifact": {"input": 16384, "output": 2048},
            "validate_against_spec": {"input": 4096, "output": 1024},
            "synthesize_outputs": {"input": 8192, "output": 4096},
            "decompose_task": {"input": 4096, "output": 2048},
            "cross_reference": {"input": 2048, "output": 4096},
        }
        
        for func_name, expected_budget in expected.items():
            actual = CONTEXT_BUDGET_DEFAULTS[func_name]
            assert actual == expected_budget, f"Mismatch for {func_name}"
    
    def test_get_context_budget_returns_budget(self) -> None:
        """AgentFunction.get_context_budget() returns correct budget."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "extract_structure"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        budget = func.get_context_budget()
        
        assert budget == {"input": 16384, "output": 2048}
    
    def test_get_context_budget_unknown_function(self) -> None:
        """Unknown function gets default budget."""
        from src.functions.base import AgentFunction, DEFAULT_CONTEXT_BUDGET
        
        class UnknownFunction(AgentFunction):
            name = "unknown_function"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = UnknownFunction()
        budget = func.get_context_budget()
        
        assert budget == DEFAULT_CONTEXT_BUDGET
    
    def test_enforce_budget_passes_under_limit(self) -> None:
        """enforce_budget() passes when under limit."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "extract_structure"  # 16384 input limit
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        # Should not raise
        func.enforce_budget(input_tokens=10000)
    
    def test_enforce_budget_raises_over_limit(self) -> None:
        """enforce_budget() raises ContextBudgetExceededError over limit."""
        from src.functions.base import AgentFunction, ContextBudgetExceededError
        
        class TestFunction(AgentFunction):
            name = "extract_structure"  # 16384 input limit
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        
        with pytest.raises(ContextBudgetExceededError) as exc_info:
            func.enforce_budget(input_tokens=20000)
        
        assert exc_info.value.function_name == "extract_structure"
        assert exc_info.value.actual == 20000
        assert exc_info.value.limit == 16384
    
    def test_context_budget_exceeded_error_message(self) -> None:
        """ContextBudgetExceededError has informative message."""
        from src.functions.base import ContextBudgetExceededError
        
        error = ContextBudgetExceededError(
            function_name="test_func",
            actual=5000,
            limit=4096,
        )
        
        assert "test_func" in str(error)
        assert "5000" in str(error)
        assert "4096" in str(error)


# =============================================================================
# AC-5.4: Preset selection mechanism
# =============================================================================

class TestPresetSelection:
    """Tests for preset selection mechanism."""
    
    def test_select_preset_default(self) -> None:
        """select_preset() returns default_preset when no hint."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        assert func.select_preset() == "S1"
    
    def test_select_preset_light(self) -> None:
        """select_preset('light') returns S1."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test"
            default_preset = "D4"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        assert func.select_preset("light") == "S1"
    
    def test_select_preset_standard(self) -> None:
        """select_preset('standard') returns D4."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        assert func.select_preset("standard") == "D4"
    
    def test_select_preset_high_quality(self) -> None:
        """select_preset('high_quality') returns D10."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        assert func.select_preset("high_quality") == "D10"
    
    def test_select_preset_unknown_hint(self) -> None:
        """Unknown hint returns default_preset."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test"
            default_preset = "S3"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        assert func.select_preset("unknown") == "S3"


# =============================================================================
# AC-5.5: FakeAgentFunction for testing
# =============================================================================

class TestFakeAgentFunction:
    """Tests for FakeAgentFunction test double."""
    
    def test_fake_agent_function_exists(self) -> None:
        """FakeAgentFunction can be imported."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        assert FakeAgentFunction is not None
    
    def test_fake_agent_function_instantiable(self) -> None:
        """FakeAgentFunction can be instantiated."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        fake = FakeAgentFunction()
        assert fake.name == "fake_function"
    
    def test_fake_agent_function_custom_name(self) -> None:
        """FakeAgentFunction accepts custom name."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        fake = FakeAgentFunction(name="extract_structure")
        assert fake.name == "extract_structure"
    
    @pytest.mark.asyncio
    async def test_fake_agent_function_returns_output(self) -> None:
        """FakeAgentFunction.run() returns configured output."""
        from tests.unit.functions.fake_agent import FakeAgentFunction, FakeOutput
        
        expected_output = FakeOutput(result="test_result")
        fake = FakeAgentFunction(output=expected_output)
        
        result = await fake.run(content="test")
        
        assert result.result == "test_result"
    
    @pytest.mark.asyncio
    async def test_fake_agent_function_raises_exception(self) -> None:
        """FakeAgentFunction.run() can raise configured exception."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        fake = FakeAgentFunction(should_raise=ValueError("test error"))
        
        with pytest.raises(ValueError, match="test error"):
            await fake.run()
    
    @pytest.mark.asyncio
    async def test_fake_agent_function_tracks_calls(self) -> None:
        """FakeAgentFunction tracks call count and arguments."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        fake = FakeAgentFunction()
        
        assert fake.call_count == 0
        
        await fake.run(content="first")
        await fake.run(content="second")
        
        assert fake.call_count == 2
        assert fake.call_args[0] == {"content": "first"}
        assert fake.call_args[1] == {"content": "second"}
    
    def test_fake_agent_function_reset(self) -> None:
        """FakeAgentFunction.reset() clears tracking."""
        from tests.unit.functions.fake_agent import FakeAgentFunction
        import asyncio
        
        fake = FakeAgentFunction()
        asyncio.get_event_loop().run_until_complete(fake.run())
        
        assert fake.call_count == 1
        
        fake.reset()
        
        assert fake.call_count == 0
        assert fake.call_args == []
    
    def test_fake_agent_function_is_agent_function(self) -> None:
        """FakeAgentFunction is a subclass of AgentFunction."""
        from src.functions.base import AgentFunction
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        assert issubclass(FakeAgentFunction, AgentFunction)


# =============================================================================
# AC-5.6: Protocol duck typing for type hints
# =============================================================================

class TestProtocolDuckTyping:
    """Tests for Protocol-based type hints."""
    
    def test_agent_function_protocol_exists(self) -> None:
        """AgentFunctionProtocol exists for duck typing."""
        from src.functions.base import AgentFunctionProtocol
        
        assert AgentFunctionProtocol is not None
    
    def test_protocol_is_runtime_checkable(self) -> None:
        """AgentFunctionProtocol is runtime checkable."""
        from typing import runtime_checkable
        from src.functions.base import AgentFunctionProtocol
        
        # If it's runtime checkable, isinstance checks should work
        assert hasattr(AgentFunctionProtocol, "__protocol_attrs__") or \
               getattr(AgentFunctionProtocol, "_is_runtime_protocol", False)
    
    def test_agent_function_implements_protocol(self) -> None:
        """AgentFunction subclasses implement AgentFunctionProtocol."""
        from src.functions.base import AgentFunction, AgentFunctionProtocol
        
        class TestFunction(AgentFunction):
            name = "test"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        
        # Should pass isinstance check
        assert isinstance(func, AgentFunctionProtocol)
    
    def test_fake_agent_function_implements_protocol(self) -> None:
        """FakeAgentFunction implements AgentFunctionProtocol."""
        from src.functions.base import AgentFunctionProtocol
        from tests.unit.functions.fake_agent import FakeAgentFunction
        
        fake = FakeAgentFunction()
        
        assert isinstance(fake, AgentFunctionProtocol)
    
    def test_protocol_has_required_methods(self) -> None:
        """Protocol defines required methods and properties."""
        from src.functions.base import AgentFunctionProtocol
        import inspect
        
        # Check for expected members
        members = [m for m in dir(AgentFunctionProtocol) if not m.startswith("_")]
        
        assert "name" in members or hasattr(AgentFunctionProtocol, "name")
        assert "default_preset" in members or hasattr(AgentFunctionProtocol, "default_preset")
        assert "run" in members or hasattr(AgentFunctionProtocol, "run")
        assert "get_context_budget" in members or hasattr(AgentFunctionProtocol, "get_context_budget")


# =============================================================================
# Additional Tests
# =============================================================================

class TestAgentFunctionRepr:
    """Tests for AgentFunction representation."""
    
    def test_repr_includes_name(self) -> None:
        """__repr__ includes function name."""
        from src.functions.base import AgentFunction
        
        class TestFunction(AgentFunction):
            name = "test_function"
            default_preset = "S1"
            
            async def run(self, **kwargs: Any) -> BaseModel:
                return BaseModel()
        
        func = TestFunction()
        repr_str = repr(func)
        
        assert "TestFunction" in repr_str
        assert "test_function" in repr_str
        assert "S1" in repr_str


class TestAgentFunctionIntegration:
    """Integration tests for AgentFunction with caching."""
    
    @pytest.mark.asyncio
    async def test_agent_function_with_handoff_cache(self) -> None:
        """AgentFunction can use HandoffCache for state."""
        from src.functions.base import AgentFunction
        from src.cache import HandoffCache
        
        class IntegrationOutput(BaseModel):
            result: str
        
        class StatefulFunction(AgentFunction):
            name = "stateful"
            
            async def run(
                self,
                *,
                cache: HandoffCache,
                key: str,
                value: str,
                **kwargs: Any,
            ) -> BaseModel:
                await cache.set(key, value)
                return IntegrationOutput(result="stored")
        
        cache = HandoffCache("test_pipeline")
        func = StatefulFunction()
        
        await func.run(cache=cache, key="result", value="test_value")
        
        retrieved = await cache.get("result")
        assert retrieved == "test_value"
