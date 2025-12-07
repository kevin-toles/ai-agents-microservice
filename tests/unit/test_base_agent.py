"""Unit tests for BaseAgent.

TDD Phase: GREEN - Tests updated to match actual implementation.
Pattern: Abstract base class testing
"""

import asyncio
import pytest
from abc import ABC
from typing import Any
from unittest.mock import AsyncMock

from pydantic import BaseModel

from src.agents.base import BaseAgent


class SampleInput(BaseModel):
    """Sample input model for tests."""
    data: str


class SampleOutput(BaseModel):
    """Sample output model for tests."""
    result: str


class TestBaseAgent:
    """Tests for BaseAgent abstract base class."""
    
    def test_base_agent_is_abstract(self) -> None:
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(name="test")  # type: ignore
    
    def test_base_agent_requires_description(self) -> None:
        """Test that subclasses must implement description property."""
        class IncompleteAgent(BaseAgent[SampleInput, SampleOutput]):
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        with pytest.raises(TypeError):
            IncompleteAgent(name="test")  # type: ignore
    
    def test_base_agent_subclass_can_be_instantiated(self) -> None:
        """Test that properly implemented subclass works."""
        class ConcreteAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "A test agent"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="success")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        agent = ConcreteAgent(name="test-agent")
        assert agent is not None
        assert agent.name == "test-agent"
    
    def test_base_agent_inherits_from_abc(self) -> None:
        """Test that BaseAgent inherits from ABC."""
        assert issubclass(BaseAgent, ABC)


class TestBaseAgentMethods:
    """Tests for BaseAgent method signatures."""
    
    def test_run_method_is_async(self) -> None:
        """Test that run() is an async method."""
        import inspect
        
        class ConcreteAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "A test agent"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        agent = ConcreteAgent(name="test")
        assert inspect.iscoroutinefunction(agent.run)
    
    def test_validate_input_is_async(self) -> None:
        """Test that validate_input() is async."""
        import inspect
        
        class ConcreteAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "A test agent"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        agent = ConcreteAgent(name="test")
        assert inspect.iscoroutinefunction(agent.validate_input)
    
    def test_description_property(self) -> None:
        """Test that description property works correctly."""
        class ConcreteAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "Cross-Reference Agent for taxonomy traversal"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        agent = ConcreteAgent(name="test")
        assert agent.description == "Cross-Reference Agent for taxonomy traversal"


class TestBaseAgentValidation:
    """Tests for BaseAgent input validation."""
    
    def test_validation_can_reject_invalid_input(self) -> None:
        """Test that validation can return False for invalid input."""
        class StrictAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "Test agent"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                if not await self.validate_input(input_data):
                    raise ValueError("Invalid input")
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return bool(input_data.data)  # Empty string is invalid
        
        agent = StrictAgent(name="test")
        
        # Valid input
        result = asyncio.run(agent.validate_input(SampleInput(data="hello")))
        assert result is True
        
        # Invalid input
        result = asyncio.run(agent.validate_input(SampleInput(data="")))
        assert result is False
    
    def test_agent_repr(self) -> None:
        """Test that agent repr includes name."""
        class ConcreteAgent(BaseAgent[SampleInput, SampleOutput]):
            @property
            def description(self) -> str:
                return "Test agent"
            
            async def run(self, input_data: SampleInput) -> SampleOutput:
                return SampleOutput(result="done")
            
            async def validate_input(self, input_data: SampleInput) -> bool:
                return True
        
        agent = ConcreteAgent(name="my-agent")
        assert "my-agent" in repr(agent)
        assert "ConcreteAgent" in repr(agent)
