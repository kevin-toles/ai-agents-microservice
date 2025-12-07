"""Base agent interface.

All agents must implement this abstract base class.

Pattern: Abstract Interface with Protocol
Source: GUIDELINES_AI_Engineering line 795 (Repository pattern), 
        Architecture Patterns with Python Ch. 2
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


# Type variables for generic agent I/O
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all AI agents.
    
    Agents are autonomous entities that:
    1. Perceive their environment (receive input)
    2. Plan actions (reason about the task)
    3. Execute tools (interact with external systems)
    4. Produce output (return results)
    
    Source: GUIDELINES_AI_Engineering Segment 27 (pp. 536-554)
    "An agent is anything that can perceive its environment and act upon
    that environment...two aspects that determine the capabilities of an
    agent: tools and planning."
    """
    
    def __init__(self, name: str) -> None:
        """Initialize agent.
        
        Args:
            name: Unique name for this agent instance
        """
        self.name = name
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return agent description for tool registration."""
        ...
    
    @abstractmethod
    async def run(self, input_data: InputT) -> OutputT:
        """Execute the agent's main workflow.
        
        Args:
            input_data: Agent input (Pydantic model)
            
        Returns:
            Agent output (Pydantic model)
            
        Raises:
            AgentExecutionError: If execution fails
            AgentTimeoutError: If execution exceeds timeout
        """
        ...
    
    @abstractmethod
    async def validate_input(self, input_data: InputT) -> bool:
        """Validate input before execution.
        
        Args:
            input_data: Agent input to validate
            
        Returns:
            True if input is valid
            
        Raises:
            ValueError: If input is invalid
        """
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
