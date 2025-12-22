"""
Base Participant - Abstract interface for conversation participants.

All participant adapters must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.conversation.models import Conversation, Participant


class BaseParticipant(ABC):
    """Abstract base class for conversation participants.
    
    All participant adapters (LLM, Tool) must implement this interface.
    """
    
    @abstractmethod
    async def respond(
        self,
        conversation: Conversation,
        participant: Participant,
    ) -> dict[str, Any]:
        """Generate a response from this participant.
        
        Args:
            conversation: Current conversation state.
            participant: The participant definition.
            
        Returns:
            Dict with:
                - content: str - The response content
                - tokens_used: int | None - Token count (LLMs only)
                - latency_ms: int - Response time in ms
                - metadata: dict - Additional metadata
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the participant service is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        pass
