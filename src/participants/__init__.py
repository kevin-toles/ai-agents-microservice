"""
Participants Package - Adapters for AI Conversation Participants

This package contains adapters for different types of participants:
- LLM participants (via llm-gateway)
- Tool participants (via Code-Orchestrator-Service)

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from src.participants.base import BaseParticipant
from src.participants.llm_participant import LLMParticipantAdapter
from src.participants.tool_participant import ToolParticipantAdapter


__all__ = [
    "BaseParticipant",
    "LLMParticipantAdapter",
    "ToolParticipantAdapter",
]
