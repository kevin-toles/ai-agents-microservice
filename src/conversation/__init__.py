"""
Conversation Module - Inter-AI Conversation Orchestration

This package provides infrastructure for orchestrating conversations
between multiple AI participants (LLMs and BERT tools).

Architecture:
- ConversationOrchestrator: Central coordinator for all communication
- Participants: Adapters for LLMs (via llm-gateway) and Tools (via Code-Orchestrator)
- Turn strategies: Round-robin, tool-first, debate

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from src.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationStatus,
    ParticipantType,
)
from src.conversation.orchestrator import ConversationOrchestrator

__all__ = [
    "Conversation",
    "ConversationMessage",
    "ConversationStatus",
    "ParticipantType",
    "ConversationOrchestrator",
]
