"""A2A Protocol Models Package.

Contains pydantic models for Agent-to-Agent (A2A) protocol integration.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A Protocol Integration
"""

from src.a2a.agent_card import generate_agent_card
from src.a2a.models import AgentCard, Capability, Skill

__all__ = [
    "AgentCard",
    "Capability",
    "Skill",
    "generate_agent_card",
]
