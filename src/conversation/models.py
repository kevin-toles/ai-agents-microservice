"""
Conversation Models - Data structures for Inter-AI Conversations

This module defines the core data structures for managing conversations
between AI participants.

Reference: docs/INTER_AI_ORCHESTRATION.md - Message Schema section
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ParticipantType(str, Enum):
    """Type of participant in the conversation."""
    
    LLM = "llm"                    # External LLM (via llm-gateway)
    TOOL = "tool"                  # BERT tool (via Code-Orchestrator)
    ORCHESTRATOR = "orchestrator"  # ai-agents itself


class ConversationStatus(str, Enum):
    """Status of a conversation."""
    
    PENDING = "pending"            # Not yet started
    IN_PROGRESS = "in_progress"    # Active conversation
    CONSENSUS = "consensus"        # Participants reached agreement
    DEADLOCK = "deadlock"          # Cannot reach agreement
    COMPLETE = "complete"          # Task finished successfully
    TIMEOUT = "timeout"            # Max rounds reached
    ERROR = "error"                # Error occurred


@dataclass
class ConversationMessage:
    """Single message in an inter-AI conversation.
    
    Attributes:
        message_id: Unique message identifier.
        conversation_id: Parent conversation ID.
        participant_id: Who sent this (e.g., "qwen", "gpt-5.2", "bertopic").
        participant_type: LLM, TOOL, or ORCHESTRATOR.
        timestamp: When the message was sent.
        role: Message role ("user", "assistant", "tool", "system").
        content: The actual message content.
        tokens_used: Token count (for LLMs).
        latency_ms: Response time in milliseconds.
        metadata: Participant-specific metadata.
    """
    
    conversation_id: str
    participant_id: str
    participant_type: ParticipantType
    role: str
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens_used: int | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "participant_id": self.participant_id,
            "participant_type": self.participant_type.value,
            "timestamp": self.timestamp.isoformat(),
            "role": self.role,
            "content": self.content,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class Participant:
    """Definition of a conversation participant.
    
    Attributes:
        id: Unique identifier (e.g., "qwen", "gpt-5.2", "bertopic").
        name: Human-readable name.
        participant_type: LLM or TOOL.
        provider: For LLMs, the provider name (e.g., "openrouter", "openai").
        model: For LLMs, the model name.
        endpoint: For tools, the API endpoint.
        system_prompt: Optional system prompt for LLMs.
        capabilities: List of what this participant can do.
    """
    
    id: str
    name: str
    participant_type: ParticipantType
    provider: str | None = None
    model: str | None = None
    endpoint: str | None = None
    system_prompt: str | None = None
    capabilities: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "participant_type": self.participant_type.value,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
        }


@dataclass
class Conversation:
    """Complete inter-AI conversation state.
    
    Attributes:
        conversation_id: Unique conversation identifier.
        task: What problem are we solving?
        participants: List of active participants.
        messages: All messages in the conversation.
        context: Shared context (book list, clusters, etc.).
        current_turn: Who should speak next.
        turn_order: Default turn sequence.
        status: Current conversation status.
        consensus_threshold: Agreement level needed (0.0-1.0).
        max_rounds: Prevent infinite loops.
        current_round: Current round number.
        created_at: When the conversation started.
        updated_at: Last activity timestamp.
        result: Final result/output of the conversation.
    """
    
    task: str
    participants: list[Participant]
    context: dict[str, Any]
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[ConversationMessage] = field(default_factory=list)
    current_turn: str | None = None
    turn_order: list[str] = field(default_factory=list)
    status: ConversationStatus = ConversationStatus.PENDING
    consensus_threshold: float = 0.8
    max_rounds: int = 10
    min_rounds: int = 1  # Minimum rounds before consensus can be declared
    current_round: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    result: dict[str, Any] | None = None
    
    def __post_init__(self):
        """Initialize turn order from participants if not set."""
        if not self.turn_order and self.participants:
            self.turn_order = [p.id for p in self.participants]
        if not self.current_turn and self.turn_order:
            self.current_turn = self.turn_order[0]
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_participant(self, participant_id: str) -> Participant | None:
        """Get a participant by ID."""
        for p in self.participants:
            if p.id == participant_id:
                return p
        return None
    
    def advance_turn(self) -> str | None:
        """Advance to the next participant's turn.
        
        Returns:
            The ID of the next participant, or None if no more turns.
        """
        if not self.turn_order:
            return None
        
        if self.current_turn is None:
            self.current_turn = self.turn_order[0]
            return self.current_turn
        
        try:
            current_idx = self.turn_order.index(self.current_turn)
            next_idx = (current_idx + 1) % len(self.turn_order)
            
            # Check if we've completed a round
            if next_idx == 0:
                self.current_round += 1
                
            self.current_turn = self.turn_order[next_idx]
            return self.current_turn
            
        except ValueError:
            # Current turn not in order, reset to first
            self.current_turn = self.turn_order[0]
            return self.current_turn
    
    def get_message_history_for_llm(
        self,
        max_messages: int = 50,
    ) -> list[dict[str, str]]:
        """Get message history formatted for LLM context.
        
        Args:
            max_messages: Maximum messages to include.
            
        Returns:
            List of {role, content} dicts for LLM.
        """
        history = []
        for msg in self.messages[-max_messages:]:
            # Format as conversation for LLM
            speaker = msg.participant_id.upper()
            formatted_content = f"[{speaker}]: {msg.content}"
            history.append({
                "role": "user" if msg.participant_type == ParticipantType.ORCHESTRATOR else "assistant",
                "content": formatted_content,
            })
        return history
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "task": self.task,
            "participants": [p.to_dict() for p in self.participants],
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "current_turn": self.current_turn,
            "turn_order": self.turn_order,
            "status": self.status.value,
            "consensus_threshold": self.consensus_threshold,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
        }
    
    def get_transcript(self) -> str:
        """Get full conversation transcript as text.
        
        Returns:
            Human-readable transcript.
        """
        lines = [
            f"=== CONVERSATION: {self.conversation_id} ===",
            f"Task: {self.task}",
            f"Status: {self.status.value}",
            f"Rounds: {self.current_round}/{self.max_rounds}",
            "",
            "--- TRANSCRIPT ---",
        ]
        
        for msg in self.messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            speaker = msg.participant_id.upper()
            lines.append(f"[{timestamp}] {speaker}: {msg.content}")
            
        if self.result:
            lines.append("")
            lines.append("--- RESULT ---")
            lines.append(str(self.result))
            
        return "\n".join(lines)
