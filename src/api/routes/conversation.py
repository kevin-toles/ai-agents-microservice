"""
Conversation API Routes - Inter-AI Conversation Endpoints

Provides REST endpoints for managing inter-AI conversations.

Endpoints:
- POST /v1/conversation/start - Start a new conversation
- GET /v1/conversation/{id} - Get conversation state
- GET /v1/conversation/{id}/transcript - Get full transcript
- POST /v1/conversation/{id}/inject - Inject human message
- POST /v1/conversation/{id}/stop - Stop conversation

Reference: docs/INTER_AI_ORCHESTRATION.md
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.conversation.models import (
    Conversation,
    ConversationStatus,
    Participant,
    ParticipantType,
)
from src.conversation.orchestrator import ConversationOrchestrator
from src.participants.llm_participant import LLMParticipantAdapter
from src.participants.tool_participant import ToolParticipantAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/conversation", tags=["conversation"])

# Global orchestrator instance (initialized on first use)
_orchestrator: ConversationOrchestrator | None = None


def get_orchestrator() -> ConversationOrchestrator:
    """Get or create the conversation orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        llm_adapter = LLMParticipantAdapter()
        tool_adapter = ToolParticipantAdapter()
        _orchestrator = ConversationOrchestrator(
            llm_client=llm_adapter,
            tool_client=tool_adapter,
        )
    return _orchestrator


# =============================================================================
# Request/Response Models
# =============================================================================


class ParticipantRequest(BaseModel):
    """Participant definition for a conversation."""
    
    id: str = Field(..., description="Unique participant ID")
    name: str = Field(..., description="Human-readable name")
    participant_type: str = Field(..., description="'llm' or 'tool'")
    provider: str | None = Field(None, description="LLM provider name")
    model: str | None = Field(None, description="LLM model name")
    system_prompt: str | None = Field(None, description="System prompt for LLMs")


class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""
    
    task: str = Field(..., description="The problem/task to solve")
    participants: list[ParticipantRequest] = Field(..., description="List of participants")
    context: dict[str, Any] = Field(default_factory=dict, description="Shared context")
    turn_order: list[str] | None = Field(None, description="Order of turns")
    max_rounds: int = Field(default=10, description="Maximum conversation rounds")


class InjectMessageRequest(BaseModel):
    """Request to inject a message into a conversation."""
    
    content: str = Field(..., description="Message content")
    from_participant: str = Field(default="human", description="Who the message is from")


class StopConversationRequest(BaseModel):
    """Request to stop a conversation."""
    
    reason: str = Field(default="Manually stopped", description="Reason for stopping")


class ConversationResponse(BaseModel):
    """Response containing conversation state."""
    
    conversation_id: str
    task: str
    status: str
    current_round: int
    max_rounds: int
    message_count: int
    participants: list[str]


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/start", response_model=ConversationResponse)
async def start_conversation(request: StartConversationRequest) -> ConversationResponse:
    """Start a new inter-AI conversation.
    
    Creates a conversation with the specified participants and context,
    then begins the discussion.
    """
    orchestrator = get_orchestrator()
    
    # Convert request participants to domain model
    participants = []
    for p in request.participants:
        participant_type = (
            ParticipantType.LLM if p.participant_type == "llm" else ParticipantType.TOOL
        )
        participants.append(
            Participant(
                id=p.id,
                name=p.name,
                participant_type=participant_type,
                provider=p.provider,
                model=p.model,
                system_prompt=p.system_prompt,
            )
        )
    
    # Start conversation
    conversation = await orchestrator.start_conversation(
        task=request.task,
        participants=participants,
        context=request.context,
        turn_order=request.turn_order,
        max_rounds=request.max_rounds,
    )
    
    logger.info(f"Started conversation {conversation.conversation_id}")
    
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        task=conversation.task,
        status=conversation.status.value,
        current_round=conversation.current_round,
        max_rounds=conversation.max_rounds,
        message_count=len(conversation.messages),
        participants=[p.id for p in conversation.participants],
    )


@router.post("/{conversation_id}/run", response_model=ConversationResponse)
async def run_conversation(conversation_id: str) -> ConversationResponse:
    """Run an existing conversation until completion.
    
    Executes turns until consensus, timeout, or max rounds reached.
    """
    orchestrator = get_orchestrator()
    
    conversation = orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Run the conversation
    conversation = await orchestrator.run_conversation(conversation)
    
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        task=conversation.task,
        status=conversation.status.value,
        current_round=conversation.current_round,
        max_rounds=conversation.max_rounds,
        message_count=len(conversation.messages),
        participants=[p.id for p in conversation.participants],
    )


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get the current state of a conversation."""
    orchestrator = get_orchestrator()
    
    conversation = orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.to_dict()


@router.get("/{conversation_id}/transcript")
async def get_transcript(conversation_id: str) -> dict[str, str]:
    """Get the full transcript of a conversation."""
    orchestrator = get_orchestrator()
    
    conversation = orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "transcript": conversation.get_transcript(),
    }


@router.post("/{conversation_id}/inject")
async def inject_message(
    conversation_id: str,
    request: InjectMessageRequest,
) -> dict[str, Any]:
    """Inject a message into an active conversation (human-in-the-loop)."""
    orchestrator = get_orchestrator()
    
    message = await orchestrator.inject_message(
        conversation_id=conversation_id,
        content=request.content,
        from_participant=request.from_participant,
    )
    
    if not message:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return message.to_dict()


@router.post("/{conversation_id}/stop")
async def stop_conversation(
    conversation_id: str,
    request: StopConversationRequest,
) -> ConversationResponse:
    """Stop an active conversation."""
    orchestrator = get_orchestrator()
    
    conversation = await orchestrator.stop_conversation(
        conversation_id=conversation_id,
        reason=request.reason,
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        task=conversation.task,
        status=conversation.status.value,
        current_round=conversation.current_round,
        max_rounds=conversation.max_rounds,
        message_count=len(conversation.messages),
        participants=[p.id for p in conversation.participants],
    )
