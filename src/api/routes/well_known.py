"""Well-Known Routes for A2A Protocol.

WBS-PI2: A2A Agent Card & Discovery - Well-Known Endpoint

Implements:
- GET /.well-known/agent-card.json - A2A Agent Card discovery endpoint

Acceptance Criteria:
- AC-PI2.4: GET /.well-known/agent-card.json returns 404 when disabled
- AC-PI2.5: GET /.well-known/agent-card.json returns valid card when enabled

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Agent Card Discovery
"""

from fastapi import APIRouter, Depends, HTTPException

from src.a2a.agent_card import generate_agent_card
from src.a2a.models import AgentCard
from src.config.feature_flags import ProtocolFeatureFlags, get_feature_flags


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    tags=["A2A Protocol"],
)


# =============================================================================
# AC-PI2.4, AC-PI2.5: Agent Card Discovery Endpoint
# =============================================================================


@router.get(
    "/.well-known/agent-card.json",
    response_model=AgentCard,
    summary="Get A2A Agent Card",
    description="Returns the A2A Agent Card for service discovery (A2A Protocol v0.3.0)",
    responses={
        200: {
            "description": "Agent Card successfully retrieved",
            "content": {
                "application/json": {
                    "example": {
                        "protocolVersion": "0.3.0",
                        "name": "ai-agents-service",
                        "description": "AI Platform Agent Functions Service",
                        "version": "1.0.0",
                        "capabilities": {
                            "streaming": True,
                            "pushNotifications": False,
                            "stateTransitionHistory": True,
                        },
                        "skills": [],
                    }
                }
            },
        },
        404: {"description": "Agent Card not available (feature disabled)"},
    },
)
async def get_agent_card(
    flags: ProtocolFeatureFlags = Depends(get_feature_flags),
) -> AgentCard:
    """Get A2A Agent Card for service discovery.
    
    Returns the Agent Card manifest declaring this service's capabilities,
    skills, and metadata. Used by A2A protocol clients for dynamic discovery.
    
    Requires both a2a_enabled and a2a_agent_card_enabled feature flags.
    
    Args:
        flags: Protocol feature flags (injected via dependency)
    
    Returns:
        AgentCard with service metadata and available skills
    
    Raises:
        HTTPException: 404 if Agent Card feature is disabled
    
    Example:
        ```bash
        export AGENTS_A2A_ENABLED=true
        export AGENTS_A2A_AGENT_CARD_ENABLED=true
        curl http://localhost:8082/.well-known/agent-card.json | jq .
        ```
    """
    # AC-PI2.4: Return 404 if feature disabled
    if not flags.a2a_enabled or not flags.a2a_agent_card_enabled:
        raise HTTPException(
            status_code=404,
            detail="Agent Card not available",
        )
    
    # AC-PI2.5: Generate and return Agent Card
    return generate_agent_card(flags=flags)


__all__ = ["router"]
