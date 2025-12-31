"""Cross-Reference Agent API Route - WBS 5.14 GREEN Phase.

This module implements the REST API endpoint for the Cross-Reference Agent.

Reference Documents:
- GUIDELINES: FastAPI dependency injection (Sinha pp. 89-91)
- GUIDELINES: Pydantic validators (Sinha pp. 193-195)
- GUIDELINES: REST constraints (Buelta pp. 92-93, 126)
- ARCHITECTURE.md (ai-agents): Cross-Reference Agent patterns
- TIER_RELATIONSHIP_DIAGRAM.md: Spider Web Model

Anti-Patterns Avoided:
- ANTI_PATTERN_ANALYSIS §1.1: Optional types with explicit None
- ANTI_PATTERN_ANALYSIS §3.1: No bare except clauses
- ANTI_PATTERN_ANALYSIS §4.1: Cognitive complexity < 15 per function
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.cross_reference.agent import CrossReferenceAgent
from src.agents.cross_reference.state import (
    Citation,
    CrossReferenceInput,
    CrossReferenceResult,
    SourceChapter,
    TierCoverage,
    TraversalConfig,
)


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration - WBS 5.13.1
# =============================================================================

router = APIRouter(
    prefix="/v1/agents/cross-reference",
    tags=["Agents"],
)


# =============================================================================
# Request/Response Models - WBS 5.13.3, 5.13.4
# =============================================================================


class CrossReferenceRequest(BaseModel):
    """Request model for cross-reference endpoint.

    Pattern: Pydantic request validation (Sinha pp. 193-195)
    """

    source: SourceChapter = Field(..., description="Source chapter to cross-reference")
    config: TraversalConfig = Field(
        default_factory=TraversalConfig,
        description="Traversal configuration"
    )
    taxonomy_id: str = Field(default="ai-ml", description="Taxonomy identifier")


class CrossReferenceResponse(BaseModel):
    """Response model for cross-reference endpoint.

    Pattern: Structured API response
    """

    annotation: str = Field(..., description="Scholarly annotation with inline citations")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")
    tier_coverage: list[TierCoverage] = Field(
        default_factory=list, description="Coverage per tier"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    model_used: str = Field(default="", description="LLM model used for synthesis")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    agent: str = Field(default="cross-reference", description="Agent name")


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


# =============================================================================
# Agent Factory - Dependency Injection
# Pattern: Factory pattern for testability (ANTI_PATTERN §4.1)
# =============================================================================

_agent: CrossReferenceAgent | None = None


def get_agent() -> CrossReferenceAgent:
    """Get or create the CrossReferenceAgent instance.

    Pattern: Lazy initialization with caching

    Returns:
        Configured CrossReferenceAgent instance
    """
    global _agent
    if _agent is None:
        _agent = CrossReferenceAgent()
    return _agent


def set_agent(agent: CrossReferenceAgent | None) -> None:
    """Set the agent instance for testing.

    Args:
        agent: Agent instance to use, or None to reset
    """
    global _agent
    _agent = agent


# =============================================================================
# Endpoints - WBS 5.13.2, 5.13.6
# =============================================================================


@router.post(
    "",
    response_model=CrossReferenceResponse,
    responses={
        200: {"description": "Successful cross-reference generation"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_cross_reference(request: CrossReferenceRequest) -> CrossReferenceResponse:
    """Generate cross-references for a source chapter.

    This endpoint executes the full cross-reference workflow:
    1. Analyze source chapter concepts
    2. Search taxonomy for matches
    3. Traverse spider web graph
    4. Retrieve content for matched chapters
    5. Synthesize scholarly annotation

    Args:
        request: Cross-reference request with source chapter and config

    Returns:
        CrossReferenceResponse with annotation, citations, and tier coverage

    Raises:
        HTTPException: On validation or processing errors
    """
    try:
        agent = get_agent()

        # Build input for agent using CrossReferenceInput model
        traversal_config = TraversalConfig(
            max_hops=request.config.max_hops,
            min_similarity=request.config.min_similarity,
            include_tier1=request.config.include_tier1,
            include_tier2=request.config.include_tier2,
            include_tier3=request.config.include_tier3,
        )
        input_data = CrossReferenceInput(
            book=request.source.book,
            chapter=request.source.chapter,
            title=request.source.title,
            tier=request.source.tier,
            content=request.source.content,
            keywords=request.source.keywords,
            concepts=request.source.concepts,
            config=traversal_config,
        )

        # Run the agent workflow
        result: CrossReferenceResult = await agent.run(input_data)

        # Convert to response model
        return CrossReferenceResponse(
            annotation=result.annotation,
            citations=result.citations,
            tier_coverage=result.tier_coverage,
            processing_time_ms=result.processing_time_ms,
            model_used=result.model_used,
        )

    except ValueError as e:
        # Validation errors - return 400
        logger.warning(f"Validation error in cross-reference: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except Exception as e:
        # Unexpected errors - return 500
        logger.exception(f"Error in cross-reference agent: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}",
        ) from e


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Health check endpoint for the cross-reference agent.

    Returns:
        HealthResponse with status and agent information
    """
    try:
        # Verify agent can be instantiated
        _ = get_agent()
        return HealthResponse(status="healthy", agent="cross-reference")
    except Exception:
        return HealthResponse(status="unhealthy", agent="cross-reference")
