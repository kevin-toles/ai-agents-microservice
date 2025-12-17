"""MSEP Router - Enrich Metadata API Route.

WBS: MSE-4.1 - MSEP Router
Implements POST /v1/agents/enrich-metadata endpoint.

Reference Documents:
- GUIDELINES: FastAPI dependency injection (Sinha pp. 89-91)
- GUIDELINES: Pydantic validators (Sinha pp. 193-195)
- GUIDELINES: REST constraints (Buelta pp. 92-93, 126)
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.1

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S3776: Cognitive complexity < 15 per function
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.msep.config import MSEPConfig
from src.agents.msep.exceptions import (
    MSEPError,
    ServiceUnavailableError,
)
from src.agents.msep.schemas import (
    ChapterMeta,
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    MSEPRequest,
    Provenance,
)

if TYPE_CHECKING:
    from src.agents.msep.orchestrator import MSEPOrchestrator


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration - MSE-4.1
# =============================================================================

router = APIRouter(
    prefix="/v1/agents/enrich-metadata",
    tags=["Agents"],
)


# =============================================================================
# Request/Response Models - Pydantic validation
# =============================================================================


class ChapterMetaRequest(BaseModel):
    """Chapter metadata in request.

    Pattern: Pydantic request validation (Sinha pp. 193-195)
    """

    book: str = Field(..., description="Book title")
    chapter: int = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    id: str | None = Field(default=None, description="Optional chapter ID")


class MSEPConfigRequest(BaseModel):
    """Configuration in request.

    Pattern: Optional config with defaults
    """

    threshold: float = Field(default=0.45, description="Similarity threshold")
    top_k: int = Field(default=5, description="Top K results per chapter")
    timeout: float = Field(default=30.0, description="Service timeout in seconds")
    same_topic_boost: float = Field(default=0.15, description="Boost for same-topic")
    use_dynamic_threshold: bool = Field(default=True, description="Dynamic threshold")
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search")


class EnrichMetadataRequest(BaseModel):
    """Request model for MSEP enrichment endpoint.

    Pattern: Pydantic request validation with field validators
    """

    corpus: list[str] = Field(..., description="Document texts")
    chapter_index: list[ChapterMetaRequest] = Field(..., description="Chapter metadata")
    config: MSEPConfigRequest | None = Field(default=None, description="MSEP config")

    @field_validator("corpus")
    @classmethod
    def corpus_not_empty(cls, v: list[str]) -> list[str]:
        """Validate corpus is not empty."""
        if not v:
            raise ValueError("corpus cannot be empty")
        return v

    @field_validator("chapter_index")
    @classmethod
    def chapter_index_not_empty(
        cls, v: list[ChapterMetaRequest]
    ) -> list[ChapterMetaRequest]:
        """Validate chapter_index is not empty."""
        if not v:
            raise ValueError("chapter_index cannot be empty")
        return v

    @model_validator(mode="after")
    def lengths_match(self) -> "EnrichMetadataRequest":
        """Validate corpus and chapter_index have same length."""
        if len(self.corpus) != len(self.chapter_index):
            raise ValueError(
                f"corpus length ({len(self.corpus)}) must match "
                f"chapter_index length ({len(self.chapter_index)})"
            )
        return self


class CrossReferenceResponse(BaseModel):
    """Cross-reference in response."""

    target: str
    score: float
    base_score: float
    topic_boost: float
    method: str


class MergedKeywordsResponse(BaseModel):
    """Merged keywords in response."""

    tfidf: list[str]
    semantic: list[str]
    merged: list[str]


class ProvenanceResponse(BaseModel):
    """Provenance in response."""

    methods_used: list[str]
    sbert_score: float
    topic_boost: float
    timestamp: str


class EnrichedChapterResponse(BaseModel):
    """Enriched chapter in response.

    Per MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md Schema Definitions.
    """

    book: str
    chapter: int
    title: str
    chapter_id: str
    cross_references: list[CrossReferenceResponse]
    keywords: MergedKeywordsResponse
    topic_id: int | None
    topic_name: str | None
    graph_relationships: list[str]
    provenance: ProvenanceResponse


class EnrichMetadataResponse(BaseModel):
    """Response model for MSEP enrichment endpoint."""

    chapters: list[EnrichedChapterResponse]
    processing_time_ms: float
    total_cross_references: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    agent: str = Field(default="msep", description="Agent name")


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


# =============================================================================
# Orchestrator Factory - Dependency Injection
# Pattern: Factory pattern for testability
# =============================================================================

_orchestrator: "MSEPOrchestrator | None" = None


def get_orchestrator() -> "MSEPOrchestrator":
    """Get or create the MSEPOrchestrator instance.

    Pattern: Lazy initialization with caching

    Returns:
        Configured MSEPOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        # Import here to avoid circular imports
        from src.agents.msep.orchestrator import MSEPOrchestrator

        _orchestrator = MSEPOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: "MSEPOrchestrator | None") -> None:
    """Set the orchestrator instance for testing.

    Args:
        orchestrator: Orchestrator instance to use, or None to reset
    """
    global _orchestrator
    _orchestrator = orchestrator


# =============================================================================
# Helper Functions - Cognitive complexity reduction
# =============================================================================


def _build_msep_request(request: EnrichMetadataRequest) -> MSEPRequest:
    """Build MSEPRequest from Pydantic request model.

    Args:
        request: Pydantic request model

    Returns:
        MSEPRequest dataclass
    """
    # Build chapter index
    chapter_index = [
        ChapterMeta(
            book=ch.book,
            chapter=ch.chapter,
            title=ch.title,
            id=ch.id or "",
        )
        for ch in request.chapter_index
    ]

    # Build config
    if request.config:
        config = MSEPConfig(
            threshold=request.config.threshold,
            top_k=request.config.top_k,
            timeout=request.config.timeout,
            same_topic_boost=request.config.same_topic_boost,
            use_dynamic_threshold=request.config.use_dynamic_threshold,
            enable_hybrid_search=request.config.enable_hybrid_search,
        )
    else:
        config = MSEPConfig()

    return MSEPRequest(
        corpus=request.corpus,
        chapter_index=chapter_index,
        config=config,
    )


def _build_response(result: EnrichedMetadata) -> EnrichMetadataResponse:
    """Build Pydantic response model from EnrichedMetadata.

    Args:
        result: EnrichedMetadata dataclass

    Returns:
        EnrichMetadataResponse Pydantic model
    """
    chapters = [
        EnrichedChapterResponse(
            book=ch.book,
            chapter=ch.chapter,
            title=ch.title,
            chapter_id=ch.chapter_id,
            cross_references=[
                CrossReferenceResponse(
                    target=xref.target,
                    score=xref.score,
                    base_score=xref.base_score,
                    topic_boost=xref.topic_boost,
                    method=xref.method,
                )
                for xref in ch.cross_references
            ],
            keywords=MergedKeywordsResponse(
                tfidf=ch.keywords.tfidf,
                semantic=ch.keywords.semantic,
                merged=ch.keywords.merged,
            ),
            topic_id=ch.topic_id,
            topic_name=ch.topic_name,
            graph_relationships=ch.graph_relationships,
            provenance=ProvenanceResponse(
                methods_used=ch.provenance.methods_used,
                sbert_score=ch.provenance.sbert_score,
                topic_boost=ch.provenance.topic_boost,
                timestamp=ch.provenance.timestamp,
            ),
        )
        for ch in result.chapters
    ]

    return EnrichMetadataResponse(
        chapters=chapters,
        processing_time_ms=result.processing_time_ms,
        total_cross_references=result.total_cross_references,
    )


# =============================================================================
# Endpoints - MSE-4.1
# =============================================================================


@router.post(
    "",
    response_model=EnrichMetadataResponse,
    responses={
        200: {"description": "Successful metadata enrichment"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def enrich_metadata(request: EnrichMetadataRequest) -> EnrichMetadataResponse:
    """Enrich metadata for a corpus using MSEP.

    This endpoint executes the MSEP workflow:
    1. Dispatch parallel calls to SBERT, TF-IDF, BERTopic
    2. Optionally run hybrid search
    3. Merge results with topic boost
    4. Return enriched metadata

    Args:
        request: MSEP enrichment request with corpus and config

    Returns:
        EnrichMetadataResponse with enriched chapters

    Raises:
        HTTPException: On validation or processing errors
    """
    try:
        orchestrator = get_orchestrator()
        msep_request = _build_msep_request(request)
        result = await orchestrator.enrich_metadata(msep_request)
        return _build_response(result)

    except ServiceUnavailableError as e:
        logger.warning(f"Service unavailable in MSEP: {e.message}")
        raise HTTPException(
            status_code=503,
            detail=e.message,
        ) from e

    except MSEPError as e:
        logger.warning(f"MSEP error: {e.message}")
        raise HTTPException(
            status_code=400,
            detail=e.message,
        ) from e

    except Exception as e:
        logger.exception(f"Unexpected error in MSEP: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}",
        ) from e


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Health check endpoint for the MSEP agent.

    Returns:
        HealthResponse with status and agent information
    """
    try:
        _ = get_orchestrator()
        return HealthResponse(status="healthy", agent="msep")
    except Exception:
        return HealthResponse(status="unhealthy", agent="msep")
