"""
Protocols API Routes - Kitchen Brigade Protocol Endpoints

Provides REST endpoints for managing and executing Kitchen Brigade protocols.
This exposes the protocol executor via microservice API endpoints.

Endpoints:
- GET /v1/protocols - List available protocols
- GET /v1/protocols/{protocol_id} - Get protocol details
- POST /v1/protocols/{protocol_id}/run - Execute protocol

Reference: src/protocols/kitchen_brigade_executor.py

Architecture (Building Microservices - Sam Newman):
    ai-agents acts as the Orchestrator, exposing Kitchen Brigade protocols
    via REST API. Protocol execution coordinates with:
    - LLM Gateway (:8080) - inference
    - Semantic Search (:8081) - cross-reference retrieval
    - Code-Orchestrator (:8083) - code analysis

TDD Status: Implementation following tests in tests/unit/routes/test_protocols.py
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.protocols.kitchen_brigade_executor import (
    KitchenBrigadeExecutor,
    TaxonomyFilter,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/protocols", tags=["protocols"])

# Protocol definitions directory
PROTOCOLS_DIR = Path(__file__).parent.parent.parent.parent / "config" / "protocols"

# Error message constants
ERROR_PROTOCOL_NOT_FOUND = "Protocol not found"
ERROR_EXECUTION_FAILED = "Protocol execution failed"


# =============================================================================
# Request/Response Models (Building Python Microservices with FastAPI pattern)
# =============================================================================


class BrigadeRoleSummary(BaseModel):
    """Summary of a brigade role configuration."""
    
    model: str = Field(..., description="Default model for this role")
    temperature: float = Field(default=0.3, description="Temperature setting")
    max_tokens: int = Field(default=4096, description="Max tokens")
    system_prompt: str = Field(..., description="System prompt for role")


class RoundDefinition(BaseModel):
    """Protocol round definition (legacy format)."""
    
    round: int = Field(..., description="Round number")
    type: str = Field(..., description="Round type: parallel, synthesis, consensus")
    name: str = Field(default="", description="Round name")
    description: str = Field(default="", description="Round description")
    roles: list[str] = Field(default_factory=list, description="Participating roles")


class StageDefinition(BaseModel):
    """Protocol stage definition (new Kitchen Brigade format)."""
    
    stage: int = Field(..., description="Stage number (1-4)")
    name: str = Field(..., description="Stage name (decompose_task, cross_reference, etc.)")
    type: str = Field(default="", description="Stage type: extract, parallel_retrieval, iterative, synthesis")
    description: str = Field(default="", description="Stage description")
    layers: list[str] | None = Field(default=None, description="Cross-reference layers for stage 2")
    max_iterations: int | None = Field(default=None, description="Max iterations for stage 3")
    can_request_more_info: bool | None = Field(default=None, description="Whether LLMs can request NEED_MORE_INFO")


class ProtocolSummary(BaseModel):
    """Protocol summary for list view."""
    
    protocol_id: str = Field(..., description="Unique protocol identifier")
    name: str = Field(..., description="Human-readable protocol name")
    description: str = Field(..., description="Protocol description")
    brigade_roles: list[str] = Field(default_factory=list, description="Role names")


class ProtocolDetail(BaseModel):
    """Full protocol detail."""
    
    protocol_id: str
    name: str
    description: str
    brigade_roles: dict[str, BrigadeRoleSummary]
    rounds: list[RoundDefinition] = Field(default_factory=list, description="Legacy rounds format")
    stages: list[StageDefinition] = Field(default_factory=list, description="New stages format (4-stage Kitchen Brigade)")
    format: str = Field(default="rounds", description="Protocol format: 'rounds' (legacy) or 'stages' (new)")


class ProtocolListResponse(BaseModel):
    """Response for listing protocols."""
    
    protocols: list[ProtocolSummary]
    count: int


class ProtocolExecutionConfig(BaseModel):
    """Configuration for protocol execution."""
    
    max_feedback_loops: int = Field(default=3, ge=0, le=10)
    allow_feedback: bool = Field(default=False)
    run_cross_reference: bool = Field(default=True)
    show_infra_status: bool = Field(default=False)


class ProtocolRunRequest(BaseModel):
    """Request to execute a protocol."""
    
    inputs: dict[str, Any] = Field(..., description="Protocol inputs (topic, documents, etc.)")
    brigade_override: dict[str, str] | None = Field(
        default=None,
        description="Override model assignments per role"
    )
    config: ProtocolExecutionConfig = Field(
        default_factory=ProtocolExecutionConfig,
        description="Execution configuration"
    )
    taxonomy_paths: list[str] | None = Field(
        default=None,
        description="Paths to taxonomy JSON files for filtering"
    )


class CrossReferenceEvidence(BaseModel):
    """Cross-reference evidence from Stage 2 retrieval."""
    
    query: str = Field(..., description="Query that was searched")
    qdrant: list[dict[str, Any]] = Field(default_factory=list, description="Qdrant hybrid search results")
    neo4j: list[dict[str, Any]] = Field(default_factory=list, description="Neo4j graph query results")
    textbooks: list[dict[str, Any]] = Field(default_factory=list, description="Textbook JSON search results")
    code_orchestrator: list[dict[str, Any]] = Field(default_factory=list, description="Code-Orchestrator ML results")
    code_reference: list[dict[str, Any]] = Field(default_factory=list, description="External repo implementations")


class AuditResult(BaseModel):
    """Audit validation result from soft/hard audit."""
    
    audit_type: str = Field(..., description="Type: 'soft' (Code-Orchestrator) or 'hard' (audit-service)")
    passed: bool = Field(..., description="Whether audit passed")
    score: float = Field(default=0.0, description="Audit confidence score")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="Audit findings")
    citations_validated: int = Field(default=0, description="Number of citations validated")
    citations_failed: int = Field(default=0, description="Number of citations that failed validation")


class CitationEntry(BaseModel):
    """Single citation extracted from LLM output."""
    
    marker: str = Field(..., description="Citation marker (e.g., '[^1]')")
    source_type: str = Field(..., description="Type: 'book', 'code', 'textbook', 'web'")
    source_id: str = Field(..., description="Source identifier (book title, repo name)")
    chapter: str | None = Field(default=None, description="Chapter reference")
    page: int | None = Field(default=None, description="Page number")
    content_claim: str = Field(default="", description="Content claim (truncated)")
    relevance_score: float = Field(default=0.0, description="Similarity score (0-1)")
    confirmed: bool = Field(default=False, description="Whether citation was validated")
    confirmed_by: str = Field(default="", description="Validation source")
    round_num: int = Field(default=0, description="Round where citation appeared")
    role: str = Field(default="", description="LLM role that generated citation")


class CitationCacheSummary(BaseModel):
    """Summary of citation cache state."""
    
    session_id: str = Field(..., description="Session ID for this cache")
    confirmed: list[CitationEntry] = Field(default_factory=list, description="Confirmed citations")
    pending: list[CitationEntry] = Field(default_factory=list, description="Pending citations")
    summary: dict[str, int] = Field(
        default_factory=lambda: {"confirmed_count": 0, "pending_count": 0, "total_markers_seen": 0},
        description="Count summary"
    )


class ProtocolRunResponse(BaseModel):
    """Response from protocol execution."""
    
    execution_id: str = Field(..., description="Unique execution ID")
    protocol_id: str = Field(..., description="Protocol that was executed")
    status: str = Field(..., description="Execution status: completed, partial, failed")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Round outputs")
    cross_reference_evidence: dict[str, CrossReferenceEvidence] | None = Field(
        default=None,
        description="Cross-reference evidence from Stage 2 retrieval (keyed by query)"
    )
    audit_results: list[AuditResult] | None = Field(
        default=None,
        description="Audit validation results (soft + hard audits)"
    )
    citation_cache: CitationCacheSummary | None = Field(
        default=None,
        description="Citation cache with confirmed/pending citations (Option A inline validation)"
    )
    trace_id: str = Field(..., description="Trace ID for debugging")
    feedback_loops_used: int = Field(default=0, description="Feedback loops consumed")
    needs_more_work: list[dict[str, Any]] | None = Field(
        default=None,
        description="Feedback needs if any"
    )
    execution_time_ms: int = Field(default=0, description="Execution time in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")


# =============================================================================
# Protocol Loading Utilities
# =============================================================================


def _load_protocol_file(protocol_id: str) -> dict[str, Any] | None:
    """Load a protocol definition from JSON file.
    
    Args:
        protocol_id: Protocol identifier (filename without .json)
        
    Returns:
        Protocol definition dict or None if not found
    """
    protocol_file = PROTOCOLS_DIR / f"{protocol_id}.json"
    if not protocol_file.exists():
        return None
    
    try:
        return json.loads(protocol_file.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse protocol {protocol_id}: {e}")
        return None


def _list_available_protocols() -> list[dict[str, Any]]:
    """List all available protocol definitions.
    
    Returns:
        List of protocol definition dicts
    """
    protocols = []
    
    if not PROTOCOLS_DIR.exists():
        logger.warning(f"Protocols directory not found: {PROTOCOLS_DIR}")
        return protocols
    
    for protocol_file in PROTOCOLS_DIR.glob("*.json"):
        try:
            protocol_data = json.loads(protocol_file.read_text())
            protocols.append(protocol_data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {protocol_file}: {e}")
            continue
    
    return protocols


def _protocol_to_summary(protocol: dict[str, Any]) -> ProtocolSummary:
    """Convert protocol dict to summary model."""
    brigade_roles = list(protocol.get("brigade_roles", {}).keys())
    return ProtocolSummary(
        protocol_id=protocol.get("protocol_id", ""),
        name=protocol.get("name", ""),
        description=protocol.get("description", ""),
        brigade_roles=brigade_roles,
    )


def _protocol_to_detail(protocol: dict[str, Any]) -> ProtocolDetail:
    """Convert protocol dict to detail model.
    
    Handles two protocol formats:
    - Legacy "rounds" format: Simple sequential/parallel rounds
    - New "stages" format: 4-stage Kitchen Brigade flow with cross-reference
    """
    brigade_roles = {}
    for role_name, role_config in protocol.get("brigade_roles", {}).items():
        brigade_roles[role_name] = BrigadeRoleSummary(
            model=role_config.get("model", ""),
            temperature=role_config.get("temperature", 0.3),
            max_tokens=role_config.get("max_tokens", 4096),
            system_prompt=role_config.get("system_prompt", ""),
        )
    
    # Determine format: "stages" (new) or "rounds" (legacy)
    protocol_format = "stages" if "stages" in protocol else "rounds"
    
    # Parse legacy rounds format
    rounds = []
    for round_def in protocol.get("rounds", []):
        rounds.append(RoundDefinition(
            round=round_def.get("round", 0),
            type=round_def.get("type", "parallel"),
            name=round_def.get("name", ""),
            description=round_def.get("description", ""),
            roles=round_def.get("roles", []),
        ))
    
    # Parse new stages format (Kitchen Brigade 4-stage flow)
    stages = []
    for stage_def in protocol.get("stages", []):
        stages.append(StageDefinition(
            stage=stage_def.get("stage", 0),
            name=stage_def.get("name", ""),
            type=stage_def.get("type", ""),
            description=stage_def.get("description", ""),
            layers=stage_def.get("layers"),
            max_iterations=stage_def.get("max_iterations"),
            can_request_more_info=stage_def.get("can_request_more_info"),
        ))
    
    return ProtocolDetail(
        protocol_id=protocol.get("protocol_id", ""),
        name=protocol.get("name", ""),
        description=protocol.get("description", ""),
        brigade_roles=brigade_roles,
        rounds=rounds,
        stages=stages,
        format=protocol_format,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=ProtocolListResponse)
async def list_protocols() -> ProtocolListResponse:
    """List all available Kitchen Brigade protocols.
    
    Returns summary information for each protocol including
    ID, name, description, and available roles.
    """
    protocols = _list_available_protocols()
    summaries = [_protocol_to_summary(p) for p in protocols]
    
    logger.info(f"Listed {len(summaries)} protocols")
    
    return ProtocolListResponse(
        protocols=summaries,
        count=len(summaries),
    )


@router.get("/{protocol_id}", response_model=ProtocolDetail)
async def get_protocol(protocol_id: str) -> ProtocolDetail:
    """Get detailed information about a specific protocol.
    
    Args:
        protocol_id: Protocol identifier (e.g., ROUNDTABLE_DISCUSSION)
        
    Returns:
        Full protocol definition including brigade roles and rounds
    """
    protocol = _load_protocol_file(protocol_id)
    
    if not protocol:
        logger.warning(f"Protocol not found: {protocol_id}")
        raise HTTPException(status_code=404, detail=ERROR_PROTOCOL_NOT_FOUND)
    
    return _protocol_to_detail(protocol)


@router.post("/{protocol_id}/run", response_model=ProtocolRunResponse)
async def run_protocol(
    protocol_id: str,
    request: ProtocolRunRequest,
) -> ProtocolRunResponse:
    """Execute a Kitchen Brigade protocol.
    
    Runs the specified protocol with provided inputs, optionally with:
    - Custom model assignments (brigade_override)
    - Cross-reference retrieval (run_cross_reference)
    - Feedback loops (max_feedback_loops)
    
    Args:
        protocol_id: Protocol identifier
        request: Execution request with inputs and configuration
        
    Returns:
        Execution results including outputs, trace, and status
    """
    execution_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Validate protocol exists
    protocol = _load_protocol_file(protocol_id)
    if not protocol:
        logger.warning(f"Protocol not found: {protocol_id}")
        raise HTTPException(status_code=404, detail=ERROR_PROTOCOL_NOT_FOUND)
    
    # Load taxonomy filter if specified
    taxonomy_filter = None
    if request.taxonomy_paths:
        taxonomy_filter = TaxonomyFilter()
        for path in request.taxonomy_paths:
            try:
                taxonomy_filter.load(path)
            except Exception as e:
                logger.warning(f"Failed to load taxonomy {path}: {e}")
    
    try:
        # Create executor
        executor = KitchenBrigadeExecutor(
            protocol_id=protocol_id,
            inputs=request.inputs,
            brigade_override=request.brigade_override,
            enable_cross_reference=request.config.run_cross_reference,
            show_infra_status=request.config.show_infra_status,
            taxonomy_filter=taxonomy_filter,
        )
        
        # Execute protocol
        result = await executor.execute(
            max_feedback_loops=request.config.max_feedback_loops,
            allow_feedback=request.config.allow_feedback,
            run_cross_reference=request.config.run_cross_reference,
        )
        
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Generate trace ID from trace data
        trace_id = result.get("trace", {}).get("start_time", execution_id)
        
        # Extract cross-reference evidence for response
        xref_evidence = result.get("cross_reference_evidence")
        formatted_xref = None
        if xref_evidence:
            formatted_xref = {
                query: CrossReferenceEvidence(
                    query=query,
                    qdrant=evidence.get("qdrant", []),
                    neo4j=evidence.get("neo4j", []),
                    textbooks=evidence.get("textbooks", []),
                    code_orchestrator=evidence.get("code_orchestrator", []),
                    code_reference=evidence.get("code_reference", []),
                )
                for query, evidence in xref_evidence.items()
                if isinstance(evidence, dict)  # Skip non-dict entries like "file_requests"
            }
        
        # Extract audit results
        audit_results = result.get("audit_results")
        formatted_audits = None
        if audit_results:
            formatted_audits = [
                AuditResult(
                    audit_type=ar.get("audit_type", "unknown"),
                    passed=ar.get("passed", False),
                    score=ar.get("score", 0.0),
                    findings=ar.get("findings", []),
                    citations_validated=ar.get("citations_validated", 0),
                    citations_failed=ar.get("citations_failed", 0),
                )
                for ar in audit_results
            ]
        
        # Extract citation cache data
        citation_cache_data = result.get("citation_cache")
        formatted_citation_cache = None
        if citation_cache_data:
            formatted_citation_cache = CitationCacheSummary(
                session_id=citation_cache_data.get("session_id", ""),
                confirmed=[
                    CitationEntry(**c) for c in citation_cache_data.get("confirmed", [])
                ],
                pending=[
                    CitationEntry(**c) for c in citation_cache_data.get("pending", [])
                ],
                summary=citation_cache_data.get("summary", {}),
            )
        
        logger.info(
            f"Protocol {protocol_id} executed - execution_id={execution_id}, "
            f"time_ms={execution_time_ms}, feedback_loops={result.get('feedback_loops_used', 0)}"
        )
        
        return ProtocolRunResponse(
            execution_id=execution_id,
            protocol_id=protocol_id,
            status="completed",
            outputs=result.get("outputs", {}),
            cross_reference_evidence=formatted_xref,
            audit_results=formatted_audits,
            citation_cache=formatted_citation_cache,
            trace_id=trace_id,
            feedback_loops_used=result.get("feedback_loops_used", 0),
            needs_more_work=result.get("needs_more_work"),
            execution_time_ms=execution_time_ms,
        )
        
    except FileNotFoundError as e:
        logger.error(f"Protocol file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
        
    except Exception as e:
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.error(f"Protocol execution failed: {e}", exc_info=True)
        
        return ProtocolRunResponse(
            execution_id=execution_id,
            protocol_id=protocol_id,
            status="failed",
            outputs={},
            trace_id=execution_id,
            execution_time_ms=execution_time_ms,
            error=str(e),
        )
