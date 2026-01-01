"""Health check API routes.

WBS-AGT18: API Routes - AC-18.3: GET /health returns service status.

Provides REST API endpoints for health monitoring and readiness checks.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Reference: llm-gateway health.py patterns

Anti-Patterns Avoided:
- ANTI_PATTERN_ANALYSIS §3.1: No bare except clauses
- ANTI_PATTERN_ANALYSIS §4.1: Cognitive complexity < 15 per function
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# =============================================================================
# Enums and Constants
# =============================================================================

class HealthStatus(str, Enum):
    """Health status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyStatus(str, Enum):
    """Dependency status enum."""

    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


# Service metadata
SERVICE_NAME = "ai-agents"
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "0.1.0")
SERVICE_PORT = 8082


# =============================================================================
# Response Models
# =============================================================================

class DependencyHealth(BaseModel):
    """Health status for a single dependency.

    Attributes:
        name: Dependency name
        status: Current status
        latency_ms: Response latency in milliseconds
        message: Optional status message
    """

    name: str = Field(..., description="Dependency name")
    status: DependencyStatus = Field(..., description="Current status")
    latency_ms: float | None = Field(
        default=None,
        description="Response latency in milliseconds",
    )
    message: str | None = Field(
        default=None,
        description="Optional status message",
    )


class HealthResponse(BaseModel):
    """Response model for health check.

    Attributes:
        status: Overall service status
        service: Service name
        version: Service version
        timestamp: Check timestamp (ISO format)
        uptime_seconds: Service uptime in seconds
        dependencies: Health status of dependencies
    """

    status: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Overall service status",
    )
    service: str = Field(
        default=SERVICE_NAME,
        description="Service name",
    )
    version: str = Field(
        default=SERVICE_VERSION,
        description="Service version",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Check timestamp",
    )
    uptime_seconds: float | None = Field(
        default=None,
        description="Service uptime in seconds",
    )
    dependencies: list[DependencyHealth] = Field(
        default_factory=list,
        description="Health status of dependencies",
    )


class ReadinessResponse(BaseModel):
    """Response model for readiness check.

    Attributes:
        ready: Whether service is ready to accept traffic
        checks: Individual check results
    """

    ready: bool = Field(
        default=True,
        description="Whether service is ready",
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual check results",
    )


class LivenessResponse(BaseModel):
    """Response model for liveness check.

    Attributes:
        alive: Whether service is alive
        timestamp: Check timestamp
    """

    alive: bool = Field(
        default=True,
        description="Whether service is alive",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Check timestamp",
    )


# =============================================================================
# Service Start Time
# =============================================================================

_service_start_time: datetime | None = None


def set_service_start_time(start_time: datetime | None = None) -> None:
    """Set the service start time for uptime calculation.

    Args:
        start_time: Service start time, defaults to now
    """
    global _service_start_time
    _service_start_time = start_time or datetime.now(UTC)


def get_uptime_seconds() -> float | None:
    """Get service uptime in seconds.

    Returns:
        Uptime in seconds, or None if start time not set
    """
    if _service_start_time is None:
        return None

    delta = datetime.now(UTC) - _service_start_time
    return delta.total_seconds()


# =============================================================================
# Dependency Checks
# =============================================================================

import httpx

# Dependency URLs (configurable via env vars)
# Use container names for Docker network communication
# Reference: ai-platform-data/docs/NETWORK_ARCHITECTURE.md
LLM_GATEWAY_URL = os.environ.get("AI_AGENTS_LLM_GATEWAY_URL", "http://llm-gateway-standalone:8080")
SEMANTIC_SEARCH_URL = os.environ.get("SEMANTIC_SEARCH_URL", "http://semantic-search-service:8081")
NEO4J_URL = os.environ.get("AI_AGENTS_NEO4J_URI", "bolt://neo4j:7687")


async def check_llm_gateway() -> DependencyHealth:
    """Check LLM Gateway connectivity.

    Returns:
        DependencyHealth for llm-gateway
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            start = datetime.now(UTC)
            response = await client.get(f"{LLM_GATEWAY_URL}/health")
            latency = (datetime.now(UTC) - start).total_seconds() * 1000
            
            if response.status_code == 200:
                return DependencyHealth(
                    name="llm-gateway",
                    status=DependencyStatus.UP,
                    latency_ms=latency,
                    message="OK",
                )
            return DependencyHealth(
                name="llm-gateway",
                status=DependencyStatus.DOWN,
                latency_ms=latency,
                message=f"HTTP {response.status_code}",
            )
    except httpx.ConnectError:
        return DependencyHealth(
            name="llm-gateway",
            status=DependencyStatus.DOWN,
            message="Connection refused",
        )
    except Exception as e:
        return DependencyHealth(
            name="llm-gateway",
            status=DependencyStatus.DOWN,
            message=str(e)[:50],
        )


async def check_semantic_search() -> DependencyHealth:
    """Check Semantic Search Service connectivity.

    Returns:
        DependencyHealth for semantic-search-service
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            start = datetime.now(UTC)
            response = await client.get(f"{SEMANTIC_SEARCH_URL}/health")
            latency = (datetime.now(UTC) - start).total_seconds() * 1000
            
            if response.status_code == 200:
                return DependencyHealth(
                    name="semantic-search-service",
                    status=DependencyStatus.UP,
                    latency_ms=latency,
                    message="OK",
                )
            return DependencyHealth(
                name="semantic-search-service",
                status=DependencyStatus.DOWN,
                latency_ms=latency,
                message=f"HTTP {response.status_code}",
            )
    except httpx.ConnectError:
        return DependencyHealth(
            name="semantic-search-service",
            status=DependencyStatus.DOWN,
            message="Connection refused",
        )
    except Exception as e:
        return DependencyHealth(
            name="semantic-search-service",
            status=DependencyStatus.DOWN,
            message=str(e)[:50],
        )


async def check_neo4j() -> DependencyHealth:
    """Check Neo4j connectivity.

    Returns:
        DependencyHealth for neo4j

    Note: Neo4j check is optional - returns UP if not configured
    """
    # Neo4j is optional for basic operation
    # TODO: Implement actual Neo4j driver ping if neo4j is configured
    if NEO4J_URL == "bolt://localhost:7687":
        return DependencyHealth(
            name="neo4j",
            status=DependencyStatus.UP,
            message="Optional (not configured)",
        )
    return DependencyHealth(
        name="neo4j",
        status=DependencyStatus.UNKNOWN,
        message="Neo4j check not implemented",
    )


async def get_all_dependency_checks() -> list[DependencyHealth]:
    """Run all dependency health checks.

    Returns:
        List of DependencyHealth objects
    """
    dependencies = []

    dependencies.append(await check_llm_gateway())
    dependencies.append(await check_semantic_search())
    dependencies.append(await check_neo4j())

    return dependencies


def calculate_overall_status(dependencies: list[DependencyHealth]) -> HealthStatus:
    """Calculate overall health status from dependencies.

    Args:
        dependencies: List of dependency health checks

    Returns:
        Overall HealthStatus
    """
    if not dependencies:
        return HealthStatus.HEALTHY

    down_count = sum(
        1 for d in dependencies
        if d.status == DependencyStatus.DOWN
    )
    unknown_count = sum(
        1 for d in dependencies
        if d.status == DependencyStatus.UNKNOWN
    )

    if down_count > 0:
        return HealthStatus.UNHEALTHY
    if unknown_count > 0:
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


# =============================================================================
# API Endpoints
# =============================================================================

@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns comprehensive health status of the service.",
)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint.

    Returns:
        HealthResponse with status, version, uptime, and dependencies
    """
    dependencies = await get_all_dependency_checks()
    status = calculate_overall_status(dependencies)

    return HealthResponse(
        status=status,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=get_uptime_seconds(),
        dependencies=dependencies,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns whether the service is ready to accept traffic.",
)
async def readiness_check() -> ReadinessResponse:
    """Kubernetes-style readiness probe.

    Returns:
        ReadinessResponse indicating if service is ready
    """
    # Check critical dependencies
    llm_gateway = await check_llm_gateway()

    checks = {
        "llm_gateway": llm_gateway.status != DependencyStatus.DOWN,
        "functions_loaded": True,  # Always true after startup
        "pipelines_loaded": True,  # Always true after startup
    }

    ready = all(checks.values())

    return ReadinessResponse(
        ready=ready,
        checks=checks,
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness check",
    description="Returns whether the service is alive.",
)
async def liveness_check() -> LivenessResponse:
    """Kubernetes-style liveness probe.

    Returns:
        LivenessResponse indicating service is alive
    """
    return LivenessResponse(
        alive=True,
        timestamp=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "DependencyHealth",
    "DependencyStatus",
    "HealthResponse",
    "HealthStatus",
    "LivenessResponse",
    "ReadinessResponse",
    "calculate_overall_status",
    "check_llm_gateway",
    "check_neo4j",
    "check_semantic_search",
    "get_all_dependency_checks",
    "get_uptime_seconds",
    "router",
    "set_service_start_time",
]
