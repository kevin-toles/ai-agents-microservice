"""Integration Test Fixtures.

WBS Reference: WBS-AGT20 Integration Testing (AGT20.7)
Pattern: Shared fixtures for integration tests
Anti-Pattern Avoided: Fixture reuse without explicit scope

Provides:
- FakeClient implementations for unit test isolation
- Real client configurations for integration tests
- Shared test data and fixtures
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from src.core.config import Settings


# =============================================================================
# Environment Configuration
# =============================================================================

# Integration test flags
LIVE_INFERENCE = os.getenv("LIVE_INFERENCE", "false").lower() == "true"
LIVE_SEMANTIC_SEARCH = os.getenv("LIVE_SEMANTIC_SEARCH", "false").lower() == "true"
LIVE_AUDIT = os.getenv("LIVE_AUDIT", "false").lower() == "true"
LIVE_ALL_SERVICES = os.getenv("LIVE_ALL_SERVICES", "false").lower() == "true"

# Service URLs from environment or defaults
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8085")
SEMANTIC_SEARCH_URL = os.getenv("SEMANTIC_SEARCH_URL", "http://localhost:8081")
AUDIT_SERVICE_URL = os.getenv("AUDIT_SERVICE_URL", "http://localhost:8084")
AI_AGENTS_URL = os.getenv("AI_AGENTS_URL", "http://localhost:8082")


# =============================================================================
# Test Settings Fixtures
# =============================================================================

@pytest.fixture
def integration_settings() -> Settings:
    """Create settings configured for integration testing."""
    return Settings(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        llm_gateway_url=os.getenv("LLM_GATEWAY_URL", "http://localhost:8080"),
        inference_service_url=INFERENCE_SERVICE_URL,
        semantic_search_url=SEMANTIC_SEARCH_URL,
        audit_service_url=AUDIT_SERVICE_URL,
        debug=True,
        log_level="DEBUG",
    )


# =============================================================================
# HTTP Client Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for integration tests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture
async def inference_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for inference-service."""
    async with httpx.AsyncClient(
        base_url=INFERENCE_SERVICE_URL,
        timeout=60.0,  # LLM calls can be slow
    ) as client:
        yield client


@pytest_asyncio.fixture
async def semantic_search_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for semantic-search-service."""
    async with httpx.AsyncClient(
        base_url=SEMANTIC_SEARCH_URL,
        timeout=30.0,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def audit_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for audit-service."""
    async with httpx.AsyncClient(
        base_url=AUDIT_SERVICE_URL,
        timeout=30.0,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def ai_agents_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for ai-agents service."""
    async with httpx.AsyncClient(
        base_url=AI_AGENTS_URL,
        timeout=60.0,
    ) as client:
        yield client


# =============================================================================
# Service Health Check Helpers
# =============================================================================

async def check_service_health(client: httpx.AsyncClient, url: str) -> bool:
    """Check if a service is healthy.
    
    Args:
        client: HTTP client to use
        url: Health endpoint URL
        
    Returns:
        True if service is healthy, False otherwise
    """
    try:
        response = await client.get(url)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


@pytest_asyncio.fixture
async def ensure_inference_service(
    inference_client: httpx.AsyncClient,
) -> httpx.AsyncClient:
    """Ensure inference-service is available, skip if not."""
    if not await check_service_health(inference_client, "/health"):
        pytest.skip("inference-service not available")
    return inference_client


@pytest_asyncio.fixture
async def ensure_semantic_search(
    semantic_search_client: httpx.AsyncClient,
) -> httpx.AsyncClient:
    """Ensure semantic-search-service is available, skip if not."""
    if not await check_service_health(semantic_search_client, "/health"):
        pytest.skip("semantic-search-service not available")
    return semantic_search_client


@pytest_asyncio.fixture
async def ensure_audit_service(
    audit_client: httpx.AsyncClient,
) -> httpx.AsyncClient:
    """Ensure audit-service is available, skip if not."""
    if not await check_service_health(audit_client, "/health"):
        pytest.skip("audit-service not available")
    return audit_client


@pytest_asyncio.fixture
async def ensure_ai_agents(
    ai_agents_client: httpx.AsyncClient,
) -> httpx.AsyncClient:
    """Ensure ai-agents service is available, skip if not."""
    if not await check_service_health(ai_agents_client, "/health"):
        pytest.skip("ai-agents service not available")
    return ai_agents_client


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_function_input() -> dict[str, Any]:
    """Sample input for function e2e tests."""
    return {
        "content": """
        Software design is fundamentally about managing complexity.
        Good modules hide implementation details behind simple interfaces.
        """,
        "artifact_type": "markdown",
    }


@pytest.fixture
def sample_pipeline_input() -> dict[str, Any]:
    """Sample input for pipeline e2e tests."""
    return {
        "book": "A Philosophy of Software Design",
        "chapter": 1,
        "title": "Introduction",
        "content": """
        The most fundamental problem in computer science is problem decomposition:
        how to take a complex problem and divide it up into pieces that can be 
        solved independently.
        """,
        "preset": "standard",
    }


@pytest.fixture
def sample_cross_reference_input() -> dict[str, Any]:
    """Sample input for cross-reference function tests."""
    return {
        "query": "complexity management in software design",
        "max_results": 5,
        "min_score": 0.7,
    }


# =============================================================================
# Mock Service Fixtures (for unit test isolation)
# =============================================================================

@pytest.fixture
def mock_inference_response() -> dict[str, Any]:
    """Mock response from inference-service."""
    return {
        "response": "This is a mock LLM response for testing.",
        "model": "granite-8b-code-128k",
        "tokens_used": 150,
        "finish_reason": "stop",
    }


@pytest.fixture
def mock_semantic_search_response() -> dict[str, Any]:
    """Mock response from semantic-search-service."""
    return {
        "results": [
            {
                "content": "Related content about software design",
                "source": "APOSD Ch.1",
                "score": 0.92,
            },
            {
                "content": "Information hiding principles",
                "source": "APOSD Ch.2",
                "score": 0.87,
            },
        ],
        "total_results": 2,
    }


@pytest.fixture
def mock_audit_response() -> dict[str, Any]:
    """Mock response from audit-service."""
    return {
        "audit_id": "audit-12345",
        "status": "recorded",
        "timestamp": "2025-12-30T12:00:00Z",
    }


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def load_test_config() -> dict[str, Any]:
    """Configuration for load tests."""
    return {
        "concurrent_requests": 5,
        "timeout_seconds": 60,
        "warmup_requests": 1,
    }
