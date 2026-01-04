"""E2E Test Configuration and Fixtures.

WBS-PI7: End-to-End Protocol Testing

Provides fixtures for E2E testing with various feature flag configurations.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Testing Strategy
Anti-Patterns: #12 (Connection Pooling), #25 (Test Isolation)
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING, AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    pass


def _create_fresh_app_with_env(env_vars: dict) -> "FastAPI":
    """Create a fresh FastAPI app with specific environment variables.
    
    This is necessary because the app/settings are typically loaded once
    at module import time. We need to reload them with new env vars.
    """
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Clear cached modules to force reimport with new env
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("src.")]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import fresh app
    from src.main import app
    
    return app


# -----------------------------------------------------------------------------
# Environment Variable Configurations
# -----------------------------------------------------------------------------


ALL_DISABLED_ENV = {
    "AGENTS_A2A_ENABLED": "false",
    "AGENTS_A2A_AGENT_CARD_ENABLED": "false",
    "AGENTS_A2A_TASK_LIFECYCLE_ENABLED": "false",
    "AGENTS_A2A_STREAMING_ENABLED": "false",
    "AGENTS_A2A_PUSH_NOTIFICATIONS": "false",
    "AGENTS_MCP_ENABLED": "false",
    "AGENTS_MCP_SERVER_ENABLED": "false",
    "AGENTS_MCP_CLIENT_ENABLED": "false",
    "AGENTS_MCP_SEMANTIC_SEARCH": "false",
    "AGENTS_MCP_TOOLBOX_NEO4J": "false",
    "AGENTS_MCP_TOOLBOX_REDIS": "false",
}

ALL_ENABLED_ENV = {
    "AGENTS_A2A_ENABLED": "true",
    "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
    "AGENTS_A2A_TASK_LIFECYCLE_ENABLED": "true",
    "AGENTS_A2A_STREAMING_ENABLED": "true",
    "AGENTS_A2A_PUSH_NOTIFICATIONS": "true",
    "AGENTS_MCP_ENABLED": "true",
    "AGENTS_MCP_SERVER_ENABLED": "true",
    "AGENTS_MCP_CLIENT_ENABLED": "true",
    "AGENTS_MCP_SEMANTIC_SEARCH": "true",
    "AGENTS_MCP_TOOLBOX_NEO4J": "true",
    "AGENTS_MCP_TOOLBOX_REDIS": "true",
}

A2A_ONLY_ENV = {
    "AGENTS_A2A_ENABLED": "true",
    "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
    "AGENTS_A2A_TASK_LIFECYCLE_ENABLED": "true",
    "AGENTS_A2A_STREAMING_ENABLED": "true",
    "AGENTS_A2A_PUSH_NOTIFICATIONS": "false",
    "AGENTS_MCP_ENABLED": "false",
    "AGENTS_MCP_SERVER_ENABLED": "false",
    "AGENTS_MCP_CLIENT_ENABLED": "false",
    "AGENTS_MCP_SEMANTIC_SEARCH": "false",
    "AGENTS_MCP_TOOLBOX_NEO4J": "false",
    "AGENTS_MCP_TOOLBOX_REDIS": "false",
}

MCP_ONLY_ENV = {
    "AGENTS_A2A_ENABLED": "false",
    "AGENTS_A2A_AGENT_CARD_ENABLED": "false",
    "AGENTS_A2A_TASK_LIFECYCLE_ENABLED": "false",
    "AGENTS_A2A_STREAMING_ENABLED": "false",
    "AGENTS_A2A_PUSH_NOTIFICATIONS": "false",
    "AGENTS_MCP_ENABLED": "true",
    "AGENTS_MCP_SERVER_ENABLED": "true",
    "AGENTS_MCP_CLIENT_ENABLED": "true",
    "AGENTS_MCP_SEMANTIC_SEARCH": "true",
    "AGENTS_MCP_TOOLBOX_NEO4J": "false",
    "AGENTS_MCP_TOOLBOX_REDIS": "false",
}


# -----------------------------------------------------------------------------
# Test Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_client_disabled() -> TestClient:
    """Test client with all protocols disabled."""
    app = _create_fresh_app_with_env(ALL_DISABLED_ENV)
    return TestClient(app)


@pytest.fixture
def test_client_enabled() -> TestClient:
    """Test client with all protocols enabled."""
    app = _create_fresh_app_with_env(ALL_ENABLED_ENV)
    return TestClient(app)


@pytest.fixture
def test_client_a2a_only() -> TestClient:
    """Test client with only A2A enabled."""
    app = _create_fresh_app_with_env(A2A_ONLY_ENV)
    return TestClient(app)


@pytest.fixture
def test_client_mcp_only() -> TestClient:
    """Test client with only MCP enabled."""
    app = _create_fresh_app_with_env(MCP_ONLY_ENV)
    return TestClient(app)


# -----------------------------------------------------------------------------
# Async Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def async_client_enabled() -> AsyncGenerator[AsyncClient, None]:
    """Async test client with all protocols enabled."""
    app = _create_fresh_app_with_env(ALL_ENABLED_ENV)
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
async def async_client_disabled() -> AsyncGenerator[AsyncClient, None]:
    """Async test client with all protocols disabled."""
    app = _create_fresh_app_with_env(ALL_DISABLED_ENV)
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# -----------------------------------------------------------------------------
# Test Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_a2a_message() -> dict:
    """Sample A2A message for testing."""
    return {
        "message": {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": "Extract keywords from this text: Python is a programming language.",
                }
            ],
        },
        "skill_id": "extract_structure",
    }


@pytest.fixture
def sample_extract_request() -> dict:
    """Sample Phase 1 extract-structure request."""
    return {
        "input": {
            "content": "# Introduction\n\nPython is a versatile programming language.",
            "extraction_type": "outline",
        }
    }
