"""Integration tests for Phase 1 endpoint stability (AC-PI1.8).

TDD Phase: RED → GREEN → REFACTOR

Reference: WBS_PROTOCOL_INTEGRATION.md → WBS-PI1 → AC-PI1.8
Purpose: Ensure Phase 1 endpoints work regardless of feature flag state.

This test validates that all existing Phase 1 REST API endpoints continue
to function correctly when Phase 2 protocol feature flags are toggled.

Anti-Patterns Avoided:
- CODING_PATTERNS_ANALYSIS §1: Full type annotations
- CODING_PATTERNS_ANALYSIS §3: No bare except clauses
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_protocol_env() -> Generator[None, None, None]:
    """Remove all AGENTS_ protocol environment variables."""
    env_vars_to_remove = [k for k in os.environ if k.startswith("AGENTS_")]
    original_values = {k: os.environ.pop(k) for k in env_vars_to_remove}
    yield
    os.environ.update(original_values)


@pytest.fixture
def all_protocols_disabled_env(
    clean_protocol_env: None,
) -> Generator[None, None, None]:
    """Ensure all protocol flags are explicitly disabled."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_A2A_ENABLED": "false",
            "AGENTS_MCP_ENABLED": "false",
        },
    ):
        yield


@pytest.fixture
def all_protocols_enabled_env(
    clean_protocol_env: None,
) -> Generator[None, None, None]:
    """Enable all protocol flags to simulate full Phase 2 deployment."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_A2A_ENABLED": "true",
            "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
            "AGENTS_A2A_STREAMING_ENABLED": "true",
            "AGENTS_A2A_PUSH_NOTIFICATIONS": "true",
            "AGENTS_MCP_ENABLED": "true",
            "AGENTS_MCP_SERVER_ENABLED": "true",
            "AGENTS_MCP_CLIENT_ENABLED": "true",
            "AGENTS_MCP_SEMANTIC_SEARCH": "true",
            "AGENTS_MCP_TOOLBOX_NEO4J": "true",
            "AGENTS_MCP_TOOLBOX_REDIS": "true",
        },
    ):
        yield


@pytest.fixture
def partial_protocols_enabled_env(
    clean_protocol_env: None,
) -> Generator[None, None, None]:
    """Enable only A2A, disable MCP."""
    with patch.dict(
        os.environ,
        {
            "AGENTS_A2A_ENABLED": "true",
            "AGENTS_A2A_AGENT_CARD_ENABLED": "true",
            "AGENTS_MCP_ENABLED": "false",
        },
    ):
        yield


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for FastAPI application."""
    from src.main import app

    return TestClient(app)


# =============================================================================
# AC-PI1.8: Phase 1 endpoints work regardless of flag state
# =============================================================================


class TestPhase1EndpointsWithProtocolsDisabled:
    """Test Phase 1 endpoints with all protocols disabled (default state)."""

    def test_health_endpoint_works(
        self,
        test_client: TestClient,
        all_protocols_disabled_env: None,
    ) -> None:
        """Health endpoint returns 200 with protocols disabled."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_functions_list_endpoint_works(
        self,
        test_client: TestClient,
        all_protocols_disabled_env: None,
    ) -> None:
        """GET /v1/functions returns 200 with protocols disabled."""
        response = test_client.get("/v1/functions")
        assert response.status_code == 200
        # Verify it returns function list in response
        data = response.json()
        assert isinstance(data, dict)
        assert "functions" in data
        functions = data["functions"]
        assert isinstance(functions, list)
        # Should include Phase 1 functions
        if functions:
            function_names = [f["name"] for f in functions]
            assert "extract-structure" in function_names


class TestPhase1EndpointsWithProtocolsEnabled:
    """Test Phase 1 endpoints with all protocols enabled."""

    def test_health_endpoint_works(
        self,
        test_client: TestClient,
        all_protocols_enabled_env: None,
    ) -> None:
        """Health endpoint returns 200 even with all protocols enabled."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_functions_list_endpoint_works(
        self,
        test_client: TestClient,
        all_protocols_enabled_env: None,
    ) -> None:
        """GET /v1/functions returns 200 even with all protocols enabled."""
        response = test_client.get("/v1/functions")
        assert response.status_code == 200


class TestPhase1EndpointsWithPartialProtocols:
    """Test Phase 1 endpoints with partial protocol enablement."""

    def test_health_endpoint_works(
        self,
        test_client: TestClient,
        partial_protocols_enabled_env: None,
    ) -> None:
        """Health endpoint returns 200 with partial protocols enabled."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_functions_list_endpoint_works(
        self,
        test_client: TestClient,
        partial_protocols_enabled_env: None,
    ) -> None:
        """GET /v1/functions returns 200 with partial protocols enabled."""
        response = test_client.get("/v1/functions")
        assert response.status_code == 200


class TestFeatureFlagsIntegration:
    """Test that feature flags are properly loaded in the application."""

    def test_feature_flags_available_in_app(
        self,
        test_client: TestClient,
        all_protocols_disabled_env: None,
    ) -> None:
        """Feature flags can be accessed via dependency injection."""
        from src.config.feature_flags import get_feature_flags

        # Clear cache to pick up environment changes
        get_feature_flags.cache_clear()

        flags = get_feature_flags()
        assert flags.a2a_enabled is False
        assert flags.mcp_enabled is False

    def test_feature_flags_respect_env_vars(
        self,
        test_client: TestClient,
        all_protocols_enabled_env: None,
    ) -> None:
        """Feature flags respect environment variable values."""
        from src.config.feature_flags import get_feature_flags

        # Clear cache to pick up environment changes
        get_feature_flags.cache_clear()

        flags = get_feature_flags()
        assert flags.a2a_enabled is True
        assert flags.mcp_enabled is True
        assert flags.a2a_available() is True
        assert flags.mcp_available() is True

        # Clean up cache
        get_feature_flags.cache_clear()
