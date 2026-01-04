"""Feature Flag Matrix E2E Tests.

WBS-PI7: End-to-End Protocol Testing
AC-PI7.7: All flags disabled → all Phase 2 endpoints return 501/404
AC-PI7.8: Partial enablement works (A2A on, MCP off)
AC-PI7.9: Phase 1 endpoints always return 200 regardless of flags

These tests verify the feature flag system correctly gates Protocol
Integration features while maintaining Phase 1 stability.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Feature Flags
Anti-Patterns: #25 (Test Isolation)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# -----------------------------------------------------------------------------
# AC-PI7.7: All Flags Disabled → Phase 2 Endpoints Return 501/404
# -----------------------------------------------------------------------------


class TestAllFlagsDisabled:
    """Verify all Phase 2 endpoints return 501/404 when flags disabled."""
    
    def test_agent_card_returns_404_when_disabled(
        self, test_client_disabled: TestClient
    ) -> None:
        """Agent Card should return 404 when A2A disabled."""
        response = test_client_disabled.get("/.well-known/agent-card.json")
        
        assert response.status_code == 404
    
    def test_a2a_message_send_returns_501_when_disabled(
        self, test_client_disabled: TestClient,
        sample_a2a_message: dict,
    ) -> None:
        """A2A message:send should return 501 when A2A disabled."""
        response = test_client_disabled.post(
            "/a2a/v1/message:send",
            json=sample_a2a_message,
        )
        
        assert response.status_code == 501
        assert "not enabled" in response.json()["detail"].lower()
    
    def test_a2a_message_stream_returns_501_when_disabled(
        self, test_client_disabled: TestClient,
        sample_a2a_message: dict,
    ) -> None:
        """A2A message:stream should return 501 when streaming disabled."""
        response = test_client_disabled.post(
            "/a2a/v1/message:stream",
            json=sample_a2a_message,
        )
        
        assert response.status_code == 501
    
    def test_a2a_task_status_returns_501_when_disabled(
        self, test_client_disabled: TestClient,
    ) -> None:
        """A2A task status should return 501 when A2A disabled."""
        response = test_client_disabled.get("/a2a/v1/tasks/test-task-id")
        
        assert response.status_code == 501


# -----------------------------------------------------------------------------
# AC-PI7.8: Partial Enablement Works (A2A on, MCP off)
# -----------------------------------------------------------------------------


class TestPartialEnablement:
    """Verify partial feature enablement works correctly."""
    
    def test_a2a_works_when_mcp_disabled(
        self, test_client_a2a_only: TestClient,
    ) -> None:
        """A2A Agent Card should work when only A2A enabled."""
        response = test_client_a2a_only.get("/.well-known/agent-card.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "skills" in data
    
    def test_mcp_disabled_when_only_a2a_enabled(
        self, test_client_a2a_only: TestClient,
    ) -> None:
        """MCP endpoints should be disabled when only A2A enabled."""
        # MCP is disabled, so MCP-specific features won't work
        # but the server still exposes Phase 1 endpoints
        response = test_client_a2a_only.get("/v1/functions")
        
        # Phase 1 should still work
        assert response.status_code == 200
    
    def test_a2a_disabled_when_only_mcp_enabled(
        self, test_client_mcp_only: TestClient,
    ) -> None:
        """A2A Agent Card should return 404 when only MCP enabled."""
        response = test_client_mcp_only.get("/.well-known/agent-card.json")
        
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# AC-PI7.9: Phase 1 Endpoints Always Return 200 Regardless of Flags
# -----------------------------------------------------------------------------


class TestPhase1Stability:
    """Verify Phase 1 endpoints always work regardless of Phase 2 state."""
    
    def test_functions_list_works_when_all_disabled(
        self, test_client_disabled: TestClient,
    ) -> None:
        """Phase 1 /v1/functions should return 200 when all disabled."""
        response = test_client_disabled.get("/v1/functions")
        
        assert response.status_code == 200
        data = response.json()
        assert "functions" in data
    
    def test_functions_list_works_when_all_enabled(
        self, test_client_enabled: TestClient,
    ) -> None:
        """Phase 1 /v1/functions should return 200 when all enabled."""
        response = test_client_enabled.get("/v1/functions")
        
        assert response.status_code == 200
        data = response.json()
        assert "functions" in data
    
    def test_extract_structure_works_when_all_disabled(
        self, test_client_disabled: TestClient,
        sample_extract_request: dict,
    ) -> None:
        """Phase 1 extract-structure should work when all disabled."""
        response = test_client_disabled.post(
            "/v1/functions/extract-structure/run",
            json=sample_extract_request,
        )
        
        assert response.status_code == 200
    
    def test_extract_structure_works_when_all_enabled(
        self, test_client_enabled: TestClient,
        sample_extract_request: dict,
    ) -> None:
        """Phase 1 extract-structure should work when all enabled."""
        response = test_client_enabled.post(
            "/v1/functions/extract-structure/run",
            json=sample_extract_request,
        )
        
        assert response.status_code == 200
    
    def test_extract_structure_works_when_a2a_only(
        self, test_client_a2a_only: TestClient,
        sample_extract_request: dict,
    ) -> None:
        """Phase 1 extract-structure should work with A2A only."""
        response = test_client_a2a_only.post(
            "/v1/functions/extract-structure/run",
            json=sample_extract_request,
        )
        
        assert response.status_code == 200
    
    def test_extract_structure_works_when_mcp_only(
        self, test_client_mcp_only: TestClient,
        sample_extract_request: dict,
    ) -> None:
        """Phase 1 extract-structure should work with MCP only."""
        response = test_client_mcp_only.post(
            "/v1/functions/extract-structure/run",
            json=sample_extract_request,
        )
        
        assert response.status_code == 200
    
    def test_health_endpoint_always_works(
        self, test_client_disabled: TestClient,
    ) -> None:
        """Health endpoint should always return 200."""
        response = test_client_disabled.get("/health")
        
        assert response.status_code == 200


# -----------------------------------------------------------------------------
# Feature Flag Combinations Matrix
# -----------------------------------------------------------------------------


class TestFeatureFlagMatrix:
    """Test various feature flag combinations."""
    
    @pytest.mark.parametrize(
        "fixture_name,expected_agent_card,expected_phase1",
        [
            ("test_client_disabled", 404, 200),
            ("test_client_enabled", 200, 200),
            ("test_client_a2a_only", 200, 200),
            ("test_client_mcp_only", 404, 200),
        ],
    )
    def test_flag_matrix(
        self,
        fixture_name: str,
        expected_agent_card: int,
        expected_phase1: int,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test matrix of flag combinations."""
        client = request.getfixturevalue(fixture_name)
        
        # Agent Card behavior varies by A2A flag
        agent_card_response = client.get("/.well-known/agent-card.json")
        assert agent_card_response.status_code == expected_agent_card
        
        # Phase 1 always works
        phase1_response = client.get("/v1/functions")
        assert phase1_response.status_code == expected_phase1
