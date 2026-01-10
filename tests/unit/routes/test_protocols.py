"""Unit Tests for Protocols Router Endpoints.

WBS-AGT21: Protocol API Route
AC-21.1: GET /v1/protocols lists available protocols
AC-21.2: GET /v1/protocols/{id} returns protocol details
AC-21.3: POST /v1/protocols/{id}/run executes protocol

Test Coverage:
- Protocol discovery (list, detail)
- Protocol execution request/response validation
- Error handling (404, 422)
- Request/response model validation

TDD Status: RED - Tests written first, implementation follows
"""

import unittest
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestProtocolsRouterDiscovery(unittest.TestCase):
    """Test suite for Protocol discovery endpoints (AC-21.1, AC-21.2)."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.main import create_app
        self.app = create_app()
        self.client = TestClient(self.app)

    # =========================================================================
    # AC-21.1: GET /v1/protocols - List Protocols
    # =========================================================================

    def test_list_protocols_returns_200(self) -> None:
        """GET /v1/protocols returns 200 with protocol list."""
        response = self.client.get("/v1/protocols")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Response structure validation
        assert "protocols" in data
        assert isinstance(data["protocols"], list)
        assert "count" in data

    def test_list_protocols_includes_roundtable(self) -> None:
        """GET /v1/protocols includes ROUNDTABLE_DISCUSSION."""
        response = self.client.get("/v1/protocols")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        protocol_ids = [p["protocol_id"] for p in data["protocols"]]
        assert "ROUNDTABLE_DISCUSSION" in protocol_ids

    def test_list_protocols_has_required_fields(self) -> None:
        """GET /v1/protocols protocols have required fields."""
        response = self.client.get("/v1/protocols")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for protocol in data["protocols"]:
            assert "protocol_id" in protocol
            assert "name" in protocol
            assert "description" in protocol
            # brigade_roles is optional in list view

    # =========================================================================
    # AC-21.2: GET /v1/protocols/{id} - Get Protocol Details
    # =========================================================================

    def test_get_protocol_roundtable_returns_200(self) -> None:
        """GET /v1/protocols/ROUNDTABLE_DISCUSSION returns 200."""
        response = self.client.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["protocol_id"] == "ROUNDTABLE_DISCUSSION"

    def test_get_protocol_has_brigade_roles(self) -> None:
        """GET /v1/protocols/{id} returns brigade_roles."""
        response = self.client.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "brigade_roles" in data
        assert "analyst" in data["brigade_roles"]
        assert "critic" in data["brigade_roles"]
        assert "synthesizer" in data["brigade_roles"]
        assert "validator" in data["brigade_roles"]

    def test_get_protocol_has_rounds(self) -> None:
        """GET /v1/protocols/{id} returns rounds definition."""
        response = self.client.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "rounds" in data
        assert len(data["rounds"]) >= 1
        
        for round_def in data["rounds"]:
            assert "round" in round_def
            assert "type" in round_def

    def test_get_protocol_not_found_returns_404(self) -> None:
        """GET /v1/protocols/INVALID returns 404."""
        response = self.client.get("/v1/protocols/NONEXISTENT_PROTOCOL_XYZ")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data


class TestProtocolsRouterExecution(unittest.TestCase):
    """Test suite for Protocol execution endpoint (AC-21.3).
    
    Note: These tests validate request/response handling, not full execution.
    Full execution requires live LLM services (see integration tests).
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.main import create_app
        self.app = create_app()
        self.client = TestClient(self.app)

    # =========================================================================
    # AC-21.3: POST /v1/protocols/{id}/run - Request Validation
    # =========================================================================

    def test_execute_protocol_missing_inputs_returns_422(self) -> None:
        """POST /v1/protocols/{id}/run without inputs returns 422."""
        response = self.client.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json={},  # Missing inputs
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_execute_protocol_invalid_protocol_returns_404(self) -> None:
        """POST /v1/protocols/INVALID/run returns 404."""
        response = self.client.post(
            "/v1/protocols/NONEXISTENT_PROTOCOL_XYZ/run",
            json={"inputs": {"topic": "test"}},
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_execute_protocol_accepts_valid_request(self) -> None:
        """POST /v1/protocols/{id}/run accepts valid request body.
        
        Note: May return 503 if LLM services unavailable, which is acceptable.
        The test validates request parsing, not full execution.
        """
        response = self.client.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json={
                "inputs": {"topic": "Test topic for validation"},
                "config": {
                    "max_feedback_loops": 0,
                    "allow_feedback": False,
                    "run_cross_reference": False,
                }
            },
            # Don't wait forever
        )
        
        # Accept 200 (success) or 503 (service unavailable)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_202_ACCEPTED,  # Async execution
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_504_GATEWAY_TIMEOUT,
        ], f"Unexpected status: {response.status_code}"

    def test_execute_protocol_with_brigade_override(self) -> None:
        """POST /v1/protocols/{id}/run accepts brigade_override."""
        response = self.client.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json={
                "inputs": {"topic": "Test with custom models"},
                "brigade_override": {
                    "analyst": "gpt-5.2",
                    "critic": "claude-opus-4-5-20251101",
                },
                "config": {
                    "max_feedback_loops": 0,
                    "allow_feedback": False,
                    "run_cross_reference": False,
                }
            },
        )
        
        # Accept valid responses
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_202_ACCEPTED,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_504_GATEWAY_TIMEOUT,
        ]


class TestProtocolsResponseModels(unittest.TestCase):
    """Test response model schemas."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from src.main import create_app
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_protocol_list_response_schema(self) -> None:
        """ProtocolListResponse has correct schema."""
        response = self.client.get("/v1/protocols")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Required fields
        assert isinstance(data.get("protocols"), list)
        assert isinstance(data.get("count"), int)
        assert data["count"] == len(data["protocols"])

    def test_protocol_detail_response_schema(self) -> None:
        """ProtocolDetailResponse has correct schema."""
        response = self.client.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Required fields
        assert isinstance(data.get("protocol_id"), str)
        assert isinstance(data.get("name"), str)
        assert isinstance(data.get("description"), str)
        assert isinstance(data.get("brigade_roles"), dict)
        assert isinstance(data.get("rounds"), list)
        
        # Brigade role schema
        for role_name, role_config in data["brigade_roles"].items():
            assert isinstance(role_config.get("model"), str)
            assert "system_prompt" in role_config


if __name__ == "__main__":
    unittest.main()
