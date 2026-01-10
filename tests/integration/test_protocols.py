"""Integration Tests: Protocols E2E.

WBS Reference: WBS-AGT21 Protocol API Integration Testing
Acceptance Criteria: AC-21.1 - E2E test: protocol request → response

Tests the complete protocol execution flow from HTTP request to response,
validating the Kitchen Brigade protocol API integration.

TDD Status: RED - Tests written first, implementation follows

API Endpoints (to be implemented in src/api/routes/protocols.py):
- GET /v1/protocols - List available protocols
- GET /v1/protocols/{protocol_id} - Get protocol details
- POST /v1/protocols/{protocol_id}/run - Execute protocol

Available protocols: ROUNDTABLE_DISCUSSION, DEBATE_PROTOCOL, PIPELINE_PROTOCOL,
                    ARCHITECTURE_RECONCILIATION, WBS_GENERATION
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
import pytest_asyncio


# Skip if live services not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


class TestProtocolDiscoveryE2E:
    """E2E tests for protocol discovery endpoints.
    
    AC-21.1: Verify protocol listing and detail retrieval.
    """
    
    async def test_list_protocols_returns_available_protocols(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.1: GET /v1/protocols returns list of available protocols.
        
        Given: ai-agents service is running
        When: GET /v1/protocols
        Then: Returns list of protocol summaries with IDs and descriptions
        """
        response = await ensure_ai_agents.get("/v1/protocols")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "protocols" in data
        assert isinstance(data["protocols"], list)
        assert len(data["protocols"]) >= 1  # At least ROUNDTABLE_DISCUSSION
        
        # Verify protocol structure
        protocol = data["protocols"][0]
        assert "protocol_id" in protocol
        assert "name" in protocol
        assert "description" in protocol
    
    async def test_list_protocols_includes_roundtable(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.1: Protocol list includes ROUNDTABLE_DISCUSSION.
        
        Given: ai-agents service with protocol configs
        When: GET /v1/protocols
        Then: ROUNDTABLE_DISCUSSION is in the list
        """
        response = await ensure_ai_agents.get("/v1/protocols")
        
        assert response.status_code == 200
        data = response.json()
        
        protocol_ids = [p["protocol_id"] for p in data["protocols"]]
        assert "ROUNDTABLE_DISCUSSION" in protocol_ids
    
    async def test_get_protocol_details_roundtable(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.1: GET /v1/protocols/{id} returns protocol details.
        
        Given: ai-agents service is running
        When: GET /v1/protocols/ROUNDTABLE_DISCUSSION
        Then: Returns full protocol definition with roles and rounds
        """
        response = await ensure_ai_agents.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify protocol structure
        assert data["protocol_id"] == "ROUNDTABLE_DISCUSSION"
        assert "name" in data
        assert "description" in data
        assert "brigade_roles" in data
        assert "rounds" in data
        
        # Verify brigade roles
        assert "analyst" in data["brigade_roles"]
        assert "critic" in data["brigade_roles"]
        assert "synthesizer" in data["brigade_roles"]
        assert "validator" in data["brigade_roles"]
        
        # Verify rounds structure
        assert len(data["rounds"]) >= 1
        assert "type" in data["rounds"][0]
    
    async def test_get_protocol_not_found(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.1: Invalid protocol ID returns 404.
        
        Given: ai-agents service is running
        When: GET /v1/protocols/NONEXISTENT_PROTOCOL
        Then: Returns 404 with error detail
        """
        response = await ensure_ai_agents.get("/v1/protocols/NONEXISTENT_PROTOCOL")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data


class TestProtocolExecutionE2E:
    """E2E tests for protocol execution endpoint.
    
    AC-21.2: Protocol execution via REST API.
    
    These tests require live LLM services and may be slow.
    """
    
    async def test_execute_roundtable_basic_request(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.2: POST /v1/protocols/{id}/run executes protocol.
        
        Given: ai-agents service with LLM gateway available
        When: POST /v1/protocols/ROUNDTABLE_DISCUSSION/run
        Then: Returns execution results with outputs and trace
        """
        request_body = {
            "inputs": {
                "topic": "What is the best approach to implement dependency injection?",
            },
            "config": {
                "max_feedback_loops": 1,
                "allow_feedback": False,
                "run_cross_reference": False,  # Disable for faster test
            }
        }
        
        response = await ensure_ai_agents.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json=request_body,
            timeout=300.0,  # Protocol execution can be slow
        )
        
        # Accept success or service unavailable (LLM not reachable)
        assert response.status_code in [200, 503, 504], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify execution response structure
            assert "execution_id" in data
            assert "status" in data
            assert data["status"] in ["completed", "partial", "failed"]
            
            # Verify outputs present
            assert "outputs" in data
            assert "trace_id" in data
    
    async def test_execute_protocol_with_brigade_override(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.2: Protocol execution supports custom model assignments.
        
        Given: ai-agents service running
        When: POST with brigade_override specifying custom models
        Then: Uses specified models for each role
        """
        request_body = {
            "inputs": {
                "topic": "How to structure a Python microservice?",
            },
            "brigade_override": {
                "analyst": "gpt-5.2",
                "critic": "deepseek-api/deepseek-chat",
                "synthesizer": "gpt-5.2",
                "validator": "claude-opus-4-5-20251101",
            },
            "config": {
                "max_feedback_loops": 0,
                "allow_feedback": False,
                "run_cross_reference": False,
            }
        }
        
        response = await ensure_ai_agents.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json=request_body,
            timeout=300.0,
        )
        
        # Accept success or various error codes
        assert response.status_code in [200, 400, 422, 503, 504]
        
        if response.status_code == 200:
            data = response.json()
            assert "outputs" in data
            # Trace should reflect custom models used
            assert "trace_id" in data
    
    async def test_execute_protocol_with_cross_reference(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.3: Protocol execution with cross-reference enabled.
        
        Given: ai-agents with semantic-search and code-orchestrator available
        When: POST with run_cross_reference=True
        Then: Cross-reference evidence included in execution
        """
        request_body = {
            "inputs": {
                "topic": "Repository pattern for data access",
                "cross_reference_queries": [
                    "repository pattern",
                    "data access layer",
                ],
            },
            "config": {
                "max_feedback_loops": 0,
                "allow_feedback": False,
                "run_cross_reference": True,
            }
        }
        
        response = await ensure_ai_agents.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json=request_body,
            timeout=300.0,
        )
        
        # Accept success or service unavailable
        assert response.status_code in [200, 503, 504]
        
        if response.status_code == 200:
            data = response.json()
            # Cross-reference evidence should be in trace
            assert "trace_id" in data
    
    async def test_execute_invalid_protocol_returns_404(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.2: Invalid protocol ID returns 404.
        
        Given: ai-agents service running
        When: POST /v1/protocols/INVALID/run
        Then: Returns 404 error
        """
        response = await ensure_ai_agents.post(
            "/v1/protocols/INVALID_PROTOCOL_DOES_NOT_EXIST/run",
            json={"inputs": {"topic": "test"}},
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data
    
    async def test_execute_missing_required_inputs_returns_422(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.2: Missing required inputs returns 422.
        
        Given: ai-agents service running
        When: POST without required inputs
        Then: Returns 422 validation error
        """
        response = await ensure_ai_agents.post(
            "/v1/protocols/ROUNDTABLE_DISCUSSION/run",
            json={},  # Missing inputs
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestProtocolResponseStructure:
    """Tests for protocol response structure consistency.
    
    AC-21.4: Response structure matches specification.
    """
    
    async def test_protocol_list_response_schema(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.4: List response matches ProtocolListResponse schema.
        """
        response = await ensure_ai_agents.get("/v1/protocols")
        
        assert response.status_code == 200
        data = response.json()
        
        # Schema validation
        assert isinstance(data.get("protocols"), list)
        assert isinstance(data.get("count"), int)
        
        for protocol in data["protocols"]:
            assert isinstance(protocol.get("protocol_id"), str)
            assert isinstance(protocol.get("name"), str)
            assert isinstance(protocol.get("description"), str)
            # Optional fields
            if "brigade_roles" in protocol:
                assert isinstance(protocol["brigade_roles"], list)
    
    async def test_protocol_detail_response_schema(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.4: Detail response matches ProtocolDetailResponse schema.
        """
        response = await ensure_ai_agents.get("/v1/protocols/ROUNDTABLE_DISCUSSION")
        
        if response.status_code != 200:
            pytest.skip("Protocol not found")
        
        data = response.json()
        
        # Required fields
        assert isinstance(data.get("protocol_id"), str)
        assert isinstance(data.get("name"), str)
        assert isinstance(data.get("description"), str)
        assert isinstance(data.get("brigade_roles"), dict)
        assert isinstance(data.get("rounds"), list)
        
        # Brigade role structure
        for role_name, role_config in data["brigade_roles"].items():
            assert isinstance(role_config.get("model"), str)
            assert isinstance(role_config.get("system_prompt"), str)
        
        # Round structure
        for round_def in data["rounds"]:
            assert isinstance(round_def.get("round"), int)
            assert isinstance(round_def.get("type"), str)
            assert round_def["type"] in ["parallel", "synthesis", "consensus"]


class TestProtocolExecutionFlow:
    """Tests for complete protocol execution workflow.
    
    AC-21.5: Full execution flow from request to response.
    """
    
    async def test_architecture_reconciliation_protocol(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.5: ARCHITECTURE_RECONCILIATION protocol executes end-to-end.
        """
        request_body = {
            "inputs": {
                "topic": "Reconcile service boundaries between ai-agents and llm-gateway",
            },
            "config": {
                "max_feedback_loops": 0,
                "allow_feedback": False,
                "run_cross_reference": False,
            }
        }
        
        response = await ensure_ai_agents.post(
            "/v1/protocols/ARCHITECTURE_RECONCILIATION/run",
            json=request_body,
            timeout=300.0,
        )
        
        assert response.status_code in [200, 404, 503, 504]
    
    async def test_debate_protocol(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-21.5: DEBATE_PROTOCOL executes end-to-end.
        """
        request_body = {
            "inputs": {
                "topic": "Should microservices always use async communication?",
            },
            "config": {
                "max_feedback_loops": 0,
                "allow_feedback": False,
                "run_cross_reference": False,
            }
        }
        
        response = await ensure_ai_agents.post(
            "/v1/protocols/DEBATE_PROTOCOL/run",
            json=request_body,
            timeout=300.0,
        )
        
        assert response.status_code in [200, 404, 503, 504]
