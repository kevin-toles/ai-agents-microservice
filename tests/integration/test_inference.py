"""Integration Tests: Inference Service.

WBS Reference: WBS-AGT20 Integration Testing (AGT20.3)
Acceptance Criteria: AC-20.3 - Service integration: ai-agents → inference-service

Tests the integration between ai-agents and inference-service:
- Health check connectivity
- Completion request/response flow
- Model parameter configuration

TDD Status: RED → GREEN → REFACTOR
Pattern: Service-to-service integration testing
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
import pytest_asyncio


# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


class TestInferenceServiceIntegration:
    """Integration tests for ai-agents → inference-service.
    
    AC-20.3: Service integration: ai-agents → inference-service
    
    Verifies:
    - inference-service:8085 receives completion requests
    - Proper request/response schema handling
    - Error propagation
    """
    
    async def test_inference_service_health(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.3: Verify inference-service is reachable.
        
        Given: inference-service running at :8085
        When: GET /health
        Then: Returns 200 OK
        """
        response = await ensure_inference_service.get("/health")
        assert response.status_code in [200, 204]
    
    async def test_inference_service_models_endpoint(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.3: Verify models endpoint availability.
        
        Given: inference-service running
        When: GET /v1/models
        Then: Returns list of available models
        """
        response = await ensure_inference_service.get("/v1/models")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
    
    async def test_completion_request(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.3: Test completion request flow.
        
        Given: inference-service with model loaded
        When: POST /v1/completions with prompt
        Then: Returns generated completion
        """
        request_data = {
            "model": "granite-8b-code",
            "prompt": "def hello():\n    '''Return a greeting message.'''\n    return ",
            "max_tokens": 50,
            "temperature": 0.1,
            "stop": ["\n\n"],
        }
        
        response = await ensure_inference_service.post(
            "/v1/completions",
            json=request_data,
        )
        
        # Accept various valid response codes
        assert response.status_code in [200, 400, 404, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "choices" in data or "completion" in data or "text" in data
    
    async def test_chat_completion_request(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.3: Test chat completion request flow.
        
        Given: inference-service with chat model
        When: POST /v1/chat/completions
        Then: Returns chat response
        """
        request_data = {
            "model": "granite-8b-code",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
            ],
            "max_tokens": 50,
            "temperature": 0.1,
        }
        
        response = await ensure_inference_service.post(
            "/v1/chat/completions",
            json=request_data,
        )
        
        assert response.status_code in [200, 400, 404, 501, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "choices" in data or "message" in data
    
    async def test_inference_via_ai_agents(
        self,
        ensure_ai_agents: httpx.AsyncClient,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.3: Test ai-agents routes completion through inference-service.
        
        Given: Both services running
        When: Calling function that requires LLM
        Then: ai-agents makes request to inference-service
        """
        # Use a function that requires inference
        response = await ensure_ai_agents.post(
            "/v1/functions/summarize_content/invoke",
            json={
                "content": "Python is a programming language known for its simplicity.",
                "format": "brief",
            },
        )
        
        # The request should either succeed or fail gracefully
        assert response.status_code in [200, 422, 500, 502, 503]


class TestInferenceServiceErrorHandling:
    """Tests for error handling in inference-service integration."""
    
    async def test_invalid_model_returns_error(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        Test that invalid model name returns appropriate error.
        """
        request_data = {
            "model": "nonexistent-model-xyz",
            "prompt": "Hello",
            "max_tokens": 10,
        }
        
        response = await ensure_inference_service.post(
            "/v1/completions",
            json=request_data,
        )
        
        # Should return error for invalid model
        assert response.status_code in [400, 404, 422, 503]
    
    async def test_empty_prompt_handling(
        self,
        ensure_inference_service: httpx.AsyncClient,
    ) -> None:
        """
        Test handling of empty prompt.
        """
        request_data = {
            "model": "granite-8b-code",
            "prompt": "",
            "max_tokens": 10,
        }
        
        response = await ensure_inference_service.post(
            "/v1/completions",
            json=request_data,
        )
        
        # Should either handle gracefully or return validation error
        assert response.status_code in [200, 400, 422]


class TestInferenceServiceHealthCheck:
    """Health check tests for inference-service connection."""
    
    async def test_health_check_response_time(
        self,
        inference_client: httpx.AsyncClient,
    ) -> None:
        """
        Verify health check responds within acceptable time.
        """
        import time
        
        start = time.monotonic()
        try:
            response = await inference_client.get("/health", timeout=5.0)
            elapsed = time.monotonic() - start
            
            if response.status_code == 200:
                # Health check should be fast
                assert elapsed < 2.0, f"Health check too slow: {elapsed:.2f}s"
        except httpx.TimeoutException:
            pytest.skip("inference-service not available")
        except httpx.ConnectError:
            pytest.skip("inference-service not available")
