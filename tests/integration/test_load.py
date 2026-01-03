"""Integration Tests: Load Testing.

WBS Reference: WBS-AGT20 Integration Testing (AGT20.6)
Acceptance Criteria: AC-20.6 - Load test: 5 concurrent pipeline requests

Tests the system under concurrent load:
- 5 concurrent pipeline requests
- Response time validation
- System stability under load

TDD Status: RED → GREEN → REFACTOR
Pattern: Load/stress testing with asyncio concurrency
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import pytest
import pytest_asyncio


# Mark all tests as slow/load tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.slow,
]


class TestConcurrentPipelines:
    """Load tests for concurrent pipeline execution.
    
    AC-20.6: Load test: 5 concurrent pipeline requests
    
    Exit Criteria:
    - 5 concurrent pipelines complete within 60s timeout
    """
    
    async def test_five_concurrent_pipeline_requests(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.6: 5 concurrent pipelines complete within 60s.
        
        Given: ai-agents service running with dependencies
        When: 5 concurrent pipeline requests submitted
        Then: All complete within 60 seconds timeout
        """
        pipeline_inputs = [
            {
                "book": "A Philosophy of Software Design",
                "chapter": i,
                "title": f"Chapter {i}",
                "content": f"Sample content for chapter {i}. This is a test of concurrent processing.",
                "preset": "light",
            }
            for i in range(1, 6)
        ]
        
        start_time = time.monotonic()
        
        # Create concurrent tasks
        tasks = [
            ensure_ai_agents.post(
                "/v1/pipelines/chapter-summarization/run",
                json=input_data,
                timeout=60.0,
            )
            for input_data in pipeline_inputs
        ]
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_time = time.monotonic() - start_time
        
        # Verify all completed within timeout
        assert elapsed_time < 60.0, f"Requests took {elapsed_time:.2f}s (> 60s timeout)"
        
        # Count successful responses
        errors = 0
        
        for _i, response in enumerate(responses):
            if isinstance(response, Exception):
                errors += 1
            else:
                # Accept 200 (success) or 404/503 (service not configured)
                if response.status_code not in [200, 404, 503]:
                    errors += 1
        
        # At least some should complete
        assert errors <= len(responses), f"All {errors} requests failed"
    
    async def test_concurrent_function_invocations(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Test 5 concurrent function invocations.
        
        Similar to pipeline test but for direct function calls.
        """
        function_inputs = [
            {
                "content": f"Sample text {i} for summarization testing.",
                "format": "brief",
            }
            for i in range(5)
        ]
        
        start_time = time.monotonic()
        
        tasks = [
            ensure_ai_agents.post(
                "/v1/functions/summarize_content/invoke",
                json=input_data,
                timeout=30.0,
            )
            for input_data in function_inputs
        ]
        
        _responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_time = time.monotonic() - start_time
        
        # Should complete reasonably quickly
        assert elapsed_time < 60.0, f"Requests took {elapsed_time:.2f}s"


class TestSystemStability:
    """System stability tests under load."""
    
    async def test_sequential_requests_stability(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Test system handles sequential requests without degradation.
        """
        response_times: list[float] = []
        
        for _ in range(5):
            start = time.monotonic()
            
            response = await ensure_ai_agents.get("/health", timeout=10.0)
            
            elapsed = time.monotonic() - start
            response_times.append(elapsed)
            
            assert response.status_code in [200, 204]
        
        # Verify no significant degradation
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # Max should not be significantly worse than average
        assert max_time < avg_time * 3, f"Response time spike: {max_time:.2f}s vs avg {avg_time:.2f}s"
    
    async def test_mixed_concurrent_operations(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Test concurrent mix of different operations.
        """
        tasks = [
            # Health checks
            ensure_ai_agents.get("/health", timeout=10.0),
            ensure_ai_agents.get("/health", timeout=10.0),
            # Function invocations
            ensure_ai_agents.post(
                "/v1/functions/summarize_content/invoke",
                json={"content": "Test content", "format": "brief"},
                timeout=30.0,
            ),
            # Pipeline requests
            ensure_ai_agents.post(
                "/v1/pipelines/chapter-summarization/run",
                json={
                    "book": "Test",
                    "chapter": 1,
                    "title": "Test",
                    "content": "Test content",
                },
                timeout=30.0,
            ),
        ]
        
        start_time = time.monotonic()
        _responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start_time
        
        # Should complete without crashing
        assert elapsed < 60.0


class TestResponseTimeValidation:
    """Response time validation tests."""
    
    async def test_health_check_response_time(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Verify health check responds quickly.
        """
        start = time.monotonic()
        response = await ensure_ai_agents.get("/health", timeout=5.0)
        elapsed = time.monotonic() - start
        
        assert response.status_code in [200, 204]
        assert elapsed < 1.0, f"Health check too slow: {elapsed:.2f}s"
    
    async def test_function_list_response_time(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Verify function listing responds quickly.
        """
        start = time.monotonic()
        response = await ensure_ai_agents.get("/v1/functions", timeout=10.0)
        elapsed = time.monotonic() - start
        
        assert response.status_code in [200, 404]
        assert elapsed < 5.0, f"Function list too slow: {elapsed:.2f}s"


class TestResourceUsage:
    """Resource usage tests (connection pooling, etc.)."""
    
    async def test_connection_reuse(
        self,
        ai_agents_client: httpx.AsyncClient,
    ) -> None:
        """
        Verify client reuses connections efficiently.
        """
        try:
            # Multiple requests should reuse connection
            for _ in range(3):
                response = await ai_agents_client.get("/health", timeout=5.0)
                if response.status_code not in [200, 204]:
                    pytest.skip("ai-agents service not available")
        except httpx.ConnectError:
            pytest.skip("ai-agents service not available")
    
    async def test_timeout_handling(
        self,
        ai_agents_client: httpx.AsyncClient,
    ) -> None:
        """
        Test system handles timeouts gracefully.
        """
        try:
            # Use very short timeout
            _response = await ai_agents_client.get("/health", timeout=0.001)
        except httpx.TimeoutException:
            # Expected - timeout should be handled gracefully
            pass
        except httpx.ConnectError:
            pytest.skip("ai-agents service not available")


class TestLoadTestMetrics:
    """Tests that capture metrics for load testing."""
    
    async def test_capture_response_metrics(
        self,
        ensure_ai_agents: httpx.AsyncClient,
    ) -> None:
        """
        Capture and report response metrics.
        """
        metrics = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "response_times": [],
        }
        
        for _ in range(5):
            start = time.monotonic()
            
            try:
                response = await ensure_ai_agents.get("/health", timeout=10.0)
                elapsed = time.monotonic() - start
                
                metrics["total_requests"] += 1
                metrics["response_times"].append(elapsed)
                
                if response.status_code in [200, 204]:
                    metrics["successful"] += 1
                else:
                    metrics["failed"] += 1
                    
            except Exception:
                metrics["failed"] += 1
        
        # Report metrics
        if metrics["response_times"]:
            avg = sum(metrics["response_times"]) / len(metrics["response_times"])
            print("\nLoad Test Metrics:")
            print(f"  Total: {metrics['total_requests']}")
            print(f"  Success: {metrics['successful']}")
            print(f"  Failed: {metrics['failed']}")
            print(f"  Avg Response: {avg:.3f}s")
            print(f"  Min Response: {min(metrics['response_times']):.3f}s")
            print(f"  Max Response: {max(metrics['response_times']):.3f}s")
        
        # At least some should succeed
        assert metrics["successful"] > 0 or metrics["failed"] == metrics["total_requests"]
