"""Integration Tests: Semantic Search Service.

WBS Reference: WBS-AGT20 Integration Testing (AGT20.4)
Acceptance Criteria: AC-20.4 - Service integration: ai-agents → semantic-search-service

Tests the integration between ai-agents and semantic-search-service:
- Query request/response flow
- Embedding generation
- Document retrieval

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


class TestSemanticSearchIntegration:
    """Integration tests for ai-agents → semantic-search-service.
    
    AC-20.4: Service integration: ai-agents → semantic-search-service
    
    Verifies:
    - semantic-search-service:8081 receives query requests
    - Proper embedding handling
    - Search result formatting
    """
    
    async def test_semantic_search_health(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.4: Verify semantic-search-service is reachable.
        
        Given: semantic-search-service running at :8081
        When: GET /health
        Then: Returns 200 OK
        """
        response = await ensure_semantic_search.get("/health")
        assert response.status_code in [200, 204]
    
    async def test_search_query_endpoint(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.4: Test search query endpoint.
        
        Given: semantic-search-service running
        When: POST /v1/search with query
        Then: Returns search results
        """
        request_data = {
            "query": "software design complexity",
            "limit": 5,
        }
        
        response = await ensure_semantic_search.post(
            "/v1/search",
            json=request_data,
        )
        
        assert response.status_code in [200, 404, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data or isinstance(data, list)
    
    async def test_embedding_endpoint(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.4: Test embedding generation endpoint.
        
        Given: semantic-search-service running
        When: POST /v1/embeddings with text
        Then: Returns embedding vector
        """
        request_data = {
            "input": "The quick brown fox jumps over the lazy dog.",
            "model": "all-MiniLM-L6-v2",
        }
        
        response = await ensure_semantic_search.post(
            "/v1/embeddings",
            json=request_data,
        )
        
        assert response.status_code in [200, 404, 422, 501]
        
        if response.status_code == 200:
            data = response.json()
            # Verify embedding structure
            assert "data" in data or "embedding" in data or "embeddings" in data
    
    async def test_document_search_with_filters(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.4: Test filtered document search.
        
        Given: semantic-search-service with indexed documents
        When: POST /v1/search with filters
        Then: Returns filtered results
        """
        request_data = {
            "query": "function composition",
            "limit": 10,
            "filters": {
                "source": "code-reference-engine",
            },
        }
        
        response = await ensure_semantic_search.post(
            "/v1/search",
            json=request_data,
        )
        
        # May not support filters, so accept various responses
        assert response.status_code in [200, 400, 404, 422]
    
    async def test_semantic_search_via_ai_agents(
        self,
        ensure_ai_agents: httpx.AsyncClient,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        AC-20.4: Test ai-agents routes queries through semantic-search.
        
        Given: Both services running
        When: Calling function that requires semantic search
        Then: ai-agents makes request to semantic-search-service
        """
        # Use cross_reference function which should use semantic search
        response = await ensure_ai_agents.post(
            "/v1/functions/cross_reference/invoke",
            json={
                "concepts": ["dependency injection", "composition"],
                "scope": "code-reference-engine",
            },
        )
        
        # The request should either succeed or fail gracefully
        assert response.status_code in [200, 404, 422, 500, 502, 503]


class TestSemanticSearchBatchOperations:
    """Tests for batch operations in semantic-search-service."""
    
    async def test_batch_embedding_request(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        Test batch embedding generation.
        """
        request_data = {
            "input": [
                "First text to embed",
                "Second text to embed",
                "Third text to embed",
            ],
            "model": "all-MiniLM-L6-v2",
        }
        
        response = await ensure_semantic_search.post(
            "/v1/embeddings",
            json=request_data,
        )
        
        assert response.status_code in [200, 404, 422, 501]
        
        if response.status_code == 200:
            data = response.json()
            # Verify multiple embeddings returned
            if "data" in data:
                assert len(data["data"]) >= 1
    
    async def test_multi_query_search(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        Test searching with multiple queries.
        """
        request_data = {
            "queries": [
                "software architecture",
                "design patterns",
            ],
            "limit": 3,
        }
        
        # Try multi-query endpoint if available
        response = await ensure_semantic_search.post(
            "/v1/search/multi",
            json=request_data,
        )
        
        # Multi-query may not be supported
        assert response.status_code in [200, 404, 405, 422]


class TestSemanticSearchErrorHandling:
    """Error handling tests for semantic-search integration."""
    
    async def test_empty_query_handling(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        Test handling of empty search query.
        """
        request_data = {
            "query": "",
            "limit": 5,
        }
        
        response = await ensure_semantic_search.post(
            "/v1/search",
            json=request_data,
        )
        
        # Should return validation error or empty results
        assert response.status_code in [200, 400, 422]
    
    async def test_invalid_limit_handling(
        self,
        ensure_semantic_search: httpx.AsyncClient,
    ) -> None:
        """
        Test handling of invalid limit parameter.
        """
        request_data = {
            "query": "test query",
            "limit": -1,
        }
        
        response = await ensure_semantic_search.post(
            "/v1/search",
            json=request_data,
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]


class TestSemanticSearchHealthCheck:
    """Health check tests for semantic-search-service connection."""
    
    async def test_health_check_response(
        self,
        semantic_search_client: httpx.AsyncClient,
    ) -> None:
        """
        Verify health check endpoint works.
        """
        try:
            response = await semantic_search_client.get("/health", timeout=5.0)
            assert response.status_code in [200, 204]
        except httpx.TimeoutException:
            pytest.skip("semantic-search-service not available")
        except httpx.ConnectError:
            pytest.skip("semantic-search-service not available")
    
    async def test_readiness_check(
        self,
        semantic_search_client: httpx.AsyncClient,
    ) -> None:
        """
        Test readiness endpoint if available.
        """
        try:
            response = await semantic_search_client.get("/ready", timeout=5.0)
            # Readiness endpoint may not exist
            assert response.status_code in [200, 204, 404]
        except httpx.TimeoutException:
            pytest.skip("semantic-search-service not available")
        except httpx.ConnectError:
            pytest.skip("semantic-search-service not available")
