"""Integration Tests: Qdrant Connectivity via Semantic Search.

PCON-8.2: ai-agents → Qdrant (via semantic-search-service) connectivity test
Acceptance Criteria: AC-8.2 - ai-agents → Qdrant (via semantic-search) test passes

Tests the connection from ai-agents to Qdrant through semantic-search-service,
which is the canonical path for vector operations.
"""

from __future__ import annotations

import os

import httpx
import pytest

# Mark all tests as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Configuration
# =============================================================================

SEMANTIC_SEARCH_BASE_URL = os.environ.get(
    "SEMANTIC_SEARCH_URL", "http://localhost:8081"
)


# =============================================================================
# Test Classes
# =============================================================================


class TestQdrantConnectivityViaSemanticSearch:
    """PCON-8.2: Qdrant connectivity tests via semantic-search-service.
    
    AC-8.2: ai-agents → Qdrant (via semantic-search) test passes
    
    Verifies:
    - semantic-search-service health shows qdrant=connected
    - Embedding endpoint works (uses the embedding model)
    - Search endpoint returns results from Qdrant
    """

    @pytest.mark.asyncio
    async def test_semantic_search_health_shows_qdrant_connected(self) -> None:
        """Test that semantic-search health shows Qdrant connected.
        
        Given: semantic-search-service running with USE_REAL_CLIENTS=true
        When: GET /health
        Then: Response shows qdrant: connected
        """
        async with httpx.AsyncClient(
            base_url=SEMANTIC_SEARCH_BASE_URL,
            timeout=30.0,
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "dependencies" in data
            assert data["dependencies"].get("qdrant") == "connected"

    @pytest.mark.asyncio
    async def test_semantic_search_embedder_loaded(self) -> None:
        """Test that semantic-search has embedder loaded.
        
        Given: semantic-search-service with SBERT model
        When: GET /health
        Then: Response shows embedder: loaded
        """
        async with httpx.AsyncClient(
            base_url=SEMANTIC_SEARCH_BASE_URL,
            timeout=30.0,
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["dependencies"].get("embedder") == "loaded"

    @pytest.mark.asyncio
    async def test_embed_endpoint_returns_vectors(self) -> None:
        """Test that /v1/embed endpoint returns 384-dim vectors.
        
        Given: semantic-search with all-MiniLM-L6-v2 model
        When: POST /v1/embed with text
        Then: Returns embeddings with 384 dimensions
        """
        async with httpx.AsyncClient(
            base_url=SEMANTIC_SEARCH_BASE_URL,
            timeout=60.0,  # Model loading can take time
        ) as client:
            response = await client.post(
                "/v1/embed",
                json={"text": "test embedding query"},  # Single text or list
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) >= 1
            
            # Verify 384 dimensions (all-MiniLM-L6-v2)
            embedding = data["embeddings"][0]
            assert len(embedding) == 384, f"Expected 384 dims, got {len(embedding)}"
            
            # Verify model name
            assert data.get("model") == "all-MiniLM-L6-v2"
            assert data.get("dimensions") == 384

    @pytest.mark.asyncio
    async def test_search_endpoint_queries_qdrant(self) -> None:
        """Test that /v1/search queries Qdrant and returns results.
        
        Given: semantic-search connected to Qdrant
        When: POST /v1/search with a query
        Then: Returns results array (may be empty if no data)
        """
        async with httpx.AsyncClient(
            base_url=SEMANTIC_SEARCH_BASE_URL,
            timeout=60.0,
        ) as client:
            response = await client.post(
                "/v1/search",
                json={
                    "query": "software design patterns",
                    "collection": "code_snippets",
                    "limit": 5,  # API uses 'limit' not 'top_k'
                },
            )
            
            # Should return 200 even if collection doesn't exist
            # or has no matching results
            if response.status_code == 200:
                data = response.json()
                assert "results" in data
                assert isinstance(data["results"], list)
            elif response.status_code == 404:
                # Collection may not exist - that's acceptable
                pass
            else:
                # Any other error should fail the test
                pytest.fail(f"Unexpected status {response.status_code}: {response.text}")

    @pytest.mark.asyncio
    async def test_collections_endpoint(self) -> None:
        """Test that /collections endpoint lists Qdrant collections.
        
        Given: semantic-search connected to Qdrant
        When: GET /collections
        Then: Returns list of collections
        """
        async with httpx.AsyncClient(
            base_url=SEMANTIC_SEARCH_BASE_URL,
            timeout=30.0,
        ) as client:
            response = await client.get("/collections")
            
            if response.status_code == 200:
                data = response.json()
                assert "collections" in data
                assert isinstance(data["collections"], list)
            elif response.status_code == 404:
                # Endpoint may not exist - skip
                pytest.skip("Collections endpoint not implemented")


class TestDirectQdrantConnectivity:
    """Direct Qdrant connectivity tests.
    
    These tests connect directly to Qdrant for verification,
    not through semantic-search-service.
    """
    
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

    @pytest.mark.asyncio
    async def test_qdrant_health(self) -> None:
        """Test direct Qdrant health endpoint.
        
        Given: Qdrant running at ai-platform-qdrant:6333
        When: GET /
        Then: Returns healthy status
        """
        async with httpx.AsyncClient(
            base_url=self.QDRANT_URL,
            timeout=10.0,
        ) as client:
            response = await client.get("/")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_qdrant_collections_api(self) -> None:
        """Test Qdrant collections API directly.
        
        Given: Qdrant running
        When: GET /collections
        Then: Returns collections object
        """
        async with httpx.AsyncClient(
            base_url=self.QDRANT_URL,
            timeout=10.0,
        ) as client:
            response = await client.get("/collections")
            assert response.status_code == 200
            
            data = response.json()
            assert "result" in data
            assert "collections" in data["result"]
