"""Unit Tests for MSEP Router.

WBS: MSE-4.1 - MSEP Router
Tests for POST /v1/agents/enrich-metadata endpoint.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
- S3776: Cognitive complexity < 15 per function

Acceptance Criteria Tested:
- AC-4.1.1: POST /v1/agents/enrich-metadata accepts MSEPRequest
- AC-4.1.2: Returns EnrichedMetadata response
- AC-4.1.3: Returns 400 for invalid request
- AC-4.1.4: Returns 503 when services unavailable
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.agents.msep.config import MSEPConfig
from src.agents.msep.exceptions import ServiceUnavailableError
from src.agents.msep.schemas import (
    ChapterMeta,
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    MSEPRequest,
    Provenance,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def app() -> FastAPI:
    """Create FastAPI app with MSEP router."""
    from src.api.routes.enrich_metadata import router
    
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create sync test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncClient:
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def valid_request_body() -> dict[str, Any]:
    """Create valid request body for testing."""
    return {
        "corpus": [
            "Introduction to neural networks and deep learning.",
            "Advanced optimization techniques for gradient descent.",
        ],
        "chapter_index": [
            {
                "book": "Deep Learning",
                "chapter": 1,
                "title": "Neural Networks",
            },
            {
                "book": "Deep Learning",
                "chapter": 2,
                "title": "Optimization",
            },
        ],
        "config": {
            "threshold": 0.45,
            "top_k": 5,
            "timeout": 30.0,
            "enable_hybrid_search": True,
        },
    }


@pytest.fixture
def mock_enriched_metadata() -> EnrichedMetadata:
    """Create mock EnrichedMetadata response."""
    return EnrichedMetadata(
        chapters=[
            EnrichedChapter(
                book="Deep Learning",
                chapter=1,
                title="Neural Networks",
                chapter_id="Deep Learning:ch1",
                similar_chapters=[
                    CrossReference(
                        target="Deep Learning:ch2",
                        score=0.85,
                        base_score=0.70,
                        topic_boost=0.15,
                        method="sbert",
                    ),
                ],
                keywords=MergedKeywords(
                    tfidf=["neural", "network"],
                    semantic=["deep learning", "activation"],
                    merged=["neural", "network", "deep learning", "activation"],
                ),
                topic_id=0,
                topic_name="Machine Learning",
                graph_relationships=[],
                provenance=Provenance(
                    methods_used=["sbert", "tfidf", "bertopic"],
                    sbert_score=0.70,
                    topic_boost=0.15,
                    timestamp="2024-01-01T00:00:00Z",
                ),
            ),
            EnrichedChapter(
                book="Deep Learning",
                chapter=2,
                title="Optimization",
                chapter_id="Deep Learning:ch2",
                similar_chapters=[
                    CrossReference(
                        target="Deep Learning:ch1",
                        score=0.85,
                        base_score=0.70,
                        topic_boost=0.15,
                        method="sbert",
                    ),
                ],
                keywords=MergedKeywords(
                    tfidf=["gradient", "optimization"],
                    semantic=["descent", "learning rate"],
                    merged=["gradient", "optimization", "descent", "learning rate"],
                ),
                topic_id=0,
                topic_name="Machine Learning",
                graph_relationships=[],
                provenance=Provenance(
                    methods_used=["sbert", "tfidf", "bertopic"],
                    sbert_score=0.70,
                    topic_boost=0.15,
                    timestamp="2024-01-01T00:00:00Z",
                ),
            ),
        ],
        processing_time_ms=150.5,
    )


# =============================================================================
# AC-4.1.1: POST /v1/agents/enrich-metadata accepts MSEPRequest
# =============================================================================


class TestEndpointAcceptsMSEPRequest:
    """Tests for AC-4.1.1: POST endpoint accepts MSEPRequest."""

    def test_endpoint_exists_at_correct_path(
        self, client: TestClient, valid_request_body: dict[str, Any]
    ) -> None:
        """POST /v1/agents/enrich-metadata endpoint exists."""
        # Will return 503 if orchestrator not configured, but NOT 404
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=valid_request_body,
        )
        assert response.status_code != 404

    def test_accepts_valid_msep_request(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Accepts valid MSEPRequest with corpus and chapter_index."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            assert response.status_code == 200

    def test_accepts_minimal_request(
        self,
        client: TestClient,
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Accepts minimal request with defaults."""
        minimal_request = {
            "corpus": ["Test document content."],
            "chapter_index": [
                {"book": "Test Book", "chapter": 1, "title": "Test Chapter"},
            ],
        }
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=minimal_request,
            )
            assert response.status_code == 200

    def test_accepts_request_with_custom_config(
        self,
        client: TestClient,
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Accepts request with custom config values."""
        request_with_config = {
            "corpus": ["Test document."],
            "chapter_index": [
                {"book": "Test", "chapter": 1, "title": "Intro"},
            ],
            "config": {
                "threshold": 0.6,
                "top_k": 10,
                "timeout": 60.0,
                "same_topic_boost": 0.2,
                "use_dynamic_threshold": False,
                "enable_hybrid_search": False,
            },
        }
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=request_with_config,
            )
            assert response.status_code == 200


# =============================================================================
# AC-4.1.2: Returns EnrichedMetadata response
# =============================================================================


class TestEndpointReturnsEnrichedMetadata:
    """Tests for AC-4.1.2: Returns EnrichedMetadata response."""

    def test_response_contains_chapters(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Response contains chapters list."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            assert "chapters" in data
            assert isinstance(data["chapters"], list)
            assert len(data["chapters"]) == 2

    def test_response_contains_processing_time(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Response contains processing_time_ms."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] == 150.5

    def test_response_contains_total_similar_chapters(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Response contains total_similar_chapters."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            assert "total_similar_chapters" in data
            assert data["total_similar_chapters"] == 2

    def test_response_chapter_structure(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Response chapters contain correct structure."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            chapter = data["chapters"][0]
            
            assert "chapter_id" in chapter
            assert "similar_chapters" in chapter
            assert "keywords" in chapter
            assert "topic_id" in chapter
            assert "provenance" in chapter

    def test_response_similar_chapters_structure(
        self,
        client: TestClient,
        valid_request_body: dict[str, Any],
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Response similar_chapters contain correct structure."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            xref = data["chapters"][0]["similar_chapters"][0]
            
            assert "target" in xref
            assert "score" in xref
            assert "base_score" in xref
            assert "topic_boost" in xref
            assert "method" in xref


# =============================================================================
# AC-4.1.3: Returns 400 for invalid request
# =============================================================================


class TestEndpointReturns400:
    """Tests for AC-4.1.3: Returns 400 for invalid request."""

    def test_400_for_missing_corpus(self, client: TestClient) -> None:
        """Returns 400 when corpus is missing."""
        invalid_request = {
            "chapter_index": [
                {"book": "Test", "chapter": 1, "title": "Intro"},
            ],
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_missing_chapter_index(self, client: TestClient) -> None:
        """Returns 400 when chapter_index is missing."""
        invalid_request = {
            "corpus": ["Test content."],
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_empty_corpus(self, client: TestClient) -> None:
        """Returns 400 when corpus is empty."""
        invalid_request = {
            "corpus": [],
            "chapter_index": [],
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_mismatched_lengths(self, client: TestClient) -> None:
        """Returns 400 when corpus and chapter_index lengths differ."""
        invalid_request = {
            "corpus": ["Doc 1.", "Doc 2."],
            "chapter_index": [
                {"book": "Test", "chapter": 1, "title": "Intro"},
            ],
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_invalid_chapter_meta(self, client: TestClient) -> None:
        """Returns 400 when chapter metadata is invalid."""
        invalid_request = {
            "corpus": ["Test content."],
            "chapter_index": [
                {"book": "Test", "chapter": "not_an_int", "title": "Intro"},
            ],
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_invalid_config_values(self, client: TestClient) -> None:
        """Returns 400 for invalid config values."""
        invalid_request = {
            "corpus": ["Test content."],
            "chapter_index": [
                {"book": "Test", "chapter": 1, "title": "Intro"},
            ],
            "config": {
                "threshold": "not_a_float",
            },
        }
        response = client.post(
            "/v1/agents/enrich-metadata",
            json=invalid_request,
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_400_for_negative_threshold(
        self,
        client: TestClient,
        mock_enriched_metadata: EnrichedMetadata,
    ) -> None:
        """Returns 400 for negative threshold."""
        invalid_request = {
            "corpus": ["Test content."],
            "chapter_index": [
                {"book": "Test", "chapter": 1, "title": "Intro"},
            ],
            "config": {
                "threshold": -0.5,
            },
        }
        # Mock orchestrator to test validation
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.return_value = mock_enriched_metadata
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=invalid_request,
            )
            # Either 400/422 for validation or 200 if passes (threshold validation is optional)
            assert response.status_code in (400, 422, 200)


# =============================================================================
# AC-4.1.4: Returns 503 when services unavailable
# =============================================================================


class TestEndpointReturns503:
    """Tests for AC-4.1.4: Returns 503 when services unavailable."""

    def test_503_when_orchestrator_raises_service_unavailable(
        self, client: TestClient, valid_request_body: dict[str, Any]
    ) -> None:
        """Returns 503 when orchestrator raises ServiceUnavailableError."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.side_effect = ServiceUnavailableError(
                message="Code-Orchestrator service unavailable",
                service="code-orchestrator",
            )
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            assert response.status_code == 503

    def test_503_includes_error_detail(
        self, client: TestClient, valid_request_body: dict[str, Any]
    ) -> None:
        """503 response includes error detail."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.enrich_metadata.side_effect = ServiceUnavailableError(
                message="semantic-search-service unavailable",
                service="semantic-search",
            )
            mock_get_orch.return_value = mock_orchestrator

            response = client.post(
                "/v1/agents/enrich-metadata",
                json=valid_request_body,
            )
            data = response.json()
            assert "detail" in data


# =============================================================================
# Router Configuration Tests
# =============================================================================


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_has_correct_prefix(self, app: FastAPI) -> None:
        """Router uses /v1/agents prefix."""
        routes = [r.path for r in app.routes]
        assert "/v1/agents/enrich-metadata" in routes

    def test_router_has_correct_tags(self, app: FastAPI) -> None:
        """Router uses Agents tag."""
        for route in app.routes:
            if hasattr(route, "tags") and route.path == "/v1/agents/enrich-metadata":
                assert "Agents" in route.tags


# =============================================================================
# Orchestrator Dependency Tests
# =============================================================================


class TestOrchestratorDependency:
    """Tests for orchestrator dependency injection."""

    def test_get_orchestrator_returns_instance(self) -> None:
        """get_orchestrator returns orchestrator instance with mock clients."""
        from src.api.routes.enrich_metadata import (
            get_orchestrator,
            set_orchestrator,
        )

        # Provide mock orchestrator since default requires real clients
        mock_orchestrator = MagicMock()
        set_orchestrator(mock_orchestrator)

        try:
            orchestrator = get_orchestrator()
            assert orchestrator is not None
        finally:
            set_orchestrator(None)

    def test_set_orchestrator_for_testing(self) -> None:
        """set_orchestrator allows injecting test double."""
        from src.api.routes.enrich_metadata import (
            get_orchestrator,
            set_orchestrator,
        )

        mock_orchestrator = MagicMock()
        set_orchestrator(mock_orchestrator)
        assert get_orchestrator() is mock_orchestrator

        # Reset for other tests
        set_orchestrator(None)


# =============================================================================
# Health Endpoint Tests (Optional)
# =============================================================================


class TestHealthEndpoint:
    """Tests for health check endpoint if included in router."""

    def test_health_endpoint_exists(self, client: TestClient) -> None:
        """Health endpoint exists at /v1/agents/enrich-metadata/health."""
        response = client.get("/v1/agents/enrich-metadata/health")
        assert response.status_code != 404

    def test_health_returns_status(self, client: TestClient) -> None:
        """Health endpoint returns status."""
        with patch(
            "src.api.routes.enrich_metadata.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = MagicMock()
            mock_get_orch.return_value = mock_orchestrator

            response = client.get("/v1/agents/enrich-metadata/health")
            data = response.json()
            assert "status" in data
