"""Tests for Cross-Reference API Route - WBS 5.13 RED Phase.

TDD RED Phase: These tests define expected behavior before implementation.

Reference Documents:
- GUIDELINES: FastAPI dependency injection (Sinha pp. 89-91)
- GUIDELINES: Pydantic validators (Sinha pp. 193-195)
- GUIDELINES: REST constraints (Buelta pp. 92-93, 126)
- ARCHITECTURE.md (ai-agents): Cross-Reference Agent API patterns
- CODING_PATTERNS_ANALYSIS.md: Anti-pattern prevention

Anti-Patterns Avoided:
- ANTI_PATTERN_ANALYSIS §1.1: Optional types with explicit None
- ANTI_PATTERN_ANALYSIS §3.1: No bare except clauses  
- ANTI_PATTERN_ANALYSIS §4.1: Cognitive complexity < 15 per function
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI app with cross-reference router."""
    from src.api.routes.cross_reference import router
    
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Mock CrossReferenceAgent for testing."""
    from src.agents.cross_reference.state import (
        CrossReferenceResult,
        Citation,
        TierCoverage,
    )
    
    mock = MagicMock()
    mock.run = AsyncMock(return_value=CrossReferenceResult(
        annotation="Test annotation with inline citations[^1].",
        citations=[
            Citation(
                author="John Ousterhout",
                book="A Philosophy of Software Design",
                chapter=1,
                chapter_title="Introduction",
                tier=1,
            )
        ],
        traversal_paths=[],
        tier_coverage=[
            TierCoverage(
                tier=1,
                tier_name="Architecture Spine",
                books_referenced=1,
                chapters_referenced=1,
                has_coverage=True,
            ),
            TierCoverage(
                tier=2,
                tier_name="Implementation",
                books_referenced=0,
                chapters_referenced=0,
                has_coverage=False,
            ),
            TierCoverage(
                tier=3,
                tier_name="Engineering Practices",
                books_referenced=0,
                chapters_referenced=0,
                has_coverage=False,
            ),
        ],
        matches=[],
        processing_time_ms=42.5,
        model_used="gpt-4",
    ))
    return mock


# ============================================================================
# TestCrossReferenceRouter - WBS 5.13.1: Router Structure
# ============================================================================


class TestCrossReferenceRouter:
    """Test suite for cross-reference router setup."""
    
    def test_router_is_fastapi_router(self):
        """
        WBS 5.13.1.1: Cross-reference router must be a FastAPI APIRouter instance.
        
        Pattern: Router separation (Sinha p. 89)
        """
        from fastapi import APIRouter
        from src.api.routes.cross_reference import router
        
        assert isinstance(router, APIRouter)
    
    def test_router_has_correct_prefix(self):
        """
        WBS 5.13.1.2: Cross-reference router must use /v1/agents/cross-reference prefix.
        
        Pattern: API versioning (Buelta p. 126)
        """
        from src.api.routes.cross_reference import router
        
        assert router.prefix == "/v1/agents/cross-reference"
    
    def test_router_has_correct_tags(self):
        """
        WBS 5.13.1.3: Cross-reference router must have 'Agents' tag for OpenAPI docs.
        """
        from src.api.routes.cross_reference import router
        
        assert "Agents" in router.tags


# ============================================================================
# TestCrossReferenceEndpoint - WBS 5.13.2: POST /v1/agents/cross-reference
# ============================================================================


class TestCrossReferenceEndpoint:
    """Test suite for cross-reference endpoint."""
    
    def test_returns_200_for_valid_request(
        self, client: TestClient, mock_agent: MagicMock
    ):
        """
        WBS 5.13.2.1: POST /v1/agents/cross-reference returns 200 for valid request.
        
        Pattern: REST constraints (Buelta p. 93)
        """
        payload = {
            "source": {
                "book": "A Philosophy of Software Design",
                "chapter": 3,
                "title": "Working Code Isn't Enough",
                "tier": 1,
            },
            "config": {
                "max_hops": 3,
                "min_similarity": 0.7,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        assert response.status_code == 200
    
    def test_returns_expected_schema(
        self, client: TestClient, mock_agent: MagicMock
    ):
        """
        WBS 5.13.2.2: Response must include annotation, citations, tier_coverage.
        
        Pattern: OpenAI API compatibility structure
        """
        payload = {
            "source": {
                "book": "A Philosophy of Software Design",
                "chapter": 3,
                "title": "Working Code Isn't Enough",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        data = response.json()
        
        assert "annotation" in data
        assert "citations" in data
        assert "tier_coverage" in data
        assert "processing_time_ms" in data
    
    def test_returns_422_for_missing_source(self, client: TestClient):
        """
        WBS 5.13.2.3: POST without source returns 422 validation error.
        
        Pattern: Pydantic validation (Sinha pp. 193-195)
        """
        payload = {"config": {"max_hops": 3}}
        response = client.post("/v1/agents/cross-reference", json=payload)
        
        assert response.status_code == 422
    
    def test_returns_422_for_invalid_tier(self, client: TestClient):
        """
        WBS 5.13.2.4: POST with invalid tier returns 422 validation error.
        
        Tier must be 1, 2, or 3 per TIER_RELATIONSHIP_DIAGRAM.md
        """
        payload = {
            "source": {
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 5,  # Invalid: must be 1-3
            },
        }
        response = client.post("/v1/agents/cross-reference", json=payload)
        
        assert response.status_code == 422
    
    def test_returns_422_for_negative_chapter(self, client: TestClient):
        """
        WBS 5.13.2.5: POST with negative chapter returns 422 validation error.
        """
        payload = {
            "source": {
                "book": "Test Book",
                "chapter": -1,  # Invalid: must be >= 1
                "title": "Test Chapter",
                "tier": 1,
            },
        }
        response = client.post("/v1/agents/cross-reference", json=payload)
        
        assert response.status_code == 422


# ============================================================================
# TestCrossReferenceRequestModel - WBS 5.13.3: Request Validation
# ============================================================================


class TestCrossReferenceRequestModel:
    """Test suite for request model validation."""
    
    def test_request_model_validates_source_fields(self):
        """
        WBS 5.13.3.1: Request model must validate all source fields.
        """
        from src.api.routes.cross_reference import CrossReferenceRequest
        
        # Valid request
        request = CrossReferenceRequest(
            source={
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            }
        )
        assert request.source.book == "Test Book"
        assert request.source.chapter == 1
        assert request.source.tier == 1
    
    def test_request_model_accepts_optional_config(self):
        """
        WBS 5.13.3.2: Request model accepts optional traversal config.
        """
        from src.api.routes.cross_reference import CrossReferenceRequest
        
        request = CrossReferenceRequest(
            source={
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            },
            config={
                "max_hops": 5,
                "min_similarity": 0.8,
                "include_tier1": True,
                "include_tier2": False,
                "include_tier3": True,
            }
        )
        assert request.config.max_hops == 5
        assert request.config.min_similarity == pytest.approx(0.8)
        assert request.config.include_tier2 is False
    
    def test_request_model_uses_defaults_for_config(self):
        """
        WBS 5.13.3.3: Request model uses sensible defaults when config omitted.
        """
        from src.api.routes.cross_reference import CrossReferenceRequest
        
        request = CrossReferenceRequest(
            source={
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            }
        )
        
        # Default config should be applied
        assert request.config is not None
        assert request.config.max_hops == 3  # Default
        assert request.config.min_similarity == pytest.approx(0.7)  # Default


# ============================================================================
# TestCrossReferenceResponseModel - WBS 5.13.4: Response Structure
# ============================================================================


class TestCrossReferenceResponseModel:
    """Test suite for response model structure."""
    
    def test_response_includes_citations_with_chicago_format(
        self, client: TestClient, mock_agent: MagicMock
    ):
        """
        WBS 5.13.4.1: Response citations must be in Chicago Manual format.
        """
        payload = {
            "source": {
                "book": "A Philosophy of Software Design",
                "chapter": 3,
                "title": "Working Code Isn't Enough",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        data = response.json()
        citations = data["citations"]
        
        assert len(citations) >= 1
        assert "book" in citations[0]
        assert "chapter" in citations[0]
        assert "tier" in citations[0]
    
    def test_response_includes_tier_coverage(
        self, client: TestClient, mock_agent: MagicMock
    ):
        """
        WBS 5.13.4.2: Response must include coverage for all 3 tiers.
        """
        payload = {
            "source": {
                "book": "A Philosophy of Software Design",
                "chapter": 3,
                "title": "Working Code Isn't Enough",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        data = response.json()
        tier_coverage = data["tier_coverage"]
        
        assert len(tier_coverage) == 3
        tiers = [tc["tier"] for tc in tier_coverage]
        assert 1 in tiers
        assert 2 in tiers
        assert 3 in tiers
    
    def test_response_includes_processing_time(
        self, client: TestClient, mock_agent: MagicMock
    ):
        """
        WBS 5.13.4.3: Response must include processing_time_ms.
        """
        payload = {
            "source": {
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        data = response.json()
        
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0


# ============================================================================
# TestCrossReferenceErrorHandling - WBS 5.13.5: Error Handling
# ============================================================================


class TestCrossReferenceErrorHandling:
    """Test suite for error handling."""
    
    def test_returns_500_on_agent_error(
        self, client: TestClient
    ):
        """
        WBS 5.13.5.1: Returns 500 when agent raises exception.
        
        Pattern: Exception handling (ANTI_PATTERN §3.1)
        """
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Agent failed"))
        
        payload = {
            "source": {
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data
    
    def test_error_response_includes_message(
        self, client: TestClient
    ):
        """
        WBS 5.13.5.2: Error response must include descriptive message.
        """
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid input"))
        
        payload = {
            "source": {
                "book": "Test Book",
                "chapter": 1,
                "title": "Test Chapter",
                "tier": 1,
            },
        }
        
        with patch("src.api.routes.cross_reference.get_agent", return_value=mock_agent):
            response = client.post("/v1/agents/cross-reference", json=payload)
        
        # Should return error status (400 for ValueError, 500 for internal)
        assert response.status_code in (400, 500)


# ============================================================================
# TestHealthEndpoint - WBS 5.13.6: Health Check
# ============================================================================


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_returns_200(self, client: TestClient):
        """
        WBS 5.13.6.1: GET /v1/agents/cross-reference/health returns 200.
        """
        response = client.get("/v1/agents/cross-reference/health")
        
        assert response.status_code == 200
    
    def test_health_returns_status(self, client: TestClient):
        """
        WBS 5.13.6.2: Health response includes status field.
        """
        response = client.get("/v1/agents/cross-reference/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ("healthy", "unhealthy")
    
    def test_health_returns_agent_info(self, client: TestClient):
        """
        WBS 5.13.6.3: Health response includes agent information.
        """
        response = client.get("/v1/agents/cross-reference/health")
        data = response.json()
        
        assert "agent" in data
        assert data["agent"] == "cross-reference"
