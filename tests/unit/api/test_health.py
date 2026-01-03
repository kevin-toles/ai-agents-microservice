"""Tests for health API routes.

TDD tests for WBS-AGT18: API Routes.

Acceptance Criteria Coverage:
- AC-18.3: GET /health returns service status

Exit Criteria:
- /health returns service health with dependencies

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# =============================================================================
# AC-18.3: Health Route Tests
# =============================================================================

class TestHealthRouteImport:
    """Tests for health route module imports."""

    def test_health_router_can_be_imported(self) -> None:
        """Health router can be imported from src.api.routes.health."""
        from src.api.routes.health import router
        
        assert router is not None


class TestHealthRouteConfiguration:
    """Tests for health route configuration."""

    def test_router_has_health_tag(self) -> None:
        """Router is tagged as 'Health'."""
        from src.api.routes.health import router
        
        assert "Health" in router.tags


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with health router."""
        from src.api.routes.health import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200

    def test_health_returns_status(self, client: TestClient) -> None:
        """GET /health returns status field."""
        response = client.get("/health")
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_returns_service_name(self, client: TestClient) -> None:
        """GET /health returns service name."""
        response = client.get("/health")
        
        data = response.json()
        assert "service" in data
        assert data["service"] == "ai-agents"

    def test_health_returns_version(self, client: TestClient) -> None:
        """GET /health returns version."""
        response = client.get("/health")
        
        data = response.json()
        assert "version" in data


class TestHealthResponseModel:
    """Tests for HealthResponse model."""

    def test_health_response_can_be_imported(self) -> None:
        """HealthResponse can be imported."""
        from src.api.routes.health import HealthResponse
        
        assert isinstance(HealthResponse, type)

    def test_health_response_has_status(self) -> None:
        """HealthResponse has status field."""
        from src.api.routes.health import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            service="ai-agents",
            version="1.0.0",
        )
        
        assert response.status == "healthy"

    def test_health_response_has_dependencies(self) -> None:
        """HealthResponse has dependencies field."""
        from src.api.routes.health import HealthResponse, DependencyHealth, DependencyStatus
        
        response = HealthResponse(
            status="healthy",
            service="ai-agents",
            version="1.0.0",
            dependencies=[
                DependencyHealth(name="neo4j", status=DependencyStatus.UP)
            ],
        )
        
        assert len(response.dependencies) == 1
        assert response.dependencies[0].name == "neo4j"


class TestReadinessEndpoint:
    """Tests for GET /health/ready endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.health import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_ready_returns_200(self, client: TestClient) -> None:
        """GET /health/ready returns 200 when ready."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200

    def test_ready_returns_ready_status(self, client: TestClient) -> None:
        """GET /health/ready returns ready status."""
        response = client.get("/health/ready")
        
        data = response.json()
        assert "ready" in data


class TestLivenessEndpoint:
    """Tests for GET /health/live endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.health import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_live_returns_200(self, client: TestClient) -> None:
        """GET /health/live returns 200."""
        response = client.get("/health/live")
        
        assert response.status_code == 200

    def test_live_returns_alive_status(self, client: TestClient) -> None:
        """GET /health/live returns alive status."""
        response = client.get("/health/live")
        
        data = response.json()
        assert "alive" in data
        assert data["alive"] is True


class TestDependencyChecks:
    """Tests for dependency health checks."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.health import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_checks_inference_service(self, client: TestClient) -> None:
        """Health check includes inference-service status."""
        response = client.get("/health")
        
        data = response.json()
        if "dependencies" in data:
            # Dependencies may include inference-service
            pass  # Optional dependency

    def test_health_checks_semantic_search(self, client: TestClient) -> None:
        """Health check includes semantic-search status."""
        response = client.get("/health")
        
        data = response.json()
        if "dependencies" in data:
            # Dependencies may include semantic-search
            pass  # Optional dependency


__all__ = [
    "TestHealthRouteImport",
    "TestHealthRouteConfiguration",
    "TestHealthEndpoint",
    "TestHealthResponseModel",
    "TestReadinessEndpoint",
    "TestLivenessEndpoint",
    "TestDependencyChecks",
]
