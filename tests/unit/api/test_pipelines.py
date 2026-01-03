"""Tests for pipeline API routes.

TDD tests for WBS-AGT18: API Routes.

Acceptance Criteria Coverage:
- AC-18.2: POST /v1/pipelines/{name}/run executes pipeline
- AC-18.4: Request validation with Pydantic

Exit Criteria:
- POST /v1/pipelines/chapter-summarization/run returns CitedContent
- Invalid pipeline name returns 404 with error schema

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


# =============================================================================
# AC-18.2: Pipeline Route Tests
# =============================================================================

class TestPipelineRouteImport:
    """Tests for pipeline route module imports."""

    def test_pipelines_router_can_be_imported(self) -> None:
        """Pipelines router can be imported from src.api.routes.pipelines."""
        from src.api.routes.pipelines import router
        
        assert router is not None

    def test_pipeline_registry_can_be_imported(self) -> None:
        """PIPELINE_REGISTRY can be imported for pipeline lookup."""
        from src.api.routes.pipelines import PIPELINE_REGISTRY
        
        assert PIPELINE_REGISTRY is not None
        assert isinstance(PIPELINE_REGISTRY, dict)


class TestPipelineRouteConfiguration:
    """Tests for pipeline route configuration."""

    def test_router_has_correct_prefix(self) -> None:
        """Router has /v1/pipelines prefix."""
        from src.api.routes.pipelines import router
        
        assert router.prefix == "/v1/pipelines"

    def test_router_has_pipelines_tag(self) -> None:
        """Router is tagged as 'Pipelines'."""
        from src.api.routes.pipelines import router
        
        assert "Pipelines" in router.tags


class TestPipelineRegistryContents:
    """Tests for pipeline registry contents."""

    def test_registry_contains_chapter_summarization(self) -> None:
        """Registry contains chapter-summarization pipeline."""
        from src.api.routes.pipelines import PIPELINE_REGISTRY
        
        assert "chapter-summarization" in PIPELINE_REGISTRY

    def test_registry_contains_code_generation(self) -> None:
        """Registry contains code-generation pipeline."""
        from src.api.routes.pipelines import PIPELINE_REGISTRY
        
        assert "code-generation" in PIPELINE_REGISTRY


class TestPipelineRunEndpoint:
    """Tests for POST /v1/pipelines/{name}/run endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with pipelines router."""
        from src.api.routes.pipelines import router
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_run_chapter_summarization_returns_200(self, client: TestClient) -> None:
        """POST /v1/pipelines/chapter-summarization/run returns 200."""
        request_data = {
            "input": {
                "chapter_content": "# Test Chapter\n\nContent here.",
                "chapter_id": "ch_001",
            },
        }
        
        async def mock_executor(**kwargs):
            await asyncio.sleep(0)  # Yield control
            return {
                "content": "Summary with [^1] citation.",
                "citations": [{"marker": 1, "source": "test"}],
                "stages_executed": ["extract", "summarize"],
            }
        
        with patch("src.api.routes.pipelines.get_pipeline_executor", side_effect=mock_executor):
            response = client.post(
                "/v1/pipelines/chapter-summarization/run",
                json=request_data,
            )
        
        assert response.status_code == 200

    def test_run_chapter_summarization_returns_cited_content(self, client: TestClient) -> None:
        """POST /v1/pipelines/chapter-summarization/run returns CitedContent."""
        request_data = {
            "input": {
                "chapter_content": "# Test Chapter\n\nContent here.",
                "chapter_id": "ch_001",
            },
        }
        
        async def mock_executor(**kwargs):
            await asyncio.sleep(0)  # Yield control
            return {
                "content": "Summary with [^1] citation.",
                "citations": [{"marker": 1, "source": "test"}],
                "stages_executed": ["extract", "summarize"],
            }
        
        with patch("src.api.routes.pipelines.get_pipeline_executor", side_effect=mock_executor):
            response = client.post(
                "/v1/pipelines/chapter-summarization/run",
                json=request_data,
            )
        
        data = response.json()
        assert "result" in data

    def test_run_code_generation_returns_200(self, client: TestClient) -> None:
        """POST /v1/pipelines/code-generation/run returns 200."""
        request_data = {
            "input": {
                "user_request": "Create a repository pattern implementation",
                "target_language": "python",
            },
        }
        
        async def mock_executor(**kwargs):
            await asyncio.sleep(0)  # Yield control
            return {
                "code": "class Repository:\n    pass",
                "language": "python",
                "stages_executed": ["analyze", "generate"],
            }
        
        with patch("src.api.routes.pipelines.get_pipeline_executor", side_effect=mock_executor):
            response = client.post(
                "/v1/pipelines/code-generation/run",
                json=request_data,
            )
        
        assert response.status_code == 200

    def test_run_invalid_pipeline_returns_404(self, client: TestClient) -> None:
        """POST /v1/pipelines/invalid-pipeline/run returns 404."""
        request_data = {"input": {"content": "test"}}
        
        response = client.post(
            "/v1/pipelines/invalid-pipeline/run",
            json=request_data,
        )
        
        assert response.status_code == 404

    def test_run_invalid_pipeline_returns_error_schema(self, client: TestClient) -> None:
        """404 response matches error schema."""
        request_data = {"input": {"content": "test"}}
        
        response = client.post(
            "/v1/pipelines/invalid-pipeline/run",
            json=request_data,
        )
        
        data = response.json()
        assert "error" in data
        assert "detail" in data


# =============================================================================
# AC-18.4: Request Validation Tests
# =============================================================================

class TestPipelineRequestValidation:
    """Tests for Pydantic request validation."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.pipelines import router
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_missing_input_returns_422(self, client: TestClient) -> None:
        """Missing input field returns 422 validation error."""
        response = client.post(
            "/v1/pipelines/chapter-summarization/run",
            json={},
        )
        
        assert response.status_code == 422


class TestPipelineRequestModel:
    """Tests for PipelineRunRequest model."""

    def test_pipeline_run_request_can_be_imported(self) -> None:
        """PipelineRunRequest can be imported."""
        from src.api.routes.pipelines import PipelineRunRequest
        
        assert isinstance(PipelineRunRequest, type)

    def test_pipeline_run_request_requires_input(self) -> None:
        """PipelineRunRequest requires input field."""
        from src.api.routes.pipelines import PipelineRunRequest
        
        with pytest.raises(Exception):  # ValidationError
            PipelineRunRequest()

    def test_pipeline_run_request_accepts_preset(self) -> None:
        """PipelineRunRequest accepts optional preset."""
        from src.api.routes.pipelines import PipelineRunRequest
        
        request = PipelineRunRequest(
            input={"chapter_content": "test"},
            preset="quality",
        )
        
        assert request.preset == "quality"


class TestPipelineResponseModel:
    """Tests for PipelineRunResponse model."""

    def test_pipeline_run_response_can_be_imported(self) -> None:
        """PipelineRunResponse can be imported."""
        from src.api.routes.pipelines import PipelineRunResponse
        
        assert isinstance(PipelineRunResponse, type)

    def test_pipeline_run_response_has_result(self) -> None:
        """PipelineRunResponse has result field."""
        from src.api.routes.pipelines import PipelineRunResponse
        
        response = PipelineRunResponse(
            result={"content": "test"},
            pipeline_name="test-pipeline",
        )
        
        assert response.result == {"content": "test"}

    def test_pipeline_run_response_has_pipeline_name(self) -> None:
        """PipelineRunResponse includes pipeline name."""
        from src.api.routes.pipelines import PipelineRunResponse
        
        response = PipelineRunResponse(
            result={},
            pipeline_name="chapter-summarization",
        )
        
        assert response.pipeline_name == "chapter-summarization"

    def test_pipeline_run_response_has_stages_completed(self) -> None:
        """PipelineRunResponse includes stages completed."""
        from src.api.routes.pipelines import PipelineRunResponse
        
        response = PipelineRunResponse(
            result={},
            pipeline_name="test",
            stages_completed=["stage1", "stage2"],
        )
        
        assert response.stages_completed == ["stage1", "stage2"]

    def test_pipeline_run_response_has_processing_time(self) -> None:
        """PipelineRunResponse includes processing time."""
        from src.api.routes.pipelines import PipelineRunResponse
        
        response = PipelineRunResponse(
            result={},
            pipeline_name="test",
            processing_time_ms=1500.0,
        )
        
        assert response.processing_time_ms == pytest.approx(1500.0)


class TestPipelineListEndpoint:
    """Tests for GET /v1/pipelines endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.pipelines import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_list_pipelines_returns_200(self, client: TestClient) -> None:
        """GET /v1/pipelines returns 200."""
        response = client.get("/v1/pipelines")
        
        assert response.status_code == 200

    def test_list_pipelines_returns_all_pipelines(self, client: TestClient) -> None:
        """GET /v1/pipelines returns all registered pipelines."""
        response = client.get("/v1/pipelines")
        
        data = response.json()
        assert "pipelines" in data
        assert len(data["pipelines"]) >= 2  # At least chapter-summarization and code-generation


class TestPipelineStatusEndpoint:
    """Tests for GET /v1/pipelines/{name}/status endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.pipelines import router
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_get_pipeline_status_returns_200(self, client: TestClient) -> None:
        """GET /v1/pipelines/chapter-summarization/status returns 200."""
        response = client.get("/v1/pipelines/chapter-summarization/status")
        
        assert response.status_code == 200

    def test_get_pipeline_status_returns_stage_info(self, client: TestClient) -> None:
        """GET /v1/pipelines/{name}/status returns stage information."""
        response = client.get("/v1/pipelines/chapter-summarization/status")
        
        data = response.json()
        assert "name" in data
        assert "stages" in data

    def test_get_invalid_pipeline_status_returns_404(self, client: TestClient) -> None:
        """GET /v1/pipelines/invalid/status returns 404."""
        response = client.get("/v1/pipelines/invalid-pipeline/status")
        
        assert response.status_code == 404


__all__ = [
    "TestPipelineRouteImport",
    "TestPipelineRouteConfiguration",
    "TestPipelineRegistryContents",
    "TestPipelineRunEndpoint",
    "TestPipelineRequestValidation",
    "TestPipelineRequestModel",
    "TestPipelineResponseModel",
    "TestPipelineListEndpoint",
    "TestPipelineStatusEndpoint",
]
