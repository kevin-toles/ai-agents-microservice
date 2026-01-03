"""Tests for function API routes.

TDD tests for WBS-AGT18: API Routes.

Acceptance Criteria Coverage:
- AC-18.1: POST /v1/functions/{name}/run executes single function
- AC-18.4: Request validation with Pydantic

Exit Criteria:
- POST /v1/functions/extract_structure/run returns StructuredOutput
- Invalid function name returns 404 with error schema

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from pydantic import BaseModel


# =============================================================================
# AC-18.1: Function Route Tests
# =============================================================================

class TestFunctionRouteImport:
    """Tests for function route module imports."""

    def test_functions_router_can_be_imported(self) -> None:
        """Functions router can be imported from src.api.routes.functions."""
        from src.api.routes.functions import router
        
        assert router is not None

    def test_function_registry_can_be_imported(self) -> None:
        """FUNCTION_REGISTRY can be imported for function lookup."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert FUNCTION_REGISTRY is not None
        assert isinstance(FUNCTION_REGISTRY, dict)


class TestFunctionRouteConfiguration:
    """Tests for function route configuration."""

    def test_router_has_correct_prefix(self) -> None:
        """Router has /v1/functions prefix."""
        from src.api.routes.functions import router
        
        assert router.prefix == "/v1/functions"

    def test_router_has_functions_tag(self) -> None:
        """Router is tagged as 'Functions'."""
        from src.api.routes.functions import router
        
        assert "Functions" in router.tags


class TestFunctionRegistryContents:
    """Tests for function registry contents."""

    def test_registry_contains_extract_structure(self) -> None:
        """Registry contains extract-structure function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "extract-structure" in FUNCTION_REGISTRY

    def test_registry_contains_summarize_content(self) -> None:
        """Registry contains summarize-content function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "summarize-content" in FUNCTION_REGISTRY

    def test_registry_contains_generate_code(self) -> None:
        """Registry contains generate-code function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "generate-code" in FUNCTION_REGISTRY

    def test_registry_contains_analyze_artifact(self) -> None:
        """Registry contains analyze-artifact function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "analyze-artifact" in FUNCTION_REGISTRY

    def test_registry_contains_validate_against_spec(self) -> None:
        """Registry contains validate-against-spec function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "validate-against-spec" in FUNCTION_REGISTRY

    def test_registry_contains_decompose_task(self) -> None:
        """Registry contains decompose-task function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "decompose-task" in FUNCTION_REGISTRY

    def test_registry_contains_synthesize_outputs(self) -> None:
        """Registry contains synthesize-outputs function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "synthesize-outputs" in FUNCTION_REGISTRY

    def test_registry_contains_cross_reference(self) -> None:
        """Registry contains cross-reference function."""
        from src.api.routes.functions import FUNCTION_REGISTRY
        
        assert "cross-reference" in FUNCTION_REGISTRY


class TestFunctionRunEndpoint:
    """Tests for POST /v1/functions/{name}/run endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with functions router."""
        from src.api.routes.functions import router
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_run_extract_structure_returns_200(self, client: TestClient) -> None:
        """POST /v1/functions/extract-structure/run returns 200."""
        request_data = {
            "input": {
                "content": "# Test\n\nSome content",
                "artifact_type": "markdown",
            },
        }
        
        async def mock_executor(**kwargs):
            await asyncio.sleep(0)  # Yield control to event loop
            return {
                "headings": [{"level": 1, "text": "Test"}],
                "sections": [],
                "code_blocks": [],
            }
        
        with patch("src.api.routes.functions.get_function_executor", side_effect=mock_executor):
            response = client.post(
                "/v1/functions/extract-structure/run",
                json=request_data,
            )
        
        assert response.status_code == 200

    def test_run_extract_structure_returns_structured_output(self, client: TestClient) -> None:
        """POST /v1/functions/extract-structure/run returns StructuredOutput."""
        request_data = {
            "input": {
                "content": "# Test\n\nSome content",
                "artifact_type": "markdown",
            },
        }
        
        async def mock_executor(**kwargs):
            await asyncio.sleep(0)  # Yield control to event loop
            return {
                "headings": [{"level": 1, "text": "Test"}],
                "sections": [{"heading": "Test", "content": "Some content"}],
                "code_blocks": [],
            }
        
        with patch("src.api.routes.functions.get_function_executor", side_effect=mock_executor):
            response = client.post(
                "/v1/functions/extract-structure/run",
                json=request_data,
            )
        
        data = response.json()
        assert "result" in data
        assert "headings" in data["result"]

    def test_run_invalid_function_returns_404(self, client: TestClient) -> None:
        """POST /v1/functions/invalid-func/run returns 404."""
        request_data = {"input": {"content": "test"}}
        
        response = client.post(
            "/v1/functions/invalid-func/run",
            json=request_data,
        )
        
        assert response.status_code == 404

    def test_run_invalid_function_returns_error_schema(self, client: TestClient) -> None:
        """404 response matches error schema."""
        request_data = {"input": {"content": "test"}}
        
        response = client.post(
            "/v1/functions/invalid-func/run",
            json=request_data,
        )
        
        data = response.json()
        assert "error" in data
        assert "detail" in data


# =============================================================================
# AC-18.4: Request Validation Tests
# =============================================================================

class TestFunctionRequestValidation:
    """Tests for Pydantic request validation."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.functions import router
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
            "/v1/functions/extract-structure/run",
            json={},
        )
        
        assert response.status_code == 422

    def test_invalid_json_returns_422(self, client: TestClient) -> None:
        """Invalid JSON returns 422."""
        response = client.post(
            "/v1/functions/extract-structure/run",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == 422


class TestFunctionRequestModel:
    """Tests for FunctionRunRequest model."""

    def test_function_run_request_can_be_imported(self) -> None:
        """FunctionRunRequest can be imported."""
        from src.api.routes.functions import FunctionRunRequest
        
        assert isinstance(FunctionRunRequest, type)

    def test_function_run_request_requires_input(self) -> None:
        """FunctionRunRequest requires input field."""
        from src.api.routes.functions import FunctionRunRequest
        
        with pytest.raises(Exception):  # ValidationError
            FunctionRunRequest()

    def test_function_run_request_accepts_preset(self) -> None:
        """FunctionRunRequest accepts optional preset."""
        from src.api.routes.functions import FunctionRunRequest
        
        request = FunctionRunRequest(
            input={"content": "test"},
            preset="S1",
        )
        
        assert request.preset == "S1"


class TestFunctionResponseModel:
    """Tests for FunctionRunResponse model."""

    def test_function_run_response_can_be_imported(self) -> None:
        """FunctionRunResponse can be imported."""
        from src.api.routes.functions import FunctionRunResponse
        
        assert isinstance(FunctionRunResponse, type)

    def test_function_run_response_has_result(self) -> None:
        """FunctionRunResponse has result field."""
        from src.api.routes.functions import FunctionRunResponse
        
        response = FunctionRunResponse(
            result={"output": "test"},
            function_name="test-func",
        )
        
        assert response.result == {"output": "test"}

    def test_function_run_response_has_function_name(self) -> None:
        """FunctionRunResponse includes function name."""
        from src.api.routes.functions import FunctionRunResponse
        
        response = FunctionRunResponse(
            result={},
            function_name="extract-structure",
        )
        
        assert response.function_name == "extract-structure"

    def test_function_run_response_has_processing_time(self) -> None:
        """FunctionRunResponse includes processing time."""
        from src.api.routes.functions import FunctionRunResponse
        
        response = FunctionRunResponse(
            result={},
            function_name="test",
            processing_time_ms=150.5,
        )
        
        assert response.processing_time_ms == pytest.approx(150.5)


class TestFunctionListEndpoint:
    """Tests for GET /v1/functions endpoint."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app."""
        from src.api.routes.functions import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_list_functions_returns_200(self, client: TestClient) -> None:
        """GET /v1/functions returns 200."""
        response = client.get("/v1/functions")
        
        assert response.status_code == 200

    def test_list_functions_returns_all_functions(self, client: TestClient) -> None:
        """GET /v1/functions returns all registered functions."""
        response = client.get("/v1/functions")
        
        data = response.json()
        assert "functions" in data
        assert len(data["functions"]) >= 8  # At least 8 core functions


__all__ = [
    "TestFunctionRouteImport",
    "TestFunctionRouteConfiguration",
    "TestFunctionRegistryContents",
    "TestFunctionRunEndpoint",
    "TestFunctionRequestValidation",
    "TestFunctionRequestModel",
    "TestFunctionResponseModel",
    "TestFunctionListEndpoint",
]
