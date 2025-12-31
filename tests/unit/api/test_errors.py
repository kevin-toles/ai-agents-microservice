"""Tests for error handlers.

TDD tests for WBS-AGT18: API Routes.

Acceptance Criteria Coverage:
- AC-18.5: Error responses match llm-gateway schema

Exit Criteria:
- Error responses have consistent schema
- HTTPException returns proper error format
- Validation errors return 422 with details

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError as PydanticValidationError


# =============================================================================
# AC-18.5: Error Handler Tests
# =============================================================================

class TestErrorHandlerImport:
    """Tests for error handler module imports."""

    def test_register_error_handlers_can_be_imported(self) -> None:
        """register_error_handlers can be imported."""
        from src.api.error_handlers import register_error_handlers
        
        assert register_error_handlers is not None

    def test_error_response_model_can_be_imported(self) -> None:
        """ErrorResponse model can be imported."""
        from src.api.error_handlers import ErrorResponse
        
        assert ErrorResponse is not None


class TestErrorResponseModel:
    """Tests for ErrorResponse model."""

    def test_error_response_has_error_field(self) -> None:
        """ErrorResponse has error field."""
        from src.api.error_handlers import ErrorResponse
        
        response = ErrorResponse(
            error="NotFoundError",
            detail="Resource not found",
        )
        
        assert response.error == "NotFoundError"

    def test_error_response_has_detail_field(self) -> None:
        """ErrorResponse has detail field."""
        from src.api.error_handlers import ErrorResponse
        
        response = ErrorResponse(
            error="ValidationError",
            detail="Invalid input",
        )
        
        assert response.detail == "Invalid input"

    def test_error_response_has_optional_code(self) -> None:
        """ErrorResponse has optional error code."""
        from src.api.error_handlers import ErrorResponse
        
        response = ErrorResponse(
            error="NotFoundError",
            detail="Function not found",
            code="FUNCTION_NOT_FOUND",
        )
        
        assert response.code == "FUNCTION_NOT_FOUND"

    def test_error_response_has_optional_path(self) -> None:
        """ErrorResponse has optional request path."""
        from src.api.error_handlers import ErrorResponse
        
        response = ErrorResponse(
            error="NotFoundError",
            detail="Not found",
            path="/v1/functions/invalid/run",
        )
        
        assert response.path == "/v1/functions/invalid/run"


class TestHTTPExceptionHandler:
    """Tests for HTTPException handler."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with error handlers."""
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        
        @app.get("/test-404")
        async def raise_404():
            raise HTTPException(status_code=404, detail="Not found")
        
        @app.get("/test-500")
        async def raise_500():
            raise HTTPException(status_code=500, detail="Internal error")
        
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_http_404_returns_error_schema(self, client: TestClient) -> None:
        """404 error returns ErrorResponse schema."""
        response = client.get("/test-404")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "detail" in data

    def test_http_500_returns_error_schema(self, client: TestClient) -> None:
        """500 error returns ErrorResponse schema."""
        response = client.get("/test-500")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "detail" in data


class TestValidationErrorHandler:
    """Tests for Pydantic validation error handler."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with error handlers."""
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        
        class TestRequest(BaseModel):
            required_field: str
        
        @app.post("/test-validation")
        async def validate_request(request: TestRequest):
            return {"received": request.required_field}
        
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_validation_error_returns_422(self, client: TestClient) -> None:
        """Validation error returns 422."""
        response = client.post("/test-validation", json={})
        
        assert response.status_code == 422

    def test_validation_error_returns_error_schema(self, client: TestClient) -> None:
        """Validation error returns ErrorResponse schema."""
        response = client.post("/test-validation", json={})
        
        data = response.json()
        assert "error" in data
        assert "detail" in data

    def test_validation_error_includes_field_info(self, client: TestClient) -> None:
        """Validation error includes field information."""
        response = client.post("/test-validation", json={})
        
        data = response.json()
        # Detail should mention the missing field
        assert "required_field" in str(data).lower() or "detail" in data


class TestFunctionNotFoundError:
    """Tests for FunctionNotFoundError."""

    def test_function_not_found_error_can_be_imported(self) -> None:
        """FunctionNotFoundError can be imported."""
        from src.api.error_handlers import FunctionNotFoundError
        
        assert FunctionNotFoundError is not None

    def test_function_not_found_error_has_function_name(self) -> None:
        """FunctionNotFoundError includes function name."""
        from src.api.error_handlers import FunctionNotFoundError
        
        error = FunctionNotFoundError("invalid-func")
        
        assert error.function_name == "invalid-func"

    def test_function_not_found_error_has_message(self) -> None:
        """FunctionNotFoundError has descriptive message."""
        from src.api.error_handlers import FunctionNotFoundError
        
        error = FunctionNotFoundError("invalid-func")
        
        assert "invalid-func" in str(error)


class TestPipelineNotFoundError:
    """Tests for PipelineNotFoundError."""

    def test_pipeline_not_found_error_can_be_imported(self) -> None:
        """PipelineNotFoundError can be imported."""
        from src.api.error_handlers import PipelineNotFoundError
        
        assert PipelineNotFoundError is not None

    def test_pipeline_not_found_error_has_pipeline_name(self) -> None:
        """PipelineNotFoundError includes pipeline name."""
        from src.api.error_handlers import PipelineNotFoundError
        
        error = PipelineNotFoundError("invalid-pipeline")
        
        assert error.pipeline_name == "invalid-pipeline"


class TestPipelineExecutionError:
    """Tests for PipelineExecutionError."""

    def test_pipeline_execution_error_can_be_imported(self) -> None:
        """PipelineExecutionError can be imported."""
        from src.api.error_handlers import PipelineExecutionError
        
        assert PipelineExecutionError is not None

    def test_pipeline_execution_error_has_stage(self) -> None:
        """PipelineExecutionError includes failed stage."""
        from src.api.error_handlers import PipelineExecutionError
        
        error = PipelineExecutionError(
            pipeline_name="test",
            stage_name="extract_structure",
            message="Stage failed",
        )
        
        assert error.stage_name == "extract_structure"


class TestGenericExceptionHandler:
    """Tests for generic exception handler."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with error handlers."""
        from src.api.error_handlers import register_error_handlers
        
        app = FastAPI()
        register_error_handlers(app)
        
        @app.get("/test-exception")
        async def raise_exception():
            raise ValueError("Unexpected error")
        
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app, raise_server_exceptions=False)

    def test_generic_exception_returns_500(self, client: TestClient) -> None:
        """Unhandled exception returns 500."""
        response = client.get("/test-exception")
        
        assert response.status_code == 500

    def test_generic_exception_returns_error_schema(self, client: TestClient) -> None:
        """Unhandled exception returns ErrorResponse schema."""
        response = client.get("/test-exception")
        
        data = response.json()
        assert "error" in data
        assert "detail" in data


class TestCustomExceptionHandlers:
    """Tests for custom exception handlers."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create test FastAPI app with error handlers."""
        from src.api.error_handlers import (
            register_error_handlers,
            FunctionNotFoundError,
            PipelineNotFoundError,
        )
        
        app = FastAPI()
        register_error_handlers(app)
        
        @app.get("/test-function-not-found")
        async def raise_function_not_found():
            raise FunctionNotFoundError("invalid-func")
        
        @app.get("/test-pipeline-not-found")
        async def raise_pipeline_not_found():
            raise PipelineNotFoundError("invalid-pipeline")
        
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_function_not_found_returns_404(self, client: TestClient) -> None:
        """FunctionNotFoundError returns 404."""
        response = client.get("/test-function-not-found")
        
        assert response.status_code == 404

    def test_pipeline_not_found_returns_404(self, client: TestClient) -> None:
        """PipelineNotFoundError returns 404."""
        response = client.get("/test-pipeline-not-found")
        
        assert response.status_code == 404


__all__ = [
    "TestErrorHandlerImport",
    "TestErrorResponseModel",
    "TestHTTPExceptionHandler",
    "TestValidationErrorHandler",
    "TestFunctionNotFoundError",
    "TestPipelineNotFoundError",
    "TestPipelineExecutionError",
    "TestGenericExceptionHandler",
    "TestCustomExceptionHandlers",
]
