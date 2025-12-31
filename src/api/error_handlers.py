"""Error handlers for API routes.

WBS-AGT18: API Routes - AC-18.5: Error responses match llm-gateway schema.

Provides consistent error response format across all API endpoints.
Follows the ErrorResponse schema pattern from llm-gateway.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Anti-Patterns Avoided:
- ANTI_PATTERN_ANALYSIS §3.1: No bare except clauses
- ANTI_PATTERN_ANALYSIS §4.1: Cognitive complexity < 15 per function
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# Type alias for exception handler
ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]


# =============================================================================
# Error Response Model - AC-18.5
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model.

    Matches llm-gateway error schema for consistency.

    Attributes:
        error: Error type/category
        detail: Human-readable error description
        code: Optional machine-readable error code
        path: Optional request path that caused the error
    """

    error: str = Field(
        ...,
        description="Error type or category",
    )
    detail: str = Field(
        ...,
        description="Human-readable error description",
    )
    code: str | None = Field(
        default=None,
        description="Machine-readable error code",
    )
    path: str | None = Field(
        default=None,
        description="Request path that caused the error",
    )


# =============================================================================
# Custom Exceptions
# =============================================================================

class FunctionNotFoundError(Exception):
    """Raised when a function is not found in the registry.

    Attributes:
        function_name: Name of the function that was not found
    """

    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        super().__init__(f"Function '{function_name}' not found in registry")


class PipelineNotFoundError(Exception):
    """Raised when a pipeline is not found in the registry.

    Attributes:
        pipeline_name: Name of the pipeline that was not found
    """

    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name
        super().__init__(f"Pipeline '{pipeline_name}' not found in registry")


class PipelineExecutionError(Exception):
    """Raised when a pipeline execution fails.

    Attributes:
        pipeline_name: Name of the pipeline that failed
        stage_name: Name of the stage that failed
        message: Error message
    """

    def __init__(
        self,
        pipeline_name: str,
        stage_name: str,
        message: str,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.stage_name = stage_name
        self.message = message
        super().__init__(f"Pipeline '{pipeline_name}' failed at stage '{stage_name}': {message}")


class FunctionExecutionError(Exception):
    """Raised when a function execution fails.

    Attributes:
        function_name: Name of the function that failed
        message: Error message
    """

    def __init__(
        self,
        function_name: str,
        message: str,
    ) -> None:
        self.function_name = function_name
        self.message = message
        super().__init__(f"Function '{function_name}' failed: {message}")


# =============================================================================
# Exception Handlers
# =============================================================================

async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle HTTPException with ErrorResponse schema.

    Args:
        request: FastAPI request object
        exc: HTTPException raised

    Returns:
        JSONResponse with ErrorResponse format
    """
    error_type = {
        400: "BadRequest",
        401: "Unauthorized",
        403: "Forbidden",
        404: "NotFound",
        422: "ValidationError",
        500: "InternalServerError",
        503: "ServiceUnavailable",
    }.get(exc.status_code, "Error")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=error_type,
            detail=str(exc.detail),
            path=str(request.url.path),
        ).model_dump(),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: RequestValidationError raised

    Returns:
        JSONResponse with ErrorResponse format and field details
    """
    # Extract field errors
    errors = exc.errors()
    field_errors = []

    for error in errors:
        loc = ".".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Invalid value")
        field_errors.append(f"{loc}: {msg}")

    detail = "; ".join(field_errors) if field_errors else "Validation error"

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            detail=detail,
            code="VALIDATION_ERROR",
            path=str(request.url.path),
        ).model_dump(),
    )


async def function_not_found_handler(
    request: Request,
    exc: FunctionNotFoundError,
) -> JSONResponse:
    """Handle FunctionNotFoundError.

    Args:
        request: FastAPI request object
        exc: FunctionNotFoundError raised

    Returns:
        JSONResponse with 404 status
    """
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NotFound",
            detail=f"Function '{exc.function_name}' not found",
            code="FUNCTION_NOT_FOUND",
            path=str(request.url.path),
        ).model_dump(),
    )


async def pipeline_not_found_handler(
    request: Request,
    exc: PipelineNotFoundError,
) -> JSONResponse:
    """Handle PipelineNotFoundError.

    Args:
        request: FastAPI request object
        exc: PipelineNotFoundError raised

    Returns:
        JSONResponse with 404 status
    """
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NotFound",
            detail=f"Pipeline '{exc.pipeline_name}' not found",
            code="PIPELINE_NOT_FOUND",
            path=str(request.url.path),
        ).model_dump(),
    )


async def pipeline_execution_handler(
    request: Request,
    exc: PipelineExecutionError,
) -> JSONResponse:
    """Handle PipelineExecutionError.

    Args:
        request: FastAPI request object
        exc: PipelineExecutionError raised

    Returns:
        JSONResponse with 500 status
    """
    logger.error(
        "Pipeline execution failed",
        extra={
            "pipeline": exc.pipeline_name,
            "stage": exc.stage_name,
            "error": exc.message,
        },
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="PipelineExecutionError",
            detail=f"Pipeline failed at stage '{exc.stage_name}': {exc.message}",
            code="PIPELINE_EXECUTION_ERROR",
            path=str(request.url.path),
        ).model_dump(),
    )


async def function_execution_handler(
    request: Request,
    exc: FunctionExecutionError,
) -> JSONResponse:
    """Handle FunctionExecutionError.

    Args:
        request: FastAPI request object
        exc: FunctionExecutionError raised

    Returns:
        JSONResponse with 500 status
    """
    logger.error(
        "Function execution failed",
        extra={
            "function": exc.function_name,
            "error": exc.message,
        },
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="FunctionExecutionError",
            detail=f"Function execution failed: {exc.message}",
            code="FUNCTION_EXECUTION_ERROR",
            path=str(request.url.path),
        ).model_dump(),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle uncaught exceptions.

    Args:
        request: FastAPI request object
        exc: Exception raised

    Returns:
        JSONResponse with 500 status
    """
    logger.exception(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "error_type": type(exc).__name__,
        },
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred",
            code="INTERNAL_ERROR",
            path=str(request.url.path),
        ).model_dump(),
    )


# =============================================================================
# Registration Function
# =============================================================================

def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Cast handlers to match FastAPI's expected signature
    app.add_exception_handler(
        HTTPException,
        http_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        FunctionNotFoundError,
        function_not_found_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        PipelineNotFoundError,
        pipeline_not_found_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        PipelineExecutionError,
        pipeline_execution_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        FunctionExecutionError,
        function_execution_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        Exception,
        generic_exception_handler,
    )


__all__ = [
    "ErrorResponse",
    "FunctionExecutionError",
    "FunctionNotFoundError",
    "PipelineExecutionError",
    "PipelineNotFoundError",
    "register_error_handlers",
]
