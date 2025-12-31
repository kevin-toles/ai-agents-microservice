"""Function API routes.

WBS-AGT18: API Routes - AC-18.1: POST /v1/functions/{name}/run executes single function.

Provides REST API endpoints for executing agent functions.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
Service Endpoints:
- POST /v1/functions/{name}/run - Execute function
- GET /v1/functions - List available functions

Anti-Patterns Avoided:
- ANTI_PATTERN_ANALYSIS §3.1: No bare except clauses
- ANTI_PATTERN_ANALYSIS §4.1: Cognitive complexity < 15 per function
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.error_handlers import FunctionExecutionError, FunctionNotFoundError
from src.functions import (
    AnalyzeArtifactFunction,
    DecomposeTaskFunction,
    ExtractStructureFunction,
    GenerateCodeFunction,
    SummarizeContentFunction,
    ValidateAgainstSpecFunction,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    prefix="/v1/functions",
    tags=["Functions"],
)


# =============================================================================
# Function Registry - Maps URL names to function classes
# =============================================================================

FUNCTION_REGISTRY: dict[str, type] = {
    "extract-structure": ExtractStructureFunction,
    "summarize-content": SummarizeContentFunction,
    "generate-code": GenerateCodeFunction,
    "analyze-artifact": AnalyzeArtifactFunction,
    "validate-against-spec": ValidateAgainstSpecFunction,
    "decompose-task": DecomposeTaskFunction,
    "synthesize-outputs": SummarizeContentFunction,  # Uses same base
    "cross-reference": ExtractStructureFunction,  # Placeholder
}


# =============================================================================
# Request/Response Models - AC-18.4
# =============================================================================

class FunctionRunRequest(BaseModel):
    """Request model for function execution.

    Attributes:
        input: Function input data (schema varies by function)
        preset: Optional preset for context budget (e.g., "S1", "D4")
    """

    input: dict[str, Any] = Field(
        ...,
        description="Function input data",
    )
    preset: str | None = Field(
        default=None,
        description="Preset for context budget selection",
    )


class FunctionRunResponse(BaseModel):
    """Response model for function execution.

    Attributes:
        result: Function output data
        function_name: Name of the executed function
        processing_time_ms: Execution time in milliseconds
        preset_used: Preset that was used
    """

    result: dict[str, Any] = Field(
        ...,
        description="Function output data",
    )
    function_name: str = Field(
        ...,
        description="Name of the executed function",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Execution time in milliseconds",
    )
    preset_used: str | None = Field(
        default=None,
        description="Preset that was used for execution",
    )


class FunctionInfo(BaseModel):
    """Information about a registered function.

    Attributes:
        name: Function name (URL path)
        description: Function description
        endpoint: Full endpoint path
    """

    name: str = Field(
        ...,
        description="Function name",
    )
    description: str = Field(
        default="",
        description="Function description",
    )
    endpoint: str = Field(
        ...,
        description="Full endpoint path",
    )


class FunctionListResponse(BaseModel):
    """Response model for listing functions.

    Attributes:
        functions: List of available functions
        total: Total number of functions
    """

    functions: list[FunctionInfo] = Field(
        default_factory=list,
        description="List of available functions",
    )
    total: int = Field(
        default=0,
        description="Total number of functions",
    )


# =============================================================================
# Function Executor Factory
# =============================================================================

async def get_function_executor(
    name: str,
    input_data: dict[str, Any],
    preset: str | None = None,
) -> dict[str, Any]:
    """Execute a function and return its result.

    This is the actual executor that runs the function.
    Can be mocked in tests.

    Args:
        name: Function name
        input_data: Function input data
        preset: Optional preset

    Returns:
        Function output as dict

    Raises:
        FunctionNotFoundError: If function not in registry
        FunctionExecutionError: If execution fails
    """
    if name not in FUNCTION_REGISTRY:
        raise FunctionNotFoundError(name)

    function_class = FUNCTION_REGISTRY[name]

    try:
        # Create function instance
        function = function_class()

        # Execute function
        result = await function.run(**input_data)

        # Convert result to dict if needed
        if hasattr(result, "model_dump"):
            output: dict[str, Any] = result.model_dump()
            return output
        elif hasattr(result, "__dict__"):
            output = dict(result.__dict__)
            return output
        else:
            return {"result": str(result)}

    except Exception as e:
        logger.exception("Function execution failed", extra={"function": name})
        raise FunctionExecutionError(name, str(e)) from e


# =============================================================================
# API Endpoints
# =============================================================================

@router.get(
    "",
    response_model=FunctionListResponse,
    summary="List available functions",
    description="Returns a list of all registered agent functions.",
)
async def list_functions() -> FunctionListResponse:
    """List all available agent functions.

    Returns:
        FunctionListResponse with all registered functions
    """
    functions = []

    for name, func_class in FUNCTION_REGISTRY.items():
        functions.append(
            FunctionInfo(
                name=name,
                description=func_class.__doc__ or "",
                endpoint=f"/v1/functions/{name}/run",
            )
        )

    return FunctionListResponse(
        functions=functions,
        total=len(functions),
    )


@router.post(
    "/{name}/run",
    response_model=FunctionRunResponse,
    summary="Execute a function",
    description="Executes the specified agent function with the provided input.",
    responses={
        404: {"description": "Function not found"},
        422: {"description": "Validation error"},
        500: {"description": "Execution error"},
    },
)
async def run_function(
    name: str,
    request: FunctionRunRequest,
) -> FunctionRunResponse:
    """Execute a single agent function.

    Args:
        name: Function name (from URL path)
        request: Function run request with input data

    Returns:
        FunctionRunResponse with execution result

    Raises:
        FunctionNotFoundError: If function not in registry
        FunctionExecutionError: If execution fails
    """
    # Validate function exists
    if name not in FUNCTION_REGISTRY:
        raise FunctionNotFoundError(name)

    logger.info(
        "Executing function",
        extra={
            "function": name,
            "preset": request.preset,
        },
    )

    # Time the execution
    start_time = time.perf_counter()

    # Execute function
    result = await get_function_executor(
        name=name,
        input_data=request.input,
        preset=request.preset,
    )

    # Calculate processing time
    processing_time = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Function execution completed",
        extra={
            "function": name,
            "processing_time_ms": processing_time,
        },
    )

    return FunctionRunResponse(
        result=result,
        function_name=name,
        processing_time_ms=processing_time,
        preset_used=request.preset,
    )


__all__ = [
    "FUNCTION_REGISTRY",
    "FunctionInfo",
    "FunctionListResponse",
    "FunctionRunRequest",
    "FunctionRunResponse",
    "get_function_executor",
    "router",
]
