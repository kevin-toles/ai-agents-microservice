"""API routes module for ai-agents service.

WBS-AGT18: API Routes - Centralized route exports.

This module exports all API routers for registration in main.py.
"""

from src.api.routes.cross_reference import router as cross_reference_router
from src.api.routes.enrich_metadata import router as enrich_metadata_router
from src.api.routes.functions import router as functions_router
from src.api.routes.health import router as health_router
from src.api.routes.pipelines import router as pipelines_router


__all__ = [
    "cross_reference_router",
    "enrich_metadata_router",
    "functions_router",
    "health_router",
    "pipelines_router",
]
