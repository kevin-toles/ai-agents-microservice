"""
Main entry point for ai-agents service.

Creates the FastAPI application instance for uvicorn.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.cross_reference import router as cross_reference_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Agents Service",
        description="LangGraph-based AI agents for scholarly cross-referencing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "ai-agents"}

    # Include routers
    app.include_router(cross_reference_router)

    return app


# Create application instance
app = create_app()
