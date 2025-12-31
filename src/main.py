"""
Main entry point for ai-agents service.

Creates the FastAPI application instance for uvicorn.

Kitchen Brigade Architecture:
    ai-agents acts as the Expeditor (:8082), orchestrating workflow
    execution and coordinating with downstream services.

Reference: WBS-AGT2 AC-2.3, WBS-AGT18, AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.cross_reference.nodes.retrieve_content import (
    set_content_client,
)
from src.agents.cross_reference.nodes.search_taxonomy import (
    set_neo4j_client as set_node_neo4j_client,
)
from src.api.error_handlers import register_error_handlers
from src.api.routes.cross_reference import router as cross_reference_router
from src.api.routes.enrich_metadata import router as enrich_metadata_router
from src.api.routes.functions import router as functions_router
from src.api.routes.health import router as health_router
from src.api.routes.health import set_service_start_time
from src.api.routes.pipelines import router as pipelines_router
from src.core.clients.content_adapter import SemanticSearchContentAdapter
from src.core.clients.neo4j import (
    create_neo4j_client_from_env,
)
from src.core.clients.neo4j import (
    set_neo4j_client as set_global_neo4j_client,
)
from src.core.clients.semantic_search import SemanticSearchClient
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger


# Configure structured logging on module load
configure_logging()
logger = get_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    On startup: Initialize Neo4j client and SemanticSearch content client
    On shutdown: Clean up resources

    Kitchen Brigade Architecture:
    - Neo4j client: Direct access for taxonomy search
    - SemanticSearchClient: Content retrieval via semantic-search-service

    WBS-AGT18: Sets service start time for health endpoint uptime tracking.
    """
    settings = get_settings()
    logger.info("Starting ai-agents service", port=settings.port, role="Expeditor")

    # Set service start time for health endpoint
    set_service_start_time()

    # Initialize Neo4j client
    neo4j_client = create_neo4j_client_from_env()
    if neo4j_client:
        # Set client in both locations for compatibility
        set_global_neo4j_client(neo4j_client)
        set_node_neo4j_client(neo4j_client)

        # Health check
        try:
            if await neo4j_client.health_check():
                logger.info("Neo4j connected successfully")
                app.state.neo4j_status = "connected"
            else:
                logger.warning("Neo4j health check failed")
                app.state.neo4j_status = "unhealthy"
        except Exception as e:
            logger.error("Neo4j connection error", error=str(e))
            app.state.neo4j_status = f"error: {e}"
    else:
        logger.warning("Neo4j client not configured")
        app.state.neo4j_status = "not_configured"

    app.state.neo4j_client = neo4j_client

    # Initialize SemanticSearchClient for content retrieval
    # Kitchen Brigade: ai-agents (Expeditor) → semantic-search (Cookbook) → Neo4j (Pantry)
    semantic_search_client = SemanticSearchClient()
    content_adapter = SemanticSearchContentAdapter(semantic_search_client)
    set_content_client(content_adapter)
    app.state.semantic_search_client = semantic_search_client
    logger.info("SemanticSearch content client initialized")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down ai-agents service")

    # Close semantic search client
    if semantic_search_client:
        await semantic_search_client.close()
        logger.info("SemanticSearch client closed")

    # Close Neo4j
    if neo4j_client:
        neo4j_client.close()
        logger.info("Neo4j driver closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    WBS-AGT18: Registers all API routers including:
    - functions_router: POST /v1/functions/{name}/run
    - pipelines_router: POST /v1/pipelines/{name}/run
    - health_router: GET /health, /health/ready, /health/live
    """
    app = FastAPI(
        title="AI Agents Service",
        description="LangGraph-based AI agents for scholarly cross-referencing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register error handlers (WBS-AGT18 AC-18.5)
    register_error_handlers(app)

    # Include routers (WBS-AGT18)
    app.include_router(functions_router)
    app.include_router(pipelines_router)
    app.include_router(health_router)

    # Legacy routers
    app.include_router(cross_reference_router)
    app.include_router(enrich_metadata_router)

    return app


# Create application instance
app = create_app()
