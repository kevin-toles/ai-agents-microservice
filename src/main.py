"""
Main entry point for ai-agents service.

Creates the FastAPI application instance for uvicorn.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.cross_reference import router as cross_reference_router
from src.api.routes.enrich_metadata import router as enrich_metadata_router
from src.core.clients.neo4j import (
    create_neo4j_client_from_env,
    set_neo4j_client as set_global_neo4j_client,
)
from src.core.clients.semantic_search import SemanticSearchClient
from src.core.clients.content_adapter import SemanticSearchContentAdapter
from src.agents.cross_reference.nodes.search_taxonomy import (
    set_neo4j_client as set_node_neo4j_client,
)
from src.agents.cross_reference.nodes.retrieve_content import (
    set_content_client,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.
    
    On startup: Initialize Neo4j client and SemanticSearch content client
    On shutdown: Clean up resources
    
    Kitchen Brigade Architecture:
    - Neo4j client: Direct access for taxonomy search
    - SemanticSearchClient: Content retrieval via semantic-search-service
    """
    logger.info("Starting ai-agents service...")
    
    # Initialize Neo4j client
    neo4j_client = create_neo4j_client_from_env()
    if neo4j_client:
        # Set client in both locations for compatibility
        set_global_neo4j_client(neo4j_client)
        set_node_neo4j_client(neo4j_client)
        
        # Health check
        try:
            if await neo4j_client.health_check():
                logger.info("Neo4j: connected")
                app.state.neo4j_status = "connected"
            else:
                logger.warning("Neo4j: health check failed")
                app.state.neo4j_status = "unhealthy"
        except Exception as e:
            logger.error("Neo4j connection error: %s", e)
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
    logger.info("SemanticSearch content client: initialized")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down ai-agents service...")
    
    # Close semantic search client
    if semantic_search_client:
        await semantic_search_client.close()
        logger.info("SemanticSearch client closed")
    
    # Close Neo4j
    if neo4j_client:
        neo4j_client.close()
        logger.info("Neo4j driver closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        neo4j_status = getattr(app.state, "neo4j_status", "unknown")
        return {
            "status": "healthy",
            "service": "ai-agents",
            "dependencies": {
                "neo4j": neo4j_status,
            },
        }

    # Include routers
    app.include_router(cross_reference_router)
    app.include_router(enrich_metadata_router)

    return app


# Create application instance
app = create_app()
