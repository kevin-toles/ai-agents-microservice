"""
Main entry point for ai-agents service.

Creates the FastAPI application instance for uvicorn.

Kitchen Brigade Architecture:
    ai-agents acts as the Expeditor (:8082), orchestrating workflow
    execution and coordinating with downstream services.

Reference: WBS-AGT2 AC-2.3, WBS-AGT18, AGENT_FUNCTIONS_ARCHITECTURE.md
PCON-4: Consolidated Neo4j client from src/clients/neo4j_client.py
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
from src.api.routes.conversation import router as conversation_router
from src.api.routes.cross_reference import router as cross_reference_router
from src.api.routes.enrich_metadata import router as enrich_metadata_router
from src.api.routes.functions import router as functions_router
from src.api.routes.health import router as health_router
from src.api.routes.health import set_service_start_time
from src.api.routes.pipelines import router as pipelines_router
from src.core.clients.content_adapter import SemanticSearchContentAdapter
# PCON-4: Use consolidated Neo4j client from src/clients/
from src.clients.neo4j_client import (
    Neo4jClient,
    create_neo4j_client_from_env,
    set_neo4j_client as set_global_neo4j_client,
)
from src.core.clients.semantic_search import SemanticSearchClient
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger

# PCON-5: WBS-AGT21-24 client imports
from src.clients.code_reference import CodeReferenceClient, CodeReferenceConfig
from src.clients.book_passage import BookPassageClient, BookPassageClientConfig
from src.retrieval.unified_retriever import UnifiedRetriever, UnifiedRetrieverConfig


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
    PCON-4: Uses consolidated Neo4jClient from src/clients/neo4j_client.py
    """
    settings = get_settings()
    logger.info("Starting ai-agents service", port=settings.port, role="Expeditor")

    # Set service start time for health endpoint
    set_service_start_time()

    # Initialize Neo4j client (PCON-4: consolidated client)
    neo4j_client = await create_neo4j_client_from_env()
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

    # =========================================================================
    # PCON-5: Initialize WBS-AGT21-24 Clients
    # =========================================================================

    # Initialize CodeReferenceClient (WBS-AGT21)
    code_ref_config = CodeReferenceConfig.from_env()
    code_ref_client = None
    if code_ref_config.registry_path:
        try:
            code_ref_client = CodeReferenceClient(code_ref_config)
            await code_ref_client.__aenter__()  # Initialize HTTP client
            logger.info("CodeReferenceClient initialized")
        except Exception as e:
            logger.warning("CodeReferenceClient initialization failed", error=str(e))
            code_ref_client = None
    else:
        logger.warning("CodeReferenceClient not configured (missing CODE_REFERENCE_REGISTRY)")
    app.state.code_ref_client = code_ref_client

    # Initialize BookPassageClient (WBS-AGT23)
    book_passage_config = BookPassageClientConfig(
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.book_passages_collection,
        books_dir=settings.book_passages_dir,
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
        embedding_model=settings.embedding_model,
    )
    book_passage_client = None
    try:
        book_passage_client = BookPassageClient(book_passage_config)
        await book_passage_client.connect()
        logger.info("BookPassageClient initialized")
    except Exception as e:
        logger.warning("BookPassageClient initialization failed", error=str(e))
        book_passage_client = None
    app.state.book_passage_client = book_passage_client

    # Initialize UnifiedRetriever (WBS-AGT24)
    unified_retriever = UnifiedRetriever(
        config=UnifiedRetrieverConfig(),
        code_client=code_ref_client,
        neo4j_client=neo4j_client,
        book_client=book_passage_client,
    )
    app.state.unified_retriever = unified_retriever
    logger.info("UnifiedRetriever initialized", 
                code_client=code_ref_client is not None,
                neo4j_client=neo4j_client is not None,
                book_client=book_passage_client is not None)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down ai-agents service")

    # PCON-5: Close WBS-AGT21-24 clients
    if code_ref_client:
        try:
            await code_ref_client.__aexit__(None, None, None)
            logger.info("CodeReferenceClient closed")
        except Exception as e:
            logger.warning("CodeReferenceClient close failed", error=str(e))

    if book_passage_client:
        try:
            await book_passage_client.close()
            logger.info("BookPassageClient closed")
        except Exception as e:
            logger.warning("BookPassageClient close failed", error=str(e))

    # Close semantic search client
    if semantic_search_client:
        await semantic_search_client.close()
        logger.info("SemanticSearch client closed")

    # Close Neo4j (PCON-4: async close)
    if neo4j_client:
        await neo4j_client.close()
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

    # Inter-AI Conversation Orchestration (INTER_AI_ORCHESTRATION.md)
    app.include_router(conversation_router)

    return app


# Create application instance
app = create_app()
