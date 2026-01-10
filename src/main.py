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
from src.api.routes.protocols import router as protocols_router
from src.api.routes.well_known import router as well_known_router
from src.api.routes.a2a import router as a2a_router
from src.core.clients.content_adapter import SemanticSearchContentAdapter
# PCON-4: Use consolidated Neo4j client from src/clients/
from src.clients.neo4j_client import (
    Neo4jClient,
    create_neo4j_client_from_env,
    set_neo4j_client as set_global_neo4j_client,
)
from src.core.clients.semantic_search import (
    SemanticSearchClient,
    set_semantic_search_client,
)
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.config.feature_flags import ProtocolFeatureFlags

# PCON-5: WBS-AGT21-24 client imports
from src.clients.code_reference import CodeReferenceClient, CodeReferenceConfig
from src.clients.book_passage import BookPassageClient, BookPassageClientConfig
from src.retrieval.unified_retriever import UnifiedRetriever, UnifiedRetrieverConfig

# WBS-PI5b: MCP Server Lifecycle Integration
from src.mcp.agent_functions_server import create_agent_functions_mcp_server


# Configure structured logging on module load
configure_logging()
logger = get_logger(__name__)


# =============================================================================
# Lifespan Helper Functions (Refactored for reduced cognitive complexity)
# =============================================================================


async def _init_neo4j_client(app: FastAPI) -> Neo4jClient | None:
    """Initialize Neo4j client and set global references."""
    neo4j_client = await create_neo4j_client_from_env()
    if not neo4j_client:
        logger.warning("Neo4j client not configured")
        app.state.neo4j_status = "not_configured"
        return None

    set_global_neo4j_client(neo4j_client)
    set_node_neo4j_client(neo4j_client)
    app.state.neo4j_status = await _check_neo4j_health(neo4j_client)
    return neo4j_client


async def _check_neo4j_health(client: Neo4jClient) -> str:
    """Check Neo4j health and return status string."""
    try:
        if await client.health_check():
            logger.info("Neo4j connected successfully")
            return "connected"
        logger.warning("Neo4j health check failed")
        return "unhealthy"
    except Exception as e:
        logger.error("Neo4j connection error", error=str(e))
        return f"error: {e}"


async def _init_code_ref_client() -> CodeReferenceClient | None:
    """Initialize CodeReferenceClient (WBS-AGT21)."""
    code_ref_config = CodeReferenceConfig.from_env()
    if not code_ref_config.registry_path:
        logger.warning("CodeReferenceClient not configured (missing CODE_REFERENCE_REGISTRY)")
        return None

    try:
        client = CodeReferenceClient(code_ref_config)
        await client.__aenter__()
        logger.info("CodeReferenceClient initialized")
        return client
    except Exception as e:
        logger.warning("CodeReferenceClient initialization failed", error=str(e))
        return None


async def _init_book_passage_client(settings) -> BookPassageClient | None:
    """Initialize BookPassageClient (WBS-AGT23)."""
    config = BookPassageClientConfig(
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.book_passages_collection,
        books_dir=settings.book_passages_dir,
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
        embedding_model=settings.embedding_model,
    )
    try:
        client = BookPassageClient(config)
        await client.connect()
        logger.info("BookPassageClient initialized")
        return client
    except Exception as e:
        logger.warning("BookPassageClient initialization failed", error=str(e))
        return None


async def _close_client(client, name: str, close_method: str = "close") -> None:
    """Safely close a client with logging."""
    if not client:
        return
    try:
        if close_method == "__aexit__":
            await client.__aexit__(None, None, None)
        else:
            await getattr(client, close_method)()
        logger.info(f"{name} closed")
    except Exception as e:
        logger.warning(f"{name} close failed", error=str(e))


async def _init_mcp_server(flags: ProtocolFeatureFlags) -> dict | None:
    """Initialize MCP server if feature flags enabled.
    
    WBS-PI5b.1: FastAPI lifecycle integration
    AC-PI5b.2, AC-PI5b.3: Feature flag guards
    
    Args:
        flags: Protocol feature flags configuration
        
    Returns:
        MCP server dict if enabled, None otherwise
    """
    if not (flags.mcp_enabled and flags.mcp_server_enabled):
        logger.info("MCP server disabled", 
                    mcp_enabled=flags.mcp_enabled,
                    mcp_server_enabled=flags.mcp_server_enabled)
        return None
    
    try:
        mcp_server = await create_agent_functions_mcp_server()
        logger.info("MCP server started", server_name=mcp_server["name"])
        return mcp_server
    except Exception as e:
        logger.error("MCP server initialization failed", error=str(e))
        return None


async def _close_mcp_server(mcp_server: dict | None) -> None:
    """Clean up MCP server resources.
    
    WBS-PI5b.4: Clean shutdown
    
    Args:
        mcp_server: MCP server instance or None
    """
    if mcp_server:
        logger.info("MCP server stopped")


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
    neo4j_client = await _init_neo4j_client(app)
    app.state.neo4j_client = neo4j_client

    # Initialize SemanticSearchClient for content retrieval
    # Kitchen Brigade: ai-agents (Expeditor) → semantic-search (Cookbook) → Neo4j (Pantry)
    semantic_search_client = SemanticSearchClient()
    content_adapter = SemanticSearchContentAdapter(semantic_search_client)
    set_content_client(content_adapter)
    # PCON-6: Inject client for cross_reference tools (MCP cross_reference fix)
    set_semantic_search_client(semantic_search_client)
    app.state.semantic_search_client = semantic_search_client
    logger.info("SemanticSearch content client initialized (MCP tools injected)")

    # =========================================================================
    # PCON-5: Initialize WBS-AGT21-24 Clients
    # =========================================================================

    # Initialize CodeReferenceClient (WBS-AGT21)
    code_ref_client = await _init_code_ref_client()
    app.state.code_ref_client = code_ref_client

    # Initialize BookPassageClient (WBS-AGT23)
    book_passage_client = await _init_book_passage_client(settings)
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

    # WBS-PI5b: MCP Server Lifecycle Integration
    mcp_server = await _init_mcp_server(ProtocolFeatureFlags())
    app.state.mcp_server = mcp_server

    yield

    # Cleanup on shutdown
    logger.info("Shutting down ai-agents service")

    # WBS-PI5b.4: MCP Server shutdown
    await _close_mcp_server(app.state.mcp_server)

    # PCON-5: Close WBS-AGT21-24 clients
    await _close_client(code_ref_client, "CodeReferenceClient", "__aexit__")
    await _close_client(book_passage_client, "BookPassageClient", "close")

    # Close semantic search client
    if semantic_search_client:
        await semantic_search_client.close()
        # PCON-6: Reset global client reference
        set_semantic_search_client(None)
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
    app.include_router(protocols_router)
    app.include_router(health_router)

    # A2A Protocol Integration (WBS-PI2, WBS-PI3)
    app.include_router(well_known_router)
    app.include_router(a2a_router)

    # Legacy routers
    app.include_router(cross_reference_router)
    app.include_router(enrich_metadata_router)

    # Inter-AI Conversation Orchestration (INTER_AI_ORCHESTRATION.md)
    app.include_router(conversation_router)

    return app


# Create application instance
app = create_app()
