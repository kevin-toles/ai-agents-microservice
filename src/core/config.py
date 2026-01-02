"""Application configuration using Pydantic Settings.

Follows Pydantic Settings pattern from CODING_PATTERNS_ANALYSIS.md Phase 1.
Environment variables are loaded with AI_AGENTS_ prefix.

Kitchen Brigade Service Ports (AGENT_FUNCTIONS_ARCHITECTURE.md):
- 8080: llm-gateway (Router)
- 8081: semantic-search-service (Cookbook)
- 8082: ai-agents (Expeditor) - THIS SERVICE
- 8083: Code-Orchestrator-Service (Sous Chef)
- 8084: audit-service (Auditor)
- 8085: inference-service (Line Cook)

Reference: WBS-AGT2 AC-2.1, AC-2.5
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Pattern: Pydantic Settings with Environment Variables
    Source: CODING_PATTERNS_ANALYSIS.md Phase 1, Comp_Static_Analysis #18

    Kitchen Brigade Architecture:
        ai-agents acts as the Expeditor, orchestrating workflow execution
        and coordinating with downstream services.

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Architecture Overview
    """

    # Service configuration
    service_name: str = "ai-agents"
    port: int = Field(default=8082, description="Service port (Expeditor role)")
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ==========================================================================
    # Kitchen Brigade Service URLs (AC-2.1, AC-2.5)
    # ==========================================================================

    # Router - Single entry point for external requests
    llm_gateway_url: str = Field(
        default="http://localhost:8080",
        description="LLM Gateway service URL (Router :8080)"
    )

    # Cookbook - Semantic search and retrieval
    semantic_search_url: str = Field(
        default="http://localhost:8081",
        description="Semantic Search service URL (Cookbook :8081)"
    )

    # Sous Chef - HuggingFace models (CodeT5+, GraphCodeBERT)
    code_orchestrator_url: str = Field(
        default="http://localhost:8083",
        description="Code Orchestrator service URL (Sous Chef :8083)"
    )

    # Auditor - Citation tracking and footnote generation
    audit_service_url: str = Field(
        default="http://localhost:8084",
        description="Audit service URL (Auditor :8084)"
    )

    # Line Cook - Local LLM inference via llama.cpp
    inference_service_url: str = Field(
        default="http://localhost:8085",
        description="Inference service URL (Line Cook :8085)"
    )

    # Code Reference Engine - ai-platform-data integration
    code_reference_engine_url: str = Field(
        default="http://localhost:8086",
        description="Code Reference Engine URL"
    )

    # Neo4j configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI"
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: SecretStr = Field(
        default=SecretStr("devpassword"),
        description="Neo4j password"
    )

    # Agent configuration
    default_llm_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Default LLM model for agent reasoning"
    )
    max_traversal_hops: int = Field(
        default=3,
        description="Maximum graph traversal depth"
    )
    default_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for search results"
    )
    allow_graph_cycles: bool = Field(
        default=True,
        description="Allow revisiting tiers during traversal"
    )

    # Feature flags - Phase 6 Production Ready (WBS 6.7)
    enable_cross_reference_agent: bool = Field(
        default=True,  # Enabled after WBS 6.1-6.3 validation
        description="Enable Cross-Reference Agent endpoint"
    )

    # Request timeouts
    llm_timeout_seconds: int = Field(default=60, description="LLM request timeout")
    search_timeout_seconds: int = Field(default=30, description="Search request timeout")
    graph_timeout_seconds: int = Field(default=30, description="Graph query timeout")
    http_timeout_seconds: int = Field(default=30, description="Default HTTP client timeout")

    # ==========================================================================
    # Vector Database Configuration (Qdrant)
    # ==========================================================================
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant URL from host and port."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # ==========================================================================
    # Book Passage Client Configuration (WBS-AGT23 / PCON-5)
    # ==========================================================================
    book_passages_collection: str = Field(
        default="book_passages",
        description="Qdrant collection name for book passages"
    )
    book_passages_dir: str = Field(
        default="",
        description="Path to enriched book JSON files"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vector search"
    )

    # ==========================================================================
    # Cache Configuration (Redis for user: prefix tier)
    # ==========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for user: prefix cache tier (24h TTL)"
    )

    # ==========================================================================
    # Agent Functions Configuration
    # ==========================================================================
    default_preset: str = Field(
        default="D4",
        description="Default preset (S1=Light, D4=Standard, D10=High Quality)"
    )
    max_pipeline_stages: int = Field(
        default=10,
        description="Maximum number of stages in a pipeline"
    )
    pipeline_timeout_seconds: int = Field(
        default=300,
        description="Pipeline execution timeout"
    )

    model_config = SettingsConfigDict(
        env_prefix="AI_AGENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings singleton
    """
    return Settings()
