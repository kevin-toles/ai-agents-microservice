"""Application configuration using Pydantic Settings.

Follows Pydantic Settings pattern from CODING_PATTERNS_ANALYSIS.md Phase 1.
Environment variables are loaded with AI_AGENTS_ prefix.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Pattern: Pydantic Settings with Environment Variables
    Source: CODING_PATTERNS_ANALYSIS.md Phase 1, Comp_Static_Analysis #18
    """
    
    # Service configuration
    service_name: str = "ai-agents"
    port: int = 8082
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # External service URLs
    llm_gateway_url: str = Field(
        default="http://localhost:8080",
        description="LLM Gateway service URL"
    )
    semantic_search_url: str = Field(
        default="http://localhost:8081",
        description="Semantic Search service URL"
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
