"""Protocol Feature Flags for Phase 2 Protocol Integration.

This module provides environment-based feature flags that control all Phase 2
protocol features (A2A and MCP). This ensures zero impact on existing Phase 1
functionality and enables safe incremental rollout.

Reference: WBS_PROTOCOL_INTEGRATION.md → WBS-PI1: Feature Flags & Configuration
Architecture: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Feature Flags

Kitchen Brigade Architecture:
    Feature flags enable safe experimentation with A2A (Agent-to-Agent) and
    MCP (Model Context Protocol) without affecting existing Phase 1 operations.

Environment Variables:
    # A2A Protocol Features
    AGENTS_A2A_ENABLED=false           # Master switch for A2A
    AGENTS_A2A_AGENT_CARD_ENABLED=false # Expose /.well-known/agent-card.json
    AGENTS_A2A_STREAMING_ENABLED=false  # SSE streaming for task updates
    AGENTS_A2A_PUSH_NOTIFICATIONS=false # Webhook notifications

    # MCP Features
    AGENTS_MCP_ENABLED=false           # Master switch for MCP
    AGENTS_MCP_SERVER_ENABLED=false    # Expose agent functions as MCP tools
    AGENTS_MCP_CLIENT_ENABLED=false    # Consume external MCP servers
    AGENTS_MCP_SEMANTIC_SEARCH=false   # MCP wrapper for semantic-search-service (hybrid RAG)
    AGENTS_MCP_TOOLBOX_NEO4J=false     # MCP Toolbox for Neo4j
    AGENTS_MCP_TOOLBOX_REDIS=false     # MCP Toolbox for Redis

Anti-Patterns Avoided:
    - CODING_PATTERNS_ANALYSIS §1: Full type annotations on all methods
    - CODING_PATTERNS_ANALYSIS §2: Cognitive complexity < 15 per method
    - CODING_PATTERNS_ANALYSIS §3: No bare except clauses
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProtocolFeatureFlags(BaseSettings):
    """Phase 2 Protocol Integration feature flags.

    All flags default to False for safe rollout.
    Enable incrementally in development → staging → production.

    Attributes:
        a2a_enabled: Master switch for all A2A protocol features.
        a2a_agent_card_enabled: Expose Agent Card at /.well-known/agent-card.json
        a2a_streaming_enabled: Enable SSE streaming for task updates.
        a2a_push_notifications: Enable webhook push notifications.
        mcp_enabled: Master switch for all MCP features.
        mcp_server_enabled: Expose agent functions as MCP tools.
        mcp_client_enabled: Consume external MCP servers.
        mcp_semantic_search: Enable MCP wrapper for semantic-search-service hybrid RAG.
        mcp_toolbox_neo4j: Enable MCP Toolbox for Neo4j graph operations.
        mcp_toolbox_redis: Enable MCP Toolbox for Redis key-value operations.

    Example:
        >>> flags = ProtocolFeatureFlags()
        >>> flags.a2a_enabled  # False by default
        False
        >>> flags.a2a_available()  # No sub-features enabled
        False
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTS_",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # A2A Protocol Flags
    # =========================================================================

    a2a_enabled: bool = Field(
        default=False,
        description="Master switch for all A2A protocol features",
    )
    a2a_agent_card_enabled: bool = Field(
        default=False,
        description="Expose Agent Card at /.well-known/agent-card.json",
    )
    a2a_task_lifecycle_enabled: bool = Field(
        default=False,
        description="Enable A2A task lifecycle endpoints (message:send, tasks)",
    )
    a2a_streaming_enabled: bool = Field(
        default=False,
        description="Enable SSE streaming for real-time task updates",
    )
    a2a_push_notifications: bool = Field(
        default=False,
        description="Enable webhook push notifications for task events",
    )

    # =========================================================================
    # MCP Protocol Flags
    # =========================================================================

    mcp_enabled: bool = Field(
        default=False,
        description="Master switch for all MCP features",
    )
    mcp_server_enabled: bool = Field(
        default=False,
        description="Expose agent functions as MCP tools for external clients",
    )
    mcp_client_enabled: bool = Field(
        default=False,
        description="Enable consumption of external MCP servers",
    )
    mcp_semantic_search: bool = Field(
        default=False,
        description="Enable MCP wrapper for semantic-search-service hybrid RAG layer",
    )
    mcp_toolbox_neo4j: bool = Field(
        default=False,
        description="Enable MCP Toolbox for Neo4j graph operations",
    )
    mcp_toolbox_redis: bool = Field(
        default=False,
        description="Enable MCP Toolbox for Redis key-value operations",
    )

    # =========================================================================
    # A2A Availability Methods
    # =========================================================================

    def a2a_available(self) -> bool:
        """Check if any A2A feature is available.

        Returns True only when:
        1. Master switch (a2a_enabled) is True, AND
        2. At least one sub-feature is enabled

        Returns:
            True if A2A protocol features are available, False otherwise.
        """
        if not self.a2a_enabled:
            return False
        return any([
            self.a2a_agent_card_enabled,
            self.a2a_streaming_enabled,
            self.a2a_push_notifications,
        ])

    @property
    def a2a_agent_card_available(self) -> bool:
        """Check if A2A Agent Card endpoint should be exposed.

        Returns:
            True if master switch AND agent_card_enabled are both True.
        """
        return self.a2a_enabled and self.a2a_agent_card_enabled

    @property
    def a2a_streaming_available(self) -> bool:
        """Check if A2A SSE streaming should be enabled.

        Returns:
            True if master switch AND streaming_enabled are both True.
        """
        return self.a2a_enabled and self.a2a_streaming_enabled

    @property
    def a2a_push_notifications_available(self) -> bool:
        """Check if A2A push notifications should be enabled.

        Returns:
            True if master switch AND push_notifications are both True.
        """
        return self.a2a_enabled and self.a2a_push_notifications

    # =========================================================================
    # MCP Availability Methods
    # =========================================================================

    def mcp_available(self) -> bool:
        """Check if any MCP feature is available.

        Returns True only when:
        1. Master switch (mcp_enabled) is True, AND
        2. At least one sub-feature is enabled

        Returns:
            True if MCP features are available, False otherwise.
        """
        if not self.mcp_enabled:
            return False
        return any([
            self.mcp_server_enabled,
            self.mcp_client_enabled,
            self.mcp_semantic_search,
            self.mcp_toolbox_neo4j,
            self.mcp_toolbox_redis,
        ])

    @property
    def mcp_server_available(self) -> bool:
        """Check if MCP server should expose agent functions as tools.

        Returns:
            True if master switch AND server_enabled are both True.
        """
        return self.mcp_enabled and self.mcp_server_enabled

    @property
    def mcp_client_available(self) -> bool:
        """Check if MCP client should consume external servers.

        Returns:
            True if master switch AND client_enabled are both True.
        """
        return self.mcp_enabled and self.mcp_client_enabled

    @property
    def mcp_semantic_search_available(self) -> bool:
        """Check if MCP wrapper for semantic-search-service should be enabled.

        Returns:
            True if master switch AND mcp_semantic_search are both True.
        """
        return self.mcp_enabled and self.mcp_semantic_search

    @property
    def mcp_toolbox_neo4j_available(self) -> bool:
        """Check if MCP Toolbox for Neo4j should be enabled.

        Returns:
            True if master switch AND toolbox_neo4j are both True.
        """
        return self.mcp_enabled and self.mcp_toolbox_neo4j

    @property
    def mcp_toolbox_redis_available(self) -> bool:
        """Check if MCP Toolbox for Redis should be enabled.

        Returns:
            True if master switch AND toolbox_redis are both True.
        """
        return self.mcp_enabled and self.mcp_toolbox_redis


# =============================================================================
# FastAPI Dependency (AC-PI1.7)
# =============================================================================


@lru_cache
def get_feature_flags() -> ProtocolFeatureFlags:
    """Get cached ProtocolFeatureFlags instance.

    This function is designed to be used with FastAPI's Depends() pattern
    for dependency injection. The lru_cache decorator ensures a singleton
    instance is returned for the lifetime of the application.

    Usage:
        from fastapi import Depends
        from src.config.feature_flags import ProtocolFeatureFlags, get_feature_flags

        @router.get("/some-endpoint")
        async def some_endpoint(
            flags: ProtocolFeatureFlags = Depends(get_feature_flags),
        ):
            if flags.a2a_enabled:
                # Handle A2A case
                ...

    Returns:
        ProtocolFeatureFlags: Cached singleton instance loaded from environment.
    """
    return ProtocolFeatureFlags()
