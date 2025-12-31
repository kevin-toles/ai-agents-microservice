"""Cache Management Package.

This package implements ADK-aligned state management with prefix conventions:
- temp: Pipeline-local state (HandoffCache)
- user: Cross-session persistence (CompressionCache → Redis)
- app:  Application-wide artifacts (Qdrant/Neo4j)

Pattern: ADK State Prefix Conventions
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration
"""

from src.cache.artifact import (
    Artifact,
    ArtifactReference,
)
from src.cache.compression import (
    DEFAULT_USER_TTL_SECONDS,
    CacheEntry,
    CompressionCache,
    RedisClientProtocol,
)
from src.cache.handoff import (
    HandoffCache,
    HandoffEntry,
)
from src.cache.state import (
    STATE_PREFIX_APP,
    STATE_PREFIX_TEMP,
    STATE_PREFIX_USER,
    StatePrefix,
    build_cache_key,
    get_cache_tier,
    parse_cache_key,
)


__all__ = [
    "DEFAULT_USER_TTL_SECONDS",
    "STATE_PREFIX_APP",
    "STATE_PREFIX_TEMP",
    "STATE_PREFIX_USER",
    # Artifact management
    "Artifact",
    "ArtifactReference",
    "CacheEntry",
    # CompressionCache (user: prefix)
    "CompressionCache",
    # HandoffCache (temp: prefix)
    "HandoffCache",
    "HandoffEntry",
    "RedisClientProtocol",
    # State prefix constants and utilities
    "StatePrefix",
    "build_cache_key",
    "get_cache_tier",
    "parse_cache_key",
]
