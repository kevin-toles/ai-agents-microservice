"""Cache Management Package.

This package implements ADK-aligned state management with prefix conventions:
- temp: Pipeline-local state (HandoffCache)
- user: Cross-session persistence (CompressionCache → Redis)
- app:  Application-wide artifacts (Qdrant/Neo4j)

Pattern: ADK State Prefix Conventions
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration
"""

from src.cache.state import (
    StatePrefix,
    STATE_PREFIX_TEMP,
    STATE_PREFIX_USER,
    STATE_PREFIX_APP,
    build_cache_key,
    parse_cache_key,
    get_cache_tier,
)
from src.cache.artifact import (
    Artifact,
    ArtifactReference,
)
from src.cache.handoff import (
    HandoffCache,
    HandoffEntry,
)
from src.cache.compression import (
    CompressionCache,
    CacheEntry,
    RedisClientProtocol,
    DEFAULT_USER_TTL_SECONDS,
)


__all__ = [
    # State prefix constants and utilities
    "StatePrefix",
    "STATE_PREFIX_TEMP",
    "STATE_PREFIX_USER",
    "STATE_PREFIX_APP",
    "build_cache_key",
    "parse_cache_key",
    "get_cache_tier",
    # Artifact management
    "Artifact",
    "ArtifactReference",
    # HandoffCache (temp: prefix)
    "HandoffCache",
    "HandoffEntry",
    # CompressionCache (user: prefix)
    "CompressionCache",
    "CacheEntry",
    "RedisClientProtocol",
    "DEFAULT_USER_TTL_SECONDS",
]
