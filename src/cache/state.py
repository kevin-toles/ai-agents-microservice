"""ADK State Management - State prefix constants and cache key builder.

Implements ADK-aligned state prefix conventions for the AI Platform:
- temp: Pipeline-local state (discarded after pipeline completes)
- user: Cross-session persistence (Redis with 24h TTL)
- app:  Application-wide artifacts (Qdrant/Neo4j permanent storage)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK Pattern Integration
"""

from enum import Enum
from typing import Literal


class StatePrefix(str, Enum):
    """ADK state prefix conventions.

    Each prefix maps to a specific cache tier with different
    lifetime and storage characteristics.

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → State Prefix Conventions
    """
    TEMP = "temp:"   # Pipeline handoff, discarded after pipeline completes
    USER = "user:"   # User session state, persists across invocations (Redis 24h)
    APP = "app:"     # Application-wide artifacts, permanent storage


# String constants for direct use (backwards compatible with __init__.py)
STATE_PREFIX_TEMP: Literal["temp:"] = "temp:"
STATE_PREFIX_USER: Literal["user:"] = "user:"
STATE_PREFIX_APP: Literal["app:"] = "app:"


def build_cache_key(prefix: str, namespace: str, key: str) -> str:
    """Build cache key using ADK prefix conventions.

    Args:
        prefix: One of STATE_PREFIX_TEMP, STATE_PREFIX_USER, STATE_PREFIX_APP
        namespace: Agent function name or pipeline ID
        key: Unique identifier within namespace

    Returns:
        Formatted cache key: "{prefix}{namespace}:{key}"

    Raises:
        ValueError: If prefix is not a valid ADK state prefix

    Example:
        >>> build_cache_key(STATE_PREFIX_TEMP, "extract_structure", "chapter_1")
        'temp:extract_structure:chapter_1'

        >>> build_cache_key("user:", "summarize", "user_123:session_abc")
        'user:summarize:user_123:session_abc'
    """
    # Validate prefix
    valid_prefixes = {STATE_PREFIX_TEMP, STATE_PREFIX_USER, STATE_PREFIX_APP}
    if prefix not in valid_prefixes:
        raise ValueError(
            f"Invalid prefix '{prefix}'. Must be one of: {valid_prefixes}"
        )

    # Validate namespace and key are non-empty
    if not namespace:
        raise ValueError("namespace cannot be empty")
    if not key:
        raise ValueError("key cannot be empty")

    return f"{prefix}{namespace}:{key}"


def parse_cache_key(cache_key: str) -> tuple[str, str, str]:
    """Parse a cache key into its components.

    Args:
        cache_key: A key built by build_cache_key()

    Returns:
        Tuple of (prefix, namespace, key)

    Raises:
        ValueError: If the cache key format is invalid

    Example:
        >>> parse_cache_key("temp:extract_structure:chapter_1")
        ('temp:', 'extract_structure', 'chapter_1')
    """
    for prefix in (STATE_PREFIX_TEMP, STATE_PREFIX_USER, STATE_PREFIX_APP):
        if cache_key.startswith(prefix):
            remainder = cache_key[len(prefix):]
            parts = remainder.split(":", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid cache key format: '{cache_key}'. "
                    f"Expected format: '{{prefix}}{{namespace}}:{{key}}'"
                )
            return prefix, parts[0], parts[1]

    raise ValueError(
        f"Invalid cache key prefix in '{cache_key}'. "
        f"Must start with 'temp:', 'user:', or 'app:'"
    )


def get_cache_tier(prefix: str) -> str:
    """Get the cache tier name for a state prefix.

    Args:
        prefix: ADK state prefix

    Returns:
        Human-readable cache tier name

    Example:
        >>> get_cache_tier(STATE_PREFIX_TEMP)
        'handoff_cache'
    """
    tier_map = {
        STATE_PREFIX_TEMP: "handoff_cache",
        STATE_PREFIX_USER: "compression_cache",
        STATE_PREFIX_APP: "artifact_store",
    }
    return tier_map.get(prefix, "unknown")
