"""MSEP Configuration.

WBS: MSE-2.3 - Configuration Dataclass
Defines MSEPConfig with environment variable support.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- #2.2: Full type annotations
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.agents.msep.constants import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TIMEOUT,
    SAME_TOPIC_BOOST,
)

# Environment variable prefix for MSEP config
_ENV_PREFIX: str = "MSEP_"


def _parse_bool(value: str) -> bool:
    """Parse boolean from string (env var)."""
    return value.lower() in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class MSEPConfig:
    """Configuration for MSEP orchestration.

    This is a frozen (immutable) dataclass that holds all
    configuration values for the Multi-Stage Enrichment Pipeline.

    Attributes:
        threshold: Similarity threshold for filtering cross-references.
        top_k: Number of top cross-references to return per chapter.
        timeout: Timeout in seconds for service calls.
        same_topic_boost: Boost applied when chapters share same topic.
        use_dynamic_threshold: Whether to adjust threshold by corpus size.
        enable_hybrid_search: Whether to include hybrid search results.
    """

    threshold: float = field(default=DEFAULT_THRESHOLD)
    top_k: int = field(default=DEFAULT_TOP_K)
    timeout: float = field(default=DEFAULT_TIMEOUT)
    same_topic_boost: float = field(default=SAME_TOPIC_BOOST)
    use_dynamic_threshold: bool = field(default=True)
    enable_hybrid_search: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "MSEPConfig":
        """Create MSEPConfig from environment variables.

        Environment variables are prefixed with MSEP_:
        - MSEP_THRESHOLD: float
        - MSEP_TOP_K: int
        - MSEP_TIMEOUT: float
        - MSEP_SAME_TOPIC_BOOST: float
        - MSEP_USE_DYNAMIC_THRESHOLD: bool (true/false)
        - MSEP_ENABLE_HYBRID_SEARCH: bool (true/false)

        Returns:
            MSEPConfig instance with values from environment.
        """
        threshold_str = os.environ.get(f"{_ENV_PREFIX}THRESHOLD")
        top_k_str = os.environ.get(f"{_ENV_PREFIX}TOP_K")
        timeout_str = os.environ.get(f"{_ENV_PREFIX}TIMEOUT")
        same_topic_boost_str = os.environ.get(f"{_ENV_PREFIX}SAME_TOPIC_BOOST")
        use_dynamic_threshold_str = os.environ.get(f"{_ENV_PREFIX}USE_DYNAMIC_THRESHOLD")
        enable_hybrid_search_str = os.environ.get(f"{_ENV_PREFIX}ENABLE_HYBRID_SEARCH")

        return cls(
            threshold=float(threshold_str) if threshold_str else DEFAULT_THRESHOLD,
            top_k=int(top_k_str) if top_k_str else DEFAULT_TOP_K,
            timeout=float(timeout_str) if timeout_str else DEFAULT_TIMEOUT,
            same_topic_boost=(
                float(same_topic_boost_str)
                if same_topic_boost_str
                else SAME_TOPIC_BOOST
            ),
            use_dynamic_threshold=(
                _parse_bool(use_dynamic_threshold_str)
                if use_dynamic_threshold_str
                else True
            ),
            enable_hybrid_search=(
                _parse_bool(enable_hybrid_search_str)
                if enable_hybrid_search_str
                else True
            ),
        )
