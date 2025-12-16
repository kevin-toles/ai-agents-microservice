"""MSEP Constants Module.

WBS: MSE-2.5 - Constants Module
All magic numbers and duplicated strings extracted to constants.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: All strings/numbers as constants
- #2.2: Full type annotations
"""

from __future__ import annotations

# =============================================================================
# Topic Boost Constants (AC-2.5.1)
# =============================================================================

SAME_TOPIC_BOOST: float = 0.15
"""Boost applied when source and target chapters share the same BERTopic cluster."""

# =============================================================================
# Threshold Constants (AC-2.5.2, Dynamic threshold from MSE-5.2)
# =============================================================================

DEFAULT_THRESHOLD: float = 0.5
"""Default similarity threshold for cross-reference filtering."""

MIN_THRESHOLD: float = 0.3
"""Minimum allowed threshold value (clamp lower bound)."""

MAX_THRESHOLD: float = 0.6
"""Maximum allowed threshold value (clamp upper bound)."""

THRESHOLD_ADJUSTMENT: float = 0.1
"""Amount to adjust threshold based on corpus size."""

# =============================================================================
# Top-K Constants (AC-2.5.3)
# =============================================================================

DEFAULT_TOP_K: int = 5
"""Default number of top cross-references to return per chapter."""

# =============================================================================
# Timeout Constants (AC-2.5.4)
# =============================================================================

DEFAULT_TIMEOUT: float = 30.0
"""Default timeout in seconds for service calls."""

# =============================================================================
# Corpus Size Thresholds (MSE-5.2 Dynamic Threshold)
# =============================================================================

SMALL_CORPUS_THRESHOLD: int = 100
"""Corpus size below which threshold is increased."""

LARGE_CORPUS_THRESHOLD: int = 500
"""Corpus size above which threshold is decreased."""

# =============================================================================
# Service Names (S1192 compliance)
# =============================================================================

SERVICE_CODE_ORCHESTRATOR: str = "code-orchestrator"
"""Name constant for Code-Orchestrator-Service."""

SERVICE_SEMANTIC_SEARCH: str = "semantic-search"
"""Name constant for semantic-search-service."""

# =============================================================================
# Enrichment Method Names (S1192 compliance)
# =============================================================================

METHOD_SBERT: str = "sbert"
"""SBERT embedding similarity method."""

METHOD_TFIDF: str = "tfidf"
"""TF-IDF keyword extraction method."""

METHOD_BERTOPIC: str = "bertopic"
"""BERTopic clustering method."""

METHOD_HYBRID: str = "hybrid"
"""Hybrid search method (vector + keyword)."""

# =============================================================================
# Chapter ID Format
# =============================================================================

CHAPTER_ID_FORMAT: str = "{book}:ch{chapter}"
"""Format string for auto-generating chapter IDs."""
