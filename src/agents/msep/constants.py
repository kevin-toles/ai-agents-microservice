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

# =============================================================================
# API Endpoints (S1192 compliance - MSE-3)
# =============================================================================

# Code-Orchestrator-Service endpoints
ENDPOINT_SBERT_EMBEDDINGS: str = "/v1/sbert/embeddings"
"""Endpoint for SBERT embeddings."""

ENDPOINT_SBERT_SIMILARITY: str = "/v1/sbert/similarity"
"""Endpoint for SBERT similarity matrix."""

ENDPOINT_BERTOPIC_CLUSTER: str = "/v1/bertopic/cluster"
"""Endpoint for BERTopic clustering."""

ENDPOINT_KEYWORDS_EXTRACT: str = "/v1/keywords/extract"
"""Endpoint for TF-IDF keyword extraction."""

# Semantic-Search-Service endpoints
ENDPOINT_SEARCH_HYBRID: str = "/v1/search/hybrid"
"""Endpoint for hybrid search."""

ENDPOINT_GRAPH_RELATIONSHIPS: str = "/v1/graph/relationships"
"""Endpoint for graph relationships."""

ENDPOINT_GRAPH_RELATIONSHIPS_BATCH: str = "/v1/graph/relationships/batch"
"""Endpoint for batch graph relationships."""

# =============================================================================
# Service Base URLs (Environment-Configurable)
# =============================================================================

import os

SERVICE_CODE_ORCHESTRATOR_URL: str = os.environ.get(
    "CODE_ORCHESTRATOR_URL", "http://localhost:8082"
)
"""Base URL for Code-Orchestrator-Service (env: CODE_ORCHESTRATOR_URL)."""

SERVICE_SEMANTIC_SEARCH_URL: str = os.environ.get(
    "SEMANTIC_SEARCH_URL", "http://localhost:8083"
)
"""Base URL for semantic-search-service (env: SEMANTIC_SEARCH_URL)."""
