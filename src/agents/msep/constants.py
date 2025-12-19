"""MSEP Constants Module.

WBS: MSE-2.5 - Constants Module
All magic numbers and duplicated strings extracted to constants.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: All strings/numbers as constants
- #2.2: Full type annotations
"""

from __future__ import annotations


# =============================================================================
# Taxonomy Constants (AC-TAX-2)
# =============================================================================

DEFAULT_TAXONOMY: str | None = None
"""Default taxonomy value - None means no filtering applied."""

ENV_TAXONOMY_KEY: str = "MSEP_TAXONOMY"
"""Environment variable key for taxonomy configuration."""

# =============================================================================
# Topic Boost Constants (AC-2.5.1)
# =============================================================================

SAME_TOPIC_BOOST: float = 0.15
"""Boost applied when source and target chapters share the same BERTopic cluster."""

# =============================================================================
# EEP-3.1: Similarity Fusion Weights (AC-3.1.1 to AC-3.1.3)
# =============================================================================

FUSION_WEIGHT_SBERT: float = 0.45
"""Weight for SBERT semantic similarity in fusion score.

Rationale: SBERT captures deep semantic meaning and is the primary signal
for document similarity. Higher weight reflects its reliability for
natural language content per AI_CODING_PLATFORM_ARCHITECTURE.md.
"""

FUSION_WEIGHT_CODEBERT: float = 0.15
"""Weight for CodeBERT code similarity in fusion score.

Rationale: Code blocks are less common than prose in documentation.
When present, CodeBERT provides valuable signal for technical content.
Weight is redistributed to SBERT when no code blocks are present.
"""

FUSION_WEIGHT_CONCEPT: float = 0.25
"""Weight for concept overlap (Jaccard) in fusion score.

Rationale: Extracted concepts from taxonomy provide domain-specific
matching that complements embedding similarity. Per EEP-2 requirements.
"""

FUSION_WEIGHT_KEYWORD: float = 0.15
"""Weight for keyword Jaccard in fusion score.

Rationale: TF-IDF keywords capture surface-level lexical similarity.
Lower weight since embeddings already capture semantic meaning.
"""

FUSION_WEIGHT_TOPIC_BOOST: float = 0.15
"""Additive boost when chapters share the same BERTopic cluster.

Rationale: Same topic cluster indicates strong thematic relationship
independent of similarity scores. Matches existing SAME_TOPIC_BOOST.
"""

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

# Code-Orchestrator-Service endpoints (Port 8083)
ENDPOINT_SBERT_EMBEDDINGS: str = "/v1/embeddings"
"""Endpoint for SBERT embeddings."""

ENDPOINT_SBERT_SIMILARITY: str = "/v1/similarity/matrix"
"""Endpoint for SBERT similarity matrix (full corpus pairwise)."""

ENDPOINT_BERTOPIC_CLUSTER: str = "/api/v1/cluster"
"""Endpoint for BERTopic clustering."""

ENDPOINT_KEYWORDS_EXTRACT: str = "/api/v1/keywords"
"""Endpoint for TF-IDF keyword extraction."""

# Semantic-Search-Service endpoints (Port 8081)
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
    "CODE_ORCHESTRATOR_URL", "http://localhost:8083"
)
"""Base URL for Code-Orchestrator-Service (env: CODE_ORCHESTRATOR_URL). Port 8083."""

SERVICE_SEMANTIC_SEARCH_URL: str = os.environ.get(
    "SEMANTIC_SEARCH_URL", "http://localhost:8081"
)
"""Base URL for semantic-search-service (env: SEMANTIC_SEARCH_URL). Port 8081."""

# =============================================================================
# Audit Service Constants (MSE-8.4)
# =============================================================================

SERVICE_AUDIT_SERVICE: str = "audit-service"
"""Name constant for audit-service."""

SERVICE_AUDIT_URL: str = os.environ.get(
    "AUDIT_SERVICE_URL", "http://audit-service:8084"
)
"""Base URL for audit-service (env: AUDIT_SERVICE_URL). Port 8084."""

ENDPOINT_AUDIT_CROSS_REF: str = "/v1/audit/cross-reference"
"""Endpoint for cross-reference audit."""

ENV_ENABLE_AUDIT_KEY: str = "MSEP_ENABLE_AUDIT"
"""Environment variable key for audit validation flag."""
