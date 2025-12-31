"""MSEP module for Multi-Stage Enrichment Pipeline.

This module contains all MSEP-related schemas, configuration, constants,
and exceptions for the ai-agents service (Expeditor role in Kitchen Brigade).

Kitchen Brigade Role: EXPEDITOR
- Orchestrates MSEP workflow
- Calls Code-Orchestrator-Service (Sous Chef) for SBERT, BERTopic, TF-IDF
- Calls semantic-search-service (Cookbook) for hybrid search
"""

from __future__ import annotations

from src.agents.msep.constants import (
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    LARGE_CORPUS_THRESHOLD,
    MAX_THRESHOLD,
    METHOD_BERTOPIC,
    METHOD_HYBRID,
    METHOD_SBERT,
    METHOD_TFIDF,
    MIN_THRESHOLD,
    SAME_TOPIC_BOOST,
    SERVICE_CODE_ORCHESTRATOR,
    SERVICE_SEMANTIC_SEARCH,
    SMALL_CORPUS_THRESHOLD,
    THRESHOLD_ADJUSTMENT,
)
from src.agents.msep.exceptions import (
    EnrichmentTimeoutError,
    MSEPError,
    ServiceUnavailableError,
)


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TOP_K",
    "LARGE_CORPUS_THRESHOLD",
    "MAX_THRESHOLD",
    "METHOD_BERTOPIC",
    "METHOD_HYBRID",
    "METHOD_SBERT",
    "METHOD_TFIDF",
    "MIN_THRESHOLD",
    # Constants
    "SAME_TOPIC_BOOST",
    "SERVICE_CODE_ORCHESTRATOR",
    "SERVICE_SEMANTIC_SEARCH",
    "SMALL_CORPUS_THRESHOLD",
    "THRESHOLD_ADJUSTMENT",
    "EnrichmentTimeoutError",
    # Exceptions
    "MSEPError",
    "ServiceUnavailableError",
]
