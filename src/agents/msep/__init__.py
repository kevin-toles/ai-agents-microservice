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
    SAME_TOPIC_BOOST,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TIMEOUT,
    MIN_THRESHOLD,
    MAX_THRESHOLD,
    THRESHOLD_ADJUSTMENT,
    SMALL_CORPUS_THRESHOLD,
    LARGE_CORPUS_THRESHOLD,
    SERVICE_CODE_ORCHESTRATOR,
    SERVICE_SEMANTIC_SEARCH,
    METHOD_SBERT,
    METHOD_TFIDF,
    METHOD_BERTOPIC,
    METHOD_HYBRID,
)
from src.agents.msep.exceptions import (
    MSEPError,
    EnrichmentTimeoutError,
    ServiceUnavailableError,
)

__all__ = [
    # Constants
    "SAME_TOPIC_BOOST",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_TIMEOUT",
    "MIN_THRESHOLD",
    "MAX_THRESHOLD",
    "THRESHOLD_ADJUSTMENT",
    "SMALL_CORPUS_THRESHOLD",
    "LARGE_CORPUS_THRESHOLD",
    "SERVICE_CODE_ORCHESTRATOR",
    "SERVICE_SEMANTIC_SEARCH",
    "METHOD_SBERT",
    "METHOD_TFIDF",
    "METHOD_BERTOPIC",
    "METHOD_HYBRID",
    # Exceptions
    "MSEPError",
    "EnrichmentTimeoutError",
    "ServiceUnavailableError",
]
