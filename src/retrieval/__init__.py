"""Unified Knowledge Retrieval Module.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval
Acceptance Criteria:
- AC-24.1: Single interface queries all knowledge sources
- AC-24.2: Orchestrates: Qdrant → Neo4j → code-reference-engine → books
- AC-24.3: Merges and ranks results across sources
- AC-24.4: Returns unified RetrievalResult with mixed citations
- AC-24.5: cross_reference agent function uses this retriever
- AC-24.6: Supports scope filtering (code-only, books-only, all)

This module provides unified knowledge retrieval across multiple sources:
- Code Reference Engine (WBS-AGT21)
- Neo4j Graph (WBS-AGT22)
- Book Passages (WBS-AGT23)
- Semantic Search (semantic-search-service)
"""

from src.retrieval.unified_retriever import (
    UnifiedRetriever,
    UnifiedRetrieverConfig,
    UnifiedRetrieverProtocol,
    FakeUnifiedRetriever,
)
from src.retrieval.merger import ResultMerger, DiversityAwareMerger
from src.retrieval.ranker import CrossSourceRanker

__all__ = [
    "UnifiedRetriever",
    "UnifiedRetrieverConfig",
    "UnifiedRetrieverProtocol",
    "FakeUnifiedRetriever",
    "ResultMerger",
    "DiversityAwareMerger",
    "CrossSourceRanker",
]
