"""Evidence Gatherer for Iterative Evidence Retrieval.

WBS Reference: WBS-KB3 - Iterative Evidence Gathering
Tasks: KB3.1-KB3.4 - Create EvidenceGatherer class with gather(), filtering, priority
Acceptance Criteria:
- AC-KB3.1: evidence_gatherer.gather() takes list[InformationRequest], returns evidence
- AC-KB3.2: Gatherer calls UnifiedRetriever with request queries
- AC-KB3.5: High-priority requests processed before medium/low
- AC-KB3.6: Gatherer respects source_types filter on requests

Anti-Patterns Avoided:
- S1192: No duplicated literals (constants at module level)
- Frozen dataclasses for immutability
- Proper type annotations throughout
- Protocol-based dependency injection for testability

Pattern: Strategy Pattern for source filtering
Reference: KITCHEN_BRIGADE_ARCHITECTURE.md → Cross-Reference Pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.discussion.models import CrossReferenceEvidence, InformationRequest
from src.schemas.retrieval_models import (
    RetrievalItem,
    RetrievalResult,
    RetrievalScope,
    SourceType,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
DEFAULT_MAX_RESULTS = 10
DEFAULT_TIMEOUT = 10.0

# Source type mappings
CODE_SOURCE_TYPES = frozenset({"code"})
BOOK_SOURCE_TYPES = frozenset({"books", "textbooks", "book"})
GRAPH_SOURCE_TYPES = frozenset({"graph"})


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retriever dependency injection.
    
    Enables duck typing for UnifiedRetriever and test doubles.
    """
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve results from knowledge sources."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvidenceGathererConfig:
    """Configuration for EvidenceGatherer.
    
    Attributes:
        max_results_per_request: Maximum results per information request
        timeout_seconds: Timeout for each retrieval call
    """
    
    max_results_per_request: int = DEFAULT_MAX_RESULTS
    timeout_seconds: float = DEFAULT_TIMEOUT


# =============================================================================
# GatherResult
# =============================================================================


@dataclass(frozen=True, slots=True)
class GatherResult:
    """Result from gathering evidence for information requests.
    
    Attributes:
        evidence: List of gathered evidence items
        total_items: Total number of items gathered
        requests_processed: Number of requests that were processed
        errors: List of error messages encountered
    """
    
    evidence: list[CrossReferenceEvidence]
    total_items: int
    requests_processed: int
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence": [e.to_dict() for e in self.evidence],
            "total_items": self.total_items,
            "requests_processed": self.requests_processed,
            "errors": list(self.errors),
        }


# =============================================================================
# EvidenceGatherer Implementation
# =============================================================================


class EvidenceGatherer:
    """Gathers evidence from UnifiedRetriever based on InformationRequests.
    
    WBS: KB3.1, KB3.2 - Create EvidenceGatherer class with gather()
    AC-KB3.1: gather() takes list[InformationRequest], returns new evidence
    AC-KB3.2: Gatherer calls UnifiedRetriever with request queries
    AC-KB3.5: High-priority requests processed before medium/low
    AC-KB3.6: Gatherer respects source_types filter on requests
    
    Attributes:
        config: Gatherer configuration
        retriever: UnifiedRetriever or compatible protocol
    """
    
    def __init__(
        self,
        config: EvidenceGathererConfig,
        retriever: RetrieverProtocol,
    ) -> None:
        """Initialize EvidenceGatherer.
        
        Args:
            config: Gatherer configuration
            retriever: UnifiedRetriever or compatible protocol
        """
        self.config = config
        self.retriever = retriever
    
    async def gather(
        self,
        requests: list[InformationRequest],
    ) -> GatherResult:
        """Gather evidence for a list of information requests.
        
        AC-KB3.1: Takes list[InformationRequest], returns GatherResult
        AC-KB3.5: High-priority requests processed first
        
        Args:
            requests: List of information requests from LLM participants
        
        Returns:
            GatherResult with gathered evidence
        """
        if not requests:
            return GatherResult(
                evidence=[],
                total_items=0,
                requests_processed=0,
                errors=[],
            )
        
        # Sort requests by priority (AC-KB3.5)
        sorted_requests = self._sort_by_priority(requests)
        
        all_evidence: list[CrossReferenceEvidence] = []
        errors: list[str] = []
        
        for request in sorted_requests:
            try:
                # Determine scope based on source_types (AC-KB3.6)
                scope = self._determine_scope(request.source_types)
                
                # Call retriever (AC-KB3.2)
                result = await self.retriever.retrieve(
                    query=request.query,
                    scope=scope,
                    top_k=self.config.max_results_per_request,
                )
                
                # Convert retrieval items to CrossReferenceEvidence
                evidence_items = self._convert_to_evidence(result.results)
                all_evidence.extend(evidence_items)
                
                # Collect errors from retrieval
                if result.errors:
                    errors.extend(result.errors)
                    
            except Exception as e:
                error_msg = f"Failed to gather evidence for '{request.query}': {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        return GatherResult(
            evidence=all_evidence,
            total_items=len(all_evidence),
            requests_processed=len(sorted_requests),
            errors=errors,
        )
    
    def _sort_by_priority(
        self,
        requests: list[InformationRequest],
    ) -> list[InformationRequest]:
        """Sort requests by priority (high → medium → low).
        
        AC-KB3.5: High-priority requests processed before medium/low.
        Uses stable sort to preserve order within same priority.
        
        Args:
            requests: Unsorted list of requests
        
        Returns:
            List sorted by priority
        """
        return sorted(
            requests,
            key=lambda r: PRIORITY_ORDER.get(r.priority, PRIORITY_ORDER["medium"]),
        )
    
    def _determine_scope(
        self,
        source_types: list[str],
    ) -> RetrievalScope:
        """Determine retrieval scope from source_types.
        
        AC-KB3.6: Gatherer respects source_types filter on requests.
        
        Args:
            source_types: List of requested source types
        
        Returns:
            Appropriate RetrievalScope
        """
        source_set = frozenset(s.lower() for s in source_types)
        
        # Check for single-scope requests
        is_code_only = source_set.issubset(CODE_SOURCE_TYPES)
        is_books_only = source_set.issubset(BOOK_SOURCE_TYPES)
        is_graph_only = source_set.issubset(GRAPH_SOURCE_TYPES)
        
        if is_code_only and source_set:
            return RetrievalScope.CODE_ONLY
        if is_books_only and source_set:
            return RetrievalScope.BOOKS_ONLY
        if is_graph_only and source_set:
            return RetrievalScope.GRAPH_ONLY
        
        # Mixed or unrecognized → query all
        return RetrievalScope.ALL
    
    def _convert_to_evidence(
        self,
        items: list[RetrievalItem],
    ) -> list[CrossReferenceEvidence]:
        """Convert RetrievalItems to CrossReferenceEvidence.
        
        Args:
            items: List of retrieval items
        
        Returns:
            List of CrossReferenceEvidence
        """
        return [
            CrossReferenceEvidence(
                source_type=self._source_type_to_string(item.source_type),
                content=item.content,
                source_id=item.source_id,
            )
            for item in items
        ]
    
    def _source_type_to_string(self, source_type: SourceType) -> str:
        """Convert SourceType enum to string."""
        return source_type.value
