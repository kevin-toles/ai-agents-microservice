"""Result Merger for Unified Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.3)
Acceptance Criteria:
- AC-24.3: Merges and ranks results across sources

Implements the merging strategy for combining results from multiple sources:
- Code Reference Engine
- Neo4j Graph
- Book Passages

Pattern: Strategy pattern for merging algorithms
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
Anti-Pattern: No mutable default arguments (AP-1.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from src.schemas.retrieval_models import RetrievalItem, SourceType


# =============================================================================
# Merger Protocol
# =============================================================================


class MergerProtocol(Protocol):
    """Protocol for result mergers.
    
    Enables duck typing for test doubles (FakeMerger).
    """
    
    def merge(
        self,
        *item_lists: list[RetrievalItem],
    ) -> list[RetrievalItem]:
        """Merge multiple lists of retrieval items.
        
        Args:
            item_lists: Variable number of item lists to merge
        
        Returns:
            Merged list of items
        """
        ...


# =============================================================================
# Result Merger Implementation
# =============================================================================


@dataclass
class ResultMerger:
    """Merges results from multiple knowledge sources.
    
    WBS: AGT24.3 - Implement result merging strategy
    AC-24.3: Merges and ranks results across sources
    
    Combines results from code, book, and graph sources with:
    - Content deduplication (optional)
    - Source diversity preservation
    - Relevance-based selection for duplicates
    
    Attributes:
        dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
        preserve_diversity: Whether to preserve source diversity
        max_per_source: Maximum items to keep per source (0 = unlimited)
    """
    
    dedup_threshold: float = 0.95
    preserve_diversity: bool = True
    max_per_source: int = 0
    
    def merge(
        self,
        *item_lists: list[Any],
    ) -> list[Any]:
        """Merge multiple lists of retrieval items.
        
        Args:
            item_lists: Variable number of item lists to merge
        
        Returns:
            Merged list of items, deduplicated if enabled
        """
        from src.schemas.retrieval_models import RetrievalItem
        
        # Flatten all lists
        all_items: list[RetrievalItem] = []
        for items in item_lists:
            all_items.extend(items)
        
        if not all_items:
            return []
        
        # Apply max per source limit if set
        if self.max_per_source > 0:
            all_items = self._apply_source_limit(all_items)
        
        # Deduplicate if threshold is set
        if self.dedup_threshold < 1.0:
            all_items = self._deduplicate(all_items)
        
        return all_items
    
    def _apply_source_limit(
        self,
        items: list[Any],
    ) -> list[Any]:
        """Apply maximum items per source limit.
        
        Keeps highest-scoring items per source type.
        """
        from src.schemas.retrieval_models import SourceType
        
        # Group by source type
        by_source: dict[SourceType, list[Any]] = {}
        for item in items:
            source = item.source_type
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)
        
        # Sort each group by relevance and limit
        limited: list[Any] = []
        for source, source_items in by_source.items():
            sorted_items = sorted(
                source_items,
                key=lambda x: x.relevance_score,
                reverse=True,
            )
            limited.extend(sorted_items[:self.max_per_source])
        
        return limited
    
    def _deduplicate(
        self,
        items: list[Any],
    ) -> list[Any]:
        """Remove duplicate content across sources.
        
        Uses content similarity to detect duplicates.
        Keeps the item with higher relevance score.
        """
        if not items:
            return []
        
        # Sort by relevance so we keep higher-scored duplicates
        sorted_items = sorted(
            items,
            key=lambda x: x.relevance_score,
            reverse=True,
        )
        
        seen_content: dict[str, Any] = {}
        unique_items: list[Any] = []
        
        for item in sorted_items:
            # Normalize content for comparison
            normalized = self._normalize_content(item.content)
            
            # Check for similar existing content
            is_duplicate = False
            for seen_norm, seen_item in seen_content.items():
                similarity = self._content_similarity(normalized, seen_norm)
                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content[normalized] = item
                unique_items.append(item)
        
        return unique_items
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison.
        
        Lowercases and removes extra whitespace.
        """
        return " ".join(content.lower().split())
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings.
        
        Uses Jaccard similarity on word sets.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Specialized Mergers
# =============================================================================


@dataclass
class DiversityAwareMerger(ResultMerger):
    """Merger that ensures diversity across source types.
    
    Interleaves results from different sources to ensure
    the merged list represents all source types evenly.
    """
    
    interleave: bool = True
    
    def merge(
        self,
        *item_lists: list[Any],
    ) -> list[Any]:
        """Merge with interleaving for source diversity."""
        # First do standard merge
        merged = super().merge(*item_lists)
        
        if not self.interleave or not merged:
            return merged
        
        # Interleave by source type
        return self._interleave_by_source(merged)
    
    def _interleave_by_source(
        self,
        items: list[Any],
    ) -> list[Any]:
        """Interleave items by source type.
        
        Round-robin selection from each source type.
        """
        from src.schemas.retrieval_models import SourceType
        
        # Group by source type
        by_source: dict[SourceType, list[Any]] = {}
        for item in items:
            source = item.source_type
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)
        
        # Sort each group by relevance
        for source in by_source:
            by_source[source].sort(
                key=lambda x: x.relevance_score,
                reverse=True,
            )
        
        # Interleave
        interleaved: list[Any] = []
        source_iterators = {
            source: iter(items) for source, items in by_source.items()
        }
        
        while source_iterators:
            exhausted = []
            for source, iterator in source_iterators.items():
                try:
                    item = next(iterator)
                    interleaved.append(item)
                except StopIteration:
                    exhausted.append(source)
            
            for source in exhausted:
                del source_iterators[source]
        
        return interleaved


# =============================================================================
# Factory Function
# =============================================================================


def create_merger(
    strategy: str = "default",
    **kwargs: Any,
) -> ResultMerger:
    """Create a merger with the specified strategy.
    
    Args:
        strategy: Merger strategy ("default", "diversity", "strict_dedup")
        **kwargs: Additional configuration for the merger
    
    Returns:
        Configured ResultMerger instance
    """
    if strategy == "diversity":
        return DiversityAwareMerger(**kwargs)
    elif strategy == "strict_dedup":
        return ResultMerger(dedup_threshold=0.8, **kwargs)
    else:
        return ResultMerger(**kwargs)
