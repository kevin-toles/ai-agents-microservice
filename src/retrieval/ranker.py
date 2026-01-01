"""Cross-Source Ranker for Unified Knowledge Retrieval.

WBS Reference: WBS-AGT24 Unified Knowledge Retrieval (AGT24.4)
Acceptance Criteria:
- AC-24.3: Merges and ranks results across sources

Implements cross-source ranking strategies for ordering results
from multiple knowledge sources by relevance.

Pattern: Strategy pattern for ranking algorithms
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pipeline Composition
Anti-Pattern: No mutable default arguments (AP-1.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.schemas.retrieval_models import RetrievalItem, SourceType


# =============================================================================
# Ranker Protocol
# =============================================================================


class RankerProtocol(Protocol):
    """Protocol for result rankers.
    
    Enables duck typing for test doubles (FakeRanker).
    """
    
    def rank(
        self,
        items: list[RetrievalItem],
        top_k: int | None = None,
    ) -> list[RetrievalItem]:
        """Rank retrieval items by relevance.
        
        Args:
            items: List of items to rank
            top_k: Optional limit on results
        
        Returns:
            Ranked list of items
        """
        ...


# =============================================================================
# Cross-Source Ranker Implementation
# =============================================================================


@dataclass
class CrossSourceRanker:
    """Ranks results from multiple knowledge sources.
    
    WBS: AGT24.4 - Implement cross-source ranking
    AC-24.3: Merges and ranks results across sources
    
    Ranks items considering:
    - Base relevance score from source
    - Source-type boost factors
    - Recency (if available)
    - Query-specific weighting
    
    Attributes:
        source_boosts: Boost factors per source type
        normalize_scores: Whether to normalize scores to 0-1
        score_decay: Decay factor for position-based scoring
    """
    
    source_boosts: dict[Any, float] = field(default_factory=dict)
    normalize_scores: bool = True
    score_decay: float = 0.0
    
    def rank(
        self,
        items: list[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        """Rank retrieval items by relevance.
        
        Args:
            items: List of items to rank
            top_k: Optional limit on results
        
        Returns:
            Ranked list of items, highest relevance first
        """
        if not items:
            return []
        
        # Calculate effective scores with boosts
        scored_items = [
            (item, self._calculate_effective_score(item))
            for item in items
        ]
        
        # Sort by effective score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Extract items
        ranked = [item for item, score in scored_items]
        
        # Apply top_k limit
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _calculate_effective_score(self, item: Any) -> float:
        """Calculate effective score with boosts applied.
        
        Args:
            item: Retrieval item to score
        
        Returns:
            Effective relevance score
        """
        base_score = item.relevance_score
        
        # Apply source boost if configured
        boost = self.source_boosts.get(item.source_type, 1.0)
        effective = base_score * boost
        
        # Normalize to 0-1 range if enabled
        if self.normalize_scores and effective > 1.0:
            effective = 1.0
        
        return effective
    
    def rank_with_scores(
        self,
        items: list[Any],
        top_k: int | None = None,
    ) -> list[tuple[Any, float]]:
        """Rank items and return with effective scores.
        
        Args:
            items: List of items to rank
            top_k: Optional limit on results
        
        Returns:
            List of (item, effective_score) tuples
        """
        if not items:
            return []
        
        # Calculate effective scores
        scored_items = [
            (item, self._calculate_effective_score(item))
            for item in items
        ]
        
        # Sort by effective score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None and top_k > 0:
            scored_items = scored_items[:top_k]
        
        return scored_items


# =============================================================================
# Specialized Rankers
# =============================================================================


@dataclass
class RecencyAwareRanker(CrossSourceRanker):
    """Ranker that considers content recency.
    
    Applies a boost to more recent items based on timestamp.
    """
    
    recency_weight: float = 0.1
    max_age_days: int = 365
    
    def _calculate_effective_score(self, item: Any) -> float:
        """Calculate score with recency boost."""
        base_score = super()._calculate_effective_score(item)
        
        # Check for timestamp in metadata
        timestamp = item.metadata.get("timestamp") or item.metadata.get("updated_at")
        if timestamp:
            recency_boost = self._calculate_recency_boost(timestamp)
            base_score = base_score * (1 + recency_boost * self.recency_weight)
        
        return min(base_score, 1.0) if self.normalize_scores else base_score
    
    def _calculate_recency_boost(self, timestamp: str) -> float:
        """Calculate recency boost from timestamp.
        
        Returns value between 0 (old) and 1 (recent).
        """
        from datetime import datetime, timezone
        
        try:
            # Parse ISO timestamp
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                return 0.0
            
            # Calculate age in days
            now = datetime.now(timezone.utc)
            age_days = (now - dt).days
            
            # Linear decay
            if age_days >= self.max_age_days:
                return 0.0
            
            return 1.0 - (age_days / self.max_age_days)
        except (ValueError, TypeError):
            return 0.0


@dataclass
class QueryRelevanceRanker(CrossSourceRanker):
    """Ranker that considers query-term overlap.
    
    Boosts items that contain query terms in title/content.
    """
    
    query_boost: float = 0.2
    title_weight: float = 2.0
    
    def rank(
        self,
        items: list[Any],
        top_k: int | None = None,
        query: str = "",
    ) -> list[Any]:
        """Rank with query-term boost.
        
        Args:
            items: Items to rank
            top_k: Optional limit
            query: Query string for term matching
        
        Returns:
            Ranked items
        """
        if not query:
            return super().rank(items, top_k)
        
        # Store query for scoring
        self._current_query = query.lower().split()
        
        result = super().rank(items, top_k)
        
        # Clean up
        self._current_query = []
        
        return result
    
    def _calculate_effective_score(self, item: Any) -> float:
        """Calculate score with query-term boost."""
        base_score = super()._calculate_effective_score(item)
        
        if not hasattr(self, "_current_query") or not self._current_query:
            return base_score
        
        # Check title overlap
        title = (item.title or "").lower()
        title_matches = sum(1 for term in self._current_query if term in title)
        
        # Check content overlap
        content = (item.content or "").lower()
        content_matches = sum(1 for term in self._current_query if term in content)
        
        # Calculate boost
        total_terms = len(self._current_query)
        if total_terms > 0:
            title_score = (title_matches / total_terms) * self.title_weight
            content_score = content_matches / total_terms
            query_boost = (title_score + content_score) / (1 + self.title_weight)
            base_score = base_score * (1 + query_boost * self.query_boost)
        
        return min(base_score, 1.0) if self.normalize_scores else base_score


# =============================================================================
# Factory Function
# =============================================================================


def create_ranker(
    strategy: str = "default",
    **kwargs: Any,
) -> CrossSourceRanker:
    """Create a ranker with the specified strategy.
    
    Args:
        strategy: Ranker strategy ("default", "recency", "query_relevance")
        **kwargs: Additional configuration for the ranker
    
    Returns:
        Configured ranker instance
    """
    if strategy == "recency":
        return RecencyAwareRanker(**kwargs)
    elif strategy == "query_relevance":
        return QueryRelevanceRanker(**kwargs)
    else:
        return CrossSourceRanker(**kwargs)
