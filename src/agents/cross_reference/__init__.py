"""Cross-Reference Agent module.

Implements the LangGraph-based Cross-Reference Agent for taxonomy-aware
scholarly annotation generation.

Pattern: LangGraph StateGraph Workflow
Source: ARCHITECTURE.md (ai-agents), TIER_RELATIONSHIP_DIAGRAM.md
"""

from src.agents.cross_reference.state import (
    CrossReferenceState,
    SourceChapter,
    TraversalConfig,
    CrossReferenceResult,
    Citation,
    TierCoverage,
)
from src.agents.cross_reference.agent import CrossReferenceAgent

__all__ = [
    "CrossReferenceAgent",
    "CrossReferenceState",
    "SourceChapter",
    "TraversalConfig",
    "CrossReferenceResult",
    "Citation",
    "TierCoverage",
]
