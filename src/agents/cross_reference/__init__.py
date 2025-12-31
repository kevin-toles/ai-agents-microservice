"""Cross-Reference Agent module.

Implements the LangGraph-based Cross-Reference Agent for taxonomy-aware
scholarly annotation generation.

Pattern: LangGraph StateGraph Workflow
Source: ARCHITECTURE.md (ai-agents), TIER_RELATIONSHIP_DIAGRAM.md
"""

from src.agents.cross_reference.agent import CrossReferenceAgent
from src.agents.cross_reference.state import (
    Citation,
    CrossReferenceResult,
    CrossReferenceState,
    SourceChapter,
    TierCoverage,
    TraversalConfig,
)


__all__ = [
    "Citation",
    "CrossReferenceAgent",
    "CrossReferenceResult",
    "CrossReferenceState",
    "SourceChapter",
    "TierCoverage",
    "TraversalConfig",
]
