"""Evidence Merger for combining evidence across discussion cycles.

WBS Reference: WBS-KB3 - Iterative Evidence Gathering
Tasks: KB3.5, KB3.6 - Implement merge_evidence() with deduplication and provenance
Acceptance Criteria:
- AC-KB3.3: Evidence from multiple sources merged without duplicates
- AC-KB3.4: merge_evidence() combines old + new evidence, preserving provenance

Anti-Patterns Avoided:
- S1192: No duplicated literals (constants at module level)
- Frozen dataclasses for immutability
- Proper type annotations throughout

Pattern: Value Object Pattern for MergeResult
Reference: KITCHEN_BRIGADE_ARCHITECTURE.md → Cross-Reference Pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.discussion.models import CrossReferenceEvidence


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

_CYCLE_KEY = "cycle"
_SOURCE_TYPE_KEY = "source_type"


# =============================================================================
# MergeResult
# =============================================================================


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result from merging old and new evidence.
    
    AC-KB3.4: Preserves provenance (cycle, source_type) for each evidence item.
    
    Attributes:
        evidence: List of deduplicated evidence items
        duplicates_removed: Count of duplicate items removed
        provenance: Mapping of source_id → provenance info (cycle, source_type)
    """
    
    evidence: list[CrossReferenceEvidence]
    duplicates_removed: int
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    @property
    def total(self) -> int:
        """Return total count of evidence items."""
        return len(self.evidence)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence": [e.to_dict() for e in self.evidence],
            "duplicates_removed": self.duplicates_removed,
            "provenance": dict(self.provenance),
            "total": self.total,
        }


# =============================================================================
# EvidenceMerger
# =============================================================================


class EvidenceMerger:
    """Merges evidence from multiple cycles with deduplication and provenance.
    
    WBS: KB3.5, KB3.6 - Implement merge_evidence() with deduplication
    AC-KB3.3: Evidence from multiple sources merged without duplicates
    AC-KB3.4: merge_evidence() combines old + new evidence, preserving provenance
    
    Deduplication is based on source_id - evidence with the same source_id
    is considered duplicate. Provenance from the earlier cycle is preserved.
    """
    
    def merge(
        self,
        old_evidence: list[CrossReferenceEvidence],
        new_evidence: list[CrossReferenceEvidence],
        current_cycle: int = 1,
    ) -> MergeResult:
        """Merge old and new evidence with deduplication.
        
        AC-KB3.3: Deduplicates based on source_id
        AC-KB3.4: Tracks provenance (cycle, source_type) for each item
        
        Old evidence appears before new evidence in the result, preserving
        the historical order of evidence gathering.
        
        Args:
            old_evidence: Evidence from previous cycles
            new_evidence: Evidence gathered in current cycle
            current_cycle: Current discussion cycle number (1-based)
        
        Returns:
            MergeResult with deduplicated evidence and provenance
        """
        seen_source_ids: set[str] = set()
        merged: list[CrossReferenceEvidence] = []
        provenance: dict[str, dict[str, Any]] = {}
        duplicates_removed = 0
        
        # Previous cycle number for old evidence
        previous_cycle = max(1, current_cycle - 1)
        
        # Process old evidence first (preserves order, tracks as from previous cycle)
        for item in old_evidence:
            if item.source_id in seen_source_ids:
                duplicates_removed += 1
                continue
            
            seen_source_ids.add(item.source_id)
            merged.append(item)
            
            # Track provenance - old evidence came from previous cycle
            provenance[item.source_id] = {
                _CYCLE_KEY: previous_cycle,
                _SOURCE_TYPE_KEY: item.source_type,
            }
        
        # Process new evidence (add only if not duplicate)
        for item in new_evidence:
            if item.source_id in seen_source_ids:
                duplicates_removed += 1
                continue
            
            seen_source_ids.add(item.source_id)
            merged.append(item)
            
            # Track provenance - new evidence from current cycle
            provenance[item.source_id] = {
                _CYCLE_KEY: current_cycle,
                _SOURCE_TYPE_KEY: item.source_type,
            }
        
        return MergeResult(
            evidence=merged,
            duplicates_removed=duplicates_removed,
            provenance=provenance,
        )


# =============================================================================
# Convenience Function
# =============================================================================


def merge_evidence(
    old_evidence: list[CrossReferenceEvidence],
    new_evidence: list[CrossReferenceEvidence],
    current_cycle: int = 1,
) -> MergeResult:
    """Merge old and new evidence with deduplication.
    
    Convenience function that creates an EvidenceMerger and calls merge().
    
    Args:
        old_evidence: Evidence from previous cycles
        new_evidence: Evidence gathered in current cycle
        current_cycle: Current discussion cycle number (1-based)
    
    Returns:
        MergeResult with deduplicated evidence and provenance
    """
    merger = EvidenceMerger()
    return merger.merge(old_evidence, new_evidence, current_cycle)
