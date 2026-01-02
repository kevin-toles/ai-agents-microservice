"""Provenance Tracking for LLM Discussion Loop.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Tasks: KB5.1, KB5.2

Acceptance Criteria:
- AC-KB5.4: ProvenanceTracker logs: claim, source, participant, cycle

Exit Criteria:
- Audit trail shows: "Claim X from Participant A in Cycle 2, source: agents.py#L135"

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity via helper functions
- Frozen dataclasses for immutability
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.discussion.consensus import ConsensusResult, extract_claims


if TYPE_CHECKING:
    from src.discussion.models import DiscussionCycle


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_CITATION_MARKER = 0
_CONST_SENTENCE_SPLITTER = r"(?<=[.!?])\s+"

logger = logging.getLogger(__name__)


# =============================================================================
# ProvenanceEntry (AC-KB5.4)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    """A single provenance record tracking claim → source → participant → cycle.
    
    AC-KB5.4: ProvenanceTracker logs: claim, source, participant, cycle.
    
    Attributes:
        claim: The claim text being tracked
        source: Source reference (e.g., "agents.py#L135")
        participant_id: ID of the participant who made the claim
        cycle_number: Discussion cycle number when claim was made
        citation_marker: Optional citation marker number (e.g., 1 for [^1])
    """

    claim: str
    source: str
    participant_id: str
    cycle_number: int
    citation_marker: int = _CONST_DEFAULT_CITATION_MARKER

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim,
            "source": self.source,
            "participant_id": self.participant_id,
            "cycle_number": self.cycle_number,
            "citation_marker": self.citation_marker,
        }


# =============================================================================
# ProvenanceConfig
# =============================================================================


@dataclass(frozen=True, slots=True)
class ProvenanceConfig:
    """Configuration for provenance tracking.
    
    Attributes:
        track_sources: Whether to track source references
        track_participants: Whether to track participant IDs
        track_cycles: Whether to track cycle numbers
    """

    track_sources: bool = True
    track_participants: bool = True
    track_cycles: bool = True


# =============================================================================
# ProvenanceTracker (KB5.1, KB5.2)
# =============================================================================


class ProvenanceTracker:
    """Tracks provenance of claims through discussion cycles.
    
    AC-KB5.4: Logs claim, source, participant, cycle for audit trail.
    
    Exit Criteria:
        Audit trail shows: "Claim X from Participant A in Cycle 2, source: agents.py#L135"
    
    Example:
        >>> tracker = ProvenanceTracker()
        >>> tracker.track_claim(
        ...     claim="Rate limiter uses Redis.",
        ...     source="middleware.py#L135",
        ...     participant_id="llm-1",
        ...     cycle_number=2,
        ... )
        >>> trail = tracker.format_audit_trail()
        >>> print(trail)
        Claim: "Rate limiter uses Redis." from llm-1 in Cycle 2, source: middleware.py#L135
    """

    def __init__(self, config: ProvenanceConfig | None = None) -> None:
        """Initialize the provenance tracker.
        
        Args:
            config: Optional configuration for tracking behavior.
        """
        self._config = config or ProvenanceConfig()
        self._entries: list[ProvenanceEntry] = []

    @property
    def config(self) -> ProvenanceConfig:
        """Get the tracker configuration."""
        return self._config

    def track_claim(
        self,
        claim: str,
        source: str,
        participant_id: str,
        cycle_number: int,
        citation_marker: int = _CONST_DEFAULT_CITATION_MARKER,
    ) -> ProvenanceEntry:
        """Track a claim with its provenance information.
        
        AC-KB5.4: Logs claim, source, participant, cycle.
        
        Args:
            claim: The claim text
            source: Source reference (e.g., "agents.py#L135")
            participant_id: ID of the participant making the claim
            cycle_number: Discussion cycle number
            citation_marker: Optional citation marker number
            
        Returns:
            The created ProvenanceEntry.
        """
        entry = ProvenanceEntry(
            claim=claim,
            source=source if self._config.track_sources else "",
            participant_id=participant_id if self._config.track_participants else "",
            cycle_number=cycle_number if self._config.track_cycles else 0,
            citation_marker=citation_marker,
        )
        self._entries.append(entry)
        logger.debug(
            "Tracked claim: %s (source=%s, participant=%s, cycle=%d)",
            claim[:50],
            source,
            participant_id,
            cycle_number,
        )
        return entry

    def get_entries(self) -> list[ProvenanceEntry]:
        """Get all tracked provenance entries.
        
        Returns:
            List of all ProvenanceEntry objects.
        """
        return list(self._entries)

    def get_entries_by_participant(self, participant_id: str) -> list[ProvenanceEntry]:
        """Get entries filtered by participant.
        
        Args:
            participant_id: The participant ID to filter by.
            
        Returns:
            List of entries from the specified participant.
        """
        return [e for e in self._entries if e.participant_id == participant_id]

    def get_entries_by_cycle(self, cycle_number: int) -> list[ProvenanceEntry]:
        """Get entries filtered by cycle number.
        
        Args:
            cycle_number: The cycle number to filter by.
            
        Returns:
            List of entries from the specified cycle.
        """
        return [e for e in self._entries if e.cycle_number == cycle_number]

    def get_entries_by_source(self, source_pattern: str) -> list[ProvenanceEntry]:
        """Get entries filtered by source pattern.
        
        Args:
            source_pattern: Substring to match in source field.
            
        Returns:
            List of entries with matching source.
        """
        return [e for e in self._entries if source_pattern in e.source]

    def to_audit_trail(self) -> list[dict[str, Any]]:
        """Convert all entries to audit trail format.
        
        Returns:
            List of dictionaries suitable for audit logging.
        """
        return [entry.to_dict() for entry in self._entries]

    def format_audit_trail(self) -> str:
        """Format audit trail for human-readable output.
        
        Exit Criteria format:
            Claim: "X" from Participant A in Cycle 2, source: agents.py#L135
        
        Returns:
            Formatted audit trail string.
        """
        if not self._entries:
            return ""

        lines = []
        for entry in self._entries:
            line = (
                f'Claim: "{entry.claim}" from {entry.participant_id} '
                f"in Cycle {entry.cycle_number}, source: {entry.source}"
            )
            lines.append(line)
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all tracked entries."""
        self._entries.clear()

    def track_from_cycle(
        self,
        cycle: "DiscussionCycle",
        sources: list[str] | None = None,
    ) -> list[ProvenanceEntry]:
        """Track all claims from a discussion cycle.
        
        Extracts claims from each analysis and tracks them with provenance.
        
        Args:
            cycle: The DiscussionCycle to process.
            sources: Optional list of source references.
            
        Returns:
            List of created ProvenanceEntry objects.
        """
        entries = []
        sources = sources or [""]
        default_source = sources[0] if sources else ""

        for analysis in cycle.analyses:
            claims = extract_claims(analysis.content)
            for idx, claim in enumerate(claims):
                source = sources[idx] if idx < len(sources) else default_source
                entry = self.track_claim(
                    claim=claim,
                    source=source,
                    participant_id=analysis.participant_id,
                    cycle_number=cycle.cycle_number,
                )
                entries.append(entry)

        return entries

    def track_consensus(
        self,
        consensus: ConsensusResult,
        source: str,
        cycle_number: int,
    ) -> list[ProvenanceEntry]:
        """Track claims from a consensus result with provenance.
        
        Args:
            consensus: The ConsensusResult to process.
            source: Source reference for the consensus.
            cycle_number: Cycle number when consensus was reached.
            
        Returns:
            List of created ProvenanceEntry objects.
        """
        entries = []

        for claim_dict in consensus.claims:
            claim_text = claim_dict.get("text", "")
            participants = claim_dict.get("participants", [])
            # Use first participant or "consensus" as attribution
            participant_id = participants[0] if participants else "consensus"

            entry = self.track_claim(
                claim=claim_text,
                source=source,
                participant_id=participant_id,
                cycle_number=cycle_number,
            )
            entries.append(entry)

        return entries


__all__ = [
    "ProvenanceConfig",
    "ProvenanceEntry",
    "ProvenanceTracker",
]
