"""Data models for LLM Discussion Loop.

WBS Reference: WBS-KB1, WBS-KB2 - LLM Discussion Loop Core + Information Request Detection
Tasks: 
- KB1.3, KB1.4 - Create DiscussionCycle and DiscussionResult dataclasses
- KB2.1 - Create InformationRequest schema
Acceptance Criteria:
- AC-KB1.2: DiscussionCycle captures cycle_number, analyses, agreement_score
- AC-KB1.6: Discussion history preserved as list[DiscussionCycle]
- AC-KB2.1: InformationRequest captures query, source_types, priority
- AC-KB2.4: Requests specify source_types (code, books, textbooks, graph)
- AC-KB2.5: Requests have priority (high/medium/low)

Anti-Patterns Avoided:
- S1192: No duplicated literals (constants at module level)
- Frozen dataclasses for immutability
- Proper type annotations throughout
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

VALID_SOURCE_TYPES = frozenset({"code", "books", "textbooks", "graph"})
VALID_PRIORITIES = frozenset({"high", "medium", "low"})
DEFAULT_SOURCE_TYPES = list(VALID_SOURCE_TYPES)
DEFAULT_PRIORITY = "medium"


# =============================================================================
# CrossReferenceEvidence
# =============================================================================


@dataclass(frozen=True, slots=True)
class CrossReferenceEvidence:
    """Evidence from cross-referencing sources for discussion input.
    
    Attributes:
        source_type: Type of evidence source ("code", "doc", "taxonomy", etc.)
        content: The actual evidence content/text
        source_id: Identifier for the source (file path, URL, etc.)
    """

    source_type: str
    content: str
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type,
            "content": self.content,
            "source_id": self.source_id,
        }


# =============================================================================
# InformationRequest (WBS-KB2)
# =============================================================================


@dataclass(frozen=True, slots=True)
class InformationRequest:
    """Request for additional information from LLM participants.
    
    AC-KB2.1: Captures query, source_types, priority.
    AC-KB2.4: Requests specify source_types (code, books, textbooks, graph).
    AC-KB2.5: Requests have priority (high/medium/low) based on disagreement severity.
    
    Attributes:
        query: The search query for additional information
        source_types: List of source types to search (code, books, textbooks, graph)
        priority: Priority level (high, medium, low)
        reasoning: Optional explanation for why this information is needed
    """

    query: str
    source_types: list[str] = field(default_factory=lambda: list(VALID_SOURCE_TYPES))
    priority: str = DEFAULT_PRIORITY
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "source_types": list(self.source_types),
            "priority": self.priority,
            "reasoning": self.reasoning,
        }


# =============================================================================
# ParticipantAnalysis
# =============================================================================


@dataclass(frozen=True, slots=True)
class ParticipantAnalysis:
    """Analysis result from a single LLM participant.
    
    Attributes:
        participant_id: Unique identifier of the participant
        model_id: Model identifier used for this analysis
        content: The analysis content/text produced
        confidence: Confidence score for this analysis (0.0-1.0)
    """

    participant_id: str
    model_id: str
    content: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participant_id": self.participant_id,
            "model_id": self.model_id,
            "content": self.content,
            "confidence": self.confidence,
        }


# =============================================================================
# DiscussionCycle
# =============================================================================


@dataclass(frozen=True, slots=True)
class DiscussionCycle:
    """A single cycle of discussion among participants.
    
    AC-KB1.2: Captures cycle_number, analyses, agreement_score.
    AC-KB2.2: Includes information_requests extracted from analyses.
    
    Attributes:
        cycle_number: Sequential number of this cycle (1-based)
        analyses: List of analyses from all participants this cycle
        agreement_score: Computed agreement among participants (0.0-1.0)
        disagreement_points: List of identified disagreement topics
        information_requests: Extracted requests for additional information
    """

    cycle_number: int
    analyses: list[ParticipantAnalysis]
    agreement_score: float
    disagreement_points: list[str] = field(default_factory=list)
    information_requests: list["InformationRequest"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_number": self.cycle_number,
            "analyses": [a.to_dict() for a in self.analyses],
            "agreement_score": self.agreement_score,
            "disagreement_points": list(self.disagreement_points),
            "information_requests": [r.to_dict() for r in self.information_requests],
        }


# =============================================================================
# DiscussionResult
# =============================================================================


@dataclass(frozen=True, slots=True)
class DiscussionResult:
    """Final result of a multi-cycle discussion.
    
    AC-KB1.6: Discussion history preserved as list[DiscussionCycle].
    
    Attributes:
        consensus: The agreed-upon answer/conclusion
        confidence: Overall confidence in the consensus (0.0-1.0)
        cycles_used: Number of cycles executed
        history: Complete history of all discussion cycles
    """

    consensus: str
    confidence: float
    cycles_used: int
    history: list[DiscussionCycle]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "consensus": self.consensus,
            "confidence": self.confidence,
            "cycles_used": self.cycles_used,
            "history": [c.to_dict() for c in self.history],
        }
