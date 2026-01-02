"""Consensus Synthesis for LLM Discussion Loop.

WBS Reference: WBS-KB4 - Agreement/Consensus Engine
Tasks:
- KB4.7: Implement synthesize_consensus()
- KB4.8: Add participant provenance to consensus output

Acceptance Criteria:
- AC-KB4.5: synthesize_consensus() merges analyses when agreement reached
- AC-KB4.6: Consensus tracks which claims came from which participant

Exit Criteria:
- Consensus output identifies "Participant A said X, Participant B agreed"

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity via helper functions
- Frozen dataclasses for immutability
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.discussion.models import ParticipantAnalysis


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_MIN_SUPPORT = 1
_CONST_SENTENCE_SPLITTER = r"(?<=[.!?])\s+"
_CONST_CITATION_PATTERN = r"\[\^(\d+)\]"

logger = logging.getLogger(__name__)


# =============================================================================
# ConsensusConfig
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Configuration for consensus synthesis.
    
    Attributes:
        min_claim_support: Minimum number of participants to include a claim
        include_provenance: Whether to track claim provenance
    """

    min_claim_support: int = _CONST_DEFAULT_MIN_SUPPORT
    include_provenance: bool = True


# =============================================================================
# ConsensusResult (AC-KB4.5, AC-KB4.6)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Result of consensus synthesis from analyses.
    
    AC-KB4.5: Merges analyses when agreement reached.
    AC-KB4.6: Tracks which claims came from which participant.
    
    Attributes:
        content: Synthesized consensus text
        claims: List of claims with participant attribution
        confidence: Overall confidence in the consensus
        participant_contributions: Count of claims per participant
    """

    content: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    participant_contributions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "claims": list(self.claims),
            "confidence": self.confidence,
            "participant_contributions": dict(self.participant_contributions),
        }


# =============================================================================
# extract_claims
# =============================================================================


def extract_claims(content: str) -> list[str]:
    """Extract individual claims from analysis content.
    
    Splits content into sentences, treating each as a potential claim.
    Preserves citation markers for provenance tracking.
    
    Args:
        content: Analysis content text.
        
    Returns:
        List of extracted claim strings.
    """
    if not content or not content.strip():
        return []
    
    # Split by sentence-ending punctuation
    sentences = re.split(_CONST_SENTENCE_SPLITTER, content.strip())
    
    # Filter empty sentences and strip whitespace
    claims = [s.strip() for s in sentences if s.strip()]
    
    return claims


# =============================================================================
# _normalize_claim
# =============================================================================


def _normalize_claim(claim: str) -> str:
    """Normalize a claim for comparison.
    
    Removes citations, lowercases, strips extra whitespace.
    
    Args:
        claim: Original claim text.
        
    Returns:
        Normalized claim string.
    """
    # Remove citation markers
    normalized = re.sub(_CONST_CITATION_PATTERN, "", claim)
    # Lowercase and strip
    normalized = normalized.lower().strip()
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


# =============================================================================
# _find_similar_claims
# =============================================================================


def _find_similar_claims(
    claims_a: list[str],
    claims_b: list[str],
    threshold: float = 0.8,
) -> list[tuple[str, str]]:
    """Find pairs of similar claims between two lists.
    
    Args:
        claims_a: First list of claims.
        claims_b: Second list of claims.
        threshold: Similarity threshold (0.0-1.0).
        
    Returns:
        List of (claim_a, claim_b) tuples that are similar.
    """
    from difflib import SequenceMatcher
    
    similar_pairs: list[tuple[str, str]] = []
    
    for claim_a in claims_a:
        norm_a = _normalize_claim(claim_a)
        for claim_b in claims_b:
            norm_b = _normalize_claim(claim_b)
            similarity = SequenceMatcher(None, norm_a, norm_b).ratio()
            if similarity >= threshold:
                similar_pairs.append((claim_a, claim_b))
    
    return similar_pairs


# =============================================================================
# _merge_claims_with_provenance
# =============================================================================


def _merge_claims_with_provenance(
    analyses: list[ParticipantAnalysis],
    config: ConsensusConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge claims across analyses with participant provenance.
    
    AC-KB4.6: Tracks which claims came from which participant.
    
    Args:
        analyses: List of participant analyses.
        config: Consensus configuration.
        
    Returns:
        Tuple of (claims_with_provenance, participant_contributions).
    """
    if not analyses:
        return [], {}
    
    # Extract claims from each participant
    participant_claims: dict[str, list[str]] = {}
    for analysis in analyses:
        claims = extract_claims(analysis.content)
        participant_claims[analysis.participant_id] = claims
    
    # Track which participants made similar claims
    claim_participants: dict[str, set[str]] = {}  # normalized_claim -> participants
    claim_original: dict[str, str] = {}  # normalized_claim -> first original
    
    for participant_id, claims in participant_claims.items():
        for claim in claims:
            normalized = _normalize_claim(claim)
            if normalized:
                if normalized not in claim_participants:
                    claim_participants[normalized] = set()
                    claim_original[normalized] = claim
                claim_participants[normalized].add(participant_id)
    
    # Group similar normalized claims
    merged_claims: list[dict[str, Any]] = []
    processed: set[str] = set()
    
    for normalized, participants in claim_participants.items():
        if normalized in processed:
            continue
        
        # Find similar claims and merge their participants
        all_participants = set(participants)
        related_normalized: list[str] = [normalized]
        
        for other_norm, other_parts in claim_participants.items():
            if other_norm in processed or other_norm == normalized:
                continue
            # Check similarity
            from difflib import SequenceMatcher
            if SequenceMatcher(None, normalized, other_norm).ratio() >= 0.75:
                all_participants.update(other_parts)
                related_normalized.append(other_norm)
        
        # Mark all related as processed
        for norm in related_normalized:
            processed.add(norm)
        
        # Only include if meets minimum support
        if len(all_participants) >= config.min_claim_support:
            merged_claims.append({
                "text": claim_original[normalized],
                "participants": sorted(all_participants),
            })
    
    # Calculate participant contributions
    contributions: Counter[str] = Counter()
    for claim_data in merged_claims:
        for participant in claim_data["participants"]:
            contributions[participant] += 1
    
    return merged_claims, dict(contributions)


# =============================================================================
# _synthesize_content
# =============================================================================


def _synthesize_content(
    analyses: list[ParticipantAnalysis],
    claims: list[dict[str, Any]],
) -> str:
    """Synthesize consensus content from claims and analyses.
    
    Combines shared claims into coherent text.
    
    Args:
        analyses: Original analyses.
        claims: Merged claims with provenance.
        
    Returns:
        Synthesized consensus text.
    """
    if not analyses:
        return ""
    
    if not claims:
        # Fall back to first analysis content
        return analyses[0].content if analyses else ""
    
    # Build consensus from shared claims (those with 2+ participants)
    shared_claims = [c["text"] for c in claims if len(c.get("participants", [])) >= 2]
    unique_claims = [c["text"] for c in claims if len(c.get("participants", [])) == 1]
    
    # Prioritize shared claims
    if shared_claims:
        content_parts = shared_claims
    else:
        # Use all claims if no shared ones
        content_parts = [c["text"] for c in claims]
    
    # Join claims into coherent text
    content = " ".join(content_parts)
    
    # Add context from unique claims if space permits
    if unique_claims and len(content) < 500:
        # Add unique claims with attribution indicator
        # (actual attribution is in the claims list)
        pass  # Keep consensus focused on shared understanding
    
    return content


# =============================================================================
# _calculate_consensus_confidence
# =============================================================================


def _calculate_consensus_confidence(analyses: list[ParticipantAnalysis]) -> float:
    """Calculate consensus confidence from underlying analyses.
    
    Args:
        analyses: List of participant analyses.
        
    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not analyses:
        return 0.0
    
    confidences = [a.confidence for a in analyses]
    return sum(confidences) / len(confidences)


# =============================================================================
# synthesize_consensus (AC-KB4.5, AC-KB4.6)
# =============================================================================


def synthesize_consensus(
    analyses: list[ParticipantAnalysis],
    config: ConsensusConfig | None = None,
) -> ConsensusResult:
    """Synthesize consensus from list of analyses.
    
    AC-KB4.5: Merges analyses when agreement reached.
    AC-KB4.6: Tracks which claims came from which participant.
    
    Exit Criteria:
    - Consensus output identifies "Participant A said X, Participant B agreed"
    
    Args:
        analyses: List of participant analyses to merge.
        config: Optional configuration settings.
        
    Returns:
        ConsensusResult with synthesized content and provenance.
    """
    if config is None:
        config = ConsensusConfig()
    
    if not analyses:
        return ConsensusResult(
            content="",
            claims=[],
            confidence=0.0,
            participant_contributions={},
        )
    
    # Single analysis case
    if len(analyses) == 1:
        claims = extract_claims(analyses[0].content)
        claim_data = [
            {"text": c, "participants": [analyses[0].participant_id]}
            for c in claims
        ]
        return ConsensusResult(
            content=analyses[0].content,
            claims=claim_data if config.include_provenance else [],
            confidence=analyses[0].confidence,
            participant_contributions={analyses[0].participant_id: len(claims)},
        )
    
    # Merge claims with provenance
    claims, contributions = _merge_claims_with_provenance(analyses, config)
    
    # Synthesize content
    content = _synthesize_content(analyses, claims)
    
    # Calculate confidence
    confidence = _calculate_consensus_confidence(analyses)
    
    logger.info(
        "Consensus synthesized: %d claims from %d participants, confidence=%.3f",
        len(claims),
        len(contributions),
        confidence,
    )
    
    return ConsensusResult(
        content=content,
        claims=claims if config.include_provenance else [],
        confidence=confidence,
        participant_contributions=contributions,
    )
