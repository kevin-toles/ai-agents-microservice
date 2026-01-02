"""Agreement/Consensus Engine for LLM Discussion Loop.

WBS Reference: WBS-KB4 - Agreement/Consensus Engine
Tasks:
- KB4.1: Create AgreementResult schema
- KB4.2: Implement calculate_agreement()
- KB4.3: Implement claim overlap scoring
- KB4.4: Implement citation overlap scoring
- KB4.5: Implement confidence-weighted scoring
- KB4.6: Implement extract_disagreements()

Acceptance Criteria:
- AC-KB4.1: calculate_agreement() returns score 0.0-1.0 from list of analyses
- AC-KB4.2: Agreement considers: claim overlap, citation overlap, confidence levels
- AC-KB4.3: agreement_threshold configurable (default 0.85)
- AC-KB4.4: Disagreement points extracted and logged

Exit Criteria:
- Two identical analyses → agreement_score = 1.0
- Two contradictory analyses → agreement_score < 0.5

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity via helper functions
- Frozen dataclasses for immutability
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.discussion.models import ParticipantAnalysis


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_THRESHOLD = 0.85
_CONST_DEFAULT_CLAIM_WEIGHT = 0.4
_CONST_DEFAULT_CITATION_WEIGHT = 0.3
_CONST_DEFAULT_CONFIDENCE_WEIGHT = 0.3
_CONST_CITATION_PATTERN = r"\[\^(\d+)\]"
_CONST_SENTENCE_PATTERN = r"[.!?]+"

logger = logging.getLogger(__name__)


# =============================================================================
# AgreementConfig (AC-KB4.3)
# =============================================================================


@dataclass(frozen=True, slots=True)
class AgreementConfig:
    """Configuration for agreement calculation.
    
    AC-KB4.3: agreement_threshold configurable (default 0.85).
    
    Attributes:
        threshold: Agreement score threshold for consensus (0.0-1.0)
        claim_weight: Weight for claim overlap in final score
        citation_weight: Weight for citation overlap in final score
        confidence_weight: Weight for confidence score in final
    """

    threshold: float = _CONST_DEFAULT_THRESHOLD
    claim_weight: float = _CONST_DEFAULT_CLAIM_WEIGHT
    citation_weight: float = _CONST_DEFAULT_CITATION_WEIGHT
    confidence_weight: float = _CONST_DEFAULT_CONFIDENCE_WEIGHT


# =============================================================================
# AgreementResult (AC-KB4.1)
# =============================================================================


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Result of agreement calculation between analyses.
    
    AC-KB4.1: Returns score 0.0-1.0 from list of analyses.
    AC-KB4.2: Includes component scores for claim, citation, confidence.
    AC-KB4.4: Includes disagreement points.
    
    Attributes:
        score: Overall agreement score (0.0-1.0)
        claim_overlap: Score for claim overlap (0.0-1.0)
        citation_overlap: Score for citation overlap (0.0-1.0)
        confidence_score: Score for confidence alignment (0.0-1.0)
        disagreements: List of identified disagreement points
    """

    score: float
    claim_overlap: float
    citation_overlap: float
    confidence_score: float
    disagreements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "claim_overlap": self.claim_overlap,
            "citation_overlap": self.citation_overlap,
            "confidence_score": self.confidence_score,
            "disagreements": list(self.disagreements),
        }


# =============================================================================
# calculate_claim_overlap (AC-KB4.2)
# =============================================================================


def calculate_claim_overlap(analyses: list[ParticipantAnalysis]) -> float:
    """Calculate claim overlap between analyses.
    
    AC-KB4.2: Agreement considers claim overlap.
    Uses sequence matching to compare content similarity.
    
    Args:
        analyses: List of participant analyses to compare.
        
    Returns:
        Overlap score between 0.0 and 1.0.
    """
    if not analyses:
        return 0.0
    
    if len(analyses) == 1:
        return 1.0
    
    # Pairwise comparison of all analyses
    total_similarity = 0.0
    pair_count = 0
    
    for i, analysis_a in enumerate(analyses):
        for analysis_b in analyses[i + 1:]:
            similarity = SequenceMatcher(
                None,
                analysis_a.content.lower(),
                analysis_b.content.lower(),
            ).ratio()
            total_similarity += similarity
            pair_count += 1
    
    if pair_count == 0:
        return 1.0
    
    return total_similarity / pair_count


# =============================================================================
# calculate_citation_overlap (AC-KB4.2)
# =============================================================================


def _extract_citations(content: str) -> set[str]:
    """Extract citation markers [^N] from content."""
    return set(re.findall(_CONST_CITATION_PATTERN, content))


def calculate_citation_overlap(analyses: list[ParticipantAnalysis]) -> float:
    """Calculate citation overlap between analyses.
    
    AC-KB4.2: Agreement considers citation overlap.
    Uses Jaccard similarity for citation sets.
    
    Args:
        analyses: List of participant analyses to compare.
        
    Returns:
        Overlap score between 0.0 and 1.0.
    """
    if not analyses:
        return 0.0
    
    if len(analyses) == 1:
        return 1.0
    
    # Extract citations from all analyses
    all_citations: list[set[str]] = [
        _extract_citations(a.content) for a in analyses
    ]
    
    # If no citations in any analysis, consider it perfect agreement
    if all(len(c) == 0 for c in all_citations):
        return 1.0
    
    # Calculate pairwise Jaccard similarity
    total_similarity = 0.0
    pair_count = 0
    
    for i, citations_a in enumerate(all_citations):
        for citations_b in all_citations[i + 1:]:
            union = citations_a | citations_b
            if not union:
                # Both empty - perfect agreement
                similarity = 1.0
            else:
                intersection = citations_a & citations_b
                similarity = len(intersection) / len(union)
            total_similarity += similarity
            pair_count += 1
    
    if pair_count == 0:
        return 1.0
    
    return total_similarity / pair_count


# =============================================================================
# calculate_confidence_score (AC-KB4.2)
# =============================================================================


def calculate_confidence_score(analyses: list[ParticipantAnalysis]) -> float:
    """Calculate confidence-weighted score from analyses.
    
    AC-KB4.2: Agreement considers confidence levels.
    Penalizes large confidence gaps between participants.
    
    Args:
        analyses: List of participant analyses to compare.
        
    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not analyses:
        return 0.0
    
    confidences = [a.confidence for a in analyses]
    
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences)
    
    # Penalty for confidence spread (std dev normalized)
    if len(confidences) > 1:
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        # Normalize: max std dev for [0,1] range is 0.5
        spread_penalty = min(std_dev / 0.5, 1.0)
        # Apply penalty (more spread = lower score)
        confidence_score = avg_confidence * (1 - spread_penalty * 0.5)
    else:
        confidence_score = avg_confidence
    
    return max(0.0, min(1.0, confidence_score))


# =============================================================================
# extract_disagreements (AC-KB4.4)
# =============================================================================


def _tokenize_content(content: str) -> set[str]:
    """Tokenize content into lowercase words for comparison."""
    # Remove citations and split into words
    clean = re.sub(_CONST_CITATION_PATTERN, "", content)
    words = re.findall(r"\b\w+\b", clean.lower())
    return set(words)


def _find_unique_terms(
    content_a: str,
    content_b: str,
    min_word_length: int = 4,
) -> tuple[set[str], set[str]]:
    """Find terms unique to each content."""
    tokens_a = _tokenize_content(content_a)
    tokens_b = _tokenize_content(content_b)
    
    # Filter short words
    tokens_a = {t for t in tokens_a if len(t) >= min_word_length}
    tokens_b = {t for t in tokens_b if len(t) >= min_word_length}
    
    unique_a = tokens_a - tokens_b
    unique_b = tokens_b - tokens_a
    
    return unique_a, unique_b


def extract_disagreements(analyses: list[ParticipantAnalysis]) -> list[str]:
    """Extract disagreement points from analyses.
    
    AC-KB4.4: Disagreement points extracted and logged.
    Identifies topics where analyses differ significantly.
    
    Args:
        analyses: List of participant analyses to compare.
        
    Returns:
        List of disagreement descriptions.
    """
    if len(analyses) < 2:
        return []
    
    disagreements: list[str] = []
    
    # Compare each pair
    for i, analysis_a in enumerate(analyses):
        for j, analysis_b in enumerate(analyses[i + 1:], start=i + 1):
            # Check content similarity
            similarity = SequenceMatcher(
                None,
                analysis_a.content.lower(),
                analysis_b.content.lower(),
            ).ratio()
            
            if similarity < 0.7:  # Significant difference
                unique_a, unique_b = _find_unique_terms(
                    analysis_a.content,
                    analysis_b.content,
                )
                
                if unique_a and unique_b:
                    # Find meaningful unique terms
                    key_unique_a = sorted(unique_a, key=len, reverse=True)[:3]
                    key_unique_b = sorted(unique_b, key=len, reverse=True)[:3]
                    
                    if key_unique_a or key_unique_b:
                        disagreement = (
                            f"{analysis_a.participant_id} mentions {', '.join(key_unique_a[:2])} "
                            f"while {analysis_b.participant_id} mentions {', '.join(key_unique_b[:2])}"
                        )
                        disagreements.append(disagreement)
                        logger.debug("Disagreement found: %s", disagreement)
    
    return disagreements


# =============================================================================
# calculate_agreement (AC-KB4.1, AC-KB4.2)
# =============================================================================


def calculate_agreement(
    analyses: list[ParticipantAnalysis],
    config: AgreementConfig | None = None,
) -> AgreementResult:
    """Calculate agreement score from list of analyses.
    
    AC-KB4.1: Returns score 0.0-1.0 from list of analyses.
    AC-KB4.2: Considers claim overlap, citation overlap, confidence levels.
    AC-KB4.4: Extracts disagreement points.
    
    Exit Criteria:
    - Two identical analyses → agreement_score = 1.0
    - Two contradictory analyses → agreement_score < 0.5
    
    Args:
        analyses: List of participant analyses to evaluate.
        config: Optional configuration for weights and threshold.
        
    Returns:
        AgreementResult with score and component scores.
    """
    if config is None:
        config = AgreementConfig()
    
    if not analyses:
        return AgreementResult(
            score=0.0,
            claim_overlap=0.0,
            citation_overlap=0.0,
            confidence_score=0.0,
            disagreements=[],
        )
    
    if len(analyses) == 1:
        return AgreementResult(
            score=1.0,
            claim_overlap=1.0,
            citation_overlap=1.0,
            confidence_score=analyses[0].confidence,
            disagreements=[],
        )
    
    # Calculate component scores
    claim_overlap = calculate_claim_overlap(analyses)
    citation_overlap = calculate_citation_overlap(analyses)
    confidence_score = calculate_confidence_score(analyses)
    
    # Extract disagreements
    disagreements = extract_disagreements(analyses)
    
    # Calculate weighted final score
    final_score = (
        config.claim_weight * claim_overlap
        + config.citation_weight * citation_overlap
        + config.confidence_weight * confidence_score
    )
    
    # Clamp to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))
    
    logger.info(
        "Agreement calculated: score=%.3f (claim=%.3f, citation=%.3f, conf=%.3f)",
        final_score,
        claim_overlap,
        citation_overlap,
        confidence_score,
    )
    
    return AgreementResult(
        score=final_score,
        claim_overlap=claim_overlap,
        citation_overlap=citation_overlap,
        confidence_score=confidence_score,
        disagreements=disagreements,
    )
