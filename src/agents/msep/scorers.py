"""EEP-3 Multi-Level Similarity Scorers.

WBS: EEP-3.2 - Concept Overlap Scorer (AC-3.2.1 to AC-3.2.3)
WBS: EEP-3.3 - Keyword Jaccard Scorer (AC-3.3.1 to AC-3.3.2)

Provides multi-signal similarity scoring for cross-reference enrichment.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- #2.2: Full type annotations on all functions/classes
- S3776: Low cognitive complexity per function
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.msep.constants import (
    FUSION_WEIGHT_CODEBERT,
    FUSION_WEIGHT_CONCEPT,
    FUSION_WEIGHT_KEYWORD,
    FUSION_WEIGHT_SBERT,
    FUSION_WEIGHT_TOPIC_BOOST,
)


# =============================================================================
# EEP-3.1: Similarity Weights Dataclass (AC-3.1.1 to AC-3.1.3)
# =============================================================================


@dataclass
class SimilarityWeights:
    """Configurable weights for multi-signal similarity fusion.

    Per EEP-3 WBS, the fusion formula combines multiple similarity signals
    with configurable weights that sum to 1.0 (excluding topic_boost).

    Weight Rationale:
    - sbert (0.45): Primary semantic signal, captures deep meaning
    - codebert (0.15): Technical code similarity, redistributed when absent
    - concept (0.25): Domain-specific concept overlap from taxonomy
    - keyword (0.15): Surface-level lexical matching via TF-IDF
    - topic_boost (0.15): Additive boost for same BERTopic cluster

    Attributes:
        sbert: Weight for SBERT semantic similarity
        codebert: Weight for CodeBERT code similarity
        concept: Weight for concept Jaccard overlap
        keyword: Weight for keyword Jaccard overlap
        topic_boost: Additive boost for same topic cluster
    """

    sbert: float = field(default=FUSION_WEIGHT_SBERT)
    codebert: float = field(default=FUSION_WEIGHT_CODEBERT)
    concept: float = field(default=FUSION_WEIGHT_CONCEPT)
    keyword: float = field(default=FUSION_WEIGHT_KEYWORD)
    topic_boost: float = field(default=FUSION_WEIGHT_TOPIC_BOOST)


# =============================================================================
# EEP-3.2: Concept Overlap Scorer (AC-3.2.1 to AC-3.2.3)
# =============================================================================


@dataclass
class ConceptOverlapResult:
    """Result from concept overlap scoring.

    Attributes:
        score: Jaccard similarity score [0.0, 1.0]
        matched_concepts: List of concepts that matched directly
        parent_matches: List of concepts matched via parent relationship
    """

    score: float
    matched_concepts: list[str] = field(default_factory=list)
    parent_matches: list[str] = field(default_factory=list)


class ConceptOverlapScorer:
    """Computes concept overlap using Jaccard similarity with taxonomy weighting.

    Per EEP-3.2 requirements:
    - AC-3.2.1: Jaccard similarity between extracted concepts
    - AC-3.2.2: Parent/child relationships weighted at 0.5 of direct match
    - AC-3.2.3: Returns both score and matched concepts list

    Example:
        >>> scorer = ConceptOverlapScorer()
        >>> result = scorer.compute(["llm", "rag"], ["llm", "embedding"])
        >>> print(result.score)  # 0.333... (1 / 3)
        >>> print(result.matched_concepts)  # ["llm"]
    """

    def __init__(self, taxonomy: dict[str, list[str]] | None = None) -> None:
        """Initialize scorer with optional taxonomy for parent matching.

        Args:
            taxonomy: Dict mapping child concepts to list of parent concepts.
                      Example: {"llm_rag": ["llm", "retrieval"]}
        """
        self._taxonomy = taxonomy or {}
        # Build reverse index: parent -> children
        self._parent_to_children: dict[str, set[str]] = {}
        for child, parents in self._taxonomy.items():
            for parent in parents:
                if parent not in self._parent_to_children:
                    self._parent_to_children[parent] = set()
                self._parent_to_children[parent].add(child)

    def compute(
        self, concepts_a: list[str], concepts_b: list[str]
    ) -> ConceptOverlapResult:
        """Compute concept overlap score.

        Args:
            concepts_a: Source chapter concepts
            concepts_b: Target chapter concepts

        Returns:
            ConceptOverlapResult with score and matched concepts
        """
        if not concepts_a or not concepts_b:
            return ConceptOverlapResult(score=0.0)

        set_a = {c.lower() for c in concepts_a}
        set_b = {c.lower() for c in concepts_b}

        # Direct matches (full weight)
        direct_matches = set_a & set_b
        matched_concepts = list(direct_matches)

        # Parent matches (0.5 weight) - AC-3.2.2
        parent_matches: list[str] = []
        parent_match_score = 0.0

        if self._taxonomy:
            parent_match_score, parent_matches = self._compute_parent_matches(
                set_a, set_b, direct_matches
            )

        # Compute Jaccard: |A intersection B| / |A union B|
        intersection_size = len(direct_matches) + (parent_match_score * 0.5)
        union_size = len(set_a | set_b)

        if union_size == 0:
            return ConceptOverlapResult(score=0.0)

        score = intersection_size / union_size

        return ConceptOverlapResult(
            score=score,
            matched_concepts=matched_concepts,
            parent_matches=parent_matches,
        )

    def _compute_parent_matches(
        self,
        set_a: set[str],
        set_b: set[str],
        direct_matches: set[str],
    ) -> tuple[float, list[str]]:
        """Compute parent/child relationship matches.

        Args:
            set_a: Source concepts (lowercase)
            set_b: Target concepts (lowercase)
            direct_matches: Already matched concepts

        Returns:
            Tuple of (match_count, list of parent matches)
        """
        parent_matches: list[str] = []
        match_count = 0.0

        # Check if A's concepts have parents in B
        for concept in set_a - direct_matches:
            parents = self._taxonomy.get(concept, [])
            for parent in parents:
                if parent.lower() in set_b and parent.lower() not in direct_matches:
                    parent_matches.append(parent)
                    match_count += 1
                    break  # Count each concept only once

        # Check if B's concepts have parents in A
        for concept in set_b - direct_matches:
            parents = self._taxonomy.get(concept, [])
            for parent in parents:
                if parent.lower() in set_a and parent.lower() not in direct_matches:
                    parent_matches.append(parent)
                    match_count += 1
                    break

        return match_count, parent_matches


# =============================================================================
# EEP-3.3: Keyword Jaccard Scorer (AC-3.3.1 to AC-3.3.2)
# =============================================================================


class KeywordJaccardScorer:
    """Computes Jaccard similarity between TF-IDF keyword sets.

    Per EEP-3.3 requirements:
    - AC-3.3.1: Jaccard similarity between filtered TF-IDF keywords
    - AC-3.3.2: Apply n-gram matching (unigrams and bigrams)

    Matching is case-insensitive to handle variations in keyword extraction.

    Example:
        >>> scorer = KeywordJaccardScorer()
        >>> score = scorer.compute(["machine", "learning"], ["machine", "vision"])
        >>> print(score)  # 0.333... (1 / 3)
    """

    def compute(self, keywords_a: list[str], keywords_b: list[str]) -> float:
        """Compute Jaccard similarity between keyword sets.

        Args:
            keywords_a: Source chapter keywords (unigrams or bigrams)
            keywords_b: Target chapter keywords (unigrams or bigrams)

        Returns:
            Jaccard similarity score [0.0, 1.0]
        """
        if not keywords_a or not keywords_b:
            return 0.0

        # Case-insensitive matching (AC-3.3.2)
        set_a = {k.lower() for k in keywords_a}
        set_b = {k.lower() for k in keywords_b}

        intersection = set_a & set_b
        union = set_a | set_b

        if not union:
            return 0.0

        return len(intersection) / len(union)


# =============================================================================
# EEP-3: Fusion Score Function (Per WBS Formula)
# =============================================================================


def compute_fused_score(
    sbert_sim: float,
    codebert_sim: float | None,
    concept_jaccard: float,
    keyword_jaccard: float,
    topic_match: bool,
    weights: SimilarityWeights,
) -> float:
    """Compute fused similarity score from multiple signals.

    Per EEP-3 WBS and AI_CODING_PLATFORM_ARCHITECTURE.md:
    - SBERT for natural language similarity
    - CodeBERT for code similarity (when present)
    - Concept Jaccard for domain concept overlap
    - Keyword Jaccard for surface lexical overlap
    - Topic boost for same BERTopic cluster

    Args:
        sbert_sim: SBERT semantic similarity [0.0, 1.0]
        codebert_sim: CodeBERT code similarity [0.0, 1.0] or None if no code
        concept_jaccard: Concept overlap Jaccard score [0.0, 1.0]
        keyword_jaccard: Keyword overlap Jaccard score [0.0, 1.0]
        topic_match: Whether chapters share same topic cluster
        weights: Configurable fusion weights

    Returns:
        Fused similarity score clamped to [0.0, 1.0]
    """
    # Base score (always available)
    score = weights.sbert * sbert_sim
    score += weights.concept * concept_jaccard
    score += weights.keyword * keyword_jaccard

    # Code similarity (only if code blocks present)
    if codebert_sim is not None:
        score += weights.codebert * codebert_sim
    else:
        # Redistribute codebert weight to sbert
        score += weights.codebert * sbert_sim

    # Topic boost (binary)
    if topic_match:
        score += weights.topic_boost

    return min(score, 1.0)  # Clamp to 1.0
