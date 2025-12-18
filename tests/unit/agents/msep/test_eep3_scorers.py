"""EEP-3 Multi-Level Similarity Scorer Tests.

TDD RED Phase: Tests written BEFORE implementation.

WBS: EEP-3.1 - Similarity Fusion Weights (AC-3.1.1 to AC-3.1.3)
WBS: EEP-3.2 - Concept Overlap Scorer (AC-3.2.1 to AC-3.2.3)
WBS: EEP-3.3 - Keyword Jaccard Scorer (AC-3.3.1 to AC-3.3.2)

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- #2.2: Full type annotations
- #7: Follows pytest patterns
"""

from __future__ import annotations

import pytest

from src.agents.msep.constants import (
    FUSION_WEIGHT_CODEBERT,
    FUSION_WEIGHT_CONCEPT,
    FUSION_WEIGHT_KEYWORD,
    FUSION_WEIGHT_SBERT,
    FUSION_WEIGHT_TOPIC_BOOST,
)
from src.agents.msep.scorers import (
    ConceptOverlapScorer,
    KeywordJaccardScorer,
    SimilarityWeights,
    compute_fused_score,
)


# =============================================================================
# EEP-3.1: Similarity Fusion Weights Tests (AC-3.1.1 to AC-3.1.3)
# =============================================================================


class TestSimilarityWeights:
    """Tests for SimilarityWeights dataclass and constants."""

    # -------------------------------------------------------------------------
    # AC-3.1.1: Define configurable weights in constants.py
    # -------------------------------------------------------------------------

    def test_fusion_weights_constants_exist(self) -> None:
        """AC-3.1.1: Verify fusion weight constants are defined."""
        assert FUSION_WEIGHT_SBERT is not None
        assert FUSION_WEIGHT_CODEBERT is not None
        assert FUSION_WEIGHT_CONCEPT is not None
        assert FUSION_WEIGHT_KEYWORD is not None
        assert FUSION_WEIGHT_TOPIC_BOOST is not None

    def test_fusion_weights_are_positive(self) -> None:
        """AC-3.1.1: Verify all weights are positive floats."""
        assert FUSION_WEIGHT_SBERT > 0
        assert FUSION_WEIGHT_CODEBERT >= 0  # May be 0 if not using code
        assert FUSION_WEIGHT_CONCEPT > 0
        assert FUSION_WEIGHT_KEYWORD > 0
        assert FUSION_WEIGHT_TOPIC_BOOST >= 0

    # -------------------------------------------------------------------------
    # AC-3.1.2: Weights sum to 1.0 (normalized)
    # -------------------------------------------------------------------------

    def test_base_weights_sum_to_one(self) -> None:
        """AC-3.1.2: Verify base weights (excluding topic boost) sum to 1.0."""
        base_sum = (
            FUSION_WEIGHT_SBERT
            + FUSION_WEIGHT_CODEBERT
            + FUSION_WEIGHT_CONCEPT
            + FUSION_WEIGHT_KEYWORD
        )
        assert abs(base_sum - 1.0) < 1e-6, f"Base weights sum to {base_sum}, expected 1.0"

    def test_similarity_weights_dataclass_defaults(self) -> None:
        """AC-3.1.2: SimilarityWeights uses constant defaults."""
        weights = SimilarityWeights()
        assert weights.sbert == FUSION_WEIGHT_SBERT
        assert weights.codebert == FUSION_WEIGHT_CODEBERT
        assert weights.concept == FUSION_WEIGHT_CONCEPT
        assert weights.keyword == FUSION_WEIGHT_KEYWORD
        assert weights.topic_boost == FUSION_WEIGHT_TOPIC_BOOST

    def test_similarity_weights_custom_values(self) -> None:
        """AC-3.1.2: SimilarityWeights accepts custom values."""
        weights = SimilarityWeights(
            sbert=0.4, codebert=0.3, concept=0.2, keyword=0.1, topic_boost=0.1
        )
        assert weights.sbert == 0.4
        assert weights.codebert == 0.3
        assert weights.concept == 0.2
        assert weights.keyword == 0.1
        assert weights.topic_boost == 0.1

    # -------------------------------------------------------------------------
    # AC-3.1.3: Document weight rationale in docstrings
    # -------------------------------------------------------------------------

    def test_similarity_weights_has_docstring(self) -> None:
        """AC-3.1.3: SimilarityWeights class has docstring with rationale."""
        assert SimilarityWeights.__doc__ is not None
        assert "weight" in SimilarityWeights.__doc__.lower()


# =============================================================================
# EEP-3.2: Concept Overlap Scorer Tests (AC-3.2.1 to AC-3.2.3)
# =============================================================================


class TestConceptOverlapScorer:
    """Tests for ConceptOverlapScorer."""

    @pytest.fixture
    def scorer(self) -> ConceptOverlapScorer:
        """Create a ConceptOverlapScorer instance."""
        return ConceptOverlapScorer()

    # -------------------------------------------------------------------------
    # AC-3.2.1: Jaccard similarity between extracted concepts
    # -------------------------------------------------------------------------

    def test_identical_concepts_returns_one(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Identical concept sets return score 1.0."""
        concepts_a = ["llm", "rag", "embedding"]
        concepts_b = ["llm", "rag", "embedding"]
        result = scorer.compute(concepts_a, concepts_b)
        assert result.score == 1.0

    def test_no_overlap_returns_zero(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Disjoint concept sets return score 0.0."""
        concepts_a = ["llm", "rag", "embedding"]
        concepts_b = ["microservices", "kubernetes", "docker"]
        result = scorer.compute(concepts_a, concepts_b)
        assert result.score == 0.0

    def test_partial_overlap_jaccard(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Partial overlap calculates Jaccard correctly."""
        # Jaccard = |A intersection B| / |A union B| = 2 / 4 = 0.5
        concepts_a = ["llm", "rag", "embedding"]
        concepts_b = ["llm", "rag", "microservices"]
        result = scorer.compute(concepts_a, concepts_b)
        assert abs(result.score - 0.5) < 1e-6

    def test_empty_concepts_a_returns_zero(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Empty source concepts return 0.0."""
        result = scorer.compute([], ["llm", "rag"])
        assert result.score == 0.0

    def test_empty_concepts_b_returns_zero(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Empty target concepts return 0.0."""
        result = scorer.compute(["llm", "rag"], [])
        assert result.score == 0.0

    def test_both_empty_returns_zero(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.1: Both empty return 0.0 (no division error)."""
        result = scorer.compute([], [])
        assert result.score == 0.0

    # -------------------------------------------------------------------------
    # AC-3.2.2: Weight parent/child relationships (parent match = 0.5)
    # -------------------------------------------------------------------------

    def test_parent_child_matching_half_weight(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.2: Parent concept match contributes 0.5 of direct match."""
        # Define taxonomy: "llm_rag" is child of "llm"
        taxonomy = {"llm_rag": ["llm"], "embedding_model": ["embedding"]}
        scorer_with_tax = ConceptOverlapScorer(taxonomy=taxonomy)

        # Source has "llm_rag", target has parent "llm"
        concepts_a = ["llm_rag"]
        concepts_b = ["llm"]

        result = scorer_with_tax.compute(concepts_a, concepts_b)
        # Parent match contributes 0.5, no direct match
        assert 0.0 < result.score < 1.0

    def test_direct_match_trumps_parent_match(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.2: Direct match is preferred over parent match."""
        taxonomy = {"llm_rag": ["llm"]}
        scorer_with_tax = ConceptOverlapScorer(taxonomy=taxonomy)

        # Both have "llm" - direct match
        concepts_a = ["llm"]
        concepts_b = ["llm"]

        result = scorer_with_tax.compute(concepts_a, concepts_b)
        assert result.score == 1.0

    # -------------------------------------------------------------------------
    # AC-3.2.3: Return both score and matched concepts list
    # -------------------------------------------------------------------------

    def test_returns_matched_concepts(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.3: Result includes list of matched concepts."""
        concepts_a = ["llm", "rag", "embedding"]
        concepts_b = ["llm", "rag", "microservices"]
        result = scorer.compute(concepts_a, concepts_b)

        assert hasattr(result, "matched_concepts")
        assert set(result.matched_concepts) == {"llm", "rag"}

    def test_matched_concepts_empty_when_no_overlap(
        self, scorer: ConceptOverlapScorer
    ) -> None:
        """AC-3.2.3: matched_concepts is empty for disjoint sets."""
        concepts_a = ["llm"]
        concepts_b = ["microservices"]
        result = scorer.compute(concepts_a, concepts_b)

        assert result.matched_concepts == []


# =============================================================================
# EEP-3.3: Keyword Jaccard Scorer Tests (AC-3.3.1 to AC-3.3.2)
# =============================================================================


class TestKeywordJaccardScorer:
    """Tests for KeywordJaccardScorer."""

    @pytest.fixture
    def scorer(self) -> KeywordJaccardScorer:
        """Create a KeywordJaccardScorer instance."""
        return KeywordJaccardScorer()

    # -------------------------------------------------------------------------
    # AC-3.3.1: Jaccard similarity between filtered TF-IDF keywords
    # -------------------------------------------------------------------------

    def test_identical_keywords_returns_one(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.1: Identical keyword sets return 1.0."""
        keywords_a = ["machine", "learning", "model"]
        keywords_b = ["machine", "learning", "model"]
        result = scorer.compute(keywords_a, keywords_b)
        assert result == 1.0

    def test_no_keyword_overlap_returns_zero(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.1: Disjoint keyword sets return 0.0."""
        keywords_a = ["machine", "learning", "model"]
        keywords_b = ["kubernetes", "docker", "container"]
        result = scorer.compute(keywords_a, keywords_b)
        assert result == 0.0

    def test_partial_keyword_overlap(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.1: Partial keyword overlap calculates Jaccard correctly."""
        # Jaccard = |A intersection B| / |A union B| = 1 / 5 = 0.2
        keywords_a = ["machine", "learning", "model"]
        keywords_b = ["machine", "vision", "image"]
        result = scorer.compute(keywords_a, keywords_b)
        assert abs(result - 0.2) < 1e-6

    def test_empty_keywords_returns_zero(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.1: Empty keyword sets return 0.0."""
        assert scorer.compute([], ["machine"]) == 0.0
        assert scorer.compute(["machine"], []) == 0.0
        assert scorer.compute([], []) == 0.0

    # -------------------------------------------------------------------------
    # AC-3.3.2: Apply n-gram matching (unigrams and bigrams)
    # -------------------------------------------------------------------------

    def test_bigram_matching(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.2: Bigrams are matched correctly."""
        keywords_a = ["machine learning", "deep learning"]
        keywords_b = ["machine learning", "reinforcement learning"]
        result = scorer.compute(keywords_a, keywords_b)
        # Intersection: {"machine learning"}, Union: 3 items
        assert abs(result - 1 / 3) < 1e-6

    def test_mixed_unigram_bigram(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.2: Mixed unigrams and bigrams handled correctly."""
        keywords_a = ["machine", "machine learning", "model"]
        keywords_b = ["machine learning", "model", "training"]
        result = scorer.compute(keywords_a, keywords_b)
        # Intersection: {"machine learning", "model"} = 2
        # Union: {"machine", "machine learning", "model", "training"} = 4
        assert abs(result - 0.5) < 1e-6

    def test_case_insensitive_matching(
        self, scorer: KeywordJaccardScorer
    ) -> None:
        """AC-3.3.2: Matching is case-insensitive."""
        keywords_a = ["Machine", "Learning"]
        keywords_b = ["machine", "learning"]
        result = scorer.compute(keywords_a, keywords_b)
        assert result == 1.0


# =============================================================================
# Fusion Score Tests (Per EEP-3 WBS Formula)
# =============================================================================


class TestComputeFusedScore:
    """Tests for compute_fused_score function."""

    @pytest.fixture
    def default_weights(self) -> SimilarityWeights:
        """Create default weights for testing."""
        return SimilarityWeights()

    def test_sbert_only_score(self, default_weights: SimilarityWeights) -> None:
        """Compute fused score with only SBERT similarity."""
        score = compute_fused_score(
            sbert_sim=0.8,
            codebert_sim=None,
            concept_jaccard=0.0,
            keyword_jaccard=0.0,
            topic_match=False,
            weights=default_weights,
        )
        # SBERT weight redistributed when no CodeBERT
        expected = (
            default_weights.sbert * 0.8
            + default_weights.codebert * 0.8  # Redistributed
        )
        assert abs(score - expected) < 1e-6

    def test_full_signal_score(self, default_weights: SimilarityWeights) -> None:
        """Compute fused score with all signals."""
        score = compute_fused_score(
            sbert_sim=0.8,
            codebert_sim=0.7,
            concept_jaccard=0.6,
            keyword_jaccard=0.5,
            topic_match=False,
            weights=default_weights,
        )
        expected = (
            default_weights.sbert * 0.8
            + default_weights.codebert * 0.7
            + default_weights.concept * 0.6
            + default_weights.keyword * 0.5
        )
        assert abs(score - expected) < 1e-6

    def test_topic_boost_applied(self, default_weights: SimilarityWeights) -> None:
        """Topic match adds boost to score."""
        score_no_boost = compute_fused_score(
            sbert_sim=0.8,
            codebert_sim=None,
            concept_jaccard=0.5,
            keyword_jaccard=0.5,
            topic_match=False,
            weights=default_weights,
        )
        score_with_boost = compute_fused_score(
            sbert_sim=0.8,
            codebert_sim=None,
            concept_jaccard=0.5,
            keyword_jaccard=0.5,
            topic_match=True,
            weights=default_weights,
        )
        diff = score_with_boost - score_no_boost
        assert abs(diff - default_weights.topic_boost) < 1e-6

    def test_score_clamped_to_one(self, default_weights: SimilarityWeights) -> None:
        """Score is clamped to maximum 1.0."""
        score = compute_fused_score(
            sbert_sim=1.0,
            codebert_sim=1.0,
            concept_jaccard=1.0,
            keyword_jaccard=1.0,
            topic_match=True,
            weights=default_weights,
        )
        assert score <= 1.0

    def test_all_zero_returns_zero(self, default_weights: SimilarityWeights) -> None:
        """All zero inputs return zero score."""
        score = compute_fused_score(
            sbert_sim=0.0,
            codebert_sim=0.0,
            concept_jaccard=0.0,
            keyword_jaccard=0.0,
            topic_match=False,
            weights=default_weights,
        )
        assert score == 0.0

    def test_custom_weights(self) -> None:
        """Custom weights are applied correctly."""
        custom = SimilarityWeights(
            sbert=0.5, codebert=0.0, concept=0.3, keyword=0.2, topic_boost=0.1
        )
        score = compute_fused_score(
            sbert_sim=1.0,
            codebert_sim=None,
            concept_jaccard=1.0,
            keyword_jaccard=1.0,
            topic_match=True,
            weights=custom,
        )
        # 0.5*1.0 + 0.0*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1 = 1.1 -> clamped to 1.0
        assert score == 1.0
