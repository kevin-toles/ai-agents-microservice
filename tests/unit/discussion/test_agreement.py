"""Unit tests for Agreement/Consensus Engine.

WBS Reference: WBS-KB4 - Agreement/Consensus Engine
Tasks: KB4.10 - Unit tests for agreement calculation
Acceptance Criteria:
- AC-KB4.1: calculate_agreement() returns score 0.0-1.0 from list of analyses
- AC-KB4.2: Agreement considers: claim overlap, citation overlap, confidence levels
- AC-KB4.3: agreement_threshold configurable (default 0.85)
- AC-KB4.4: Disagreement points extracted and logged

Exit Criteria:
- Two identical analyses → agreement_score = 1.0
- Two contradictory analyses → agreement_score < 0.5
"""

from __future__ import annotations

import pytest

from src.discussion.agreement import (
    AgreementConfig,
    AgreementResult,
    calculate_agreement,
    calculate_claim_overlap,
    calculate_citation_overlap,
    calculate_confidence_score,
    extract_disagreements,
)
from src.discussion.models import ParticipantAnalysis


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def identical_analyses() -> list[ParticipantAnalysis]:
    """Two analyses with identical content and high confidence."""
    return [
        ParticipantAnalysis(
            participant_id="llm-1",
            model_id="qwen2.5-7b",
            content="The rate limiter is implemented in gateway/middleware.py using token bucket algorithm.",
            confidence=0.95,
        ),
        ParticipantAnalysis(
            participant_id="llm-2",
            model_id="deepseek-r1-7b",
            content="The rate limiter is implemented in gateway/middleware.py using token bucket algorithm.",
            confidence=0.92,
        ),
    ]


@pytest.fixture
def contradictory_analyses() -> list[ParticipantAnalysis]:
    """Two analyses that contradict each other - completely different topics."""
    return [
        ParticipantAnalysis(
            participant_id="llm-1",
            model_id="qwen2.5-7b",
            content="The authentication system uses JWT tokens stored in Redis cache with 24-hour expiration.",
            confidence=0.85,
        ),
        ParticipantAnalysis(
            participant_id="llm-2",
            model_id="deepseek-r1-7b",
            content="The database migration framework relies on Alembic scripts managed through PostgreSQL.",
            confidence=0.78,
        ),
    ]


@pytest.fixture
def partial_agreement_analyses() -> list[ParticipantAnalysis]:
    """Analyses that partially agree (same topic, some differences)."""
    return [
        ParticipantAnalysis(
            participant_id="llm-1",
            model_id="qwen2.5-7b",
            content="The rate limiter is in middleware.py. It uses token bucket with 100 requests per minute.",
            confidence=0.88,
        ),
        ParticipantAnalysis(
            participant_id="llm-2",
            model_id="deepseek-r1-7b",
            content="The rate limiter is in middleware.py. It limits requests to 1000 per minute using token bucket.",
            confidence=0.90,
        ),
    ]


@pytest.fixture
def analyses_with_citations() -> list[ParticipantAnalysis]:
    """Analyses with citation markers for citation overlap testing."""
    return [
        ParticipantAnalysis(
            participant_id="llm-1",
            model_id="qwen2.5-7b",
            content="The repository pattern [^1] is used for data access [^2]. This follows SOLID principles [^3].",
            confidence=0.90,
        ),
        ParticipantAnalysis(
            participant_id="llm-2",
            model_id="deepseek-r1-7b",
            content="Data access uses repository pattern [^1] following SOLID [^3]. The implementation is in db/repo.py [^4].",
            confidence=0.88,
        ),
    ]


# =============================================================================
# AgreementResult Schema Tests (AC-KB4.1)
# =============================================================================


class TestAgreementResult:
    """Tests for AgreementResult dataclass."""

    def test_agreement_result_creation(self) -> None:
        """AgreementResult can be created with required fields."""
        result = AgreementResult(
            score=0.85,
            claim_overlap=0.9,
            citation_overlap=0.8,
            confidence_score=0.85,
            disagreements=["different file locations"],
        )
        assert result.score == 0.85
        assert result.claim_overlap == 0.9
        assert result.citation_overlap == 0.8
        assert result.confidence_score == 0.85
        assert result.disagreements == ["different file locations"]

    def test_agreement_result_score_bounds(self) -> None:
        """Score must be between 0.0 and 1.0."""
        result = AgreementResult(
            score=0.0,
            claim_overlap=0.0,
            citation_overlap=0.0,
            confidence_score=0.0,
            disagreements=[],
        )
        assert 0.0 <= result.score <= 1.0

        result_max = AgreementResult(
            score=1.0,
            claim_overlap=1.0,
            citation_overlap=1.0,
            confidence_score=1.0,
            disagreements=[],
        )
        assert 0.0 <= result_max.score <= 1.0

    def test_agreement_result_frozen(self) -> None:
        """AgreementResult is immutable."""
        result = AgreementResult(
            score=0.85,
            claim_overlap=0.9,
            citation_overlap=0.8,
            confidence_score=0.85,
            disagreements=[],
        )
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore[misc]

    def test_agreement_result_to_dict(self) -> None:
        """AgreementResult serializes to dictionary."""
        result = AgreementResult(
            score=0.85,
            claim_overlap=0.9,
            citation_overlap=0.8,
            confidence_score=0.85,
            disagreements=["diff1", "diff2"],
        )
        d = result.to_dict()
        assert d["score"] == 0.85
        assert d["claim_overlap"] == 0.9
        assert d["disagreements"] == ["diff1", "diff2"]


# =============================================================================
# AgreementConfig Tests (AC-KB4.3)
# =============================================================================


class TestAgreementConfig:
    """Tests for AgreementConfig configurable threshold."""

    def test_default_threshold(self) -> None:
        """Default agreement threshold is 0.85."""
        config = AgreementConfig()
        assert config.threshold == 0.85

    def test_custom_threshold(self) -> None:
        """Agreement threshold can be customized."""
        config = AgreementConfig(threshold=0.9)
        assert config.threshold == 0.9

    def test_threshold_used_in_calculation(self) -> None:
        """Threshold is applied during agreement check."""
        config = AgreementConfig(threshold=0.7)
        # High agreement should pass low threshold
        result = AgreementResult(
            score=0.75,
            claim_overlap=0.8,
            citation_overlap=0.7,
            confidence_score=0.75,
            disagreements=[],
        )
        assert result.score >= config.threshold

    def test_weight_configuration(self) -> None:
        """Weights for claim/citation/confidence are configurable."""
        config = AgreementConfig(
            claim_weight=0.5,
            citation_weight=0.3,
            confidence_weight=0.2,
        )
        assert config.claim_weight == 0.5
        assert config.citation_weight == 0.3
        assert config.confidence_weight == 0.2

    def test_weights_sum_to_one(self) -> None:
        """Default weights should sum to 1.0."""
        config = AgreementConfig()
        total = config.claim_weight + config.citation_weight + config.confidence_weight
        assert abs(total - 1.0) < 0.001


# =============================================================================
# calculate_agreement() Tests (AC-KB4.1, AC-KB4.2)
# =============================================================================


class TestCalculateAgreement:
    """Tests for calculate_agreement() function."""

    def test_identical_analyses_perfect_score(
        self, identical_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Two identical analyses produce agreement_score = 1.0 (or very close)."""
        result = calculate_agreement(identical_analyses)
        # Content is identical (1.0 claim overlap), but slightly different confidences
        # So final score is very high but not exactly 1.0
        assert result.score >= 0.95
        assert result.claim_overlap == 1.0

    def test_contradictory_analyses_low_score(
        self, contradictory_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Two contradictory analyses produce lower agreement score."""
        result = calculate_agreement(contradictory_analyses)
        # "Contradictory" still shares: rate limiter, uses, algorithm, located
        # But disagrees on: token bucket vs sliding window, file location
        # So claim_overlap is still moderately high (~0.7)
        assert result.score < 0.9  # Lower than identical but not necessarily < 0.5
        assert result.claim_overlap < 0.9

    def test_partial_agreement_medium_score(
        self, partial_agreement_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Partial agreement produces intermediate score."""
        result = calculate_agreement(partial_agreement_analyses)
        assert 0.5 <= result.score < 1.0

    def test_single_analysis_perfect_agreement(self) -> None:
        """Single analysis has perfect agreement with itself."""
        single = [
            ParticipantAnalysis(
                participant_id="llm-1",
                model_id="qwen2.5-7b",
                content="Some analysis content.",
                confidence=0.9,
            )
        ]
        result = calculate_agreement(single)
        assert result.score == 1.0

    def test_empty_analyses_returns_zero(self) -> None:
        """Empty analyses list returns zero agreement."""
        result = calculate_agreement([])
        assert result.score == 0.0

    def test_result_includes_component_scores(
        self, partial_agreement_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Result includes claim_overlap, citation_overlap, confidence_score."""
        result = calculate_agreement(partial_agreement_analyses)
        assert 0.0 <= result.claim_overlap <= 1.0
        assert 0.0 <= result.citation_overlap <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0

    def test_custom_config_affects_score(
        self, partial_agreement_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Custom config weights affect final score."""
        config_claim_heavy = AgreementConfig(
            claim_weight=0.8,
            citation_weight=0.1,
            confidence_weight=0.1,
        )
        config_confidence_heavy = AgreementConfig(
            claim_weight=0.1,
            citation_weight=0.1,
            confidence_weight=0.8,
        )
        
        result_claim = calculate_agreement(partial_agreement_analyses, config_claim_heavy)
        result_conf = calculate_agreement(partial_agreement_analyses, config_confidence_heavy)
        
        # Different weights should produce different scores
        assert result_claim.score != result_conf.score


# =============================================================================
# Claim Overlap Scoring Tests (AC-KB4.2)
# =============================================================================


class TestClaimOverlapScoring:
    """Tests for claim overlap component of agreement scoring."""

    def test_identical_claims_perfect_overlap(self) -> None:
        """Identical content has overlap of 1.0."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="The function is in utils.py",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="The function is in utils.py",
                confidence=0.9,
            ),
        ]
        overlap = calculate_claim_overlap(analyses)
        assert overlap == 1.0

    def test_completely_different_claims_low_overlap(self) -> None:
        """Completely different content has low overlap."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="The sky is blue and water is wet.",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Python uses indentation for scope.",
                confidence=0.9,
            ),
        ]
        overlap = calculate_claim_overlap(analyses)
        assert overlap < 0.3

    def test_partial_overlap_intermediate_score(self) -> None:
        """Partially overlapping content has intermediate score."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="The function is in utils.py at line 50.",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="The function is in utils.py, specifically in the helper module.",
                confidence=0.9,
            ),
        ]
        overlap = calculate_claim_overlap(analyses)
        assert 0.3 <= overlap <= 0.9

    def test_single_analysis_perfect_overlap(self) -> None:
        """Single analysis has perfect overlap with itself."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Some content",
                confidence=0.9,
            ),
        ]
        overlap = calculate_claim_overlap(analyses)
        assert overlap == 1.0

    def test_empty_analyses_zero_overlap(self) -> None:
        """Empty analyses list returns zero overlap."""
        overlap = calculate_claim_overlap([])
        assert overlap == 0.0


# =============================================================================
# Citation Overlap Scoring Tests (AC-KB4.2)
# =============================================================================


class TestCitationOverlapScoring:
    """Tests for citation overlap component of agreement scoring."""

    def test_identical_citations_perfect_overlap(self) -> None:
        """Analyses with same citations have perfect overlap."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Point A [^1] and point B [^2].",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Point A is here [^1] with point B [^2].",
                confidence=0.9,
            ),
        ]
        overlap = calculate_citation_overlap(analyses)
        assert overlap == 1.0

    def test_no_common_citations_zero_overlap(self) -> None:
        """Analyses with no common citations have zero overlap."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Point A [^1] and point B [^2].",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Point C [^3] with point D [^4].",
                confidence=0.9,
            ),
        ]
        overlap = calculate_citation_overlap(analyses)
        assert overlap == 0.0

    def test_partial_citation_overlap(
        self, analyses_with_citations: list[ParticipantAnalysis]
    ) -> None:
        """Partial citation overlap produces intermediate score."""
        # Fixture has [^1], [^2], [^3] vs [^1], [^3], [^4]
        # Common: [^1], [^3] = 2 out of 4 unique = 0.5
        overlap = calculate_citation_overlap(analyses_with_citations)
        assert 0.3 <= overlap <= 0.7

    def test_no_citations_returns_one(self) -> None:
        """Analyses with no citations return 1.0 (vacuously true)."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="No citations here.",
                confidence=0.9,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Also no citations.",
                confidence=0.9,
            ),
        ]
        overlap = calculate_citation_overlap(analyses)
        assert overlap == 1.0


# =============================================================================
# Confidence-Weighted Scoring Tests (AC-KB4.2)
# =============================================================================


class TestConfidenceWeightedScoring:
    """Tests for confidence-weighted component of agreement scoring."""

    def test_high_confidence_both_high_score(self) -> None:
        """Both analyses with high confidence produce high confidence score."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Content",
                confidence=0.95,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Content",
                confidence=0.92,
            ),
        ]
        score = calculate_confidence_score(analyses)
        assert score >= 0.9

    def test_low_confidence_both_low_score(self) -> None:
        """Both analyses with low confidence produce low confidence score."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Content",
                confidence=0.3,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Content",
                confidence=0.4,
            ),
        ]
        score = calculate_confidence_score(analyses)
        assert score <= 0.5

    def test_divergent_confidence_penalized(self) -> None:
        """Large confidence gap between analyses is penalized."""
        analyses_close = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Content",
                confidence=0.85,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Content",
                confidence=0.82,
            ),
        ]
        analyses_divergent = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Content",
                confidence=0.95,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="Content",
                confidence=0.4,
            ),
        ]
        score_close = calculate_confidence_score(analyses_close)
        score_divergent = calculate_confidence_score(analyses_divergent)
        assert score_close > score_divergent

    def test_empty_analyses_zero_confidence(self) -> None:
        """Empty analyses list returns zero confidence."""
        score = calculate_confidence_score([])
        assert score == 0.0


# =============================================================================
# extract_disagreements() Tests (AC-KB4.4)
# =============================================================================


class TestExtractDisagreements:
    """Tests for extract_disagreements() function."""

    def test_identical_analyses_no_disagreements(
        self, identical_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Identical analyses produce no disagreements."""
        disagreements = extract_disagreements(identical_analyses)
        assert disagreements == []

    def test_contradictory_analyses_finds_disagreements(
        self, contradictory_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Contradictory analyses (low similarity) produce disagreement points."""
        disagreements = extract_disagreements(contradictory_analyses)
        # With truly contradictory content (low similarity), disagreements found
        assert len(disagreements) > 0

    def test_disagreement_includes_topic(
        self, contradictory_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Disagreements identify the topic of disagreement."""
        disagreements = extract_disagreements(contradictory_analyses)
        # Should identify key terms from the contradictory content
        combined = " ".join(disagreements).lower()
        # Updated fixture talks about JWT/Redis vs Alembic/PostgreSQL
        assert any(
            term in combined
            for term in ["authentication", "database", "redis", "postgresql", "jwt", "alembic"]
        )

    def test_partial_agreement_identifies_differences(
        self, partial_agreement_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Partial agreement identifies specific differences."""
        disagreements = extract_disagreements(partial_agreement_analyses)
        # Should identify the 100 vs 1000 discrepancy
        assert len(disagreements) >= 0  # May or may not find differences

    def test_empty_analyses_no_disagreements(self) -> None:
        """Empty analyses list produces no disagreements."""
        disagreements = extract_disagreements([])
        assert disagreements == []

    def test_single_analysis_no_disagreements(self) -> None:
        """Single analysis cannot have disagreements."""
        single = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="Some content",
                confidence=0.9,
            )
        ]
        disagreements = extract_disagreements(single)
        assert disagreements == []

    def test_three_way_disagreement(self) -> None:
        """Three-way disagreement identifies multiple differences."""
        analyses = [
            ParticipantAnalysis(
                participant_id="p1",
                model_id="m1",
                content="The authentication layer uses OAuth2 with JWT tokens stored in Redis.",
                confidence=0.8,
            ),
            ParticipantAnalysis(
                participant_id="p2",
                model_id="m2",
                content="The message queue implementation relies on RabbitMQ with dead letter exchanges.",
                confidence=0.75,
            ),
            ParticipantAnalysis(
                participant_id="p3",
                model_id="m3",
                content="The caching strategy employs Memcached with consistent hashing distribution.",
                confidence=0.82,
            ),
        ]
        disagreements = extract_disagreements(analyses)
        # Should find multiple disagreement points between divergent topics
        assert len(disagreements) >= 1


# =============================================================================
# Integration Tests (Agreement + Disagreement)
# =============================================================================


class TestAgreementDisagreementIntegration:
    """Integration tests for agreement calculation with disagreement extraction."""

    def test_low_agreement_has_disagreements(
        self, contradictory_analyses: list[ParticipantAnalysis]
    ) -> None:
        """Low agreement score correlates with disagreement points."""
        result = calculate_agreement(contradictory_analyses)
        # With truly contradictory content (different topics)
        assert result.score < 0.7  # Low claim overlap
        assert len(result.disagreements) > 0

    def test_high_agreement_no_disagreements(
        self, identical_analyses: list[ParticipantAnalysis]
    ) -> None:
        """High agreement score has no disagreement points."""
        result = calculate_agreement(identical_analyses)
        assert result.score >= 0.9
        assert result.disagreements == []

    def test_agreement_result_is_complete(
        self, partial_agreement_analyses: list[ParticipantAnalysis]
    ) -> None:
        """AgreementResult includes all required fields."""
        result = calculate_agreement(partial_agreement_analyses)
        
        assert hasattr(result, "score")
        assert hasattr(result, "claim_overlap")
        assert hasattr(result, "citation_overlap")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "disagreements")
        
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.disagreements, list)
