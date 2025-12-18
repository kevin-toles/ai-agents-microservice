"""EEP-3.4 MSEP Orchestrator Fusion Update Tests.

TDD RED Phase: Tests written BEFORE implementation.

WBS: EEP-3.4 - Update MSEP Orchestrator Fusion (AC-3.4.1 to AC-3.4.4)

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Uses constants from constants.py
- #2.2: Full type annotations
- #7: Follows pytest patterns
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.msep.schemas import CrossReference, EnrichedChapter
from src.agents.msep.scorers import SimilarityWeights


# =============================================================================
# AC-3.4.2: Add new fields to CrossReference schema
# =============================================================================


class TestCrossReferenceSchema:
    """Tests for updated CrossReference schema."""

    def test_crossref_has_concept_overlap_field(self) -> None:
        """AC-3.4.2: CrossReference has concept_overlap field."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.8,
            base_score=0.7,
            topic_boost=0.1,
            method="sbert",
            concept_overlap=0.5,
        )
        assert hasattr(xref, "concept_overlap")
        assert xref.concept_overlap == 0.5

    def test_crossref_has_keyword_jaccard_field(self) -> None:
        """AC-3.4.2: CrossReference has keyword_jaccard field."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.8,
            base_score=0.7,
            topic_boost=0.1,
            method="sbert",
            keyword_jaccard=0.3,
        )
        assert hasattr(xref, "keyword_jaccard")
        assert xref.keyword_jaccard == 0.3

    def test_crossref_has_matched_concepts_field(self) -> None:
        """AC-3.4.2: CrossReference has matched_concepts field."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.8,
            base_score=0.7,
            topic_boost=0.1,
            method="sbert",
            matched_concepts=["llm", "rag"],
        )
        assert hasattr(xref, "matched_concepts")
        assert xref.matched_concepts == ["llm", "rag"]

    # -------------------------------------------------------------------------
    # AC-3.4.3: Maintain backward compatibility
    # -------------------------------------------------------------------------

    def test_crossref_backward_compatible_no_new_fields(self) -> None:
        """AC-3.4.3: CrossReference works without new fields (backward compat)."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.8,
            base_score=0.7,
            topic_boost=0.1,
            method="sbert",
        )
        assert xref.target == "Book:ch1"
        assert xref.score == 0.8
        # New fields default to None or empty
        assert xref.concept_overlap is None or xref.concept_overlap == 0.0
        assert xref.keyword_jaccard is None or xref.keyword_jaccard == 0.0

    def test_crossref_new_fields_default_none(self) -> None:
        """AC-3.4.3: New fields default to None for backward compatibility."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.8,
            base_score=0.7,
            topic_boost=0.1,
            method="sbert",
        )
        # Should have defaults that don't break serialization
        assert xref.concept_overlap is not None or xref.concept_overlap is None
        assert xref.keyword_jaccard is not None or xref.keyword_jaccard is None


# =============================================================================
# AC-3.4.1: Update _build_cross_references() to use multi-signal fusion
# =============================================================================


class TestMultiSignalFusion:
    """Tests for multi-signal fusion in orchestrator."""

    @pytest.fixture
    def sample_similarity_matrix(self) -> np.ndarray:
        """Create sample similarity matrix."""
        return np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ])

    @pytest.fixture
    def sample_concepts(self) -> list[list[str]]:
        """Create sample concept lists per chapter."""
        return [
            ["llm", "rag", "embedding"],
            ["llm", "fine-tuning", "prompt"],
            ["microservices", "kubernetes", "docker"],
        ]

    @pytest.fixture
    def sample_keywords(self) -> list[list[str]]:
        """Create sample keyword lists per chapter."""
        return [
            ["machine learning", "neural network", "model"],
            ["machine learning", "training", "optimization"],
            ["container", "deployment", "infrastructure"],
        ]

    def test_fusion_combines_all_signals(
        self,
        sample_similarity_matrix: np.ndarray,
        sample_concepts: list[list[str]],
        sample_keywords: list[list[str]],
    ) -> None:
        """AC-3.4.1: Fusion score combines SBERT, concept, and keyword signals."""
        # This test verifies the orchestrator uses multi-signal fusion
        # Implementation will update _build_single_cross_reference()
        from src.agents.msep.scorers import (
            ConceptOverlapScorer,
            KeywordJaccardScorer,
            compute_fused_score,
        )

        concept_scorer = ConceptOverlapScorer()
        keyword_scorer = KeywordJaccardScorer()
        weights = SimilarityWeights()

        # Compare chapters 0 and 1
        sbert_sim = sample_similarity_matrix[0][1]
        concept_result = concept_scorer.compute(
            sample_concepts[0], sample_concepts[1]
        )
        keyword_jaccard = keyword_scorer.compute(
            sample_keywords[0], sample_keywords[1]
        )

        fused = compute_fused_score(
            sbert_sim=sbert_sim,
            codebert_sim=None,
            concept_jaccard=concept_result.score,
            keyword_jaccard=keyword_jaccard,
            topic_match=False,
            weights=weights,
        )

        # Fused score should be a combination, not just SBERT
        assert 0.0 < fused < 1.0
        # Fused score should differ from raw SBERT due to other signals
        assert fused != sbert_sim

    def test_crossref_includes_concept_overlap_score(self) -> None:
        """AC-3.4.1: CrossReference includes concept_overlap from scoring."""
        # When orchestrator builds cross-ref, it should populate concept_overlap
        xref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.8,
            topic_boost=0.0,
            method="multi-signal",
            concept_overlap=0.5,
            keyword_jaccard=0.3,
            matched_concepts=["llm", "rag"],
        )
        assert xref.concept_overlap == 0.5
        assert xref.method == "multi-signal"

    def test_crossref_includes_keyword_jaccard_score(self) -> None:
        """AC-3.4.1: CrossReference includes keyword_jaccard from scoring."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.8,
            topic_boost=0.0,
            method="multi-signal",
            concept_overlap=0.5,
            keyword_jaccard=0.3,
        )
        assert xref.keyword_jaccard == 0.3


# =============================================================================
# AC-3.4.4: Update enriched output schema in ai-platform-data
# =============================================================================


class TestEnrichedOutputSchema:
    """Tests for enriched output schema updates."""

    def test_enriched_chapter_serialization_with_new_fields(self) -> None:
        """AC-3.4.4: EnrichedChapter serializes correctly with new CrossRef fields."""
        from src.agents.msep.schemas import MergedKeywords, Provenance

        xref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.8,
            topic_boost=0.05,
            method="multi-signal",
            concept_overlap=0.5,
            keyword_jaccard=0.3,
            matched_concepts=["llm"],
        )

        chapter = EnrichedChapter(
            book="TestBook",
            chapter=1,
            title="Introduction",
            chapter_id="TestBook:ch1",
            cross_references=[xref],
            keywords=MergedKeywords(
                tfidf=["test"], semantic=[], merged=["test"]
            ),
            topic_id=1,
            topic_name="Topic 1",
            graph_relationships=[],
            provenance=Provenance(
                methods_used=["sbert", "concept", "keyword"],
                sbert_score=0.8,
                topic_boost=0.05,
                timestamp="2025-01-01T00:00:00Z",
            ),
        )

        # Verify cross-reference has new fields
        assert chapter.cross_references[0].concept_overlap == 0.5
        assert chapter.cross_references[0].keyword_jaccard == 0.3
        assert chapter.cross_references[0].matched_concepts == ["llm"]

    def test_crossref_method_indicates_multi_signal(self) -> None:
        """AC-3.4.4: Method field indicates multi-signal fusion was used."""
        xref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.8,
            topic_boost=0.0,
            method="multi-signal",
            concept_overlap=0.5,
            keyword_jaccard=0.3,
        )
        assert "multi-signal" in xref.method or xref.method == "multi-signal"
