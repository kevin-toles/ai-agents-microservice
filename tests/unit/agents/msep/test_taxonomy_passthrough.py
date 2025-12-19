"""Unit tests for MSEP taxonomy pass-through feature.

WBS: Taxonomy Pass-Through Integration
TDD Phase: RED (tests written BEFORE implementation)

Feature Summary:
- MSEPConfig should have taxonomy: str | None = None field
- MSEPRequest should accept taxonomy parameter
- MSEP Merger should apply taxonomy filter to results before returning
- llm-document-enhancer --taxonomy flag should pass through to MSEP

Acceptance Criteria Coverage:
- AC-TAX-1.1: MSEPConfig has taxonomy field with default None
- AC-TAX-1.2: MSEPConfig.from_env() loads MSEP_TAXONOMY from env
- AC-TAX-1.3: MSEPConfig remains frozen (immutable) with taxonomy
- AC-TAX-2.1: MSEPRequest accepts taxonomy in config
- AC-TAX-3.1: Merger filters cross-references by taxonomy books
- AC-TAX-3.2: Merger returns all results when taxonomy is None
- AC-TAX-3.3: Merger filters correctly for multi-book taxonomy

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Constants for test strings
- S3776: Cognitive complexity < 15
- S1172: No unused parameters
- #2.2: Full type annotations
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

# Module constants per S1192 (no duplicated literals)
ENV_PREFIX: str = "MSEP_"
TEST_TAXONOMY: str = "AI-ML_taxonomy"
TEST_TAXONOMY_PATH: str = "/path/to/AI-ML_taxonomy.json"
TEST_BOOK_IN_TAXONOMY: str = "AI Engineering"
TEST_BOOK_NOT_IN_TAXONOMY: str = "Effective Modern C++"


# =============================================================================
# AC-TAX-1: MSEPConfig Taxonomy Field Tests
# =============================================================================


class TestMSEPConfigTaxonomy:
    """Tests for MSEPConfig taxonomy field (AC-TAX-1.x)."""

    def test_config_has_taxonomy_field(self) -> None:
        """AC-TAX-1.1: MSEPConfig should have taxonomy field."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert hasattr(config, "taxonomy")

    def test_config_taxonomy_default_is_none(self) -> None:
        """AC-TAX-1.1: MSEPConfig.taxonomy should default to None."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()

        assert config.taxonomy is None

    def test_config_taxonomy_can_be_set(self) -> None:
        """AC-TAX-1.1: MSEPConfig should accept taxonomy parameter."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig(taxonomy=TEST_TAXONOMY)

        assert config.taxonomy == TEST_TAXONOMY

    def test_config_taxonomy_type_is_optional_string(self) -> None:
        """AC-TAX-1.1: taxonomy should be str | None type."""
        from src.agents.msep.config import MSEPConfig
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(MSEPConfig)}

        assert "taxonomy" in fields
        # Check that default is None (indicates Optional)
        assert fields["taxonomy"].default is None or (
            fields["taxonomy"].default_factory is not dataclasses.MISSING
            and fields["taxonomy"].default_factory() is None  # type: ignore
        )

    def test_config_from_env_loads_taxonomy(self) -> None:
        """AC-TAX-1.2: MSEPConfig.from_env() should load MSEP_TAXONOMY."""
        from src.agents.msep.config import MSEPConfig

        env_vars = {f"{ENV_PREFIX}TAXONOMY": TEST_TAXONOMY}

        with patch.dict(os.environ, env_vars, clear=False):
            config = MSEPConfig.from_env()

        assert config.taxonomy == TEST_TAXONOMY

    def test_config_from_env_taxonomy_default_none(self) -> None:
        """AC-TAX-1.2: taxonomy defaults to None if env var not set."""
        from src.agents.msep.config import MSEPConfig

        # Ensure env var is not set
        env_vars_to_remove = [f"{ENV_PREFIX}TAXONOMY"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars_to_remove}

        with patch.dict(os.environ, clean_env, clear=True):
            config = MSEPConfig.from_env()

        assert config.taxonomy is None

    def test_config_remains_frozen_with_taxonomy(self) -> None:
        """AC-TAX-1.3: MSEPConfig should remain immutable with taxonomy."""
        from src.agents.msep.config import MSEPConfig
        from dataclasses import FrozenInstanceError

        config = MSEPConfig(taxonomy=TEST_TAXONOMY)

        with pytest.raises(FrozenInstanceError):
            config.taxonomy = "different_taxonomy"  # type: ignore[misc]


# =============================================================================
# AC-TAX-2: Constants Module Tests
# =============================================================================


class TestMSEPTaxonomyConstants:
    """Tests for taxonomy-related constants (S1192 compliance)."""

    def test_default_taxonomy_constant_exists(self) -> None:
        """S1192: DEFAULT_TAXONOMY constant should exist."""
        from src.agents.msep.constants import DEFAULT_TAXONOMY

        assert DEFAULT_TAXONOMY is None

    def test_env_taxonomy_key_constant_exists(self) -> None:
        """S1192: ENV_TAXONOMY_KEY constant should exist for from_env()."""
        from src.agents.msep.constants import ENV_TAXONOMY_KEY

        assert ENV_TAXONOMY_KEY == "MSEP_TAXONOMY"


# =============================================================================
# AC-TAX-3: MSEP Merger Taxonomy Filtering Tests
# =============================================================================


class TestMSEPMergerTaxonomyFilter:
    """Tests for MSEP Merger taxonomy filtering (AC-TAX-3.x)."""

    def test_filter_by_taxonomy_function_exists(self) -> None:
        """AC-TAX-3.1: Merger should have filter_by_taxonomy function."""
        from src.agents.msep.merger import filter_by_taxonomy

        assert callable(filter_by_taxonomy)

    def test_filter_returns_all_when_taxonomy_none(self) -> None:
        """AC-TAX-3.2: Returns all results when taxonomy is None."""
        from src.agents.msep.merger import filter_by_taxonomy
        from src.agents.msep.schemas import CrossReference

        cross_refs = [
            CrossReference(
                target=f"{TEST_BOOK_IN_TAXONOMY}:ch1",
                score=0.8,
                base_score=0.7,
                topic_boost=0.1,
                method="sbert",
            ),
            CrossReference(
                target=f"{TEST_BOOK_NOT_IN_TAXONOMY}:ch2",
                score=0.7,
                base_score=0.6,
                topic_boost=0.1,
                method="sbert",
            ),
        ]

        filtered = filter_by_taxonomy(cross_refs, taxonomy_books=None)

        assert len(filtered) == 2  # All returned

    def test_filter_removes_books_not_in_taxonomy(self) -> None:
        """AC-TAX-3.1: Filters out books not in taxonomy."""
        from src.agents.msep.merger import filter_by_taxonomy
        from src.agents.msep.schemas import CrossReference

        cross_refs = [
            CrossReference(
                target=f"{TEST_BOOK_IN_TAXONOMY}:ch1",
                score=0.8,
                base_score=0.7,
                topic_boost=0.1,
                method="sbert",
            ),
            CrossReference(
                target=f"{TEST_BOOK_NOT_IN_TAXONOMY}:ch2",
                score=0.7,
                base_score=0.6,
                topic_boost=0.1,
                method="sbert",
            ),
        ]

        # Mock taxonomy with only AI Engineering book
        taxonomy_books = {TEST_BOOK_IN_TAXONOMY}

        filtered = filter_by_taxonomy(
            cross_refs,
            taxonomy_books=taxonomy_books,
        )

        assert len(filtered) == 1
        assert filtered[0].target.startswith(TEST_BOOK_IN_TAXONOMY)

    def test_filter_handles_empty_cross_refs(self) -> None:
        """AC-TAX-3.1: Handles empty cross-reference list."""
        from src.agents.msep.merger import filter_by_taxonomy

        filtered = filter_by_taxonomy([], taxonomy_books=set())

        assert filtered == []

    def test_filter_preserves_order(self) -> None:
        """AC-TAX-3.3: Preserves cross-reference order after filtering."""
        from src.agents.msep.merger import filter_by_taxonomy
        from src.agents.msep.schemas import CrossReference

        cross_refs = [
            CrossReference(
                target=f"{TEST_BOOK_IN_TAXONOMY}:ch3",
                score=0.9,
                base_score=0.8,
                topic_boost=0.1,
                method="sbert",
            ),
            CrossReference(
                target=f"{TEST_BOOK_NOT_IN_TAXONOMY}:ch1",
                score=0.85,
                base_score=0.75,
                topic_boost=0.1,
                method="sbert",
            ),
            CrossReference(
                target=f"{TEST_BOOK_IN_TAXONOMY}:ch1",
                score=0.7,
                base_score=0.6,
                topic_boost=0.1,
                method="sbert",
            ),
        ]

        taxonomy_books = {TEST_BOOK_IN_TAXONOMY}

        filtered = filter_by_taxonomy(
            cross_refs,
            taxonomy_books=taxonomy_books,
        )

        assert len(filtered) == 2
        # Verify order preserved (ch3 before ch1)
        assert ":ch3" in filtered[0].target
        assert ":ch1" in filtered[1].target


# =============================================================================
# AC-TAX-4: Integration - EnrichedMetadata with Taxonomy
# =============================================================================


class TestEnrichedMetadataWithTaxonomy:
    """Tests for EnrichedMetadata taxonomy integration."""

    def test_merge_results_accepts_taxonomy_param(self) -> None:
        """merge_results should accept taxonomy parameter."""
        from src.agents.msep.merger import merge_results

        # This should not raise - taxonomy is optional
        result = merge_results(
            similarity_matrix=[[1.0, 0.5], [0.5, 1.0]],
            topics=[0, 0],
            keywords=[["test"], ["test"]],
            chapter_ids=["Book:ch1", "Book:ch2"],
            threshold=0.3,
            top_k=5,
            _taxonomy=None,  # Optional param (underscore prefix per S1172)
        )

        assert result is not None

    def test_merge_results_filters_by_taxonomy(self) -> None:
        """merge_results should filter cross-refs by taxonomy when provided."""
        from src.agents.msep.merger import merge_results

        # Similarity matrix: chapter 0 is similar to both chapter 1 and 2
        similarity_matrix = [
            [1.0, 0.8, 0.7],  # ch0 -> ch1: 0.8, ch0 -> ch2: 0.7
            [0.8, 1.0, 0.3],
            [0.7, 0.3, 1.0],
        ]

        # Chapter IDs: one in taxonomy, one not
        chapter_ids = [
            f"{TEST_BOOK_IN_TAXONOMY}:ch1",
            f"{TEST_BOOK_IN_TAXONOMY}:ch2",
            f"{TEST_BOOK_NOT_IN_TAXONOMY}:ch1",
        ]

        result = merge_results(
            similarity_matrix=similarity_matrix,
            topics=[0, 0, 1],
            keywords=[["test"], ["test"], ["test"]],
            chapter_ids=chapter_ids,
            threshold=0.3,
            top_k=5,
            _taxonomy=TEST_TAXONOMY,
            taxonomy_books={TEST_BOOK_IN_TAXONOMY},
        )

        # First chapter's cross-refs should NOT include book not in taxonomy
        ch0_refs = result.chapters[0].cross_references
        for ref in ch0_refs:
            assert TEST_BOOK_NOT_IN_TAXONOMY not in ref.target
