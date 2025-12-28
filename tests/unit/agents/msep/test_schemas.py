"""Unit tests for MSEP schemas.

WBS: MSE-2.1 - Input Schema Dataclasses
WBS: MSE-2.2 - Output Schema Dataclasses
TDD Phase: RED (tests written BEFORE implementation)

Acceptance Criteria Coverage:
- AC-2.1.1: ChapterMeta has fields: book, chapter, title, id
- AC-2.1.2: ChapterMeta.id auto-generates as "{book}:ch{chapter}"
- AC-2.1.3: MSEPRequest has fields: corpus, chapter_index, config
- AC-2.1.4: Type annotations pass Mypy strict mode
- AC-2.2.1: CrossReference has: target, score, base_score, topic_boost, method
- AC-2.2.2: MergedKeywords has: tfidf, semantic, merged
- AC-2.2.3: Provenance has: methods_used, sbert_score, topic_boost, timestamp
- AC-2.2.4: EnrichedChapter has all required fields
- AC-2.2.5: EnrichedMetadata.total_similar_chapters computed in __post_init__
- AC-2.2.6: All schemas JSON-serializable via dataclasses.asdict()

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: No duplicated string literals
- #2.2: Full type annotations
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.agents.msep.config import MSEPConfig


# =============================================================================
# MSE-2.1: Input Schema Tests
# =============================================================================


class TestChapterMeta:
    """Tests for ChapterMeta dataclass (AC-2.1.1, AC-2.1.2)."""

    def test_chapter_meta_has_book_field(self) -> None:
        """AC-2.1.1: ChapterMeta should have book field."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Test Book", chapter=1, title="Introduction")

        assert meta.book == "Test Book"

    def test_chapter_meta_has_chapter_field(self) -> None:
        """AC-2.1.1: ChapterMeta should have chapter field (int)."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Test Book", chapter=5, title="Chapter Five")

        assert meta.chapter == 5
        assert isinstance(meta.chapter, int)

    def test_chapter_meta_has_title_field(self) -> None:
        """AC-2.1.1: ChapterMeta should have title field."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Test Book", chapter=1, title="Introduction")

        assert meta.title == "Introduction"

    def test_chapter_meta_has_id_field(self) -> None:
        """AC-2.1.1: ChapterMeta should have id field."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Test Book", chapter=1, title="Introduction")

        assert hasattr(meta, "id")
        assert meta.id is not None

    def test_chapter_meta_id_auto_generates(self) -> None:
        """AC-2.1.2: ChapterMeta.id should auto-generate as '{book}:ch{chapter}'."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Python Distilled", chapter=3, title="Types")

        assert meta.id == "Python Distilled:ch3"

    def test_chapter_meta_id_explicit_override(self) -> None:
        """AC-2.1.2: ChapterMeta.id can be explicitly set."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(
            book="Python Distilled",
            chapter=3,
            title="Types",
            id="custom:id",
        )

        assert meta.id == "custom:id"

    def test_chapter_meta_is_dataclass(self) -> None:
        """AC-2.1.4: ChapterMeta should be a dataclass."""
        from src.agents.msep.schemas import ChapterMeta

        assert dataclasses.is_dataclass(ChapterMeta)


class TestMSEPRequest:
    """Tests for MSEPRequest dataclass (AC-2.1.3)."""

    def test_msep_request_has_corpus_field(self) -> None:
        """AC-2.1.3: MSEPRequest should have corpus field."""
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig

        request = MSEPRequest(
            corpus=["doc1", "doc2"],
            chapter_index=[
                ChapterMeta(book="Book", chapter=1, title="Ch1"),
                ChapterMeta(book="Book", chapter=2, title="Ch2"),
            ],
            config=MSEPConfig(),
        )

        assert request.corpus == ["doc1", "doc2"]
        assert isinstance(request.corpus, list)

    def test_msep_request_has_chapter_index_field(self) -> None:
        """AC-2.1.3: MSEPRequest should have chapter_index field."""
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig

        chapters = [
            ChapterMeta(book="Book", chapter=1, title="Ch1"),
            ChapterMeta(book="Book", chapter=2, title="Ch2"),
        ]
        request = MSEPRequest(
            corpus=["doc1", "doc2"],
            chapter_index=chapters,
            config=MSEPConfig(),
        )

        assert len(request.chapter_index) == 2
        assert all(isinstance(ch, ChapterMeta) for ch in request.chapter_index)

    def test_msep_request_has_config_field(self) -> None:
        """AC-2.1.3: MSEPRequest should have config field."""
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()
        request = MSEPRequest(
            corpus=["doc1"],
            chapter_index=[ChapterMeta(book="Book", chapter=1, title="Ch1")],
            config=config,
        )

        assert request.config is config

    def test_msep_request_is_dataclass(self) -> None:
        """AC-2.1.4: MSEPRequest should be a dataclass."""
        from src.agents.msep.schemas import MSEPRequest

        assert dataclasses.is_dataclass(MSEPRequest)


# =============================================================================
# MSE-2.2: Output Schema Tests
# =============================================================================


class TestCrossReference:
    """Tests for CrossReference dataclass (AC-2.2.1)."""

    def test_cross_reference_has_target_field(self) -> None:
        """AC-2.2.1: CrossReference should have target field."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.target == "Book:ch1"

    def test_cross_reference_has_score_field(self) -> None:
        """AC-2.2.1: CrossReference should have score field."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.score == 0.85

    def test_cross_reference_has_base_score_field(self) -> None:
        """AC-2.2.1: CrossReference should have base_score field."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.base_score == 0.75

    def test_cross_reference_has_topic_boost_field(self) -> None:
        """AC-2.2.1: CrossReference should have topic_boost field."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.topic_boost == 0.10

    def test_cross_reference_has_method_field(self) -> None:
        """AC-2.2.1: CrossReference should have method field."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.method == "sbert"

    def test_cross_reference_is_dataclass(self) -> None:
        """AC-2.2.1: CrossReference should be a dataclass."""
        from src.agents.msep.schemas import CrossReference

        assert dataclasses.is_dataclass(CrossReference)

    def test_cross_reference_has_relationship_type_field(self) -> None:
        """AC-4.4.2: CrossReference should have relationship_type field for EEP-4."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
            relationship_type="PARALLEL",
        )

        assert ref.relationship_type == "PARALLEL"

    def test_cross_reference_relationship_type_default_none(self) -> None:
        """AC-4.4.2: relationship_type should default to None for backward compat."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )

        assert ref.relationship_type is None


class TestMergedKeywords:
    """Tests for MergedKeywords dataclass (AC-2.2.2)."""

    def test_merged_keywords_has_tfidf_field(self) -> None:
        """AC-2.2.2: MergedKeywords should have tfidf field."""
        from src.agents.msep.schemas import MergedKeywords

        kw = MergedKeywords(
            tfidf=["python", "machine learning"],
            semantic=["ai", "neural"],
            merged=["python", "machine learning", "ai"],
        )

        assert kw.tfidf == ["python", "machine learning"]

    def test_merged_keywords_has_semantic_field(self) -> None:
        """AC-2.2.2: MergedKeywords should have semantic field."""
        from src.agents.msep.schemas import MergedKeywords

        kw = MergedKeywords(
            tfidf=["python"],
            semantic=["ai", "neural"],
            merged=["python", "ai"],
        )

        assert kw.semantic == ["ai", "neural"]

    def test_merged_keywords_has_merged_field(self) -> None:
        """AC-2.2.2: MergedKeywords should have merged field."""
        from src.agents.msep.schemas import MergedKeywords

        kw = MergedKeywords(
            tfidf=["python"],
            semantic=["ai"],
            merged=["python", "ai", "neural"],
        )

        assert kw.merged == ["python", "ai", "neural"]

    def test_merged_keywords_is_dataclass(self) -> None:
        """AC-2.2.2: MergedKeywords should be a dataclass."""
        from src.agents.msep.schemas import MergedKeywords

        assert dataclasses.is_dataclass(MergedKeywords)


class TestProvenance:
    """Tests for Provenance dataclass (AC-2.2.3)."""

    def test_provenance_has_methods_used_field(self) -> None:
        """AC-2.2.3: Provenance should have methods_used field."""
        from src.agents.msep.schemas import Provenance

        prov = Provenance(
            methods_used=["sbert", "tfidf", "bertopic"],
            sbert_score=0.85,
            topic_boost=0.10,
            timestamp="2025-12-16T12:00:00Z",
        )

        assert prov.methods_used == ["sbert", "tfidf", "bertopic"]

    def test_provenance_has_sbert_score_field(self) -> None:
        """AC-2.2.3: Provenance should have sbert_score field."""
        from src.agents.msep.schemas import Provenance

        prov = Provenance(
            methods_used=["sbert"],
            sbert_score=0.85,
            topic_boost=0.0,
            timestamp="2025-12-16T12:00:00Z",
        )

        assert prov.sbert_score == 0.85

    def test_provenance_has_topic_boost_field(self) -> None:
        """AC-2.2.3: Provenance should have topic_boost field."""
        from src.agents.msep.schemas import Provenance

        prov = Provenance(
            methods_used=["sbert", "bertopic"],
            sbert_score=0.75,
            topic_boost=0.15,
            timestamp="2025-12-16T12:00:00Z",
        )

        assert prov.topic_boost == 0.15

    def test_provenance_has_timestamp_field(self) -> None:
        """AC-2.2.3: Provenance should have timestamp field."""
        from src.agents.msep.schemas import Provenance

        prov = Provenance(
            methods_used=["sbert"],
            sbert_score=0.85,
            topic_boost=0.0,
            timestamp="2025-12-16T12:00:00Z",
        )

        assert prov.timestamp == "2025-12-16T12:00:00Z"

    def test_provenance_is_dataclass(self) -> None:
        """AC-2.2.3: Provenance should be a dataclass."""
        from src.agents.msep.schemas import Provenance

        assert dataclasses.is_dataclass(Provenance)


class TestEnrichedChapter:
    """Tests for EnrichedChapter dataclass (AC-2.2.4)."""

    def test_enriched_chapter_has_chapter_id_field(self) -> None:
        """AC-2.2.4: EnrichedChapter should have chapter_id field."""
        from src.agents.msep.schemas import (
            EnrichedChapter,
            CrossReference,
            MergedKeywords,
            Provenance,
        )

        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=[],
            keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
            topic_id=0,
            topic_name=None,
            graph_relationships=[],
            provenance=Provenance(
                methods_used=["sbert"],
                sbert_score=0.0,
                topic_boost=0.0,
                timestamp="2025-12-16T12:00:00Z",
            ),
        )

        assert chapter.chapter_id == "Book:ch1"

    def test_enriched_chapter_has_similar_chapters_field(self) -> None:
        """AC-2.2.4: EnrichedChapter should have similar_chapters field."""
        from src.agents.msep.schemas import (
            EnrichedChapter,
            CrossReference,
            MergedKeywords,
            Provenance,
        )

        refs = [
            CrossReference(
                target="Other:ch2",
                score=0.8,
                base_score=0.7,
                topic_boost=0.1,
                method="sbert",
            )
        ]
        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=refs,
            keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
            topic_id=0,
            topic_name=None,
            graph_relationships=[],
            provenance=Provenance(
                methods_used=["sbert"],
                sbert_score=0.8,
                topic_boost=0.1,
                timestamp="2025-12-16T12:00:00Z",
            ),
        )

        assert len(chapter.similar_chapters) == 1

    def test_enriched_chapter_has_keywords_field(self) -> None:
        """AC-2.2.4: EnrichedChapter should have keywords field."""
        from src.agents.msep.schemas import (
            EnrichedChapter,
            MergedKeywords,
            Provenance,
        )

        kw = MergedKeywords(tfidf=["python"], semantic=["ai"], merged=["python", "ai"])
        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=[],
            keywords=kw,
            topic_id=0,
            topic_name=None,
            graph_relationships=[],
            provenance=Provenance(
                methods_used=["sbert"],
                sbert_score=0.0,
                topic_boost=0.0,
                timestamp="2025-12-16T12:00:00Z",
            ),
        )

        assert chapter.keywords is kw

    def test_enriched_chapter_has_topic_id_field(self) -> None:
        """AC-2.2.4: EnrichedChapter should have topic_id field."""
        from src.agents.msep.schemas import (
            EnrichedChapter,
            MergedKeywords,
            Provenance,
        )

        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=[],
            keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
            topic_id=3,
            topic_name="Machine Learning",
            graph_relationships=[],
            provenance=Provenance(
                methods_used=["bertopic"],
                sbert_score=0.0,
                topic_boost=0.0,
                timestamp="2025-12-16T12:00:00Z",
            ),
        )

        assert chapter.topic_id == 3

    def test_enriched_chapter_has_provenance_field(self) -> None:
        """AC-2.2.4: EnrichedChapter should have provenance field."""
        from src.agents.msep.schemas import (
            EnrichedChapter,
            MergedKeywords,
            Provenance,
        )

        prov = Provenance(
            methods_used=["sbert", "tfidf"],
            sbert_score=0.85,
            topic_boost=0.1,
            timestamp="2025-12-16T12:00:00Z",
        )
        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=[],
            keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
            topic_id=0,
            topic_name=None,
            graph_relationships=[],
            provenance=prov,
        )

        assert chapter.provenance is prov

    def test_enriched_chapter_is_dataclass(self) -> None:
        """AC-2.2.4: EnrichedChapter should be a dataclass."""
        from src.agents.msep.schemas import EnrichedChapter

        assert dataclasses.is_dataclass(EnrichedChapter)


class TestEnrichedMetadata:
    """Tests for EnrichedMetadata dataclass (AC-2.2.5)."""

    def test_enriched_metadata_has_chapters_field(self) -> None:
        """AC-2.2.5: EnrichedMetadata should have chapters field."""
        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(
            chapters=[],
            processing_time_ms=100.0,
        )

        assert metadata.chapters == []

    def test_enriched_metadata_has_processing_time_ms_field(self) -> None:
        """AC-2.2.5: EnrichedMetadata should have processing_time_ms field."""
        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(
            chapters=[],
            processing_time_ms=150.5,
        )

        assert metadata.processing_time_ms == 150.5

    def test_enriched_metadata_total_similar_chapters_computed(self) -> None:
        """AC-2.2.5: EnrichedMetadata.total_similar_chapters computed in __post_init__."""
        from src.agents.msep.schemas import (
            EnrichedMetadata,
            EnrichedChapter,
            CrossReference,
            MergedKeywords,
            Provenance,
        )

        ref1 = CrossReference(
            target="A:ch1", score=0.8, base_score=0.7, topic_boost=0.1, method="sbert"
        )
        ref2 = CrossReference(
            target="B:ch2", score=0.7, base_score=0.6, topic_boost=0.1, method="sbert"
        )
        prov = Provenance(
            methods_used=["sbert"],
            sbert_score=0.8,
            topic_boost=0.1,
            timestamp="2025-12-16T12:00:00Z",
        )

        chapters = [
            EnrichedChapter(
                book="Book",
                chapter=1,
                title="Introduction",
                chapter_id="Book:ch1",
                similar_chapters=[ref1, ref2],
                keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
                topic_id=0,
                topic_name=None,
                graph_relationships=[],
                provenance=prov,
            ),
            EnrichedChapter(
                book="Book",
                chapter=2,
                title="Fundamentals",
                chapter_id="Book:ch2",
                similar_chapters=[ref1],
                keywords=MergedKeywords(tfidf=[], semantic=[], merged=[]),
                topic_id=1,
                topic_name="Topic 1",
                graph_relationships=[],
                provenance=prov,
            ),
        ]

        metadata = EnrichedMetadata(
            chapters=chapters,
            processing_time_ms=100.0,
        )

        # 2 refs in ch1 + 1 ref in ch2 = 3 total
        assert metadata.total_similar_chapters == 3

    def test_enriched_metadata_is_dataclass(self) -> None:
        """AC-2.2.5: EnrichedMetadata should be a dataclass."""
        from src.agents.msep.schemas import EnrichedMetadata

        assert dataclasses.is_dataclass(EnrichedMetadata)


class TestSchemasSerialization:
    """Tests for JSON serialization (AC-2.2.6)."""

    def test_chapter_meta_json_serializable(self) -> None:
        """AC-2.2.6: ChapterMeta should be JSON-serializable via asdict."""
        from src.agents.msep.schemas import ChapterMeta

        meta = ChapterMeta(book="Test Book", chapter=1, title="Intro")
        result = dataclasses.asdict(meta)

        assert isinstance(result, dict)
        assert result["book"] == "Test Book"
        assert result["chapter"] == 1
        assert result["title"] == "Intro"
        assert result["id"] == "Test Book:ch1"

    def test_cross_reference_json_serializable(self) -> None:
        """AC-2.2.6: CrossReference should be JSON-serializable via asdict."""
        from src.agents.msep.schemas import CrossReference

        ref = CrossReference(
            target="Book:ch1",
            score=0.85,
            base_score=0.75,
            topic_boost=0.10,
            method="sbert",
        )
        result = dataclasses.asdict(ref)

        assert isinstance(result, dict)
        assert result["target"] == "Book:ch1"
        assert result["score"] == 0.85

    def test_enriched_metadata_json_serializable(self) -> None:
        """AC-2.2.6: EnrichedMetadata (full tree) should be JSON-serializable."""
        from src.agents.msep.schemas import (
            EnrichedMetadata,
            EnrichedChapter,
            CrossReference,
            MergedKeywords,
            Provenance,
        )

        ref = CrossReference(
            target="A:ch1", score=0.8, base_score=0.7, topic_boost=0.1, method="sbert"
        )
        prov = Provenance(
            methods_used=["sbert"],
            sbert_score=0.8,
            topic_boost=0.1,
            timestamp="2025-12-16T12:00:00Z",
        )
        kw = MergedKeywords(tfidf=["python"], semantic=["ai"], merged=["python", "ai"])
        chapter = EnrichedChapter(
            book="Book",
            chapter=1,
            title="Introduction",
            chapter_id="Book:ch1",
            similar_chapters=[ref],
            keywords=kw,
            topic_id=0,
            topic_name=None,
            graph_relationships=["PARALLEL:Book:ch2"],
            provenance=prov,
        )
        metadata = EnrichedMetadata(
            chapters=[chapter],
            processing_time_ms=100.0,
        )

        result = dataclasses.asdict(metadata)

        assert isinstance(result, dict)
        assert "chapters" in result
        assert "processing_time_ms" in result
        assert "total_similar_chapters" in result
        assert len(result["chapters"]) == 1
        # Verify new fields are serialized
        assert result["chapters"][0]["book"] == "Book"
        assert result["chapters"][0]["chapter"] == 1
        assert result["chapters"][0]["title"] == "Introduction"
        assert result["chapters"][0]["topic_name"] is None
        assert result["chapters"][0]["graph_relationships"] == ["PARALLEL:Book:ch2"]
