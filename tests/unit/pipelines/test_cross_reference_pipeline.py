"""Unit tests for CrossReferencePipeline.

WBS Reference: WBS-KB6 - Cross-Reference Pipeline Orchestration
Tasks: KB6.2-KB6.11 - Create CrossReferencePipeline class

Acceptance Criteria:
- AC-KB6.1: CrossReferencePipeline orchestrates all KB components
- AC-KB6.2: Pipeline stages: decompose → parallel_retrieval → discussion_loop → validate → format
- AC-KB6.3: Pipeline terminates when: agreement reached OR max_cycles OR validation passed
- AC-KB6.4: Final output is GroundedResponse with content, citations, confidence, metadata
- AC-KB6.5: Metadata includes: cycles_used, participants, sources_consulted, processing_time

TDD Phase: RED
Exit Criteria:
- pytest tests/unit/pipelines/test_cross_reference_pipeline.py passes

Anti-Patterns Avoided:
- #1 (S1192): Test constants at module level
- #42/#43: Proper async/await patterns with pytest-asyncio
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_TEST_QUERY = "Where is the repository pattern implemented?"
_TEST_CONTENT = "The repository pattern is implemented in src/repository.py"
_TEST_CONFIDENCE = 0.92
_TEST_AGREEMENT_THRESHOLD = 0.85


# =============================================================================
# Import Tests
# =============================================================================


class TestCrossReferencePipelineImports:
    """Tests for CrossReferencePipeline module imports."""

    def test_cross_reference_pipeline_importable(self) -> None:
        """CrossReferencePipeline should be importable."""
        from src.pipelines.cross_reference_pipeline import CrossReferencePipeline

        assert CrossReferencePipeline is not None

    def test_cross_reference_config_importable(self) -> None:
        """CrossReferenceConfig should be importable."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        assert CrossReferenceConfig is not None

    def test_pipeline_stage_importable(self) -> None:
        """PipelineStage enum should be importable."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage is not None


# =============================================================================
# PipelineStage Enum Tests (AC-KB6.2)
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_decompose_stage_exists(self) -> None:
        """DECOMPOSE stage should exist."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage.DECOMPOSE == "decompose"

    def test_parallel_retrieval_stage_exists(self) -> None:
        """PARALLEL_RETRIEVAL stage should exist."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage.PARALLEL_RETRIEVAL == "parallel_retrieval"

    def test_discussion_loop_stage_exists(self) -> None:
        """DISCUSSION_LOOP stage should exist."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage.DISCUSSION_LOOP == "discussion_loop"

    def test_validate_stage_exists(self) -> None:
        """VALIDATE stage should exist."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage.VALIDATE == "validate"

    def test_format_stage_exists(self) -> None:
        """FORMAT stage should exist."""
        from src.pipelines.cross_reference_pipeline import PipelineStage

        assert PipelineStage.FORMAT == "format"


# =============================================================================
# CrossReferenceConfig Tests
# =============================================================================


class TestCrossReferenceConfig:
    """Tests for CrossReferenceConfig."""

    def test_config_has_max_cycles(self) -> None:
        """Config should have max_cycles field."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        config = CrossReferenceConfig()
        assert hasattr(config, "max_cycles")
        assert config.max_cycles == 5  # Default

    def test_config_has_agreement_threshold(self) -> None:
        """Config should have agreement_threshold field."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        config = CrossReferenceConfig()
        assert hasattr(config, "agreement_threshold")
        assert config.agreement_threshold == _TEST_AGREEMENT_THRESHOLD

    def test_config_has_validation_enabled(self) -> None:
        """Config should have validation_enabled field."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        config = CrossReferenceConfig()
        assert hasattr(config, "validation_enabled")
        assert config.validation_enabled is True  # Default

    def test_config_has_participant_ids(self) -> None:
        """Config should have participant_ids field."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        config = CrossReferenceConfig(participant_ids=["qwen2.5-7b", "deepseek-r1-7b"])
        assert "qwen2.5-7b" in config.participant_ids

    def test_config_has_source_types(self) -> None:
        """Config should have source_types field."""
        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        config = CrossReferenceConfig(source_types=["code", "books", "graph"])
        assert "code" in config.source_types


# =============================================================================
# CrossReferencePipeline Core Tests (AC-KB6.1)
# =============================================================================


class TestCrossReferencePipelineCore:
    """Core tests for CrossReferencePipeline."""

    def test_pipeline_initializes_with_config(self) -> None:
        """Pipeline should initialize with config."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert pipeline.config is not None

    def test_pipeline_has_run_method(self) -> None:
        """Pipeline should have async run method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "run")
        assert asyncio.iscoroutinefunction(pipeline.run)

    def test_pipeline_has_stages_property(self) -> None:
        """Pipeline should expose stages property."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
            PipelineStage,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "stages")
        # Should have all 5 stages
        assert PipelineStage.DECOMPOSE in pipeline.stages
        assert PipelineStage.PARALLEL_RETRIEVAL in pipeline.stages
        assert PipelineStage.DISCUSSION_LOOP in pipeline.stages
        assert PipelineStage.VALIDATE in pipeline.stages
        assert PipelineStage.FORMAT in pipeline.stages


# =============================================================================
# Pipeline Run Tests (AC-KB6.2)
# =============================================================================


class TestCrossReferencePipelineRun:
    """Tests for CrossReferencePipeline.run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_grounded_response(self) -> None:
        """run() should return GroundedResponse."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import GroundedResponse, GroundedResponseMetadata, TerminationReason

        config = CrossReferenceConfig(participant_ids=["test-participant"])
        pipeline = CrossReferencePipeline(config=config)

        # Mock dependencies with proper GroundedResponseMetadata
        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["test"],
                    sources_consulted=["code"],
                    processing_time_seconds=1.0,
                    termination_reason=TerminationReason.AGREEMENT_REACHED,
                ),
                query=_TEST_QUERY,
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert isinstance(result, GroundedResponse)

    @pytest.mark.asyncio
    async def test_run_executes_all_stages(self) -> None:
        """run() should execute all pipeline stages."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
            PipelineStage,
        )

        config = CrossReferenceConfig(participant_ids=["test-participant"])
        pipeline = CrossReferencePipeline(config=config)

        executed_stages = []

        # Track actual stage methods that exist
        original_decompose = pipeline._stage_decompose
        original_retrieval = pipeline._stage_parallel_retrieval
        original_discussion = pipeline._stage_discussion_loop
        original_validate = pipeline._stage_validate
        original_format = pipeline._stage_format

        async def track_decompose(state: Any) -> Any:
            executed_stages.append(PipelineStage.DECOMPOSE)
            return await original_decompose(state)

        async def track_retrieval(state: Any) -> Any:
            executed_stages.append(PipelineStage.PARALLEL_RETRIEVAL)
            return await original_retrieval(state)

        async def track_discussion(state: Any) -> Any:
            executed_stages.append(PipelineStage.DISCUSSION_LOOP)
            return await original_discussion(state)

        async def track_validate(state: Any) -> Any:
            executed_stages.append(PipelineStage.VALIDATE)
            return await original_validate(state)

        async def track_format(state: Any) -> Any:
            executed_stages.append(PipelineStage.FORMAT)
            return await original_format(state)

        with patch.object(pipeline, "_stage_decompose", side_effect=track_decompose):
            with patch.object(pipeline, "_stage_parallel_retrieval", side_effect=track_retrieval):
                with patch.object(pipeline, "_stage_discussion_loop", side_effect=track_discussion):
                    with patch.object(pipeline, "_stage_validate", side_effect=track_validate):
                        with patch.object(pipeline, "_stage_format", side_effect=track_format):
                            await pipeline.run(query=_TEST_QUERY)

        # Should have executed all 5 stages
        assert len(executed_stages) == 5
        assert executed_stages == [
            PipelineStage.DECOMPOSE,
            PipelineStage.PARALLEL_RETRIEVAL,
            PipelineStage.DISCUSSION_LOOP,
            PipelineStage.VALIDATE,
            PipelineStage.FORMAT,
        ]

    @pytest.mark.asyncio
    async def test_run_records_processing_time(self) -> None:
        """run() should record processing time in metadata."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig(participant_ids=["test-participant"])
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["test"],
                    sources_consulted=["code"],
                    processing_time_seconds=1.5,
                ),
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert result.metadata.processing_time_seconds >= 0


# =============================================================================
# Pipeline Termination Tests (AC-KB6.3)
# =============================================================================


class TestCrossReferencePipelineTermination:
    """Tests for pipeline termination conditions."""

    @pytest.mark.asyncio
    async def test_terminates_on_agreement_reached(self) -> None:
        """Pipeline should terminate when agreement is reached."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import TerminationReason

        config = CrossReferenceConfig(
            participant_ids=["test-participant"],
            agreement_threshold=0.85,
        )
        pipeline = CrossReferencePipeline(config=config)

        # Simulate agreement reached
        should_terminate = pipeline._should_terminate(
            agreement_score=0.92,
            cycle_count=2,
            validation_passed=False,
        )
        assert should_terminate is True

    @pytest.mark.asyncio
    async def test_terminates_on_max_cycles(self) -> None:
        """Pipeline should terminate when max_cycles reached."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig(
            participant_ids=["test-participant"],
            max_cycles=3,
        )
        pipeline = CrossReferencePipeline(config=config)

        # Simulate max cycles reached
        should_terminate = pipeline._should_terminate(
            agreement_score=0.5,  # Not agreed
            cycle_count=3,
            validation_passed=False,
        )
        assert should_terminate is True

    @pytest.mark.asyncio
    async def test_terminates_on_validation_passed(self) -> None:
        """Pipeline should terminate when validation passes."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig(
            participant_ids=["test-participant"],
            validation_enabled=True,
        )
        pipeline = CrossReferencePipeline(config=config)

        # Simulate validation passed
        should_terminate = pipeline._should_terminate(
            agreement_score=0.5,
            cycle_count=1,
            validation_passed=True,
        )
        assert should_terminate is True

    @pytest.mark.asyncio
    async def test_continues_when_no_termination_condition(self) -> None:
        """Pipeline should continue when no termination condition met."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig(
            participant_ids=["test-participant"],
            max_cycles=5,
            agreement_threshold=0.85,
        )
        pipeline = CrossReferencePipeline(config=config)

        # No termination condition met
        should_terminate = pipeline._should_terminate(
            agreement_score=0.5,  # Not agreed
            cycle_count=2,  # Not max
            validation_passed=False,  # Not validated
        )
        assert should_terminate is False

    def test_get_termination_reason_agreement(self) -> None:
        """Should return AGREEMENT_REACHED when agreement threshold met."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import TerminationReason

        config = CrossReferenceConfig(agreement_threshold=0.85)
        pipeline = CrossReferencePipeline(config=config)

        reason = pipeline._get_termination_reason(
            agreement_score=0.92,
            cycle_count=2,
            validation_passed=False,
        )
        assert reason == TerminationReason.AGREEMENT_REACHED

    def test_get_termination_reason_max_cycles(self) -> None:
        """Should return MAX_CYCLES when max cycles reached."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import TerminationReason

        config = CrossReferenceConfig(max_cycles=3)
        pipeline = CrossReferencePipeline(config=config)

        reason = pipeline._get_termination_reason(
            agreement_score=0.5,
            cycle_count=3,
            validation_passed=False,
        )
        assert reason == TerminationReason.MAX_CYCLES

    def test_get_termination_reason_validation_passed(self) -> None:
        """Should return VALIDATION_PASSED when validation passes."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import TerminationReason

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)

        reason = pipeline._get_termination_reason(
            agreement_score=0.5,
            cycle_count=1,
            validation_passed=True,
        )
        assert reason == TerminationReason.VALIDATION_PASSED


# =============================================================================
# Pipeline Stage Tests (AC-KB6.2)
# =============================================================================


class TestCrossReferencePipelineStages:
    """Tests for individual pipeline stages."""

    @pytest.mark.asyncio
    async def test_decompose_stage_exists(self) -> None:
        """Pipeline should have decompose stage method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "_stage_decompose")

    @pytest.mark.asyncio
    async def test_parallel_retrieval_stage_exists(self) -> None:
        """Pipeline should have parallel_retrieval stage method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "_stage_parallel_retrieval")

    @pytest.mark.asyncio
    async def test_discussion_loop_stage_exists(self) -> None:
        """Pipeline should have discussion_loop stage method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "_stage_discussion_loop")

    @pytest.mark.asyncio
    async def test_validate_stage_exists(self) -> None:
        """Pipeline should have validate stage method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "_stage_validate")

    @pytest.mark.asyncio
    async def test_format_stage_exists(self) -> None:
        """Pipeline should have format stage method."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        assert hasattr(pipeline, "_stage_format")


# =============================================================================
# Metadata Collection Tests (AC-KB6.5)
# =============================================================================


class TestCrossReferencePipelineMetadata:
    """Tests for metadata collection."""

    @pytest.mark.asyncio
    async def test_metadata_includes_cycles_used(self) -> None:
        """Metadata should include cycles_used."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig(participant_ids=["test"])
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=3,
                    participants=["test"],
                    sources_consulted=["code"],
                    processing_time_seconds=1.0,
                ),
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert result.metadata.cycles_used == 3

    @pytest.mark.asyncio
    async def test_metadata_includes_participants(self) -> None:
        """Metadata should include participants."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig(participant_ids=["qwen2.5-7b", "deepseek-r1-7b"])
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["qwen2.5-7b", "deepseek-r1-7b"],
                    sources_consulted=["code"],
                    processing_time_seconds=1.0,
                ),
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert "qwen2.5-7b" in result.metadata.participants
            assert "deepseek-r1-7b" in result.metadata.participants

    @pytest.mark.asyncio
    async def test_metadata_includes_sources_consulted(self) -> None:
        """Metadata should include sources_consulted."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig(source_types=["code", "books", "graph"])
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["test"],
                    sources_consulted=["code", "books", "graph"],
                    processing_time_seconds=1.0,
                ),
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert "code" in result.metadata.sources_consulted
            assert "books" in result.metadata.sources_consulted

    @pytest.mark.asyncio
    async def test_metadata_includes_processing_time(self) -> None:
        """Metadata should include processing_time_seconds."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            mock_execute.return_value = GroundedResponse(
                content=_TEST_CONTENT,
                citations=[],
                confidence=_TEST_CONFIDENCE,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["test"],
                    sources_consulted=["code"],
                    processing_time_seconds=2.5,
                ),
            )
            result = await pipeline.run(query=_TEST_QUERY)
            assert result.metadata.processing_time_seconds == 2.5


# =============================================================================
# Integration with KB Components Tests (AC-KB6.1)
# =============================================================================


class TestCrossReferencePipelineIntegration:
    """Tests for integration with KB1-KB5 components."""

    def test_pipeline_uses_discussion_loop(self) -> None:
        """Pipeline should use LLMDiscussionLoop."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        # Should have discussion_loop attribute or method
        assert hasattr(pipeline, "_discussion_loop") or hasattr(
            pipeline, "_stage_discussion_loop"
        )

    def test_pipeline_uses_evidence_gatherer(self) -> None:
        """Pipeline should use EvidenceGatherer."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        # Should have evidence_gatherer attribute or use it in stages
        assert hasattr(pipeline, "_evidence_gatherer") or hasattr(
            pipeline, "_stage_parallel_retrieval"
        )

    def test_pipeline_uses_audit_validator(self) -> None:
        """Pipeline should use AuditServiceValidator."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        # Should have audit_validator attribute or use it in stages
        assert hasattr(pipeline, "_audit_validator") or hasattr(
            pipeline, "_stage_validate"
        )

    def test_pipeline_uses_provenance_tracker(self) -> None:
        """Pipeline should use ProvenanceTracker."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)
        # Should have provenance_tracker attribute
        assert hasattr(pipeline, "_provenance_tracker") or hasattr(
            pipeline, "_track_provenance"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestCrossReferencePipelineEdgeCases:
    """Edge case tests for CrossReferencePipeline."""

    @pytest.mark.asyncio
    async def test_handles_empty_query(self) -> None:
        """Pipeline should handle empty query gracefully."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)

        with pytest.raises(ValueError):
            await pipeline.run(query="")

    @pytest.mark.asyncio
    async def test_handles_no_evidence_found(self) -> None:
        """Pipeline should handle no evidence found."""
        from src.pipelines.cross_reference_pipeline import (
            CrossReferenceConfig,
            CrossReferencePipeline,
        )
        from src.schemas.grounded_response import (
            GroundedResponse,
            GroundedResponseMetadata,
        )

        config = CrossReferenceConfig()
        pipeline = CrossReferencePipeline(config=config)

        with patch.object(pipeline, "_execute_stages") as mock_execute:
            # Return response with low confidence due to no evidence
            mock_execute.return_value = GroundedResponse(
                content="No relevant information found.",
                citations=[],
                confidence=0.1,
                metadata=GroundedResponseMetadata(
                    cycles_used=1,
                    participants=["test"],
                    sources_consulted=["code"],
                    processing_time_seconds=1.0,
                ),
            )
            result = await pipeline.run(query="nonexistent pattern xyz123")
            assert result.confidence < 0.5

    def test_config_validation(self) -> None:
        """Config should validate parameters."""
        from pydantic import ValidationError

        from src.pipelines.cross_reference_pipeline import CrossReferenceConfig

        # max_cycles should be positive
        with pytest.raises(ValidationError):
            CrossReferenceConfig(max_cycles=0)

        # agreement_threshold should be 0-1
        with pytest.raises(ValidationError):
            CrossReferenceConfig(agreement_threshold=1.5)
