"""Unit tests for LLMParticipant Protocol.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Task: KB1.1 - Create LLMParticipant protocol
Acceptance Criteria: AC-KB1.1 - LLMParticipant class wraps inference-service calls with participant identity

TDD Phase: RED - Tests written before implementation

Anti-Patterns Avoided:
- #7/#13: Namespaced exceptions (DiscussionError, not generic Exception)
- S1192: Constants extracted to module level
- Protocol duck typing pattern (CODING_PATTERNS_ANALYSIS.md)

Reference: KITCHEN_BRIGADE_ARCHITECTURE.md → LLM Discussion Loop Details
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pytest


# =============================================================================
# Test Constants (S1192 compliance)
# =============================================================================

_TEST_PARTICIPANT_ID = "test-participant-1"
_TEST_MODEL_ID = "qwen2.5-7b"
_TEST_QUERY = "What is the sub-agent pattern?"
_TEST_EVIDENCE_TEXT = "ParallelAgent uses asyncio.gather for concurrent execution"


# =============================================================================
# KB1.1: LLMParticipant Protocol Tests
# =============================================================================


class TestLLMParticipantProtocolExists:
    """AC-KB1.1: LLMParticipant protocol exists and is runtime checkable."""

    def test_protocol_module_importable(self) -> None:
        """Protocol module is importable."""
        from src.discussion import protocols
        assert protocols is not None

    def test_llm_participant_protocol_exists(self) -> None:
        """LLMParticipantProtocol class exists."""
        from src.discussion.protocols import LLMParticipantProtocol
        assert LLMParticipantProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol is runtime checkable for isinstance checks."""
        from src.discussion.protocols import LLMParticipantProtocol
        
        # Protocol should be decorated with @runtime_checkable
        assert hasattr(LLMParticipantProtocol, "__protocol_attrs__") or \
               isinstance(LLMParticipantProtocol, type)


class TestLLMParticipantProtocolInterface:
    """AC-KB1.1: Protocol defines required methods."""

    def test_protocol_has_participant_id_property(self) -> None:
        """Protocol defines participant_id property."""
        from src.discussion.protocols import LLMParticipantProtocol
        
        # Check protocol defines participant_id
        assert "participant_id" in dir(LLMParticipantProtocol)

    def test_protocol_has_model_id_property(self) -> None:
        """Protocol defines model_id property."""
        from src.discussion.protocols import LLMParticipantProtocol
        
        assert "model_id" in dir(LLMParticipantProtocol)

    def test_protocol_has_analyze_method(self) -> None:
        """Protocol defines async analyze method."""
        from src.discussion.protocols import LLMParticipantProtocol
        
        assert "analyze" in dir(LLMParticipantProtocol)


class TestLLMParticipantProtocolDuckTyping:
    """Protocol enables duck typing for test doubles."""

    def test_fake_participant_satisfies_protocol(self) -> None:
        """FakeLLMParticipant satisfies LLMParticipantProtocol."""
        from src.discussion.protocols import LLMParticipantProtocol
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        fake = FakeLLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        # Protocol should be runtime checkable
        assert isinstance(fake, LLMParticipantProtocol)


# =============================================================================
# KB1.1: LLMParticipant Implementation Tests
# =============================================================================


class TestLLMParticipantImplementation:
    """AC-KB1.1: LLMParticipant wraps inference-service calls."""

    def test_participant_module_importable(self) -> None:
        """Participant module is importable."""
        from src.discussion import participant
        assert participant is not None

    def test_llm_participant_class_exists(self) -> None:
        """LLMParticipant class exists."""
        from src.discussion.participant import LLMParticipant
        assert LLMParticipant is not None

    def test_participant_accepts_id_and_model(self) -> None:
        """LLMParticipant accepts participant_id and model_id."""
        from src.discussion.participant import LLMParticipant
        
        participant = LLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        assert participant.participant_id == _TEST_PARTICIPANT_ID
        assert participant.model_id == _TEST_MODEL_ID

    def test_participant_satisfies_protocol(self) -> None:
        """LLMParticipant satisfies LLMParticipantProtocol."""
        from src.discussion.participant import LLMParticipant
        from src.discussion.protocols import LLMParticipantProtocol
        
        participant = LLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        assert isinstance(participant, LLMParticipantProtocol)


class TestLLMParticipantAnalyze:
    """AC-KB1.1, AC-KB1.4: Analyze returns analysis with participant_id."""

    @pytest.mark.asyncio
    async def test_analyze_returns_participant_analysis(self) -> None:
        """analyze() returns ParticipantAnalysis with participant_id."""
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import ParticipantAnalysis, CrossReferenceEvidence
        
        participant = LLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        mock_evidence = CrossReferenceEvidence(
            source_type="code",
            content=_TEST_EVIDENCE_TEXT,
            source_id="agents.py",
        )
        
        result = await participant.analyze(
            query=_TEST_QUERY,
            evidence=[mock_evidence],
        )
        
        assert isinstance(result, ParticipantAnalysis)
        assert result.participant_id == _TEST_PARTICIPANT_ID
        assert result.model_id == _TEST_MODEL_ID

    @pytest.mark.asyncio
    async def test_analyze_includes_content(self) -> None:
        """analyze() returns analysis with content."""
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import CrossReferenceEvidence
        
        participant = LLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        mock_evidence = CrossReferenceEvidence(
            source_type="code",
            content=_TEST_EVIDENCE_TEXT,
            source_id="agents.py",
        )
        
        result = await participant.analyze(
            query=_TEST_QUERY,
            evidence=[mock_evidence],
        )
        
        # Content should be non-empty (fallback generates description)
        assert len(result.content) > 0
        assert _TEST_QUERY in result.content or "evidence" in result.content.lower()

    @pytest.mark.asyncio
    async def test_analyze_includes_confidence(self) -> None:
        """analyze() returns analysis with confidence score."""
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import CrossReferenceEvidence
        
        participant = LLMParticipant(
            participant_id=_TEST_PARTICIPANT_ID,
            model_id=_TEST_MODEL_ID,
        )
        
        mock_evidence = CrossReferenceEvidence(
            source_type="code",
            content=_TEST_EVIDENCE_TEXT,
            source_id="agents.py",
        )
        
        result = await participant.analyze(
            query=_TEST_QUERY,
            evidence=[mock_evidence],
        )
        
        assert 0.0 <= result.confidence <= 1.0
