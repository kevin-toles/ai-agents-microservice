"""Unit tests for Discussion Models.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Tasks: KB1.3, KB1.4 - Create DiscussionCycle and DiscussionResult dataclasses
Acceptance Criteria:
- AC-KB1.2: DiscussionCycle captures cycle_number, analyses, agreement_score
- AC-KB1.6: Discussion history preserved as list[DiscussionCycle]

TDD Phase: RED - Tests written before implementation

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- Frozen dataclasses for immutability
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# =============================================================================
# Test Constants (S1192 compliance)
# =============================================================================

_TEST_PARTICIPANT_ID_A = "participant-a"
_TEST_PARTICIPANT_ID_B = "participant-b"
_TEST_MODEL_ID_A = "qwen2.5-7b"
_TEST_MODEL_ID_B = "deepseek-r1-7b"
_TEST_ANALYSIS_CONTENT = "ParallelAgent uses asyncio.gather"
_TEST_QUERY = "What is the sub-agent pattern?"


# =============================================================================
# KB1.3: CrossReferenceEvidence Tests
# =============================================================================


class TestCrossReferenceEvidence:
    """Evidence model for discussion input."""

    def test_evidence_module_importable(self) -> None:
        """Models module is importable."""
        from src.discussion import models
        assert models is not None

    def test_evidence_class_exists(self) -> None:
        """CrossReferenceEvidence class exists."""
        from src.discussion.models import CrossReferenceEvidence
        assert CrossReferenceEvidence is not None

    def test_evidence_accepts_required_fields(self) -> None:
        """Evidence accepts source_type, content, source_id."""
        from src.discussion.models import CrossReferenceEvidence
        
        evidence = CrossReferenceEvidence(
            source_type="code",
            content=_TEST_ANALYSIS_CONTENT,
            source_id="agents.py#L135",
        )
        
        assert evidence.source_type == "code"
        assert evidence.content == _TEST_ANALYSIS_CONTENT
        assert evidence.source_id == "agents.py#L135"

    def test_evidence_is_frozen(self) -> None:
        """Evidence is immutable (frozen dataclass)."""
        from src.discussion.models import CrossReferenceEvidence
        
        evidence = CrossReferenceEvidence(
            source_type="code",
            content=_TEST_ANALYSIS_CONTENT,
            source_id="agents.py",
        )
        
        with pytest.raises((AttributeError, TypeError)):
            evidence.content = "modified"  # type: ignore


# =============================================================================
# KB1.3: ParticipantAnalysis Tests
# =============================================================================


class TestParticipantAnalysis:
    """Analysis result from a single participant."""

    def test_analysis_class_exists(self) -> None:
        """ParticipantAnalysis class exists."""
        from src.discussion.models import ParticipantAnalysis
        assert ParticipantAnalysis is not None

    def test_analysis_accepts_required_fields(self) -> None:
        """Analysis accepts participant_id, model_id, content, confidence."""
        from src.discussion.models import ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content=_TEST_ANALYSIS_CONTENT,
            confidence=0.85,
        )
        
        assert analysis.participant_id == _TEST_PARTICIPANT_ID_A
        assert analysis.model_id == _TEST_MODEL_ID_A
        assert analysis.content == _TEST_ANALYSIS_CONTENT
        assert analysis.confidence == 0.85

    def test_analysis_default_confidence(self) -> None:
        """Analysis has default confidence of 0.5."""
        from src.discussion.models import ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content=_TEST_ANALYSIS_CONTENT,
        )
        
        assert analysis.confidence == 0.5

    def test_analysis_is_frozen(self) -> None:
        """Analysis is immutable (frozen dataclass)."""
        from src.discussion.models import ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content=_TEST_ANALYSIS_CONTENT,
        )
        
        with pytest.raises((AttributeError, TypeError)):
            analysis.content = "modified"  # type: ignore


# =============================================================================
# KB1.3: DiscussionCycle Tests (AC-KB1.2)
# =============================================================================


class TestDiscussionCycleDataclass:
    """AC-KB1.2: DiscussionCycle captures cycle_number, analyses, agreement_score."""

    def test_cycle_class_exists(self) -> None:
        """DiscussionCycle class exists."""
        from src.discussion.models import DiscussionCycle
        assert DiscussionCycle is not None

    def test_cycle_accepts_required_fields(self) -> None:
        """Cycle accepts cycle_number, analyses, agreement_score."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis_a = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis from A",
            confidence=0.8,
        )
        analysis_b = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_B,
            model_id=_TEST_MODEL_ID_B,
            content="Analysis from B",
            confidence=0.9,
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis_a, analysis_b],
            agreement_score=0.75,
        )
        
        assert cycle.cycle_number == 1
        assert len(cycle.analyses) == 2
        assert cycle.agreement_score == 0.75

    def test_cycle_has_disagreement_points(self) -> None:
        """Cycle captures disagreement_points list."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis_a = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis from A",
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis_a],
            agreement_score=0.6,
            disagreement_points=["runtime vs construction spawning"],
        )
        
        assert cycle.disagreement_points == ["runtime vs construction spawning"]

    def test_cycle_default_disagreement_points(self) -> None:
        """Cycle has empty disagreement_points by default."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis",
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.9,
        )
        
        assert cycle.disagreement_points == []

    def test_cycle_is_frozen(self) -> None:
        """Cycle is immutable (frozen dataclass)."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis",
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.9,
        )
        
        with pytest.raises((AttributeError, TypeError)):
            cycle.cycle_number = 2  # type: ignore


class TestDiscussionCycleSerialization:
    """AC-KB1.2: DiscussionCycle serializes to JSON."""

    def test_cycle_to_dict(self) -> None:
        """Cycle can be converted to dict."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis",
            confidence=0.8,
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.85,
        )
        
        result = cycle.to_dict()
        
        assert isinstance(result, dict)
        assert result["cycle_number"] == 1
        assert result["agreement_score"] == 0.85
        assert len(result["analyses"]) == 1

    def test_cycle_serializes_to_json(self) -> None:
        """Cycle serializes to valid JSON string."""
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis content",
            confidence=0.8,
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.85,
        )
        
        json_str = json.dumps(cycle.to_dict())
        parsed = json.loads(json_str)
        
        assert parsed["cycle_number"] == 1
        assert parsed["agreement_score"] == 0.85


# =============================================================================
# KB1.4: DiscussionResult Tests (AC-KB1.6)
# =============================================================================


class TestDiscussionResultDataclass:
    """AC-KB1.6: DiscussionResult contains history as list[DiscussionCycle]."""

    def test_result_class_exists(self) -> None:
        """DiscussionResult class exists."""
        from src.discussion.models import DiscussionResult
        assert DiscussionResult is not None

    def test_result_accepts_required_fields(self) -> None:
        """Result accepts consensus, confidence, cycles_used, history."""
        from src.discussion.models import (
            DiscussionResult,
            DiscussionCycle,
            ParticipantAnalysis,
        )
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis",
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.9,
        )
        
        result = DiscussionResult(
            consensus="Final agreed answer",
            confidence=0.92,
            cycles_used=1,
            history=[cycle],
        )
        
        assert result.consensus == "Final agreed answer"
        assert result.confidence == 0.92
        assert result.cycles_used == 1
        assert len(result.history) == 1

    def test_result_history_is_list_of_cycles(self) -> None:
        """Result history is list[DiscussionCycle]."""
        from src.discussion.models import (
            DiscussionResult,
            DiscussionCycle,
            ParticipantAnalysis,
        )
        
        analysis_1 = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="First analysis",
        )
        
        analysis_2 = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Second analysis",
        )
        
        cycle_1 = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis_1],
            agreement_score=0.6,
        )
        
        cycle_2 = DiscussionCycle(
            cycle_number=2,
            analyses=[analysis_2],
            agreement_score=0.9,
        )
        
        result = DiscussionResult(
            consensus="Final answer",
            confidence=0.9,
            cycles_used=2,
            history=[cycle_1, cycle_2],
        )
        
        assert all(isinstance(c, DiscussionCycle) for c in result.history)
        assert result.history[0].cycle_number == 1
        assert result.history[1].cycle_number == 2

    def test_result_history_has_one_entry_per_cycle(self) -> None:
        """AC-KB1.6: History has 1 entry per cycle executed."""
        from src.discussion.models import (
            DiscussionResult,
            DiscussionCycle,
            ParticipantAnalysis,
        )
        
        cycles = []
        for i in range(3):
            analysis = ParticipantAnalysis(
                participant_id=f"participant-{i}",
                model_id=_TEST_MODEL_ID_A,
                content=f"Analysis {i}",
            )
            cycles.append(DiscussionCycle(
                cycle_number=i + 1,
                analyses=[analysis],
                agreement_score=0.5 + (i * 0.15),
            ))
        
        result = DiscussionResult(
            consensus="Final answer",
            confidence=0.95,
            cycles_used=3,
            history=cycles,
        )
        
        assert len(result.history) == result.cycles_used
        assert [c.cycle_number for c in result.history] == [1, 2, 3]

    def test_result_is_frozen(self) -> None:
        """Result is immutable (frozen dataclass)."""
        from src.discussion.models import DiscussionResult
        
        result = DiscussionResult(
            consensus="Answer",
            confidence=0.9,
            cycles_used=1,
            history=[],
        )
        
        with pytest.raises((AttributeError, TypeError)):
            result.consensus = "modified"  # type: ignore


class TestDiscussionResultSerialization:
    """DiscussionResult serializes to JSON."""

    def test_result_to_dict(self) -> None:
        """Result can be converted to dict."""
        from src.discussion.models import (
            DiscussionResult,
            DiscussionCycle,
            ParticipantAnalysis,
        )
        
        analysis = ParticipantAnalysis(
            participant_id=_TEST_PARTICIPANT_ID_A,
            model_id=_TEST_MODEL_ID_A,
            content="Analysis",
        )
        
        cycle = DiscussionCycle(
            cycle_number=1,
            analyses=[analysis],
            agreement_score=0.9,
        )
        
        result = DiscussionResult(
            consensus="Final answer",
            confidence=0.92,
            cycles_used=1,
            history=[cycle],
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["consensus"] == "Final answer"
        assert result_dict["confidence"] == 0.92
        assert result_dict["cycles_used"] == 1
        assert len(result_dict["history"]) == 1


# =============================================================================
# KB2.6: InformationRequest Tests (AC-KB2.1, AC-KB2.4, AC-KB2.5)
# WBS Reference: WBS-KB2 - Information Request Detection
# =============================================================================

_TEST_INFO_REQUEST_QUERY = "Need implementation of AST chunking"
_TEST_SOURCE_TYPE_CODE = "code"
_TEST_SOURCE_TYPE_BOOKS = "books"
_TEST_SOURCE_TYPE_TEXTBOOKS = "textbooks"
_TEST_SOURCE_TYPE_GRAPH = "graph"


class TestInformationRequestDataclass:
    """AC-KB2.1: InformationRequest captures query, source_types, priority."""

    def test_information_request_class_exists(self) -> None:
        """InformationRequest class exists in models module."""
        from src.discussion.models import InformationRequest
        assert InformationRequest is not None

    def test_information_request_accepts_required_fields(self) -> None:
        """AC-KB2.1: InformationRequest accepts query, source_types, priority."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="high",
        )
        
        assert request.query == _TEST_INFO_REQUEST_QUERY
        assert request.source_types == [_TEST_SOURCE_TYPE_CODE]
        assert request.priority == "high"

    def test_information_request_multiple_source_types(self) -> None:
        """AC-KB2.4: Request can specify multiple source_types."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[
                _TEST_SOURCE_TYPE_CODE,
                _TEST_SOURCE_TYPE_BOOKS,
                _TEST_SOURCE_TYPE_TEXTBOOKS,
                _TEST_SOURCE_TYPE_GRAPH,
            ],
            priority="medium",
        )
        
        assert len(request.source_types) == 4
        assert _TEST_SOURCE_TYPE_CODE in request.source_types
        assert _TEST_SOURCE_TYPE_BOOKS in request.source_types
        assert _TEST_SOURCE_TYPE_TEXTBOOKS in request.source_types
        assert _TEST_SOURCE_TYPE_GRAPH in request.source_types

    def test_information_request_priority_values(self) -> None:
        """AC-KB2.5: Priority must be high, medium, or low."""
        from src.discussion.models import InformationRequest
        
        # Valid priorities
        for priority in ["high", "medium", "low"]:
            request = InformationRequest(
                query=_TEST_INFO_REQUEST_QUERY,
                source_types=[_TEST_SOURCE_TYPE_CODE],
                priority=priority,
            )
            assert request.priority == priority

    def test_information_request_is_frozen(self) -> None:
        """InformationRequest is immutable (frozen dataclass)."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="high",
        )
        
        with pytest.raises((AttributeError, TypeError)):
            request.query = "modified"  # type: ignore

    def test_information_request_default_priority(self) -> None:
        """InformationRequest has default priority of medium."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
        )
        
        assert request.priority == "medium"

    def test_information_request_default_source_types(self) -> None:
        """InformationRequest has default source_types of all types."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
        )
        
        # Default should include all source types
        assert _TEST_SOURCE_TYPE_CODE in request.source_types


class TestInformationRequestSerialization:
    """InformationRequest serializes to JSON."""

    def test_information_request_to_dict(self) -> None:
        """InformationRequest can be converted to dict."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE, _TEST_SOURCE_TYPE_BOOKS],
            priority="high",
        )
        
        request_dict = request.to_dict()
        
        assert isinstance(request_dict, dict)
        assert request_dict["query"] == _TEST_INFO_REQUEST_QUERY
        assert request_dict["source_types"] == [_TEST_SOURCE_TYPE_CODE, _TEST_SOURCE_TYPE_BOOKS]
        assert request_dict["priority"] == "high"

    def test_information_request_json_serializable(self) -> None:
        """InformationRequest dict is JSON serializable."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="low",
        )
        
        request_dict = request.to_dict()
        json_str = json.dumps(request_dict)
        
        assert isinstance(json_str, str)
        assert _TEST_INFO_REQUEST_QUERY in json_str


class TestInformationRequestValidation:
    """InformationRequest validates input fields."""

    def test_information_request_empty_query_allowed(self) -> None:
        """Empty query is allowed (validation happens at extraction)."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query="",
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="medium",
        )
        
        assert request.query == ""

    def test_information_request_reasoning_field(self) -> None:
        """InformationRequest can include reasoning for the request."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="high",
            reasoning="Participants disagree on chunking implementation",
        )
        
        assert request.reasoning == "Participants disagree on chunking implementation"

    def test_information_request_default_reasoning(self) -> None:
        """InformationRequest has empty string default for reasoning."""
        from src.discussion.models import InformationRequest
        
        request = InformationRequest(
            query=_TEST_INFO_REQUEST_QUERY,
            source_types=[_TEST_SOURCE_TYPE_CODE],
            priority="high",
        )
        
        assert request.reasoning == ""
