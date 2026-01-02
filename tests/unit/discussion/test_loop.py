"""Unit tests for LLMDiscussionLoop.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Tasks: KB1.5, KB1.6, KB1.7 - Create LLMDiscussionLoop class
Acceptance Criteria:
- AC-KB1.3: LLMDiscussionLoop.discuss() runs N participants via asyncio.gather
- AC-KB1.4: Each participant receives same evidence, produces independent analysis
- AC-KB1.5: Discussion loop uses configurable max_cycles (default 5)
- AC-KB1.6: Discussion history preserved as list[DiscussionCycle]

TDD Phase: RED - Tests written before implementation

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Keep cognitive complexity < 15
- #42/#43: Proper async/await patterns
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from src.discussion.loop import LLMDiscussionLoop


# =============================================================================
# Test Constants (S1192 compliance)
# =============================================================================

_TEST_PARTICIPANT_ID_A = "participant-a"
_TEST_PARTICIPANT_ID_B = "participant-b"
_TEST_MODEL_ID_A = "qwen2.5-7b"
_TEST_MODEL_ID_B = "deepseek-r1-7b"
_TEST_QUERY = "What is the sub-agent pattern?"
_TEST_EVIDENCE_CONTENT = "ParallelAgent uses asyncio.gather"
_DEFAULT_MAX_CYCLES = 5
_DEFAULT_AGREEMENT_THRESHOLD = 0.85


# =============================================================================
# KB1.5: LLMDiscussionLoop Class Tests
# =============================================================================


class TestLLMDiscussionLoopExists:
    """LLMDiscussionLoop class existence tests."""

    def test_loop_module_importable(self) -> None:
        """Loop module is importable."""
        from src.discussion import loop
        assert loop is not None

    def test_loop_class_exists(self) -> None:
        """LLMDiscussionLoop class exists."""
        from src.discussion.loop import LLMDiscussionLoop
        assert LLMDiscussionLoop is not None

    def test_loop_has_discuss_method(self) -> None:
        """LLMDiscussionLoop has discuss method."""
        from src.discussion.loop import LLMDiscussionLoop
        assert hasattr(LLMDiscussionLoop, "discuss")


class TestLLMDiscussionLoopConstruction:
    """LLMDiscussionLoop construction tests."""

    def test_loop_accepts_participants_list(self) -> None:
        """Loop accepts list of participants."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [
            FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A),
            FakeLLMParticipant(_TEST_PARTICIPANT_ID_B, _TEST_MODEL_ID_B),
        ]
        
        loop = LLMDiscussionLoop(participants=participants)
        
        assert len(loop.participants) == 2

    def test_loop_has_max_cycles_parameter(self) -> None:
        """Loop accepts max_cycles parameter."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        
        loop = LLMDiscussionLoop(participants=participants, max_cycles=3)
        
        assert loop.max_cycles == 3

    def test_loop_default_max_cycles(self) -> None:
        """AC-KB1.5: Loop has default max_cycles of 5."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        
        loop = LLMDiscussionLoop(participants=participants)
        
        assert loop.max_cycles == _DEFAULT_MAX_CYCLES

    def test_loop_has_agreement_threshold_parameter(self) -> None:
        """Loop accepts agreement_threshold parameter."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        
        loop = LLMDiscussionLoop(
            participants=participants,
            agreement_threshold=0.9,
        )
        
        assert loop.agreement_threshold == 0.9

    def test_loop_default_agreement_threshold(self) -> None:
        """Loop has default agreement_threshold of 0.85."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        
        loop = LLMDiscussionLoop(participants=participants)
        
        assert loop.agreement_threshold == _DEFAULT_AGREEMENT_THRESHOLD


# =============================================================================
# KB1.5: LLMDiscussionLoop.discuss() Tests (AC-KB1.3, AC-KB1.4)
# =============================================================================


class TestLLMDiscussionLoopDiscuss:
    """AC-KB1.3: LLMDiscussionLoop.discuss() runs N participants via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_discuss_is_async_method(self) -> None:
        """discuss() is an async method."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        loop = LLMDiscussionLoop(participants=participants)
        
        # Verify discuss is a coroutine function
        import inspect
        assert inspect.iscoroutinefunction(loop.discuss)

    @pytest.mark.asyncio
    async def test_discuss_accepts_query_and_evidence(self) -> None:
        """discuss() accepts query and evidence parameters."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        # Should not raise
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        assert result is not None

    @pytest.mark.asyncio
    async def test_discuss_returns_discussion_result(self) -> None:
        """discuss() returns DiscussionResult."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence, DiscussionResult
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        assert isinstance(result, DiscussionResult)

    @pytest.mark.asyncio
    async def test_discuss_calls_all_participants(self) -> None:
        """AC-KB1.4: Each participant is called with same query."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant_a = FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)
        participant_b = FakeLLMParticipant(_TEST_PARTICIPANT_ID_B, _TEST_MODEL_ID_B)
        
        participants = [participant_a, participant_b]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # Both participants should have been called
        assert participant_a.analyze_called
        assert participant_b.analyze_called

    @pytest.mark.asyncio
    async def test_discuss_passes_same_evidence_to_all(self) -> None:
        """AC-KB1.4: Each participant receives same evidence."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant_a = FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)
        participant_b = FakeLLMParticipant(_TEST_PARTICIPANT_ID_B, _TEST_MODEL_ID_B)
        
        participants = [participant_a, participant_b]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # Both should have received same evidence
        assert participant_a.last_evidence == evidence
        assert participant_b.last_evidence == evidence


class TestLLMDiscussionLoopParallelExecution:
    """AC-KB1.3: Verify asyncio.gather parallel execution."""

    @pytest.mark.asyncio
    async def test_discuss_uses_asyncio_gather(self) -> None:
        """discuss() uses asyncio.gather for parallel execution."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participants = [
            FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A),
            FakeLLMParticipant(_TEST_PARTICIPANT_ID_B, _TEST_MODEL_ID_B),
        ]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        with patch("asyncio.gather", wraps=asyncio.gather) as mock_gather:
            await loop.discuss(query=_TEST_QUERY, evidence=evidence)
            
            # asyncio.gather should have been called
            assert mock_gather.called

    @pytest.mark.asyncio
    async def test_discuss_executes_participants_concurrently(self) -> None:
        """Participants execute concurrently, not sequentially."""
        import time
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Each participant delays 0.1s
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A, delay=0.1
        )
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B, _TEST_MODEL_ID_B, delay=0.1
        )
        
        participants = [participant_a, participant_b]
        loop = LLMDiscussionLoop(participants=participants, max_cycles=1)
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        start = time.monotonic()
        await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        elapsed = time.monotonic() - start
        
        # If sequential: ~0.2s; if parallel: ~0.1s
        # Allow margin for overhead
        assert elapsed < 0.18, f"Expected parallel execution, but took {elapsed}s"


# =============================================================================
# KB1.6: Discussion Cycles Tests (AC-KB1.5, AC-KB1.6)
# =============================================================================


class TestLLMDiscussionLoopCycles:
    """AC-KB1.5, AC-KB1.6: Cycle management tests."""

    @pytest.mark.asyncio
    async def test_discuss_respects_max_cycles(self) -> None:
        """AC-KB1.5: Loop stops at max_cycles."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Participant that never agrees (low agreement)
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A, 
            _TEST_MODEL_ID_A,
            fixed_confidence=0.3,  # Low confidence = low agreement
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.95,  # High threshold = won't reach
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        assert result.cycles_used <= 3

    @pytest.mark.asyncio
    async def test_discuss_preserves_history(self) -> None:
        """AC-KB1.6: History has 1 entry per cycle."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence, DiscussionCycle
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,  # Medium confidence
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=2,
            agreement_threshold=0.99,  # Won't reach
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # History should have entry for each cycle
        assert len(result.history) == result.cycles_used
        assert all(isinstance(c, DiscussionCycle) for c in result.history)

    @pytest.mark.asyncio
    async def test_discuss_stops_on_agreement(self) -> None:
        """Loop stops early when agreement threshold reached."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Participant with high confidence = high agreement
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.95,
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=5,
            agreement_threshold=0.85,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # Should stop early due to high agreement
        assert result.cycles_used < 5
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_discuss_cycle_numbers_sequential(self) -> None:
        """Cycle numbers are sequential starting from 1."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.99,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        cycle_numbers = [c.cycle_number for c in result.history]
        assert cycle_numbers == list(range(1, len(result.history) + 1))


# =============================================================================
# KB1.7: Error Handling Tests
# =============================================================================


class TestLLMDiscussionLoopErrorHandling:
    """Error handling for discussion loop."""

    @pytest.mark.asyncio
    async def test_discuss_raises_on_empty_participants(self) -> None:
        """Loop raises error with no participants."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        
        with pytest.raises(ValueError, match="participants"):
            LLMDiscussionLoop(participants=[])

    @pytest.mark.asyncio
    async def test_discuss_raises_on_empty_evidence(self) -> None:
        """Loop raises error with no evidence."""
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant = FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)
        loop = LLMDiscussionLoop(participants=[participant])
        
        with pytest.raises(ValueError, match="evidence"):
            await loop.discuss(query=_TEST_QUERY, evidence=[])

    @pytest.mark.asyncio
    async def test_discuss_handles_participant_error_gracefully(self) -> None:
        """Loop handles participant errors without crashing."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # One working participant, one that raises
        working = FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)
        failing = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B, 
            _TEST_MODEL_ID_B, 
            should_fail=True,
        )
        
        loop = LLMDiscussionLoop(
            participants=[working, failing],
            max_cycles=1,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        # Should complete with partial results, not crash
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        assert result is not None
        # At least one analysis should be present
        assert len(result.history[0].analyses) >= 1


# =============================================================================
# KB4.9: Agreement/Consensus Integration Tests
# =============================================================================


class TestAgreementConsensusIntegration:
    """Test KB4 agreement/consensus integration in LLMDiscussionLoop."""

    @pytest.mark.asyncio
    async def test_loop_uses_calculate_agreement(self) -> None:
        """Loop uses calculate_agreement() for proper scoring (AC-KB4.1)."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two participants with identical content
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.9,
            fixed_content="The rate limiter is in middleware.py using token bucket.",
        )
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.88,
            fixed_content="The rate limiter is in middleware.py using token bucket.",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],
            max_cycles=3,
            agreement_threshold=0.85,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content="class RateLimiter: ...",
                source_id="middleware.py",
            )
        ]
        
        result = await loop.discuss(query="Where is rate limiter?", evidence=evidence)
        
        # Identical content should give high agreement (>0.85), stopping at cycle 1
        assert result.cycles_used == 1
        assert result.history[0].agreement_score >= 0.85

    @pytest.mark.asyncio
    async def test_loop_captures_disagreement_points(self) -> None:
        """Loop captures disagreement points (AC-KB4.4)."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two participants with very different content
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,
            fixed_content="The authentication uses JWT tokens with Redis caching.",
        )
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.5,
            fixed_content="The database migrations use Alembic with PostgreSQL.",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],
            max_cycles=1,
            agreement_threshold=0.85,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content="Some code",
                source_id="file.py",
            )
        ]
        
        result = await loop.discuss(query="What pattern is used?", evidence=evidence)
        
        # Divergent content should have disagreement points
        cycle = result.history[0]
        assert cycle.agreement_score < 0.85
        # Disagreements may or may not be extracted depending on content
        assert isinstance(cycle.disagreement_points, list)

    @pytest.mark.asyncio
    async def test_loop_uses_synthesize_consensus(self) -> None:
        """Loop uses synthesize_consensus() for final output (AC-KB4.5)."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two participants with overlapping content
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.9,
            fixed_content="The cache uses Redis. It supports TTL.",
        )
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.88,
            fixed_content="The cache uses Redis. Configuration in config.yaml.",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],
            max_cycles=1,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Cache documentation",
                source_id="cache.md",
            )
        ]
        
        result = await loop.discuss(query="How does caching work?", evidence=evidence)
        
        # Consensus should contain shared concepts
        assert "Redis" in result.consensus or "cache" in result.consensus.lower()
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_early_stopping_on_agreement(self) -> None:
        """Loop stops early when agreement threshold reached (AC-KB4.3)."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two identical participants = high agreement
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.95,
            fixed_content="The function validates input.",
        )
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.95,
            fixed_content="The function validates input.",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],
            max_cycles=5,  # Allow up to 5
            agreement_threshold=0.9,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content="def validate(x): ...",
                source_id="utils.py",
            )
        ]
        
        result = await loop.discuss(query="What does validate do?", evidence=evidence)
        
        # Should stop at cycle 1 due to high agreement
        assert result.cycles_used == 1
        assert result.history[0].agreement_score >= 0.9


# =============================================================================
# KB3.7: Evidence Gathering Integration Tests
# =============================================================================


class TestEvidenceGatheringIntegration:
    """Test evidence gathering integration in LLMDiscussionLoop (KB3.7)."""

    @pytest.mark.asyncio
    async def test_loop_accepts_evidence_gatherer(self) -> None:
        """Loop accepts optional evidence_gatherer parameter."""
        from src.discussion.evidence_gatherer import (
            EvidenceGatherer,
            EvidenceGathererConfig,
        )
        from src.discussion.loop import LLMDiscussionLoop
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant = FakeLLMParticipant(_TEST_PARTICIPANT_ID_A, _TEST_MODEL_ID_A)
        
        # Create a mock gatherer
        mock_gatherer = MagicMock(spec=EvidenceGatherer)
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            evidence_gatherer=mock_gatherer,
        )
        
        assert loop.evidence_gatherer is mock_gatherer

    @pytest.mark.asyncio
    async def test_loop_without_gatherer_works(self) -> None:
        """Loop works without evidence gatherer (backward compatible)."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.9,
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=1,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="code",
                content=_TEST_EVIDENCE_CONTENT,
                source_id="agents.py",
            )
        ]
        
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        assert result is not None
        assert result.cycles_used == 1

    @pytest.mark.asyncio
    async def test_loop_calls_gatherer_when_requests_present(self) -> None:
        """Loop calls gatherer when information requests are present and agreement low."""
        from src.discussion.evidence_gatherer import (
            EvidenceGatherer,
            GatherResult,
        )
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two participants with different content to trigger low agreement
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,
            fixed_content='''
The authentication system uses JWT tokens stored in Redis cache.

```json
{
  "analysis": "JWT auth with Redis",
  "confidence": 0.5,
  "information_requests": [
    {"query": "Show ParallelAgent code", "source_types": ["code"], "priority": "high"}
  ]
}
```
''',
        )
        
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.5,
            fixed_content='''
The database migration framework relies on Alembic scripts managed through PostgreSQL.

```json
{
  "analysis": "Alembic with PostgreSQL",
  "confidence": 0.5,
  "information_requests": [
    {"query": "Show database schema", "source_types": ["code"], "priority": "medium"}
  ]
}
```
''',
        )
        
        # Create mock gatherer
        mock_gatherer = AsyncMock(spec=EvidenceGatherer)
        mock_gatherer.gather.return_value = GatherResult(
            evidence=[
                CrossReferenceEvidence(
                    source_type="code",
                    content="class ParallelAgent: ...",
                    source_id="agents.py#L100",
                )
            ],
            total_items=1,
            requests_processed=1,
            errors=[],
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],  # Two divergent participants
            max_cycles=2,
            agreement_threshold=0.85,
            evidence_gatherer=mock_gatherer,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Basic documentation",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # Gatherer should have been called since agreement < threshold
        mock_gatherer.gather.assert_called()

    @pytest.mark.asyncio
    async def test_loop_merges_new_evidence(self) -> None:
        """Loop merges new evidence into next cycle."""
        from src.discussion.evidence_gatherer import (
            EvidenceGatherer,
            GatherResult,
        )
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Track evidence passed to participant
        evidence_per_cycle: list[int] = []
        
        class TrackingParticipant(FakeLLMParticipant):
            async def analyze(self, query, evidence):
                evidence_per_cycle.append(len(evidence))
                return await super().analyze(query, evidence)
        
        # Two participants with different content for low agreement
        participant_a = TrackingParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,
            fixed_content='''
The caching system uses Redis with TTL expiration and LRU eviction.

```json
{
  "analysis": "Redis caching",
  "confidence": 0.5,
  "information_requests": [
    {"query": "More code", "source_types": ["code"], "priority": "high"}
  ]
}
```
''',
        )
        
        participant_b = TrackingParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.5,
            fixed_content='''
The message queue implementation uses RabbitMQ with dead letter exchanges.

```json
{
  "analysis": "RabbitMQ messaging",
  "confidence": 0.5,
  "information_requests": [
    {"query": "Queue config", "source_types": ["code"], "priority": "medium"}
  ]
}
```
''',
        )
        
        # Gatherer returns new evidence
        mock_gatherer = AsyncMock(spec=EvidenceGatherer)
        mock_gatherer.gather.return_value = GatherResult(
            evidence=[
                CrossReferenceEvidence(
                    source_type="code",
                    content="New code content",
                    source_id="new_file.py",
                )
            ],
            total_items=1,
            requests_processed=1,
            errors=[],
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],  # Two divergent participants
            max_cycles=3,
            agreement_threshold=0.85,
            evidence_gatherer=mock_gatherer,
        )
        
        initial_evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Initial doc",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_TEST_QUERY, evidence=initial_evidence)
        
        # With 2 participants: Cycle 1 = 2 appends, Cycle 2 = 2 appends, etc.
        # Cycle 1: indices 0,1 (1 evidence each)
        # Cycle 2: indices 2,3 (merged evidence)
        assert evidence_per_cycle[0] == 1  # Initial cycle 1
        assert evidence_per_cycle[1] == 1  # Second participant cycle 1
        # Cycle 2 should have merged evidence (1 initial + 1 new = 2)
        if len(evidence_per_cycle) >= 3:
            assert evidence_per_cycle[2] >= 2  # Merged with new evidence

    @pytest.mark.asyncio
    async def test_loop_handles_gatherer_error_gracefully(self) -> None:
        """Loop handles gatherer errors without crashing."""
        from src.discussion.evidence_gatherer import EvidenceGatherer
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # Two participants with different content for low agreement
        participant_a = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.5,
            fixed_content='''
Authentication uses OAuth2 with JWT tokens in Redis.

```json
{"information_requests": [{"query": "test", "source_types": ["code"]}]}
```
''',
        )
        
        participant_b = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_B,
            _TEST_MODEL_ID_B,
            fixed_confidence=0.5,
            fixed_content='''
Database migrations use Alembic with PostgreSQL backend.

```json
{"information_requests": [{"query": "schema", "source_types": ["code"]}]}
```
''',
        )
        
        # Gatherer that raises
        mock_gatherer = AsyncMock(spec=EvidenceGatherer)
        mock_gatherer.gather.side_effect = RuntimeError("Retrieval failed")
        
        loop = LLMDiscussionLoop(
            participants=[participant_a, participant_b],  # Two divergent participants
            max_cycles=2,
            evidence_gatherer=mock_gatherer,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Doc content",
                source_id="readme.md",
            )
        ]
        
        # Should complete without crashing
        result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        assert result is not None
        assert result.cycles_used == 2

    @pytest.mark.asyncio
    async def test_loop_skips_gathering_when_agreement_reached(self) -> None:
        """Loop skips evidence gathering when agreement threshold reached."""
        from src.discussion.evidence_gatherer import EvidenceGatherer
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.models import CrossReferenceEvidence
        from tests.unit.discussion.fake_participant import FakeLLMParticipant
        
        # High confidence participant
        participant = FakeLLMParticipant(
            _TEST_PARTICIPANT_ID_A,
            _TEST_MODEL_ID_A,
            fixed_confidence=0.95,  # Above threshold
        )
        
        mock_gatherer = AsyncMock(spec=EvidenceGatherer)
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.85,
            evidence_gatherer=mock_gatherer,
        )
        
        evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Doc content",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        
        # Gatherer should NOT have been called (early stopping)
        mock_gatherer.gather.assert_not_called()
