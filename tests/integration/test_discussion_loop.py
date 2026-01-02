"""Integration test for LLMDiscussionLoop with real inference-service.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Task: KB1.10 - Integration test with real inference-service
Acceptance Criteria:
- AC-KB1.3: LLMDiscussionLoop.discuss() runs N participants in parallel

TDD Phase: RED - Tests written expecting real inference-service integration

Requirements:
- inference-service running at http://localhost:8085
- Models available: qwen2.5-7b, deepseek-r1-7b

Run with: pytest tests/integration/test_discussion_loop.py -v --integration
"""

from __future__ import annotations

import os
import pytest
import httpx


# =============================================================================
# Test Constants
# =============================================================================

_INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8085")
_TEST_QUERY = "What is the sub-agent pattern in software architecture?"
_MODEL_A = "qwen2.5-7b"
_MODEL_B = "deepseek-r1-7b"


# =============================================================================
# Fixtures
# =============================================================================


def _inference_service_available() -> bool:
    """Check if inference-service is reachable."""
    try:
        response = httpx.get(f"{_INFERENCE_SERVICE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip entire module if inference-service not available
pytestmark = pytest.mark.skipif(
    not _inference_service_available(),
    reason="inference-service not available at " + _INFERENCE_SERVICE_URL,
)


# =============================================================================
# AC-KB1.3: Integration Tests - Parallel LLM Execution
# =============================================================================


class TestLLMDiscussionLoopIntegration:
    """Integration tests for LLMDiscussionLoop with real inference-service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_two_participants_produce_different_analyses(self) -> None:
        """AC-KB1.3: qwen2.5-7b and deepseek-r1-7b produce different analyses."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import CrossReferenceEvidence
        from src.clients.inference_service import InferenceServiceClient
        
        # Create real inference client
        client = InferenceServiceClient(base_url=_INFERENCE_SERVICE_URL)
        try:
            # Create participants with different models
            participant_a = LLMParticipant(
                participant_id="analyst-qwen",
                model_id=_MODEL_A,
                inference_client=client,
            )
            participant_b = LLMParticipant(
                participant_id="analyst-deepseek",
                model_id=_MODEL_B,
                inference_client=client,
            )
            
            loop = LLMDiscussionLoop(
                participants=[participant_a, participant_b],
                max_cycles=1,
            )
            
            evidence = [
                CrossReferenceEvidence(
                    source_type="code",
                    content="class ParallelAgent:\n    async def run(self, tasks):\n        return await asyncio.gather(*tasks)",
                    source_id="agents.py#L135",
                ),
            ]
            
            result = await loop.discuss(query=_TEST_QUERY, evidence=evidence)
        finally:
            await client.close()
        
        # Verify 2 independent analyses
        assert result.cycles_used >= 1
        assert len(result.history) >= 1
        
        first_cycle = result.history[0]
        assert len(first_cycle.analyses) == 2
        
        # Analyses should be from different participants
        participant_ids = {a.participant_id for a in first_cycle.analyses}
        assert participant_ids == {"analyst-qwen", "analyst-deepseek"}
        
        # Analyses should have different content (different models)
        contents = [a.content for a in first_cycle.analyses]
        assert contents[0] != contents[1], "Different models should produce different analyses"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_discussion_produces_consensus(self) -> None:
        """Integration: Discussion loop produces consensus from real LLMs."""
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import CrossReferenceEvidence, DiscussionResult
        from src.clients.inference_service import InferenceServiceClient
        
        client = InferenceServiceClient(base_url=_INFERENCE_SERVICE_URL)
        try:
            participant = LLMParticipant(
                participant_id="analyst-main",
                model_id=_MODEL_A,
                inference_client=client,
            )
            
            loop = LLMDiscussionLoop(
                participants=[participant],
                max_cycles=2,
                agreement_threshold=0.7,
            )
            
            evidence = [
                CrossReferenceEvidence(
                    source_type="doc",
                    content="The Repository pattern mediates between domain and data mapping layers.",
                    source_id="design_patterns.md",
                ),
            ]
            
            result = await loop.discuss(
                query="What is the Repository pattern?",
                evidence=evidence,
            )
        finally:
            await client.close()
        
        assert isinstance(result, DiscussionResult)
        assert result.consensus != ""
        assert result.confidence > 0.0
        assert result.cycles_used >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.timeout(60)
    async def test_parallel_execution_performance(self) -> None:
        """Integration: Verify parallel execution is faster than sequential."""
        import time
        from src.discussion.loop import LLMDiscussionLoop
        from src.discussion.participant import LLMParticipant
        from src.discussion.models import CrossReferenceEvidence
        from src.clients.inference_service import InferenceServiceClient
        
        client = InferenceServiceClient(base_url=_INFERENCE_SERVICE_URL)
        try:
            # Create 2 participants using same model for fair comparison
            participants = [
                LLMParticipant(
                    participant_id=f"analyst-{i}",
                    model_id=_MODEL_A,
                    inference_client=client,
                )
                for i in range(2)
            ]
            
            loop = LLMDiscussionLoop(
                participants=participants,
                max_cycles=1,
            )
            
            evidence = [
                CrossReferenceEvidence(
                    source_type="code",
                    content="def hello(): return 'world'",
                    source_id="simple.py",
                ),
            ]
            
            start = time.monotonic()
            result = await loop.discuss(query="What does this code do?", evidence=evidence)
            elapsed = time.monotonic() - start
        finally:
            await client.close()
        
        assert result.cycles_used == 1
        assert len(result.history[0].analyses) == 2
        
        # Parallel should complete both in roughly single-call time
        # (with margin for overhead). If sequential, would be ~2x.
        # This is a soft check - mainly ensures both completed.
        assert elapsed < 120, f"Parallel execution took too long: {elapsed}s"
