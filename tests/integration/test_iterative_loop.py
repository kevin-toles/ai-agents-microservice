"""Integration tests for Iterative Evidence Gathering Loop.

WBS Reference: WBS-KB3 - Iterative Evidence Gathering
Task: KB3.9 - Integration test: loop retrieves new evidence and continues
Acceptance Criteria:
- AC-KB3.2: Gatherer calls UnifiedRetriever with request queries

Exit Criteria:
- Integration test shows: Cycle 1 → info request → Cycle 2 with new evidence

This test demonstrates the full iterative loop:
1. Initial discussion cycle with insufficient evidence
2. LLMs generate information requests
3. EvidenceGatherer retrieves new evidence
4. Cycle 2 runs with merged evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.discussion.evidence_gatherer import (
    EvidenceGatherer,
    EvidenceGathererConfig,
    GatherResult,
)
from src.discussion.loop import LLMDiscussionLoop
from src.discussion.models import CrossReferenceEvidence, InformationRequest
from src.schemas.retrieval_models import (
    RetrievalItem,
    RetrievalResult,
    RetrievalScope,
    SourceType,
)


# =============================================================================
# Test Constants
# =============================================================================

_INITIAL_QUERY = "How does the sub-agent pattern work in this codebase?"
_CODE_QUERY = "Show ParallelAgent implementation"


# =============================================================================
# Fake Retriever for Integration Testing
# =============================================================================


@dataclass
class FakeUnifiedRetriever:
    """Fake retriever that simulates real retrieval behavior.
    
    Returns different results based on query, simulating how real
    UnifiedRetriever would respond to information requests.
    """
    
    call_log: list[dict[str, Any]] = field(default_factory=list)
    
    async def retrieve(
        self,
        query: str,
        scope: RetrievalScope = RetrievalScope.ALL,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Simulate retrieval based on query."""
        self.call_log.append({"query": query, "scope": scope})
        
        # Simulate finding code for ParallelAgent query
        if "ParallelAgent" in query or "parallel" in query.lower():
            return RetrievalResult(
                query=query,
                results=[
                    RetrievalItem(
                        source_type=SourceType.CODE,
                        source_id="src/pipelines/agents.py#L135-L180",
                        content="""class ParallelAgent:
    \"\"\"Agent that executes sub-agents in parallel using asyncio.gather.\"\"\"
    
    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents
    
    async def run(self, state: AgentState) -> AgentState:
        tasks = [agent.run(state) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(results, state)
""",
                        relevance_score=0.95,
                        title="ParallelAgent class",
                        metadata={"language": "python", "lines": "135-180"},
                    )
                ],
                total_count=1,
                citations=[],
                errors=[],
                scope=scope,
            )
        
        # Default: empty results
        return RetrievalResult(
            query=query,
            results=[],
            total_count=0,
            citations=[],
            errors=[],
            scope=scope,
        )


# =============================================================================
# Fake Participant for Integration Testing
# =============================================================================


class CycleAwareParticipant:
    """Participant that changes behavior based on evidence provided.
    
    - With minimal evidence: produces low confidence + info requests
    - With code evidence: produces high confidence answer
    """
    
    def __init__(self, participant_id: str, model_id: str) -> None:
        self._participant_id = participant_id
        self._model_id = model_id
        self._call_count = 0
    
    @property
    def participant_id(self) -> str:
        return self._participant_id
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    async def analyze(
        self,
        query: str,
        evidence: list[CrossReferenceEvidence],
    ):
        """Analyze query with evidence.
        
        Returns different responses based on evidence quality.
        """
        from src.discussion.models import ParticipantAnalysis
        
        self._call_count += 1
        
        # Check if we have code evidence about ParallelAgent
        has_code = any(
            "ParallelAgent" in e.content or "asyncio.gather" in e.content
            for e in evidence
        )
        
        if has_code:
            # High confidence response with code evidence
            return ParticipantAnalysis(
                participant_id=self._participant_id,
                model_id=self._model_id,
                content="""
Based on the code evidence, the sub-agent pattern is implemented using ParallelAgent.

The ParallelAgent class wraps multiple sub-agents and executes them concurrently 
using `asyncio.gather`. This allows for parallel execution of independent tasks.

## Confidence: 0.92

The implementation is clearly visible in agents.py.

```json
{
  "analysis": "ParallelAgent uses asyncio.gather for concurrent sub-agent execution",
  "confidence": 0.92,
  "key_claims": [
    {"claim": "ParallelAgent executes sub-agents in parallel", "evidence_source": "agents.py"}
  ],
  "information_requests": []
}
```
""",
                confidence=0.92,
            )
        else:
            # Low confidence, need more info
            return ParticipantAnalysis(
                participant_id=self._participant_id,
                model_id=self._model_id,
                content="""
The evidence mentions agents but doesn't show implementation details.

## Confidence: 0.45

I need to see the actual code to understand how sub-agents work.

```json
{
  "analysis": "Evidence insufficient - need code",
  "confidence": 0.45,
  "key_claims": [],
  "information_requests": [
    {
      "query": "Show ParallelAgent implementation in src/pipelines/agents.py",
      "source_types": ["code"],
      "priority": "high",
      "reasoning": "Need to see asyncio.gather usage for sub-agent execution"
    }
  ]
}
```
""",
                confidence=0.45,
            )


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestIterativeEvidenceGatheringLoop:
    """Integration test: loop retrieves new evidence and continues."""

    @pytest.mark.asyncio
    async def test_cycle1_generates_info_request_cycle2_has_new_evidence(
        self,
    ) -> None:
        """Exit Criteria: Cycle 1 → info request → Cycle 2 with new evidence.
        
        This test demonstrates the full iterative loop:
        1. Cycle 1: Participant has low confidence, requests ParallelAgent code
        2. EvidenceGatherer retrieves the code
        3. Cycle 2: Participant has high confidence with code evidence
        """
        # Setup fake retriever
        fake_retriever = FakeUnifiedRetriever()
        
        # Setup evidence gatherer
        config = EvidenceGathererConfig(max_results_per_request=5)
        gatherer = EvidenceGatherer(config=config, retriever=fake_retriever)
        
        # Setup participant
        participant = CycleAwareParticipant(
            participant_id="analyst-alpha",
            model_id="test-model",
        )
        
        # Setup loop with gatherer
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.85,
            evidence_gatherer=gatherer,
        )
        
        # Initial evidence - just documentation, no code
        initial_evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="The system uses agents for various orchestration tasks.",
                source_id="architecture.md",
            )
        ]
        
        # Run the discussion
        result = await loop.discuss(query=_INITIAL_QUERY, evidence=initial_evidence)
        
        # Verify iterative behavior
        assert result.cycles_used >= 2, "Should have run at least 2 cycles"
        
        # Cycle 1 should have info request
        cycle1 = result.history[0]
        assert cycle1.agreement_score < 0.85, "Cycle 1 should have low agreement"
        assert len(cycle1.information_requests) > 0, "Cycle 1 should have info requests"
        
        # Verify retriever was called
        assert len(fake_retriever.call_log) > 0, "Retriever should have been called"
        assert any(
            "ParallelAgent" in call["query"]
            for call in fake_retriever.call_log
        ), "Should have queried for ParallelAgent"
        
        # Final result should have high confidence
        assert result.confidence >= 0.85, "Final confidence should be high"

    @pytest.mark.asyncio
    async def test_evidence_count_increases_across_cycles(self) -> None:
        """Evidence count should increase as new evidence is gathered."""
        evidence_counts: list[int] = []
        
        class TrackingParticipant(CycleAwareParticipant):
            async def analyze(self, query, evidence):
                evidence_counts.append(len(evidence))
                return await super().analyze(query, evidence)
        
        fake_retriever = FakeUnifiedRetriever()
        config = EvidenceGathererConfig()
        gatherer = EvidenceGatherer(config=config, retriever=fake_retriever)
        
        participant = TrackingParticipant(
            participant_id="analyst",
            model_id="test-model",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.85,
            evidence_gatherer=gatherer,
        )
        
        initial_evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Basic docs",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_INITIAL_QUERY, evidence=initial_evidence)
        
        # Evidence should increase
        assert evidence_counts[0] == 1, "Cycle 1 has initial evidence"
        assert evidence_counts[1] > evidence_counts[0], "Cycle 2 has more evidence"

    @pytest.mark.asyncio
    async def test_gatherer_receives_correct_requests(self) -> None:
        """EvidenceGatherer receives information requests from cycle."""
        fake_retriever = FakeUnifiedRetriever()
        config = EvidenceGathererConfig()
        gatherer = EvidenceGatherer(config=config, retriever=fake_retriever)
        
        participant = CycleAwareParticipant(
            participant_id="analyst",
            model_id="test-model",
        )
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=2,
            agreement_threshold=0.85,
            evidence_gatherer=gatherer,
        )
        
        initial_evidence = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Basic docs",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_INITIAL_QUERY, evidence=initial_evidence)
        
        # Check retriever received the query
        assert len(fake_retriever.call_log) > 0
        
        # Should have queried for code (from info request)
        code_queries = [
            call for call in fake_retriever.call_log
            if call["scope"] == RetrievalScope.CODE_ONLY
        ]
        assert len(code_queries) > 0, "Should have code-scoped query"


@pytest.mark.integration
class TestEvidenceMerging:
    """Test evidence merging across cycles."""

    @pytest.mark.asyncio
    async def test_no_duplicate_evidence_across_cycles(self) -> None:
        """Merged evidence should not have duplicates."""
        # Retriever that returns same item twice
        class DuplicatingRetriever:
            async def retrieve(
                self,
                query: str,
                scope: RetrievalScope = RetrievalScope.ALL,
                top_k: int = 10,
            ) -> RetrievalResult:
                return RetrievalResult(
                    query=query,
                    results=[
                        RetrievalItem(
                            source_type=SourceType.CODE,
                            source_id="same_file.py",  # Same ID
                            content="Same content",
                            relevance_score=0.9,
                        )
                    ],
                    total_count=1,
                    citations=[],
                    errors=[],
                    scope=scope,
                )
        
        evidence_per_cycle: list[list[CrossReferenceEvidence]] = []
        
        class TrackingParticipant(CycleAwareParticipant):
            async def analyze(self, query, evidence):
                evidence_per_cycle.append(list(evidence))
                # Always return low confidence to keep iterating
                from src.discussion.models import ParticipantAnalysis
                return ParticipantAnalysis(
                    participant_id=self.participant_id,
                    model_id=self.model_id,
                    content='```json\n{"information_requests": [{"query": "more", "source_types": ["code"]}]}\n```',
                    confidence=0.4,
                )
        
        gatherer = EvidenceGatherer(
            config=EvidenceGathererConfig(),
            retriever=DuplicatingRetriever(),
        )
        
        participant = TrackingParticipant("analyst", "test")
        
        loop = LLMDiscussionLoop(
            participants=[participant],
            max_cycles=3,
            agreement_threshold=0.85,
            evidence_gatherer=gatherer,
        )
        
        initial = [
            CrossReferenceEvidence(
                source_type="doc",
                content="Initial",
                source_id="readme.md",
            )
        ]
        
        await loop.discuss(query=_INITIAL_QUERY, evidence=initial)
        
        # Check no duplicates by source_id
        for cycle_evidence in evidence_per_cycle:
            source_ids = [e.source_id for e in cycle_evidence]
            assert len(source_ids) == len(set(source_ids)), "No duplicate source_ids"
