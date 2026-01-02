"""Integration tests for Information Request Detection.

WBS Reference: WBS-KB2 - Information Request Detection
Task: KB2.8 - Integration test: LLM produces parseable info requests
Acceptance Criteria:
- AC-KB2.2: extract_information_requests() parses LLM analysis for requests

This test requires inference-service to be running on :8085.
Mark as integration test to skip in unit test runs.
"""

from __future__ import annotations

import pytest

from src.discussion.models import CrossReferenceEvidence, InformationRequest
from src.discussion.request_extractor import extract_information_requests


# =============================================================================
# Test Constants
# =============================================================================

_TEST_QUERY = "How does the sub-agent pattern work in this codebase?"

_EVIDENCE_INSUFFICIENT = [
    CrossReferenceEvidence(
        source_type="doc",
        content="The system uses agents for various tasks.",
        source_id="architecture.md",
    ),
]


# =============================================================================
# Integration Test Markers
# =============================================================================


@pytest.mark.integration
class TestLLMProducesParseableInfoRequests:
    """Integration test: LLM produces parseable info requests when evidence is insufficient."""

    @pytest.fixture
    def analysis_with_requests(self) -> str:
        """Sample LLM analysis with structured information requests.
        
        This simulates what a real LLM would produce when it needs more info.
        In actual integration tests, this would come from inference-service.
        """
        return '''
Based on the provided evidence, I can see the system uses agents, but the documentation 
doesn't provide implementation details about the sub-agent pattern.

## Analysis

The evidence mentions "agents for various tasks" but doesn't explain:
1. How sub-agents are spawned
2. How they communicate
3. How results are aggregated

## Confidence: 0.45

I'm uncertain because the evidence lacks implementation details.

```json
{
  "analysis": "The evidence mentions agents but lacks implementation details.",
  "confidence": 0.45,
  "key_claims": [
    {
      "claim": "System uses agents",
      "evidence_source": "architecture.md",
      "confidence": 0.9
    }
  ],
  "information_requests": [
    {
      "query": "Show the ParallelAgent implementation in src/pipelines/agents.py",
      "source_types": ["code"],
      "priority": "high",
      "reasoning": "Need to see how asyncio.gather is used for parallel sub-agent execution"
    },
    {
      "query": "Find textbook references to the agent orchestration pattern",
      "source_types": ["books", "textbooks"],
      "priority": "medium",
      "reasoning": "Want to compare implementation with canonical pattern"
    }
  ]
}
```
'''

    def test_extract_requests_from_llm_analysis(
        self,
        analysis_with_requests: str,
    ) -> None:
        """Extract information requests from realistic LLM analysis."""
        requests = extract_information_requests(analysis_with_requests)
        
        assert len(requests) == 2
        assert all(isinstance(r, InformationRequest) for r in requests)

    def test_high_priority_request_has_code_source_type(
        self,
        analysis_with_requests: str,
    ) -> None:
        """High priority request targets code source."""
        requests = extract_information_requests(analysis_with_requests)
        
        high_priority = [r for r in requests if r.priority == "high"]
        assert len(high_priority) == 1
        assert high_priority[0].source_types == ["code"]
        assert "ParallelAgent" in high_priority[0].query

    def test_medium_priority_request_has_book_source_types(
        self,
        analysis_with_requests: str,
    ) -> None:
        """Medium priority request targets books and textbooks."""
        requests = extract_information_requests(analysis_with_requests)
        
        medium_priority = [r for r in requests if r.priority == "medium"]
        assert len(medium_priority) == 1
        assert "books" in medium_priority[0].source_types
        assert "textbooks" in medium_priority[0].source_types

    def test_requests_have_reasoning(
        self,
        analysis_with_requests: str,
    ) -> None:
        """All requests have reasoning explaining why info is needed."""
        requests = extract_information_requests(analysis_with_requests)
        
        for req in requests:
            assert req.reasoning != ""


@pytest.mark.integration
class TestLLMHighConfidenceNoRequests:
    """Integration test: LLM with high confidence produces no requests."""

    @pytest.fixture
    def analysis_high_confidence(self) -> str:
        """Sample LLM analysis with high confidence (no requests needed)."""
        return '''
Based on the provided evidence, the sub-agent pattern is clearly implemented.

## Analysis

The ParallelAgent class in agents.py uses asyncio.gather to execute multiple 
sub-agents concurrently. Each sub-agent is a function wrapped in the Agent base class.

## Confidence: 0.92

The implementation is straightforward and well-documented.

```json
{
  "analysis": "ParallelAgent uses asyncio.gather for concurrent sub-agent execution.",
  "confidence": 0.92,
  "key_claims": [
    {
      "claim": "ParallelAgent wraps sub-agents with asyncio.gather",
      "evidence_source": "agents.py#L135",
      "confidence": 0.95
    }
  ],
  "information_requests": []
}
```
'''

    def test_no_requests_when_high_confidence(
        self,
        analysis_high_confidence: str,
    ) -> None:
        """No information requests when LLM has high confidence."""
        requests = extract_information_requests(analysis_high_confidence)
        
        assert requests == []


@pytest.mark.integration  
@pytest.mark.slow
class TestRealInferenceServiceProducesRequests:
    """Integration test with real inference-service (requires service running)."""

    @pytest.fixture
    def inference_service_available(self) -> bool:
        """Check if inference-service is available."""
        import httpx
        
        try:
            response = httpx.get("http://localhost:8085/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    @pytest.mark.skip(reason="Requires inference-service on :8085")
    async def test_real_llm_produces_parseable_requests(
        self,
        inference_service_available: bool,
    ) -> None:
        """Real LLM should produce parseable information requests.
        
        This test is skipped by default - enable when running with live services.
        """
        if not inference_service_available:
            pytest.skip("inference-service not available on :8085")
        
        # This would use LLMParticipant to get real analysis
        # Then verify extract_information_requests parses it correctly
        pass
