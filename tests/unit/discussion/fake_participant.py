"""Fake LLMParticipant for testing.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Purpose: Test double that satisfies LLMParticipantProtocol for unit testing

This fake allows:
- Tracking if analyze() was called
- Inspecting what query/evidence were passed
- Configurable delays for parallel execution tests
- Configurable failures for error handling tests
- Configurable confidence for agreement tests

Anti-Patterns Avoided:
- S1192: Constants at module level
- Proper async context management
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.discussion.models import CrossReferenceEvidence, ParticipantAnalysis


# =============================================================================
# Test Constants
# =============================================================================

_DEFAULT_ANALYSIS_CONTENT = "Fake analysis response"
_DEFAULT_CONFIDENCE = 0.75


class FakeLLMParticipant:
    """Test double for LLMParticipant.
    
    Satisfies LLMParticipantProtocol via duck typing.
    """

    def __init__(
        self,
        participant_id: str,
        model_id: str,
        *,
        delay: float = 0.0,
        should_fail: bool = False,
        fixed_confidence: float = _DEFAULT_CONFIDENCE,
        fixed_content: str = _DEFAULT_ANALYSIS_CONTENT,
    ) -> None:
        """Initialize fake participant.
        
        Args:
            participant_id: Unique identifier for this participant.
            model_id: Model identifier this participant represents.
            delay: Simulated delay in seconds for async testing.
            should_fail: If True, analyze() raises an exception.
            fixed_confidence: Confidence value to return.
            fixed_content: Content to return in analysis.
        """
        self._participant_id = participant_id
        self._model_id = model_id
        self._delay = delay
        self._should_fail = should_fail
        self._fixed_confidence = fixed_confidence
        self._fixed_content = fixed_content
        
        # Tracking for test assertions
        self.analyze_called = False
        self.last_query: str | None = None
        self.last_evidence: list[CrossReferenceEvidence] | None = None
        self.call_count = 0

    @property
    def participant_id(self) -> str:
        """Unique identifier for this participant."""
        return self._participant_id

    @property
    def model_id(self) -> str:
        """Model identifier this participant uses."""
        return self._model_id

    async def analyze(
        self,
        query: str,
        evidence: list[CrossReferenceEvidence],
    ) -> ParticipantAnalysis:
        """Produce analysis given query and evidence.
        
        Args:
            query: The question to analyze.
            evidence: Cross-reference evidence to consider.
            
        Returns:
            ParticipantAnalysis with fake content and configured confidence.
            
        Raises:
            RuntimeError: If should_fail is True.
        """
        # Import here to avoid circular imports
        from src.discussion.models import ParticipantAnalysis
        
        # Track the call
        self.analyze_called = True
        self.last_query = query
        self.last_evidence = evidence
        self.call_count += 1
        
        # Simulate delay if configured
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        
        # Simulate failure if configured
        if self._should_fail:
            raise RuntimeError(f"Fake failure for {self._participant_id}")
        
        return ParticipantAnalysis(
            participant_id=self._participant_id,
            model_id=self._model_id,
            content=self._fixed_content,
            confidence=self._fixed_confidence,
        )

    def reset(self) -> None:
        """Reset tracking state for reuse in tests."""
        self.analyze_called = False
        self.last_query = None
        self.last_evidence = None
        self.call_count = 0
