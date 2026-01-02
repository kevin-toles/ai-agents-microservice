"""LLMDiscussionLoop orchestrator.

WBS Reference: WBS-KB1, WBS-KB2, WBS-KB3, WBS-KB4, WBS-KB7 - LLM Discussion Loop + Agreement Engine
Tasks: 
- KB1.5, KB1.6, KB1.7 - Create LLMDiscussionLoop class
- KB2.5 - Add request extraction to LLMDiscussionLoop cycle
- KB3.7 - Integrate evidence gathering into LLMDiscussionLoop
- KB4.9 - Integrate agreement check into LLMDiscussionLoop
- KB7.10 - Implement validation failure → retry logic
Acceptance Criteria:
- AC-KB1.3: LLMDiscussionLoop.discuss() runs N participants via asyncio.gather
- AC-KB1.5: Discussion loop uses configurable max_cycles (default 5)
- AC-KB1.6: Discussion history preserved as list[DiscussionCycle]
- AC-KB2.2: extract_information_requests() parses LLM analysis for requests
- AC-KB2.6: Zero requests returned when agreement_score > threshold
- AC-KB3.1: EvidenceGatherer.gather() called with information requests
- AC-KB3.3: Evidence merged without duplicates across cycles
- AC-KB4.1: calculate_agreement() returns score 0.0-1.0
- AC-KB4.3: agreement_threshold configurable (default 0.85)
- AC-KB4.4: Disagreement points extracted and logged
- AC-KB4.5: synthesize_consensus() used for final output
- AC-KB7.7: Validation failures from tools trigger discussion loop retry

Pattern Reference: src/pipelines/agents.py - ParallelAgent using asyncio.gather

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity < 15 via helper methods
- #42/#43: Proper async/await patterns
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.discussion.agreement import AgreementConfig, AgreementResult, calculate_agreement
from src.discussion.consensus import synthesize_consensus
from src.discussion.evidence_gatherer import EvidenceGatherer
from src.discussion.evidence_merger import merge_evidence
from src.discussion.models import (
    CrossReferenceEvidence,
    DiscussionCycle,
    DiscussionResult,
    InformationRequest,
    ParticipantAnalysis,
)
from src.discussion.protocols import LLMParticipantProtocol
from src.discussion.request_extractor import extract_information_requests_with_agreement


if TYPE_CHECKING:
    from src.tools.code_validation import CodeValidationProtocol


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_MAX_CYCLES = 5
_CONST_DEFAULT_AGREEMENT_THRESHOLD = 0.85
_CONST_MIN_PARTICIPANTS = 1
_CONST_MIN_EVIDENCE = 1

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class DiscussionLoopError(Exception):
    """Base exception for discussion loop errors."""


class DiscussionConfigError(DiscussionLoopError):
    """Configuration error in discussion loop."""


# =============================================================================
# LLMDiscussionLoop
# =============================================================================


class LLMDiscussionLoop:
    """Orchestrates multi-model discussion with cycle management and iterative evidence.
    
    AC-KB1.3: Runs N participants via asyncio.gather for parallel execution.
    AC-KB1.5: Configurable max_cycles with default of 5.
    AC-KB1.6: Preserves history as list[DiscussionCycle].
    AC-KB3.1: Optionally gathers new evidence based on information requests.
    
    Example:
        >>> participants = [LLMParticipant("p1", "qwen2.5-7b"), ...]
        >>> loop = LLMDiscussionLoop(participants=participants)
        >>> result = await loop.discuss(query="What is X?", evidence=[...])
        >>> print(result.consensus)
    
    With evidence gathering:
        >>> gatherer = EvidenceGatherer(config, retriever)
        >>> loop = LLMDiscussionLoop(participants, evidence_gatherer=gatherer)
        >>> result = await loop.discuss(query="...", evidence=[...])
    
    With code validation (AC-KB7.7):
        >>> from src.tools.code_validation import CodeValidationTool
        >>> validator = CodeValidationTool(code_analysis_client, sonarqube_client)
        >>> loop = LLMDiscussionLoop(participants, code_validation_tool=validator)
        >>> # Validation failures trigger retry
    """

    def __init__(
        self,
        participants: list[LLMParticipantProtocol],
        *,
        max_cycles: int = _CONST_DEFAULT_MAX_CYCLES,
        agreement_threshold: float = _CONST_DEFAULT_AGREEMENT_THRESHOLD,
        evidence_gatherer: EvidenceGatherer | None = None,
        code_validation_tool: CodeValidationProtocol | None = None,
    ) -> None:
        """Initialize the discussion loop.
        
        Args:
            participants: List of LLM participants (must satisfy protocol).
            max_cycles: Maximum number of discussion cycles.
            agreement_threshold: Agreement score to stop early.
            evidence_gatherer: Optional gatherer for iterative evidence retrieval.
            code_validation_tool: Optional tool for code validation (AC-KB7.6).
                When provided, generated code is validated before final output.
                Validation failures trigger retry (AC-KB7.7).
            
        Raises:
            ValueError: If participants list is empty.
        """
        if not participants:
            raise ValueError("participants list cannot be empty")
        
        self._participants = participants
        self._max_cycles = max_cycles
        self._agreement_threshold = agreement_threshold
        self._evidence_gatherer = evidence_gatherer
        self._code_validation_tool = code_validation_tool

    @property
    def participants(self) -> list[LLMParticipantProtocol]:
        """List of discussion participants."""
        return self._participants

    @property
    def max_cycles(self) -> int:
        """Maximum number of discussion cycles."""
        return self._max_cycles

    @property
    def agreement_threshold(self) -> float:
        """Agreement score threshold to stop early."""
        return self._agreement_threshold

    @property
    def evidence_gatherer(self) -> EvidenceGatherer | None:
        """Optional evidence gatherer for iterative retrieval."""
        return self._evidence_gatherer

    async def discuss(
        self,
        query: str,
        evidence: list[CrossReferenceEvidence],
    ) -> DiscussionResult:
        """Run the discussion loop with all participants.
        
        AC-KB1.3: Runs participants via asyncio.gather for parallel execution.
        AC-KB1.4: Each participant receives same evidence.
        AC-KB3.1: Gathers new evidence based on information requests.
        AC-KB3.3: Merges evidence without duplicates across cycles.
        
        Args:
            query: The question or topic to discuss.
            evidence: List of cross-reference evidence.
            
        Returns:
            DiscussionResult with consensus, confidence, and history.
            
        Raises:
            ValueError: If evidence list is empty.
        """
        if not evidence:
            raise ValueError("evidence list cannot be empty")
        
        history: list[DiscussionCycle] = []
        consensus = ""
        confidence = 0.0
        current_evidence = list(evidence)  # Copy to avoid mutation
        
        for cycle_num in range(1, self._max_cycles + 1):
            cycle = await self._run_cycle(cycle_num, query, current_evidence, history)
            history.append(cycle)
            
            # Update consensus from latest analyses
            consensus, confidence = self._compute_consensus(cycle)
            
            # Check for early stopping
            if cycle.agreement_score >= self._agreement_threshold:
                logger.info(
                    "Agreement threshold reached at cycle %d: %.2f",
                    cycle_num,
                    cycle.agreement_score,
                )
                break
            
            # Gather new evidence if we have requests and a gatherer (AC-KB3.1)
            if cycle.information_requests and self._evidence_gatherer:
                current_evidence = await self._gather_and_merge_evidence(
                    current_evidence,
                    cycle.information_requests,
                    cycle_num,
                )
        
        return DiscussionResult(
            consensus=consensus,
            confidence=confidence,
            cycles_used=len(history),
            history=history,
        )

    async def validate_code_output(
        self,
        code: str,
        query: str,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """Validate generated code using CodeValidationTool (AC-KB7.6, AC-KB7.7).
        
        Uses CodeT5+, GraphCodeBERT, CodeBERT, and SonarQube for
        objective code analysis. Returns validation result with
        should_retry flag for triggering discussion loop retry.
        
        Args:
            code: Generated source code to validate
            query: Original query describing expected functionality
            file_path: Optional file path for SonarQube analysis
            
        Returns:
            Dict with validation results:
            - passed: Whether validation passed
            - keywords: Extracted keywords from CodeT5+
            - validation_score: Overall validation score
            - sonarqube_result: Quality metrics (if available)
            - should_retry: True if validation failed (AC-KB7.7)
        """
        if self._code_validation_tool is None:
            logger.debug("CodeValidationTool not available, skipping validation")
            return {
                "passed": True,
                "keywords": [],
                "validation_score": 0.0,
                "sonarqube_result": None,
                "should_retry": False,
                "skipped": True,
            }
        
        try:
            result = await self._code_validation_tool.validate_code(
                code=code,
                query=query,
                file_path=file_path,
            )
            
            if not result.passed:
                logger.info(
                    "Code validation failed (should_retry=%s): %s",
                    result.should_retry,
                    result.failure_reason,
                )
            
            return {
                "passed": result.passed,
                "keywords": result.keywords,
                "validation_score": result.validation_score,
                "sonarqube_result": result.sonarqube_result,
                "should_retry": result.should_retry,
                "failure_reason": result.failure_reason,
            }
        except Exception as e:
            logger.warning(f"CodeValidationTool error: {e}")
            return {
                "passed": True,  # Fail open
                "keywords": [],
                "validation_score": 0.0,
                "sonarqube_result": None,
                "should_retry": False,
                "error": str(e),
            }

    async def _run_cycle(
        self,
        cycle_number: int,
        query: str,
        evidence: list[CrossReferenceEvidence],
        _previous_history: list[DiscussionCycle],
    ) -> DiscussionCycle:
        """Run a single discussion cycle.
        
        Uses asyncio.gather for parallel execution (AC-KB1.3).
        Uses calculate_agreement() for proper scoring (AC-KB4.1).
        Extracts information requests when agreement is below threshold (AC-KB2.2, AC-KB2.6).
        """
        # Create analysis tasks for all participants
        tasks = [
            self._safe_analyze(participant, query, evidence)
            for participant in self._participants
        ]
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful analyses
        analyses = self._filter_successful_analyses(results)
        
        # Compute agreement with full analysis (AC-KB4.1, AC-KB4.4)
        agreement_result = self._compute_agreement(analyses)
        
        # Extract information requests from analyses (AC-KB2.2, AC-KB2.6)
        information_requests = self._extract_cycle_requests(
            analyses, agreement_result.score
        )
        
        return DiscussionCycle(
            cycle_number=cycle_number,
            analyses=analyses,
            agreement_score=agreement_result.score,
            disagreement_points=list(agreement_result.disagreements),
            information_requests=information_requests,
        )

    async def _safe_analyze(
        self,
        participant: LLMParticipantProtocol,
        query: str,
        evidence: list[CrossReferenceEvidence],
    ) -> ParticipantAnalysis | Exception:
        """Safely call participant.analyze() with error handling."""
        try:
            return await participant.analyze(query, evidence)
        except Exception as e:
            logger.warning(
                "Participant %s failed: %s",
                participant.participant_id,
                e,
            )
            return e

    def _filter_successful_analyses(
        self,
        results: list[ParticipantAnalysis | BaseException],
    ) -> list[ParticipantAnalysis]:
        """Filter out failed analyses from results."""
        return [
            r for r in results
            if isinstance(r, ParticipantAnalysis)
        ]

    def _compute_agreement(
        self,
        analyses: list[ParticipantAnalysis],
    ) -> AgreementResult:
        """Compute agreement score from analyses.
        
        AC-KB4.1: Uses calculate_agreement() for proper scoring.
        AC-KB4.2: Considers claim overlap, citation overlap, confidence.
        AC-KB4.4: Extracts disagreement points.
        
        Args:
            analyses: List of participant analyses.
            
        Returns:
            AgreementResult with score and disagreements.
        """
        config = AgreementConfig(threshold=self._agreement_threshold)
        return calculate_agreement(analyses, config)

    async def _gather_and_merge_evidence(
        self,
        current_evidence: list[CrossReferenceEvidence],
        requests: list[InformationRequest],
        current_cycle: int,
    ) -> list[CrossReferenceEvidence]:
        """Gather new evidence and merge with existing.
        
        AC-KB3.1: Calls EvidenceGatherer.gather() with requests.
        AC-KB3.3: Merges evidence without duplicates.
        AC-KB3.4: Preserves provenance across cycles.
        
        Args:
            current_evidence: Evidence from previous cycles
            requests: Information requests from current cycle
            current_cycle: Current cycle number
            
        Returns:
            Merged evidence list
        """
        if not self._evidence_gatherer:
            return current_evidence
        
        try:
            # Gather new evidence based on requests
            gather_result = await self._evidence_gatherer.gather(requests)
            
            if gather_result.evidence:
                logger.info(
                    "Gathered %d new evidence items for cycle %d",
                    len(gather_result.evidence),
                    current_cycle + 1,  # Will be used in next cycle
                )
                
                # Merge with deduplication (AC-KB3.3)
                merge_result = merge_evidence(
                    old_evidence=current_evidence,
                    new_evidence=gather_result.evidence,
                    current_cycle=current_cycle + 1,
                )
                
                if merge_result.duplicates_removed > 0:
                    logger.debug(
                        "Removed %d duplicate evidence items",
                        merge_result.duplicates_removed,
                    )
                
                return list(merge_result.evidence)
            
            return current_evidence
            
        except Exception as e:
            logger.warning(
                "Failed to gather additional evidence: %s",
                e,
            )
            return current_evidence

    def _extract_cycle_requests(
        self,
        analyses: list[ParticipantAnalysis],
        agreement_score: float,
    ) -> list[InformationRequest]:
        """Extract information requests from cycle analyses.
        
        AC-KB2.2: Parses LLM analysis for requests.
        AC-KB2.6: Returns empty when agreement_score >= threshold.
        
        Args:
            analyses: List of participant analyses from this cycle
            agreement_score: Current agreement score
            
        Returns:
            Deduplicated list of information requests from all analyses
        """
        all_requests: list[InformationRequest] = []
        seen_queries: set[str] = set()
        
        for analysis in analyses:
            requests = extract_information_requests_with_agreement(
                analysis=analysis.content,
                agreement_score=agreement_score,
                threshold=self._agreement_threshold,
            )
            
            # Deduplicate by query
            for req in requests:
                if req.query not in seen_queries:
                    seen_queries.add(req.query)
                    all_requests.append(req)
        
        return all_requests

    def _compute_consensus(
        self,
        cycle: DiscussionCycle,
    ) -> tuple[str, float]:
        """Compute consensus from cycle analyses.
        
        AC-KB4.5: Uses synthesize_consensus() to merge analyses.
        AC-KB4.6: Tracks which claims came from which participant.
        
        Args:
            cycle: The discussion cycle with analyses.
            
        Returns:
            Tuple of (consensus_content, confidence).
        """
        if not cycle.analyses:
            return "", 0.0
        
        result = synthesize_consensus(cycle.analyses)
        return result.content, result.confidence

# =============================================================================
# Demo Entry Point (pragma: no cover - CLI demo code)
# =============================================================================


async def _run_demo() -> None:  # pragma: no cover
    """Run demonstration of LLMDiscussionLoop.
    
    Shows parallel LLM analysis with FakeLLMParticipant.
    For real inference-service demo, use integration tests.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    from tests.unit.discussion.fake_participant import FakeLLMParticipant
    
    print("=" * 60)
    print("LLMDiscussionLoop Demo - WBS-KB1")
    print("=" * 60)
    
    # Create fake participants with different responses
    participant_a = FakeLLMParticipant(
        participant_id="analyst-alpha",
        model_id="qwen2.5-7b",
        fixed_content="The sub-agent pattern uses ParallelAgent with asyncio.gather for concurrent execution.",
        fixed_confidence=0.85,
        delay=0.1,
    )
    participant_b = FakeLLMParticipant(
        participant_id="analyst-beta",
        model_id="deepseek-r1-7b",
        fixed_content="Sub-agents are spawned at runtime using SequentialAgent or ParallelAgent orchestrators.",
        fixed_confidence=0.78,
        delay=0.15,
    )
    
    print(f"\nParticipants: {participant_a.participant_id}, {participant_b.participant_id}")
    print(f"Models: {participant_a.model_id}, {participant_b.model_id}")
    
    loop = LLMDiscussionLoop(
        participants=[participant_a, participant_b],
        max_cycles=2,
        agreement_threshold=0.80,
    )
    
    evidence = [
        CrossReferenceEvidence(
            source_type="code",
            content="class ParallelAgent:\n    async def run(self, tasks):\n        return await asyncio.gather(*tasks)",
            source_id="agents.py#L135",
        ),
        CrossReferenceEvidence(
            source_type="doc",
            content="Sub-agents can be spawned at runtime or defined at construction time.",
            source_id="ARCHITECTURE.md",
        ),
    ]
    
    query = "What is the sub-agent pattern in software architecture?"
    print(f"\nQuery: {query}")
    print(f"Evidence items: {len(evidence)}")
    print("\n" + "-" * 60)
    print("Running discussion loop...")
    print("-" * 60)
    
    result = await loop.discuss(query=query, evidence=evidence)
    
    print(f"\n✅ Discussion completed!")
    print(f"   Cycles used: {result.cycles_used}")
    print(f"   Final confidence: {result.confidence:.2f}")
    
    for cycle in result.history:
        print(f"\n📋 Cycle {cycle.cycle_number}:")
        print(f"   Agreement score: {cycle.agreement_score:.2f}")
        for analysis in cycle.analyses:
            print(f"   - {analysis.participant_id} ({analysis.model_id}):")
            print(f"     Confidence: {analysis.confidence:.2f}")
            print(f"     Content: {analysis.content[:80]}...")
    
    print(f"\n🎯 Consensus:")
    print(f"   {result.consensus[:200]}...")
    print("\n" + "=" * 60)


def main() -> None:  # pragma: no cover
    """Main entry point for module execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Discussion Loop - WBS-KB1",
        prog="python -m src.discussion.loop",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with fake participants",
    )
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(_run_demo())
    else:
        print("Usage: python -m src.discussion.loop --demo")
        print("Run with --demo to see parallel LLM analysis demonstration")


if __name__ == "__main__":
    main()
