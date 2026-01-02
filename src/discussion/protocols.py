"""Protocol definitions for LLM Discussion Loop participants.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Tasks: KB1.1, KB1.2 - Define LLMParticipant protocol
Acceptance Criteria:
- AC-KB1.1: LLMParticipant wraps inference-service with participant_id
- AC-KB1.4: Each participant receives same evidence, produces independent analysis

Pattern Reference: src/clients/protocols.py - Protocol duck typing pattern

Anti-Patterns Avoided:
- S1192: No duplicated literals
- Clean protocol with runtime_checkable for isinstance checks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.discussion.models import CrossReferenceEvidence, ParticipantAnalysis


@runtime_checkable
class LLMParticipantProtocol(Protocol):
    """Protocol for LLM discussion participants.
    
    This defines the duck-typed interface that any discussion participant
    must implement. Uses @runtime_checkable for isinstance() support.
    
    AC-KB1.1: Participant wraps inference-service with participant_id.
    AC-KB1.4: Each participant receives same evidence, produces independent analysis.
    
    Example:
        >>> class MyParticipant:
        ...     @property
        ...     def participant_id(self) -> str: ...
        ...     @property
        ...     def model_id(self) -> str: ...
        ...     async def analyze(self, query, evidence) -> ParticipantAnalysis: ...
        >>> 
        >>> isinstance(MyParticipant(), LLMParticipantProtocol)
        True
    """

    @property
    def participant_id(self) -> str:
        """Unique identifier for this participant."""
        ...

    @property
    def model_id(self) -> str:
        """Model identifier this participant uses for inference."""
        ...

    async def analyze(
        self,
        query: str,
        evidence: list[CrossReferenceEvidence],
    ) -> ParticipantAnalysis:
        """Produce an analysis given the query and evidence.
        
        Args:
            query: The question or topic to analyze.
            evidence: List of cross-reference evidence to consider.
            
        Returns:
            ParticipantAnalysis containing the participant's analysis.
        """
        ...
