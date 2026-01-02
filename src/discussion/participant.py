"""LLMParticipant implementation wrapping inference-service.

WBS Reference: WBS-KB1 - LLM Discussion Loop Core
Tasks: KB1.1, KB1.2 - Implement LLMParticipant class
Acceptance Criteria:
- AC-KB1.1: LLMParticipant wraps inference-service with participant_id
- AC-KB1.4: Each participant receives same evidence, produces independent analysis

Pattern Reference: src/clients/inference_service.py - InferenceServiceClient

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity < 15
- #12: Connection pooling via shared client
- #7/#13: Namespaced exceptions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.discussion.models import CrossReferenceEvidence, ParticipantAnalysis

if TYPE_CHECKING:
    from src.clients.inference_service import InferenceServiceClient


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_CONFIDENCE = 0.75
_CONST_SYSTEM_PROMPT = """You are a participant in a multi-model discussion.
Analyze the provided evidence and question carefully.
Provide a clear, well-reasoned analysis based on the evidence.
Be specific and cite evidence when making claims."""


# =============================================================================
# Exceptions (namespaced to avoid #7/#13 exception shadowing)
# =============================================================================


class ParticipantError(Exception):
    """Base exception for participant errors."""


class ParticipantInferenceError(ParticipantError):
    """Error during inference call."""


# =============================================================================
# LLMParticipant
# =============================================================================


class LLMParticipant:
    """LLM participant that wraps inference-service.
    
    AC-KB1.1: Wraps inference-service with participant_id.
    
    Attributes:
        participant_id: Unique identifier for this participant.
        model_id: Model identifier used for inference.
    """

    def __init__(
        self,
        participant_id: str,
        model_id: str,
        inference_client: InferenceServiceClient | None = None,
    ) -> None:
        """Initialize the LLM participant.
        
        Args:
            participant_id: Unique identifier for this participant.
            model_id: Model identifier to use for inference.
            inference_client: Optional inference client (for testing).
        """
        self._participant_id = participant_id
        self._model_id = model_id
        self._client = inference_client

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
        """Produce an analysis given query and evidence.
        
        AC-KB1.4: Receives evidence and produces independent analysis.
        
        Args:
            query: The question or topic to analyze.
            evidence: List of cross-reference evidence to consider.
            
        Returns:
            ParticipantAnalysis with content and confidence.
            
        Raises:
            ParticipantInferenceError: If inference call fails.
        """
        # Format evidence for prompt
        evidence_text = self._format_evidence(evidence)
        
        # Build prompt
        prompt = self._build_prompt(query, evidence_text)
        
        # Call inference service if client available
        if self._client is not None:
            try:
                response = await self._call_inference(prompt)
                content = response.get("content", "")
                confidence = self._extract_confidence(response)
            except Exception as e:
                raise ParticipantInferenceError(
                    f"Inference failed for {self._participant_id}: {e}"
                ) from e
        else:
            # Fallback for testing without client
            content = f"Analysis of '{query}' based on {len(evidence)} evidence items."
            confidence = _CONST_DEFAULT_CONFIDENCE
        
        return ParticipantAnalysis(
            participant_id=self._participant_id,
            model_id=self._model_id,
            content=content,
            confidence=confidence,
        )

    def _format_evidence(self, evidence: list[CrossReferenceEvidence]) -> str:
        """Format evidence list into prompt text."""
        if not evidence:
            return "No evidence provided."
        
        parts = []
        for i, ev in enumerate(evidence, 1):
            parts.append(f"[{i}] ({ev.source_type}) {ev.source_id}:\n{ev.content}")
        
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, evidence_text: str) -> str:
        """Build the full prompt for inference."""
        return f"""Question: {query}

Evidence:
{evidence_text}

Provide your analysis based on the evidence above."""

    async def _call_inference(self, prompt: str) -> dict[str, Any]:
        """Call the inference service.
        
        Args:
            prompt: The prompt to send.
            
        Returns:
            Response dictionary from inference service.
        """
        if self._client is None:
            return {"content": "", "confidence": _CONST_DEFAULT_CONFIDENCE}
        
        # Use the inference client's complete method
        response = await self._client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_id,
            system_prompt=_CONST_SYSTEM_PROMPT,
        )
        
        return {"content": response, "confidence": _CONST_DEFAULT_CONFIDENCE}

    def _extract_confidence(self, response: dict[str, Any]) -> float:
        """Extract confidence score from response."""
        return response.get("confidence", _CONST_DEFAULT_CONFIDENCE)
