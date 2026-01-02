"""Discussion module for LLM Discussion Loop.

WBS Reference: WBS-KB1, WBS-KB2, WBS-KB3, WBS-KB4 - LLM Discussion Loop + Evidence + Agreement
Purpose: Multi-model deliberation with cross-reference evidence and iterative retrieval

This module provides:
- LLMParticipantProtocol: Duck-typed interface for discussion participants
- LLMParticipant: Concrete participant wrapping inference-service
- LLMDiscussionLoop: Orchestrator running N participants via asyncio.gather
- Data models for cycles, results, evidence, and information requests
- Information request extraction from LLM analyses
- EvidenceGatherer: Iterative evidence retrieval based on information requests
- EvidenceMerger: Merges evidence across cycles with deduplication and provenance
- AgreementEngine: Agreement calculation and disagreement extraction (WBS-KB4)
- ConsensusEngine: Consensus synthesis from agreeing analyses (WBS-KB4)
"""

from src.discussion.agreement import (
    AgreementConfig,
    AgreementResult,
    calculate_agreement,
    calculate_citation_overlap,
    calculate_claim_overlap,
    calculate_confidence_score,
    extract_disagreements,
)
from src.discussion.consensus import (
    ConsensusConfig,
    ConsensusResult,
    extract_claims,
    synthesize_consensus,
)
from src.discussion.evidence_gatherer import (
    EvidenceGatherer,
    EvidenceGathererConfig,
    GatherResult,
)
from src.discussion.evidence_merger import (
    EvidenceMerger,
    MergeResult,
    merge_evidence,
)
from src.discussion.loop import LLMDiscussionLoop
from src.discussion.models import (
    CrossReferenceEvidence,
    DiscussionCycle,
    DiscussionResult,
    InformationRequest,
    ParticipantAnalysis,
)
from src.discussion.participant import LLMParticipant
from src.discussion.protocols import LLMParticipantProtocol
from src.discussion.provenance import (
    ProvenanceConfig,
    ProvenanceEntry,
    ProvenanceTracker,
)
from src.discussion.audit_validator import (
    AuditServiceValidator,
    ValidationConfig,
    ValidationResult,
)
from src.discussion.request_extractor import (
    calculate_priority_from_disagreement,
    extract_information_requests,
    extract_information_requests_with_agreement,
)

__all__ = [
    "AgreementConfig",
    "AgreementResult",
    "AuditServiceValidator",
    "ConsensusConfig",
    "ConsensusResult",
    "CrossReferenceEvidence",
    "DiscussionCycle",
    "DiscussionResult",
    "EvidenceGatherer",
    "EvidenceGathererConfig",
    "EvidenceMerger",
    "GatherResult",
    "InformationRequest",
    "LLMDiscussionLoop",
    "LLMParticipant",
    "LLMParticipantProtocol",
    "MergeResult",
    "ParticipantAnalysis",
    "ProvenanceConfig",
    "ProvenanceEntry",
    "ProvenanceTracker",
    "ValidationConfig",
    "ValidationResult",
    "calculate_agreement",
    "calculate_citation_overlap",
    "calculate_claim_overlap",
    "calculate_confidence_score",
    "calculate_priority_from_disagreement",
    "extract_claims",
    "extract_disagreements",
    "extract_information_requests",
    "extract_information_requests_with_agreement",
    "merge_evidence",
    "synthesize_consensus",
]
