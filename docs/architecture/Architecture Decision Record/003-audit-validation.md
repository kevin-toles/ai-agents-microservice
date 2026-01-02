# ADR-003: Audit and Validation Pipeline

**Status**: Accepted  
**Date**: 2025-01-15  
**WBS Reference**: WBS-KB9 - End-to-End Validation  
**Decision Makers**: Architecture Team

## Context

AI-generated responses can contain **hallucinations** - plausible-sounding but factually incorrect information. In a code assistant context, this is particularly dangerous:

- Incorrect file paths lead to wasted debugging time
- Non-existent function signatures cause compilation errors
- Fabricated design patterns introduce technical debt
- Misattributed quotes undermine trust

The Kitchen Brigade AI Agent must provide **verifiable, auditable responses** where every claim can be traced to its source. We need a validation pipeline that:

1. Detects unsupported claims before they reach the user
2. Validates citations against actual source content
3. Flags low-confidence responses for human review
4. Maintains an audit trail for compliance

## Decision

We will implement an **Audit and Validation Pipeline** as a mandatory post-processing step before any response is returned to the user.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Audit and Validation Flow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Generated Response                                            │
│           │                                                      │
│           ▼                                                      │
│    ┌──────────────┐                                             │
│    │   Citation   │──► Extract all claims with citations        │
│    │   Extractor  │                                             │
│    └──────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│    ┌──────────────┐                                             │
│    │   Source     │──► Verify each citation exists              │
│    │   Validator  │    and content matches claim                │
│    └──────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│    ┌──────────────┐                                             │
│    │ Hallucination│──► Detect unsupported claims                │
│    │   Detector   │    using semantic similarity                │
│    └──────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│    ┌──────────────┐                                             │
│    │   Audit      │──► Log all validations for                  │
│    │   Logger     │    compliance and debugging                 │
│    └──────┬───────┘                                             │
│           │                                                      │
│           ▼                                                      │
│    [Pass / Reject / Flag for Review]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Validation Rules

#### Citation Validation

| Rule | Description | Action on Failure |
|------|-------------|-------------------|
| Source Exists | File/document path is valid | Reject citation |
| Line Range Valid | Line numbers exist in source | Reject citation |
| Content Match | Cited text matches source | Flag for review |
| Semantic Alignment | Claim aligns with cited content | Flag for review |

#### Hallucination Detection

```python
class HallucinationDetector:
    """Detects claims not supported by retrieved context."""
    
    CONFIDENCE_THRESHOLD = 0.7
    
    def detect(self, response: str, context: list[Citation]) -> list[Claim]:
        """Identify unsupported claims in the response."""
        claims = self._extract_claims(response)
        unsupported = []
        
        for claim in claims:
            support_score = self._calculate_support(claim, context)
            if support_score < self.CONFIDENCE_THRESHOLD:
                unsupported.append(Claim(
                    text=claim,
                    support_score=support_score,
                    nearest_citation=self._find_nearest(claim, context)
                ))
        
        return unsupported
    
    def _calculate_support(self, claim: str, context: list[Citation]) -> float:
        """Calculate how well the context supports the claim."""
        claim_embedding = self.embedder.encode(claim)
        
        max_similarity = 0.0
        for citation in context:
            citation_embedding = self.embedder.encode(citation.content)
            similarity = cosine_similarity(claim_embedding, citation_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
```

### Audit Log Schema

```json
{
  "audit_id": "uuid",
  "timestamp": "ISO8601",
  "session_id": "uuid",
  "query": "user query text",
  "response_id": "uuid",
  "validation_result": {
    "status": "PASS | REJECT | REVIEW",
    "citations_validated": 5,
    "citations_failed": 0,
    "hallucinations_detected": 1,
    "confidence_score": 0.92
  },
  "citations": [
    {
      "claim": "The DelegationEngine class is defined in...",
      "source": "src/delegation/engine.py",
      "line_range": "45-120",
      "validation_status": "VALID",
      "semantic_score": 0.95
    }
  ],
  "processing_time_ms": 150,
  "model_version": "qwen2.5-7b"
}
```

### Response Handling

| Validation Status | Action |
|-------------------|--------|
| PASS | Return response to user |
| REJECT | Regenerate with stricter citation requirements |
| REVIEW | Return with confidence warning |

### Rejection Response Format

When a response is rejected, the system returns:

```json
{
  "status": "rejected",
  "reason": "Unable to verify citation for file path",
  "suggestion": "Please rephrase your question or specify the repository",
  "partial_response": "Based on available sources...",
  "audit_id": "uuid-for-debugging"
}
```

## Consequences

### Positive

- **Trust**: Users can verify every claim
- **Quality**: Hallucinations caught before reaching users
- **Compliance**: Full audit trail for regulated environments
- **Debugging**: Failed validations help improve retrieval

### Negative

- **Latency**: Validation adds ~100-200ms per response
- **Rejection rate**: Some valid responses may be incorrectly rejected
- **Complexity**: Additional service dependency

### Mitigations

- Parallel validation (validate citations concurrently)
- Adjustable thresholds per environment (stricter for production)
- Graceful degradation (return with warning if audit service unavailable)
- User feedback loop to improve false positive rate

## Alternatives Considered

### Alternative 1: No Validation

Trust the LLM output without verification.

**Rejected because**: Unacceptable hallucination risk in a code assistant context. User trust would erode quickly.

### Alternative 2: User-Side Verification

Let users verify citations manually.

**Rejected because**: Shifts burden to users. Most won't verify, defeating the purpose of citations.

### Alternative 3: Sampling Validation

Only validate a random sample of responses.

**Rejected because**: Allows hallucinations to slip through. Inconsistent user experience.

### Alternative 4: Pre-Generation Validation

Validate context before generation instead of post-validation.

**Rejected because**: LLM can still hallucinate even with valid context. Post-validation catches generation errors.

## References

- Audit Service: `audit-service/`
- Hallucination Tests: `tests/e2e/test_kitchen_brigade_e2e.py::TestHallucinationRejectionE2E`
- Citation Model: `src/models/citation.py`

## Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Validation latency | < 200ms | ~150ms |
| False rejection rate | < 5% | 3.2% |
| Hallucination catch rate | > 95% | 97% |
| Audit log completeness | 100% | 100% |
| Citation accuracy | > 95% | 97% |

## Compliance Notes

This audit pipeline is designed to support:

- SOC 2 Type II requirements for audit trails
- GDPR right to explanation (audit logs explain AI decisions)
- Internal AI governance policies

Audit logs are retained for 90 days by default, configurable via `AUDIT_RETENTION_DAYS` environment variable.
