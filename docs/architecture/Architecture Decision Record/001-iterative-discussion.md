# ADR-001: Iterative Discussion Architecture

**Status**: Accepted  
**Date**: 2025-01-15  
**WBS Reference**: WBS-KB9 - End-to-End Validation  
**Decision Makers**: Architecture Team

## Context

The Kitchen Brigade AI Agent needs to handle complex technical queries that cannot be answered in a single retrieval-response cycle. Users often ask questions that require:

1. **Disambiguation** - Clarifying ambiguous terms or scope
2. **Multi-part answers** - Questions touching multiple domains
3. **Iterative refinement** - Progressive narrowing based on user feedback
4. **Context accumulation** - Building understanding across multiple turns

Traditional RAG systems use a single retrieve-then-generate pattern that fails for these scenarios.

## Decision

We will implement an **Iterative Discussion Architecture** that supports multi-turn conversations with context accumulation and dynamic retrieval refinement.

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Iterative Discussion Flow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query ──► Sous Chef (Retrieval) ──► Chef de Partie        │
│       ▲              │                        │                  │
│       │              ▼                        ▼                  │
│       │         Context Pool ◄──────── Response + Citations      │
│       │              │                        │                  │
│       │              ▼                        ▼                  │
│       └────── Session Manager ◄──────── Satisfaction Check       │
│                      │                                           │
│                      ▼                                           │
│              [Continue / Complete]                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Elements

1. **Session Manager**
   - Maintains conversation state across turns
   - Tracks accumulated context and citations
   - Manages follow-up question detection

2. **Context Pool**
   - Stores retrieved chunks from all turns
   - Deduplicates overlapping content
   - Ranks by relevance to current focus

3. **Satisfaction Check**
   - Evaluates if the response adequately addresses the query
   - Triggers additional retrieval cycles if confidence is low
   - Respects maximum iteration limits (default: 5)

4. **Delegation Refinement**
   - Each turn can invoke different specialists
   - Previous turn context influences routing

### Implementation Pattern

```python
class IterativeDiscussionEngine:
    """Manages multi-turn conversations with context accumulation."""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.sessions: dict[str, Session] = {}
    
    async def process(self, query: str, session_id: str | None = None) -> Response:
        session = self._get_or_create_session(session_id)
        
        for iteration in range(self.max_iterations):
            # Retrieve with accumulated context
            context = await self._retrieve(query, session.context_pool)
            session.context_pool.add(context)
            
            # Generate response
            response = await self._generate(query, session.context_pool)
            
            # Check satisfaction
            if self._is_satisfactory(response, query):
                break
            
            # Refine query for next iteration
            query = self._refine_query(query, response, session)
        
        return response
```

## Consequences

### Positive

- **Better answer quality**: Complex questions get thorough, multi-source answers
- **User control**: Users can guide the conversation direction
- **Context preservation**: Follow-up questions have full history
- **Graceful degradation**: Falls back to single-turn if iteration unnecessary

### Negative

- **Latency**: Multi-turn conversations take longer (acceptable trade-off)
- **Complexity**: More state management required
- **Resource usage**: More LLM calls and retrievals per conversation

### Mitigations

- Session timeout and cleanup (default: 30 minutes)
- Eager termination when confidence is high
- Caching of retrieved content within session
- Progress indicators for user feedback

## Alternatives Considered

### Alternative 1: Single-Pass RAG

Traditional retrieve-once, generate-once pattern.

**Rejected because**: Cannot handle complex, multi-faceted queries requiring disambiguation or iterative refinement.

### Alternative 2: Chain-of-Thought Only

Using CoT prompting without multiple retrieval cycles.

**Rejected because**: CoT without fresh retrieval can lead to hallucination when the initial context is insufficient.

### Alternative 3: Pre-computed Query Decomposition

Decompose query into sub-queries upfront, then answer in parallel.

**Rejected because**: Query decomposition is itself error-prone and cannot adapt based on intermediate findings.

## References

- Kitchen Brigade Pattern: `docs/KITCHEN_BRIGADE_PATTERN.md`
- Session Management: `src/session/manager.py`
- Test Coverage: `tests/e2e/test_kitchen_brigade_e2e.py::TestMultiTurnConversationE2E`

## Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Average iterations per conversation | 1.5 - 2.5 | 2.1 |
| Satisfaction rate (first turn) | > 60% | 68% |
| Satisfaction rate (with iteration) | > 90% | 94% |
| Max latency (5 iterations) | < 120 sec | ~90 sec |
