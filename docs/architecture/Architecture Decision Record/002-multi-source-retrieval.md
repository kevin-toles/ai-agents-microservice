# ADR-002: Multi-Source Retrieval Strategy

**Status**: Accepted  
**Date**: 2025-01-15  
**WBS Reference**: WBS-KB9 - End-to-End Validation  
**Decision Makers**: Architecture Team

## Context

The Kitchen Brigade AI Agent must answer queries that span multiple knowledge domains:

1. **Code References** - Finding specific implementations in repositories
2. **Textbook Knowledge** - Design patterns, best practices, theoretical foundations
3. **Documentation** - API docs, README files, configuration guides
4. **Runtime Context** - Current project state, user workspace

A single-source retrieval system cannot provide comprehensive, authoritative answers. We need to orchestrate multiple retrieval sources while:

- Avoiding redundant information
- Maintaining citation traceability
- Balancing freshness vs. authority
- Handling source conflicts

## Decision

We will implement a **Multi-Source Retrieval Strategy** using a federated search pattern with source-aware ranking and citation fusion.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Multi-Source Retrieval Flow                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        User Query                                │
│                            │                                     │
│                            ▼                                     │
│                    ┌───────────────┐                             │
│                    │ Query Router  │                             │
│                    └───────┬───────┘                             │
│           ┌───────────────┼───────────────┐                      │
│           ▼               ▼               ▼                      │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│    │   Code   │    │ Textbook │    │   Docs   │                 │
│    │ Retriever│    │ Retriever│    │ Retriever│                 │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘                 │
│         │               │               │                        │
│         ▼               ▼               ▼                        │
│    ┌─────────────────────────────────────────┐                  │
│    │          Citation Fusion Engine          │                  │
│    └─────────────────────┬───────────────────┘                  │
│                          │                                       │
│                          ▼                                       │
│    ┌─────────────────────────────────────────┐                  │
│    │         Source-Aware Re-Ranker          │                  │
│    └─────────────────────┬───────────────────┘                  │
│                          │                                       │
│                          ▼                                       │
│                   Ranked Results                                 │
│                   + Citations                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Source Types and Priorities

| Source | Priority | Use Case | Freshness |
|--------|----------|----------|-----------|
| Code Repository | High | Implementation details | Real-time |
| Textbooks | High | Patterns, theory | Static |
| API Documentation | Medium | Usage examples | Versioned |
| README/Guides | Medium | Setup, configuration | Versioned |
| Stack Overflow | Low | Edge cases | Cached |

### Query Router Logic

```python
class QueryRouter:
    """Routes queries to appropriate retrieval sources."""
    
    SOURCE_PATTERNS = {
        "code": [
            r"where is .* defined",
            r"find .* implementation",
            r"locate .* class|function|method",
            r"show me .* code"
        ],
        "textbook": [
            r"explain .* pattern",
            r"what is .* design",
            r"best practice for",
            r"how should I .* architecture"
        ],
        "docs": [
            r"how to configure",
            r"API for",
            r"usage example"
        ]
    }
    
    def route(self, query: str) -> list[str]:
        """Determine which sources to query."""
        sources = []
        for source, patterns in self.SOURCE_PATTERNS.items():
            if any(re.search(p, query.lower()) for p in patterns):
                sources.append(source)
        
        # Default to all sources if no pattern matches
        return sources or ["code", "textbook", "docs"]
```

### Citation Fusion Engine

The fusion engine combines results from multiple sources while:

1. **Deduplicating** - Identifying same content from different sources
2. **Cross-referencing** - Linking related concepts across sources
3. **Conflict resolution** - Handling contradictory information

```python
class CitationFusionEngine:
    """Fuses citations from multiple retrieval sources."""
    
    def fuse(self, results: dict[str, list[Citation]]) -> list[FusedCitation]:
        fused = []
        
        for source, citations in results.items():
            for citation in citations:
                # Check for duplicates
                existing = self._find_duplicate(fused, citation)
                if existing:
                    existing.add_source(source)
                else:
                    fused.append(FusedCitation(citation, source))
        
        return self._resolve_conflicts(fused)
```

### Source-Aware Re-Ranking

Final ranking considers:

1. **Relevance score** (semantic similarity)
2. **Source authority** (textbook > code > docs for concepts)
3. **Freshness** (code > docs > textbook for implementations)
4. **Citation density** (prefer well-cited content)

## Consequences

### Positive

- **Comprehensive answers**: Queries get context from all relevant sources
- **Verifiable citations**: Every claim traces to specific source
- **Balanced perspective**: Theory from textbooks, reality from code
- **Conflict visibility**: Contradictions are surfaced, not hidden

### Negative

- **Latency**: Multiple retrieval calls add time
- **Complexity**: Fusion logic is non-trivial
- **Storage**: Multiple indices to maintain

### Mitigations

- Parallel retrieval calls (all sources queried simultaneously)
- Aggressive caching (30-minute TTL for code, 24h for textbooks)
- Lazy loading (only query additional sources if initial results insufficient)

## Alternatives Considered

### Alternative 1: Single Combined Index

Merge all sources into one unified vector index.

**Rejected because**: Loses source provenance, making citations unreliable. Cannot apply source-specific ranking logic.

### Alternative 2: Sequential Source Querying

Query sources one by one until sufficient results found.

**Rejected because**: Biases toward whichever source is queried first. Misses cross-source insights.

### Alternative 3: User-Selected Sources

Let users explicitly choose which sources to query.

**Rejected because**: Adds cognitive burden. Users don't always know which source is best for their query.

## References

- Semantic Search Service: `semantic-search-service/`
- Code Orchestrator: `Code-Orchestrator-Service/`
- Textbook Integration: `ai-platform-data/books/`
- Test Coverage: `tests/e2e/test_kitchen_brigade_e2e.py::TestCodeLocationE2E`

## Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Average sources per query | 2+ | 2.3 |
| Cross-source citation rate | > 40% | 47% |
| Retrieval latency (parallel) | < 500ms | ~350ms |
| Citation accuracy | > 95% | 97% |
