# Kitchen Brigade Architecture Implementation WBS

> **Version:** 1.1.0  
> **Created:** 2025-12-31  
> **Updated:** 2025-12-31  
> **Status:** Planning Phase  
> **Reference:** [KITCHEN_BRIGADE_ARCHITECTURE.md](./KITCHEN_BRIGADE_ARCHITECTURE.md)  
> **Prerequisite:** [WBS.md](./WBS.md) (WBS-AGT1 through WBS-AGT24)

---

## Relationship to WBS.md (Foundation Blocks)

This WBS **extends** the foundation established in [WBS.md](./WBS.md). The following blocks from WBS.md are **prerequisites**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WBS.md (FOUNDATION)                                 │
│                                                                             │
│   WBS-AGT1-5    Core Infrastructure, State, Schemas, Base Class            │
│        │                                                                    │
│        ▼                                                                    │
│   WBS-AGT6-13   8 Agent Functions (extract, summarize, generate, etc.)     │
│        │                                                                    │
│        ▼                                                                    │
│   WBS-AGT14     Pipeline Orchestrator (SequentialAgent, ParallelAgent)     │
│        │                                                                    │
│        ├────────────────────────────────────────────────────────┐          │
│        ▼                                                        ▼          │
│   WBS-AGT15-17  Pipelines + Citations                    WBS-AGT21-24      │
│   (Chapter, CodeGen, Audit)                              (Knowledge        │
│        │                                                  Retrieval)       │
│        │                                                       │           │
│        ▼                                                       │           │
│   WBS-AGT18-19  API Routes + Anti-Pattern                      │           │
│        │                                                       │           │
│        ▼                                                       │           │
│   WBS-AGT20     Integration Testing (Basic)                    │           │
│                                                                │           │
└────────────────────────────────────────────────────────────────┼───────────┘
                                                                 │
                 ┌───────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WBS_KITCHEN_BRIGADE.md (THIS DOCUMENT)                   │
│                                                                             │
│   WBS-KB1-4     LLM Discussion Loop + Agreement Engine                      │
│        │                                                                    │
│        ▼                                                                    │
│   WBS-KB5-6     Provenance + Cross-Reference Pipeline ◄── Uses AGT21-24    │
│        │                                                                    │
│        ▼                                                                    │
│   WBS-KB7-8     Code-Orchestrator Tools + VS Code MCP                       │
│        │                                                                    │
│        ▼                                                                    │
│   WBS-KB9       End-to-End Validation (Extends AGT20)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Dependencies from WBS.md

| Kitchen Brigade Block | Depends On | Why |
|----------------------|------------|-----|
| WBS-KB1 (Discussion Loop) | WBS-AGT14 | Uses ParallelAgent for concurrent LLM calls |
| WBS-KB3 (Evidence Gathering) | **WBS-AGT24** | UnifiedRetriever is the evidence source |
| WBS-KB3 (Evidence Gathering) | WBS-AGT21-23 | Code, Neo4j, Book clients provide evidence |
| WBS-KB5 (Provenance) | WBS-AGT17 | Extends CitationManager with provenance tracking |
| WBS-KB6 (Pipeline) | WBS-AGT14 | Uses PipelineOrchestrator as base |
| WBS-KB9 (E2E) | WBS-AGT20 | Extends basic integration tests |

### What WBS.md Provides (Do NOT Duplicate)

The following are **already implemented** by WBS.md and should NOT be re-implemented:

- ✅ `UnifiedRetriever` (WBS-AGT24) — multi-source retrieval
- ✅ `CodeReferenceClient` (WBS-AGT21) — code search
- ✅ `Neo4jClient` (WBS-AGT22) — graph traversal
- ✅ `BookPassageClient` (WBS-AGT23) — textbook search
- ✅ `CitationManager` (WBS-AGT17) — citation tracking
- ✅ `ParallelAgent` (WBS-AGT14) — concurrent execution
- ✅ Basic integration tests (WBS-AGT20)

### What This WBS Adds (New Capabilities)

- 🆕 `LLMDiscussionLoop` — multi-LLM debate with iterations
- 🆕 `InformationRequest` — LLMs articulate what they need
- 🆕 `EvidenceGatherer` — iterative retrieval based on requests
- 🆕 `AgreementEngine` — consensus calculation
- 🆕 `ProvenanceTracker` — claim → source → participant → cycle
- 🆕 `AuditValidator` — pre-delivery citation validation
- 🆕 `CodeOrchestratorClient` — CodeT5+, GraphCodeBERT tools
- 🆕 MCP Server — VS Code integration
- 🆕 E2E tests for iterative discussion pipeline

---

## Executive Summary

This WBS implements the **Kitchen Brigade 2.0 Architecture** — specifically the **iterative LLM discussion loop** and **multi-source cross-reference pipeline** that enables agents to work **alongside or in lieu of VS Code Copilot**.

### End-State Vision

Upon completion, you will be able to:
1. **Ask complex questions** to local agents instead of (or alongside) Copilot
2. **Receive grounded, cited responses** from multiple LLMs that have discussed and refined their answers
3. **Iteratively request additional information** — agents loop back for more evidence when needed
4. **Trust the output** — audit-service validates all citations before delivery
5. **Use Code-Orchestrator tools** for objective validation (CodeT5+, GraphCodeBERT, SonarQube)

### Self-Enclosed Acceptance Criteria Philosophy

Each WBS block can be **closed independently**:
- AC/EC do NOT require future blocks to validate
- Integration tests use existing services (inference-service, semantic-search, audit-service)
- Each block produces a **working subsystem** that can be demonstrated

---

## WBS Summary

| Block | Name | Dependencies | Est. Effort | Milestone |
|-------|------|--------------|-------------|-----------|
| WBS-KB1 | LLM Discussion Loop Core | WBS-AGT14 | 12 hours | Multi-LLM debate works |
| WBS-KB2 | Information Request Detection | WBS-KB1 | 8 hours | Agents request more info |
| WBS-KB3 | Iterative Evidence Gathering | WBS-KB2, **WBS-AGT21-24** | 10 hours | Loop retrieves new evidence |
| WBS-KB4 | Agreement/Consensus Engine | WBS-KB1 | 8 hours | Disagreement → agreement |
| WBS-KB5 | Provenance & Audit Integration | WBS-KB4, **WBS-AGT17** | 10 hours | All claims validated |
| WBS-KB6 | Cross-Reference Pipeline Orchestration | WBS-KB3, WBS-KB4, WBS-KB5 | 12 hours | Full pipeline works |
| WBS-KB7 | Code-Orchestrator Tool Integration | WBS-KB6 | 12 hours | CodeT5+, GraphCodeBERT |
| WBS-KB8 | VS Code MCP Server | WBS-KB6 | 16 hours | Copilot replacement/tandem |
| WBS-KB9 | End-to-End Validation | All prior, **WBS-AGT20** | 12 hours | Production-ready |

**Total Estimated Effort:** ~100 hours (on top of WBS.md ~136 hours)

**Combined Total:** ~236 hours for complete Kitchen Brigade implementation

---

## Critical Path

```
WBS-AGT14 (Pipeline Orchestrator) ─────────────────────────────────────────────┐
        │                                                                       │
        ▼                                                                       │
┌───────────────────────────────────────────────────────────────────────────┐  │
│                        PHASE 1: LLM Discussion Foundation                 │  │
│                                                                           │  │
│   WBS-KB1 (Discussion Loop) ──► WBS-KB2 (Info Request) ──► WBS-KB4       │  │
│         │                             │                   (Agreement)     │  │
│         │                             │                        │          │  │
│         └──────────────────────►┬─────┘                        │          │  │
│                                 │                              │          │  │
│                                 ▼                              │          │  │
│                          WBS-KB3 (Iterative)                   │          │  │
│                          (Evidence Gathering)                  │          │  │
│                                 │                              │          │  │
│                                 └──────────────────────────────┘          │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                      │                                        │
                                      ▼                                        │
┌───────────────────────────────────────────────────────────────────────────┐  │
│                        PHASE 2: Grounding & Validation                    │  │
│                                                                           │  │
│   WBS-KB5 (Provenance) ◄────────────────────────────────────── WBS-AGT17 │  │
│         │                                                                 │  │
│         ▼                                                                 │  │
│   WBS-KB6 (Cross-Reference Pipeline Orchestration)                        │  │
│         │                                                                 │  │
│         ├───────────────────────────────────────────────────────────────┐│  │
│         │                                                               ││  │
│         ▼                                                               ▼│  │
│   WBS-KB7 (Code-Orchestrator Tools)                              WBS-KB8│◄─┘
│   [CodeT5+, GraphCodeBERT, SonarQube]                        (VS Code MCP)
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: Production Readiness                      │
│                                                                           │
│                          WBS-KB9 (End-to-End Validation)                  │
│                                                                           │
│   ✓ User can ask: "Where is the rate limiter implemented?"                │
│   ✓ Multiple LLMs discuss, request additional info, reach consensus       │
│   ✓ Response includes citations to actual code files                      │
│   ✓ audit-service validates all citations before delivery                 │
│   ✓ Works alongside OR instead of VS Code Copilot                         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## WBS-KB1: LLM Discussion Loop Core

**Dependencies:** WBS-AGT14 (Pipeline Orchestrator)  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Cross-Reference Pipeline (Iterative)

### Purpose

Implement the foundation where **multiple LLMs analyze evidence in parallel**, producing independent analyses that can be compared for agreement/disagreement.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB1.1 | `LLMParticipant` class wraps inference-service calls with participant identity | Unit test: participant returns analysis with `participant_id` |
| AC-KB1.2 | `DiscussionCycle` dataclass captures cycle_number, analyses, agreement_score | Unit test: dataclass serializes to JSON |
| AC-KB1.3 | `LLMDiscussionLoop.discuss()` runs N participants in parallel via `asyncio.gather` | Unit test: 2+ participants called concurrently |
| AC-KB1.4 | Each participant receives same evidence and query, produces independent analysis | Unit test: analyses differ based on participant preset |
| AC-KB1.5 | Discussion loop uses configurable `max_cycles` (default 5) | Unit test: loop terminates after max_cycles |
| AC-KB1.6 | Discussion history preserved as `list[DiscussionCycle]` | Unit test: history has 1 entry per cycle executed |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB1.1 | Create `LLMParticipant` protocol | AC-KB1.1 | `src/discussion/protocols.py` |
| KB1.2 | Implement `LLMParticipant` class | AC-KB1.1, AC-KB1.4 | `src/discussion/participant.py` |
| KB1.3 | Create `DiscussionCycle` dataclass | AC-KB1.2 | `src/discussion/models.py` |
| KB1.4 | Create `DiscussionResult` dataclass | AC-KB1.2, AC-KB1.6 | `src/discussion/models.py` |
| KB1.5 | Implement `LLMDiscussionLoop.__init__` | AC-KB1.5 | `src/discussion/loop.py` |
| KB1.6 | Implement `LLMDiscussionLoop.discuss()` with asyncio.gather | AC-KB1.3 | `src/discussion/loop.py` |
| KB1.7 | Implement single-cycle execution logic | AC-KB1.4, AC-KB1.6 | `src/discussion/loop.py` |
| KB1.8 | Create `FakeLLMParticipant` for testing | AC-KB1.1-6 | `tests/unit/discussion/fake_participant.py` |
| KB1.9 | Write unit tests for LLMDiscussionLoop | AC-KB1.1-6 | `tests/unit/discussion/test_loop.py` |
| KB1.10 | Integration test with real inference-service | AC-KB1.3 | `tests/integration/test_discussion_loop.py` |

### Exit Criteria

- [ ] `pytest tests/unit/discussion/` passes with 100% coverage
- [ ] `LLMDiscussionLoop` with 2 participants produces 2 independent analyses per cycle
- [ ] Integration test: qwen2.5-7b and deepseek-r1-7b produce different analyses for same evidence
- [ ] `discussion_result.history` contains correct number of cycles
- [ ] **DEMO:** Run `python -m src.discussion.loop --demo` shows parallel LLM analysis

---

## WBS-KB2: Information Request Detection

**Dependencies:** WBS-KB1  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → LLM Discussion Loop Details

### Purpose

When LLMs disagree or have low confidence, they should **articulate what additional information would help** — not just say "I don't know."

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB2.1 | `InformationRequest` model captures: query, source_types, priority | Unit test: model validates |
| AC-KB2.2 | `extract_information_requests()` parses LLM analysis for requests | Unit test: extracts requests from structured output |
| AC-KB2.3 | LLM prompt template includes section for "What additional information would help?" | Manual: review prompt template |
| AC-KB2.4 | Requests specify `source_types` (code, books, textbooks, graph) | Unit test: source_types parsed correctly |
| AC-KB2.5 | Requests have `priority` (high/medium/low) based on disagreement severity | Unit test: priority correlates with confidence gap |
| AC-KB2.6 | Zero requests returned when agreement_score > threshold | Unit test: no requests when agreed |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB2.1 | Create `InformationRequest` schema | AC-KB2.1, AC-KB2.4, AC-KB2.5 | `src/discussion/models.py` |
| KB2.2 | Create analysis prompt template with info request section | AC-KB2.3 | `src/discussion/prompts/analysis_prompt.md` |
| KB2.3 | Implement `extract_information_requests()` parser | AC-KB2.2 | `src/discussion/request_extractor.py` |
| KB2.4 | Implement priority scoring based on disagreement | AC-KB2.5 | `src/discussion/request_extractor.py` |
| KB2.5 | Add request extraction to `LLMDiscussionLoop` cycle | AC-KB2.2-6 | `src/discussion/loop.py` |
| KB2.6 | Unit tests for InformationRequest | AC-KB2.1 | `tests/unit/discussion/test_models.py` |
| KB2.7 | Unit tests for extract_information_requests | AC-KB2.2-6 | `tests/unit/discussion/test_request_extractor.py` |
| KB2.8 | Integration test: LLM produces parseable info requests | AC-KB2.2 | `tests/integration/test_info_requests.py` |

### Exit Criteria

- [ ] `pytest tests/unit/discussion/test_request_extractor.py` passes with 100% coverage
- [ ] LLM analysis includes structured `information_requests` section
- [ ] `extract_information_requests(analysis)` returns `list[InformationRequest]`
- [ ] Disagreement between LLMs produces high-priority requests
- [ ] Agreement produces zero requests
- [ ] **DEMO:** Show LLM asking "Need to see AST chunking implementation" when evidence is insufficient

---

## WBS-KB3: Iterative Evidence Gathering

**Dependencies:** WBS-KB2, WBS-AGT24 (Unified Knowledge Retrieval)  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Cross-Reference Pipeline

### Purpose

When LLMs request additional information, the system **automatically retrieves new evidence** and feeds it back into the discussion loop.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB3.1 | `evidence_gatherer.gather()` takes `list[InformationRequest]`, returns new evidence | Unit test: gatherer returns CrossReferenceResult |
| AC-KB3.2 | Gatherer calls UnifiedRetriever (from WBS-AGT24) with request queries | Integration test: hits Qdrant, Neo4j, code-reference |
| AC-KB3.3 | Evidence from multiple sources merged without duplicates | Unit test: deduplication works |
| AC-KB3.4 | `merge_evidence()` combines old + new evidence, preserving provenance | Unit test: merged result tracks source cycle |
| AC-KB3.5 | High-priority requests processed before medium/low | Unit test: priority ordering |
| AC-KB3.6 | Gatherer respects `source_types` filter on requests | Unit test: code-only request doesn't hit books |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB3.1 | Create `EvidenceGatherer` class | AC-KB3.1 | `src/discussion/evidence_gatherer.py` |
| KB3.2 | Implement `gather()` method with UnifiedRetriever | AC-KB3.1, AC-KB3.2 | `src/discussion/evidence_gatherer.py` |
| KB3.3 | Implement source_types filtering | AC-KB3.6 | `src/discussion/evidence_gatherer.py` |
| KB3.4 | Implement priority-based ordering | AC-KB3.5 | `src/discussion/evidence_gatherer.py` |
| KB3.5 | Implement `merge_evidence()` with deduplication | AC-KB3.3, AC-KB3.4 | `src/discussion/evidence_merger.py` |
| KB3.6 | Add cycle provenance tracking to merged evidence | AC-KB3.4 | `src/discussion/evidence_merger.py` |
| KB3.7 | Integrate evidence gathering into `LLMDiscussionLoop` | AC-KB3.1-6 | `src/discussion/loop.py` |
| KB3.8 | Unit tests for EvidenceGatherer | AC-KB3.1-6 | `tests/unit/discussion/test_evidence_gatherer.py` |
| KB3.9 | Integration test: loop retrieves new evidence and continues | AC-KB3.2 | `tests/integration/test_iterative_loop.py` |

### Exit Criteria

- [ ] `pytest tests/unit/discussion/test_evidence_gatherer.py` passes with 100% coverage
- [ ] `evidence_gatherer.gather([request])` returns evidence from correct sources
- [ ] Evidence merger produces no duplicates across cycles
- [ ] Integration test shows: Cycle 1 → info request → Cycle 2 with new evidence
- [ ] **DEMO:** Ask "sub-agent patterns" → LLM requests "ParallelAgent implementation" → system retrieves agents.py → LLM refines answer

---

## WBS-KB4: Agreement/Consensus Engine

**Dependencies:** WBS-KB1  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → LLM Discussion Loop

### Purpose

Determine when LLMs **agree enough to stop iterating**, and synthesize their consensus into a coherent answer.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB4.1 | `calculate_agreement()` returns score 0.0-1.0 from list of analyses | Unit test: known inputs produce expected scores |
| AC-KB4.2 | Agreement considers: claim overlap, citation overlap, confidence levels | Unit test: each factor affects score |
| AC-KB4.3 | `agreement_threshold` configurable (default 0.85) | Unit test: threshold checked correctly |
| AC-KB4.4 | Disagreement points extracted and logged | Unit test: disagreements identified |
| AC-KB4.5 | `synthesize_consensus()` merges analyses when agreement reached | Unit test: produces single coherent output |
| AC-KB4.6 | Consensus tracks which claims came from which participant | Unit test: provenance preserved |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB4.1 | Create `AgreementResult` schema | AC-KB4.1, AC-KB4.4 | `src/discussion/models.py` |
| KB4.2 | Implement `calculate_agreement()` | AC-KB4.1, AC-KB4.2 | `src/discussion/agreement.py` |
| KB4.3 | Implement claim overlap scoring | AC-KB4.2 | `src/discussion/agreement.py` |
| KB4.4 | Implement citation overlap scoring | AC-KB4.2 | `src/discussion/agreement.py` |
| KB4.5 | Implement confidence-weighted scoring | AC-KB4.2 | `src/discussion/agreement.py` |
| KB4.6 | Implement `extract_disagreements()` | AC-KB4.4 | `src/discussion/agreement.py` |
| KB4.7 | Implement `synthesize_consensus()` | AC-KB4.5, AC-KB4.6 | `src/discussion/consensus.py` |
| KB4.8 | Add participant provenance to consensus output | AC-KB4.6 | `src/discussion/consensus.py` |
| KB4.9 | Integrate agreement check into `LLMDiscussionLoop` | AC-KB4.3 | `src/discussion/loop.py` |
| KB4.10 | Unit tests for agreement calculation | AC-KB4.1-4 | `tests/unit/discussion/test_agreement.py` |
| KB4.11 | Unit tests for consensus synthesis | AC-KB4.5-6 | `tests/unit/discussion/test_consensus.py` |

### Exit Criteria

- [ ] `pytest tests/unit/discussion/test_agreement.py` passes with 100% coverage
- [ ] Two identical analyses → agreement_score = 1.0
- [ ] Two contradictory analyses → agreement_score < 0.5
- [ ] Consensus output identifies "Participant A said X, Participant B agreed"
- [ ] **DEMO:** Show two LLMs initially disagree (0.6), then agree (0.92) after new evidence

---

## WBS-KB5: Provenance & Audit Integration

**Dependencies:** WBS-KB4, WBS-AGT17 (Citation Flow & Audit)  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → audit-service Integration

### Purpose

**Every claim in the final output must be traceable** to a source. audit-service validates citations before delivery.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB5.1 | All claims in consensus output have `[^N]` citation markers | Unit test: no uncited claims |
| AC-KB5.2 | Citations validated via audit-service:8084/v1/validate | Integration test: audit-service called |
| AC-KB5.3 | Invalid citations rejected, trigger additional evidence gathering | Integration test: retry on invalid |
| AC-KB5.4 | `ProvenanceTracker` logs: claim, source, participant, cycle | Unit test: tracker captures all fields |
| AC-KB5.5 | Audit trail includes discussion_history with all cycles | Integration test: full history in audit |
| AC-KB5.6 | Chicago-style footnotes generated for all citation types | Unit test: book, code, graph citations formatted |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB5.1 | Create `ProvenanceTracker` class | AC-KB5.4 | `src/discussion/provenance.py` |
| KB5.2 | Implement claim → source → participant → cycle tracking | AC-KB5.4 | `src/discussion/provenance.py` |
| KB5.3 | Implement citation marker injection in consensus | AC-KB5.1 | `src/discussion/consensus.py` |
| KB5.4 | Create `AuditServiceValidator` class | AC-KB5.2 | `src/discussion/audit_validator.py` |
| KB5.5 | Implement citation validation via audit-service | AC-KB5.2, AC-KB5.3 | `src/discussion/audit_validator.py` |
| KB5.6 | Implement retry logic for invalid citations | AC-KB5.3 | `src/discussion/loop.py` |
| KB5.7 | Add discussion_history to audit payload | AC-KB5.5 | `src/discussion/audit_validator.py` |
| KB5.8 | Implement Chicago formatter for all citation types | AC-KB5.6 | `src/citations/chicago_formatter.py` |
| KB5.9 | Unit tests for ProvenanceTracker | AC-KB5.4 | `tests/unit/discussion/test_provenance.py` |
| KB5.10 | Integration test with audit-service | AC-KB5.2, AC-KB5.5 | `tests/integration/test_audit_validation.py` |

### Exit Criteria

- [ ] `pytest tests/unit/discussion/test_provenance.py` passes with 100% coverage
- [ ] Every claim in output has citation marker
- [ ] audit-service receives validation request with citations
- [ ] Invalid citation (fake source) triggers loop retry
- [ ] Audit trail shows: "Claim X from Participant A in Cycle 2, source: agents.py#L135"
- [ ] **DEMO:** Show audit-service rejecting hallucinated citation, system retrying with real evidence

---

## WBS-KB6: Cross-Reference Pipeline Orchestration

**Dependencies:** WBS-KB3, WBS-KB4, WBS-KB5  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Complete Flow Example

### Purpose

Wire everything together: the full **cross_reference → discuss → iterate → validate → deliver** pipeline.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB6.1 | `CrossReferencePipeline` orchestrates all KB components | Integration test: full pipeline runs |
| AC-KB6.2 | Pipeline stages: decompose → parallel_retrieval → discussion_loop → validate → format | Integration test: all stages execute |
| AC-KB6.3 | Pipeline terminates when: agreement reached OR max_cycles OR validation passed | Unit test: all termination conditions |
| AC-KB6.4 | Final output is `GroundedResponse` with content, citations, confidence, metadata | Unit test: schema validates |
| AC-KB6.5 | Metadata includes: cycles_used, participants, sources_consulted, processing_time | Unit test: all metadata fields present |
| AC-KB6.6 | Pipeline registered as `/v1/pipelines/cross-reference/run` | Integration test: API endpoint works |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB6.1 | Create `GroundedResponse` schema | AC-KB6.4, AC-KB6.5 | `src/schemas/grounded_response.py` |
| KB6.2 | Create `CrossReferencePipeline` class | AC-KB6.1 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.3 | Implement stage 1: decompose_task integration | AC-KB6.2 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.4 | Implement stage 2: parallel_retrieval (UnifiedRetriever) | AC-KB6.2 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.5 | Implement stage 3: LLMDiscussionLoop integration | AC-KB6.2 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.6 | Implement stage 4: audit validation | AC-KB6.2 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.7 | Implement stage 5: response formatting | AC-KB6.2 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.8 | Implement termination conditions | AC-KB6.3 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.9 | Implement metadata collection | AC-KB6.5 | `src/pipelines/cross_reference_pipeline.py` |
| KB6.10 | Register API route | AC-KB6.6 | `src/api/routes/pipelines.py` |
| KB6.11 | Unit tests for CrossReferencePipeline | AC-KB6.1-5 | `tests/unit/pipelines/test_cross_reference_pipeline.py` |
| KB6.12 | Integration test: full pipeline execution | AC-KB6.1-6 | `tests/integration/test_cross_reference_full.py` |

### Exit Criteria

- [ ] `pytest tests/unit/pipelines/test_cross_reference_pipeline.py` passes
- [ ] POST `/v1/pipelines/cross-reference/run` returns GroundedResponse
- [ ] Response includes all metadata fields
- [ ] Pipeline completes in <60s for typical queries
- [ ] **DEMO:** Ask "Where is the repository pattern implemented?" → get grounded answer with citations to code-reference-engine files

---

## WBS-KB7: Code-Orchestrator Tool Integration

**Dependencies:** WBS-KB6  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Agent → Tool/Service Mapping

### Purpose

Integrate **objective code analysis tools** (CodeT5+, GraphCodeBERT, SonarQube) so LLM claims can be validated against actual code metrics.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB7.1 | `CodeOrchestratorClient` wraps Code-Orchestrator:8083 API | Unit test: client methods work |
| AC-KB7.2 | `keyword_extraction` tool uses CodeT5+ via Code-Orchestrator | Integration test: extracts keywords |
| AC-KB7.3 | `term_validation` tool uses GraphCodeBERT via Code-Orchestrator | Integration test: validates terms |
| AC-KB7.4 | `code_ranking` tool uses CodeBERT via Code-Orchestrator | Integration test: ranks results |
| AC-KB7.5 | `sonarqube_analyze` tool integrated for quality metrics | Integration test: returns findings |
| AC-KB7.6 | Tools available to `analyze_artifact` and `validate_against_spec` agents | Integration test: agents use tools |
| AC-KB7.7 | Validation failures from tools trigger discussion loop retry | Integration test: retry on bad code |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB7.1 | Create `CodeOrchestratorClient` | AC-KB7.1 | `src/clients/code_orchestrator.py` |
| KB7.2 | Implement `extract_keywords()` wrapper | AC-KB7.2 | `src/clients/code_orchestrator.py` |
| KB7.3 | Implement `validate_terms()` wrapper | AC-KB7.3 | `src/clients/code_orchestrator.py` |
| KB7.4 | Implement `rank_code_results()` wrapper | AC-KB7.4 | `src/clients/code_orchestrator.py` |
| KB7.5 | Create `SonarQubeClient` | AC-KB7.5 | `src/clients/sonarqube.py` |
| KB7.6 | Implement `analyze_file()` method | AC-KB7.5 | `src/clients/sonarqube.py` |
| KB7.7 | Create `CodeValidationTool` combining all tools | AC-KB7.6 | `src/tools/code_validation.py` |
| KB7.8 | Integrate CodeValidationTool into analyze_artifact | AC-KB7.6 | `src/functions/analyze_artifact.py` |
| KB7.9 | Integrate CodeValidationTool into validate_against_spec | AC-KB7.6 | `src/functions/validate_against_spec.py` |
| KB7.10 | Implement validation failure → retry logic | AC-KB7.7 | `src/discussion/loop.py` |
| KB7.11 | Unit tests with FakeCodeOrchestratorClient | AC-KB7.1-4 | `tests/unit/clients/test_code_orchestrator.py` |
| KB7.12 | Integration tests with real Code-Orchestrator | AC-KB7.2-5 | `tests/integration/test_code_orchestrator.py` |

### Exit Criteria

- [ ] `pytest tests/unit/clients/test_code_orchestrator.py` passes
- [ ] `extract_keywords("class Repository")` returns ["repository", "pattern", ...]
- [ ] `validate_terms(["repositry"], query)` catches typo, returns low score
- [ ] SonarQube analysis returns complexity, security findings
- [ ] LLM claim "this code has CC < 10" validated against actual metrics
- [ ] **DEMO:** Generate code → CodeT5+ extracts keywords → GraphCodeBERT validates → SonarQube checks quality → all pass before delivery

---

## WBS-KB8: VS Code MCP Server

**Dependencies:** WBS-KB6  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Output Flow Architecture

### Purpose

Create an **MCP (Model Context Protocol) server** that VS Code can use **alongside or instead of Copilot**. This is the user-facing interface.

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB8.1 | MCP server implements `tools/list` and `tools/call` | Integration test: MCP protocol works |
| AC-KB8.2 | Tool: `cross_reference` — search knowledge bases, get grounded answer | Manual test: works in VS Code |
| AC-KB8.3 | Tool: `generate_code` — generate code with citations | Manual test: works in VS Code |
| AC-KB8.4 | Tool: `analyze_code` — analyze selection with Code-Orchestrator | Manual test: works in VS Code |
| AC-KB8.5 | Tool: `explain_code` — explain with textbook references | Manual test: works in VS Code |
| AC-KB8.6 | MCP server configurable via `mcp.json` in workspace | Documentation: setup instructions |
| AC-KB8.7 | Responses include inline citations `[^N]` and footnotes block | Manual test: citations visible |
| AC-KB8.8 | User can ask follow-up questions (triggers additional iteration) | Manual test: follow-up works |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB8.1 | Create MCP server scaffold | AC-KB8.1 | `src/mcp/server.py` |
| KB8.2 | Implement `tools/list` handler | AC-KB8.1 | `src/mcp/handlers/tools.py` |
| KB8.3 | Implement `tools/call` handler | AC-KB8.1 | `src/mcp/handlers/tools.py` |
| KB8.4 | Implement `cross_reference` tool | AC-KB8.2 | `src/mcp/tools/cross_reference.py` |
| KB8.5 | Implement `generate_code` tool | AC-KB8.3 | `src/mcp/tools/generate_code.py` |
| KB8.6 | Implement `analyze_code` tool | AC-KB8.4 | `src/mcp/tools/analyze_code.py` |
| KB8.7 | Implement `explain_code` tool | AC-KB8.5 | `src/mcp/tools/explain_code.py` |
| KB8.8 | Create `mcp.json` template | AC-KB8.6 | `config/mcp.json.template` |
| KB8.9 | Implement citation formatting for MCP responses | AC-KB8.7 | `src/mcp/formatters/citation_formatter.py` |
| KB8.10 | Implement session state for follow-ups | AC-KB8.8 | `src/mcp/session.py` |
| KB8.11 | Write setup documentation | AC-KB8.6 | `docs/MCP_SETUP.md` |
| KB8.12 | Integration test: MCP protocol compliance | AC-KB8.1 | `tests/integration/test_mcp_server.py` |
| KB8.13 | Manual test script for VS Code | AC-KB8.2-8 | `scripts/test_mcp_vscode.sh` |

### Exit Criteria

- [ ] MCP server starts on configured port
- [ ] VS Code detects tools via `mcp.json`
- [ ] "cross_reference: Where is rate limiter?" returns grounded answer
- [ ] "generate_code: Repository pattern in Python" returns code with citations
- [ ] Response includes: content, `[^1]`, `[^2]`, and footnotes block
- [ ] Follow-up "show me the tests too" triggers additional iteration
- [ ] **DEMO:** Screen recording of VS Code using local agents instead of Copilot

---

## WBS-KB9: End-to-End Validation

**Dependencies:** All prior WBS-KB blocks, **WBS-AGT20** (Integration Testing from WBS.md)  
**Reference:** KITCHEN_BRIGADE_ARCHITECTURE.md → Complete Flow Example

### Purpose

Validate the entire system works **end-to-end** for the core use case: answering complex questions about a large codebase with grounded, cited responses.

**Note:** This block **extends** WBS-AGT20 from WBS.md. The basic service integration tests (inference-service, semantic-search, audit-service) are already covered by WBS-AGT20. This block adds tests specific to the **iterative discussion pipeline**.

### Relationship to WBS-AGT20

| WBS-AGT20 (Already Covered) | WBS-KB9 (New Tests) |
|-----------------------------|---------------------|
| Function → response E2E | Multi-cycle discussion E2E |
| Pipeline → citations E2E | Iterative evidence gathering E2E |
| inference-service integration | Multi-LLM debate integration |
| semantic-search integration | Discussion loop + retrieval integration |
| audit-service integration | Pre-delivery validation integration |
| 5 concurrent pipeline load test | Discussion loop performance test |

### Acceptance Criteria

| ID | Requirement | Validation Method |
|----|-------------|-------------------|
| AC-KB9.1 | E2E test: "Where is the rate limiter implemented?" returns code location with citation | Automated E2E test |
| AC-KB9.2 | E2E test: "Explain the repository pattern" returns explanation with book + code citations | Automated E2E test |
| AC-KB9.3 | E2E test: "Generate a cache service" produces code validated by Code-Orchestrator | Automated E2E test |
| AC-KB9.4 | E2E test: Deliberate hallucination rejected by audit-service | Automated E2E test |
| AC-KB9.5 | E2E test: Multi-turn conversation maintains context | Automated E2E test |
| AC-KB9.6 | Performance: <60s for typical query, <120s for complex multi-cycle | Performance test |
| AC-KB9.7 | Documentation: Complete setup guide for new users | Documentation review |
| AC-KB9.8 | Documentation: Architecture decision records (ADRs) | Documentation review |

### WBS Tasks

| ID | Task | AC | File(s) |
|----|------|-----|---------|
| KB9.1 | Write E2E test: code location query | AC-KB9.1 | `tests/e2e/test_code_location.py` |
| KB9.2 | Write E2E test: concept explanation | AC-KB9.2 | `tests/e2e/test_concept_explanation.py` |
| KB9.3 | Write E2E test: code generation | AC-KB9.3 | `tests/e2e/test_code_generation.py` |
| KB9.4 | Write E2E test: hallucination rejection | AC-KB9.4 | `tests/e2e/test_hallucination_rejection.py` |
| KB9.5 | Write E2E test: multi-turn conversation | AC-KB9.5 | `tests/e2e/test_multi_turn.py` |
| KB9.6 | Write performance benchmarks | AC-KB9.6 | `tests/e2e/test_performance.py` |
| KB9.7 | Create docker-compose.e2e.yml | AC-KB9.1-6 | `docker/docker-compose.e2e.yml` |
| KB9.8 | Write complete setup guide | AC-KB9.7 | `docs/GETTING_STARTED.md` |
| KB9.9 | Write ADR: Why iterative discussion | AC-KB9.8 | `docs/adr/001-iterative-discussion.md` |
| KB9.10 | Write ADR: Why multi-source retrieval | AC-KB9.8 | `docs/adr/002-multi-source-retrieval.md` |
| KB9.11 | Write ADR: Why audit validation | AC-KB9.8 | `docs/adr/003-audit-validation.md` |
| KB9.12 | Create demo video/script | AC-KB9.7 | `docs/DEMO.md` |

### Exit Criteria

- [ ] `pytest tests/e2e/` passes (all 5 E2E tests)
- [ ] Performance benchmarks meet targets (<60s typical, <120s complex)
- [ ] `docs/GETTING_STARTED.md` allows new user to set up in <30 minutes
- [ ] All 3 ADRs document key architectural decisions
- [ ] **DEMO VIDEO:** 5-minute walkthrough showing:
  1. Ask complex question
  2. Watch LLMs discuss (visible in logs)
  3. See request for additional info
  4. See new evidence gathered
  5. Receive grounded answer with citations
  6. Verify citation links to actual source

---

## Milestone Summary

| Phase | Blocks | Milestone | User Capability |
|-------|--------|-----------|-----------------|
| **Phase 1** | WBS-KB1-4 | LLM Discussion Foundation | Multiple LLMs debate and reach consensus |
| **Phase 2** | WBS-KB5-7 | Grounding & Validation | All claims validated against sources + code tools |
| **Phase 3** | WBS-KB8-9 | Production Readiness | VS Code integration, works alongside Copilot |

### Phase 1 Completion Criteria
> "I can run a query and see multiple LLMs discuss, request more info, and eventually agree on an answer."

### Phase 2 Completion Criteria  
> "Every claim in the answer has a citation I can verify. Hallucinations are rejected."

### Phase 3 Completion Criteria
> "I can use this in VS Code instead of (or with) Copilot for complex code questions."

---

## Risk Mitigation

| Risk | Mitigation | WBS Block |
|------|-----------|-----------|
| LLMs never agree | Set max_cycles, fallback to highest-confidence response | WBS-KB4 |
| Information requests too vague | Prompt engineering + structured output format | WBS-KB2 |
| Evidence retrieval too slow | Parallel retrieval, caching | WBS-KB3, WBS-AGT24 |
| Citation validation bottleneck | Batch validation, async audit calls | WBS-KB5 |
| MCP protocol changes | Abstract MCP layer, version pinning | WBS-KB8 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial WBS based on KITCHEN_BRIGADE_ARCHITECTURE.md |
| 1.1.0 | 2025-12-31 | Added "Relationship to WBS.md" section; clarified dependencies on WBS-AGT17, WBS-AGT20, WBS-AGT21-24; updated WBS-KB9 to show extension of WBS-AGT20 |

---

## Appendix A: Key Insights from Cross-Reference Pipeline Demonstration

The following insights were learned through the actual cross-reference process that this WBS implements:

1. **Iteration is Essential** — Single-pass retrieval is insufficient. LLMs need multiple cycles to refine understanding.

2. **Disagreement Drives Discovery** — When LLMs disagree, the system should gather more evidence, not just pick a winner.

3. **All Claims Need Citations** — Uncited claims are hallucination candidates. audit-service must validate before delivery.

4. **Tools Provide Objectivity** — CodeT5+, GraphCodeBERT, SonarQube provide objective validation that LLMs cannot self-assess.

5. **Provenance is Trackable** — Every claim should trace back: claim → participant → cycle → source → line number.

6. **The Process IS the Product** — The conversation we had to design this architecture demonstrated exactly how the agents should behave.
