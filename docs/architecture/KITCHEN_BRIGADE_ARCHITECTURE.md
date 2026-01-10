# Kitchen Brigade 2.0 Architecture

## Complete Agent-Service Integration for LLM-Powered Code Understanding

**Version:** 2.1  
**Last Updated:** 2026-01-07  
**Status:** Living Document

---

## Implementation Status (January 7, 2026)

| Component | Location | Status |
|-----------|----------|--------|
| **Infrastructure Config** | `src/infrastructure_config.py` | ✅ Complete |
| **Protocol Executor** | `src/protocols/kitchen_brigade_executor.py` | ✅ Complete |
| **Workflow Composer** | `src/protocols/workflow_composer.py` | ✅ Complete |
| **Protocol Definitions** | `config/protocols/*.json` | ✅ 5 protocols |
| **Prompt Templates** | `config/prompts/kitchen_brigade/*.txt` | ✅ 16 templates |
| **Brigade Recommendations** | `config/brigade_recommendations.yaml` | ✅ Complete |
| **Agent Guide** | `docs/KITCHEN_BRIGADE_AGENT_GUIDE.md` | ✅ Complete |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Principles](#core-principles)
3. [Infrastructure Modes](#infrastructure-modes)
4. [The 8 Agent Functions](#the-8-agent-functions)
5. [Cross-Reference Pipeline (Iterative)](#cross-reference-pipeline-iterative)
6. [Agent → Tool/Service Mapping](#agent--toolservice-mapping)
7. [Output Flow Architecture](#output-flow-architecture)
8. [Protected Configurations](#protected-configurations)
9. [Complete Flow Example](#complete-flow-example)
10. [Kitchen Brigade Metaphor](#kitchen-brigade-metaphor)

---

## Executive Summary

The Kitchen Brigade architecture orchestrates 8 agent functions that interact with specialized services (Code-Orchestrator, semantic-search, inference-service, audit-service) to answer complex queries about code understanding. 

**Key Innovation:** The cross-reference process is **iterative and multi-loop** — LLMs actively discuss, request additional information, and refine their understanding through multiple cycles before producing a grounded, validated response.

**January 2026 Updates:**
- Infrastructure-aware endpoint resolution via `infrastructure_config.py`
- Multi-LLM protocol executor with Stage 2 cross-reference integration
- Workflow composer for chaining multiple protocols
- Brigade tier system (local_only, balanced, premium)

### Core Use Case
> "Design a scalable LLM-powered code understanding system for a 20M-line monorepo"
> 
> Focus: multi-stage chunking, embeddings + hierarchical retrieval, indexing strategies, incremental refresh pipeline, grounding LLM outputs, hallucination-hardening

---

## Infrastructure Modes

The protocol executor dynamically resolves service endpoints based on deployment mode:

| Mode | Set Via | Service URLs | Use Case |
|------|---------|--------------|----------|
| **docker** | `INFRASTRUCTURE_MODE=docker` | Docker DNS (e.g., `llm-gateway:8080`) | Full containerized deployment |
| **hybrid** | `INFRASTRUCTURE_MODE=hybrid` | localhost (e.g., `localhost:8080`) | Development: DBs in Docker, services native |
| **native** | `INFRASTRUCTURE_MODE=native` | localhost (e.g., `localhost:8080`) | Fully native development |

**Configuration Source**: `src/infrastructure_config.py` implements `PlatformConfig` dataclass with:
- Service URLs (llm-gateway, semantic-search, code-orchestrator, inference-service, audit-service)
- Database URLs (Qdrant, Neo4j, Redis)
- Data paths (textbooks, books_enriched, books_metadata)
- Credentials (Neo4j user/password)

---

## Core Principles

### 1. Agents Are Stateless Executors
```
"Agents do not remember, do not chat, do not accumulate context.
 They read from caches and write new state back."
```

### 2. Composition Over Spawning
- Agents do **NOT** create sub-agents at runtime
- Composition happens via `SequentialAgent`, `ParallelAgent`, `LoopAgent`
- Pipeline Orchestrator (not agents) executes the DAG

### 3. Tools Are Service Calls
- Agents USE tools (query Qdrant, search Neo4j, call LLM)
- Tools abstract service complexity
- Agents select tools, services handle execution

### 4. Iterative Refinement
- Cross-referencing is NOT a single-pass operation
- LLMs request additional information through multiple loops
- Validation gates trigger retries when needed

---

## The 8 Agent Functions

| Function | Purpose | Default Preset | Primary Tools |
|----------|---------|----------------|---------------|
| `extract_structure` | Extract keywords, concepts, entities, outline | S1 | keyword_extraction, ast_parser |
| `summarize_content` | Compress while preserving invariants | D4 | tokenizer, llm_compress |
| `generate_code` | Generate code from spec + context | D4 | pattern_lookup, cross_reference |
| `analyze_artifact` | Analyze for patterns, issues, quality | D4 | sonarqube, term_validator |
| `validate_against_spec` | Check against criteria/constraints | D4 | citation_validator, llm_critique |
| `synthesize_outputs` | Combine multiple artifacts | S1 | conflict_resolver, provenance_tracker |
| `decompose_task` | Break task into subtasks | S2 | agent_registry, dependency_analyzer |
| `cross_reference` | Find related content across sources | S4 | semantic_search, code_search, textbook_search |

### Sufficiency Analysis for Use Case

| Requirement | Agent Function | How It Handles |
|-------------|----------------|----------------|
| Multi-stage chunking | `extract_structure` | Extracts outline, concepts by AST/semantic units |
| Embeddings + retrieval | `cross_reference` | Semantic search via Qdrant, Neo4j tools |
| Indexing strategies | `decompose_task` + `generate_code` | Breaks into subtasks, generates schemas |
| Incremental refresh | `analyze_artifact` + `generate_code` | Analyzes diffs, generates refresh logic |
| Grounding outputs | `cross_reference` + audit-service | Citations with Chicago-style footnotes |
| Hallucination-hardening | `validate_against_spec` | LLM critique + citation validation |

**Verdict:** ✅ The 8 agent functions ARE sufficient. Gaps are in TOOLS, not agents.

---

## Cross-Reference Pipeline (Iterative)

### The Living Example

This pipeline was demonstrated in our conversation — what I did IS what the agents should do:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     USER QUERY (Ambiguous/Complex)                          │
│   "Do agents create sub-agents? Industry says yes, your docs say no."       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: decompose_task                                                    │
│  ───────────────────────                                                    │
│  Extract: "sub-agent patterns", "parallel processing", "ADK architecture"   │
│  Output: Search terms + source priorities                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: cross_reference_full (Parallel 5-Source Search)                   │
│  ─────────────────────────────────────────────────────────                  │
│                                                                             │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │  QDRANT     │ │  NEO4J      │ │  TEXTBOOKS  │ │  CODE       │          │
│   │  (chapters) │ │  (graph)    │ │  (JSON)     │ │  ORCHESTR.  │          │
│   │             │ │             │ │             │ │             │          │
│   │  Textbook   │ │  "What      │ │  "AI Agents │ │  ML-based   │          │
│   │  chapters   │ │  relates?"  │ │  In Action" │ │  code       │          │
│   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │
│          │               │               │               │                  │
│   ┌──────┴───────────────┴───────────────┴───────────────┴───────┐         │
│   │                                                              │         │
│   │  ┌─────────────────────────────────────────────────────┐     │         │
│   │  │  CODE_CHUNKS (Qdrant collection)                    │     │         │
│   │  │  ─────────────────────────────────                  │     │         │
│   │  │  Actual GitHub code snippets with:                  │     │         │
│   │  │  • content (source code)                            │     │         │
│   │  │  • repo_url, file_path, start_line, end_line        │     │         │
│   │  │  • domain, concepts, patterns                       │     │         │
│   │  │  → Citable as [^N] with GitHub permalink            │     │         │
│   │  └──────────────────────────────────┬──────────────────┘     │         │
│   │                                     │                        │         │
│   └─────────────────────────────────────┼────────────────────────┘         │
│                                         │                                   │
│                          Promise.all (5 parallel)                           │
│                                         │                                   │
│                                         ▼                                   │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │  Results: [qdrant_hits, neo4j_relations, textbook_excerpts,     │      │
│   │            code_orchestrator_results, code_chunks]              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM DISCUSSION LOOP (ITERATIVE - Multiple Cycles)                 │
│  ──────────────────────────────────────────────────────────                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CYCLE 1: Initial Analysis                                          │   │
│  │  ─────────────────────────────                                      │   │
│  │  LLM-A (qwen2.5-7b): "Architecture doc says stateless executors"    │   │
│  │  LLM-B (deepseek-r1): "But textbook mentions supervisor pattern"    │   │
│  │                                                                     │   │
│  │  DISAGREEMENT DETECTED → Request additional information             │   │
│  │  → cross_reference("ParallelAgent ADK pattern")                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CYCLE 2: Refined Analysis                                          │   │
│  │  ──────────────────────────────                                     │   │
│  │  New evidence: ai-agents/src/pipelines/agents.py (ParallelAgent)    │   │
│  │                                                                     │   │
│  │  LLM-A: "ParallelAgent uses asyncio.gather for concurrent exec"     │   │
│  │  LLM-B: "This IS the sub-agent pattern - workflow composition"      │   │
│  │                                                                     │   │
│  │  AGREEMENT DETECTED → Proceed to synthesis                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CYCLE 3: Reconciliation (if needed)                                │   │
│  │  ───────────────────────────────────                                │   │
│  │  Reconciler LLM synthesizes: "Sub-agents = workflow composition     │   │
│  │  at construction time, not runtime spawning"                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Loop continues until: agreement_score > threshold OR max_iterations       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: synthesize_outputs (Fusion/Reconciliation)                        │
│  ──────────────────────────────────────────────────                         │
│                                                                             │
│  Input: Reconciled understanding from LLM discussion                        │
│  Output: Coherent answer with tracked provenance                            │
│                                                                             │
│  "Sub-agents = workflow composition at construction time,                   │
│   not runtime spawning. ParallelAgent IS the pattern."                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: validate_against_spec (Grounding Check)                           │
│  ─────────────────────────────────────────────────                          │
│                                                                             │
│  ✓ Citations exist (ADK_MIGRATION_GUIDE.md lines 175-180)                   │
│  ✓ Code references valid (ai-agents/src/pipelines/agents.py)                │
│  ✓ Textbook excerpts match source (AI Agents and Applications Ch.12)       │
│  ✓ No hallucinated claims                                                   │
│                                                                             │
│  If validation fails → LoopAgent triggers retry from Stage 2                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: audit-service Integration                                         │
│  ──────────────────────────────────                                         │
│                                                                             │
│  POST audit-service:8084/v1/footnotes                                       │
│  → Generate Chicago-style citations                                         │
│  → Record audit trail (task_id, citations, models_used)                     │
│  → Verify source documents exist                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GROUNDED RESPONSE                                  │
│                                                                             │
│  "Your contention is VALID. The architecture supports sub-agent patterns    │
│   via ParallelAgent composition. Here's the evidence from 3 sources..."     │
│                                                                             │
│  Citations:                                                                 │
│  [^1]: textbooks/pending/platform/ADK_MIGRATION_GUIDE.md#L175               │
│  [^2]: ai-agents/src/pipelines/agents.py#L135-174                           │
│  [^3]: AI Agents and Applications, Chapter 12 "Multi-agent Systems"         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM Discussion Loop Details

The discussion loop is the **heart of cross-referencing**. It's not a single-pass operation:

```python
class LLMDiscussionLoop:
    """
    Iterative LLM discussion for cross-reference reconciliation.
    
    Key principles:
    1. Multiple LLMs analyze the same evidence
    2. Disagreements trigger additional cross_reference calls
    3. Loop continues until agreement or max iterations
    4. All cycles contribute to provenance tracking
    """
    
    def __init__(
        self,
        participants: list[LLMParticipant],
        cross_reference_tool: CrossReferenceTool,
        max_cycles: int = 5,
        agreement_threshold: float = 0.85,
    ):
        self.participants = participants
        self.cross_reference = cross_reference_tool
        self.max_cycles = max_cycles
        self.agreement_threshold = agreement_threshold
        self.discussion_history: list[DiscussionCycle] = []
    
    async def discuss(
        self, 
        initial_evidence: CrossReferenceResult,
        query: str,
    ) -> DiscussionResult:
        """
        Run iterative discussion until agreement or max cycles.
        
        Flow:
        1. All participants analyze evidence
        2. Calculate agreement score
        3. If disagreement, extract information_requests
        4. Call cross_reference for additional evidence
        5. Repeat until agreement or max_cycles
        """
        current_evidence = initial_evidence
        
        for cycle in range(self.max_cycles):
            # Phase 1: Parallel analysis by all participants
            analyses = await asyncio.gather(*[
                p.analyze(current_evidence, query) 
                for p in self.participants
            ])
            
            # Phase 2: Calculate agreement
            agreement = calculate_agreement(analyses)
            
            # Phase 3: Record cycle
            self.discussion_history.append(DiscussionCycle(
                cycle_number=cycle + 1,
                analyses=analyses,
                agreement_score=agreement.score,
                disagreement_points=agreement.disagreements,
            ))
            
            # Phase 4: Check termination condition
            if agreement.score >= self.agreement_threshold:
                return DiscussionResult(
                    consensus=synthesize_consensus(analyses),
                    confidence=agreement.score,
                    cycles_used=cycle + 1,
                    history=self.discussion_history,
                )
            
            # Phase 5: Request additional information
            information_requests = extract_information_requests(
                agreement.disagreements
            )
            
            # Phase 6: Parallel cross-reference for new evidence
            new_evidence = await asyncio.gather(*[
                self.cross_reference.search(req) 
                for req in information_requests
            ])
            
            # Phase 7: Merge new evidence
            current_evidence = merge_evidence(current_evidence, new_evidence)
        
        # Max cycles reached - return best effort
        return DiscussionResult(
            consensus=synthesize_consensus(analyses),
            confidence=agreement.score,
            cycles_used=self.max_cycles,
            history=self.discussion_history,
            max_cycles_reached=True,
        )
```

### Integration Points

| Component | Role in Discussion Loop | Service |
|-----------|------------------------|---------|
| **inference-service** | Hosts LLM participants | :8085 |
| **Code-Orchestrator** | CodeT5+/GraphCodeBERT for code analysis | :8083 |
| **semantic-search** | Qdrant/Neo4j for evidence retrieval | :8081 |
| **audit-service** | Validates citations, tracks provenance | :8084 |
| **ai-platform-data** | Textbook JSON files, repo_registry | filesystem |

---

## Agent → Tool/Service Mapping

### cross_reference Agent (Primary Example)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ cross_reference                                                             │
│        │                                                                    │
│        ├──► TOOL: semantic_search                                           │
│        │         └──► semantic-search:8081/v1/search                        │
│        │                    ├──► Qdrant (vector search)                     │
│        │                    └──► Neo4j (graph traversal)                    │
│        │                                                                    │
│        ├──► TOOL: code_search                                               │
│        │         └──► Code-Orchestrator:8083/v1/search                      │
│        │                    ├──► CodeT5+ (keyword extraction)               │
│        │                    ├──► GraphCodeBERT (term validation)            │
│        │                    └──► CodeBERT (ranking)                         │
│        │                                                                    │
│        ├──► TOOL: textbook_search                                           │
│        │         └──► ai-platform-data/textbooks_json/                      │
│        │                    └──► JSON file loader (256 files)               │
│        │                                                                    │
│        ├──► TOOL: github_fetch                                              │
│        │         └──► code-reference-engine (GitHubClient)                  │
│        │                    └──► On-demand code retrieval                   │
│        │                                                                    │
│        └──► TOOL: repo_registry                                             │
│                  └──► ai-platform-data/repos/repo_registry.json             │
│                             └──► Domain/concept/pattern lookup              │
│                                                                             │
│  The TOOL abstracts complexity. Agent says:                                 │
│  search(scope=['code','books','textbooks'], query="...")                    │
│  Tool decides which services to call based on scope.                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complete Agent → Tool Mapping

| Agent Function | Tools | Services Called |
|----------------|-------|-----------------|
| `extract_structure` | keyword_extraction, ast_parser, file_reader | Code-Orchestrator:8083 |
| `summarize_content` | tokenizer, importance_scorer, llm_compress | inference-service:8085 |
| `generate_code` | cross_reference, pattern_lookup, template_engine | inference-service:8085 |
| `analyze_artifact` | sonarqube_analyze, term_validator, complexity_analyzer | Code-Orchestrator:8083 |
| `validate_against_spec` | spec_comparator, citation_validator, llm_critique | audit-service:8084, inference:8085 |
| `synthesize_outputs` | conflict_resolver, provenance_tracker, format_converter | audit-service:8084 |
| `decompose_task` | agent_registry, capability_matcher, token_estimator | inference-service:8085 |
| `cross_reference` | semantic_search, code_search, textbook_search, github_fetch | semantic-search:8081, Code-Orchestrator:8083 |

---

## Output Flow Architecture

### Cache-Based Handoff

```
┌─────────────────────────────┐
│     AGENT OUTPUT            │
│  (Pydantic model result)    │
└─────────────┬───────────────┘
              │
      ┌───────┼───────┐
      │       │       │
      ▼       ▼       ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ TEMP     │ │ USER     │ │ APP      │
│ CACHE    │ │ CACHE    │ │ CACHE    │
│ (temp:)  │ │ (user:)  │ │ (app:)   │
│          │ │          │ │          │
│ Pipeline │ │ Session  │ │ Permanent│
│ handoff  │ │ persist  │ │ storage  │
│ Ephemeral│ │ 24h TTL  │ │ Qdrant/  │
│          │ │ Redis    │ │ Neo4j    │
└──────────┘ └──────────┘ └──────────┘
```

### Citation Flow

```
Step 1: Agent generates with citation markers
─────────────────────────────────────────────
generate_code output:
{
  "code": "class Repository(ABC):...",
  "explanation": "The Repository pattern[^1] provides...",
  "citations": [
    {"marker": "[^1]", "source_id": "ref_001", "type": "book"}
  ]
}

Step 2: audit-service formats footnotes
───────────────────────────────────────
POST audit-service:8084/v1/footnotes

Response:
{
  "footnotes": {
    "[^1]": "Fowler, Martin, *Patterns of Enterprise Application Architecture*..."
  }
}

Step 3: Final response to VS Code
─────────────────────────────────
{
  "content": "The Repository pattern[^1]...\n\n---\n\n[^1]: Fowler, Martin...",
  "metadata": {
    "citations_used": 1,
    "models_used": ["qwen2.5-7b"],
    "confidence": 0.87
  }
}
```

---

## Protected Configurations

### What Agents CANNOT Modify

| Configuration Type | Location | Who Can Modify |
|-------------------|----------|----------------|
| LLM Model Loading | inference-service/config/models.yaml | Admin only (restart required) |
| Preset Definitions | inference-service/config/presets.yaml | Admin only |
| Qdrant Collections | semantic-search-service/config/ | Admin only (seeder scripts) |
| Neo4j Schema | semantic-search-service/graph/schema.py | Admin only (migrations) |
| HuggingFace Models | Code-Orchestrator/models/registry.py | Admin only (pre-loaded) |
| Pipeline Definitions | ai-agents/config/pipelines.yaml | Admin only |
| SonarQube Rules | sonar-project.properties | Admin only |

### What Agents CAN Do

- Invoke tools (query Qdrant, search Neo4j, call LLM, read files)
- Pass parameters to tools (top_k, query, scope)
- Select presets by NAME ("D4", "S1")
- Read from and write to cache (temp:, user:, app:)
- Request additional cross-reference cycles

**Principle:** Agents are OPERATORS, not ADMINISTRATORS. They use equipment at current settings; they don't reconfigure the kitchen.

---

## Complete Flow Example

### Use Case: "Design LLM-Powered Code Understanding System"

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  VS Code Extension                                                         │
│        │                                                                   │
│        │ POST /chat/completions                                            │
│        │ { "message": "Design a scalable LLM-powered code..." }            │
│        ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ llm-gateway:8080 (ROUTER)                                            │  │
│  │ • Auth check ✓                                                       │  │
│  │ • Rate limit check ✓                                                 │  │
│  │ • Query classification: "architecture_design" → route to ai-agents   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ai-agents:8082 (EXPEDITOR - Pipeline Orchestrator)                   │  │
│  │                                                                      │  │
│  │ Pipeline: architecture-design                                        │  │
│  │                                                                      │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 1: decompose_task                                        ║   │  │
│  │ ║ Output: 6 subtasks with dependencies                           ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                         │                                            │  │
│  │                         ▼                                            │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 2: ParallelAgent(cross_reference × 5)                    ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ For each subtask, runs 4-layer parallel retrieval:             ║   │  │
│  │ ║ • Qdrant (vectors) - semantic similarity search                ║   │  │
│  │ ║ • Neo4j (graph) - relationship traversal                       ║   │  │
│  │ ║ • Textbooks (JSON) - reference material lookup                 ║   │  │
│  │ ║ • Code-Orchestrator (Full ML Stack):                           ║   │  │
│  │ ║   - SBERT: NL→semantic embeddings, similar chapters            ║   │  │
│  │ ║   - CodeT5+: keyword extraction from code                      ║   │  │
│  │ ║   - GraphCodeBERT: term validation, false positive filtering   ║   │  │
│  │ ║   - CodeBERT: NL↔Code ranking, relevance scoring               ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                         │                                            │  │
│  │                         ▼                                            │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 3: LLM Discussion Loop (ITERATIVE)                       ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ CYCLE 1: Initial analysis                                      ║   │  │
│  │ ║   LLM-A: Analyzes evidence from cross_reference                ║   │  │
│  │ ║   LLM-B: Critiques and identifies gaps                         ║   │  │
│  │ ║   → Disagreement on chunking strategy                          ║   │  │
│  │ ║   → Request: cross_reference("AST vs semantic chunking")       ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ CYCLE 2: Refined analysis                                      ║   │  │
│  │ ║   New evidence: code-reference-engine patterns                 ║   │  │
│  │ ║   LLM-A: "Hybrid approach - AST for structure, semantic..."    ║   │  │
│  │ ║   LLM-B: "Agrees, cites Building Microservices Ch.12"          ║   │  │
│  │ ║   → Agreement score: 0.91 > threshold                          ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ CYCLE 3: Reconciliation                                        ║   │  │
│  │ ║   Synthesize consensus from both analyses                      ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                         │                                            │  │
│  │                         ▼                                            │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 4: SequentialAgent(generate_code × 5)                    ║   │  │
│  │ ║ Generate architecture sections from consensus                  ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                         │                                            │  │
│  │                         ▼                                            │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 5: synthesize_outputs                                    ║   │  │
│  │ ║ Merge 5 sections into coherent document                        ║   │  │
│  │ ║ Track provenance: which source said what                       ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                         │                                            │  │
│  │                         ▼                                            │  │
│  │ ╔════════════════════════════════════════════════════════════════╗   │  │
│  │ ║ STAGE 6: validate_against_spec                                 ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ CHECKS:                                                        ║   │  │
│  │ ║ ✓ All 6 focus areas addressed?                                 ║   │  │
│  │ ║ ✓ Citations traceable to sources?                              ║   │  │
│  │ ║ ✓ No hallucinated claims?                                      ║   │  │
│  │ ║ ✓ Code samples syntactically valid?                            ║   │  │
│  │ ║                                                                ║   │  │
│  │ ║ If !valid → LoopAgent triggers retry from Stage 3              ║   │  │
│  │ ╚════════════════════════════════════════════════════════════════╝   │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ audit-service:8084 (AUDITOR)                                         │  │
│  │                                                                      │  │
│  │ • Generate Chicago-style footnotes                                   │  │
│  │ • Verify all cited sources exist                                     │  │
│  │ • Record audit trail                                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ FINAL RESPONSE                                                       │  │
│  │                                                                      │  │
│  │ {                                                                    │  │
│  │   "content": "# LLM-Powered Code Understanding System\n\n...",       │  │
│  │   "metadata": {                                                      │  │
│  │     "pipeline": "architecture-design",                               │  │
│  │     "stages_completed": 6,                                           │  │
│  │     "discussion_cycles": 3,                                          │  │
│  │     "models_used": ["deepseek-r1-7b", "qwen2.5-7b", "phi-4"],        │  │
│  │     "citations_count": 18,                                           │  │
│  │     "confidence": 0.94,                                              │  │
│  │     "sources": { "books": 6, "code": 8, "textbooks": 4 }             │  │
│  │   }                                                                  │  │
│  │ }                                                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Kitchen Brigade Metaphor

| Concept | Kitchen Equivalent | Implementation |
|---------|-------------------|----------------|
| **Microservices** | Kitchen equipment (ovens, mixers) | Pre-configured, staff can't rewire |
| **Model wrappers** | Specialized attachments (pasta maker) | CodeT5+, SBERT attached to equipment |
| **Tools** | Utensils (knives, spatulas) | Agents pick up and use them |
| **Agent Functions** | Kitchen staff roles (8 chefs) | Defined specialties |
| **Pipelines** | Recipes | Sequence of steps using multiple chefs |
| **Presets** | Cooking modes ("sauté", "simmer") | Staff select mode, can't rewire stove |
| **Discussion Loop** | Kitchen meeting | Chefs discuss, request more ingredients |
| **Audit Service** | Quality inspector | Validates final dish, checks sources |

---

## New Tools Needed

Based on this architecture, the following tools need implementation:

| Tool | Agent User | Service | Status |
|------|------------|---------|--------|
| `textbook_search` | cross_reference | JSON file loader | 🔴 Not implemented |
| `code_reference` | cross_reference | CodeReferenceEngine | 🟡 Exists, not wired |
| `ast_parser` | extract_structure | Code-Orchestrator | 🔴 Not implemented |
| `template_engine` | generate_code | Jinja2 | 🔴 Not implemented |
| `discussion_loop` | cross_reference | inference-service | 🔴 Not implemented |

---

## References

- [AGENT_FUNCTIONS_ARCHITECTURE.md](../../textbooks/pending/platform/AGENT_FUNCTIONS_ARCHITECTURE.md)
- [ADK_MIGRATION_GUIDE.md](../../textbooks/pending/platform/ADK_MIGRATION_GUIDE.md)
- [ai-agents/src/pipelines/agents.py](../src/pipelines/agents.py) - ParallelAgent, SequentialAgent, LoopAgent
- [inference-service/src/orchestration/modes/debate.py](../../inference-service/src/orchestration/modes/debate.py) - asyncio.gather pattern
- [code-reference-engine/docs/CODE_REFERENCE_ENGINE_SETUP.md](../../code-reference-engine/docs/CODE_REFERENCE_ENGINE_SETUP.md) - 3-layer retrieval

---

*This document synthesizes the cross-reference pipeline demonstration and integrates it with the Kitchen Brigade architecture. The key insight: cross-referencing is an iterative, multi-loop process where LLMs actively discuss, request additional information, and refine their understanding before producing a grounded response.*
