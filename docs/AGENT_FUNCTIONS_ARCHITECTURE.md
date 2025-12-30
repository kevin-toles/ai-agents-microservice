# Agent Functions Architecture

> **Version:** 1.2.0  
> **Created:** 2025-12-29  
> **Updated:** 2025-12-29  
> **Status:** Design Phase  
> **Reference:** [ARCHITECTURE.md](ARCHITECTURE.md), [inference-service/MODEL_LIBRARY.md](../../inference-service/docs/MODEL_LIBRARY.md), [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md), [PROTOCOL_INTEGRATION_ARCHITECTURE.md](PROTOCOL_INTEGRATION_ARCHITECTURE.md)

## Overview

This document defines the **Agent Function** architecture for the AI Platform. Agent functions are **stateless executors over cached artifacts**—not chat personas or conversational agents.

### Design Philosophy

> "Agents do not remember, do not chat, do not accumulate context. They read from caches and write new state back."

This philosophy aligns with Google's Agent Development Kit (ADK) pattern of **shared session state** and **explicit state management**. While we maintain our stateless executor model, we adopt ADK's state prefix conventions and artifact patterns for consistency with industry standards.

| Concept | Traditional Agent | Agent Function (Our Model) | ADK Alignment |
|---------|-------------------|---------------------------|---------------|
| **State** | Accumulated memory | Stateless; reads from cache | ✅ Shared session state |
| **Identity** | Persona ("helpful assistant") | Function signature | ✅ Tool-based agents |
| **Context** | Growing conversation | Fixed input schema | ✅ Explicit context injection |
| **Output** | Chat response | Typed artifact + metadata | ✅ Artifacts service |
| **Composition** | Nested calls | Pipeline DAG | ✅ Workflow agents |

> **Architecture Evolution:**
> - **Phase 1 (This Document):** Agent Functions + ADK Patterns
> - **Phase 2:** Protocol Integration (A2A + MCP) - See [PROTOCOL_INTEGRATION_ARCHITECTURE.md](PROTOCOL_INTEGRATION_ARCHITECTURE.md)
> - **Phase 3:** Full ADK Migration - See [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [ADK Pattern Integration (Option C)](#adk-pattern-integration-option-c)
3. [Agent Functions](#agent-functions)
4. [Pipeline Composition](#pipeline-composition)
5. [Citation Flow](#citation-flow)
6. [Preset Selection](#preset-selection)
7. [Pydantic Schemas](#pydantic-schemas)
8. [Integration Points](#integration-points)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI PLATFORM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   External Client (VS Code, Web, CLI)                                       │
│                     │                                                        │
│                     ▼                                                        │
│   ┌─────────────────────────────────────┐                                   │
│   │         llm-gateway :8080           │  ◀── Single entry point          │
│   │         (Router / Auth)             │      for all external requests    │
│   └─────────────────┬───────────────────┘                                   │
│                     │                                                        │
│   ┌─────────────────┼───────────────────────────────────────────────┐       │
│   │                 ▼           INTERNAL NETWORK                    │       │
│   │   ┌─────────────────────────────────┐                           │       │
│   │   │       ai-agents :8082           │  ◀── Expeditor            │       │
│   │   │       (Workflow Orchestrator)   │      Pipeline execution   │       │
│   │   └────────┬─────────────┬──────────┘                           │       │
│   │            │             │                                      │       │
│   │   ┌────────▼────────┐    │                                      │       │
│   │   │ semantic-search │    │                                      │       │
│   │   │ :8081 (Cookbook)│    │                                      │       │
│   │   └────────┬────────┘    │                                      │       │
│   │            │             │                                      │       │
│   │   ┌────────▼────────┐    │                                      │       │
│   │   │   Qdrant/Neo4j  │    │                                      │       │
│   │   │   (Indexes)     │    │                                      │       │
│   │   └─────────────────┘    │                                      │       │
│   │            ▲             │                                      │       │
│   │            │             │                                      │       │
│   │   ┌────────┴────────┐    │                                      │       │
│   │   │ ai-platform-data│    │                                      │       │
│   │   │ code-ref-engine │    │  (indexed at startup)                │       │
│   │   │ textbooks/      │    │                                      │       │
│   │   └─────────────────┘    │                                      │       │
│   │                          │                                      │       │
│   │   ┌──────────────────────▼──────────┐                           │       │
│   │   │   Code-Orchestrator :8083       │  ◀── Sous Chef            │       │
│   │   │   (HuggingFace models)          │      CodeT5+, GraphCodeBERT│       │
│   │   └─────────────────┬───────────────┘                           │       │
│   │                     │                                           │       │
│   │   ┌─────────────────▼───────────────┐                           │       │
│   │   │   inference-service :8085       │  ◀── Line Cook            │       │
│   │   │   (GGUF models via llama.cpp)   │      8 local LLMs         │       │
│   │   └─────────────────┬───────────────┘                           │       │
│   │                     │                                           │       │
│   │   ┌─────────────────▼───────────────┐                           │       │
│   │   │   audit-service :8084           │  ◀── Auditor              │       │
│   │   │   (Citation tracking)           │      Footnote generation  │       │
│   │   └─────────────────────────────────┘                           │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ADK Pattern Integration (Option C)

This section documents the **cherry-picked patterns** from Google's Agent Development Kit (ADK) that enhance our existing architecture without requiring full framework adoption. For complete ADK migration path, see [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md).

### State Prefix Conventions

ADK uses prefix conventions for state management. We adopt these for our caching tiers:

| ADK Prefix | ADK Purpose | Platform Mapping | Cache Tier |
|------------|-------------|------------------|------------|
| `temp:` | Current invocation only | `handoff_cache` | Pipeline-local |
| `user:` | Cross-session persistence | `compression_cache` | Redis (24h TTL) |
| `app:` | Application-wide shared | `artifact_store` | Qdrant/Neo4j |

**Implementation Pattern:**

```python
# State key conventions aligned with ADK prefixes
STATE_PREFIX_TEMP = "temp:"      # Pipeline handoff, discarded after pipeline completes
STATE_PREFIX_USER = "user:"      # User session state, persists across invocations
STATE_PREFIX_APP = "app:"        # Application-wide artifacts, permanent storage

def build_cache_key(prefix: str, namespace: str, key: str) -> str:
    """Build cache key using ADK prefix conventions.
    
    Args:
        prefix: One of STATE_PREFIX_TEMP, STATE_PREFIX_USER, STATE_PREFIX_APP
        namespace: Agent function name or pipeline ID
        key: Unique identifier within namespace
    
    Returns:
        Formatted cache key: "{prefix}{namespace}:{key}"
    
    Example:
        >>> build_cache_key(STATE_PREFIX_TEMP, "extract_structure", "chapter_1")
        "temp:extract_structure:chapter_1"
    """
    return f"{prefix}{namespace}:{key}"
```

### Artifact Conventions

ADK's Artifact service provides versioned binary storage. Our platform maps this to existing caches:

| Artifact Type | ADK Pattern | Platform Implementation |
|---------------|-------------|-------------------------|
| **Binary Blobs** | `artifacts.save_artifact(name, data)` | Redis binary keys |
| **Versioned Files** | `filename="{artifact}_v{version}"` | Git-backed `ai-platform-data/` |
| **Namespaced Access** | `{user_id}/{artifact_name}` | `{pipeline_id}/{stage_id}:{artifact}` |

**Artifact Versioning Pattern:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Artifact:
    """ADK-aligned artifact with version tracking.
    
    Follows ADK's artifact pattern for versioned binary storage
    while integrating with platform's cache hierarchy.
    """
    name: str
    namespace: str                    # Pipeline or agent function ID
    version: int                      # Auto-incremented
    mime_type: str                    # e.g., "application/json", "text/plain"
    data: bytes
    created_at: datetime
    metadata: dict[str, Any]          # Additional context (source, checksum)
    
    @property
    def qualified_name(self) -> str:
        """Return ADK-style qualified artifact name."""
        return f"{self.namespace}/{self.name}_v{self.version}"
    
    @property
    def cache_key(self) -> str:
        """Return platform cache key with app: prefix."""
        return f"app:artifact:{self.qualified_name}"
```

### AgentTool Pattern

ADK's `AgentTool` wraps agents as callable tools. Our agent functions already follow this pattern:

| ADK Pattern | Platform Equivalent | Notes |
|-------------|---------------------|-------|
| `AgentTool(agent=sub_agent)` | `agent_function.run(input)` | Agent as tool |
| `tool.run(tool_context)` | `POST /v1/functions/{name}/run` | HTTP callable |
| `output_key="result"` | `output: TypedDict` | Structured return |

**Our agent functions are already ADK-compatible tools:**

```python
# ADK AgentTool pattern
class AgentTool:
    def __init__(self, agent: Agent):
        self.agent = agent
    
    def run(self, tool_context: ToolContext) -> Any:
        return self.agent.execute(tool_context.input)

# Our equivalent pattern (already implemented)
class AgentFunction(ABC):
    """Base class for agent functions - ADK AgentTool compatible."""
    
    @abstractmethod
    def run(self, input: BaseModel, **kwargs: Any) -> BaseModel:
        """Execute agent function. Uses **kwargs for ABC flexibility.
        
        Follows CODING_PATTERNS_ANALYSIS.md ABC signature pattern
        to allow subclasses with different parameter needs.
        """
        ...
```

### Workflow Agent Mapping

ADK's workflow agents map to our pipeline composition:

| ADK Workflow Agent | Platform Equivalent | Use Case |
|--------------------|---------------------|----------|
| `SequentialAgent` | Pipeline stages in order | Chapter summarization |
| `ParallelAgent` | Parallel cross_reference calls | Multi-repo search |
| `LoopAgent` | Retry with feedback | Validation → regenerate |

**Current implementation uses explicit pipeline DAGs:**

```yaml
# Platform pipeline (equivalent to ADK SequentialAgent)
pipeline:
  stages:
    - extract_structure    # Stage 1
    - cross_reference      # Stage 2 (can parallelize)
    - summarize_content    # Stage 3
    - validate_against_spec # Stage 4 (LoopAgent pattern: retry if !valid)
```

> **Note:** Full ADK workflow agent adoption is documented in [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md). Current implementation maintains explicit control over pipeline execution for observability.

### Context Management

ADK's context injection aligns with our `context_budget` pattern:

| ADK Concept | Platform Implementation |
|-------------|-------------------------|
| `ReadonlyContext` | Immutable input schemas |
| `InvocationContext` | Pipeline execution context |
| `tool_context.state` | `handoff_cache` for inter-stage data |

**Context Budget Enforcement:**

```python
CONTEXT_BUDGET_DEFAULTS = {
    "extract_structure": {"input": 16384, "output": 2048},
    "summarize_content": {"input": 8192, "output": 4096},
    "generate_code": {"input": 4096, "output": 8192},
    "analyze_artifact": {"input": 16384, "output": 2048},
    "validate_against_spec": {"input": 4096, "output": 1024},
    "synthesize_outputs": {"input": 8192, "output": 4096},
    "decompose_task": {"input": 4096, "output": 2048},
    "cross_reference": {"input": 2048, "output": 4096},
}
```

---

## Agent Functions

### Function Registry

| Function | Purpose | Default Preset | Tools |
|----------|---------|----------------|-------|
| `extract_structure` | Extract structured data from content | S1 | none |
| `summarize_content` | Compress while preserving invariants | D4 | none |
| `generate_code` | Generate code from spec + context | D4 | cross_reference |
| `analyze_artifact` | Analyze for patterns/issues/quality | D4 | sonarqube |
| `validate_against_spec` | Check artifact against criteria | D4 | none |
| `synthesize_outputs` | Combine multiple artifacts | S1 | none |
| `decompose_task` | Break task into subtasks | S2 | none |
| `cross_reference` | Find related content | S4 | semantic_search |

---

### 1. `extract_structure`

**Purpose:** Extract structured data from unstructured content.

```yaml
extract_structure:
  description: "Extract keywords, concepts, entities, or outline from raw content"
  
  input:
    content: str              # Raw text/code
    extraction_type: enum     # keywords | concepts | entities | outline
    domain: str               # ai-ml | systems | web | general
  
  output:
    extracted: list[dict]     # Structured items with confidence scores
    raw_positions: list       # Source locations for audit
    compressed_summary: str   # Max 500 tokens for downstream
  
  presets:
    default: S1               # phi-4 for general extraction
    code_heavy: S6            # granite-8b-code-128k
    long_input: S5            # phi-3-medium-128k
  
  context_budget:
    input: 4096 tokens
    output: 1024 tokens
```

---

### 2. `summarize_content`

**Purpose:** Compress content while preserving invariants.

```yaml
summarize_content:
  description: "Compress text to target size while preserving key facts"
  
  input:
    content: str
    target_tokens: int        # Compression target
    preserve: list[str]       # Must-include concepts
    style: enum               # technical | executive | bullets
  
  output:
    summary: str              # Human-readable
    invariants: list[str]     # Key facts preserved (for validation)
    compression_ratio: float
    audit_hash: str           # For cache key
  
  presets:
    short_input: S4           # llama-3.2-3b (fast)
    long_input: S5            # phi-3-medium-128k
    technical: D4             # deepseek + qwen (critique)
  
  context_budget:
    input: variable
    output: target_tokens
```

---

### 3. `generate_code`

**Purpose:** Generate code from specification + context.

```yaml
generate_code:
  description: "Generate code artifacts from spec with context awareness"
  
  input:
    specification: str        # What to build
    context_artifacts: list   # From handoff cache
    language: str
    patterns_to_follow: list  # From CODING_PATTERNS_ANALYSIS
    constraints: list         # Must-haves
  
  output:
    code: str
    explanation: str          # Why these choices
    test_hints: list[str]     # Suggested test cases
    compressed_intent: str    # For downstream validation
    citations: list[Citation] # Sources used
  
  presets:
    simple: S3                # qwen2.5-7b solo
    quality: D4               # qwen (gen) + deepseek (critique)
    long_file: S6             # granite-8b-code-128k
  
  tools:
    - cross_reference
    - semantic_search
  
  context_budget:
    input: 8192 tokens
    output: 4096 tokens
```

---

### 4. `analyze_artifact`

**Purpose:** Analyze code/document for patterns, issues, quality.

```yaml
analyze_artifact:
  description: "Analyze artifact for quality, security, patterns"
  
  input:
    artifact: str             # Code or document
    artifact_type: enum       # code | document | config
    analysis_type: enum       # quality | security | patterns | dependencies
    checklist: list[str]      # Optional specific checks
  
  output:
    findings: list[Finding]   # Issues with severity, location, fix hint
    metrics: dict             # CC, LOC, etc. for code
    pass: bool                # Overall gate
    compressed_report: str    # For downstream
  
  presets:
    code: D4                  # Think + Code critique
    security: D3              # Debate for high-stakes
    quick: S3                 # Qwen solo for speed
  
  tools:
    - sonarqube_analyze_file (if code)
  
  context_budget:
    input: 16384 tokens
    output: 2048 tokens
```

---

### 5. `validate_against_spec`

**Purpose:** Check artifact against criteria/constraints.

```yaml
validate_against_spec:
  description: "Validate artifact meets specification and criteria"
  
  input:
    artifact: str
    specification: str        # Original requirement
    invariants: list[str]     # From upstream summarize_content
    acceptance_criteria: list
  
  output:
    valid: bool
    violations: list[Violation]
    confidence: float         # 0.0-1.0
    remediation_hints: list
  
  presets:
    default: D4               # Critique mode
    high_stakes: D3           # Debate for critical
  
  context_budget:
    input: 4096 tokens each
    output: 1024 tokens
```

---

### 6. `synthesize_outputs`

**Purpose:** Combine multiple artifacts into coherent result.

```yaml
synthesize_outputs:
  description: "Combine multiple outputs into coherent result"
  
  input:
    artifacts: list[dict]     # Multiple outputs
    synthesis_strategy: enum  # merge | reconcile | vote
    conflict_policy: enum     # first_wins | consensus | flag
  
  output:
    synthesized: str
    agreement_score: float
    conflicts: list[Conflict]
    provenance: dict          # Which artifact contributed what
  
  presets:
    default: S1               # phi-4 for general
    code: D4                  # Code-aware synthesis
    debate_reconciliation: D3
  
  context_budget:
    input: 8192 tokens
    output: 4096 tokens
```

---

### 7. `decompose_task`

**Purpose:** Break complex task into executable subtasks.

```yaml
decompose_task:
  description: "Decompose complex task into subtask DAG"
  
  input:
    task: str                 # High-level objective
    constraints: list         # Time, scope, dependencies
    available_agents: list    # Which agent functions exist
    context: str              # Domain/project context
  
  output:
    subtasks: list[Subtask]   # Ordered execution plan
    dependencies: DAG         # Which subtasks depend on which
    agent_assignments: dict   # subtask_id -> agent_function
    estimated_tokens: int     # Total budget estimate
  
  presets:
    default: S2               # DeepSeek for chain-of-thought
    complex: D3               # Debate for architecture
  
  context_budget:
    input: 4096 tokens
    output: 2048 tokens
```

---

### 8. `cross_reference`

**Purpose:** Find related content across knowledge bases.

```yaml
cross_reference:
  description: "Find related content via semantic search"
  
  input:
    query_artifact: str       # Source content
    search_scope: list        # Which repositories
    match_type: enum          # semantic | keyword | hybrid
    top_k: int
  
  output:
    references: list[Reference]
    similarity_scores: list
    compressed_context: str   # For downstream
    citations: list[Citation] # For footnotes
  
  presets:
    ranking: S4               # LLM for post-ranking only
  
  tools:
    - semantic_search (via semantic-search-service)
    - keyword_extraction (via Code-Orchestrator)
  
  context_budget:
    input: 2048 tokens
    output: 4096 tokens
```

---

## Pipeline Composition

### Chapter Summarization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                Chapter Summarization Pipeline                    │
│                Pipeline ID: chapter-summarization                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: extract_structure                                     │
│  ├── Input: chapter_text                                        │
│  ├── Preset: S5 (phi-3-medium-128k for long chapters)          │
│  ├── Output: {keywords, concepts, outline}                      │
│  └── Cache: handoff_cache[stage_1_output]                      │
│                     │                                           │
│                     ▼                                           │
│  Stage 2: cross_reference                                       │
│  ├── Input: keywords + concepts from Stage 1                    │
│  ├── Tools: semantic_search → Qdrant/Neo4j                     │
│  ├── Output: {related_chapters, similarity_scores, citations}   │
│  └── Cache: handoff_cache[stage_2_output]                      │
│                     │                                           │
│                     ▼                                           │
│  Stage 3: summarize_content                                     │
│  ├── Input: chapter_text + extracted structure + citations     │
│  ├── Preset: D4 (critique mode for quality)                    │
│  ├── Output: {summary, invariants, footnotes}                  │
│  └── Cache: compression_cache[stage_3_output]                  │
│                     │                                           │
│                     ▼                                           │
│  Stage 4: validate_against_spec                                 │
│  ├── Input: summary + original outline (as spec)               │
│  ├── Preset: D4 (critique)                                     │
│  ├── Output: {valid, violations}                               │
│  └── If !valid → Retry Stage 3 with feedback                   │
│                     │                                           │
│                     ▼                                           │
│  Final: Emit to llm-document-enhancer                          │
│  ├── summary (with inline [^N] markers)                        │
│  ├── footnotes (Chicago-style definitions)                     │
│  └── metadata (compression_ratio, sources_used)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  Code Generation Pipeline                        │
│                  Pipeline ID: code-generation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: decompose_task                                        │
│  ├── Input: user_request                                        │
│  ├── Preset: S2 (DeepSeek for reasoning)                       │
│  └── Output: {subtasks, dependencies, assignments}             │
│                     │                                           │
│                     ▼                                           │
│  Stage 2: cross_reference (parallel for each subtask)          │
│  ├── Input: subtask specification                               │
│  ├── Scope: code-reference-engine/, ai-platform-data/          │
│  └── Output: {patterns, examples, citations}                   │
│                     │                                           │
│                     ▼                                           │
│  Stage 3: generate_code (per subtask)                          │
│  ├── Input: subtask + patterns + context                        │
│  ├── Preset: D4 (critique mode)                                │
│  └── Output: {code, explanation, citations}                    │
│                     │                                           │
│                     ▼                                           │
│  Stage 4: synthesize_outputs                                    │
│  ├── Input: all generated code artifacts                        │
│  ├── Strategy: merge                                            │
│  └── Output: {combined_code, provenance}                       │
│                     │                                           │
│                     ▼                                           │
│  Stage 5: analyze_artifact                                      │
│  ├── Input: combined_code                                       │
│  ├── Type: quality + security                                  │
│  └── Output: {findings, pass/fail}                             │
│                     │                                           │
│                     ▼                                           │
│  Stage 6: validate_against_spec                                 │
│  ├── Input: code + original_request                            │
│  └── Output: {valid, violations} → if !valid, retry Stage 3   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Citation Flow

### Footnote-Augmented Output Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CITATION FLOW (FOOTNOTE GENERATION)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. RETRIEVAL (semantic-search-service)                                     │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │ Query: "Repository pattern implementation"           │                  │
│  │                                                      │                  │
│  │ Returns:                                             │                  │
│  │ ┌────────────────────────────────────────────────┐  │                  │
│  │ │ chunk: "The Repository pattern provides..."    │  │                  │
│  │ │ source_type: "book"                            │  │                  │
│  │ │ author: "Fowler, Martin"                       │  │                  │
│  │ │ title: "Patterns of Enterprise Application..." │  │                  │
│  │ │ publisher: "Addison-Wesley"                    │  │                  │
│  │ │ year: 2002                                     │  │                  │
│  │ │ pages: "322-327"                               │  │                  │
│  │ │ similarity: 0.89                               │  │                  │
│  │ └────────────────────────────────────────────────┘  │                  │
│  │ ┌────────────────────────────────────────────────┐  │                  │
│  │ │ chunk: "class BaseRepository(ABC):..."         │  │                  │
│  │ │ source_type: "code"                            │  │                  │
│  │ │ repo: "code-reference-engine"                  │  │                  │
│  │ │ file_path: "backend/ddd/repository.py"         │  │                  │
│  │ │ line_range: "12-45"                            │  │                  │
│  │ │ commit_hash: "a1b2c3d"                         │  │                  │
│  │ │ similarity: 0.85                               │  │                  │
│  │ └────────────────────────────────────────────────┘  │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                          │                                                  │
│                          ▼                                                  │
│  2. GENERATION (inference-service)                                          │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │ System Prompt includes:                              │                  │
│  │ "When citing sources, use [^N] inline markers.       │                  │
│  │  Available sources:                                  │                  │
│  │  [^1] = Fowler, PEAA, pp. 322-327                   │                  │
│  │  [^2] = code-reference-engine/backend/ddd/repo...   │                  │
│  │  Always cite when referencing patterns or code."    │                  │
│  │                                                      │                  │
│  │ LLM generates:                                       │                  │
│  │ "The Repository pattern[^1] provides a collection-  │                  │
│  │  like interface for accessing domain objects. Our   │                  │
│  │  implementation[^2] uses an abstract base class..." │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                          │                                                  │
│                          ▼                                                  │
│  3. FORMATTING (audit-service)                                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │ Input markers: [^1], [^2]                            │                  │
│  │ Source metadata: (from retrieval)                    │                  │
│  │                                                      │                  │
│  │ Output footnotes (Chicago style):                    │                  │
│  │                                                      │                  │
│  │ [^1]: Fowler, Martin, *Patterns of Enterprise       │                  │
│  │       Application Architecture* (Boston: Addison-   │                  │
│  │       Wesley, 2002), 322-327.                        │                  │
│  │                                                      │                  │
│  │ [^2]: `code-reference-engine/backend/ddd/           │                  │
│  │       repository.py`, commit `a1b2c3d`,             │                  │
│  │       lines 12-45.                                   │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                          │                                                  │
│                          ▼                                                  │
│  4. AUDIT RECORD                                                            │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │ conversation_id: uuid                                │                  │
│  │ message_id: uuid                                     │                  │
│  │ citations_used: [citation_1_id, citation_2_id]      │                  │
│  │ retrieval_scores: [0.89, 0.85]                      │                  │
│  │ timestamp: 2025-12-29T...                           │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Citation Format Templates

| Source Type | Chicago Format |
|-------------|----------------|
| **Book** | `[^N]: LastName, FirstName, *Title* (City: Publisher, Year), pages.` |
| **Code** | `[^N]: \`repo/path/file.py\`, commit \`hash\`, lines X-Y.` |
| **Schema** | `[^N]: \`repo/schemas/file.json\`, version X.Y.Z.` |
| **Internal Doc** | `[^N]: service, *Document* (Date), §Section.` |

---

## Preset Selection

### Quality/Latency Tradeoff Matrix

| Agent Function | Light (Fast) | Standard | High Quality |
|----------------|--------------|----------|--------------|
| `extract_structure` | S4 (2.0 GB) | S1 (8.4 GB) | S5 (8.6 GB) |
| `summarize_content` | S4 | D4 (9.2 GB) | T1 (14.9 GB) |
| `generate_code` | S3 (4.5 GB) | D4 | T1 |
| `analyze_artifact` | S3 | D4 | D3 (13.1 GB) |
| `validate_against_spec` | S4 | D4 | D3 |
| `synthesize_outputs` | S1 | S1 | D3 |
| `decompose_task` | S2 (4.7 GB) | D3 | T4 (17.6 GB) |
| `cross_reference` | S4 | S4 | S4 |

### Preset → Orchestration Mode

| Preset | Mode | Models | Best For |
|--------|------|--------|----------|
| S1-S8 | single | 1 model | Fast, simple tasks |
| D1-D10 | critique/debate/pipeline | 2 models | Quality + validation |
| T1-T10 | pipeline/ensemble/debate | 3 models | Complex multi-step |
| Q1-Q5 | ensemble/pipeline/debate | 4 models | Server, high accuracy |
| P1-P3 | ensemble/pipeline/debate | 5 models | Maximum quality |

---

## Pydantic Schemas

> **Status:** Pending finalization of architecture design

Core types to be defined:
- `SourceMetadata` - Provenance for citations
- `Citation` - Chicago-style footnote
- `CitedContent` - Content with embedded citations
- Input/Output schemas for each agent function
- `PipelineDefinition` - DAG of agent functions
- `PipelineExecutionRequest/Response`

---

## Integration Points

### Service Endpoints

| Service | Endpoint | Agent Function |
|---------|----------|----------------|
| ai-agents | `POST /v1/functions/extract-structure/run` | `extract_structure` |
| ai-agents | `POST /v1/functions/summarize-content/run` | `summarize_content` |
| ai-agents | `POST /v1/functions/generate-code/run` | `generate_code` |
| ai-agents | `POST /v1/functions/analyze-artifact/run` | `analyze_artifact` |
| ai-agents | `POST /v1/functions/validate-against-spec/run` | `validate_against_spec` |
| ai-agents | `POST /v1/functions/synthesize-outputs/run` | `synthesize_outputs` |
| ai-agents | `POST /v1/functions/decompose-task/run` | `decompose_task` |
| ai-agents | `POST /v1/functions/cross-reference/run` | `cross_reference` |
| ai-agents | `POST /v1/pipelines/chapter-summarization/run` | Pipeline |
| ai-agents | `POST /v1/pipelines/code-generation/run` | Pipeline |

### Request Flow

```
External Client
      │
      ▼
┌─────────────┐
│ llm-gateway │ ─── Auth, Rate Limiting, Logging
│   :8080     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ai-agents  │ ─── Pipeline Orchestration
│   :8082     │
└──────┬──────┘
       │
       ├──────────────────┬───────────────────┐
       ▼                  ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│semantic-srch│    │Code-Orch    │    │inference-svc│
│   :8081     │    │   :8083     │    │   :8085     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                   │
       └──────────────────┴───────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │audit-service│ ─── Citation Tracking
                   │   :8084     │
                   └─────────────┘
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-29 | Initial agent functions architecture || 1.1.0 | 2025-12-29 | ADK Pattern Integration (Option C): state prefixes, artifact conventions, AgentTool mapping, workflow agent equivalents. Added reference to ADK_MIGRATION_GUIDE.md for full Option A migration path. |

---

## References

### Internal Documentation
- [ADK_MIGRATION_GUIDE.md](ADK_MIGRATION_GUIDE.md) - Full ADK adoption roadmap (Option A)
- [ARCHITECTURE.md](ARCHITECTURE.md) - ai-agents service architecture
- [AI_CODING_PLATFORM_ARCHITECTURE.md](../../textbooks/pending/platform/AI_CODING_PLATFORM_ARCHITECTURE.md) - Kitchen Brigade model
- [CODING_PATTERNS_ANALYSIS.md](../../textbooks/Guidelines/CODING_PATTERNS_ANALYSIS.md) - Anti-pattern catalog

### External Documentation
- [Google ADK Documentation](https://google.github.io/adk-docs/) - Agent Development Kit
- [ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/) - Workflow agents
- [ADK Artifacts](https://google.github.io/adk-docs/artifacts/) - Artifact service patterns