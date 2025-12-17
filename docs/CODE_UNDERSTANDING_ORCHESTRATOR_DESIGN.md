# Code Understanding Orchestrator Service

> ## ⚠️ DEPRECATED
> **Date**: January 2025  
> **Reason**: This document is a duplicate. The canonical version lives in the Code-Orchestrator-Service repository.
> 
> **Canonical Document**:
> - [Code-Orchestrator-Service/docs/ARCHITECTURE.md](/Code-Orchestrator-Service/docs/ARCHITECTURE.md) - Sous Chef architecture
>
> **Platform Context**:
> - [AI_CODING_PLATFORM_ARCHITECTURE.md](/textbooks/pending/platform/AI_CODING_PLATFORM_ARCHITECTURE.md) - Kitchen Brigade overview
>
> This document is retained for historical reference only. Do not update.

## Executive Summary

A standalone microservice that coordinates multiple specialized code understanding models (CodeT5+, GraphCodeBERT, CodeBERT) to dynamically extract, validate, and rank search terms from natural language queries. This service replaces hardcoded keyword mappings with intelligent, context-aware term generation.

This service acts as the **"Sous Chef"** in the Kitchen Brigade architecture—interpreting orders (queries), preparing ingredients (keywords), curating results, and auditing output before serving to the customer.

---

## Kitchen Brigade Architecture Model

### The Analogy

The platform follows a **Kitchen Brigade** organizational model where each service has a specific role:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          🍽️  KITCHEN BRIGADE MODEL                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  👤 CUSTOMER (Claude/GPT/User)                                              │
│     └─→ Places order: "I need code for document chunking with overlap"      │
│                                                                              │
│  👨‍🍳 SOUS CHEF (Code Understanding Orchestrator) ← THIS SERVICE             │
│     └─→ SMART: Interprets the order                                         │
│     └─→ Extracts keywords/concepts using code understanding models          │
│     └─→ Sends keyword list to Cookbook                                      │
│                                                                              │
│  📖 COOKBOOK (Semantic Search Service) ← DUMB RETRIEVAL                     │
│     └─→ Takes keywords as INPUT (does NOT generate them)                    │
│     └─→ Queries vector DBs (Qdrant, Neo4j) where content lives              │
│     └─→ Returns ALL matches without filtering or judgment                   │
│     └─→ Just a retrieval engine - like looking up recipes in a book         │
│                                                                              │
│  👨‍🍳 CHEF DE PARTIE (Orchestrator - Curation Phase)                         │
│     └─→ Receives raw results from Cookbook                                  │
│     └─→ SMART: Filters out irrelevant results (C++ "chunk of memory")       │
│     └─→ Ranks by domain relevance                                           │
│     └─→ Prepares curated instructions for Line Cook                         │
│                                                                              │
│  👨‍🍳 LINE COOK (Code Llama via LLM Gateway)                                 │
│     └─→ Receives curated context + instructions                             │
│     └─→ Generates actual code from the instructions                         │
│                                                                              │
│  👨‍🍳 CHEF DE PARTIE (Orchestrator - Audit Phase)                            │
│     └─→ Validates generated code quality                                    │
│     └─→ Ensures code matches original intent                                │
│                                                                              │
│  👤 CUSTOMER receives the final plated dish (working code)                  │
│     └─→ Implements the code in their project                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Service Responsibility Matrix

| Service | Role | Intelligence | What It Does | What It Does NOT Do |
|---------|------|--------------|--------------|---------------------|
| **LLM Gateway** | Router | Routing only | Routes requests to appropriate models | Make decisions about content |
| **Code Understanding Orchestrator** | Sous Chef + Chef de Partie | **SMART** | Extracts keywords, curates results, audits output | Store content, execute searches |
| **Semantic Search Service** | Cookbook | **DUMB** | Takes keywords as input, queries vector DBs, returns all matches | Generate keywords, filter results, make judgments |
| **Code Llama** | Line Cook | Executor | Generates code from curated instructions | Decide what to generate |
| **Vector DBs (Qdrant/Neo4j)** | Pantry | Storage | Stores embeddings and relationships | Nothing else |

### Key Insight: Semantic Search is DUMB

The **Semantic Search Service** is intentionally dumb:
- It does NOT contain knowledge itself—it queries databases that contain knowledge
- It does NOT generate keywords—it receives them as input
- It does NOT filter results—it returns ALL matches
- It's just a query executor, like looking up recipes in a cookbook

The **intelligence lives in the Orchestrator**, which:
1. **Interprets** the customer's order (query understanding)
2. **Generates** the right keywords to search for
3. **Curates** the raw results (filters irrelevant matches)
4. **Instructs** the line cook (prepares context for code generation)
5. **Audits** the final output (validates generated code)

---

## Problem Statement

### Current State
The existing cross-reference system uses **hardcoded `FOCUS_SEARCH_TERMS`** mappings:

```python
FOCUS_SEARCH_TERMS = {
    "multi-stage chunking": [
        "chunk", "chunking", "split", "segment", ...  # Static, brittle
    ],
}
```

### Issues
1. **False Positives**: "chunk" matches C++ memory allocation ("chunk of memory") instead of LLM document chunking
2. **Not Portable**: Hardcoded terms don't transfer across taxonomies/domains
3. **Maintenance Burden**: Manual updates required for new concepts
4. **Limited Coverage**: Misses semantically related terms not in the list

### Proposed Solution
A multi-model orchestration service that dynamically generates contextually-relevant search terms.

---

## Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     👤 CUSTOMER (Claude/GPT/User)                            │
│                "I need code for document chunking with overlap"              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Request
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              👨‍🍳 CODE UNDERSTANDING ORCHESTRATOR (Sous Chef)                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         API Gateway                                    │  │
│  │                    /extract, /validate, /search                        │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                           │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                      Agent Orchestrator                                │  │
│  │                   (LangGraph State Machine)                            │  │
│  └───┬───────────────────────┬───────────────────────┬───────────────────┘  │
│      │                       │                       │                       │
│      ▼                       ▼                       ▼                       │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐                │
│  │  CodeT5+    │       │GraphCodeBERT│       │  CodeBERT   │                │
│  │  Agent      │       │   Agent     │       │   Agent     │                │
│  │ (Generator) │       │ (Validator) │       │  (Ranker)   │                │
│  └─────────────┘       └─────────────┘       └─────────────┘                │
│                                                                              │
│  Output: ["chunking", "text_splitter", "overlap", "RAG", "embedding"]       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Keywords (INPUT to Cookbook)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                📖 SEMANTIC SEARCH SERVICE (Cookbook) - DUMB                  │
│                                                                              │
│  Input:  Keywords from Orchestrator                                          │
│  Action: Query vector databases                                              │
│  Output: ALL matches (no filtering, no judgment)                            │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Qdrant        │  │   Neo4j Graph   │  │   Hybrid        │             │
│  │   Retriever     │  │   Retriever     │  │   Search        │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                        │
│           └────────────────────┼────────────────────┘                        │
│                                │                                             │
│           Returns: [C++ memory chunk, LLM chunking, game chunks, ...]       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Raw Results
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           👨‍🍳 ORCHESTRATOR (Chef de Partie) - Curation Phase                 │
│                                                                              │
│  ✓ Filter: Remove C++ "chunk of memory" (wrong domain)                      │
│  ✓ Rank: Score by relevance to LLM/AI context                               │
│  ✓ Prepare: Curated context for Line Cook                                   │
│                                                                              │
│  Output: Curated references + instructions for code generation              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Curated Context
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    👨‍🍳 LINE COOK (Code Llama via LLM Gateway)                │
│                                                                              │
│  Input:  Curated context + generation instructions                          │
│  Action: Generate code based on best practices from references              │
│  Output: Working code implementation                                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Generated Code
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            👨‍🍳 ORCHESTRATOR (Chef de Partie) - Audit Phase                   │
│                                                                              │
│  ✓ Validate: Code quality checks                                            │
│  ✓ Verify: Matches original intent                                          │
│  ✓ Format: Prepare final output                                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Final Result
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         👤 CUSTOMER receives final dish                      │
│                      (Working code ready to implement)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Where Content Actually Lives

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🗄️  DATA LAYER (Pantry)                           │
│                                                                              │
│  These are the ACTUAL STORAGE systems - where content lives:                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  QDRANT (Vector Database)                                            │   │
│  │  └─→ Stores: Document embeddings, chunk vectors                      │   │
│  │  └─→ Contains: Textbook content, code patterns, technical docs       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NEO4J (Graph Database)                                              │   │
│  │  └─→ Stores: Relationships between concepts, cross-references        │   │
│  │  └─→ Contains: Book→Chapter→Section→Concept relationships           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JSON FILES (Local Textbooks)                                        │   │
│  │  └─→ Stores: Raw textbook JSON files                                 │   │
│  │  └─→ Location: /Users/kevintoles/POC/textbooks/JSON Texts/           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

The Semantic Search Service QUERIES these systems - it doesn't contain them.
```

---

## Multi-Model Coordination Flow

### Agent Conversation Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Query                                       │
│          "LLM code understanding with multi-stage chunking for RAG"          │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR STATE MACHINE                            │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 1: GENERATION                                                      │ │
│  │ ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ CodeT5+ Agent                                                        │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  "Extract technical search terms for: LLM code understanding  │ │ │
│  │ │          with multi-stage chunking for RAG"                          │ │ │
│  │ │                                                                       │ │ │
│  │ │ Output: {                                                             │ │ │
│  │ │   "primary_terms": ["chunking", "RAG", "embedding", "LLM"],          │ │ │
│  │ │   "related_terms": ["tokenization", "vector", "retrieval"],          │ │ │
│  │ │   "code_patterns": ["text_splitter", "chunk_size", "overlap"]        │ │ │
│  │ │ }                                                                     │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 2: VALIDATION                                                      │ │
│  │ ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ GraphCodeBERT Agent                                                  │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  Generated terms + Original query + Domain context            │ │ │
│  │ │                                                                       │ │ │
│  │ │ Validation Rules:                                                     │ │ │
│  │ │   ✓ "chunking" - Valid (LLM context, not memory allocation)          │ │ │
│  │ │   ✓ "RAG" - Valid (retrieval augmented generation)                   │ │ │
│  │ │   ✓ "embedding" - Valid (vector representations)                     │ │ │
│  │ │   ✗ "split" - Rejected (too generic, high false positive rate)       │ │ │
│  │ │                                                                       │ │ │
│  │ │ Expansions Added:                                                     │ │ │
│  │ │   + "semantic_search" (related to RAG)                               │ │ │
│  │ │   + "context_window" (related to chunking)                           │ │ │
│  │ │   + "HNSW" (related to vector indexing)                              │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 3: RANKING                                                         │ │
│  │ ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ CodeBERT Agent                                                       │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  Validated terms + Original query embedding                   │ │ │
│  │ │                                                                       │ │ │
│  │ │ Similarity Scoring:                                                   │ │ │
│  │ │   1. chunking         → 0.95 (highest relevance)                     │ │ │
│  │ │   2. RAG              → 0.92                                         │ │ │
│  │ │   3. embedding        → 0.89                                         │ │ │
│  │ │   4. context_window   → 0.85                                         │ │ │
│  │ │   5. semantic_search  → 0.82                                         │ │ │
│  │ │   6. tokenization     → 0.78                                         │ │ │
│  │ │   7. vector           → 0.75                                         │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 4: CONSENSUS                                                       │ │
│  │                                                                           │ │
│  │ Agreement Filter: Terms must be approved by ≥2 models                    │ │
│  │                                                                           │ │
│  │ Final Output:                                                             │ │
│  │ {                                                                         │ │
│  │   "search_terms": [                                                       │ │
│  │     {"term": "chunking", "score": 0.95, "models_agreed": 3},             │ │
│  │     {"term": "RAG", "score": 0.92, "models_agreed": 3},                  │ │
│  │     {"term": "embedding", "score": 0.89, "models_agreed": 3},            │ │
│  │     {"term": "context_window", "score": 0.85, "models_agreed": 2},       │ │
│  │     {"term": "semantic_search", "score": 0.82, "models_agreed": 2}       │ │
│  │   ],                                                                      │ │
│  │   "excluded_terms": [                                                     │ │
│  │     {"term": "split", "reason": "Too generic", "models_agreed": 1}       │ │
│  │   ]                                                                       │ │
│  │ }                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection Rationale

### The Trio: Why These Three Models?

| Model | Role | Strength | HuggingFace ID |
|-------|------|----------|----------------|
| **CodeT5+** | Generator | Encoder-decoder architecture enables text generation; trained on NL↔Code pairs | `Salesforce/codet5p-220m` |
| **GraphCodeBERT** | Validator | Understands code structure via data flow graphs; catches semantic mismatches | `microsoft/graphcodebert-base` |
| **CodeBERT** | Ranker | Fast embeddings for similarity scoring; well-established baseline | `microsoft/codebert-base` |

### Model Comparison Matrix

```
┌────────────────────┬────────────────┬────────────────┬────────────────┐
│     Capability     │    CodeT5+     │ GraphCodeBERT  │    CodeBERT    │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Text Generation    │       ✅       │       ❌       │       ❌       │
│ Code Structure     │       ⚠️       │       ✅       │       ⚠️       │
│ Embeddings         │       ✅       │       ✅       │       ✅       │
│ Zero-shot Ready    │       ✅       │       ⚠️       │       ⚠️       │
│ Parameters         │    220M-6B     │     125M       │     125M       │
│ Inference Speed    │    Medium      │     Fast       │     Fast       │
└────────────────────┴────────────────┴────────────────┴────────────────┘

Legend: ✅ Excellent  ⚠️ Partial  ❌ Not supported
```

---

## Service API Design

### REST Endpoints

```yaml
openapi: 3.0.0
info:
  title: Code Understanding Orchestrator API
  version: 1.0.0

paths:
  /api/v1/extract:
    post:
      summary: Extract search terms from query
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  example: "LLM code understanding with multi-stage chunking"
                domain:
                  type: string
                  example: "ai-ml"
                options:
                  type: object
                  properties:
                    min_confidence:
                      type: number
                      default: 0.7
                    max_terms:
                      type: integer
                      default: 10
                    require_consensus:
                      type: boolean
                      default: true
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExtractionResult'

  /api/v1/validate:
    post:
      summary: Validate terms against domain context
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                terms:
                  type: array
                  items:
                    type: string
                query:
                  type: string
                domain:
                  type: string

  /v1/search:
    post:
      summary: Full pipeline - extract, validate, and search
      description: Combines extraction with semantic search service

components:
  schemas:
    ExtractionResult:
      type: object
      properties:
        search_terms:
          type: array
          items:
            type: object
            properties:
              term:
                type: string
              score:
                type: number
              models_agreed:
                type: integer
        excluded_terms:
          type: array
        metadata:
          type: object
          properties:
            processing_time_ms:
              type: integer
            models_used:
              type: array
```

---

## Repository Structure

```
code-understanding-orchestrator/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── extract.py           # /extract endpoint
│   │   │   ├── validate.py          # /validate endpoint
│   │   │   └── search.py            # /search endpoint
│   │   └── schemas/
│   │       ├── requests.py
│   │       └── responses.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent abstract class
│   │   ├── codet5_agent.py          # CodeT5+ Generator
│   │   ├── graphcodebert_agent.py   # GraphCodeBERT Validator
│   │   └── codebert_agent.py        # CodeBERT Ranker
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── state_machine.py         # LangGraph state definitions
│   │   ├── graph.py                 # Orchestration graph
│   │   └── consensus.py             # Multi-model agreement logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py              # Model loading/caching
│   │   └── inference.py             # Inference utilities
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   └── semantic_search.py       # Downstream service client
│   │
│   └── config/
│       ├── __init__.py
│       ├── settings.py              # Pydantic settings
│       └── models.yaml              # Model configurations
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_orchestrator.py
│   │   └── test_consensus.py
│   ├── integration/
│   │   └── test_full_pipeline.py
│   └── fixtures/
│       └── sample_queries.json
│
├── scripts/
│   ├── download_models.py           # Pre-download HF models
│   └── benchmark.py                 # Performance benchmarking
│
├── deploy/
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   └── helm/
│       └── code-understanding-orchestrator/
│
└── docs/
    ├── API.md
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```

---

## Core Implementation

### State Machine Definition (LangGraph)

```python
# src/orchestrator/state_machine.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class OrchestratorState(TypedDict):
    """State shared across all agents."""
    query: str
    domain: str
    
    # Generator output
    generated_terms: list[str]
    related_terms: list[str]
    code_patterns: list[str]
    
    # Validator output
    validated_terms: list[str]
    rejected_terms: list[dict]
    expanded_terms: list[str]
    
    # Ranker output
    ranked_terms: list[dict]
    
    # Final output
    final_terms: list[dict]
    excluded_terms: list[dict]
    
    # Metadata
    processing_steps: list[str]
    errors: list[str]


def create_orchestrator_graph() -> StateGraph:
    """Create the multi-model orchestration graph."""
    
    graph = StateGraph(OrchestratorState)
    
    # Add nodes
    graph.add_node("generate", generate_terms)
    graph.add_node("validate", validate_terms)
    graph.add_node("rank", rank_terms)
    graph.add_node("consensus", build_consensus)
    
    # Define edges
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "rank")
    graph.add_edge("rank", "consensus")
    graph.add_edge("consensus", END)
    
    return graph.compile()
```

### Agent Base Class

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Any

class BaseCodeAgent(ABC):
    """Base class for code understanding agents."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from HuggingFace."""
        pass
    
    @abstractmethod
    def process(self, state: dict) -> dict:
        """Process state and return updated state."""
        pass
    
    def health_check(self) -> bool:
        """Verify model is loaded and functional."""
        return self.model is not None
```

### CodeT5+ Generator Agent

```python
# src/agents/codet5_agent.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import BaseCodeAgent

class CodeT5PlusAgent(BaseCodeAgent):
    """Generator agent using CodeT5+ for term extraction."""
    
    def __init__(self, model_size: str = "220m"):
        model_name = f"Salesforce/codet5p-{model_size}"
        super().__init__(model_name)
    
    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    def process(self, state: dict) -> dict:
        """Extract search terms from query."""
        query = state["query"]
        domain = state.get("domain", "general")
        
        prompt = f"""Extract technical search terms and concepts from this query.
Domain: {domain}
Query: {query}

Output format: term1, term2, term3, ..."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        terms = [t.strip() for t in result.split(",")]
        
        state["generated_terms"] = terms
        state["processing_steps"].append("CodeT5+ generation complete")
        
        return state
```

### GraphCodeBERT Validator Agent

```python
# src/agents/graphcodebert_agent.py
from transformers import AutoTokenizer, AutoModel
import torch
from .base import BaseCodeAgent

class GraphCodeBERTAgent(BaseCodeAgent):
    """Validator agent using GraphCodeBERT for semantic validation."""
    
    def __init__(self):
        super().__init__("microsoft/graphcodebert-base")
        self.domain_embeddings = {}  # Cached domain concept embeddings
    
    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            return outputs.last_hidden_state[:, 0, :]
    
    def process(self, state: dict) -> dict:
        """Validate terms against query context."""
        query = state["query"]
        terms = state["generated_terms"]
        
        query_embedding = self.get_embedding(query)
        
        validated = []
        rejected = []
        
        for term in terms:
            term_embedding = self.get_embedding(term)
            similarity = torch.cosine_similarity(query_embedding, term_embedding)
            
            if similarity > 0.5:  # Threshold
                validated.append(term)
            else:
                rejected.append({
                    "term": term,
                    "reason": "Low semantic similarity to query",
                    "score": similarity.item()
                })
        
        # Expand with related terms
        expanded = self._expand_terms(query_embedding, validated)
        
        state["validated_terms"] = validated
        state["rejected_terms"] = rejected
        state["expanded_terms"] = expanded
        state["processing_steps"].append("GraphCodeBERT validation complete")
        
        return state
    
    def _expand_terms(self, query_embedding: torch.Tensor, 
                      validated: list[str]) -> list[str]:
        """Expand with semantically related terms."""
        # Implementation: compare against domain concept bank
        expansions = []
        domain_concepts = [
            "semantic_search", "context_window", "tokenizer",
            "HNSW", "vector_store", "retrieval_augmented"
        ]
        
        for concept in domain_concepts:
            if concept not in validated:
                concept_embedding = self.get_embedding(concept)
                similarity = torch.cosine_similarity(
                    query_embedding, concept_embedding
                )
                if similarity > 0.6:
                    expansions.append(concept)
        
        return expansions
```

---

## Deployment Architecture

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  orchestrator:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_CACHE_DIR=/models
      - DEVICE=cpu
      - LOG_LEVEL=INFO
    volumes:
      - model-cache:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: GPU-enabled inference
  orchestrator-gpu:
    build: .
    runtime: nvidia
    environment:
      - DEVICE=cuda
      - MODEL_CACHE_DIR=/models
    volumes:
      - model-cache:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  model-cache:
```

### Kubernetes Deployment

```yaml
# deploy/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-understanding-orchestrator
  labels:
    app: code-understanding-orchestrator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: code-understanding-orchestrator
  template:
    metadata:
      labels:
        app: code-understanding-orchestrator
    spec:
      containers:
        - name: orchestrator
          image: code-understanding-orchestrator:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
          env:
            - name: MODEL_CACHE_DIR
              value: "/models"
            - name: SEMANTIC_SEARCH_URL
              valueFrom:
                configMapKeyRef:
                  name: orchestrator-config
                  key: semantic_search_url
          volumeMounts:
            - name: model-cache
              mountPath: /models
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
```

---

## Integration Points

### With Semantic Search Service

```python
# src/clients/semantic_search.py
import httpx
from typing import Optional

class SemanticSearchClient:
    """Client for semantic-search-service integration."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(
        self,
        terms: list[str],
        taxonomy: Optional[str] = None,
        limit: int = 20
    ) -> dict:
        """Execute search with extracted terms."""
        response = await self.client.post(
            f"{self.base_url}/v1/search",
            json={
                "query_terms": terms,
                "taxonomy_filter": taxonomy,
                "limit": limit,
                "search_type": "hybrid"  # vector + keyword
            }
        )
        return response.json()
```

### With AI Agents Service

```python
# Example integration in ai-agents
from code_understanding_client import CodeUnderstandingClient

class CrossReferenceAgent:
    def __init__(self):
        self.orchestrator = CodeUnderstandingClient(
            url="http://code-understanding-orchestrator:8080"
        )
        self.search = SemanticSearchClient(...)
    
    async def find_references(self, query: str, taxonomy: str) -> list:
        # Step 1: Extract terms via orchestrator
        extraction = await self.orchestrator.extract(
            query=query,
            domain="ai-ml",
            options={"min_confidence": 0.7}
        )
        
        # Step 2: Search with extracted terms
        terms = [t["term"] for t in extraction["search_terms"]]
        results = await self.search.search(terms, taxonomy=taxonomy)
        
        return results
```

---

## Use Cases Beyond Cross-References

This service is designed for reuse across multiple applications:

| Use Case | Description |
|----------|-------------|
| **Code Search** | Extract search terms from natural language queries about code |
| **Documentation Retrieval** | Find relevant docs based on technical questions |
| **API Discovery** | Match user intent to available API endpoints |
| **Codebase Q&A** | Power RAG systems for code understanding |
| **Technical Support** | Route support tickets to relevant knowledge base articles |
| **Code Review** | Identify related code patterns and best practices |

---

## Performance Considerations

### Model Loading Strategy

```python
# Lazy loading with caching
class ModelRegistry:
    _instances = {}
    
    @classmethod
    def get_model(cls, model_type: str):
        if model_type not in cls._instances:
            if model_type == "codet5":
                cls._instances[model_type] = CodeT5PlusAgent()
            elif model_type == "graphcodebert":
                cls._instances[model_type] = GraphCodeBERTAgent()
            elif model_type == "codebert":
                cls._instances[model_type] = CodeBERTAgent()
            cls._instances[model_type].load_model()
        return cls._instances[model_type]
```

### Batch Processing

```python
# For high-throughput scenarios
async def batch_extract(queries: list[str]) -> list[dict]:
    """Process multiple queries in parallel."""
    tasks = [extract_single(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### Caching Layer

```python
# Redis caching for repeated queries
@cache(ttl=3600)  # 1 hour
async def extract_terms(query: str, domain: str) -> dict:
    return await orchestrator.process(query, domain)
```

---

## Monitoring & Observability

### Metrics to Track

| Metric | Description |
|--------|-------------|
| `extraction_duration_seconds` | Time to extract terms |
| `model_inference_duration_seconds` | Per-model inference time |
| `terms_generated_total` | Number of terms generated |
| `terms_validated_ratio` | Ratio of validated vs rejected terms |
| `consensus_agreement_rate` | How often models agree |
| `cache_hit_rate` | Effectiveness of caching |

### Structured Logging

```python
logger.info(
    "Extraction complete",
    extra={
        "query_hash": hash(query),
        "terms_count": len(final_terms),
        "models_used": ["codet5", "graphcodebert", "codebert"],
        "processing_time_ms": elapsed,
        "consensus_rate": agreement_rate
    }
)
```

---

## Next Steps

1. **Phase 1**: Create repository and basic FastAPI structure
2. **Phase 2**: Implement CodeT5+ generator agent
3. **Phase 3**: Add GraphCodeBERT validator agent
4. **Phase 4**: Add CodeBERT ranker agent
5. **Phase 5**: Implement LangGraph orchestration
6. **Phase 6**: Integration tests with semantic-search-service
7. **Phase 7**: Docker/Kubernetes deployment
8. **Phase 8**: Performance optimization and caching

---

## References

- [CodeT5+ Paper](https://arxiv.org/abs/2305.07922)
- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
