# Inter-AI Conversation Orchestration Architecture

## Overview

This document defines the architecture for **Inter-AI Conversation Orchestration** - a system where multiple AI participants (LLMs and BERT tools) engage in structured dialogue to collaboratively solve problems.

**Design Philosophy**: ai-agents acts as the **moderator/orchestrator** that facilitates conversations between AI participants. No AI talks directly to another - all communication is mediated.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ai-agents                                           │
│                    (Inter-AI Conversation Orchestrator)                          │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    ConversationOrchestrator                                │  │
│  │                                                                            │  │
│  │  • Manages turn-taking between AI participants                            │  │
│  │  • Maintains conversation state and context                               │  │
│  │  • Routes messages to appropriate services                                │  │
│  │  • Determines consensus/completion criteria                               │  │
│  │  • Logs full conversation for audit/review                                │  │
│  │  • Injects tool outputs into conversation when needed                     │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                          │
│           ┌───────────────────────────┼───────────────────────────┐             │
│           │                           │                           │             │
│           ▼                           ▼                           ▼             │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐     │
│  │  LLM Adapter    │        │  LLM Adapter    │        │  Tool Adapter   │     │
│  │  (Qwen)         │        │  (GPT-5.2)      │        │  (BERT Tools)   │     │
│  └────────┬────────┘        └────────┬────────┘        └────────┬────────┘     │
│           │                          │                          │               │
└───────────┼──────────────────────────┼──────────────────────────┼───────────────┘
            │                          │                          │
            ▼                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              llm-gateway (Port 8080)                               │
│                         (ALL LLM Routing - External & Future Local)               │
│                                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────────┐ │
│   │                           Provider Adapters                                  │ │
│   │                                                                              │ │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│   │  │   OpenAI     │  │  Anthropic   │  │  OpenRouter  │  │ Local LLM    │    │ │
│   │  │  (GPT-5.2)   │  │  (Claude)    │  │  (Qwen POC)  │  │ (Future)     │    │ │
│   │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│   │                                                                              │ │
│   │  Future: Local LLM servers on dedicated hardware route through gateway      │ │
│   │  (handles concurrency for 5-10 users with concurrent calls)                 │ │
│   └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────┐
│                      Code-Orchestrator-Service (Port 8083)                         │
│                            (BERT Tools ONLY - No LLMs)                             │
│                                                                                    │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│   │     SBERT      │  │ GraphCodeBERT  │  │   BERTopic     │  │    CodeT5      │ │
│   │                │  │                │  │                │  │                │ │
│   │ Semantic       │  │ Code-aware     │  │ Topic          │  │ Code           │ │
│   │ embeddings     │  │ validation     │  │ clustering     │  │ understanding  │ │
│   └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘ │
│                                                                                    │
│   ┌────────────────┐  ┌────────────────┐                                         │
│   │    HDBSCAN     │  │ ConceptValid.  │                                         │
│   │                │  │                │                                         │
│   │ Density        │  │ Concept vs     │                                         │
│   │ clustering     │  │ keyword filter │                                         │
│   └────────────────┘  └────────────────┘                                         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Communication Flow

**CRITICAL RULE**: ai-agents NEVER lets participants talk directly to each other.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          COMMUNICATION PATTERN                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ALL COMMUNICATION IS MEDIATED:                                                │
│                                                                                  │
│   LLM A → ai-agents → LLM B → ai-agents → Tool → ai-agents → LLM A             │
│                                                                                  │
│   ✅ Qwen responds → ai-agents receives → ai-agents sends to GPT                │
│   ✅ GPT responds → ai-agents receives → ai-agents queries BERTopic             │
│   ✅ BERTopic returns → ai-agents formats → ai-agents sends to Qwen             │
│                                                                                  │
│   ❌ Qwen → GPT (NEVER - no direct communication)                               │
│   ❌ LLM → Tool (NEVER - orchestrator mediates)                                 │
│                                                                                  │
│   BENEFITS:                                                                      │
│   • Full conversation logging                                                   │
│   • Ability to inject context/corrections                                       │
│   • Rate limiting per participant                                               │
│   • Graceful handling of failures                                               │
│   • Human-in-the-loop capability                                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Participant Types

### 1. LLM Participants (via llm-gateway)

| Participant | Provider | Model | Role |
|------------|----------|-------|------|
| Qwen | OpenRouter | qwen/qwen3-coder | Technical/code analysis |
| GPT-5.2 | OpenAI | gpt-5.2 | General reasoning, validation |
| Claude | Anthropic | claude-sonnet-4-5-20250929 | Document analysis (future) |
| Local LLM | Custom API | Llama/Mistral | Cost-effective inference (future) |

### 2. BERT Tool Participants (via Code-Orchestrator-Service)

| Participant | Tool | Capability |
|------------|------|------------|
| BERTopic | Topic modeling | Cluster terms into semantic groups |
| SBERT | Embeddings | Compute semantic similarity/coherence |
| GraphCodeBERT | Code validation | Validate code-related terms |
| CodeT5 | Code understanding | Analyze code snippets |
| HDBSCAN | Clustering | Density-based term clustering |
| ConceptValidator | Filtering | Distinguish concepts from keywords |

---

## Conversation Protocol

### Message Schema

```python
@dataclass
class ConversationMessage:
    """Single message in an inter-AI conversation."""
    
    message_id: str                    # Unique message identifier
    conversation_id: str               # Parent conversation ID
    participant_id: str                # Who sent this (e.g., "qwen", "gpt-5.2", "bertopic")
    participant_type: ParticipantType  # LLM | TOOL
    timestamp: datetime
    
    # Content
    role: str                          # "user" | "assistant" | "tool"
    content: str                       # The actual message
    
    # Metadata
    tokens_used: int | None            # For LLMs
    latency_ms: int                    # Response time
    metadata: dict                     # Participant-specific metadata


class ParticipantType(Enum):
    LLM = "llm"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"      # ai-agents itself
```

### Conversation State

```python
@dataclass
class Conversation:
    """Complete inter-AI conversation state."""
    
    conversation_id: str
    task: str                          # What are we solving?
    participants: list[str]            # Active participants
    messages: list[ConversationMessage]
    
    # Turn management
    current_turn: str                  # Who should speak next
    turn_order: list[str]              # Default turn sequence
    
    # Context
    context: dict                      # Shared context (book list, clusters, etc.)
    
    # Status
    status: ConversationStatus         # IN_PROGRESS | CONSENSUS | DEADLOCK | COMPLETE
    consensus_threshold: float         # Agreement level needed (e.g., 0.8)
    max_rounds: int                    # Prevent infinite loops
    current_round: int


class ConversationStatus(Enum):
    IN_PROGRESS = "in_progress"
    CONSENSUS = "consensus"            # Participants agree
    DEADLOCK = "deadlock"              # Cannot reach agreement
    COMPLETE = "complete"              # Task finished
    TIMEOUT = "timeout"                # Max rounds reached
```

---

## Turn-Taking Strategies

### Strategy 1: Round-Robin (Default)

```
Round 1: BERTopic → Qwen → GPT → SBERT
Round 2: BERTopic → Qwen → GPT → SBERT
...until consensus or max_rounds
```

### Strategy 2: Tool-First

```
1. All BERT tools provide analysis
2. LLMs discuss tool outputs
3. Repeat if needed
```

### Strategy 3: Debate

```
1. LLM A proposes
2. LLM B critiques
3. Tools provide evidence
4. LLMs reach consensus
```

---

## POC Use Case: Concept Validation

### Task

*"Here are extracted terms from 201 technical books. Collaborate to identify which are true CONCEPTS (abstract ideas) vs just KEYWORDS (frequent words). Produce a validated concept vocabulary."*

### Input Context

```json
{
  "task": "concept_validation",
  "context": {
    "book_list": [
      {"title": "Domain-Driven Design", "domain": "Architecture"},
      {"title": "Kubernetes Up & Running", "domain": "DevOps"},
      ...
    ],
    "concepts_by_book": {
      "Domain-Driven Design": ["bounded context", "aggregate", "ubiquitous language"],
      "Kubernetes Up & Running": ["pod", "deployment", "service mesh"],
      ...
    },
    "bertopic_clusters": [
      {
        "cluster_id": 3,
        "terms": ["test", "tests", "testing", "unit", "integration"],
        "quality_score": 0.93
      },
      ...
    ]
  }
}
```

### Example Dialogue

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ROUND 1                                                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│ [ORCHESTRATOR → BERTopic]                                                       │
│ "What clusters did you find from the 10,625 extracted terms?"                   │
│                                                                                  │
│ [BERTopic → ORCHESTRATOR]                                                       │
│ {                                                                                │
│   "cluster_count": 562,                                                         │
│   "noise_terms": 2400,                                                          │
│   "top_clusters": [                                                             │
│     {"id": 3, "terms": ["test", "tests", "testing", "unit"], "count": 57},     │
│     {"id": 13, "terms": ["architecture", "architectures", "monolithic"], ...}  │
│   ]                                                                             │
│ }                                                                                │
│                                                                                  │
│ [ORCHESTRATOR → SBERT]                                                          │
│ "What's the semantic coherence of Cluster 3?"                                   │
│                                                                                  │
│ [SBERT → ORCHESTRATOR]                                                          │
│ {"cluster_id": 3, "coherence_score": 0.87, "verdict": "high_coherence"}        │
│                                                                                  │
│ [ORCHESTRATOR → Qwen]                                                           │
│ "Given Cluster 3 (terms: test, tests, testing, unit, integration) with         │
│  coherence 0.87, what abstract concept does this represent?                     │
│  Context: These terms appear in books: Clean Code, TDD By Example,              │
│  Continuous Delivery"                                                           │
│                                                                                  │
│ [Qwen → ORCHESTRATOR]                                                           │
│ {                                                                                │
│   "cluster_id": 3,                                                              │
│   "concept_name": "Software Testing",                                           │
│   "confidence": "high",                                                         │
│   "reasoning": "Terms all relate to the practice of validating software..."    │
│ }                                                                                │
│                                                                                  │
│ [ORCHESTRATOR → GPT-5.2]                                                        │
│ "Qwen labeled Cluster 3 as 'Software Testing'. Do you agree?                    │
│  Terms: [test, tests, testing, unit, integration]                               │
│  Book sources: Clean Code, TDD By Example, Continuous Delivery"                 │
│                                                                                  │
│ [GPT-5.2 → ORCHESTRATOR]                                                        │
│ {                                                                                │
│   "agreement": true,                                                            │
│   "suggested_refinement": "Software Testing Methodology",                       │
│   "reasoning": "More precise - distinguishes from testing tools/frameworks"    │
│ }                                                                                │
│                                                                                  │
│ [ORCHESTRATOR] CONSENSUS REACHED                                                │
│ Final label: "Software Testing" (both LLMs agree on core concept)               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure (in ai-agents)

```
ai-agents/
├── src/
│   ├── conversation/                    # NEW - Inter-AI Orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py              # ConversationOrchestrator class
│   │   ├── models.py                    # Conversation, Message dataclasses
│   │   ├── turn_strategies.py           # Round-robin, tool-first, debate
│   │   ├── consensus.py                 # Consensus detection logic
│   │   └── logging.py                   # Conversation transcript logging
│   │
│   ├── participants/                    # NEW - Participant Adapters
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract Participant interface
│   │   ├── llm_participant.py           # LLM adapter (calls llm-gateway)
│   │   └── tool_participant.py          # Tool adapter (calls Code-Orchestrator)
│   │
│   ├── api/
│   │   └── routes/
│   │       └── conversation.py          # NEW - POST /v1/conversation/start
│   │
│   └── ...existing folders...
│
├── docs/
│   ├── INTER_AI_ORCHESTRATION.md        # THIS FILE
│   └── ...existing docs...
```

---

## API Endpoints (New)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/conversation/start` | Start a new inter-AI conversation |
| GET | `/v1/conversation/{id}` | Get conversation state |
| GET | `/v1/conversation/{id}/transcript` | Get full transcript |
| POST | `/v1/conversation/{id}/inject` | Inject human message (human-in-loop) |
| POST | `/v1/conversation/{id}/stop` | Stop conversation early |

---

## Gateway Requirements (llm-gateway)

### New Provider: OpenRouter

```python
# llm-gateway needs to add OpenRouter provider
class OpenRouterProvider:
    """OpenRouter provider for accessing Qwen and other models."""
    
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    
    supported_models = [
        "qwen/qwen3-coder",
        "meta-llama/llama-3.1-70b-instruct",
        # ...
    ]
```

### Environment Variables

```bash
# ~/.zshrc (already configured)
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPSEEK_API_KEY="sk-..."
```

---

## Future Extensions

### 1. Local LLM Integration

When local LLM servers are deployed on dedicated hardware:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Local LLM Server                               │
│                 (Dedicated Mini-Server)                           │
│                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│   │   Ollama    │  │    vLLM     │  │  TGI        │             │
│   │             │  │             │  │             │             │
│   │  Llama 3    │  │  Mistral    │  │  CodeLlama  │             │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│          │                │                │                     │
│          └────────────────┴────────────────┘                     │
│                           │                                      │
│                    API Wrapper (FastAPI)                         │
│                    OpenAI-compatible endpoint                    │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  llm-gateway  │  ← Routes like any other provider
                    │  (Port 8080)  │
                    └───────────────┘
```

### 2. IDE Integration (Claude Code, Copilot)

```
┌──────────────────────────────────────────────────────────────────┐
│                    External AI Integration                        │
│                                                                   │
│   Claude Code (VS Code)  ←→  ai-agents (via Gateway)             │
│   GitHub Copilot         ←→  ai-agents (via Gateway)             │
│                                                                   │
│   These become additional "participants" in conversations         │
│   ai-agents coordinates them with internal LLMs and tools        │
└──────────────────────────────────────────────────────────────────┘
```

### 3. BERT Tool Inter-Communication

```
┌──────────────────────────────────────────────────────────────────┐
│           Tool Chaining (orchestrated by ai-agents)              │
│                                                                   │
│   BERTopic clusters → SBERT validates coherence                  │
│   SBERT embeddings  → HDBSCAN clusters                           │
│   GraphCodeBERT     → validates code terms from BERTopic         │
│                                                                   │
│   ai-agents can chain tool calls when one tool needs             │
│   output from another                                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria for POC

| Criteria | Measurement |
|----------|-------------|
| Conversation completes | All 562 clusters processed |
| Consensus rate | > 80% of clusters get agreement |
| Concept quality | Manual review of 50 random concepts |
| Latency | < 5 min for full vocabulary generation |
| Transcript logging | Full conversation saved to JSON |

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) - ai-agents service architecture
- [Kitchen Brigade Model](ARCHITECTURE.md#kitchen-brigade-role-expeditor) - Service roles
- Gateway-First Pattern - All external communication through gateway
