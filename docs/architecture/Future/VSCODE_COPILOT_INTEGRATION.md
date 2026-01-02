# VS Code Copilot Integration with AI Platform

> **Version:** 1.0.0  
> **Created:** 2026-01-01  
> **Status:** Implementation Guide  
> **Reference:** inference-service:8085, llm-gateway:8080

## Overview

This document describes three methods to integrate your local AI platform with VS Code Copilot, enabling you to use your self-hosted models (phi-4, deepseek-r1-7b, qwen2.5-7b, llama-3.2-3b, etc.) directly within VS Code.

---

## Method 1: OpenAI-Compatible API Endpoints (BYOK)

Your **inference-service** already exposes OpenAI-compatible endpoints on port 8085. This is the easiest integration path.

### Prerequisites

1. **VS Code Insiders** - This feature requires the Insiders build
2. **inference-service running** - Start with the task "Start Inference Service (Native Metal)"

### Step 1: Verify inference-service is Running

```bash
# Check health
curl http://localhost:8085/health

# List available models
curl http://localhost:8085/v1/models

# Test chat completion (OpenAI-compatible format)
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 100
  }'
```

### Step 2: Configure VS Code Settings

Open VS Code Settings (`Cmd+Shift+P` > "Preferences: Open User Settings (JSON)") and add:

```json
{
  "github.copilot.chat.customOAIModels": [
    {
      "name": "phi-4",
      "displayName": "Phi-4 (Local)",
      "endpoint": "http://localhost:8085/v1/chat/completions",
      "model": "phi-4",
      "apiKey": "not-required",
      "capabilities": {
        "chat": true,
        "codeCompletion": true
      }
    },
    {
      "name": "deepseek-r1",
      "displayName": "DeepSeek R1 7B (Local CoT)",
      "endpoint": "http://localhost:8085/v1/chat/completions",
      "model": "deepseek-r1-7b",
      "apiKey": "not-required",
      "capabilities": {
        "chat": true,
        "reasoning": true
      }
    },
    {
      "name": "qwen-coder",
      "displayName": "Qwen 2.5 7B (Local Coder)",
      "endpoint": "http://localhost:8085/v1/chat/completions",
      "model": "qwen2.5-7b",
      "apiKey": "not-required",
      "capabilities": {
        "chat": true,
        "codeCompletion": true
      }
    },
    {
      "name": "llama-fast",
      "displayName": "Llama 3.2 3B (Local Fast)",
      "endpoint": "http://localhost:8085/v1/chat/completions",
      "model": "llama-3.2-3b",
      "apiKey": "not-required",
      "capabilities": {
        "chat": true
      }
    }
  ]
}
```

### Step 3: Use in Copilot Chat

1. Open Copilot Chat (`Cmd+Shift+I`)
2. Click the model selector dropdown
3. Select your local model (e.g., "Phi-4 (Local)")
4. Chat as normal - requests go to your local inference-service

### Available Models & Use Cases

| Model | Size | Best For | Endpoint Model ID |
|-------|------|----------|-------------------|
| phi-4 | 8.4GB | General reasoning, summarization | `phi-4` |
| deepseek-r1-7b | 4.7GB | Chain-of-thought reasoning | `deepseek-r1-7b` |
| qwen2.5-7b | 4.5GB | Code generation, technical tasks | `qwen2.5-7b` |
| llama-3.2-3b | 2.0GB | Fast responses, simple queries | `llama-3.2-3b` |
| phi-3-medium-128k | 8.6GB | Long context (128K), doc analysis | `phi-3-medium-128k` |
| granite-8b-code-128k | 4.5GB | Code analysis, enterprise | `granite-8b-code-128k` |

---

## Method 2: Custom Agent File (.agent.md)

Create a custom agent that uses your local models with specific instructions.

### Step 1: Create Workspace Agent

Run `Cmd+Shift+P` > "Chat: New Custom Agent" > Select "Workspace"

This creates `.github/agents/local-assistant.agent.md`

### Step 2: Define the Agent

```markdown
---
name: local-ai
description: Local AI assistant using self-hosted models
model: phi-4
endpoint: http://localhost:8085/v1/chat/completions
apiKey: not-required
tools:
  - codebase
  - terminal
  - web
---

# Local AI Assistant

You are a helpful coding assistant running on local infrastructure. You have access to:

## Available Models

When reasoning is needed, suggest switching to deepseek-r1-7b.
For code generation, use qwen2.5-7b.
For quick responses, use llama-3.2-3b.

## Knowledge Sources

You can reference:
- **code-reference-engine**: Curated code examples at semantic-search-service:8081
- **Books**: Technical documentation from ai-platform-data/books/
- **Neo4j Graph**: Concept relationships and cross-references

## Instructions

1. When asked about design patterns, query code-reference-engine first
2. Always provide citations when referencing book content
3. Use Chicago-style citations for technical references
4. Prefer local model inference over external APIs

## Context

The user is working in a multi-service AI platform with:
- inference-service:8085 (local LLM inference)
- llm-gateway:8080 (routing to external/local LLMs)
- semantic-search-service:8081 (Qdrant vector search)
- ai-agents:8082 (agent orchestration)
- Code-Orchestrator-Service:8083 (workflow orchestration)
```

### Step 3: Create a Tools-Enabled Agent

For more powerful agents with tool access, create:

`.github/agents/unified-retriever.agent.md`:

```markdown
---
name: knowledge-retriever
description: Unified knowledge retrieval agent with access to all knowledge sources
model: phi-4
endpoint: http://localhost:8085/v1/chat/completions
tools:
  - codebase
  - terminal
---

# Unified Knowledge Retriever

You are a knowledge retrieval agent that can query:

1. **Qdrant (semantic-search-service:8081)**
   - Vector similarity search for concepts
   - Query: `curl http://localhost:8081/v1/search -d '{"query": "..."}'`

2. **Neo4j (bolt://localhost:7687)**
   - Graph traversal for relationships
   - Cypher queries for concept → code mappings

3. **code-reference-engine**
   - Curated code examples by pattern/concept
   - GitHub repo integration

4. **Books/JSON**
   - Technical textbooks in ai-platform-data/books/
   - Enriched JSON with citations

## Workflow

When asked a question:
1. Decompose into concepts
2. Query semantic-search for relevant passages
3. Query Neo4j for related code references
4. Synthesize response with citations

## Citation Format

Use Chicago-style footnotes:
[^1]: Author, "Title" (Publisher, Year), Page.
```

---

## Method 3: VS Code Extension with Agents Toolkit

For deep integration, build a VS Code extension that uses your ai-agents service.

### Step 1: Scaffold Extension

```bash
# Create extension scaffold
npx yo code

# Select: New Extension (TypeScript)
# Name: ai-platform-copilot
# Publisher: your-org
```

### Step 2: Add Chat Participant

In `src/extension.ts`:

```typescript
import * as vscode from 'vscode';

const INFERENCE_SERVICE_URL = 'http://localhost:8085';
const AI_AGENTS_URL = 'http://localhost:8082';

export function activate(context: vscode.ExtensionContext) {
    // Register chat participant
    const participant = vscode.chat.createChatParticipant(
        'ai-platform.assistant',
        async (
            request: vscode.ChatRequest,
            context: vscode.ChatContext,
            response: vscode.ChatResponseStream,
            token: vscode.CancellationToken
        ) => {
            // Determine model based on command
            let model = 'phi-4';
            if (request.command === 'code') {
                model = 'qwen2.5-7b';
            } else if (request.command === 'think') {
                model = 'deepseek-r1-7b';
            } else if (request.command === 'fast') {
                model = 'llama-3.2-3b';
            }

            // Stream response from inference-service
            const res = await fetch(`${INFERENCE_SERVICE_URL}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    messages: [
                        { role: 'system', content: 'You are a helpful coding assistant.' },
                        { role: 'user', content: request.prompt }
                    ],
                    stream: true
                })
            });

            // Handle SSE stream
            const reader = res.body?.getReader();
            const decoder = new TextDecoder();
            
            while (reader) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.startsWith('data: '));
                
                for (const line of lines) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    
                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content) {
                            response.markdown(content);
                        }
                    } catch (e) {
                        // Skip malformed chunks
                    }
                }
            }
        }
    );

    // Add commands
    participant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'icon.png');
    participant.followupProvider = {
        provideFollowups: () => [
            { prompt: 'Explain this code', label: 'Explain' },
            { prompt: 'Write tests', label: 'Tests' },
            { prompt: 'Find similar patterns', label: 'Patterns' }
        ]
    };

    context.subscriptions.push(participant);
}
```

### Step 3: Add Unified Retrieval Tool

```typescript
// src/tools/unifiedRetrieval.ts
import * as vscode from 'vscode';

const SEMANTIC_SEARCH_URL = 'http://localhost:8081';

export function registerUnifiedRetrievalTool(context: vscode.ExtensionContext) {
    const tool = vscode.lm.registerTool('ai-platform.unifiedRetrieval', {
        description: 'Search across code-reference-engine, books, and Neo4j',
        inputSchema: {
            type: 'object',
            properties: {
                query: { type: 'string', description: 'Search query' },
                scope: { 
                    type: 'string', 
                    enum: ['all', 'code', 'books', 'graph'],
                    description: 'Knowledge source scope'
                }
            },
            required: ['query']
        },
        run: async (input: { query: string; scope?: string }) => {
            const { query, scope = 'all' } = input;
            
            // Query semantic-search-service
            const response = await fetch(`${SEMANTIC_SEARCH_URL}/v1/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    filters: scope !== 'all' ? { source_type: scope } : undefined,
                    top_k: 5
                })
            });
            
            const results = await response.json();
            return new vscode.LanguageModelToolResult([
                new vscode.LanguageModelTextPart(JSON.stringify(results, null, 2))
            ]);
        }
    });

    context.subscriptions.push(tool);
}
```

### Step 4: Package.json Contributions

```json
{
  "contributes": {
    "chatParticipants": [
      {
        "id": "ai-platform.assistant",
        "name": "ai-platform",
        "description": "Local AI assistant using self-hosted models",
        "isSticky": true,
        "commands": [
          {
            "name": "code",
            "description": "Generate code using Qwen 2.5"
          },
          {
            "name": "think",
            "description": "Deep reasoning using DeepSeek R1"
          },
          {
            "name": "fast",
            "description": "Quick response using Llama 3.2"
          },
          {
            "name": "search",
            "description": "Search knowledge base"
          }
        ]
      }
    ],
    "languageModelTools": [
      {
        "name": "ai-platform.unifiedRetrieval",
        "displayName": "Unified Knowledge Search",
        "canBeReferencedInPrompt": true,
        "toolReferenceName": "search",
        "modelDescription": "Search across code examples, books, and concept graphs"
      }
    ]
  }
}
```

---

## Integration with Kitchen Brigade Architecture (WBS_KITCHEN_BRIGADE.md)

The VS Code extension becomes the **user interface** for the full Kitchen Brigade pipeline. Here's how a query flows through the system:

### Complete Flow: VS Code → Kitchen Brigade → Grounded Response

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VS Code Extension (Method 3)                         │
│                                                                              │
│   User: @ai-platform /search "Where is the rate limiter implemented?"       │
│                                                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ai-agents:8082 MCP Server (WBS-KB8)                    │
│                                                                              │
│   POST /v1/pipelines/cross-reference/run                                    │
│   { "query": "Where is the rate limiter implemented?", "scope": "all" }     │
│                                                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               CrossReferencePipeline (WBS-KB6)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: Initial Evidence Gathering (WBS-AGT24 UnifiedRetriever)       │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │   Qdrant    │  │   Neo4j     │  │ code-ref    │  │    Books    │  │ │
│  │  │  (Vector)   │  │   (Graph)   │  │   engine    │  │   (JSON)    │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │ │
│  │         │                │                │                │          │ │
│  │         └────────────────┴────────────────┴────────────────┘          │ │
│  │                                   │                                    │ │
│  │                                   ▼                                    │ │
│  │                    MixedCitation Evidence (v1)                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: LLM Discussion Loop (WBS-KB1) ──── ITERATION CYCLE ────────┐ │ │
│  │                                                                      │ │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │ │ │
│  │  │  phi-4        │  │  qwen2.5-7b   │  │  deepseek-r1  │           │ │ │
│  │  │  (General)    │  │  (Coder)      │  │  (Thinker)    │           │ │ │
│  │  │               │  │               │  │               │           │ │ │
│  │  │  Analysis A   │  │  Analysis B   │  │  Analysis C   │           │ │ │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │ │ │
│  │          │                  │                  │                    │ │ │
│  │          └──────────────────┴──────────────────┘                    │ │ │
│  │                             │                                       │ │ │
│  │                             ▼                                       │ │ │
│  │  ┌────────────────────────────────────────────────────────────────┐│ │ │
│  │  │ Agreement Engine (WBS-KB4)                                     ││ │ │
│  │  │                                                                ││ │ │
│  │  │  Agreement Score: 0.67 (below 0.85 threshold)                  ││ │ │
│  │  │  Disagreement: "Location of rate limiter middleware"           ││ │ │
│  │  └────────────────────────────────────────────────────────────────┘│ │ │
│  │                             │                                       │ │ │
│  │                             ▼                                       │ │ │
│  │  ┌────────────────────────────────────────────────────────────────┐│ │ │
│  │  │ Information Request Detection (WBS-KB2)                        ││ │ │
│  │  │                                                                ││ │ │
│  │  │  Request: "Need middleware configuration in llm-gateway"       ││ │ │
│  │  │  Source Types: ["code"], Priority: HIGH                        ││ │ │
│  │  └────────────────────────────────────────────────────────────────┘│ │ │
│  │                             │                                       │ │ │
│  │                             ▼                                       │ │ │
│  │  ┌────────────────────────────────────────────────────────────────┐│ │ │
│  │  │ Iterative Evidence Gathering (WBS-KB3)                         ││ │ │
│  │  │                                                                ││ │ │
│  │  │  → Query: "rate limiter middleware llm-gateway"                ││ │ │
│  │  │  → Found: src/api/middleware/rate_limit.py L45-89              ││ │ │
│  │  │  → Merged into evidence (v2)                                   ││ │ │
│  │  └────────────────────────────────────────────────────────────────┘│ │ │
│  │                             │                                       │ │ │
│  │                             │  ◄─── CYCLE 2 with new evidence       │ │ │
│  │                             │                                       │ │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │ │ │
│  │  │  phi-4        │  │  qwen2.5-7b   │  │  deepseek-r1  │           │ │ │
│  │  │  Analysis A'  │  │  Analysis B'  │  │  Analysis C'  │           │ │ │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │ │ │
│  │          │                  │                  │                    │ │ │
│  │          └──────────────────┴──────────────────┘                    │ │ │
│  │                             │                                       │ │ │
│  │                             ▼                                       │ │ │
│  │  ┌────────────────────────────────────────────────────────────────┐│ │ │
│  │  │ Agreement Engine: Score 0.92 ✓ (above threshold)               ││ │ │
│  │  └────────────────────────────────────────────────────────────────┘│ │ │
│  └───────────────────────────────────────────────────────────────────┘│ │
│                                      │                                  │ │
│                                      ▼                                  │ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: Consensus Synthesis (WBS-KB4)                             │ │
│  │                                                                     │ │
│  │  Synthesize analyses into unified response with provenance:        │ │
│  │  "The rate limiter is implemented in llm-gateway at                │ │
│  │   src/api/middleware/rate_limit.py [^1]. It uses a token bucket    │ │
│  │   algorithm [^2] with Redis for distributed state [^3]."           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                      │                                   │
│                                      ▼                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 4: Audit Validation (WBS-KB5)                                │ │
│  │                                                                     │ │
│  │  POST audit-service:8084/v1/validate                               │ │
│  │  { citations: [...], discussion_history: [...] }                   │ │
│  │                                                                     │ │
│  │  ✓ [^1] rate_limit.py exists, line 45-89 verified                  │ │
│  │  ✓ [^2] Token bucket pattern matches code-reference-engine         │ │
│  │  ✓ [^3] Redis integration confirmed in config                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                      │                                   │
│                                      ▼                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 5: Code Validation (WBS-KB7 - Optional)                      │ │
│  │                                                                     │ │
│  │  Code-Orchestrator:8083                                            │ │
│  │  → CodeT5+ keyword extraction                                      │ │
│  │  → GraphCodeBERT term validation                                   │ │
│  │  → SonarQube quality metrics                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GroundedResponse (WBS-KB6)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  {                                                                           │
│    "content": "The rate limiter is implemented in llm-gateway at             │
│                src/api/middleware/rate_limit.py [^1]. It uses a token        │
│                bucket algorithm [^2] with Redis for distributed state [^3].",│
│    "citations": [                                                            │
│      { "id": 1, "source": "llm-gateway/src/api/middleware/rate_limit.py",   │
│        "lines": "45-89", "type": "code" },                                   │
│      { "id": 2, "source": "code-reference-engine/backend/rate-limiting",    │
│        "type": "pattern" },                                                  │
│      { "id": 3, "source": "llm-gateway/config/redis.yaml",                  │
│        "type": "config" }                                                    │
│    ],                                                                        │
│    "footnotes": [                                                            │
│      "[^1]: llm-gateway, rate_limit.py, Lines 45-89",                       │
│      "[^2]: Token Bucket Pattern, code-reference-engine",                    │
│      "[^3]: Redis Configuration, llm-gateway"                                │
│    ],                                                                        │
│    "metadata": {                                                             │
│      "cycles_used": 2,                                                       │
│      "participants": ["phi-4", "qwen2.5-7b", "deepseek-r1-7b"],             │
│      "agreement_score": 0.92,                                                │
│      "sources_consulted": ["Qdrant", "Neo4j", "code-reference-engine"],     │
│      "processing_time_ms": 8420                                              │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VS Code Extension Renders Response                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ The rate limiter is implemented in llm-gateway at                      │ │
│  │ [src/api/middleware/rate_limit.py](vscode://file/...) [^1].            │ │
│  │                                                                        │ │
│  │ It uses a **token bucket algorithm** [^2] with Redis for distributed  │ │
│  │ state [^3].                                                            │ │
│  │                                                                        │ │
│  │ ────────────────────────────────────────────────────────────────────── │ │
│  │ **References:**                                                        │ │
│  │ [^1]: llm-gateway, rate_limit.py, Lines 45-89                         │ │
│  │ [^2]: Token Bucket Pattern, code-reference-engine                      │ │
│  │ [^3]: Redis Configuration, llm-gateway                                 │ │
│  │                                                                        │ │
│  │ *2 discussion cycles | 3 LLMs agreed (92%) | 8.4s*                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### VS Code Extension → Kitchen Brigade Tool Mapping

| VS Code Tool | Kitchen Brigade Component | WBS Block |
|--------------|--------------------------|-----------|
| `cross_reference` | CrossReferencePipeline | WBS-KB6 |
| `generate_code` | CodeGenerationPipeline + discussion | WBS-AGT16 + KB1 |
| `analyze_code` | AnalyzeArtifact + Code-Orchestrator | WBS-KB7 |
| `explain_code` | Discussion Loop + Book citations | WBS-KB1 + KB3 |
| `summarize` | SummarizationPipeline (Map-Reduce) | WBS-KB10 |

### Extension Implementation: Calling Kitchen Brigade

```typescript
// src/tools/crossReference.ts
import * as vscode from 'vscode';

const AI_AGENTS_URL = 'http://localhost:8082';

export function registerCrossReferenceTool(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.lm.registerTool('ai-platform.crossReference', {
            description: 'Search across code, books, and graphs with multi-LLM discussion',
            inputSchema: {
                type: 'object',
                properties: {
                    query: { type: 'string', description: 'Your question' },
                    scope: { 
                        type: 'string', 
                        enum: ['all', 'code', 'books', 'graph'],
                        default: 'all'
                    },
                    maxCycles: { type: 'number', default: 5 }
                },
                required: ['query']
            },
            run: async (input) => {
                // Call the Kitchen Brigade CrossReferencePipeline
                const response = await fetch(
                    `${AI_AGENTS_URL}/v1/pipelines/cross-reference/run`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: input.query,
                            scope: input.scope || 'all',
                            max_cycles: input.maxCycles || 5,
                            participants: ['phi-4', 'qwen2.5-7b', 'deepseek-r1-7b']
                        })
                    }
                );

                const result: GroundedResponse = await response.json();
                
                // Format for VS Code display
                const formatted = formatGroundedResponse(result);
                
                return new vscode.LanguageModelToolResult([
                    new vscode.LanguageModelTextPart(formatted)
                ]);
            }
        })
    );
}

function formatGroundedResponse(response: GroundedResponse): string {
    let output = response.content + '\n\n';
    output += '---\n**References:**\n';
    
    for (const footnote of response.footnotes) {
        output += footnote + '\n';
    }
    
    output += `\n*${response.metadata.cycles_used} cycles | `;
    output += `${response.metadata.participants.length} LLMs agreed `;
    output += `(${Math.round(response.metadata.agreement_score * 100)}%) | `;
    output += `${(response.metadata.processing_time_ms / 1000).toFixed(1)}s*`;
    
    return output;
}
```

### MCP Server Alternative (WBS-KB8)

The Kitchen Brigade also exposes an **MCP Server** that VS Code can use directly without building an extension:

```json
// .vscode/mcp.json (in your workspace)
{
    "servers": {
        "ai-platform": {
            "command": "python",
            "args": ["-m", "src.mcp.server"],
            "cwd": "/Users/kevintoles/POC/ai-agents",
            "env": {
                "INFERENCE_SERVICE_URL": "http://localhost:8085",
                "SEMANTIC_SEARCH_URL": "http://localhost:8081",
                "AUDIT_SERVICE_URL": "http://localhost:8084"
            }
        }
    }
}
```

With MCP configured, VS Code Copilot gains these tools automatically:
- `cross_reference` — Full Kitchen Brigade pipeline
- `generate_code` — Code generation with citations
- `analyze_code` — Code analysis with Code-Orchestrator
- `explain_code` — Explanations with textbook references
- `summarize` — Map-Reduce summarization for long documents

---

## Quick Start Commands

```bash
# Start inference-service with Metal acceleration
cd /Users/kevintoles/POC/inference-service
source .venv/bin/activate
export INFERENCE_GPU_LAYERS=-1
export INFERENCE_MODELS_DIR=/Users/kevintoles/POC/ai-models/models
export INFERENCE_DEFAULT_PRESET=D4
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085

# Or use the VS Code task:
# Cmd+Shift+P > "Tasks: Run Task" > "Start Inference Service (Native Metal)"
```

---

## Troubleshooting

### Model Not Loading

```bash
# Check available models
curl http://localhost:8085/v1/models

# Load specific model
curl -X POST http://localhost:8085/v1/models/phi-4/load
```

### Connection Refused

Ensure inference-service is running on port 8085:

```bash
lsof -i :8085
```

### Slow Responses

For Mac Metal acceleration, ensure GPU layers are configured:
- phi-4: 35 layers (hybrid for 16GB RAM)
- Smaller models: -1 (all GPU)

---

---

## Multi-User Deployment (Mini-Server)

Once the platform is hosted on your mini-server, additional users can connect from their VS Code instances over the network.

### Server-Side Configuration

#### 1. Network Exposure

Ensure all services bind to `0.0.0.0` (not `127.0.0.1`) so they're accessible from the network:

```bash
# inference-service (already configured)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085

# ai-agents
python -m uvicorn src.main:app --host 0.0.0.0 --port 8082

# semantic-search-service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8081

# llm-gateway
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
```

#### 2. Service Discovery (docker-compose.yml)

For production deployment, use Docker Compose:

```yaml
# /Users/kevintoles/POC/ai-platform-data/docker/docker-compose.production.yml
version: "3.9"
services:
  inference-service:
    build: ../inference-service
    ports:
      - "8085:8085"
    environment:
      - INFERENCE_GPU_LAYERS=-1
      - INFERENCE_MODELS_DIR=/models
    volumes:
      - /Users/kevintoles/POC/ai-models/models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  ai-agents:
    build: ../ai-agents
    ports:
      - "8082:8082"
    environment:
      - INFERENCE_SERVICE_URL=http://inference-service:8085
      - SEMANTIC_SEARCH_URL=http://semantic-search:8081
    depends_on:
      - inference-service
      - semantic-search

  semantic-search:
    build: ../semantic-search-service
    ports:
      - "8081:8081"
    environment:
      - QDRANT_URL=http://qdrant:6333

  llm-gateway:
    build: ../llm-gateway
    ports:
      - "8080:8080"
    environment:
      - INFERENCE_SERVICE_URL=http://inference-service:8085
      - AGENTS_SERVICE_URL=http://ai-agents:8082
    depends_on:
      - inference-service
      - ai-agents

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your-password
    volumes:
      - neo4j_data:/data

volumes:
  qdrant_data:
  neo4j_data:
```

#### 3. Firewall Configuration (macOS)

```bash
# Allow ports through macOS firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/python3

# Or use pfctl for specific ports
echo "pass in proto tcp from any to any port {8080, 8081, 8082, 8085}" | sudo pfctl -ef -
```

#### 4. DNS/Hostname Setup

For easier access, configure a hostname:

```bash
# On the mini-server, edit /etc/hosts or use mDNS
# Mini-server hostname: ai-platform.local

# Users can then use:
# http://ai-platform.local:8085/v1/chat/completions
```

---

### Client-Side Configuration (Remote Users)

#### Method 1: Direct OpenAI-Compatible API

Remote users update their VS Code settings to point to the server:

```json
{
  "github.copilot.chat.customOAIModels": [
    {
      "name": "phi-4-team",
      "displayName": "Phi-4 (Team Server)",
      "endpoint": "http://ai-platform.local:8085/v1/chat/completions",
      "model": "phi-4",
      "apiKey": "team-api-key-here",
      "capabilities": {
        "chat": true,
        "codeCompletion": true
      }
    },
    {
      "name": "deepseek-team",
      "displayName": "DeepSeek R1 (Team Server)",
      "endpoint": "http://ai-platform.local:8085/v1/chat/completions",
      "model": "deepseek-r1-7b",
      "apiKey": "team-api-key-here",
      "capabilities": {
        "chat": true,
        "reasoning": true
      }
    }
  ]
}
```

#### Method 2: Custom Agent File (Shared)

Create a team agent file in a shared repository:

```markdown
<!-- .github/copilot-instructions.md or .agent.md -->
# Team AI Platform Agent

## Instructions
You have access to the team AI platform at ai-platform.local.
Use the Kitchen Brigade pipeline for grounded responses.

## API Endpoints
- Inference: http://ai-platform.local:8085/v1/chat/completions
- Agents: http://ai-platform.local:8082/v1/pipelines/
- Search: http://ai-platform.local:8081/v1/search

## Capabilities
- Multi-model selection (phi-4, deepseek-r1, qwen2.5)
- Citation-backed responses via Kitchen Brigade
- Code analysis via Code-Orchestrator
```

#### Method 3: VS Code Extension (Distributed)

Package the extension with server URL as configuration:

```json
// extension's package.json - contributes.configuration
{
  "contributes": {
    "configuration": {
      "title": "AI Platform",
      "properties": {
        "aiPlatform.serverUrl": {
          "type": "string",
          "default": "http://localhost:8085",
          "description": "AI Platform server URL (e.g., http://ai-platform.local:8085)"
        },
        "aiPlatform.agentsUrl": {
          "type": "string",
          "default": "http://localhost:8082",
          "description": "AI Agents service URL"
        },
        "aiPlatform.apiKey": {
          "type": "string",
          "default": "",
          "description": "API key for authentication"
        }
      }
    }
  }
}
```

Remote users configure in their VS Code settings:

```json
{
  "aiPlatform.serverUrl": "http://ai-platform.local:8085",
  "aiPlatform.agentsUrl": "http://ai-platform.local:8082",
  "aiPlatform.apiKey": "user-specific-key"
}
```

#### Method 4: MCP Server (Remote)

Remote users configure their MCP settings to connect to the server:

```json
// User's ~/.vscode/mcp.json
{
    "servers": {
        "ai-platform-remote": {
            "type": "http",
            "url": "http://ai-platform.local:8082/mcp",
            "headers": {
                "Authorization": "Bearer user-api-key"
            }
        }
    }
}
```

---

### Authentication & Multi-Tenancy

#### Simple API Key Authentication

Add API key validation to your services:

```python
# src/middleware/auth.py
from fastapi import HTTPException, Header
from typing import Optional
import os

API_KEYS = os.getenv("API_KEYS", "").split(",")  # Comma-separated keys

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEYS or API_KEYS == ['']:
        return None  # No auth required (dev mode)
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    # Support "Bearer <key>" or just "<key>"
    key = authorization.replace("Bearer ", "")
    if key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return key
```

Add to FastAPI app:

```python
# src/main.py
from fastapi import Depends
from src.middleware.auth import verify_api_key

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    # ... existing logic
```

#### User-Specific Rate Limiting

```python
# src/middleware/rate_limit.py
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_key: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[user_key] = [
            ts for ts in self.requests[user_key] if ts > minute_ago
        ]
        
        if len(self.requests[user_key]) >= self.rpm:
            return False
        
        self.requests[user_key].append(now)
        return True
```

---

### Network Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MINI-SERVER (ai-platform.local)                  │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ inference-svc   │    │    ai-agents    │    │ semantic-search │     │
│  │     :8085       │◄───│      :8082      │───►│     :8081       │     │
│  │                 │    │                 │    │                 │     │
│  │  phi-4, qwen,   │    │ Kitchen Brigade │    │     Qdrant      │     │
│  │  deepseek, etc  │    │    Pipeline     │    │   Vector DB     │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           ▲                      │                      │              │
│           │                      ▼                      │              │
│           │              ┌─────────────────┐            │              │
│           │              │   llm-gateway   │            │              │
│           └──────────────│      :8080      │────────────┘              │
│                          │   (Unified API) │                           │
│                          └─────────────────┘                           │
│                                  │                                     │
│                                  │ 0.0.0.0 (all interfaces)            │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │        LAN / VPN            │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Developer 1  │          │  Developer 2  │          │  Developer 3  │
│    VS Code    │          │    VS Code    │          │    VS Code    │
│               │          │               │          │               │
│ Settings:     │          │ Settings:     │          │ Settings:     │
│ endpoint:     │          │ endpoint:     │          │ endpoint:     │
│ ai-platform   │          │ ai-platform   │          │ ai-platform   │
│ .local:8085   │          │ .local:8085   │          │ .local:8085   │
└───────────────┘          └───────────────┘          └───────────────┘
```

---

### Quick Setup Script for Remote Users

Create a script users can run to configure their VS Code:

```bash
#!/bin/bash
# setup-ai-platform.sh
# Usage: ./setup-ai-platform.sh <server-url> <api-key>

SERVER_URL="${1:-http://ai-platform.local:8085}"
API_KEY="${2:-demo-key}"

# VS Code settings file location
SETTINGS_DIR="$HOME/Library/Application Support/Code/User"
SETTINGS_FILE="$SETTINGS_DIR/settings.json"

# Backup existing settings
cp "$SETTINGS_FILE" "$SETTINGS_FILE.backup" 2>/dev/null

# Add AI Platform configuration using jq
jq --arg url "$SERVER_URL" --arg key "$API_KEY" '
  . + {
    "github.copilot.chat.customOAIModels": [
      {
        "name": "phi-4-team",
        "displayName": "Phi-4 (Team Server)",
        "endpoint": ($url + "/v1/chat/completions"),
        "model": "phi-4",
        "apiKey": $key,
        "capabilities": {"chat": true, "codeCompletion": true}
      },
      {
        "name": "deepseek-team",
        "displayName": "DeepSeek R1 (Team Server)",
        "endpoint": ($url + "/v1/chat/completions"),
        "model": "deepseek-r1-7b",
        "apiKey": $key,
        "capabilities": {"chat": true, "reasoning": true}
      }
    ]
  }
' "$SETTINGS_FILE" > "$SETTINGS_FILE.tmp" && mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"

echo "✅ AI Platform configured!"
echo "   Server: $SERVER_URL"
echo "   Restart VS Code to apply changes."
```

---

### Security Considerations

| Concern | Recommendation |
|---------|----------------|
| **Network Exposure** | Use VPN or restrict to LAN only |
| **API Keys** | Rotate keys monthly; use unique keys per user |
| **HTTPS** | Add reverse proxy (nginx/caddy) with TLS for production |
| **Rate Limiting** | Implement per-user rate limits to prevent abuse |
| **Audit Logging** | Log all requests to audit-service for compliance |
| **Model Access** | Consider per-model permissions (some users get phi-4, others get deepseek) |
| **Google Workspace SSO** | Restrict access to your organization's domain only |

---

## Google Workspace SSO Integration

Restrict platform access to members of your Google Workspace organization only. This ensures only `@yourcompany.com` users can authenticate.

### Prerequisites

1. **Google Cloud Console access** - Admin access to create OAuth 2.0 credentials
2. **Google Workspace domain** - e.g., `yourcompany.com`
3. **HTTPS enabled** - Required for OAuth (use Caddy or nginx)

### Step 1: Create Google OAuth 2.0 Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Navigate to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **OAuth client ID**
5. Configure the OAuth consent screen:
   - User Type: **Internal** (restricts to your Workspace domain)
   - App name: "AI Platform"
   - Authorized domains: `ai-platform.local` or your domain
6. Create OAuth client ID:
   - Application type: **Web application**
   - Authorized redirect URIs:
     ```
     https://ai-platform.local/auth/callback
     https://ai-platform.local:8080/auth/callback
     ```
7. Save **Client ID** and **Client Secret**

### Step 2: Install Dependencies

```bash
cd /Users/kevintoles/POC/llm-gateway
pip install authlib httpx python-jose[cryptography]
```

### Step 3: Implement Google OAuth Middleware

```python
# src/auth/google_sso.py
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from fastapi import HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional
import os

# Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
ALLOWED_DOMAIN = os.getenv("GOOGLE_ALLOWED_DOMAIN", "yourcompany.com")
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# OAuth setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'hd': ALLOWED_DOMAIN,  # Restricts to your Workspace domain
    }
)


class GoogleUser:
    """Authenticated Google Workspace user."""
    def __init__(self, email: str, name: str, picture: str, domain: str):
        self.email = email
        self.name = name
        self.picture = picture
        self.domain = domain
    
    @property
    def is_valid_domain(self) -> bool:
        return self.email.endswith(f"@{ALLOWED_DOMAIN}")


def create_access_token(user: GoogleUser) -> str:
    """Create JWT token for authenticated user."""
    payload = {
        "sub": user.email,
        "name": user.name,
        "domain": user.domain,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(request: Request) -> GoogleUser:
    """
    Dependency to get current authenticated user.
    Checks for JWT in Authorization header or cookie.
    """
    # Check Authorization header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    else:
        # Fall back to cookie
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login at /auth/login",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Verify domain restriction
    email = payload.get("sub", "")
    if not email.endswith(f"@{ALLOWED_DOMAIN}"):
        raise HTTPException(
            status_code=403,
            detail=f"Access restricted to {ALLOWED_DOMAIN} organization members"
        )
    
    return GoogleUser(
        email=email,
        name=payload.get("name", ""),
        picture="",
        domain=ALLOWED_DOMAIN
    )


# Optional: Less strict dependency that allows unauthenticated access
async def get_optional_user(request: Request) -> Optional[GoogleUser]:
    """Get user if authenticated, None otherwise."""
    try:
        return await get_current_user(request)
    except HTTPException:
        return None
```

### Step 4: Add Auth Routes

```python
# src/api/routes/auth.py
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from src.auth.google_sso import (
    oauth, GoogleUser, create_access_token, 
    ALLOWED_DOMAIN, get_current_user
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth login."""
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        
        if not user_info:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to get user info from Google"}
            )
        
        email = user_info.get("email", "")
        
        # Verify domain
        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            return JSONResponse(
                status_code=403,
                content={
                    "error": f"Access denied. Only {ALLOWED_DOMAIN} members allowed.",
                    "your_email": email
                }
            )
        
        # Create user and JWT
        user = GoogleUser(
            email=email,
            name=user_info.get("name", ""),
            picture=user_info.get("picture", ""),
            domain=email.split("@")[1]
        )
        
        access_token = create_access_token(user)
        
        # Return token (for API clients) or set cookie (for browser)
        response = RedirectResponse(url="/auth/success")
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,  # Requires HTTPS
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Authentication failed: {str(e)}"}
        )


@router.get("/success")
async def auth_success(request: Request, user: GoogleUser = Depends(get_current_user)):
    """Show authentication success with token for VS Code."""
    token = request.cookies.get("access_token")
    return JSONResponse({
        "status": "authenticated",
        "user": {
            "email": user.email,
            "name": user.name,
            "domain": user.domain
        },
        "token": token,
        "instructions": {
            "vs_code_settings": {
                "aiPlatform.apiKey": token,
                "aiPlatform.serverUrl": "https://ai-platform.local:8080"
            },
            "curl_example": f"curl -H 'Authorization: Bearer {token}' https://ai-platform.local:8080/v1/models"
        }
    })


@router.get("/logout")
async def logout():
    """Clear authentication cookie."""
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response


@router.get("/me")
async def get_me(user: GoogleUser = Depends(get_current_user)):
    """Get current authenticated user info."""
    return {
        "email": user.email,
        "name": user.name,
        "domain": user.domain,
        "organization": ALLOWED_DOMAIN
    }
```

### Step 5: Protect API Endpoints

```python
# src/main.py
from fastapi import FastAPI, Depends
from starlette.middleware.sessions import SessionMiddleware
from src.auth.google_sso import get_current_user, GoogleUser
from src.api.routes import auth

app = FastAPI(title="AI Platform")

# Required for OAuth state management
app.add_middleware(SessionMiddleware, secret_key="your-session-secret")

# Include auth routes (unprotected)
app.include_router(auth.router)


# Protected endpoints
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    user: GoogleUser = Depends(get_current_user)  # Requires auth
):
    # Log user for audit
    logger.info(f"Chat request from {user.email}")
    # ... existing logic


@app.get("/v1/models")
async def list_models(user: GoogleUser = Depends(get_current_user)):
    # ... existing logic


# Health endpoint remains unprotected
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Step 6: Environment Configuration

```bash
# .env (server-side)
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_ALLOWED_DOMAIN=yourcompany.com
JWT_SECRET=generate-a-secure-random-string-here

# Generate a secure JWT secret:
# python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 7: VS Code Client Authentication Flow

Users authenticate via browser, then copy their token to VS Code:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VS Code Client Authentication Flow                    │
│                                                                         │
│  1. User opens: https://ai-platform.local/auth/login                    │
│                                                                         │
│  2. Redirects to Google:                                                │
│     accounts.google.com/o/oauth2/auth?hd=yourcompany.com                │
│                                                                         │
│  3. User signs in with @yourcompany.com account                         │
│     ⚠️  Non-org emails are REJECTED                                     │
│                                                                         │
│  4. Callback returns JWT token                                          │
│                                                                         │
│  5. User copies token to VS Code settings:                              │
│     {                                                                   │
│       "aiPlatform.apiKey": "eyJhbGciOiJIUzI1NiIs..."                   │
│     }                                                                   │
│                                                                         │
│  6. All API requests include: Authorization: Bearer <token>             │
└─────────────────────────────────────────────────────────────────────────┘
```

### VS Code Extension: Token Refresh Command

Add a command to your VS Code extension for easy re-authentication:

```typescript
// src/extension.ts
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    // Command to open browser for authentication
    const loginCommand = vscode.commands.registerCommand(
        'aiPlatform.login',
        async () => {
            const serverUrl = vscode.workspace
                .getConfiguration('aiPlatform')
                .get<string>('serverUrl', 'https://ai-platform.local');
            
            // Open browser to login page
            vscode.env.openExternal(
                vscode.Uri.parse(`${serverUrl}/auth/login`)
            );
            
            // Prompt user to paste token
            const token = await vscode.window.showInputBox({
                prompt: 'Paste your authentication token from the browser',
                password: true,
                placeHolder: 'eyJhbGciOiJIUzI1NiIs...'
            });
            
            if (token) {
                // Save to settings
                await vscode.workspace
                    .getConfiguration('aiPlatform')
                    .update('apiKey', token, vscode.ConfigurationTarget.Global);
                
                vscode.window.showInformationMessage(
                    'AI Platform: Authentication successful!'
                );
            }
        }
    );
    
    context.subscriptions.push(loginCommand);
}
```

### Docker Compose with Google SSO

```yaml
# docker-compose.production.yml (updated)
services:
  llm-gateway:
    build: ../llm-gateway
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
      - GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
      - GOOGLE_ALLOWED_DOMAIN=yourcompany.com
      - JWT_SECRET=${JWT_SECRET}
      - INFERENCE_SERVICE_URL=http://inference-service:8085
    depends_on:
      - inference-service
```

### Token Validation in Other Services

For services behind llm-gateway, pass the validated user:

```python
# src/middleware/internal_auth.py
from fastapi import Header, HTTPException
from typing import Optional

async def verify_internal_user(
    x_user_email: Optional[str] = Header(None),
    x_user_domain: Optional[str] = Header(None)
):
    """
    Verify user from internal service calls.
    llm-gateway sets these headers after Google SSO validation.
    """
    if not x_user_email:
        raise HTTPException(status_code=401, detail="Missing user context")
    
    return {"email": x_user_email, "domain": x_user_domain}
```

### Security Summary

| Feature | Implementation |
|---------|----------------|
| **Domain Restriction** | `hd=yourcompany.com` in OAuth scope |
| **Token Expiration** | JWT expires after 24 hours |
| **HTTPS Required** | OAuth requires secure redirect URIs |
| **Audit Trail** | User email logged with every request |
| **Token Refresh** | Users re-authenticate via browser |
| **Internal User Type** | Only "Internal" apps visible to org |

---

#### HTTPS with Caddy (Recommended for Production)

```bash
# Caddyfile
ai-platform.local {
    reverse_proxy /v1/chat/completions localhost:8085
    reverse_proxy /v1/pipelines/* localhost:8082
    reverse_proxy /v1/search/* localhost:8081
    reverse_proxy /* localhost:8080
    
    tls internal  # Self-signed cert for local network
}
```

---

## Related Documents

- [inference-service/ARCHITECTURE.md](../inference-service/docs/ARCHITECTURE.md)
- [ai-agents/WBS.md](../ai-agents/docs/WBS.md) - WBS-AGT24 Unified Knowledge Retrieval
- [AGENT_FUNCTIONS_ARCHITECTURE.md](pending/platform/AGENT_FUNCTIONS_ARCHITECTURE.md)
