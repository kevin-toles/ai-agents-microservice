#!/bin/bash
# Kitchen Brigade Interview Preparation Script
# Creates a technical overview for Google interview

OUTPUT_DIR="/tmp/kitchen_brigade"
mkdir -p "$OUTPUT_DIR"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║     KITCHEN BRIGADE - GOOGLE INTERVIEW TECHNICAL OVERVIEW            ║"
echo "║     Generating platform documentation for technical presentation     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Write condensed platform context
cat > "$OUTPUT_DIR/interview_context.txt" <<'CONTEXT'
=== AI CODING PLATFORM - TECHNICAL OVERVIEW ===

DEVELOPMENT TIMELINE:
- llm-document-enhancer: Nov 11, 2025 (original application)
- Core Platform (llm-gateway, semantic-search, ai-agents): Dec 1, 2025
- Code-Orchestrator-Service: Dec 8, 2025
- ai-platform-data, audit-service: Dec 13, 2025
- inference-service: Dec 27, 2025

ARCHITECTURE: "Kitchen Brigade" Model
- Router (llm-gateway:8080): Entry point, routes to LLM providers
- Expeditor (ai-agents:8082): Orchestrates agent workflows
- Cookbook (semantic-search:8081): Hybrid RAG (Qdrant vectors + Neo4j graph)
- Sous Chef (Code-Orchestrator:8083): CodeBERT/GraphCodeBERT/CodeT5+
- Line Cook (inference-service:8085): Local LLM inference (llama-cpp-python + Metal)
- Auditor (audit-service:8084): Pattern compliance, security scanning
- Pantry (ai-platform-data): Reference materials, taxonomies

KEY PROTOCOLS:
1. A2A (Agent-to-Agent): JSON-RPC protocol for agent discovery & communication
2. MCP (Model Context Protocol): Tool standardization for LLM integrations
3. ADK (Agent Development Kit): Google's agent framework patterns

8 AGENT FUNCTIONS:
1. extract_structure - Keywords, concepts, entities
2. summarize_content - Compress while preserving invariants
3. generate_code - Code from spec + context
4. analyze_artifact - Pattern analysis, quality
5. validate_against_spec - Constraint checking
6. synthesize_outputs - Combine artifacts
7. decompose_task - Break into subtasks
8. cross_reference - Find related content across sources

WHAT JUST HAPPENED (Kitchen Brigade Discussion):
- 3 LLMs (Qwen3-8B local, DeepSeek-Chat, GPT-5.2) held iterative discussions
- Round 1: Initial proposals
- Round 2: Research loop via semantic-search (hybrid RAG)
- Round 3: Comparison of outputs
- Resolution Round: Explicit voting on 3 blocking decisions
- Final: Collaborative architecture document generation

DECISIONS MADE:
1. Mode Detection: EXPLICIT with diagnostic auto-detect (unanimous)
2. Config Source: Generated artifact (tiebreaker vote)
3. Implementation: Mode-switch + preflight + readiness specs

WHY INFRASTRUCTURE SHIFT (Hybrid Mode):
- Run local LLMs on Apple Metal for cost reduction
- Use Docker for databases (Neo4j, Qdrant, Redis)
- Run Python services natively for debugging
- Need deterministic startup across 3 modes (Docker/Hybrid/Native)
CONTEXT

echo "━━━ Generating Technical Overview with DeepSeek ━━━"
echo "Creating platform overview for technical and non-technical audiences..."

OVERVIEW_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/interview_context.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a senior technical writer creating documentation for a Google interview. Create a clear, professional technical overview that works for both technical and non-technical audiences. Use clear headings, bullet points, and avoid jargon where possible. Include the ADK and MCP frameworks prominently."
      },
      {
        "role": "user",
        "content": ($context + "\n\nCreate a TECHNICAL OVERVIEW document for a Google interview that:\n\n1. Explains what this platform does in 2-3 sentences for executives\n2. Describes the Kitchen Brigade architecture (with the metaphor explained)\n3. Explains how we just used the platform (multi-LLM discussion with research loops)\n4. Highlights Google ADK and MCP integration\n5. Explains the infrastructure shift to hybrid mode (why local LLMs on Metal)\n6. Shows development velocity (built in ~5 weeks)\n\nFormat as a clean markdown document with clear sections. Make it presentable for a Google interview.")
      }
    ],
    "max_tokens": 3000,
    "temperature": 0.7
  }')

OVERVIEW=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$OVERVIEW_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .')

echo "$OVERVIEW" > "$OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md"
echo "✓ Technical overview saved"
echo ""

echo "━━━ Generating ADK/MCP Deep Dive with GPT-5.2 ━━━"

ADK_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/interview_context.txt" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a Google engineer explaining Agent Development Kit (ADK) and Model Context Protocol (MCP) to an interview panel."
      },
      {
        "role": "user",
        "content": ($context + "\n\nCreate a focused section on ADK and MCP that:\n\n1. Explains what ADK is (Google'\''s agent framework) and how we'\''re using its patterns\n2. Explains what MCP is (Anthropic'\''s tool protocol) and how it standardizes tool calls\n3. Shows how A2A (Agent-to-Agent) protocol enables discovery\n4. Explains why these matter for enterprise AI platforms\n5. Shows our implementation status (Phase 2 of 3)\n\nBe specific about the technical value proposition. This is for Google engineers evaluating architectural decisions.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

ADK_SECTION=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$ADK_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .')

echo "$ADK_SECTION" >> "$OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md"
echo "✓ ADK/MCP section added"
echo ""

echo "━━━ Generating Hybrid Infrastructure Section with Qwen3-8B ━━━"

INFRA_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/interview_context.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are an infrastructure architect explaining a hybrid local development setup."
      },
      {
        "role": "user",
        "content": ($context + "\n\nCreate a section explaining the INFRASTRUCTURE SHIFT:\n\n1. Why hybrid mode? (Local LLMs on Apple Metal for cost, Docker for DBs, native Python for debugging)\n2. The 3 deployment modes (Docker-only, Hybrid, Native)\n3. What we just solved (explicit mode detection, generated config, readiness gates)\n4. Next steps (implement the architecture decision)\n\nKeep it concise and technical. Focus on the practical benefits.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

INFRA_SECTION=$(curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$INFRA_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .')

echo "$INFRA_SECTION" >> "$OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md"
echo "✓ Infrastructure section added"
echo ""

# Add development timeline
cat >> "$OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md" <<'TIMELINE'

---

## Development Timeline

| Date | Milestone | Repository |
|------|-----------|------------|
| Nov 11, 2025 | Original document enhancer application | llm-document-enhancer |
| Dec 1, 2025 | Core platform services launched | llm-gateway, semantic-search, ai-agents |
| Dec 8, 2025 | Code understanding capabilities | Code-Orchestrator-Service |
| Dec 13, 2025 | Reference data & auditing | ai-platform-data, audit-service |
| Dec 27, 2025 | Local LLM inference (Metal) | inference-service |
| Jan 5-6, 2026 | Multi-LLM discussion system | Kitchen Brigade architecture |

**Total Development Time:** ~8 weeks from concept to multi-LLM orchestration platform

---

## What We Just Demonstrated

The Kitchen Brigade discussion you witnessed was the platform orchestrating itself:

1. **3 LLMs participated**: Qwen3-8B (local/Metal), DeepSeek-Chat (API), GPT-5.2 (API)
2. **Iterative refinement**: LLMs requested additional research via `RESEARCH_NEEDED` markers
3. **Semantic search integration**: Queries routed to hybrid RAG (Qdrant vectors + Neo4j graph)
4. **Voting protocol**: Unanimous decision on mode detection, tiebreaker on config source
5. **Collaborative authoring**: Final architecture document written by all 3 LLMs

This demonstrates the platform's core value: **grounded, multi-perspective AI reasoning with retrieval augmentation**.

---

## Key Differentiators

| Capability | Traditional RAG | Kitchen Brigade Platform |
|------------|-----------------|--------------------------|
| LLM Count | Single model | Multi-model orchestration |
| Retrieval | Vector-only | Hybrid (vectors + graph + taxonomy) |
| Grounding | Prompt injection | Citation tracking + audit trails |
| Reasoning | Single-pass | Iterative with research loops |
| Validation | None | Pattern compliance + security scanning |

---

*Document generated by Kitchen Brigade Multi-LLM Discussion System*
*Interview Date: January 6, 2026*
TIMELINE

echo "═══════════════════════════════════════════════════════════════════════"
echo "GOOGLE INTERVIEW OVERVIEW COMPLETE"
echo "Document saved to: $OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

cat "$OUTPUT_DIR/GOOGLE_INTERVIEW_OVERVIEW.md"
