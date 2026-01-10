#!/bin/bash
# Kitchen Brigade Multi-LLM Discussion Loop
# Calls Qwen3-8B (local), DeepSeek, and GPT-5.2 with full context

OUTPUT_DIR="/tmp/kitchen_brigade"
mkdir -p "$OUTPUT_DIR"

# Write context packet to temp file for jq --rawfile
cat > "$OUTPUT_DIR/context_packet.txt" <<'CONTEXT'
=== STAGE 2 CROSS-REFERENCE RESULTS ===

QDRANT VECTOR SEARCH (10 chapters):
| Score | Book | Key Topics |
|-------|------|------------|
| 0.573 | Docker Networking ch5 | Docker Swarm, service discovery |
| 0.569 | Production Kubernetes ch21 | Service Mesh, Istio, Envoy |
| 0.563 | Microservice Architecture ch10 | Bounded contexts, decoupling |
| 0.555 | Seeking SRE ch24 | Service Mesh, sidecar proxy |
| 0.554 | Microservices Theory ch5 | Circuit breaker, bulkhead |

NEO4J GRAPH QUERY (concepts with chapter counts):
| Concept | Chapters | Sample Books |
|---------|----------|--------------|
| service discovery | 22 | Prometheus, Production Kubernetes, Docker Networking |
| Service mesh | 13 | Production Kubernetes, API Security, Infrastructure as Code |
| DEPLOYMENT PIPELINE | 23 | Multiple books |
| microservice architecture | 36 | Distributed patterns books |
| Circuit Breaker | 5 books | Microservices for Enterprise, Microservices in Action |

TAXONOMY (3 tiers, 43 books):
- Tier 1 (Architecture): Building Microservices, Clean Architecture, Release It!, Designing Distributed Systems
- Tier 2 (Implementation): Infrastructure as Code, Terraform, Kubernetes, Docker Deep Dive, Testing books
- Tier 3 (Operational): SRE, Observability Engineering, Prometheus, Security books

=== PREVIOUS SOLUTION PROPOSAL ===

1. DISCOVERY: Mode-aware URL resolution function
   - Docker mode: Use container DNS names (ai-platform-neo4j:7687)
   - Hybrid mode: DBs use localhost (exposed ports), services use localhost
   - Native mode: All localhost

2. CONFIG: Single canonical env var names (no service prefixes)
   - NEO4J_URI (not AI_AGENTS_NEO4J_URI)
   - QDRANT_URL, REDIS_URL
   - INFRASTRUCTURE_MODE env var to switch modes

3. HEALTH: Platform health aggregator endpoint
   - /platform/health that checks all services
   - Returns DEGRADED status visibly (not silent)

4. CONTRACTS: OpenAPI runtime validation middleware

5. PORTS: Fixed port assignments per service

6. STARTUP: Existing tasks.json with mode detection
CONTEXT

# Write problem statement to temp file
cat > "$OUTPUT_DIR/problem.txt" <<'PROBLEM'
PROBLEM STATEMENT:
Local development platform with these components:

SERVICES (Python/FastAPI):
- llm-gateway:8080 (routes to LLM providers)
- semantic-search:8081 (Qdrant + Neo4j hybrid search)
- ai-agents:8082 (orchestrates agents)
- code-orchestrator:8083 (code analysis)
- inference-service:8085 (local LLM inference)

DATABASES:
- Qdrant:6333 (vector DB)
- Neo4j:7687 (graph DB)  
- Redis:6379 (cache)

DEPLOYMENT MODES:
1. Docker-only: All services + DBs in containers
2. Hybrid: DBs in Docker, services run natively (for debugging)
3. Native: Everything runs natively (rare)

CURRENT ISSUES:
1. Service discovery chaos - some use container names, some use localhost
2. Env var proliferation - NEO4J_URI vs AI_AGENTS_NEO4J_URI vs SEMANTIC_SEARCH_NEO4J_URI
3. Silent fallbacks - services fail silently when dependencies are down
4. No contract tests - API schema drift between services
5. Port conflicts when switching modes
6. No unified startup - have to start services manually in order
PROBLEM

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║          KITCHEN BRIGADE MULTI-LLM DISCUSSION LOOP                   ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  ROUND 1: Initial proposals from each LLM                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# ROUND 1: QWEN3-8B (Proposer) via local inference
# ============================================================================
echo "━━━ ROUND 1.1: Qwen3-8B (Proposer) ━━━"
echo "Calling local inference service..."

# Build JSON payload using jq with --rawfile for multiline content
QWEN_PAYLOAD=$(jq -n \
  --rawfile problem "$OUTPUT_DIR/problem.txt" \
  --rawfile context "$OUTPUT_DIR/context_packet.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are an infrastructure architect. You are the PROPOSER in a multi-LLM discussion. Based on the cross-reference research provided, propose improvements to the solution. If you need more research, say RESEARCH_NEEDED: [query]. Be specific with code examples."
      },
      {
        "role": "user",
        "content": ($problem + "\n\n" + $context + "\n\nYour task: Review the previous proposal and suggest specific improvements. Focus on:\n1. Is the mode-aware URL resolution the right approach?\n2. How should services discover each other in Hybrid mode specifically?\n3. What'\''s missing from the health aggregator design?\n\nProvide code examples. You may request RESEARCH_NEEDED if you want patterns from specific books.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round1_qwen.txt" 2>&1

echo "Qwen3-8B response saved to $OUTPUT_DIR/round1_qwen.txt"
echo ""

# ============================================================================
# ROUND 1: DEEPSEEK (Architect) via gateway
# ============================================================================
echo "━━━ ROUND 1.2: DeepSeek-Chat (Architect) ━━━"
echo "Calling via LLM Gateway..."

DEEPSEEK_PAYLOAD=$(jq -n \
  --rawfile problem "$OUTPUT_DIR/problem.txt" \
  --rawfile context "$OUTPUT_DIR/context_packet.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a senior infrastructure architect. You are the ARCHITECT in a multi-LLM discussion. Review the proposal critically and suggest architectural improvements. If you need more research from books, say RESEARCH_NEEDED: [query]."
      },
      {
        "role": "user",
        "content": ($problem + "\n\n" + $context + "\n\nYour task as ARCHITECT:\n1. Critique the mode-aware URL resolution approach - is it too simple or too complex?\n2. The previous proposal lacks detail on Hybrid mode discovery - how exactly should a native Python service find a Docker-hosted Neo4j?\n3. Propose a concrete implementation for the health aggregator\n4. What contract testing approach works for FastAPI services?\n\nProvide architectural diagrams (ASCII) and code examples.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round1_deepseek.txt" 2>&1

echo "DeepSeek response saved to $OUTPUT_DIR/round1_deepseek.txt"
echo ""

# ============================================================================
# ROUND 1: GPT-5.2 (Critic) via gateway
# ============================================================================
echo "━━━ ROUND 1.3: GPT-5.2 (Critic) ━━━"
echo "Calling via LLM Gateway..."

GPT_PAYLOAD=$(jq -n \
  --rawfile problem "$OUTPUT_DIR/problem.txt" \
  --rawfile context "$OUTPUT_DIR/context_packet.txt" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a critical systems engineer. You are the CRITIC in a multi-LLM discussion. Find flaws, edge cases, and failure modes in proposals. If you need more research, say RESEARCH_NEEDED: [query]."
      },
      {
        "role": "user",
        "content": ($problem + "\n\n" + $context + "\n\nYour task as CRITIC:\n1. What failure modes are NOT addressed by the current proposal?\n2. What happens when switching from Docker to Hybrid mode mid-session?\n3. The env var standardization will break existing code - what'\''s the migration path?\n4. Is the health aggregator a single point of failure?\n5. How do you test contract compliance without breaking CI/CD speed?\n\nBe harsh but constructive. Identify specific gaps.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round1_gpt52.txt" 2>&1

echo "GPT-5.2 response saved to $OUTPUT_DIR/round1_gpt52.txt"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "ROUND 1 COMPLETE - Responses saved to $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════════════"

# Display results
echo ""
echo "━━━ QWEN3-8B (Proposer) Response: ━━━"
cat "$OUTPUT_DIR/round1_qwen.txt"
echo ""
echo "━━━ DEEPSEEK (Architect) Response: ━━━"
cat "$OUTPUT_DIR/round1_deepseek.txt"
echo ""
echo "━━━ GPT-5.2 (Critic) Response: ━━━"
cat "$OUTPUT_DIR/round1_gpt52.txt"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Checking for RESEARCH_NEEDED requests..."
grep -l "RESEARCH_NEEDED" "$OUTPUT_DIR"/*.txt 2>/dev/null && echo "Some LLMs requested more research - will trigger Stage 2 loop"
echo "═══════════════════════════════════════════════════════════════════════"
