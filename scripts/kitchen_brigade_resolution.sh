#!/bin/bash
# Kitchen Brigade Resolution Round
# Addresses blocking items and brings Qwen3-8B back with condensed context

OUTPUT_DIR="/tmp/kitchen_brigade"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║     KITCHEN BRIGADE - RESOLUTION ROUND                               ║"
echo "║     Addressing Blocking Items for Sign-Off                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "BLOCKING ITEMS TO RESOLVE:"
echo "  1. Explicit mode vs auto-detect (DECISION REQUIRED)"
echo "  2. Generated config artifact vs runtime service (DECISION REQUIRED)"
echo "  3. Mode-switch + preflight + readiness spec (DRAFT REQUIRED)"
echo ""

# ============================================================================
# Create condensed context for Qwen3-8B (must stay under 4096 tokens ~3000 words)
# ============================================================================
cat > "$OUTPUT_DIR/condensed_context.txt" <<'CONDENSED'
=== PLATFORM CONTEXT (CONDENSED) ===

SERVICES: llm-gateway:8080, semantic-search:8081, ai-agents:8082, code-orchestrator:8083, inference-service:8085
DATABASES: Qdrant:6333, Neo4j:7687, Redis:6379
MODES: Docker (all containers), Hybrid (DBs in Docker, services native), Native (all native)

CURRENT PROBLEMS:
- Service discovery chaos (container names vs localhost)
- Env var proliferation (NEO4J_URI vs AI_AGENTS_NEO4J_URI)
- Silent fallbacks when dependencies down
- Port conflicts when switching modes

ROUND 2 CONSENSUS POINTS:
- Need single canonical config source
- Fail-fast on config ambiguity
- Shared config library required
- Readiness gates before health aggregator

BLOCKING ITEMS (must decide):
1. MODE DETECTION: Auto-detect vs explicit mode declaration
2. CONFIG SOURCE: Generated artifact vs runtime config service
3. SPEC NEEDED: Mode-switch + preflight + readiness specification
CONDENSED

# ============================================================================
# DECISION 1: Explicit Mode vs Auto-Detect
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "DECISION 1: MODE DETECTION STRATEGY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

DECISION1_PROMPT='DECISION REQUIRED: Mode Detection Strategy

OPTION A - EXPLICIT MODE:
- Developer must declare mode: `platform up --mode=hybrid`
- Mode stored in `.platform-mode` file
- No auto-detection of running containers
- Deterministic, reproducible behavior
- Requires manual mode switch command

OPTION B - AUTO-DETECT MODE:
- System detects mode from running containers/processes
- Convenient, no manual declaration needed
- Risk: non-deterministic if partial stack running
- Risk: stale processes cause wrong mode detection

OPTION C - EXPLICIT WITH DIAGNOSTIC:
- Explicit mode required for startup
- Auto-detect only for diagnostics: `platform doctor` shows detected state
- Best of both: deterministic startup + helpful debugging

Vote: A, B, or C with brief justification (2-3 sentences max).
Then provide the implementation spec for your chosen option.'

echo "━━━ Qwen3-8B Vote ━━━"
QWEN_D1_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION1_PROMPT" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are an infrastructure architect. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_D1_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d1_qwen.txt" 2>&1

echo "Qwen3-8B: $(head -3 "$OUTPUT_DIR/resolution_d1_qwen.txt")"
echo ""

echo "━━━ DeepSeek Vote ━━━"
DEEPSEEK_D1_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION1_PROMPT" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a senior architect. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_D1_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d1_deepseek.txt" 2>&1

echo "DeepSeek: $(head -3 "$OUTPUT_DIR/resolution_d1_deepseek.txt")"
echo ""

echo "━━━ GPT-5.2 Vote ━━━"
GPT_D1_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION1_PROMPT" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {"role": "system", "content": "You are a critical systems engineer. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_D1_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d1_gpt52.txt" 2>&1

echo "GPT-5.2: $(head -3 "$OUTPUT_DIR/resolution_d1_gpt52.txt")"
echo ""

# ============================================================================
# DECISION 2: Generated Config Artifact vs Runtime Service
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "DECISION 2: CONFIG SOURCE STRATEGY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

DECISION2_PROMPT='DECISION REQUIRED: Configuration Source Strategy

OPTION A - GENERATED ARTIFACT:
- Build script creates `endpoints.generated.json` from topology.yaml + mode
- File committed/generated at startup, consumed by all services
- No runtime dependency on config service
- Simple, no SPOF, but requires regeneration on topology changes

OPTION B - RUNTIME CONFIG SERVICE:
- Lightweight config service runs in all modes
- Services query at startup for resolved endpoints
- Dynamic updates possible without restart
- Adds complexity and potential SPOF

OPTION C - HYBRID (Generated + Optional Service):
- Generated artifact as primary source
- Optional config service for advanced scenarios (dynamic discovery)
- Services default to generated file, fallback to service if available

Vote: A, B, or C with brief justification (2-3 sentences max).
Then provide the implementation spec for your chosen option.'

echo "━━━ Qwen3-8B Vote ━━━"
QWEN_D2_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION2_PROMPT" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are an infrastructure architect. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_D2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d2_qwen.txt" 2>&1

echo "Qwen3-8B: $(head -3 "$OUTPUT_DIR/resolution_d2_qwen.txt")"
echo ""

echo "━━━ DeepSeek Vote ━━━"
DEEPSEEK_D2_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION2_PROMPT" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a senior architect. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_D2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d2_deepseek.txt" 2>&1

echo "DeepSeek: $(head -3 "$OUTPUT_DIR/resolution_d2_deepseek.txt")"
echo ""

echo "━━━ GPT-5.2 Vote ━━━"
GPT_D2_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION2_PROMPT" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {"role": "system", "content": "You are a critical systems engineer. Make a clear decision with brief justification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 800,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_D2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d2_gpt52.txt" 2>&1

echo "GPT-5.2: $(head -3 "$OUTPUT_DIR/resolution_d2_gpt52.txt")"
echo ""

# ============================================================================
# DECISION 3: Draft Mode-Switch + Preflight + Readiness Spec
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "DECISION 3: DRAFT SPECIFICATION"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

DECISION3_PROMPT='TASK: Draft the Mode-Switch + Preflight + Readiness Specification

Provide a concise spec covering:

1. MODE-SWITCH PROTOCOL:
   - Command syntax for switching modes
   - What gets torn down (containers, processes, ports)
   - What gets cleaned up (volumes, temp files)
   - Verification steps after switch

2. PREFLIGHT CHECKS:
   - Port availability checks (which ports, how to remediate)
   - Docker daemon check
   - Required env vars check
   - Volume/data directory checks

3. READINESS CONTRACT:
   - Required /ready endpoint format
   - What constitutes "ready" for each service type
   - Dependency readiness checks
   - Schema/migration version checks

Format as a structured specification document (YAML or markdown).
Keep it actionable - this will be implemented this week.'

echo "━━━ Qwen3-8B Spec Draft ━━━"
QWEN_D3_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION3_PROMPT" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are an infrastructure architect. Provide a clear, implementable specification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 1200,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_D3_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d3_qwen.txt" 2>&1

echo "Qwen3-8B spec saved."
echo ""

echo "━━━ DeepSeek Spec Draft ━━━"
DEEPSEEK_D3_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION3_PROMPT" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a senior architect. Provide a clear, implementable specification."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_D3_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d3_deepseek.txt" 2>&1

echo "DeepSeek spec saved."
echo ""

echo "━━━ GPT-5.2 Spec Review ━━━"
# GPT-5.2 reviews the specs instead of drafting
GPT_D3_PAYLOAD=$(jq -n \
  --rawfile context "$OUTPUT_DIR/condensed_context.txt" \
  --arg prompt "$DECISION3_PROMPT

After drafting, identify:
- Any gaps in the spec
- Potential edge cases not covered
- What would make you sign off on implementation" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {"role": "system", "content": "You are a critical systems engineer. Provide spec and identify gaps."},
      {"role": "user", "content": ($context + "\n\n" + $prompt)}
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_D3_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/resolution_d3_gpt52.txt" 2>&1

echo "GPT-5.2 review saved."
echo ""

# ============================================================================
# FINAL TALLY AND CONSENSUS
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "RESOLUTION ROUND COMPLETE - TALLYING VOTES"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

echo "━━━ DECISION 1: MODE DETECTION ━━━"
echo ""
echo "QWEN3-8B:"
cat "$OUTPUT_DIR/resolution_d1_qwen.txt"
echo ""
echo "---"
echo "DEEPSEEK:"
cat "$OUTPUT_DIR/resolution_d1_deepseek.txt"
echo ""
echo "---"
echo "GPT-5.2:"
cat "$OUTPUT_DIR/resolution_d1_gpt52.txt"
echo ""

echo "━━━ DECISION 2: CONFIG SOURCE ━━━"
echo ""
echo "QWEN3-8B:"
cat "$OUTPUT_DIR/resolution_d2_qwen.txt"
echo ""
echo "---"
echo "DEEPSEEK:"
cat "$OUTPUT_DIR/resolution_d2_deepseek.txt"
echo ""
echo "---"
echo "GPT-5.2:"
cat "$OUTPUT_DIR/resolution_d2_gpt52.txt"
echo ""

echo "━━━ DECISION 3: SPEC DRAFTS ━━━"
echo ""
echo "QWEN3-8B SPEC:"
cat "$OUTPUT_DIR/resolution_d3_qwen.txt"
echo ""
echo "---"
echo "DEEPSEEK SPEC:"
cat "$OUTPUT_DIR/resolution_d3_deepseek.txt"
echo ""
echo "---"
echo "GPT-5.2 REVIEW:"
cat "$OUTPUT_DIR/resolution_d3_gpt52.txt"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "Resolution Round Complete!"
echo "All outputs saved to: $OUTPUT_DIR/resolution_*.txt"
echo "═══════════════════════════════════════════════════════════════════════"
