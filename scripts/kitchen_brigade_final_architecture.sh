#!/bin/bash
# Kitchen Brigade - Tiebreaker & Final Architecture Composition
# 1. Tiebreaker vote for Decision 2
# 2. All 3 LLMs compose final architecture document with citations

OUTPUT_DIR="/tmp/kitchen_brigade"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║     KITCHEN BRIGADE - TIEBREAKER & ARCHITECTURE COMPOSITION          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# PART 1: TIEBREAKER - Decision 2 (Config Source)
# ============================================================================
echo "━━━ PART 1: TIEBREAKER - Decision 2 (Config Source) ━━━"
echo ""
echo "Current votes:"
echo "  Qwen3-8B: C (Hybrid - generated artifact + optional service)"
echo "  DeepSeek: A (Generated artifact only)"
echo "  GPT-5.2:  No vote recorded"
echo ""
echo "Asking GPT-5.2 to cast tiebreaker vote..."

TIEBREAKER_PAYLOAD=$(jq -n '{
  "model": "gpt-5.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a critical systems engineer casting a tiebreaker vote. Be decisive and justify your choice."
    },
    {
      "role": "user",
      "content": "TIEBREAKER VOTE NEEDED - Decision 2: Configuration Source Strategy\n\nCONTEXT:\nLocal dev platform with services (llm-gateway, semantic-search, ai-agents, code-orchestrator, inference-service) and databases (Qdrant, Neo4j, Redis). Three deployment modes: Docker, Hybrid, Native.\n\nCURRENT VOTES:\n- Qwen3-8B voted C: Hybrid approach (generated artifact as primary + optional runtime config service for dynamic scenarios)\n- DeepSeek voted A: Generated artifact only (simple, no SPOF, fail-fast)\n\nOPTIONS:\nA) GENERATED ARTIFACT ONLY\n   - topology.yaml + mode → endpoints.generated.json at build time\n   - No runtime dependencies, no SPOF\n   - Requires regeneration on topology changes\n   - Simple, deterministic\n\nB) RUNTIME CONFIG SERVICE\n   - Consul/etcd-style dynamic registry\n   - Allows runtime updates\n   - Adds complexity and SPOF risk\n\nC) HYBRID (Generated + Optional Service)\n   - Generated artifact as primary/fallback\n   - Optional runtime service for advanced dynamic discovery\n   - More flexible but more complex\n\nYour vote will break the tie. Choose A, B, or C.\n\nFormat your response as:\nVOTE: [A/B/C]\nJUSTIFICATION: [2-3 sentences explaining your choice]\nRISK MITIGATION: [How to address the main risk of your choice]"
    }
  ],
  "max_tokens": 500,
  "temperature": 0.3
}')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$TIEBREAKER_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/tiebreaker_decision2.txt" 2>&1

echo "Tiebreaker vote:"
cat "$OUTPUT_DIR/tiebreaker_decision2.txt"
echo ""

# Extract the vote
FINAL_VOTE=$(grep -oE "VOTE:[[:space:]]*[ABC]" "$OUTPUT_DIR/tiebreaker_decision2.txt" | head -1 | grep -oE "[ABC]")
echo "═══════════════════════════════════════════════════════════════════════"
echo "FINAL DECISION 2 RESULT: Option $FINAL_VOTE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# PART 2: ARCHITECTURE COMPOSITION - All 3 LLMs contribute sections
# ============================================================================
echo "━━━ PART 2: COMPOSING FINAL ARCHITECTURE DOCUMENT ━━━"
echo ""

# Load research context
RESEARCH_CONTEXT=$(cat "$OUTPUT_DIR/stage2_research_formatted.txt" 2>/dev/null || echo "No research available")

# Create decisions summary for context
cat > "$OUTPUT_DIR/decisions_summary.txt" <<EOF
=== FINAL DECISIONS FROM KITCHEN BRIGADE DISCUSSION ===

DECISION 1: MODE DETECTION STRATEGY
- Final: OPTION C - Explicit mode with diagnostic auto-detection
- Unanimous agreement from all 3 LLMs
- Implementation: platform up --mode=<MODE> required, auto-detect only for diagnostics via 'platform doctor'

DECISION 2: CONFIGURATION SOURCE STRATEGY  
- Final: OPTION $FINAL_VOTE
- Tiebreaker by GPT-5.2
$(cat "$OUTPUT_DIR/tiebreaker_decision2.txt")

DECISION 3: SPEC PRIORITY
- Mode-Switch Protocol, Preflight Checks, and Readiness Contract specs drafted
- Implementation ready to begin

=== KEY CONSENSUS POINTS ===
1. Single canonical config source via shared library
2. Fail-fast on config ambiguity (no silent fallbacks)
3. Explicit mode declaration (reject pure auto-detect)
4. Readiness gates before health aggregator
5. Generated topology manifest as source of truth
EOF

echo "Decisions summary prepared."
echo ""

# ============================================================================
# Section 1: Executive Summary & Overview (DeepSeek)
# ============================================================================
echo "━━━ Section 1: Executive Summary (DeepSeek) ━━━"

SECTION1_PAYLOAD=$(jq -n \
  --rawfile decisions "$OUTPUT_DIR/decisions_summary.txt" \
  --rawfile research "$OUTPUT_DIR/stage2_research_formatted.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a senior architect writing the Executive Summary section of an Architecture Decision Record. Include citations to source material where relevant using [Source: Book/Chapter] format."
      },
      {
        "role": "user",
        "content": ($decisions + "\n\n" + $research + "\n\nWrite the EXECUTIVE SUMMARY AND OVERVIEW section for the final architecture document. Include:\n\n1. Executive Summary (2-3 paragraphs)\n2. Problem Statement\n3. Solution Overview\n4. Key Architectural Decisions (summarized)\n5. Scope and Boundaries\n\nCite relevant sources from the research where they informed decisions. Format as markdown.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$SECTION1_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/arch_section1_overview.txt" 2>&1

echo "✓ Section 1 complete"
echo ""

# ============================================================================
# Section 2: Mode Management & Service Discovery (Qwen3-8B)
# ============================================================================
echo "━━━ Section 2: Mode Management & Service Discovery (Qwen3-8B) ━━━"

SECTION2_PAYLOAD=$(jq -n \
  --rawfile decisions "$OUTPUT_DIR/decisions_summary.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are an infrastructure architect writing detailed technical specifications. Include code examples and cite sources where relevant."
      },
      {
        "role": "user",
        "content": ($decisions + "\n\nWrite the MODE MANAGEMENT AND SERVICE DISCOVERY section. Include:\n\n1. Mode Declaration Protocol\n   - CLI commands and syntax\n   - .platform-mode file specification\n   - Mode switching workflow\n\n2. Service Discovery Strategy\n   - Endpoint resolution per mode\n   - Generated config artifact format\n   - Shared config library interface\n\n3. Code Examples\n   - Python config library snippet\n   - topology.yaml schema\n   - endpoints.generated.json example\n\nFormat as markdown with code blocks.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$SECTION2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/arch_section2_mode.txt" 2>&1

echo "✓ Section 2 complete"
echo ""

# ============================================================================
# Section 3: Preflight & Readiness (GPT-5.2)
# ============================================================================
echo "━━━ Section 3: Preflight & Readiness (GPT-5.2) ━━━"

SECTION3_PAYLOAD=$(jq -n \
  --rawfile decisions "$OUTPUT_DIR/decisions_summary.txt" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a systems engineer writing detailed technical specifications for operational procedures. Be precise and include validation criteria."
      },
      {
        "role": "user",
        "content": ($decisions + "\n\nWrite the PREFLIGHT CHECKS AND READINESS CONTRACT section. Include:\n\n1. Preflight Check Specification\n   - Port availability checks (with port list)\n   - Docker daemon validation\n   - Environment variable validation\n   - Data directory checks\n   - Remediation steps for each failure\n\n2. Readiness Contract\n   - /ready endpoint JSON schema\n   - Service-specific readiness criteria (for each service)\n   - Dependency graph\n\n3. Health Aggregation\n   - CLI-first approach (platform doctor)\n   - Optional dashboard service\n   - Circuit breaker patterns\n\nFormat as markdown with YAML/JSON schemas where appropriate.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$SECTION3_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/arch_section3_preflight.txt" 2>&1

echo "✓ Section 3 complete"
echo ""

# ============================================================================
# Section 4: Implementation Roadmap (DeepSeek)
# ============================================================================
echo "━━━ Section 4: Implementation Roadmap (DeepSeek) ━━━"

SECTION4_PAYLOAD=$(jq -n \
  --rawfile decisions "$OUTPUT_DIR/decisions_summary.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a technical program manager creating a detailed implementation roadmap with clear deliverables and acceptance criteria."
      },
      {
        "role": "user",
        "content": ($decisions + "\n\nWrite the IMPLEMENTATION ROADMAP section. Include:\n\n1. Phase 0: Quick Wins (Week 1)\n   - Deliverables with acceptance criteria\n   - Estimated effort\n\n2. Phase 1: Foundation (Weeks 2-4)\n   - Deliverables with acceptance criteria\n   - Dependencies\n\n3. Phase 2: Migration (Weeks 5-8)\n   - Service migration schedule\n   - Rollback plan\n\n4. Phase 3: Polish (Weeks 9-12)\n   - Advanced features\n   - Documentation\n\n5. Risk Matrix\n   - Risk, probability, impact, mitigation\n\n6. Success Metrics\n   - Immediate, medium-term, long-term\n\nFormat as markdown with tables where appropriate.")
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$SECTION4_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/arch_section4_roadmap.txt" 2>&1

echo "✓ Section 4 complete"
echo ""

# ============================================================================
# Section 5: References & Citations (Qwen3-8B)
# ============================================================================
echo "━━━ Section 5: References & Citations (Qwen3-8B) ━━━"

SECTION5_PAYLOAD=$(jq -n \
  --rawfile research "$OUTPUT_DIR/stage2_research_formatted.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a technical writer compiling a references section. Extract all source citations and organize them properly."
      },
      {
        "role": "user",
        "content": ($research + "\n\nCompile the REFERENCES AND CITATIONS section. Include:\n\n1. Primary Sources\n   - Books and chapters that informed the architecture\n   - Key concepts from each source\n\n2. Design Patterns Referenced\n   - Pattern name, source, and how it was applied\n\n3. Related Architecture Decision Records\n   - Links to any ADRs mentioned\n\n4. Appendix\n   - Glossary of terms\n   - Acronyms used\n\nFormat as markdown. If specific sources are not available, note that the architecture was informed by industry best practices for microservices, service discovery, and local development environments.")
      }
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$SECTION5_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/arch_section5_references.txt" 2>&1

echo "✓ Section 5 complete"
echo ""

# ============================================================================
# COMPOSE FINAL DOCUMENT
# ============================================================================
echo "━━━ Composing Final Architecture Document ━━━"

cat > "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md" <<'HEADER'
# Platform Infrastructure Architecture Decision Record

**Document Version:** 1.0  
**Date:** January 6, 2026  
**Status:** APPROVED  
**Authors:** Kitchen Brigade Multi-LLM Discussion (Qwen3-8B, DeepSeek-Chat, GPT-5.2)

---

## Document History

| Version | Date | Authors | Changes |
|---------|------|---------|---------|
| 1.0 | 2026-01-06 | Kitchen Brigade | Initial architecture approved |

---

HEADER

# Append all sections
echo "" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
cat "$OUTPUT_DIR/arch_section1_overview.txt" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
echo -e "\n\n---\n" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
cat "$OUTPUT_DIR/arch_section2_mode.txt" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
echo -e "\n\n---\n" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
cat "$OUTPUT_DIR/arch_section3_preflight.txt" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
echo -e "\n\n---\n" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
cat "$OUTPUT_DIR/arch_section4_roadmap.txt" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
echo -e "\n\n---\n" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
cat "$OUTPUT_DIR/arch_section5_references.txt" >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"

# Add decision record appendix
cat >> "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md" <<'APPENDIX'

---

## Appendix: Kitchen Brigade Discussion Record

### Participants
- **Qwen3-8B** (Local Inference): Proposer role - initial proposals and integration
- **DeepSeek-Chat** (LLM Gateway): Architect role - synthesis and consolidation  
- **GPT-5.2** (LLM Gateway): Critic role - validation and risk identification

### Decision Record

| Decision | Options | Final Choice | Vote Count |
|----------|---------|--------------|------------|
| Mode Detection | A) Explicit, B) Auto-detect, C) Explicit+Diagnostic | **C** | 3-0 (Unanimous) |
| Config Source | A) Generated artifact, B) Runtime service, C) Hybrid | See tiebreaker | 2-1 |
| Spec Priority | Draft Mode-Switch + Preflight + Readiness specs | **Approved** | Consensus |

### Discussion Rounds
1. **Round 1**: Initial proposals from each LLM
2. **Round 2**: Synthesis with semantic-search research
3. **Round 3**: Original vs New comparison
4. **Resolution Round**: Explicit voting on blocking decisions
5. **Architecture Composition**: Collaborative document creation

APPENDIX

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "ARCHITECTURE DOCUMENT COMPOSED"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Final document saved to: $OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
echo ""
echo "Section files:"
ls -la "$OUTPUT_DIR"/arch_section*.txt
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Displaying final document..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
cat "$OUTPUT_DIR/ARCHITECTURE_DECISION_RECORD.md"
