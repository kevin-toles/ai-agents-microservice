#!/bin/bash
# Kitchen Brigade Stage 2 - Research Loop with Comparison
# 1. Fresh run with semantic-search research
# 2. Comparison loop: LLMs review original vs new outputs

OUTPUT_DIR="/tmp/kitchen_brigade"
ORIGINAL_DIR="/tmp/kitchen_brigade_original"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║     KITCHEN BRIGADE STAGE 2 - ENHANCED RESEARCH LOOP                 ║"
echo "║     With Original vs New Comparison                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Verify original backup exists
if [ ! -d "$ORIGINAL_DIR" ]; then
    echo "ERROR: Original outputs not found at $ORIGINAL_DIR"
    echo "Please backup original outputs first: cp -r /tmp/kitchen_brigade /tmp/kitchen_brigade_original"
    exit 1
fi

# ============================================================================
# STEP 1: Extract RESEARCH_NEEDED queries from Round 1
# ============================================================================
echo "━━━ STEP 1: Extracting RESEARCH_NEEDED queries ━━━"

RESEARCH_QUERIES=""
for file in "$OUTPUT_DIR"/round1_*.txt; do
    if grep -q "RESEARCH_NEEDED" "$file"; then
        echo "Found request in: $(basename $file)"
        QUERY=$(grep -oE "RESEARCH_NEEDED:.*" "$file" | sed 's/RESEARCH_NEEDED:[[:space:]]*//')
        echo "  Query: $QUERY"
        RESEARCH_QUERIES="$RESEARCH_QUERIES
$QUERY"
    fi
done

if [ -z "$RESEARCH_QUERIES" ]; then
    echo "No RESEARCH_NEEDED queries found. Stage 2 not required."
    exit 0
fi

echo ""

# ============================================================================
# STEP 2: Call Semantic Search Service for additional research
# ============================================================================
echo "━━━ STEP 2: Fetching research from Semantic Search ━━━"

# Check if semantic-search is available
if ! curl -s http://localhost:8081/health > /dev/null 2>&1; then
    echo "ERROR: Semantic-search service not available on port 8081"
    exit 1
fi

# Build search payload for hybrid search
SEARCH_PAYLOAD=$(jq -n \
  --arg query "$RESEARCH_QUERIES" \
  '{
    "query": $query,
    "limit": 15,
    "collection": "chapters",
    "include_graph": true,
    "tier_boost": true
  }')

echo "Calling semantic-search service (hybrid search)..."
RESEARCH_RESULTS=$(curl -s -X POST http://localhost:8081/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d "$SEARCH_PAYLOAD")

# Check if we got results
if echo "$RESEARCH_RESULTS" | jq -e '.results' > /dev/null 2>&1; then
    RESULT_COUNT=$(echo "$RESEARCH_RESULTS" | jq '.results | length')
    echo "✓ Received $RESULT_COUNT research results!"
    echo ""
    echo "Top results:"
    echo "$RESEARCH_RESULTS" | jq -r '.results[:10][] | "• [\(.score | tostring | .[0:5])] \(.metadata.book // .metadata.source // "unknown"): \(.metadata.chapter // .metadata.title // "N/A")"' 2>/dev/null
else
    echo "Warning: Unexpected response format"
    echo "$RESEARCH_RESULTS" | head -5
fi

# Save research results
echo "$RESEARCH_RESULTS" > "$OUTPUT_DIR/stage2_research.json"
echo ""

# Format research for LLM context
RESEARCH_CONTEXT=$(echo "$RESEARCH_RESULTS" | jq -r '
  "=== STAGE 2 ADDITIONAL RESEARCH ===\n\nResearch query: " + 
  "Service discovery patterns for hybrid local dev environments\n\n" +
  "Results from textbook corpus:\n\n" +
  (.results // [] | .[:10] | map(
    "---\nSource: " + (.metadata.book // .metadata.source // "unknown") + "\n" +
    "Chapter: " + (.metadata.chapter // .metadata.title // "N/A") + "\n" +
    "Score: " + (.score | tostring) + "\n" +
    "Content:\n" + (.content // .text // "No content" | .[0:800]) + "\n"
  ) | join("\n"))
' 2>/dev/null)

if [ -z "$RESEARCH_CONTEXT" ] || [ "$RESEARCH_CONTEXT" = "null" ]; then
    RESEARCH_CONTEXT="=== STAGE 2 ADDITIONAL RESEARCH ===

Research was requested but results were limited. Proceeding with synthesis based on Round 1 discussion."
fi

echo "$RESEARCH_CONTEXT" > "$OUTPUT_DIR/stage2_research_formatted.txt"
echo "Research context saved."
echo ""

# ============================================================================
# STEP 3: Load Round 1 responses for context
# ============================================================================
echo "━━━ STEP 3: Loading Round 1 context ━━━"

ROUND1_QWEN=$(cat "$OUTPUT_DIR/round1_qwen.txt" 2>/dev/null || echo "No Qwen response")
ROUND1_DEEPSEEK=$(cat "$OUTPUT_DIR/round1_deepseek.txt" 2>/dev/null || echo "No DeepSeek response")
ROUND1_GPT=$(cat "$OUTPUT_DIR/round1_gpt52.txt" 2>/dev/null || echo "No GPT response")

cat > "$OUTPUT_DIR/round1_summary.txt" <<EOF
=== ROUND 1 DISCUSSION SUMMARY ===

--- PROPOSER (Qwen3-8B) ---
$ROUND1_QWEN

--- ARCHITECT (DeepSeek) ---
$ROUND1_DEEPSEEK

--- CRITIC (GPT-5.2) ---
$ROUND1_GPT
EOF

echo "Round 1 context loaded."
echo ""

# ============================================================================
# ROUND 2: Synthesis with new research (FRESH RUN)
# ============================================================================
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  ROUND 2: Fresh Synthesis with Research                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ROUND 2.1: Qwen3-8B
echo "━━━ ROUND 2.1: Qwen3-8B (Integration) ━━━"
echo "Calling local inference service..."

QWEN_R2_PAYLOAD=$(jq -n \
  --rawfile round1 "$OUTPUT_DIR/round1_summary.txt" \
  --rawfile research "$OUTPUT_DIR/stage2_research_formatted.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are an infrastructure architect. Integrate the new research findings into your recommendations for the local development platform."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nBased on the Round 1 discussion and new research, provide UPDATED RECOMMENDATIONS:\n1. How does the research inform the hybrid mode discovery approach?\n2. What specific patterns from the research should we adopt?\n3. Provide concrete code examples incorporating these patterns.\n4. What is your recommended priority order for implementation?")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_qwen.txt" 2>&1

echo "✓ Qwen3-8B Round 2 response saved."
echo ""

# ROUND 2.2: DeepSeek
echo "━━━ ROUND 2.2: DeepSeek-Chat (Synthesis) ━━━"
echo "Calling via LLM Gateway..."

DEEPSEEK_R2_PAYLOAD=$(jq -n \
  --rawfile round1 "$OUTPUT_DIR/round1_summary.txt" \
  --rawfile research "$OUTPUT_DIR/stage2_research_formatted.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a senior infrastructure architect. Synthesize all inputs into a final architectural recommendation with a concrete implementation plan."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nProduce FINAL ARCHITECTURAL RECOMMENDATION:\n1. Service discovery approach (final decision)\n2. Concrete implementation plan with phases\n3. Migration path for env var standardization\n4. Health aggregator design (addressing SPOF concerns)\n5. Contract testing strategy\n6. Estimated effort per phase\n\nProvide actionable, specific recommendations.")
      }
    ],
    "max_tokens": 2500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_deepseek.txt" 2>&1

echo "✓ DeepSeek Round 2 response saved."
echo ""

# ROUND 2.3: GPT-5.2
echo "━━━ ROUND 2.3: GPT-5.2 (Final Review) ━━━"
echo "Calling via LLM Gateway..."

GPT_R2_PAYLOAD=$(jq -n \
  --rawfile round1 "$OUTPUT_DIR/round1_summary.txt" \
  --rawfile research "$OUTPUT_DIR/stage2_research_formatted.txt" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a critical systems engineer. Provide final review identifying remaining gaps, risks, and quick wins."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nFINAL CRITICAL REVIEW:\n1. Remaining gaps in the proposal?\n2. Top 3 risks with recommended approach?\n3. Quick wins for immediate implementation?\n4. What needs more investigation?\n5. Overall assessment: Is this plan ready for implementation?\n\nBe concise and actionable.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_gpt52.txt" 2>&1

echo "✓ GPT-5.2 Round 2 response saved."
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "ROUND 2 COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# ROUND 3: COMPARISON LOOP - Original vs New
# ============================================================================
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  ROUND 3: COMPARISON - Original vs New Outputs                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Load original Round 2 outputs
ORIG_QWEN=$(cat "$ORIGINAL_DIR/round2_qwen.txt" 2>/dev/null || echo "No original Qwen response")
ORIG_DEEPSEEK=$(cat "$ORIGINAL_DIR/round2_deepseek.txt" 2>/dev/null || echo "No original DeepSeek response")
ORIG_GPT=$(cat "$ORIGINAL_DIR/round2_gpt52.txt" 2>/dev/null || echo "No original GPT response")

# Load new Round 2 outputs
NEW_QWEN=$(cat "$OUTPUT_DIR/round2_qwen.txt" 2>/dev/null || echo "No new Qwen response")
NEW_DEEPSEEK=$(cat "$OUTPUT_DIR/round2_deepseek.txt" 2>/dev/null || echo "No new DeepSeek response")
NEW_GPT=$(cat "$OUTPUT_DIR/round2_gpt52.txt" 2>/dev/null || echo "No new GPT response")

# Create comparison context file
cat > "$OUTPUT_DIR/comparison_context.txt" <<EOF
=== COMPARISON: ORIGINAL VS NEW OUTPUTS ===

The original run was done WITHOUT semantic-search research (the service was not available).
The new run was done WITH semantic-search research results.

=== ORIGINAL OUTPUTS (without research) ===

--- ORIGINAL Qwen3-8B ---
$ORIG_QWEN

--- ORIGINAL DeepSeek ---
$ORIG_DEEPSEEK

--- ORIGINAL GPT-5.2 ---
$ORIG_GPT

=== NEW OUTPUTS (with research) ===

--- NEW Qwen3-8B ---
$NEW_QWEN

--- NEW DeepSeek ---
$NEW_DEEPSEEK

--- NEW GPT-5.2 ---
$NEW_GPT
EOF

echo "━━━ ROUND 3.1: Qwen3-8B (Comparison Analysis) ━━━"
echo "Calling local inference service..."

QWEN_COMPARE_PAYLOAD=$(jq -n \
  --rawfile comparison "$OUTPUT_DIR/comparison_context.txt" \
  '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "system",
        "content": "You are an infrastructure architect reviewing two versions of recommendations. Compare the original (without research) to the new (with research) outputs."
      },
      {
        "role": "user",
        "content": ($comparison + "\n\nANALYZE THE DIFFERENCES:\n1. What key insights did the research add that were missing originally?\n2. Did any recommendations change significantly? Why?\n3. Which version is more actionable?\n4. What consensus emerged across both versions?\n5. Final recommendation: What should we implement?")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_COMPARE_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round3_comparison_qwen.txt" 2>&1

echo "✓ Qwen3-8B comparison saved."
echo ""

echo "━━━ ROUND 3.2: DeepSeek-Chat (Synthesis of Comparison) ━━━"
echo "Calling via LLM Gateway..."

DEEPSEEK_COMPARE_PAYLOAD=$(jq -n \
  --rawfile comparison "$OUTPUT_DIR/comparison_context.txt" \
  '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a senior architect synthesizing two rounds of multi-LLM discussion. Create a final consolidated recommendation."
      },
      {
        "role": "user",
        "content": ($comparison + "\n\nCREATE FINAL CONSOLIDATED RECOMMENDATION:\n1. Merge the best insights from both versions\n2. Resolve any conflicts between original and new recommendations\n3. Produce a SINGLE prioritized implementation plan\n4. Include specific acceptance criteria for each phase\n5. Identify what can be done this week vs next month\n\nThis is the FINAL output that will guide implementation.")
      }
    ],
    "max_tokens": 2500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_COMPARE_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round3_comparison_deepseek.txt" 2>&1

echo "✓ DeepSeek comparison saved."
echo ""

echo "━━━ ROUND 3.3: GPT-5.2 (Final Validation) ━━━"
echo "Calling via LLM Gateway..."

GPT_COMPARE_PAYLOAD=$(jq -n \
  --rawfile comparison "$OUTPUT_DIR/comparison_context.txt" \
  '{
    "model": "gpt-5.2",
    "messages": [
      {
        "role": "system",
        "content": "You are a critical systems engineer performing final validation of multi-LLM discussion outputs."
      },
      {
        "role": "user",
        "content": ($comparison + "\n\nFINAL VALIDATION:\n1. Did the research improve the quality of recommendations? Score 1-10 with justification.\n2. Are there any remaining blind spots or risks?\n3. Is the team ready to start implementation? YES/NO with conditions.\n4. What is the single most important thing to get right?\n5. Final sign-off or concerns that must be addressed first.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_COMPARE_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round3_comparison_gpt52.txt" 2>&1

echo "✓ GPT-5.2 comparison saved."
echo ""

# ============================================================================
# FINAL OUTPUT
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "ALL ROUNDS COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  Round 1: $OUTPUT_DIR/round1_*.txt"
echo "  Round 2: $OUTPUT_DIR/round2_*.txt"
echo "  Round 3 (Comparison): $OUTPUT_DIR/round3_comparison_*.txt"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "━━━ ROUND 2 RESULTS (Fresh with Research) ━━━"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "━━━ QWEN3-8B (Integration): ━━━"
cat "$OUTPUT_DIR/round2_qwen.txt"
echo ""
echo "━━━ DEEPSEEK (Synthesis): ━━━"
cat "$OUTPUT_DIR/round2_deepseek.txt"
echo ""
echo "━━━ GPT-5.2 (Final Review): ━━━"
cat "$OUTPUT_DIR/round2_gpt52.txt"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "━━━ ROUND 3 COMPARISON RESULTS ━━━"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "━━━ QWEN3-8B (Comparison Analysis): ━━━"
cat "$OUTPUT_DIR/round3_comparison_qwen.txt"
echo ""
echo "━━━ DEEPSEEK (Final Consolidated Plan): ━━━"
cat "$OUTPUT_DIR/round3_comparison_deepseek.txt"
echo ""
echo "━━━ GPT-5.2 (Final Validation): ━━━"
cat "$OUTPUT_DIR/round3_comparison_gpt52.txt"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Kitchen Brigade Enhanced Discussion Complete!"
echo "All outputs saved to: $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════════════"
