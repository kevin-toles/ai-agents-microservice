#!/bin/bash
# Kitchen Brigade Stage 2 - Research Loop
# Reads Round 1 results, extracts RESEARCH_NEEDED queries, fetches research, runs Round 2

OUTPUT_DIR="/tmp/kitchen_brigade"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║          KITCHEN BRIGADE STAGE 2 - RESEARCH LOOP                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 1: Extract RESEARCH_NEEDED queries from Round 1
# ============================================================================
echo "━━━ STEP 1: Extracting RESEARCH_NEEDED queries ━━━"

RESEARCH_QUERIES=""
for file in "$OUTPUT_DIR"/round1_*.txt; do
    if grep -q "RESEARCH_NEEDED" "$file"; then
        echo "Found request in: $(basename $file)"
        # Extract the query after RESEARCH_NEEDED:
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
echo "Combined research queries:"
echo "$RESEARCH_QUERIES"
echo ""

# ============================================================================
# STEP 2: Call Semantic Search Service for additional research
# ============================================================================
echo "━━━ STEP 2: Fetching research from Semantic Search ━━━"

# Build search payload
SEARCH_PAYLOAD=$(jq -n \
  --arg query "$RESEARCH_QUERIES" \
  '{
    "query": $query,
    "top_k": 15,
    "search_type": "hybrid"
  }')

echo "Calling semantic-search service..."
RESEARCH_RESULTS=$(curl -s -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d "$SEARCH_PAYLOAD")

# Check if we got results
if echo "$RESEARCH_RESULTS" | jq -e '.results' > /dev/null 2>&1; then
    echo "Research results received!"
    echo "$RESEARCH_RESULTS" | jq -r '.results[] | "• [\(.score | tostring | .[0:5])] \(.metadata.book // .metadata.source): \(.metadata.chapter // .metadata.title)"' 2>/dev/null | head -20
else
    echo "Warning: Could not parse research results, using raw response"
fi

# Save research results
echo "$RESEARCH_RESULTS" > "$OUTPUT_DIR/stage2_research.json"
echo ""

# Format research for LLM context
RESEARCH_CONTEXT=$(echo "$RESEARCH_RESULTS" | jq -r '
  "=== STAGE 2 ADDITIONAL RESEARCH ===\n\n" +
  (.results // [] | map(
    "Source: \(.metadata.book // .metadata.source // "unknown")\n" +
    "Chapter: \(.metadata.chapter // .metadata.title // "N/A")\n" +
    "Score: \(.score)\n" +
    "Content: \(.content // .text | .[0:500])...\n---"
  ) | join("\n"))
' 2>/dev/null || echo "Research results: $RESEARCH_RESULTS")

# Save formatted research
echo "$RESEARCH_CONTEXT" > "$OUTPUT_DIR/stage2_research_formatted.txt"

# ============================================================================
# STEP 3: Load Round 1 responses for context
# ============================================================================
echo "━━━ STEP 3: Loading Round 1 context ━━━"

ROUND1_QWEN=$(cat "$OUTPUT_DIR/round1_qwen.txt" 2>/dev/null || echo "No Qwen response")
ROUND1_DEEPSEEK=$(cat "$OUTPUT_DIR/round1_deepseek.txt" 2>/dev/null || echo "No DeepSeek response")
ROUND1_GPT=$(cat "$OUTPUT_DIR/round1_gpt52.txt" 2>/dev/null || echo "No GPT response")

# Write combined context to file for jq
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
# STEP 4: ROUND 2 - Synthesis with new research
# ============================================================================
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  ROUND 2: Synthesis with Additional Research                         ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# ROUND 2.1: Qwen3-8B - Integrate research findings
# ============================================================================
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
        "content": "You are an infrastructure architect. You have received new research based on your RESEARCH_NEEDED request. Integrate these findings into the discussion and update your recommendations."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nBased on the new research above, update your recommendations. Specifically address:\n1. How does the research inform the hybrid mode discovery approach?\n2. What patterns from the books should we adopt?\n3. Provide updated code examples incorporating the research findings.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$QWEN_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_qwen.txt" 2>&1

echo "Qwen3-8B Round 2 response saved."
echo ""

# ============================================================================
# ROUND 2.2: DeepSeek - Architectural synthesis
# ============================================================================
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
        "content": "You are a senior infrastructure architect. Synthesize the Round 1 discussion with the new research to produce a final architectural recommendation."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nSynthesize all inputs into a FINAL ARCHITECTURAL RECOMMENDATION. Include:\n1. Final decision on service discovery approach\n2. Concrete implementation plan with priorities\n3. Migration path for env var standardization\n4. Health aggregator design (addressing single-point-of-failure concerns)\n5. Contract testing strategy\n\nProvide a prioritized action plan with estimated effort.")
      }
    ],
    "max_tokens": 2500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$DEEPSEEK_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_deepseek.txt" 2>&1

echo "DeepSeek Round 2 response saved."
echo ""

# ============================================================================
# ROUND 2.3: GPT-5.2 - Final review and gaps
# ============================================================================
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
        "content": "You are a critical systems engineer. Provide final review of the synthesized recommendations, identifying any remaining gaps or risks."
      },
      {
        "role": "user",
        "content": ($round1 + "\n\n" + $research + "\n\nProvide FINAL REVIEW:\n1. Are there any remaining gaps in the proposal?\n2. What are the top 3 risks with the recommended approach?\n3. What quick wins can be implemented immediately?\n4. What requires more investigation before implementation?\n\nBe concise and actionable.")
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7
  }')

curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$GPT_R2_PAYLOAD" \
  | jq -r '.choices[0].message.content // .error // .' > "$OUTPUT_DIR/round2_gpt52.txt" 2>&1

echo "GPT-5.2 Round 2 response saved."
echo ""

# ============================================================================
# FINAL OUTPUT
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "STAGE 2 COMPLETE - All responses saved to $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Display Round 2 results
echo "━━━ ROUND 2 RESULTS ━━━"
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
echo "Kitchen Brigade Discussion Complete!"
echo "All outputs saved to: $OUTPUT_DIR/"
echo "═══════════════════════════════════════════════════════════════════════"
