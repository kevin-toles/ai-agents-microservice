#!/bin/bash
# Gemini 2.5 Pro Assessment of Platform Documentation
# Sends both documents for review with development timeline context

OUTPUT_DIR="/tmp/kitchen_brigade"
mkdir -p "$OUTPUT_DIR"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║     GEMINI 2.5 PRO - PLATFORM ASSESSMENT                              ║"
echo "║     Reviewing AI Coding Platform documentation                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set"
    echo "Run: export GEMINI_API_KEY='your-key-here'"
    exit 1
fi

# Read both documents
DOC1=$(cat "/Users/kevintoles/POC/Platform Documentation/GOOGLE_INTERVIEW_OVERVIEW.md")
DOC2=$(cat "/Users/kevintoles/POC/Platform Documentation/ARCHITECTURE_DECISION_RECORD.md")

# Create the prompt
PROMPT="You are a senior Google engineer reviewing documentation for an AI Coding Platform built by a candidate. This platform was developed from December 1, 2025 to January 5, 2026 (~5 weeks).

Please provide a thorough assessment covering:

1. **Architecture Quality**: Evaluate the Kitchen Brigade design, service decomposition, and overall system architecture
2. **ADK & MCP Integration**: Assess how well Google's Agent Development Kit patterns and Model Context Protocol are implemented
3. **Technical Decisions**: Review the architecture decisions (explicit mode detection, generated config, readiness gates)
4. **Development Velocity**: Comment on the 5-week timeline and what was accomplished
5. **Hybrid Infrastructure**: Evaluate the local LLM on Metal + Docker DBs approach
6. **Areas for Improvement**: What would you suggest for Phase 3 and beyond?
7. **Interview Readiness**: Overall impression - is this candidate ready for a Google technical interview?

Be specific, technical, and honest. This assessment will help the candidate prepare.

--- DOCUMENT 1: GOOGLE_INTERVIEW_OVERVIEW.md ---

$DOC1

--- DOCUMENT 2: ARCHITECTURE_DECISION_RECORD.md ---

$DOC2"

echo "━━━ Sending to Gemini 2.5 Pro for assessment... ━━━"
echo ""

# Create JSON payload using jq
PAYLOAD=$(jq -n \
  --arg prompt "$PROMPT" \
  '{
    "contents": [{
      "parts": [{
        "text": $prompt
      }]
    }],
    "generationConfig": {
      "maxOutputTokens": 8000,
      "temperature": 0.7
    }
  }')

# Call Gemini API
RESPONSE=$(curl -s -X POST \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-05-06:generateContent?key=$GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

# Extract the text response
ASSESSMENT=$(echo "$RESPONSE" | jq -r '.candidates[0].content.parts[0].text // .error.message // .')

# Save and display
echo "$ASSESSMENT" > "$OUTPUT_DIR/GEMINI_ASSESSMENT.md"

echo "═══════════════════════════════════════════════════════════════════════"
echo "GEMINI 2.5 PRO ASSESSMENT"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "$ASSESSMENT"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Assessment saved to: $OUTPUT_DIR/GEMINI_ASSESSMENT.md"
echo "═══════════════════════════════════════════════════════════════════════"
