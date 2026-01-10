#!/usr/bin/env python3
"""
Gemini Platform Review Script

Sends platform documentation to Gemini 2.5 Pro for technical assessment.
Uses the new GeminiProvider from llm-gateway.

Usage:
    export GEMINI_API_KEY='your-key'
    python scripts/gemini_review.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add llm-gateway to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "llm-gateway"))

try:
    from src.providers.gemini import GeminiProvider
    USE_PROVIDER = True
except ImportError:
    # Fallback to direct httpx if provider not available
    USE_PROVIDER = False
    import httpx


DOCS_DIR = Path(__file__).parent.parent.parent / "Platform Documentation"
OUTPUT_DIR = Path("/tmp/gemini_review")


async def load_documents() -> tuple[str, str]:
    """Load the two main platform documents."""
    overview_path = DOCS_DIR / "GOOGLE_INTERVIEW_OVERVIEW.md"
    adr_path = DOCS_DIR / "ARCHITECTURE_DECISION_RECORD.md"
    
    with open(overview_path, "r") as f:
        overview = f.read()
    
    with open(adr_path, "r") as f:
        adr = f.read()
    
    return overview, adr


def build_review_prompt(overview: str, adr: str) -> str:
    """Build the review prompt for Gemini."""
    return f"""You are a principal systems architect conducting a system design review for a **POC/MVP platform**. 

CRITICAL CONTEXT:
- This is a PROOF OF CONCEPT / MVP, NOT a production system
- Target users: 5-10 people at POC stage, 25-50 at MVP stage
- Built in 5 weeks by a single developer
- Purpose: Demonstrate viability of AI-assisted code generation approach
- Infrastructure: Local development environment (not cloud-deployed)

Evaluate this design WITH THIS SCOPE IN MIND. Don't critique missing enterprise features that aren't needed for 25-50 users.

Provide a rigorous but appropriately-scoped assessment:

## 1. POC/MVP ARCHITECTURE ASSESSMENT
- Is this architecture appropriate for the stated scale (5-50 users)?
- What's working well for a POC?
- What complexity is justified vs unnecessary at this stage?
- Service boundaries: Right-sized for POC or over/under-engineered?

## 2. SCALABILITY PATH
- What needs to change to go from POC (5-10) to MVP (25-50)?
- What are the natural scaling breakpoints?
- Which components would need rework vs just horizontal scaling?

## 3. DESIGN DECISIONS EVALUATION
- Explicit mode detection: Appropriate for local dev POC?
- Generated config: Right choice for small team?
- Hybrid local/Docker: Smart or overcomplicating things?
- ADK/MCP/A2A: Over-engineering for POC or good foundation?

## 4. TECHNICAL DEBT vs INTENTIONAL SIMPLIFICATION
- What's acceptable technical debt for a POC?
- What shortcuts are fine vs what will bite them later?
- Security: What's acceptable risk for internal POC vs what's still critical?

## 5. WHAT'S IMPRESSIVE
- What stands out positively for a 5-week single-developer POC?
- Which architectural choices show good engineering judgment?
- What would you keep as-is?

## 6. RECOMMENDED CHANGES
- What 3-5 changes would have the highest impact for MVP readiness?
- What should be deferred until actual production (post-MVP)?
- What can stay as-is for the 25-50 user target?

## 7. VERDICT
- POC Readiness Score (1-10): Is it ready to demo and get feedback?
- MVP Readiness Score (1-10): Ready for 25-50 internal users?
- Top risks for THIS SCALE (not enterprise scale)

Be realistic about scope. A POC doesn't need enterprise HA, disaster recovery, or complex security. Evaluate against actual requirements.

--- DOCUMENT 1: PLATFORM OVERVIEW ---

{overview}

--- DOCUMENT 2: ARCHITECTURE DECISION RECORD ---

{adr}"""


async def review_with_provider(prompt: str) -> str:
    """Use the GeminiProvider to get review."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    provider = GeminiProvider(api_key=api_key)
    
    # Import request model
    from src.models.requests import ChatCompletionRequest, Message
    
    request = ChatCompletionRequest(
        model="gemini-2.5-pro",
        messages=[Message(role="user", content=prompt)],
        max_tokens=8000,
        temperature=0.7,
    )
    
    response = await provider.complete(request)
    return response.choices[0].message.content


async def review_with_httpx(prompt: str) -> str:
    """Direct httpx call - try OpenRouter first, fallback to direct Gemini API."""
    
    # Try OpenRouter first with different models
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        # Try multiple models in order of preference
        models_to_try = [
            "openai/gpt-5.2",  # GPT-5.2 first
            "openai/o1",  # o1 as backup
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
        ]
        
        for model in models_to_try:
            print(f"    Trying {model} via OpenRouter...")
            url = "https://openrouter.ai/api/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8000,
                "temperature": 0.7,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_key}",
                "HTTP-Referer": "https://github.com/kevin-toles/ai-platform",
                "X-Title": "AI Platform Review",
            }
            
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            print(f"    ✓ Success with {model}")
                            return choices[0].get("message", {}).get("content", "")
                    else:
                        print(f"    ✗ {model} returned {response.status_code}")
            except Exception as e:
                print(f"    ✗ {model} error: {e}")
                continue
    
    # Fallback to direct Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Neither OPENROUTER_API_KEY nor GEMINI_API_KEY is set")
    
    print("    Falling back to direct Gemini API...")
    # Use gemini-2.0-flash for better quota limits (gemini-2.5-pro has strict free tier limits)
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8000,
            "temperature": 0.7,
        }
    }
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
        
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise Exception(f"No candidates in response: {data}")
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        return "".join(p.get("text", "") for p in parts)


async def main():
    """Main review workflow."""
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║     GEMINI 2.5 PRO - PLATFORM ASSESSMENT                              ║")
    print("║     Reviewing AI Coding Platform documentation                        ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set")
        print("Run: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Load documents
    print("━━━ Loading platform documentation... ━━━")
    overview, adr = await load_documents()
    print(f"  ✓ Loaded GOOGLE_INTERVIEW_OVERVIEW.md ({len(overview)} chars)")
    print(f"  ✓ Loaded ARCHITECTURE_DECISION_RECORD.md ({len(adr)} chars)")
    print()
    
    # Build prompt
    prompt = build_review_prompt(overview, adr)
    print(f"━━━ Built review prompt ({len(prompt)} chars) ━━━")
    print()
    
    # Get review
    print("━━━ Sending to Gemini 2.5 Pro for assessment... ━━━")
    print("    (This may take 30-60 seconds)")
    print()
    
    try:
        if USE_PROVIDER:
            print("    Using GeminiProvider from llm-gateway")
            assessment = await review_with_provider(prompt)
        else:
            print("    Using direct httpx (GeminiProvider not available)")
            assessment = await review_with_httpx(prompt)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "GEMINI_ASSESSMENT.md"
    
    with open(output_file, "w") as f:
        f.write(f"# Gemini 2.5 Pro Platform Assessment\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        f.write(assessment)
    
    # Display
    print("═══════════════════════════════════════════════════════════════════════")
    print("GEMINI 2.5 PRO ASSESSMENT")
    print("═══════════════════════════════════════════════════════════════════════")
    print()
    print(assessment)
    print()
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"Assessment saved to: {output_file}")
    print("═══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
