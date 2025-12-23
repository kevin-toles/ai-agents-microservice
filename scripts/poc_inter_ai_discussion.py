#!/usr/bin/env python3
"""
POC: Inter-AI Discussion System

True multi-agent orchestration where LLMs actively discuss and debate
term classifications, seeing each other's reasoning and responding.

Flow for each term:
1. GPT-5.2 gives initial assessment with reasoning
2. Claude Opus 4.5 sees GPT's assessment and responds (agree/disagree + why)
3. DeepSeek Reasoner sees both and provides final arbitration
4. If still no consensus after 2 rounds, majority wins

This creates actual AI-to-AI dialogue, not isolated voting.

Usage:
    python scripts/poc_inter_ai_discussion.py --dry-run  # Test with 10 terms
    python scripts/poc_inter_ai_discussion.py            # Full run
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.context.book_context import BookContextLookup, TermContext

# =============================================================================
# Configuration
# =============================================================================

LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
DATA_DIR = Path(__file__).parent.parent / "data" / "concept_validation"
OUTPUT_DIR = DATA_DIR

# The 3 AI participants
PARTICIPANTS = [
    {"id": "gpt", "model": "gpt-5.2-2025-12-11", "name": "GPT-5.2"},
    {"id": "claude", "model": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5"},
    {"id": "deepseek", "model": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
]

MAX_DISCUSSION_ROUNDS = 2
REQUEST_DELAY = 1.5  # seconds between API calls

# =============================================================================
# Prompts for Discussion
# =============================================================================

INITIAL_ASSESSMENT_PROMPT = """You are participating in a collaborative discussion with other AI models to determine whether a technical term is a CONCEPT or KEYWORD.

DEFINITIONS:
- CONCEPT: An abstract idea representing meaningful architectural, design, or methodological knowledge in software engineering. Examples: "microservice architecture", "event sourcing", "dependency injection"
- KEYWORD: A generic term, identifier, or technical detail without deep conceptual meaning. Examples: "code", "file", "data", "config"

CONTEXT FROM TECHNICAL BOOKS:
{context}

TERM TO EVALUATE: "{term}"

Provide your initial assessment:
1. Your classification: CONCEPT or KEYWORD
2. Your reasoning (2-3 sentences explaining why)
3. Confidence level: HIGH, MEDIUM, or LOW

Format your response as:
CLASSIFICATION: [CONCEPT/KEYWORD]
REASONING: [Your explanation]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

RESPONSE_TO_PEER_PROMPT = """You are participating in a collaborative discussion with other AI models about whether "{term}" is a CONCEPT or KEYWORD.

DEFINITIONS:
- CONCEPT: An abstract idea representing meaningful architectural, design, or methodological knowledge
- KEYWORD: A generic term without deep conceptual meaning

CONTEXT FROM TECHNICAL BOOKS:
{context}

PREVIOUS ASSESSMENT BY {peer_name}:
{peer_assessment}

Based on {peer_name}'s assessment:
1. Do you AGREE or DISAGREE with their classification?
2. If you disagree, explain why and provide your alternative classification
3. If you agree, add any supporting evidence or nuance

Format your response as:
STANCE: [AGREE/DISAGREE]
CLASSIFICATION: [CONCEPT/KEYWORD]
REASONING: [Your response to their argument]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

DEEPSEEK_DISCUSSION_PROMPT = """You are participating in a collaborative discussion with GPT-5.2 and Claude Opus 4.5 about whether "{term}" is a CONCEPT or KEYWORD.

DEFINITIONS:
- CONCEPT: An abstract idea representing meaningful architectural, design, or methodological knowledge
- KEYWORD: A generic term without deep conceptual meaning

CONTEXT FROM TECHNICAL BOOKS:
{context}

DISCUSSION SO FAR:
{discussion_history}

Now it's your turn to contribute:
1. Review the arguments from GPT-5.2 and Claude Opus 4.5
2. Do you AGREE with their conclusions or have a DIFFERENT perspective?
3. Provide your own classification with reasoning

Format your response as:
STANCE: [AGREE/DISAGREE with majority]
CLASSIFICATION: [CONCEPT/KEYWORD]
REASONING: [Your analysis - what did the others miss or get right?]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""


# =============================================================================
# LLM Communication
# =============================================================================


async def call_llm(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int = 500,
) -> tuple[str, str | None]:
    """Call LLM via gateway."""
    try:
        await asyncio.sleep(REQUEST_DELAY)
        
        response = await client.post(
            f"{LLM_GATEWAY_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.4,
            },
            timeout=90.0,
        )
        
        if response.status_code != 200:
            try:
                err = response.json().get("error", {}).get("message", "")[:80]
            except Exception:
                err = f"HTTP {response.status_code}"
            return "", err
        
        content = response.json()["choices"][0]["message"]["content"]
        return content, None
        
    except Exception as e:
        return "", str(e)[:80]


def format_context(context: TermContext) -> str:
    """Format book context for prompt."""
    if context.books_found_in == 0:
        return "No direct evidence found in the technical book corpus."
    
    lines = [f"Found in {context.books_found_in} books, {context.chapters_found_in} chapters:"]
    
    for occ in context.occurrences[:3]:
        lines.append(f"- [{occ.source_type}] {occ.book_title}, Ch.{occ.chapter_number}")
        lines.append(f"  \"{occ.context_snippet[:150]}...\"")
    
    return "\n".join(lines)


def parse_assessment(response: str) -> dict:
    """Parse structured assessment from LLM response."""
    result = {
        "classification": None,
        "reasoning": "",
        "confidence": "MEDIUM",
        "stance": None,
        "raw": response,
    }
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("CLASSIFICATION:"):
            val = line.replace("CLASSIFICATION:", "").strip().upper()
            if "CONCEPT" in val:
                result["classification"] = "CONCEPT"
            elif "KEYWORD" in val:
                result["classification"] = "KEYWORD"
        elif line.startswith("FINAL_CLASSIFICATION:"):
            val = line.replace("FINAL_CLASSIFICATION:", "").strip().upper()
            if "CONCEPT" in val:
                result["classification"] = "CONCEPT"
            elif "KEYWORD" in val:
                result["classification"] = "KEYWORD"
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            val = line.replace("CONFIDENCE:", "").strip().upper()
            if val in ["HIGH", "MEDIUM", "LOW"]:
                result["confidence"] = val
        elif line.startswith("STANCE:"):
            val = line.replace("STANCE:", "").strip().upper()
            if "AGREE" in val:
                result["stance"] = "AGREE"
            elif "DISAGREE" in val:
                result["stance"] = "DISAGREE"
    
    return result


# =============================================================================
# Discussion Orchestration
# =============================================================================


async def run_discussion(
    client: httpx.AsyncClient,
    term: str,
    context: TermContext,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a full multi-agent discussion for one term."""
    
    context_text = format_context(context)
    discussion = {
        "term": term,
        "rounds": [],
        "final_classification": None,
        "consensus_achieved": False,
        "participants": {},
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DISCUSSING: \"{term}\"")
        print(f"{'='*60}")
    
    # ==========================================================================
    # ROUND 1: Initial assessments and responses
    # ==========================================================================
    
    round1 = {"round": 1, "exchanges": []}
    
    # --- GPT gives initial assessment ---
    gpt = PARTICIPANTS[0]
    if verbose:
        print(f"\n[{gpt['name']}] Providing initial assessment...")
    
    prompt = INITIAL_ASSESSMENT_PROMPT.format(term=term, context=context_text)
    response, error = await call_llm(client, gpt["model"], prompt)
    
    if error:
        if verbose:
            print(f"  ❌ Error: {error}")
        discussion["participants"]["gpt"] = {"error": error}
        gpt_assessment = None
    else:
        gpt_assessment = parse_assessment(response)
        discussion["participants"]["gpt"] = gpt_assessment
        round1["exchanges"].append({
            "speaker": gpt["name"],
            "type": "initial",
            "assessment": gpt_assessment,
        })
        if verbose:
            print(f"  → {gpt_assessment['classification']} ({gpt_assessment['confidence']})")
            print(f"  → {gpt_assessment['reasoning'][:100]}...")
    
    # --- Claude responds to GPT ---
    claude = PARTICIPANTS[1]
    if verbose:
        print(f"\n[{claude['name']}] Responding to {gpt['name']}...")
    
    if gpt_assessment:
        peer_text = f"CLASSIFICATION: {gpt_assessment['classification']}\nREASONING: {gpt_assessment['reasoning']}\nCONFIDENCE: {gpt_assessment['confidence']}"
        prompt = RESPONSE_TO_PEER_PROMPT.format(
            term=term,
            context=context_text,
            peer_name=gpt["name"],
            peer_assessment=peer_text,
        )
    else:
        # Fall back to initial assessment if GPT failed
        prompt = INITIAL_ASSESSMENT_PROMPT.format(term=term, context=context_text)
    
    response, error = await call_llm(client, claude["model"], prompt)
    
    if error:
        if verbose:
            print(f"  ❌ Error: {error}")
        discussion["participants"]["claude"] = {"error": error}
        claude_assessment = None
    else:
        claude_assessment = parse_assessment(response)
        discussion["participants"]["claude"] = claude_assessment
        round1["exchanges"].append({
            "speaker": claude["name"],
            "type": "response",
            "responding_to": gpt["name"],
            "assessment": claude_assessment,
        })
        if verbose:
            stance = claude_assessment.get('stance', 'N/A')
            print(f"  → {stance} → {claude_assessment['classification']} ({claude_assessment['confidence']})")
            print(f"  → {claude_assessment['reasoning'][:100]}...")
    
    discussion["rounds"].append(round1)
    
    # ==========================================================================
    # Collect classifications so far
    # ==========================================================================
    
    classifications = []
    if gpt_assessment and gpt_assessment.get("classification"):
        classifications.append(("gpt", gpt_assessment["classification"]))
    if claude_assessment and claude_assessment.get("classification"):
        classifications.append(("claude", claude_assessment["classification"]))
    
    # ==========================================================================
    # ROUND 2: DeepSeek ALWAYS participates (not just as tie-breaker)
    # ==========================================================================
    
    if verbose:
        print(f"\n[{PARTICIPANTS[2]['name']}] Joining the discussion...")
    
    round2 = {"round": 2, "exchanges": []}
    deepseek = PARTICIPANTS[2]
    
    # Build discussion history for DeepSeek to respond to
    history_lines = []
    if gpt_assessment:
        history_lines.append(f"{gpt['name']} ({gpt_assessment['confidence']} confidence):")
        history_lines.append(f"  Classification: {gpt_assessment['classification']}")
        history_lines.append(f"  Reasoning: {gpt_assessment['reasoning']}")
        history_lines.append("")
    
    if claude_assessment:
        stance_text = f" (Stance: {claude_assessment.get('stance', 'N/A')})" if claude_assessment.get('stance') else ""
        history_lines.append(f"{claude['name']}{stance_text} ({claude_assessment['confidence']} confidence):")
        history_lines.append(f"  Classification: {claude_assessment['classification']}")
        history_lines.append(f"  Reasoning: {claude_assessment['reasoning']}")
    
    discussion_history = "\n".join(history_lines)
    
    prompt = DEEPSEEK_DISCUSSION_PROMPT.format(
        term=term,
        context=context_text,
        discussion_history=discussion_history,
    )
    
    response, error = await call_llm(client, deepseek["model"], prompt)
    
    if error:
        if verbose:
            print(f"  ❌ Error: {error}")
        discussion["participants"]["deepseek"] = {"error": error}
        deepseek_assessment = None
    else:
        deepseek_assessment = parse_assessment(response)
        discussion["participants"]["deepseek"] = deepseek_assessment
        round2["exchanges"].append({
            "speaker": deepseek["name"],
            "type": "discussion",
            "assessment": deepseek_assessment,
        })
        
        if deepseek_assessment.get("classification"):
            classifications.append(("deepseek", deepseek_assessment["classification"]))
        
        if verbose:
            stance = deepseek_assessment.get('stance', 'N/A')
            print(f"  → {stance} → {deepseek_assessment['classification']} ({deepseek_assessment['confidence']})")
            print(f"  → {deepseek_assessment['reasoning'][:100]}...")
    
    discussion["rounds"].append(round2)
    
    # ==========================================================================
    # FINAL CONSENSUS: Majority vote of all 3 participants
    # ==========================================================================
    
    if verbose:
        print(f"\n📊 FINAL VOTE COUNT:")
    
    votes = [c[1] for c in classifications]
    concept_votes = votes.count("CONCEPT")
    keyword_votes = votes.count("KEYWORD")
    
    vote_summary = []
    for model, vote in classifications:
        vote_summary.append(f"{model}={vote}")
    
    if verbose:
        print(f"   {' | '.join(vote_summary)}")
        print(f"   CONCEPT: {concept_votes} votes | KEYWORD: {keyword_votes} votes")
    
    if concept_votes > keyword_votes:
        discussion["final_classification"] = "CONCEPT"
        discussion["consensus_achieved"] = concept_votes == len(votes)
    elif keyword_votes > concept_votes:
        discussion["final_classification"] = "KEYWORD"
        discussion["consensus_achieved"] = keyword_votes == len(votes)
    else:
        # Tie - use DeepSeek's vote as tiebreaker if available
        if deepseek_assessment and deepseek_assessment.get("classification"):
            discussion["final_classification"] = deepseek_assessment["classification"]
        else:
            discussion["final_classification"] = "KEYWORD"  # Default to keyword on tie
        discussion["consensus_achieved"] = False
    
    discussion["consensus_method"] = "majority_vote_3way"
    discussion["vote_counts"] = {"concept": concept_votes, "keyword": keyword_votes}
    
    if verbose:
        consensus_str = "✅ UNANIMOUS" if discussion["consensus_achieved"] else "⚖️ MAJORITY"
        print(f"\n{consensus_str}: {discussion['final_classification']}")
    
    return discussion


# =============================================================================
# Main Processing
# =============================================================================


async def load_disputed_terms() -> list[dict]:
    """Load disputed terms needing resolution."""
    # First try the remaining terms file (for resume)
    remaining_file = DATA_DIR / "disputed_terms_remaining.json"
    
    if remaining_file.exists():
        with open(remaining_file) as f:
            data = json.load(f)
        all_disputed = data.get("all_disputed", [])
        print(f"  Loaded {len(all_disputed)} remaining terms from {remaining_file.name}")
        return [{"term": t} for t in all_disputed]
    
    # Then try the full disputed terms file
    disputed_file = DATA_DIR / "disputed_terms_needing_resolution.json"
    
    if disputed_file.exists():
        with open(disputed_file) as f:
            data = json.load(f)
        all_disputed = data.get("all_disputed", [])
        print(f"  Loaded {len(all_disputed)} terms from {disputed_file.name}")
        return [{"term": t} for t in all_disputed]
    
    # Fallback to old file
    old_file = DATA_DIR / "dispute_retry_20251222_191538.json"
    if old_file.exists():
        with open(old_file) as f:
            data = json.load(f)
        return data.get("still_disputed", [])
    
    return []


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inter-AI Discussion System")
    parser.add_argument("--dry-run", action="store_true", help="Test with 10 terms")
    parser.add_argument("--terms", type=str, help="Comma-separated terms to discuss")
    parser.add_argument("--use-disputed", action="store_true", help="Use disputed terms from previous run")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("INTER-AI DISCUSSION SYSTEM")
    print("GPT-5.2 ↔ Claude Opus 4.5 ↔ DeepSeek Reasoner")
    print("=" * 70)
    
    # Check gateway
    print("\nChecking LLM Gateway...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LLM_GATEWAY_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                print(f"✅ Gateway reachable")
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        return
    
    # Load terms
    if args.terms:
        terms = [{"term": t.strip()} for t in args.terms.split(",")]
        print(f"\nUsing {len(terms)} specified terms")
    elif args.use_disputed:
        terms = await load_disputed_terms()
        print(f"\n✅ Loaded {len(terms)} disputed terms")
    else:
        # Default test terms
        terms = [
            {"term": "microservice architecture"},
            {"term": "event sourcing"},
            {"term": "dependency injection"},
            {"term": "code"},
            {"term": "file"},
            {"term": "access routines"},
            {"term": "circuit breaker pattern"},
            {"term": "data"},
            {"term": "continuous integration"},
            {"term": "config"},
        ]
        print(f"\nUsing {len(terms)} default test terms")
    
    if args.dry_run:
        terms = terms[:10]
        print(f"DRY RUN: Limited to {len(terms)} terms")
    
    # Initialize book context
    print("\nInitializing book context lookup...")
    lookup = BookContextLookup()
    
    # Process terms
    print(f"\n{'='*70}")
    print("STARTING DISCUSSIONS")
    print(f"{'='*70}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_terms": len(terms),
        "concepts": [],
        "keywords": [],
        "errors": [],
        "discussions": [],
    }
    
    async with httpx.AsyncClient() as client:
        for i, term_data in enumerate(terms, 1):
            term = term_data["term"] if isinstance(term_data, dict) else term_data
            
            print(f"\n[{i}/{len(terms)}] Processing: {term}")
            
            # Get book context
            context = lookup.find_term_context(term, max_results=5)
            
            # Run discussion
            discussion = await run_discussion(
                client, term, context, verbose=not args.quiet
            )
            
            results["discussions"].append(discussion)
            
            if discussion["final_classification"] == "CONCEPT":
                results["concepts"].append(term)
                if args.quiet:
                    print(f"  → CONCEPT ✅")
            elif discussion["final_classification"] == "KEYWORD":
                results["keywords"].append(term)
                if args.quiet:
                    print(f"  → KEYWORD")
            else:
                results["errors"].append({"term": term, "reason": "no_classification"})
                if args.quiet:
                    print(f"  → ERROR ❌")
    
    # Summary
    print(f"\n{'='*70}")
    print("DISCUSSION RESULTS")
    print(f"{'='*70}")
    print(f"Total discussed: {len(terms)}")
    print(f"Classified as CONCEPT: {len(results['concepts'])}")
    print(f"Classified as KEYWORD: {len(results['keywords'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results["concepts"]:
        print(f"\nCONCEPTS identified:")
        for c in results["concepts"][:20]:
            print(f"  ✅ {c}")
        if len(results["concepts"]) > 20:
            print(f"  ... and {len(results['concepts']) - 20} more")
    
    # Save results
    if not args.dry_run:
        output_file = OUTPUT_DIR / f"inter_ai_discussion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nDry run - results not saved")


if __name__ == "__main__":
    asyncio.run(main())
