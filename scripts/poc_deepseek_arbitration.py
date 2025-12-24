#!/usr/bin/env python3
"""
DeepSeek Reasoner Arbitration Script

This script uses DeepSeek Reasoner to arbitrate disputed terms where
GPT-5.2 and Claude Opus 4.5 disagreed on classification.

DeepSeek Reasoner is particularly well-suited for this task because:
1. It excels at step-by-step reasoning
2. It can weigh competing arguments
3. It provides transparent justifications

The script:
1. Loads disputed terms from the retry results
2. Gathers book context evidence for each term
3. Presents both model's classifications to DeepSeek
4. DeepSeek makes a final decision with reasoning
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.context.book_context import BookContextLookup

# =============================================================================
# Configuration - Dynamically discovered
# =============================================================================

GATEWAY_URL = "http://localhost:8080"
DEEPSEEK_MODEL = "deepseek-reasoner"

# Rate limiting for DeepSeek (be conservative)
MIN_DELAY = 1.0  # Minimum delay between requests
MAX_DELAY = 2.0  # Maximum delay between requests
BATCH_SIZE = 5   # Process in small batches
BATCH_DELAY = 3.0  # Delay between batches

# Paths - dynamically discovered
DATA_DIR = PROJECT_ROOT / "data" / "concept_validation"


def find_latest_retry_results() -> Path | None:
    """Find the most recent retry results file."""
    retry_files = list(DATA_DIR.glob("dispute_retry_*.json"))
    if not retry_files:
        return None
    return max(retry_files, key=lambda p: p.stat().st_mtime)


def load_disputed_terms(results_file: Path) -> list[dict]:
    """Load disputed terms from retry results."""
    with open(results_file) as f:
        data = json.load(f)
    return data.get("still_disputed", [])


async def check_gateway() -> bool:
    """Check if llm-gateway is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GATEWAY_URL}/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def check_deepseek_available() -> bool:
    """Check if DeepSeek model is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GATEWAY_URL}/v1/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return DEEPSEEK_MODEL in models
    except Exception:
        pass
    return False


def build_arbitration_prompt(
    term: str,
    votes: dict[str, str],
    evidence: dict,
) -> str:
    """Build a prompt for DeepSeek to arbitrate the dispute."""
    
    # Extract the models and their votes
    model_votes = []
    for model_id, vote in votes.items():
        # Simplify model names
        if "gpt" in model_id.lower():
            model_name = "GPT-5.2"
        elif "claude" in model_id.lower():
            model_name = "Claude Opus 4.5"
        else:
            model_name = model_id
        model_votes.append(f"- {model_name}: {vote}")
    
    # Build evidence section
    evidence_text = ""
    if evidence.get("books_found"):
        evidence_text = f"\n\nEVIDENCE FROM {evidence['total_books']} TECHNICAL BOOKS:\n"
        for source_type, items in evidence.get("evidence", {}).items():
            if items:
                evidence_text += f"\n{source_type.upper()}:\n"
                for item in items[:3]:  # Limit to 3 items per source
                    evidence_text += f"  - {item}\n"
    else:
        evidence_text = "\n\nNo direct evidence found in the book corpus."
    
    prompt = f"""You are an expert technical taxonomist arbitrating a classification dispute.

TERM TO CLASSIFY: "{term}"

TWO AI MODELS DISAGREE ON THE CLASSIFICATION:
{chr(10).join(model_votes)}

CLASSIFICATION DEFINITIONS:
- CONCEPT: A technical term that represents a distinct idea, technology, pattern, or methodology 
  that requires understanding and has specific technical meaning (e.g., "microservices", "dependency injection", "neural network")
- KEYWORD: A commonly used technical term that's useful for search/tagging but doesn't represent 
  a distinct teachable concept (e.g., "implementation", "configuration", "deployment")
- REJECT: Not a meaningful technical term - too generic, misspelled, or not related to software/tech
{evidence_text}

YOUR TASK:
1. Consider both models' classifications
2. Review any available evidence from technical books
3. Make a final decision: CONCEPT, KEYWORD, or REJECT
4. Provide brief reasoning (1-2 sentences)

RESPOND IN THIS EXACT FORMAT:
DECISION: [CONCEPT|KEYWORD|REJECT]
REASONING: [Your brief explanation]"""

    return prompt


async def call_deepseek(
    prompt: str,
    client: httpx.AsyncClient,
    timeout: float = 60.0,
) -> tuple[str | None, str | None]:
    """
    Call DeepSeek Reasoner for arbitration.
    
    Returns:
        Tuple of (decision, reasoning) or (None, error_message)
    """
    try:
        response = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temperature for consistent reasoning
                "max_tokens": 200,
            },
            timeout=timeout,
        )
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:100]}"
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Parse response
        decision = None
        reasoning = None
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip().upper()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        if decision not in ["CONCEPT", "KEYWORD", "REJECT"]:
            # Try to extract from free-form response
            content_upper = content.upper()
            if "CONCEPT" in content_upper:
                decision = "CONCEPT"
            elif "KEYWORD" in content_upper:
                decision = "KEYWORD"
            elif "REJECT" in content_upper:
                decision = "REJECT"
            reasoning = content[:200] if not reasoning else reasoning
        
        return decision, reasoning
        
    except httpx.TimeoutException:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


async def arbitrate_disputes(
    disputed_terms: list[dict],
    book_lookup: BookContextLookup,
    dry_run: bool = False,
) -> dict:
    """
    Run DeepSeek arbitration on disputed terms.
    
    Returns:
        Results dictionary with arbitration outcomes
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": DEEPSEEK_MODEL,
        "total_disputed": len(disputed_terms),
        "arbitrated_concepts": [],
        "arbitrated_keywords": [],
        "arbitrated_rejected": [],
        "still_undecided": [],
        "errors": [],
        "detailed_results": {},
    }
    
    async with httpx.AsyncClient() as client:
        batches = [
            disputed_terms[i:i + BATCH_SIZE]
            for i in range(0, len(disputed_terms), BATCH_SIZE)
        ]
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Arbitrating")):
            for item in batch:
                term = item["term"]
                votes = item["votes"]
                
                # Get evidence
                evidence = book_lookup.find_term_context(term)
                evidence_summary = {
                    "books_found": evidence.get("books_found", 0) > 0,
                    "total_books": evidence.get("books_found", 0),
                    "evidence": evidence.get("evidence", {}),
                }
                
                # Build prompt
                prompt = build_arbitration_prompt(term, votes, evidence_summary)
                
                if dry_run:
                    # Just show what would be done
                    tqdm.write(f"  📋 Would arbitrate: '{term}'")
                    tqdm.write(f"     Votes: {votes}")
                    tqdm.write(f"     Evidence: {evidence_summary['total_books']} books")
                    continue
                
                # Call DeepSeek
                decision, reasoning = await call_deepseek(prompt, client)
                
                # Random delay
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                
                # Record result
                result_entry = {
                    "term": term,
                    "original_votes": votes,
                    "deepseek_decision": decision,
                    "deepseek_reasoning": reasoning,
                    "evidence_books": evidence_summary["total_books"],
                }
                results["detailed_results"][term] = result_entry
                
                if decision == "CONCEPT":
                    results["arbitrated_concepts"].append(term)
                    tqdm.write(f"  ✅ CONCEPT: {term}")
                elif decision == "KEYWORD":
                    results["arbitrated_keywords"].append(term)
                    tqdm.write(f"  📝 KEYWORD: {term}")
                elif decision == "REJECT":
                    results["arbitrated_rejected"].append(term)
                    tqdm.write(f"  ❌ REJECTED: {term}")
                else:
                    results["still_undecided"].append(term)
                    results["errors"].append({"term": term, "error": reasoning})
                    tqdm.write(f"  ⚠️  ERROR: {term} - {reasoning}")
            
            # Batch delay
            if batch_idx < len(batches) - 1 and not dry_run:
                tqdm.write(f"  💤 Batch complete, waiting {BATCH_DELAY}s...")
                await asyncio.sleep(BATCH_DELAY)
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek Reasoner arbitration for disputed terms"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of terms to process",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("DEEPSEEK REASONER ARBITRATION")
    print("=" * 70)
    print()
    
    # Check gateway
    print("Checking LLM Gateway...")
    if not await check_gateway():
        print("❌ Gateway not reachable at", GATEWAY_URL)
        return 1
    print("✅ Gateway reachable")
    
    # Check DeepSeek availability
    print("Checking DeepSeek availability...")
    if not await check_deepseek_available():
        print(f"❌ {DEEPSEEK_MODEL} not available")
        print("   Make sure DEEPSEEK_API_KEY is set and gateway is restarted")
        return 1
    print(f"✅ {DEEPSEEK_MODEL} available")
    
    # Find retry results
    results_file = find_latest_retry_results()
    if not results_file:
        print("❌ No retry results found in", DATA_DIR)
        return 1
    print(f"✅ Using results from: {results_file.name}")
    
    # Load disputed terms
    disputed_terms = load_disputed_terms(results_file)
    print(f"✅ Loaded {len(disputed_terms)} disputed terms")
    
    if args.limit:
        disputed_terms = disputed_terms[:args.limit]
        print(f"   Limited to {args.limit} terms")
    
    # Estimate time
    total_batches = (len(disputed_terms) + BATCH_SIZE - 1) // BATCH_SIZE
    est_time = total_batches * (BATCH_SIZE * (MIN_DELAY + MAX_DELAY) / 2 + BATCH_DELAY)
    print(f"⏱️  Estimated time: {int(est_time / 60)} minutes")
    
    print()
    print("-" * 70)
    print("ARBITRATING DISPUTES")
    print("-" * 70)
    
    # Initialize book lookup
    print("Initializing book context lookup...")
    book_lookup = BookContextLookup()
    
    if args.dry_run:
        print("DRY RUN: Processing only 10 terms")
        disputed_terms = disputed_terms[:10]
    
    # Run arbitration
    results = await arbitrate_disputes(
        disputed_terms,
        book_lookup,
        dry_run=args.dry_run,
    )
    
    print()
    print("=" * 70)
    print("ARBITRATION SUMMARY")
    print("=" * 70)
    print(f"  Arbitrated as CONCEPTS: {len(results['arbitrated_concepts'])}")
    print(f"  Arbitrated as KEYWORDS: {len(results['arbitrated_keywords'])}")
    print(f"  Arbitrated as REJECTED: {len(results['arbitrated_rejected'])}")
    print(f"  Errors/Undecided: {len(results['still_undecided'])}")
    
    if results["arbitrated_concepts"]:
        print(f"\nSample arbitrated concepts: {results['arbitrated_concepts'][:10]}")
    
    # Save results
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DATA_DIR / f"deepseek_arbitration_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {output_file}")
        
        # Also save a simple summary
        summary = {
            "timestamp": results["timestamp"],
            "total_arbitrated": len(disputed_terms),
            "concepts": results["arbitrated_concepts"],
            "keywords": results["arbitrated_keywords"],
            "rejected": results["arbitrated_rejected"],
            "errors": len(results["errors"]),
        }
        summary_file = DATA_DIR / "deepseek_arbitration_latest.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_file}")
    else:
        print("\nDry run complete - no files saved")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
