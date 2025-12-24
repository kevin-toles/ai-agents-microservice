#!/usr/bin/env python3
"""
Retry Failed Dispute Resolution - Retry terms with multiple models including DeepSeek.

This script:
1. Loads disputed terms where GPT and Claude disagreed
2. Adds DeepSeek Reasoner as a third model for tie-breaking
3. Uses majority voting (2 out of 3) to resolve disputes
4. Retries with LONGER delays (2-3 seconds between requests)

Usage:
    python scripts/poc_dispute_retry_failed.py
    python scripts/poc_dispute_retry_failed.py --dry-run  # Test with 10 terms
    python scripts/poc_dispute_retry_failed.py --include-deepseek  # Add DeepSeek to voting
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import httpx
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.context.book_context import BookContextLookup, TermContext

# =============================================================================
# Configuration - SLOWER for reliability
# =============================================================================

LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
DATA_DIR = Path(__file__).parent.parent / "data" / "concept_validation"

# Failed terms from previous run
FAILED_TERMS_FILE = DATA_DIR / "claude_failed_terms.json"

# Previous results to merge with
PREVIOUS_RESULTS_FILE = DATA_DIR / "dispute_resolution_20251222_121726.json"

# Still-disputed terms from retry
DISPUTED_TERMS_FILE = DATA_DIR / "dispute_retry_20251222_191538.json"

# SLOWER SETTINGS to avoid rate limiting
BATCH_SIZE = 2  # Down from 5
MIN_DELAY_BETWEEN_REQUESTS = 2.0  # seconds
MAX_DELAY_BETWEEN_REQUESTS = 4.0  # seconds
BATCH_DELAY = 5.0  # seconds between batches

# Preferred models for voting (will use what's available)
PREFERRED_MODELS = [
    "gpt-5.2-2025-12-11",       # OpenAI GPT-5.2
    "claude-opus-4-5-20251101", # Anthropic Claude Opus 4.5
    "deepseek-reasoner",        # DeepSeek Reasoner (tie-breaker)
]


# =============================================================================
# Gateway Model Discovery
# =============================================================================


async def discover_available_models() -> list[str]:
    """Query gateway to find available models."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LLM_GATEWAY_URL}/v1/models", timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


def select_models_for_voting(
    available_models: list[str],
    include_deepseek: bool = False,
    previous_models: list[str] | None = None,
) -> list[dict]:
    """
    Select models for voting based on availability.
    
    Priority:
    1. Use previous models if provided (for consistency)
    2. Add DeepSeek if --include-deepseek flag is set
    3. Fall back to preferred models that are available
    """
    selected = []
    
    # First, check if previous models are still available
    if previous_models:
        for model_id in previous_models:
            # Allow partial matching for model IDs
            if model_id in available_models or any(model_id in m for m in available_models):
                matched = model_id if model_id in available_models else next(
                    (m for m in available_models if model_id in m), model_id
                )
                selected.append({"model_id": matched, "name": matched.split("-")[0]})
    
    # If no previous models, use preferred models
    if not selected:
        for model_id in PREFERRED_MODELS[:2]:  # GPT and Claude
            if model_id in available_models:
                selected.append({"model_id": model_id, "name": model_id.split("-")[0]})
    
    # Add DeepSeek if requested and available
    if include_deepseek:
        deepseek_model = "deepseek-reasoner"
        if deepseek_model in available_models:
            if not any(m["model_id"] == deepseek_model for m in selected):
                selected.append({"model_id": deepseek_model, "name": "deepseek"})
    
    return selected


# =============================================================================
# Prompt Building
# =============================================================================


def build_dispute_prompt(term: str, context: TermContext, source: str) -> str:
    """Build prompt for dispute resolution with context evidence."""
    evidence_lines = []
    for occ in context.occurrences[:5]:
        evidence_lines.append(f"  - [{occ.source_type}] {occ.book_title}, Ch.{occ.chapter_number}")
        evidence_lines.append(f"    \"{occ.context_snippet[:200]}...\"")
    
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "  No direct evidence found in corpus."
    
    stats = []
    if context.appears_in_concept_lists:
        stats.append("appears in existing concept lists")
    if context.appears_in_keyword_lists:
        stats.append("appears in keyword lists")
    if context.appears_in_summaries:
        stats.append("found in chapter summaries")
    if context.appears_in_raw_text:
        stats.append("found in raw text")
    stats_text = ", ".join(stats) if stats else "not found in any source"
    
    prompt = f"""You are evaluating whether "{term}" is a valid TECHNICAL CONCEPT.

BACKGROUND:
- This term was previously classified as a concept by {source.upper()} but NOT by the other model.
- We are re-evaluating with actual evidence from our 201-book technical corpus.

CORPUS EVIDENCE:
- Found in: {context.books_found_in} books, {context.chapters_found_in} chapters
- Status: {stats_text}

EVIDENCE SNIPPETS:
{evidence_text}

CLASSIFICATION CRITERIA:
A valid technical concept should be:
1. A TECHNICAL term (programming, architecture, AI/ML, data science, etc.)
2. TEACHABLE - something that can be learned and applied
3. ACTIONABLE - has practical implications for software/system design
4. Not just a common English word or generic phrase

Based on the evidence above, is "{term}" a valid technical concept?

Respond with ONLY one word: CONCEPT or KEYWORD or REJECT

- CONCEPT: Clear technical concept (e.g., "microservice", "bounded context", "ACID")
- KEYWORD: Related technical term but not a core concept (e.g., "database", "server")
- REJECT: Not a valid technical term (common word, noise, false positive)
"""
    return prompt


# =============================================================================
# LLM Evaluation with Rate Limiting
# =============================================================================


async def evaluate_with_delay(
    client: httpx.AsyncClient,
    term: str,
    context: TermContext,
    source: str,
    model_id: str,
) -> tuple[str, str | None]:
    """Evaluate a term with built-in delay to avoid rate limiting."""
    prompt = build_dispute_prompt(term, context, source)
    
    # Random delay to avoid thundering herd
    delay = random.uniform(MIN_DELAY_BETWEEN_REQUESTS, MAX_DELAY_BETWEEN_REQUESTS)
    await asyncio.sleep(delay)
    
    try:
        response = await client.post(
            f"{LLM_GATEWAY_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1,
            },
            timeout=60.0,  # Longer timeout
        )
        
        if response.status_code != 200:
            # Try to get error details from response body
            try:
                error_body = response.json()
                error_msg = error_body.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_msg = f"HTTP {response.status_code}"
            return "", error_msg
        
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        
        if "CONCEPT" in content:
            return "CONCEPT", None
        elif "KEYWORD" in content:
            return "KEYWORD", None
        elif "REJECT" in content:
            return "REJECT", None
        else:
            return "UNKNOWN", f"Unparseable: {content[:50]}"
            
    except httpx.TimeoutException:
        return "", "Timeout"
    except Exception as e:
        return "", str(e)


# =============================================================================
# Main Retry Logic
# =============================================================================


async def retry_failed_terms(
    failed_terms: list[str],
    models: list[dict],
    dry_run: bool = False,
    previous_votes: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Retry failed terms with slower processing.
    
    Args:
        failed_terms: List of terms to process
        models: List of models to use for NEW evaluations
        dry_run: If True, only process 10 terms
        previous_votes: Dict of term -> {model_id: vote} from previous runs
                        If provided, only call models not already in votes
    """
    
    print("Initializing book context lookup...")
    lookup = BookContextLookup()
    
    if dry_run:
        failed_terms = failed_terms[:10]
        print(f"DRY RUN: Processing only {len(failed_terms)} terms")
    
    print(f"Processing {len(failed_terms)} failed terms...")
    print(f"Settings: batch_size={BATCH_SIZE}, delay={MIN_DELAY_BETWEEN_REQUESTS}-{MAX_DELAY_BETWEEN_REQUESTS}s")
    
    if previous_votes:
        print(f"📊 Using {len(previous_votes)} previous vote records (only calling new models)")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_retried": len(failed_terms),
        "settings": {
            "batch_size": BATCH_SIZE,
            "min_delay": MIN_DELAY_BETWEEN_REQUESTS,
            "max_delay": MAX_DELAY_BETWEEN_REQUESTS,
            "batch_delay": BATCH_DELAY,
        },
        "newly_agreed_concepts": [],
        "newly_agreed_keywords": [],
        "both_rejected": [],
        "still_disputed": [],
        "errors": [],
        "detailed_results": {},
    }
    
    async with httpx.AsyncClient() as client:
        for i in tqdm(range(0, len(failed_terms), BATCH_SIZE), desc="Retrying"):
            batch = failed_terms[i:i + BATCH_SIZE]
            
            for term in batch:
                # Get book context
                context = lookup.find_term_context(term, max_results=5)
                
                # Log evidence
                if context.books_found_in > 0:
                    evidence_parts = []
                    if context.appears_in_concept_lists:
                        evidence_parts.append("concepts")
                    if context.appears_in_keyword_lists:
                        evidence_parts.append("keywords")
                    if context.appears_in_summaries:
                        evidence_parts.append("summaries")
                    tqdm.write(f"  📚 '{term}': {context.books_found_in} books [{', '.join(evidence_parts)}]")
                else:
                    tqdm.write(f"  ⚠️  '{term}': NO evidence")
                
                # Evaluate with each model (sequentially to avoid rate limits)
                model_votes = {}
                
                for model in models:
                    classification, error = await evaluate_with_delay(
                        client, term, context, "previous_run", model["model_id"]
                    )
                    
                    if error:
                        model_votes[model["model_id"]] = f"ERROR: {error}"
                        results["errors"].append({
                            "term": term,
                            "model": model["model_id"],
                            "error": error,
                        })
                    else:
                        model_votes[model["model_id"]] = classification
                
                # Save detailed result
                results["detailed_results"][term] = {
                    "books_found_in": context.books_found_in,
                    "chapters_found_in": context.chapters_found_in,
                    "model_votes": model_votes,
                }
                
                # Determine outcome using majority voting
                votes = [v for v in model_votes.values() if not str(v).startswith("ERROR")]
                vote_str = " vs ".join([f"{m.split('-')[0]}={v}" for m, v in model_votes.items()])
                
                num_models = len([v for v in model_votes.values() if not str(v).startswith("ERROR")])
                majority_threshold = (num_models // 2) + 1  # e.g., 2 out of 3
                
                if len(votes) >= 2:
                    # Count votes for each classification
                    concept_votes = sum(1 for v in votes if v == "CONCEPT")
                    keyword_votes = sum(1 for v in votes if v == "KEYWORD")
                    reject_votes = sum(1 for v in votes if v == "REJECT")
                    
                    if concept_votes >= majority_threshold:
                        results["newly_agreed_concepts"].append(term)
                        tqdm.write(f"    ✅ CONCEPT: {term} ({vote_str}) [{concept_votes}/{num_models} votes]")
                    elif keyword_votes >= majority_threshold:
                        results["newly_agreed_keywords"].append(term)
                        tqdm.write(f"    📝 KEYWORD: {term} ({vote_str}) [{keyword_votes}/{num_models} votes]")
                    elif reject_votes >= majority_threshold:
                        results["both_rejected"].append(term)
                        tqdm.write(f"    ❌ REJECTED: {term} ({vote_str}) [{reject_votes}/{num_models} votes]")
                    else:
                        results["still_disputed"].append({
                            "term": term,
                            "votes": model_votes,
                        })
                        tqdm.write(f"    ⚡ DISPUTED: {term} ({vote_str}) [no majority]")
                else:
                    tqdm.write(f"    ⚠️  INCOMPLETE: {term} ({vote_str})")
            
            # Longer delay between batches
            if i + BATCH_SIZE < len(failed_terms):
                tqdm.write(f"  💤 Batch complete, waiting {BATCH_DELAY}s...")
                await asyncio.sleep(BATCH_DELAY)
    
    return results


def get_models_from_previous_results() -> list[dict]:
    """Get models from previous results file."""
    with open(PREVIOUS_RESULTS_FILE) as f:
        data = json.load(f)
    
    models_used = data.get("models_used", [])
    return [{"model_id": m, "name": m.split("-")[0]} for m in models_used]


def merge_with_previous(retry_results: dict, previous_file: Path) -> dict:
    """Merge retry results with previous run."""
    with open(previous_file) as f:
        previous = json.load(f)
    
    # Get previous successful results
    prev_concepts = set(previous.get("newly_agreed_concepts", []))
    prev_keywords = set(previous.get("newly_agreed_keywords", []))
    prev_rejected = set(previous.get("both_rejected", []))
    
    # Add new results
    new_concepts = set(retry_results.get("newly_agreed_concepts", []))
    new_keywords = set(retry_results.get("newly_agreed_keywords", []))
    new_rejected = set(retry_results.get("both_rejected", []))
    
    merged = {
        "timestamp": datetime.now().isoformat(),
        "source": "Merged dispute resolution (original + retry)",
        "summary": {
            "original_concepts": len(prev_concepts),
            "retry_concepts": len(new_concepts),
            "total_concepts": len(prev_concepts | new_concepts),
            "original_keywords": len(prev_keywords),
            "retry_keywords": len(new_keywords),
            "total_keywords": len(prev_keywords | new_keywords),
            "total_rejected": len(prev_rejected | new_rejected),
            "retry_still_disputed": len(retry_results.get("still_disputed", [])),
            "retry_errors": len(retry_results.get("errors", [])),
        },
        "newly_agreed_concepts": sorted(prev_concepts | new_concepts),
        "newly_agreed_keywords": sorted(prev_keywords | new_keywords),
        "both_rejected": sorted(prev_rejected | new_rejected),
        "still_disputed": retry_results.get("still_disputed", []),
        "retry_errors": retry_results.get("errors", []),
    }
    
    return merged


# =============================================================================
# Main
# =============================================================================


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Retry Failed Dispute Resolution")
    parser.add_argument("--dry-run", action="store_true", help="Test with only 10 terms")
    parser.add_argument("--include-deepseek", action="store_true", 
                        help="Include DeepSeek Reasoner as third model for tie-breaking")
    parser.add_argument("--use-disputed", action="store_true",
                        help="Process still-disputed terms from previous retry (with DeepSeek tie-breaker)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RETRY DISPUTE RESOLUTION" + (" WITH DEEPSEEK" if args.include_deepseek else ""))
    print("=" * 70)
    
    # Check gateway
    print("\nChecking LLM Gateway...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LLM_GATEWAY_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                print(f"✅ Gateway reachable at {LLM_GATEWAY_URL}")
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        return
    
    # Discover available models
    print("\nDiscovering available models...")
    available_models = await discover_available_models()
    print(f"✅ Found {len(available_models)} models from gateway")
    
    # Load terms to process
    print("\nLoading terms to process...")
    
    if args.use_disputed:
        # Load still-disputed terms from previous retry (these need DeepSeek tie-breaker)
        if not DISPUTED_TERMS_FILE.exists():
            print(f"❌ Disputed terms file not found: {DISPUTED_TERMS_FILE}")
            return
        
        with open(DISPUTED_TERMS_FILE) as f:
            disputed_data = json.load(f)
        
        # Extract just the term names from disputed items
        still_disputed = disputed_data.get("still_disputed", [])
        failed_terms = [item["term"] if isinstance(item, dict) else item for item in still_disputed]
        print(f"✅ Loaded {len(failed_terms)} still-disputed terms (need tie-breaker)")
        
        # Force DeepSeek for tie-breaking
        args.include_deepseek = True
        
    else:
        # Load failed terms from original run
        if not FAILED_TERMS_FILE.exists():
            print(f"❌ Failed terms file not found: {FAILED_TERMS_FILE}")
            return
        
        with open(FAILED_TERMS_FILE) as f:
            failed_data = json.load(f)
        
        failed_terms = failed_data.get("terms", [])
        print(f"✅ Loaded {len(failed_terms)} failed terms")
    
    # Get models from previous results
    previous_models = None
    if PREVIOUS_RESULTS_FILE.exists():
        try:
            with open(PREVIOUS_RESULTS_FILE) as f:
                prev_data = json.load(f)
            previous_models = prev_data.get("models_used", [])
        except Exception:
            pass
    
    # Select models for voting
    models = select_models_for_voting(
        available_models,
        include_deepseek=args.include_deepseek,
        previous_models=previous_models,
    )
    
    if not models:
        print("❌ No models available for voting!")
        return
    
    print(f"✅ Using {len(models)} models: {[m['model_id'] for m in models]}")
    
    if args.include_deepseek and any("deepseek" in m["model_id"] for m in models):
        print("   🧠 DeepSeek Reasoner included for tie-breaking!")
    
    # Estimate time
    estimated_time = len(failed_terms) * (MIN_DELAY_BETWEEN_REQUESTS + MAX_DELAY_BETWEEN_REQUESTS) / 2 * len(models)
    estimated_time += (len(failed_terms) / BATCH_SIZE) * BATCH_DELAY
    print(f"⏱️  Estimated time: {estimated_time/60:.0f} minutes")
    
    # Run retry
    print("\n" + "-" * 70)
    print("RETRYING WITH SLOWER RATE")
    print("-" * 70)
    
    retry_results = await retry_failed_terms(
        failed_terms,
        models,
        dry_run=args.dry_run,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("RETRY SUMMARY")
    print("=" * 70)
    print(f"  Newly agreed CONCEPTS: {len(retry_results['newly_agreed_concepts'])}")
    print(f"  Newly agreed KEYWORDS: {len(retry_results['newly_agreed_keywords'])}")
    print(f"  Both rejected: {len(retry_results['both_rejected'])}")
    print(f"  Still disputed: {len(retry_results['still_disputed'])}")
    print(f"  Errors: {len(retry_results['errors'])}")
    
    if retry_results['newly_agreed_concepts']:
        print(f"\nSample newly agreed concepts: {retry_results['newly_agreed_concepts'][:10]}")
    
    # Save and merge
    if not args.dry_run:
        # Save retry results
        retry_file = DATA_DIR / f"dispute_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(retry_file, "w") as f:
            json.dump(retry_results, f, indent=2)
        print(f"\nSaved retry results to: {retry_file}")
        
        # Merge with previous
        print("\nMerging with previous results...")
        merged = merge_with_previous(retry_results, PREVIOUS_RESULTS_FILE)
        
        merged_file = DATA_DIR / "dispute_resolution_merged.json"
        with open(merged_file, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Saved merged results to: {merged_file}")
        
        print("\n" + "=" * 70)
        print("FINAL MERGED TOTALS")
        print("=" * 70)
        for key, value in merged["summary"].items():
            print(f"  {key}: {value}")
    else:
        print("\nDry run complete - no files saved")


if __name__ == "__main__":
    asyncio.run(main())
