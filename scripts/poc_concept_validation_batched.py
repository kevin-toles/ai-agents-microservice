#!/usr/bin/env python3
"""
POC: Batched Inter-AI Concept Validation

This script processes ALL extracted terms (10,000+) in batches of 250,
having the LLMs determine which are true CONCEPTS vs KEYWORDS.

Each batch:
1. Send 250 terms to the 3 LLMs (Qwen, GPT-5.2, Claude Opus 4.5)
2. LLMs categorize each term as CONCEPT or KEYWORD
3. Aggregate results across all batches
4. Final consensus on the complete concept vocabulary

Usage:
    python scripts/poc_concept_validation_batched.py
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.poc_concept_validation import (
    METADATA_DIR,
    collect_all_terms,
    load_book_list,
)

# Configure logging - suppress httpx noise for clean progress output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 250
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "concept_validation"
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")

# Models to use - 3-way Inter-AI Orchestration
MODELS = [
    {"id": "gpt", "provider": "openai", "model": "gpt-5.2-2025-12-11"},
    {"id": "claude", "provider": "anthropic", "model": "claude-opus-4-5-20251101"},
    {"id": "deepseek", "provider": "deepseek", "model": "deepseek-reasoner"},
]

# =============================================================================
# LLM Client
# =============================================================================


async def call_llm(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4000,
) -> tuple[str, str | None]:
    """Call LLM via gateway.
    
    Returns:
        Tuple of (response_content, error_message or None)
    """
    try:
        response = await client.post(
            f"{LLM_GATEWAY_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more consistent categorization
            },
            timeout=120.0,
        )
        
        if response.status_code != 200:
            # Extract short error reason
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", "")[:50]
            except Exception:
                err_msg = f"HTTP {response.status_code}"
            return "", err_msg
        
        data = response.json()
        return data["choices"][0]["message"]["content"], None
        
    except Exception as e:
        return "", str(e)[:50]


# =============================================================================
# Batch Processing
# =============================================================================

SYSTEM_PROMPT = """You are an expert at identifying software engineering CONCEPTS vs mere KEYWORDS.

CONCEPT = An abstract or generic idea generalized from particular instances.
- Represents a meaningful architectural, design, or methodological idea
- Examples: "microservice architecture", "event sourcing", "test-driven development", "dependency injection"

KEYWORD = A frequently occurring word without deep conceptual meaning.
- Generic terms, noise, identifiers, or overly specific technical details
- Examples: "code", "test", "file", "run", "build", "config", "data"

Your task: For each term provided, output ONLY:
- CONCEPT: term_name
- KEYWORD: term_name

One term per line. No explanations. Just categorize each term."""


async def process_batch(
    client: httpx.AsyncClient,
    batch_num: int,
    terms: list[str],
    model_config: dict,
) -> dict[str, list[str]]:
    """Process a batch of terms through one LLM.
    
    Returns dict with 'concepts' and 'keywords' lists.
    """
    model_id = model_config["id"]
    model_name = model_config["model"]
    
    # Format terms for the prompt
    terms_text = "\n".join(f"- {term}" for term in terms)
    user_prompt = f"""Categorize each of these {len(terms)} terms as CONCEPT or KEYWORD:

{terms_text}

Output format (one per line):
CONCEPT: term_name
KEYWORD: term_name"""

    print(f"  → {model_id}: calling {model_name}...", end="", flush=True)
    
    response, error = await call_llm(client, model_name, SYSTEM_PROMPT, user_prompt)
    
    if error:
        print(f" ⚠️  SKIPPED ({error})", flush=True)
        return {"concepts": [], "keywords": [], "skipped": True}
    
    if not response:
        print(" ⚠️  SKIPPED (empty response)", flush=True)
        return {"concepts": [], "keywords": [], "skipped": True}
    
    # Parse response
    concepts = []
    keywords = []
    
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("CONCEPT:"):
            term = line.replace("CONCEPT:", "").strip()
            if term:
                concepts.append(term)
        elif line.startswith("KEYWORD:"):
            term = line.replace("KEYWORD:", "").strip()
            if term:
                keywords.append(term)
    
    print(f" ✓ {len(concepts)} concepts, {len(keywords)} keywords", flush=True)
    
    return {"concepts": concepts, "keywords": keywords, "skipped": False}


async def process_all_batches(all_terms: list[str]) -> dict[str, Any]:
    """Process all terms through all LLMs in batches."""
    
    # Split into batches
    batches = [
        all_terms[i:i + BATCH_SIZE]
        for i in range(0, len(all_terms), BATCH_SIZE)
    ]
    
    print(f"\n=== BATCHED CONCEPT VALIDATION ===")
    print(f"Total terms: {len(all_terms)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(batches)}")
    print(f"Models: {[m['id'] for m in MODELS]}")
    print()
    
    # Track all results
    all_results = {
        model["id"]: {"concepts": set(), "keywords": set()}
        for model in MODELS
    }
    
    async with httpx.AsyncClient() as client:
        for batch_num, batch_terms in enumerate(batches, 1):
            print(f"\n━━━ Batch {batch_num}/{len(batches)} ({len(batch_terms)} terms) ━━━", flush=True)
            
            # Process batch through each model
            for model_config in MODELS:
                result = await process_batch(client, batch_num, batch_terms, model_config)
                
                all_results[model_config["id"]]["concepts"].update(result["concepts"])
                all_results[model_config["id"]]["keywords"].update(result["keywords"])
            
            # Progress update
            processed = min(batch_num * BATCH_SIZE, len(all_terms))
            pct = 100 * processed / len(all_terms)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\n  [{bar}] {pct:.1f}% ({processed}/{len(all_terms)} terms)", flush=True)
    
    return all_results


def compute_consensus(results: dict[str, Any], threshold: float = 0.67, min_models: int = 2) -> dict[str, Any]:
    """Compute consensus across models, adapting to failures.
    
    A term is a CONCEPT if >= threshold of ACTIVE models agree.
    Only counts models that actually returned data.
    
    Args:
        results: Dict of model_id -> {concepts: set, keywords: set}
        threshold: Fraction of active models that must agree (default 67%)
        min_models: Minimum number of active models required (default 2)
    
    Returns:
        Dict with concepts, keywords, and metadata about the consensus
    """
    # Determine which models actually returned data
    active_models = []
    inactive_models = []
    
    for model_id, model_results in results.items():
        total_classified = len(model_results["concepts"]) + len(model_results["keywords"])
        if total_classified > 0:
            active_models.append(model_id)
        else:
            inactive_models.append(model_id)
    
    num_active = len(active_models)
    
    print(f"\n=== CONSENSUS CALCULATION ===", flush=True)
    print(f"Active models: {active_models} ({num_active})", flush=True)
    if inactive_models:
        print(f"⚠️  Inactive models (failed/skipped): {inactive_models}", flush=True)
    
    # Check minimum models requirement
    if num_active < min_models:
        print(f"❌ ERROR: Only {num_active} models responded, need at least {min_models}", flush=True)
        return {
            "concepts": [],
            "keywords": [],
            "error": f"Insufficient models: {num_active} < {min_models}",
            "active_models": active_models,
            "inactive_models": inactive_models,
        }
    
    # Recalculate threshold based on active models
    votes_needed = max(1, int(num_active * threshold + 0.5))  # Round up
    print(f"Threshold: {threshold:.0%} of {num_active} = {votes_needed} votes needed", flush=True)
    
    # Collect all terms from active models only
    all_terms = set()
    for model_id in active_models:
        all_terms.update(results[model_id]["concepts"])
        all_terms.update(results[model_id]["keywords"])
    
    concepts = []
    keywords = []
    
    for term in all_terms:
        concept_votes = sum(
            1 for model_id in active_models
            if term in results[model_id]["concepts"]
        )
        
        if concept_votes >= votes_needed:
            concepts.append(term)
        else:
            keywords.append(term)
    
    print(f"Results: {len(concepts)} concepts, {len(keywords)} keywords", flush=True)
    
    return {
        "concepts": sorted(concepts),
        "keywords": sorted(keywords),
        "active_models": active_models,
        "inactive_models": inactive_models,
        "votes_needed": votes_needed,
    }


# =============================================================================
# Main
# =============================================================================


async def main():
    """Main entry point."""
    start_time = datetime.now()
    
    print("=" * 60)
    print("POC: BATCHED INTER-AI CONCEPT VALIDATION")
    print("=" * 60)
    
    # Step 1: Load all terms
    print("\n[1/4] Loading all terms from metadata...")
    all_terms = collect_all_terms(METADATA_DIR)
    print(f"      Loaded {len(all_terms)} unique terms")
    
    # Step 2: Process in batches
    print("\n[2/4] Processing terms through LLMs...")
    results = await process_all_batches(all_terms)
    
    # Step 3: Compute consensus
    print("\n[3/4] Computing consensus...")
    consensus = compute_consensus(results, threshold=0.67, min_models=2)
    
    # Check for errors
    if "error" in consensus:
        print(f"\n❌ CONSENSUS FAILED: {consensus['error']}")
        print("Saving partial results anyway...")
    else:
        print(f"\n=== CONSENSUS RESULTS ===")
        print(f"Active models: {consensus.get('active_models', [])}")
        print(f"Votes needed: {consensus.get('votes_needed', 'N/A')}")
        print(f"Total CONCEPTS identified: {len(consensus['concepts'])}")
        print(f"Total KEYWORDS identified: {len(consensus['keywords'])}")
    
    # Step 4: Save results
    print("\n[4/4] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    output_file = OUTPUT_DIR / f"concept_validation_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_terms": len(all_terms),
            "batch_size": BATCH_SIZE,
            "models": [m["model"] for m in MODELS],
            "consensus_threshold": 0.67,
            "results": {
                model_id: {
                    "concepts": sorted(list(data["concepts"])),
                    "keywords": sorted(list(data["keywords"])),
                }
                for model_id, data in results.items()
            },
            "consensus": consensus,
        }, f, indent=2)
    
    print(f"      Full results saved to: {output_file}")
    
    # Save just the concepts list
    concepts_file = OUTPUT_DIR / f"validated_concepts_{timestamp}.json"
    with open(concepts_file, "w") as f:
        json.dump({
            "concepts": consensus["concepts"],
            "count": len(consensus["concepts"]),
            "generated": timestamp,
        }, f, indent=2)
    
    print(f"      Concepts list saved to: {concepts_file}")
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Active models: {consensus.get('active_models', [])}")
    if consensus.get('inactive_models'):
        print(f"⚠️  Failed models: {consensus.get('inactive_models', [])}")
    print(f"Concepts identified: {len(consensus['concepts'])}")
    
    if consensus['concepts']:
        print(f"\nSample concepts (first 20):")
        for concept in consensus["concepts"][:20]:
            print(f"  - {concept}")
    elif "error" not in consensus:
        print("\n⚠️  No concepts met consensus threshold. Check individual model outputs.")
    
    return consensus


if __name__ == "__main__":
    asyncio.run(main())
