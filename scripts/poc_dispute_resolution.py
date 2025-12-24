#!/usr/bin/env python3
"""
Dispute Resolution POC - Re-evaluate terms where GPT and Claude disagreed.

This script:
1. Loads the 1,603 disputed terms (terms classified by one model but not the other)
2. Fetches book context evidence for each term from the corpus
3. Has GPT and Claude re-debate each term WITH the evidence
4. Aggregates newly agreed terms with the original consensus

This is the final step in the Inter-AI Orchestration POC:
- Step 1: Batched validation (completed) → 2,688 consensus terms
- Step 2: Dispute resolution (this script) → re-evaluate 1,603 disputed terms
- Step 3: Final output → merged validated concepts

Usage:
    python scripts/poc_dispute_resolution.py
    python scripts/poc_dispute_resolution.py --dry-run  # Test with 10 terms
"""

import asyncio
import json
import logging
import os
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
# Configuration
# =============================================================================

LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
DATA_DIR = Path(__file__).parent.parent / "data" / "concept_validation"


def find_latest_results_file() -> Path:
    """Find the most recent concept validation results file."""
    pattern = "concept_validation_*.json"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files found matching {pattern} in {DATA_DIR}")
    
    # Sort by modification time, most recent first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]


def get_models_from_results(results_file: Path) -> list[dict]:
    """Extract the models that participated in the original validation run.
    
    This avoids hardcoding model names - we simply use whatever models
    had results in the previous run.
    """
    with open(results_file) as f:
        data = json.load(f)
    
    # The batched script saves full model IDs in the "models" field
    full_model_ids = data.get("models", [])
    
    # Map short names to full IDs
    short_to_full = {}
    for full_id in full_model_ids:
        # Extract short name (e.g., "gpt-5.2-2025-12-11" -> "gpt")
        short_name = full_id.split("-")[0] if "-" in full_id else full_id.split("/")[-1]
        short_to_full[short_name] = full_id
    
    models = []
    for model_id, model_data in data["results"].items():
        # Only include models that actually returned results
        if model_data.get("concepts") or model_data.get("keywords"):
            # Look up the full model ID
            full_id = short_to_full.get(model_id, model_id)
            models.append({
                "name": model_id,      # Short name (e.g., "gpt")
                "model_id": full_id,   # Full ID (e.g., "gpt-5.2-2025-12-11")
            })
    
    return models


async def discover_models_from_gateway() -> list[dict]:
    """Dynamically discover available models from the LLM gateway.
    
    This provides real-time model discovery, independent of previous results.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{LLM_GATEWAY_URL}/v1/models", timeout=10.0)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            models = []
            
            for model in data.get("data", []):
                model_id = model.get("id", "")
                provider = model.get("owned_by", "unknown")
                short_name = model_id.split("-")[0] if "-" in model_id else model_id.split("/")[-1]
                models.append({
                    "name": short_name,
                    "model_id": model_id,
                    "provider": provider,
                })
            
            return models
        except Exception as e:
            print(f"⚠️  Could not discover models from gateway: {e}")
            return []

# =============================================================================
# Dispute Resolution Logic
# =============================================================================


def load_disputed_terms(results_file: Path) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    """Load terms where models disagreed.
    
    Dynamically identifies which models had results and finds disagreements.
    
    Returns:
        - exclusive_to_first: Terms first model classified but second didn't
        - exclusive_to_second: Terms second model classified but first didn't
        - original_data: Full original results
        - model_ids: List of model IDs that had results
    """
    with open(results_file) as f:
        data = json.load(f)
    
    # Dynamically find models that had results (non-zero concepts)
    active_models = []
    model_concepts = {}
    
    for model_id, model_data in data["results"].items():
        concepts = set(c.lower() for c in model_data.get("concepts", []))
        if concepts:  # Only include models that returned concepts
            active_models.append(model_id)
            model_concepts[model_id] = concepts
    
    if len(active_models) < 2:
        raise ValueError(f"Need at least 2 models with results, found: {active_models}")
    
    # Find disagreements between the first two active models
    model1, model2 = active_models[0], active_models[1]
    concepts1 = model_concepts[model1]
    concepts2 = model_concepts[model2]
    
    exclusive_to_first = list(concepts1 - concepts2)
    exclusive_to_second = list(concepts2 - concepts1)
    
    return exclusive_to_first, exclusive_to_second, data, active_models


def build_dispute_prompt(term: str, context: TermContext, source: str) -> str:
    """Build prompt for dispute resolution with context evidence.
    
    Args:
        term: The disputed term
        context: Book context evidence for the term
        source: Which model originally classified it ("gpt" or "claude")
    
    Returns:
        Prompt string for the LLM
    """
    # Build evidence section
    evidence_lines = []
    for occ in context.occurrences[:5]:  # Top 5 pieces of evidence
        evidence_lines.append(f"  - [{occ.source_type}] {occ.book_title}, Ch.{occ.chapter_number}")
        evidence_lines.append(f"    \"{occ.context_snippet[:200]}...\"")
    
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "  No direct evidence found in corpus."
    
    # Statistics
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


async def evaluate_dispute(
    client: httpx.AsyncClient,
    term: str,
    context: TermContext,
    source: str,
    model: dict,
) -> tuple[str, str | None]:
    """Have a single model re-evaluate a disputed term with evidence.
    
    Returns:
        Tuple of (classification, error_message)
    """
    prompt = build_dispute_prompt(term, context, source)
    
    try:
        response = await client.post(
            f"{LLM_GATEWAY_URL}/v1/chat/completions",
            json={
                "model": model["model_id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1,  # Low temp for consistent classification
            },
            timeout=30.0,
        )
        
        if response.status_code != 200:
            return "", f"HTTP {response.status_code}"
        
        result = response.json()
        # Parse OpenAI-style response format
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        
        # Parse response
        if "CONCEPT" in content:
            return "CONCEPT", None
        elif "KEYWORD" in content:
            return "KEYWORD", None
        elif "REJECT" in content:
            return "REJECT", None
        else:
            return "UNKNOWN", f"Unparseable: {content[:50]}"
            
    except Exception as e:
        return "", str(e)


async def resolve_disputes(
    model1_only: list[str],
    model2_only: list[str],
    models: list[dict],
    model_ids: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run dispute resolution for all disagreed terms.
    
    Args:
        model1_only: Terms first model classified but second didn't
        model2_only: Terms second model classified but first didn't
        models: Available models from gateway
        model_ids: Model IDs that had results in original run
        dry_run: If True, only process 10 terms
    
    Returns:
        Results dictionary with resolution outcomes
    """
    # Initialize context lookup
    print("Initializing book context lookup...")
    lookup = BookContextLookup()
    
    # Prepare disputed terms with source tracking
    disputes = []
    for term in model1_only:
        disputes.append({"term": term, "source": model_ids[0]})
    for term in model2_only:
        disputes.append({"term": term, "source": model_ids[1]})
    
    if dry_run:
        disputes = disputes[:10]
        print(f"DRY RUN: Processing only {len(disputes)} terms")
    
    print(f"Processing {len(disputes)} disputed terms...")
    print(f"Using models: {[m['model_id'] for m in models]}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_disputes": len(disputes),
        "models_used": [m["model_id"] for m in models],
        "newly_agreed_concepts": [],  # Both models now agree it's a concept
        "newly_agreed_keywords": [],  # Both models now agree it's a keyword
        "still_disputed": [],         # Models still disagree
        "both_rejected": [],          # Both models now reject
        "errors": [],
        "detailed_results": {},
    }
    
    async with httpx.AsyncClient() as client:
        # Process in batches to avoid overwhelming the gateway
        batch_size = 5
        
        for i in tqdm(range(0, len(disputes), batch_size), desc="Resolving disputes"):
            batch = disputes[i:i + batch_size]
            
            for dispute in batch:
                term = dispute["term"]
                source = dispute["source"]
                
                # Get book context
                context = lookup.find_term_context(term, max_results=5)
                
                # Log evidence found (verbose mode)
                if context.books_found_in > 0:
                    evidence_summary = []
                    if context.appears_in_concept_lists:
                        evidence_summary.append("concepts")
                    if context.appears_in_keyword_lists:
                        evidence_summary.append("keywords")
                    if context.appears_in_summaries:
                        evidence_summary.append("summaries")
                    if context.appears_in_raw_text:
                        evidence_summary.append("raw_text")
                    tqdm.write(f"  📚 '{term}': {context.books_found_in} books, {context.chapters_found_in} chapters [{', '.join(evidence_summary)}]")
                else:
                    tqdm.write(f"  ⚠️  '{term}': NO evidence found in corpus")
                
                # Evaluate with all available models
                model_votes = {}
                
                for model in models:
                    classification, error = await evaluate_dispute(
                        client, term, context, source, model
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
                
                # Aggregate votes
                results["detailed_results"][term] = {
                    "original_source": source,
                    "books_found_in": context.books_found_in,
                    "chapters_found_in": context.chapters_found_in,
                    "model_votes": model_votes,
                }
                
                # Determine outcome dynamically based on all model votes
                votes = [v for v in model_votes.values() if not str(v).startswith("ERROR")]
                
                # Log the votes
                vote_str = " vs ".join([f"{m.split('-')[0]}={v}" for m, v in model_votes.items()])
                
                if len(votes) >= 2:  # Need at least 2 valid votes
                    if all(v == "CONCEPT" for v in votes):
                        results["newly_agreed_concepts"].append(term)
                        tqdm.write(f"    ✅ CONCEPT: {term} ({vote_str})")
                    elif all(v == "KEYWORD" for v in votes):
                        results["newly_agreed_keywords"].append(term)
                        tqdm.write(f"    📝 KEYWORD: {term} ({vote_str})")
                    elif all(v == "REJECT" for v in votes):
                        results["both_rejected"].append(term)
                        tqdm.write(f"    ❌ REJECTED: {term} ({vote_str})")
                    else:
                        results["still_disputed"].append({
                            "term": term,
                            "votes": model_votes,
                            "evidence_found": context.books_found_in > 0,
                        })
                        tqdm.write(f"    ⚡ DISPUTED: {term} ({vote_str})")
            
            # Small delay between batches
            await asyncio.sleep(0.5)
    
    return results


def merge_results(
    original_file: Path,
    dispute_results: dict[str, Any],
) -> dict[str, Any]:
    """Merge original consensus with newly resolved disputes.
    
    Dynamically finds models and computes consensus.
    
    Returns:
        Final merged results with all validated concepts.
    """
    # Load original consensus
    with open(original_file) as f:
        original = json.load(f)
    
    # Dynamically find active models and their concepts
    model_concepts = {}
    for model_id, model_data in original["results"].items():
        concepts = set(c.lower() for c in model_data.get("concepts", []))
        if concepts:  # Only include models that had results
            model_concepts[model_id] = concepts
    
    # Compute original consensus (intersection of all active models)
    if model_concepts:
        original_consensus = set.intersection(*model_concepts.values())
    else:
        original_consensus = set()
    
    # Add newly agreed concepts
    newly_agreed = set(c.lower() for c in dispute_results["newly_agreed_concepts"])
    
    final_concepts = original_consensus | newly_agreed
    
    return {
        "timestamp": datetime.now().isoformat(),
        "source": "Inter-AI Orchestration POC - Final Results",
        "summary": {
            "original_consensus_concepts": len(original_consensus),
            "newly_agreed_concepts": len(newly_agreed),
            "final_validated_concepts": len(final_concepts),
            "keywords_identified": len(dispute_results.get("newly_agreed_keywords", [])),
            "rejected_terms": len(dispute_results.get("both_rejected", [])),
            "still_disputed": len(dispute_results.get("still_disputed", [])),
        },
        "validated_concepts": sorted(final_concepts),
        "keywords": sorted(dispute_results.get("newly_agreed_keywords", [])),
        "dispute_resolution_details": dispute_results,
    }


# =============================================================================
# Main
# =============================================================================


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dispute Resolution POC")
    parser.add_argument("--dry-run", action="store_true", help="Test with only 10 terms")
    args = parser.parse_args()
    
    print("=" * 70)
    print("INTER-AI ORCHESTRATION POC - DISPUTE RESOLUTION")
    print("=" * 70)
    
    # Check gateway connectivity
    print("\nChecking LLM Gateway connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LLM_GATEWAY_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                print(f"✅ LLM Gateway reachable at {LLM_GATEWAY_URL}")
            else:
                print(f"⚠️  Gateway returned status {resp.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach LLM Gateway: {e}")
        print("Please start the gateway first: docker compose up -d")
        return
    
    # Find latest results file
    print("\nFinding latest results file...")
    try:
        results_file = find_latest_results_file()
        print(f"✅ Using: {results_file.name}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get models from the results (the ones that actually participated)
    available_models = get_models_from_results(results_file)
    print(f"✅ Found {len(available_models)} models from results: {[m['model_id'] for m in available_models]}")
    
    if len(available_models) < 2:
        print("❌ Need at least 2 models for dispute resolution")
        return
    
    # Load disputed terms
    print("\nLoading disputed terms...")
    model1_only, model2_only, original_data, model_ids = load_disputed_terms(results_file)
    
    print(f"  Active models in results: {model_ids}")
    print(f"  {model_ids[0]}-only concepts: {len(model1_only)}")
    print(f"  {model_ids[1]}-only concepts: {len(model2_only)}")
    print(f"  Total disputes to resolve: {len(model1_only) + len(model2_only)}")
    
    # Show some examples
    print("\nSample disputed terms:")
    print(f"  {model_ids[0]}-only: {model1_only[:5]}")
    print(f"  {model_ids[1]}-only: {model2_only[:5]}")
    
    # Run dispute resolution
    print("\n" + "-" * 70)
    print("RUNNING DISPUTE RESOLUTION WITH BOOK EVIDENCE")
    print("-" * 70)
    
    dispute_results = await resolve_disputes(
        model1_only, 
        model2_only,
        available_models,
        model_ids,
        dry_run=args.dry_run,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("DISPUTE RESOLUTION SUMMARY")
    print("=" * 70)
    print(f"  Newly agreed as CONCEPT: {len(dispute_results['newly_agreed_concepts'])}")
    print(f"  Newly agreed as KEYWORD: {len(dispute_results['newly_agreed_keywords'])}")
    print(f"  Both rejected: {len(dispute_results['both_rejected'])}")
    print(f"  Still disputed: {len(dispute_results['still_disputed'])}")
    print(f"  Errors: {len(dispute_results['errors'])}")
    
    if dispute_results['newly_agreed_concepts']:
        print(f"\nSample newly agreed concepts: {dispute_results['newly_agreed_concepts'][:10]}")
    
    # Merge and save final results
    if not args.dry_run:
        print("\nMerging with original consensus...")
        final_results = merge_results(results_file, dispute_results)
        
        # Save dispute results
        dispute_file = DATA_DIR / f"dispute_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dispute_file, "w") as f:
            json.dump(dispute_results, f, indent=2)
        print(f"Saved dispute results to: {dispute_file}")
        
        # Save final merged results
        final_file = DATA_DIR / "final_validated_concepts.json"
        with open(final_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"Saved final results to: {final_file}")
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Original consensus concepts: {final_results['summary']['original_consensus_concepts']}")
        print(f"  + Newly agreed concepts: {final_results['summary']['newly_agreed_concepts']}")
        print(f"  = Total validated concepts: {final_results['summary']['final_validated_concepts']}")
    else:
        print("\nDry run complete - no files saved")
        print(f"\nDetailed results:")
        for term, details in dispute_results["detailed_results"].items():
            print(f"  {term}: {details['model_votes']} (found in {details['books_found_in']} books)")


if __name__ == "__main__":
    asyncio.run(main())
