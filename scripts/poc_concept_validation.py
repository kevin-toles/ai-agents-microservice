#!/usr/bin/env python3
"""
POC: Inter-AI Concept Validation Conversation

This script runs the proof-of-concept for Inter-AI Conversation Orchestration.

Task: Have LLMs (Qwen, GPT-5.2, Claude Opus 4) and BERT tools (BERTopic, SBERT) collaborate
to validate extracted concepts from 201 technical books.

Flow:
1. Load book list and extracted concepts from metadata
2. Run BERTopic clustering on terms
3. Start conversation between participants
4. LLMs discuss and validate concept labels
5. Save validated concept vocabulary

Reference: docs/INTER_AI_ORCHESTRATION.md

Usage:
    python scripts/poc_concept_validation.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path (parent of src)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationStatus,
    Participant,
    ParticipantType,
)
from src.conversation.orchestrator import ConversationOrchestrator
from src.participants.llm_participant import LLMParticipantAdapter
from src.participants.tool_participant import ToolParticipantAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "conversations"
CONCEPT_OUTPUT = Path(__file__).parent.parent / "config" / "validated_concepts.json"

# Service URLs
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8080")
CODE_ORCHESTRATOR_URL = os.getenv("CODE_ORCHESTRATOR_URL", "http://localhost:8083")


# =============================================================================
# Data Preparation
# =============================================================================


def load_book_list(metadata_dir: Path) -> list[dict]:
    """Load book list from metadata files.
    
    The metadata files are lists of chapter objects with keywords/concepts.
    
    Returns:
        List of {title, keywords_count, file} dicts.
    """
    books = []
    
    for file_path in sorted(metadata_dir.glob("*.json")):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Data is a list of chapters
            if isinstance(data, list):
                # Extract title from filename
                title = file_path.stem.replace("_metadata", "")
                
                # Count total keywords across chapters
                keywords_count = 0
                for chapter in data:
                    keywords_count += len(chapter.get("keywords", []))
                
                books.append({
                    "title": title,
                    "keywords_count": keywords_count,
                    "file": file_path.name,
                })
            else:
                logger.warning(f"Unexpected format in {file_path}: not a list")
                
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    return books


def load_concepts_by_book(metadata_dir: Path) -> dict[str, list[str]]:
    """Load keywords organized by book.
    
    Returns:
        Dict mapping book title to list of keywords.
    """
    keywords_by_book = {}
    
    for file_path in sorted(metadata_dir.glob("*.json")):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
                
            title = file_path.stem.replace("_metadata", "")
            
            # Collect keywords from all chapters
            keywords = set()
            for chapter in data:
                chapter_keywords = chapter.get("keywords", [])
                keywords.update(chapter_keywords)
            
            keywords_by_book[title] = sorted(list(keywords))
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    return keywords_by_book


def collect_all_terms(metadata_dir: Path) -> list[str]:
    """Collect all terms (keywords) from metadata.
    
    Returns:
        List of all unique terms.
    """
    all_terms = set()
    
    for file_path in sorted(metadata_dir.glob("*.json")):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
                
            for chapter in data:
                keywords = chapter.get("keywords", [])
                all_terms.update(keywords)
                
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    return sorted(list(all_terms))


# =============================================================================
# Participants
# =============================================================================


def create_participants() -> list[Participant]:
    """Create the conversation participants."""
    return [
        # BERT Tools
        Participant(
            id="bertopic",
            name="BERTopic Clustering Engine",
            participant_type=ParticipantType.TOOL,
            endpoint=f"{CODE_ORCHESTRATOR_URL}/api/v1/nlp/concepts/discover",
            capabilities=["topic_clustering", "term_grouping"],
        ),
        
        # LLMs
        Participant(
            id="qwen",
            name="Qwen3 Coder",
            participant_type=ParticipantType.LLM,
            provider="openrouter",
            model="qwen/qwen3-coder",
            system_prompt=(
                "You are Qwen3-Coder, a technical AI assistant specialized in software engineering. "
                "Your role is to analyze term clusters and identify which represent true CONCEPTS "
                "(abstract ideas like 'microservice architecture', 'test-driven development') vs "
                "mere KEYWORDS (frequent words like 'code', 'test', 'deploy'). "
                "A concept is an abstract or generic idea generalized from particular instances."
            ),
            capabilities=["code_analysis", "concept_reasoning"],
        ),
        Participant(
            id="gpt",
            name="GPT-5.2",
            participant_type=ParticipantType.LLM,
            provider="openai",
            model="gpt-5.2",
            system_prompt=(
                "You are GPT, an advanced reasoning AI. Your role is to validate and refine "
                "concept labels proposed by other participants. Review the term clusters, "
                "assess whether the proposed concept names accurately capture the abstract idea, "
                "and suggest improvements. When you agree with another participant, say so explicitly."
            ),
            capabilities=["reasoning", "validation", "refinement"],
        ),
        Participant(
            id="claude",
            name="Claude Opus 4",
            participant_type=ParticipantType.LLM,
            provider="anthropic",
            model="claude-opus-4-20250514",
            system_prompt=(
                "You are Claude, an AI assistant focused on synthesis and analysis. Your role is to "
                "integrate insights from other participants, identify patterns across their suggestions, "
                "and help synthesize a coherent final concept vocabulary. Focus on clarity and precision "
                "in concept definitions. When the group reaches consensus, summarize the agreed-upon concepts."
            ),
            capabilities=["synthesis", "analysis", "summarization"],
        ),
        
        # Additional tool for semantic validation
        Participant(
            id="sbert",
            name="SBERT Semantic Analyzer",
            participant_type=ParticipantType.TOOL,
            endpoint=f"{CODE_ORCHESTRATOR_URL}/api/v1/nlp/similarity",
            capabilities=["semantic_similarity", "coherence_scoring"],
        ),
    ]


# =============================================================================
# Conversation Runner
# =============================================================================


async def run_concept_validation_conversation():
    """Run the concept validation conversation."""
    
    print("=" * 60)
    print("POC: INTER-AI CONCEPT VALIDATION CONVERSATION")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/5] Loading metadata...")
    
    book_list = load_book_list(METADATA_DIR)
    concepts_by_book = load_concepts_by_book(METADATA_DIR)
    all_terms = collect_all_terms(METADATA_DIR)
    
    print(f"      Loaded {len(book_list)} books")
    print(f"      Found {len(all_terms)} unique terms")
    
    # Step 2: Prepare context
    print("\n[2/5] Preparing conversation context...")
    
    # Sample of books and concepts for context (full list too large for prompt)
    sample_books = book_list[:20]
    sample_concepts = {k: v[:10] for k, v in list(concepts_by_book.items())[:10]}
    
    context = {
        "total_books": len(book_list),
        "total_terms": len(all_terms),
        "sample_books": sample_books,
        "concepts_by_book_sample": sample_concepts,
        "terms": all_terms[:500],  # First 500 terms for BERTopic
    }
    
    print(f"      Context prepared with {len(context['terms'])} terms for clustering")
    
    # Step 3: Create orchestrator and participants
    print("\n[3/5] Initializing orchestrator...")
    
    orchestrator = ConversationOrchestrator(
        llm_gateway_url=LLM_GATEWAY_URL,
        tool_service_url=CODE_ORCHESTRATOR_URL,
        transcript_dir=OUTPUT_DIR,
    )
    
    participants = create_participants()
    print(f"      Created {len(participants)} participants")
    
    # Step 4: Define turn order
    # Tool-first strategy: BERTopic clusters → LLMs discuss → SBERT validates
    turn_order = ["bertopic", "qwen", "gpt", "sbert", "qwen", "gpt"]
    
    # Step 5: Start and run conversation
    print("\n[4/5] Starting conversation...")
    
    task = (
        "Analyze the extracted terms from 201 technical books and collaborate to:\n"
        "1. BERTopic: Cluster the terms into semantic groups\n"
        "2. Qwen: For each cluster, propose a concept name (2-4 words, abstract idea)\n"
        "3. GPT: Validate/refine Qwen's proposals\n"
        "4. SBERT: Compute semantic coherence of the clusters\n"
        "5. Reach consensus on which clusters represent true CONCEPTS vs just KEYWORDS\n\n"
        "Output: A validated list of concept names with their representative terms."
    )
    
    conversation = await orchestrator.start_conversation(
        task=task,
        participants=participants,
        context=context,
        turn_order=turn_order,
        max_rounds=3,  # 3 rounds for POC
        consensus_threshold=0.8,
    )
    
    print(f"      Conversation ID: {conversation.conversation_id}")
    print(f"      Turn order: {turn_order}")
    
    # Run the conversation with live output
    def on_message(msg: ConversationMessage):
        print(f"\n[{msg.participant_id.upper()}] ({msg.latency_ms}ms):")
        print(f"  {msg.content[:500]}...")
    
    print("\n[5/5] Running conversation...\n")
    print("-" * 60)
    
    conversation = await orchestrator.run_conversation(
        conversation,
        on_message=on_message,
    )
    
    print("-" * 60)
    
    # Summary
    print(f"\n=== CONVERSATION COMPLETE ===")
    print(f"Status: {conversation.status.value}")
    print(f"Rounds: {conversation.current_round}/{conversation.max_rounds}")
    print(f"Messages: {len(conversation.messages)}")
    print(f"Transcript saved to: {OUTPUT_DIR / f'{conversation.conversation_id}.txt'}")
    
    # Close clients
    await llm_adapter.close()
    await tool_adapter.close()
    
    return conversation


# =============================================================================
# Main
# =============================================================================


async def main():
    """Main entry point."""
    try:
        conversation = await run_concept_validation_conversation()
        
        # Print final transcript
        print("\n" + "=" * 60)
        print("FINAL TRANSCRIPT")
        print("=" * 60)
        print(conversation.get_transcript())
        
    except Exception as e:
        logger.error(f"Error running POC: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
