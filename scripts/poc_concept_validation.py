#!/usr/bin/env python3
"""
POC: Inter-AI Concept Validation Conversation

This script runs the proof-of-concept for Inter-AI Conversation Orchestration.

Task: Have LLMs (Qwen, GPT-5.2, Claude Opus 4) collaborate to validate extracted 
concepts from 201 technical books. LLMs are informed about available tools 
(BERTopic, SBERT) and decide when/how to use them.

Architecture:
- Orchestrator presents: task, data, available tools
- LLMs drive the discussion and request tool calls when needed
- Tools execute on demand and return results to the conversation
- LLMs analyze results and continue discussion until consensus

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
    
    Filters out noise like hex values, version numbers, timestamps, etc.
    
    Returns:
        List of all unique, cleaned terms.
    """
    import re
    
    all_terms = set()
    
    # Patterns to filter out (noise)
    noise_patterns = [
        r'^[0-9a-f]{2,}$',           # Hex strings like "0f", "4d78", etc.
        r'^[0-9]+$',                  # Pure numbers
        r'^\d+[\s_.-]\d+',            # Version numbers like "10.12", "2024-12"
        r'^[0-9]{4}[-/]\d{2}',        # Dates like "2024-12", "2024/12"
        r'^\d+:\d+',                  # Times like "12:30"
        r'^0x',                       # Hex prefixed
        r'^__',                       # Python dunder methods
        r'^\d+\s+(cid|ps|kb|mb)',     # Numeric with units
        r'^[a-f0-9]{8,}$',            # Long hex strings
        r'localhost|127\.\d+',        # IP addresses
        r'^\d+\.\d+\.\d+',            # IP-like patterns
        r'^https?://',                # URLs
    ]
    
    # Compile patterns for efficiency
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in noise_patterns]
    
    def is_valid_term(term: str) -> bool:
        """Check if term is a valid concept candidate."""
        # Too short
        if len(term) < 3:
            return False
        # Pure punctuation or whitespace
        if not any(c.isalpha() for c in term):
            return False
        # Matches noise pattern
        for pattern in compiled_patterns:
            if pattern.search(term):
                return False
        return True
    
    for file_path in sorted(metadata_dir.glob("*.json")):
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                continue
                
            for chapter in data:
                keywords = chapter.get("keywords", [])
                for kw in keywords:
                    if is_valid_term(kw):
                        all_terms.add(kw)
                
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    # Sort by length (longer terms more likely to be concepts)
    return sorted(list(all_terms), key=lambda x: (-len(x), x))


# =============================================================================
# Participants
# =============================================================================


def create_participants() -> list[Participant]:
    """Create the conversation participants.
    
    LLMs are the primary participants who drive the discussion.
    Tools are available for LLMs to request when needed.
    """
    
    # Tool descriptions for LLM system prompts
    tools_description = """
AVAILABLE TOOLS (request by saying "I'd like to use [tool_name] to..."):

1. BERTopic Clustering Engine
   - Purpose: Cluster terms into semantic groups using transformer embeddings
   - Use when: You want to see how terms naturally group together
   - Request format: "Use BERTopic to cluster these terms: [list of terms]"

2. SBERT Semantic Analyzer  
   - Purpose: Compute semantic similarity between terms/concepts
   - Use when: You want to validate if terms belong together, or compare concepts
   - Request format: "Use SBERT to compare similarity of: [term1] vs [term2]"

When you request a tool, the orchestrator will execute it and share results with all participants.
"""

    base_task_context = """
You are participating in a collaborative discussion with other AI models to analyze 
concepts extracted from 201 technical books. Your goal is to determine which extracted 
terms represent true CONCEPTS (abstract ideas) vs mere KEYWORDS (common words).

CONCEPT = An abstract or generic idea generalized from particular instances
  Examples: "microservice architecture", "test-driven development", "event sourcing"

KEYWORD = A frequently occurring word without deep semantic meaning
  Examples: "code", "test", "file", "run", "build"

The discussion should result in a validated concept vocabulary.
"""

    return [
        # LLMs - Primary participants who drive the discussion
        Participant(
            id="qwen",
            name="Qwen3 Coder",
            participant_type=ParticipantType.LLM,
            provider="openrouter",
            model="qwen/qwen3-coder",
            system_prompt=(
                base_task_context + 
                "\nYou are Qwen3-Coder, specialized in software engineering concepts. "
                "Analyze the data from a technical/code perspective. Identify patterns "
                "in how terms relate to software development practices.\n" +
                tools_description
            ),
            capabilities=["code_analysis", "concept_reasoning", "tool_requester"],
        ),
        Participant(
            id="gpt",
            name="GPT-5.2",
            participant_type=ParticipantType.LLM,
            provider="openai",
            model="gpt-5.2-thinking",
            system_prompt=(
                base_task_context +
                "\nYou are GPT-5.2, an advanced reasoning AI. Your role is to validate "
                "and challenge proposals from other participants. Be critical but constructive. "
                "If you disagree, explain why and propose alternatives.\n" +
                tools_description
            ),
            capabilities=["reasoning", "validation", "refinement", "tool_requester"],
        ),
        Participant(
            id="claude",
            name="Claude Opus 4.5",
            participant_type=ParticipantType.LLM,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
            system_prompt=(
                base_task_context +
                "\nYou are Claude Opus 4, focused on synthesis and clarity. Help integrate "
                "insights from other participants. Identify when the group is converging on "
                "consensus and summarize agreed-upon concepts.\n" +
                tools_description
            ),
            capabilities=["synthesis", "analysis", "summarization", "tool_requester"],
        ),
        
        # Tools - Available on request from LLMs
        Participant(
            id="bertopic",
            name="BERTopic Clustering Engine",
            participant_type=ParticipantType.TOOL,
            endpoint=f"{CODE_ORCHESTRATOR_URL}/api/v1/nlp/concepts/discover",
            capabilities=["topic_clustering", "term_grouping"],
        ),
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
    print(f"      Found {len(all_terms)} unique terms (after filtering noise)")
    
    # Step 2: Prepare context - organize data for LLMs to analyze
    print("\n[2/5] Preparing conversation context...")
    
    # Organize concepts by book for LLMs to see the full picture
    # Limit to representative sample for context window
    sample_books = book_list[:30]  # 30 representative books
    sample_concepts_by_book = {}
    for book in sample_books:
        title = book["title"]
        if title in concepts_by_book:
            sample_concepts_by_book[title] = concepts_by_book[title][:20]  # Top 20 per book
    
    # Get top terms across all books (most frequent/important)
    top_terms = all_terms[:200]  # Top 200 terms for analysis
    
    context = {
        "total_books": len(book_list),
        "total_unique_terms": len(all_terms),
        "sample_book_titles": [b["title"] for b in sample_books],
        "concepts_by_book": sample_concepts_by_book,
        "top_terms_for_analysis": top_terms,
    }
    
    print(f"      {len(sample_books)} sample books with concepts")
    print(f"      {len(top_terms)} top terms for analysis")
    
    # Step 3: Create orchestrator and participants
    print("\n[3/5] Initializing orchestrator...")
    
    orchestrator = ConversationOrchestrator(
        llm_gateway_url=LLM_GATEWAY_URL,
        tool_service_url=CODE_ORCHESTRATOR_URL,
        transcript_dir=OUTPUT_DIR,
    )
    
    participants = create_participants()
    llm_count = len([p for p in participants if p.participant_type == ParticipantType.LLM])
    tool_count = len([p for p in participants if p.participant_type == ParticipantType.TOOL])
    print(f"      Created {llm_count} LLM participants, {tool_count} tools available")
    
    # Step 4: Define turn order - LLMs only, they request tools when needed
    # Each round: Qwen → GPT → Claude (all must participate)
    turn_order = [
        "qwen", "gpt", "claude",  # Round 1: Initial analysis
        "qwen", "gpt", "claude",  # Round 2: Deeper discussion  
        "qwen", "gpt", "claude",  # Round 3: Consensus building
    ]
    
    # Step 5: Start and run conversation
    print("\n[4/5] Starting conversation...")
    
    task = """
Analyze the extracted terms from 201 technical books and collaborate to build a 
validated CONCEPT VOCABULARY.

YOUR DATA:
- {total_books} books analyzed
- {total_terms} unique terms extracted
- Sample books and their concepts provided in context
- Top terms for analysis provided in context

YOUR TASK:
1. Review the extracted terms and concepts organized by book
2. Discuss which terms represent true CONCEPTS (abstract ideas worth including in vocabulary)
3. Challenge each other's proposals - debate and refine
4. You may request tool assistance (BERTopic for clustering, SBERT for similarity)
5. Build consensus on a final concept list

REMEMBER:
- CONCEPT = Abstract idea generalized from instances (e.g., "microservice architecture")
- KEYWORD = Common word without deep meaning (e.g., "code", "test", "file")
- ALL of you must participate before consensus can be declared
- Challenge weak proposals - this should be a substantive discussion

OUTPUT: A final list of validated concepts with brief definitions.
""".format(total_books=context["total_books"], total_terms=context["total_unique_terms"])
    
    conversation = await orchestrator.start_conversation(
        task=task,
        participants=participants,
        context=context,
        turn_order=turn_order,
        max_rounds=5,
        min_rounds=2,  # IMPORTANT: Prevents premature consensus (requires at least 2 full rounds)
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
    
    # Close orchestrator clients
    await orchestrator.close()
    
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
