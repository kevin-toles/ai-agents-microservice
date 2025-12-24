#!/usr/bin/env python3
"""
================================================================================
FILE: poc_guideline_textbook_crossref.py
--------------------------------------------------------------------------------
INTENT: Inter-AI orchestration script that uses GPT-5.2, Claude Opus 4.5, and 
        DeepSeek Reasoner to cross-reference Guidelines against JSON textbooks
        per the document hierarchy in TIER_RELATIONSHIP_DIAGRAM.md.

SOURCES:
- GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md (Priority 1)
- AI_CODING_PLATFORM_ARCHITECTURE.md (Priority 2)
- ARCHITECTURE.md - llm-gateway (Priority 3)
- AI-ML_taxonomy_20251128.json (Priority 4)
- CODING_PATTERNS_ANALYSIS.md (Priority 5)

DEPENDENCIES:
- httpx (async HTTP client)
- llm-gateway running on localhost:8080
- JSON textbooks in /Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json/

CREATED: 2024-12-24
AUTHOR: AI Agent Orchestration (GPT-5.2 + Claude Opus 4.5 + DeepSeek Reasoner)
================================================================================
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
import httpx

LLM_GATEWAY_URL = 'http://localhost:8080'
TEXTBOOKS_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json')
OUTPUT_DIR = Path('/Users/kevintoles/POC/ai-agents/data/textbook_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {'id': 'gpt', 'model': 'gpt-5.2-2025-12-11'},
    {'id': 'claude', 'model': 'claude-opus-4-5-20251101'},
    {'id': 'deepseek', 'model': 'deepseek-reasoner'},
]

# Target textbooks from taxonomy - Architecture Spine + ML Design books
TARGET_BOOKS = [
    'Machine Learning Design Patterns.json',
    'Designing Machine Learning Systems An Iterative Process for Production-Ready Applications.json',
    'Reliable Machine Learning.json',
    'AntiPatterns.json',
    'Architecture Patterns with Python.json',
    'Clean Architecture - Robert C Martin.json',
    'Design Patterns - Elements of Reusable Object-Oriented Software - Gang of Four.json',
    'microservices-antipatterns-and-pitfalls.json',
]

ANALYSIS_PROMPT = """You are analyzing technical textbook content to find patterns relevant to building a **Hybrid Tiered Classification System** for software concepts.

CONTEXT: We're building a 4-tier classifier following the Kitchen Brigade Architecture:
- Tier 1: Alias/Exact Match Lookup (O(1) dictionary)
- Tier 2: ML Classifier (SBERT embeddings + LogisticRegression)
- Tier 3: Heuristic Rules (noise filtering from noise_terms.yaml)
- Tier 4: LLM Fallback (for edge cases via llm-gateway)

ANTI-PATTERNS TO AVOID (from CODING_PATTERNS_ANALYSIS.md):
- #7: Exception Shadowing (don't shadow builtins like ConnectionError)
- #12: Connection Pooling (use repository pattern with FakeClient for tests)
- #13: Custom Exception Naming (prefix with domain, e.g., Neo4jConnectionError)

TEXTBOOK CONTENT:
{content}

TASK: Extract relevant patterns, principles, and recommendations that apply to:
1. **Tiered/Cascade classification** - fallback patterns, confidence thresholds
2. **ML model selection** - when to use embeddings vs rules vs LLMs
3. **Performance optimization** - caching, lazy loading, batch processing
4. **Error handling** - graceful degradation, fallback strategies
5. **Anti-patterns to avoid** - common mistakes in classification systems

OUTPUT FORMAT (JSON):
{{
  "book_title": "<title>",
  "relevant_patterns": [
    {{
      "pattern_name": "<name>",
      "description": "<2-3 sentences>",
      "applicable_tier": "tier1|tier2|tier3|tier4|all",
      "source_reference": "<chapter/section if available>"
    }}
  ],
  "anti_patterns": [
    {{
      "name": "<name>",
      "description": "<why to avoid>",
      "mitigation": "<how to prevent>"
    }}
  ],
  "key_recommendations": ["<recommendation1>", "<recommendation2>", ...]
}}

Return ONLY valid JSON. No markdown, no explanation."""


async def load_textbook_excerpt(book_name: str, max_chars: int = 12000) -> str:
    """Load excerpt from a textbook JSON."""
    book_path = TEXTBOOKS_DIR / book_name
    if not book_path.exists():
        return None
    
    with open(book_path) as f:
        data = json.load(f)
    
    # Extract text content
    content_parts = []
    if isinstance(data, dict):
        for key in ['content', 'text', 'chapters', 'sections', 'body', 'pages']:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    content_parts.append(val[:max_chars])
                elif isinstance(val, list):
                    for item in val[:15]:
                        if isinstance(item, str):
                            content_parts.append(item[:2000])
                        elif isinstance(item, dict):
                            content_parts.append(json.dumps(item)[:2000])
                break
    
    if not content_parts:
        content_parts.append(json.dumps(data)[:max_chars])
    
    return '\n'.join(content_parts)[:max_chars]


async def query_model(client: httpx.AsyncClient, model_id: str, model_name: str, 
                      prompt: str, book_title: str) -> dict:
    """Query a single model for textbook analysis."""
    print(f'    [{model_id}] Querying...')
    
    try:
        resp = await client.post(
            f'{LLM_GATEWAY_URL}/v1/chat/completions',
            json={
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert ML systems architect analyzing technical literature for patterns applicable to classification systems.'},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 2000,
                'temperature': 0.2,
            },
            timeout=120.0,
        )
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                parsed = json.loads(content)
                patterns_count = len(parsed.get('relevant_patterns', []))
                anti_count = len(parsed.get('anti_patterns', []))
                print(f'    [{model_id}] ✅ {patterns_count} patterns, {anti_count} anti-patterns')
                return {'status': 'success', 'model': model_id, 'data': parsed}
            except json.JSONDecodeError:
                print(f'    [{model_id}] ⚠️  JSON parse error')
                return {'status': 'parse_error', 'model': model_id, 'raw': content[:500]}
        else:
            print(f'    [{model_id}] ❌ HTTP {resp.status_code}')
            return {'status': 'error', 'model': model_id, 'error': f'HTTP {resp.status_code}'}
    
    except Exception as e:
        print(f'    [{model_id}] ❌ {str(e)[:50]}')
        return {'status': 'error', 'model': model_id, 'error': str(e)}


async def analyze_textbook(client: httpx.AsyncClient, book_name: str, book_num: int, total: int) -> dict:
    """Have all 3 models analyze a single textbook."""
    print(f'\n[{book_num}/{total}] 📖 {book_name}')
    
    content = await load_textbook_excerpt(book_name)
    if not content:
        print(f'    ⚠️  Could not load {book_name}')
        return None
    
    print(f'    Loaded {len(content)} chars')
    prompt = ANALYSIS_PROMPT.format(content=content)
    book_title = book_name.replace('.json', '')
    
    results = {}
    for model in MODELS:
        result = await query_model(client, model['id'], model['model'], prompt, book_title)
        results[model['id']] = result
        await asyncio.sleep(2)
    
    return {'book': book_title, 'model_responses': results}


async def main():
    print('=' * 70)
    print('INTER-AI GUIDELINE-TEXTBOOK CROSS-REFERENCE')
    print('=' * 70)
    print(f'Gateway: {LLM_GATEWAY_URL}')
    print(f'Models: GPT-5.2, Claude Opus 4.5, DeepSeek Reasoner')
    print(f'Textbooks: {TEXTBOOKS_DIR}')
    print(f'Target books: {len(TARGET_BOOKS)}')
    print()
    
    # Check gateway
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f'{LLM_GATEWAY_URL}/health', timeout=5.0)
            if health.status_code != 200:
                print('❌ Gateway not healthy!')
                return
            print('✅ Gateway healthy')
        except Exception as e:
            print(f'❌ Gateway error: {e}')
            return
    
    all_results = []
    
    async with httpx.AsyncClient() as client:
        for i, book in enumerate(TARGET_BOOKS, 1):
            result = await analyze_textbook(client, book, i, len(TARGET_BOOKS))
            if result:
                all_results.append(result)
            await asyncio.sleep(3)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'guideline_crossref_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models': [m['model'] for m in MODELS],
            'books_analyzed': len(all_results),
            'results': all_results
        }, f, indent=2)
    
    print()
    print('=' * 70)
    print('ANALYSIS COMPLETE')
    print(f'Results: {output_file}')
    print('=' * 70)
    
    # Aggregate and show consensus
    all_patterns = []
    pattern_votes = {}  # pattern_name -> list of (model, tier)
    
    for result in all_results:
        for model_id, response in result['model_responses'].items():
            if response.get('status') == 'success' and 'data' in response:
                for p in response['data'].get('relevant_patterns', []):
                    p['source_book'] = result['book']
                    p['identified_by'] = model_id
                    all_patterns.append(p)
                    
                    key = p.get('pattern_name', 'unknown')
                    if key not in pattern_votes:
                        pattern_votes[key] = []
                    pattern_votes[key].append((model_id, p.get('applicable_tier', 'unknown')))
    
    print(f'\nTotal patterns: {len(all_patterns)}')
    
    # Show patterns with 2+ model consensus
    consensus_patterns = {k: v for k, v in pattern_votes.items() if len(v) >= 2}
    if consensus_patterns:
        print(f'\n🎯 CONSENSUS PATTERNS (2+ models agree):')
        for pattern, votes in sorted(consensus_patterns.items(), key=lambda x: -len(x[1])):
            models = ', '.join(v[0] for v in votes)
            tier = votes[0][1] if votes else 'unknown'
            print(f'  [{len(votes)}/3] {pattern} → {tier} ({models})')


if __name__ == '__main__':
    asyncio.run(main())
