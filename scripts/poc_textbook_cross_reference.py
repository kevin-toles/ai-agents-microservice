#!/usr/bin/env python3
"""
================================================================================
FILE: poc_textbook_cross_reference.py
--------------------------------------------------------------------------------
INTENT: Inter-AI orchestration script that uses GPT-5.2, Claude Opus 4.5, and 
        DeepSeek Reasoner to analyze JSON textbooks and extract architectural 
        patterns relevant to building a Hybrid Tiered Classifier.

SOURCES:
- Kitchen Brigade Architecture Model (TIER_RELATIONSHIP_DIAGRAM.md)
- ML Design Patterns (Machine Learning Design Patterns.json)
- Designing ML Systems (Designing Machine Learning Systems.json)
- GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md

DEPENDENCIES:
- httpx (async HTTP client)
- llm-gateway running on localhost:8080
- JSON textbooks in /Users/kevintoles/POC/textbooks/textbooks_json/

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
TEXTBOOKS_DIR = Path('/Users/kevintoles/POC/textbooks/textbooks_json')
OUTPUT_DIR = Path('/Users/kevintoles/POC/ai-agents/data/textbook_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {'id': 'gpt', 'model': 'gpt-5.2-2025-12-11'},
    {'id': 'claude', 'model': 'claude-opus-4-5-20251101'},
    {'id': 'deepseek', 'model': 'deepseek-reasoner'},
]

# Target textbooks for Hybrid Tiered Classifier architecture
TARGET_BOOKS = [
    'Designing Machine Learning Systems.json',
    'Machine Learning Design Patterns.json',
    'Building Machine Learning Pipelines.json',
    'Reliable Machine Learning.json',
    'Practical MLOps.json',
    'Machine Learning Engineering.json',
    'Introducing MLOps.json',
    'Clean Code.json',
    'Clean Architecture.json',
    'Design Patterns.json',
]

ANALYSIS_PROMPT = """You are analyzing technical textbook content to find patterns relevant to building a **Hybrid Tiered Classification System** for software concepts.

CONTEXT: We're building a 4-tier classifier:
- Tier 1: Alias/Exact Match Lookup (O(1) dictionary)
- Tier 2: ML Classifier (SBERT embeddings + LogisticRegression)
- Tier 3: Heuristic Rules (noise filtering, regex patterns)
- Tier 4: LLM Fallback (for edge cases)

TEXTBOOK CONTENT:
{content}

TASK: Extract relevant patterns, principles, and recommendations from this content that apply to:
1. **Tiered/Cascade classification systems** - fallback patterns, confidence thresholds
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


async def load_textbook_excerpt(book_name: str, max_chars: int = 15000) -> str:
    """Load first N characters from a textbook JSON."""
    book_path = TEXTBOOKS_DIR / book_name
    if not book_path.exists():
        return None
    
    with open(book_path) as f:
        data = json.load(f)
    
    # Extract text content (structure varies by book)
    content_parts = []
    if isinstance(data, dict):
        for key in ['content', 'text', 'chapters', 'sections']:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    content_parts.append(val)
                elif isinstance(val, list):
                    for item in val[:10]:  # First 10 chapters/sections
                        if isinstance(item, str):
                            content_parts.append(item)
                        elif isinstance(item, dict):
                            content_parts.append(json.dumps(item)[:2000])
    
    if not content_parts:
        # Fallback: stringify the whole thing
        content_parts.append(json.dumps(data)[:max_chars])
    
    return '\n'.join(content_parts)[:max_chars]


async def query_model(client: httpx.AsyncClient, model_id: str, model_name: str, 
                      prompt: str, book_title: str) -> dict:
    """Query a single model for textbook analysis."""
    print(f'  [{model_id}] Analyzing {book_title}...')
    
    try:
        resp = await client.post(
            f'{LLM_GATEWAY_URL}/v1/chat/completions',
            json={
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert ML systems architect analyzing technical literature.'},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 2000,
                'temperature': 0.2,
            },
            timeout=120.0,
        )
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            # Try to parse JSON from response
            try:
                # Clean up potential markdown wrapping
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                return {'status': 'success', 'model': model_id, 'data': json.loads(content)}
            except json.JSONDecodeError:
                return {'status': 'parse_error', 'model': model_id, 'raw': content[:500]}
        else:
            return {'status': 'error', 'model': model_id, 'error': f'HTTP {resp.status_code}'}
    
    except Exception as e:
        return {'status': 'error', 'model': model_id, 'error': str(e)}


async def analyze_textbook(client: httpx.AsyncClient, book_name: str) -> dict:
    """Have all 3 models analyze a single textbook."""
    content = await load_textbook_excerpt(book_name)
    if not content:
        print(f'  ⚠️  Could not load {book_name}')
        return None
    
    prompt = ANALYSIS_PROMPT.format(content=content)
    book_title = book_name.replace('.json', '')
    
    results = {}
    for model in MODELS:
        result = await query_model(client, model['id'], model['model'], prompt, book_title)
        results[model['id']] = result
        await asyncio.sleep(1)  # Rate limiting
    
    return {'book': book_title, 'model_responses': results}


async def main():
    print('=' * 70)
    print('INTER-AI TEXTBOOK CROSS-REFERENCE ANALYSIS')
    print('=' * 70)
    print(f'Gateway: {LLM_GATEWAY_URL}')
    print(f'Models: GPT-5.2, Claude Opus 4.5, DeepSeek Reasoner')
    print(f'Target books: {len(TARGET_BOOKS)}')
    print()
    
    all_results = []
    
    async with httpx.AsyncClient() as client:
        for i, book in enumerate(TARGET_BOOKS, 1):
            print(f'\n[{i}/{len(TARGET_BOOKS)}] Processing: {book}')
            result = await analyze_textbook(client, book)
            if result:
                all_results.append(result)
            await asyncio.sleep(2)  # Rate limiting between books
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'textbook_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models': [m['model'] for m in MODELS],
            'books_analyzed': len(all_results),
            'results': all_results
        }, f, indent=2)
    
    print()
    print('=' * 70)
    print(f'ANALYSIS COMPLETE')
    print(f'Results saved to: {output_file}')
    print('=' * 70)
    
    # Print summary
    successful = sum(1 for r in all_results if r)
    print(f'\nBooks analyzed: {successful}/{len(TARGET_BOOKS)}')
    
    # Aggregate patterns found
    all_patterns = []
    for result in all_results:
        for model_id, response in result['model_responses'].items():
            if response.get('status') == 'success' and 'data' in response:
                patterns = response['data'].get('relevant_patterns', [])
                for p in patterns:
                    p['source_book'] = result['book']
                    p['identified_by'] = model_id
                all_patterns.append(patterns)
    
    print(f'Total patterns extracted: {sum(len(p) for p in all_patterns)}')


if __name__ == '__main__':
    asyncio.run(main())
