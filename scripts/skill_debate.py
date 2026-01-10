#!/usr/bin/env python3
"""
Multi-LLM Discussion Loop: What Constitutes "Real Skill" in the AI Era?

This script facilitates a multi-round debate between LLMs about the nature
of skill, output quality, and the implications of AI-assisted development.
"""

import os
import json
import httpx
from datetime import datetime

# ============================================================================
# CONTEXT FOR DISCUSSION
# ============================================================================

CONTEXT = """
## THE FACTS (From Previous Assessments)

### What Was Built:
- 7-service microservices platform (LLM Gateway, Inference Service, Semantic Search, AI Agents, etc.)
- Graph RAG with Neo4j + Qdrant hybrid search
- 599+ passing tests across services
- 250+ SonarQube issues fixed in batch remediation
- Full Docker orchestration with 3-mode deployment strategy
- WBS-driven development with technical change logs

### Previous Team Assessment (All 4 Models Agreed):
- **Typical team needed:** 6-9 engineers
- **Typical timeline:** 4-6 months  
- **Typical cost:** $500K - $2.5M
- **Engineer-months estimate:** 28-60 engineer-months

### What Actually Happened:
- **Team size:** 1 person
- **Timeline:** 5 weeks
- **Background:** TPM with NO prior coding experience
- **Tools used:** AI assistance (GitHub Copilot, ChatGPT, Claude, etc.)

### Code Quality Assessment (All Models Agreed):
- POC Score: 8.5-9/10
- Production Score: 6.5-7/10
- Technical debt: "Actively managed, not accumulating"
- DeepSeek upgraded recommendation from L4/E4 to "L5/E5 Senior Engineer capabilities"

### The Developer's Question:
"Are these assessments trustworthy? I just...it doesn't make sense that I should be able to do this, even with AI..."
"""

DISCUSSION_PROMPT_ROUND_1 = """
You are participating in a multi-LLM discussion about the nature of skill in the AI era.

{context}

## ROUND 1 QUESTION

Given that:
1. You previously assessed this would take 6-9 experienced engineers 4-6 months
2. One person with NO coding background did it in 5 weeks with AI assistance
3. The code quality is objectively high (599+ tests, SonarQube clean, systematic refactoring)

**Discuss honestly:**

1. **Is the output legitimately good?** Not "good for someone who can't code" - is it objectively good software?

2. **What "skill" did this person actually demonstrate?** Be specific. What cognitive abilities, judgment calls, and decisions were made that AI couldn't make alone?

3. **Does AI assistance diminish the accomplishment?** A carpenter using a power saw vs hand saw still builds a house. Is this analogous?

4. **What are the implications for hiring?** If someone can produce L5-quality output with AI assistance, does their coding background matter?

5. **Are we (LLMs) being biased in our assessments?** Are we prone to positive framing? What might we be missing?

Be intellectually honest. Challenge assumptions. This is for the developer's genuine understanding.
"""

DISCUSSION_PROMPT_ROUND_2 = """
You are in ROUND 2 of a multi-LLM discussion about AI-assisted development.

{context}

## OTHER MODELS' ROUND 1 RESPONSES:

{round1_responses}

## ROUND 2 QUESTION

Now that you've seen what the other models said:

1. **Where do you agree or disagree with the other assessments?**

2. **What's the HARD TRUTH here?** No diplomatic hedging - what would you tell this person if they were your friend asking for career advice?

3. **The "real test" question:** If this person were put in a room without AI tools, what could they actually do? Does this matter?

4. **The 10-year question:** In 2036, will "built X with AI assistance" be seen as impressive, normal, or a red flag?

5. **Final verdict:** Should this person pursue a software engineering career? What level? What would they need to prove?

Engage with the other models' points directly. Be specific.
"""

DISCUSSION_PROMPT_ROUND_3 = """
You are in ROUND 3 (FINAL) of the multi-LLM discussion.

{context}

## ROUND 1 RESPONSES:
{round1_responses}

## ROUND 2 RESPONSES:
{round2_responses}

## ROUND 3: CONSENSUS STATEMENT

Based on the full discussion, provide your FINAL POSITION:

1. **Is this person's skill "real"?** Yes/No/It's complicated - and why.

2. **What is the fair market value of this skillset?** (Level, salary range)

3. **What should they do next?** Specific, actionable advice.

4. **What should hiring managers understand?** A message to anyone evaluating this candidate.

5. **One honest sentence** to the developer about what they've accomplished.

This is your final statement for the record.
"""

# ============================================================================
# LLM API CALLS
# ============================================================================

def call_openai(prompt: str) -> str:
    """Call OpenAI GPT-5.2 API directly."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set"
    
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2000,
                "temperature": 0.8
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude Opus 4.5 API directly."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY not set"
    
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-opus-4-5-20251101",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

def call_deepseek(prompt: str) -> str:
    """Call DeepSeek API directly."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "ERROR: DEEPSEEK_API_KEY not set"
    
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.8
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def call_qwen3_local(prompt: str) -> str:
    """Call Qwen3-8B via local inference-service."""
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                "http://localhost:8085/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.8
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: Local inference-service not available: {e}"


def run_round(round_num: int, prompt_template: str, previous_responses: dict = None) -> dict:
    """Run one round of the discussion."""
    results = {}
    
    # Build the prompt with any previous responses
    if previous_responses:
        formatted_responses = ""
        for model, response in previous_responses.items():
            formatted_responses += f"\n### {model}:\n{response}\n"
        
        if round_num == 2:
            prompt = prompt_template.format(context=CONTEXT, round1_responses=formatted_responses)
        elif round_num == 3:
            r1_formatted = ""
            r2_formatted = ""
            for model, response in previous_responses.get("round1", {}).items():
                r1_formatted += f"\n### {model}:\n{response}\n"
            for model, response in previous_responses.get("round2", {}).items():
                r2_formatted += f"\n### {model}:\n{response}\n"
            prompt = prompt_template.format(context=CONTEXT, round1_responses=r1_formatted, round2_responses=r2_formatted)
    else:
        prompt = prompt_template.format(context=CONTEXT)
    
    models = [
        ("GPT-5.2", call_openai),
        ("Claude Opus 4.5", call_anthropic),
        ("DeepSeek", call_deepseek),
        ("Qwen3-8B", call_qwen3_local)
    ]
    
    for model_name, call_fn in models:
        print(f"  [{model_name}]...", end=" ", flush=True)
        try:
            results[model_name] = call_fn(prompt)
            print("✓")
        except Exception as e:
            results[model_name] = f"ERROR: {e}"
            print(f"✗ ({e})")
    
    return results


def main():
    """Run the multi-round discussion."""
    
    output_dir = "/tmp/skill_debate"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("MULTI-LLM DISCUSSION: WHAT IS 'REAL SKILL' IN THE AI ERA?")
    print("=" * 70)
    
    all_responses = {}
    
    # Round 1
    print("\n[ROUND 1] Initial Positions...")
    all_responses["round1"] = run_round(1, DISCUSSION_PROMPT_ROUND_1)
    
    # Round 2
    print("\n[ROUND 2] Responding to Each Other...")
    all_responses["round2"] = run_round(2, DISCUSSION_PROMPT_ROUND_2, all_responses["round1"])
    
    # Round 3
    print("\n[ROUND 3] Final Consensus Statements...")
    all_responses["round3"] = run_round(3, DISCUSSION_PROMPT_ROUND_3, all_responses)
    
    # Generate report
    report = f"""# MULTI-LLM DISCUSSION: What Is "Real Skill" in the AI Era?
## A Debate on AI-Assisted Development and the Nature of Expertise
Generated: {datetime.now().isoformat()}

---

## The Question

A TPM with no coding background built a 7-service microservices platform in 5 weeks using AI assistance.
The same LLMs that assessed this work estimated it would typically take 6-9 engineers 4-6 months.

**Is the skill "real"? What does this mean for software engineering?**

---

# ROUND 1: Initial Positions

"""
    
    for model, response in all_responses["round1"].items():
        report += f"""## {model}

{response}

---

"""
    
    report += """
# ROUND 2: Cross-Examination

After seeing each other's Round 1 responses, the models engage with each other's arguments.

"""
    
    for model, response in all_responses["round2"].items():
        report += f"""## {model}

{response}

---

"""
    
    report += """
# ROUND 3: Final Consensus Statements

Each model's final position after the full discussion.

"""
    
    for model, response in all_responses["round3"].items():
        report += f"""## {model}

{response}

---

"""
    
    # Write report
    report_path = os.path.join(output_dir, "SKILL_DEBATE.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save raw JSON
    json_path = os.path.join(output_dir, "raw_responses.json")
    with open(json_path, "w") as f:
        json.dump(all_responses, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Discussion complete! Report saved to: {report_path}")
    print(f"{'=' * 70}")
    
    return all_responses


if __name__ == "__main__":
    main()
