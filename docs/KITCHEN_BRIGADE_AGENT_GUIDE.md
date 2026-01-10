# Kitchen Brigade Protocol - Agent Interaction Guide

**Version:** 2.1  
**Updated:** 2026-01-07  
**Status:** Production Ready

## Overview

This document defines how AI agents (like GitHub Copilot, Claude, etc.) should interact with users when initiating Kitchen Brigade multi-model protocols.

### January 2026 Updates
- Infrastructure-aware configuration via `INFRASTRUCTURE_MODE` environment variable
- Stage 2 cross-reference evidence collection (`--enable-cross-reference`)
- Resume capability with `--resume-from` trace files
- Workflow Composer for multi-stage protocol chaining

## Core Principle: Ask, Don't Assume

Before executing a Kitchen Brigade protocol, agents **SHOULD** offer the user control over:
1. **Protocol Type** - Which collaboration pattern to use
2. **LLM Selection** - Which models participate (local, external, or mixed)
3. **Configuration Details** - Number of rounds, temperature, etc.

Users can also delegate decisions with phrases like:
- "you decide"
- "use your best judgment"
- "whatever you think is best"
- "defaults are fine"

---

## Protocol Types

### 1. Round Table (`ROUNDTABLE_DISCUSSION`)
**Pattern**: All LLMs participate in every round  
**Use When**: 
- Exploring new ideas
- Brainstorming sessions
- When you want diverse perspectives throughout
- Complex problems requiring multiple viewpoints

**Structure**:
- Round 1: All share initial perspectives
- Round 2: All respond to each other
- Round 3: All refine positions
- Round 4: All present final views

### 2. Design Review (`ARCHITECTURE_RECONCILIATION`)
**Pattern**: Parallel analysis → Synthesis → Parallel resolution → Consensus  
**Use When**:
- Formal validation/review
- Document reconciliation
- When structured synthesis is needed
- Decision-making with clear deliverables

**Structure**:
- Round 1: All analyze (parallel)
- Round 2: Architect synthesizes (single LLM)
- Round 3: All propose resolutions (parallel)
- Round 4: Architect builds consensus (single LLM)

### 3. Debate (`DEBATE_PROTOCOL`)
**Pattern**: Adversarial discussion with moderator  
**Use When**:
- Trade-off analysis
- Decision between alternatives
- When pros/cons need explicit exploration

### 4. Pipeline (`PIPELINE_PROTOCOL`)
**Pattern**: Sequential handoff  
**Use When**:
- Code generation workflows
- Refinement chains
- When output of one should feed the next

---

## LLM Brigade Options

### Local Only (Zero Cost)
Best for: Regular usage, cost-sensitive, privacy-focused
```yaml
analyst: deepseek-r1-7b
critic: qwen3-8b
synthesizer: phi-4
validator: codellama-13b-instruct
```

### Balanced (Mixed Local + External)
Best for: Important tasks, good quality/cost balance
```yaml
analyst: claude-sonnet-4.5  # External - strong reasoning
critic: qwen3-8b            # Local - fast feedback
synthesizer: gpt-5-mini     # External - good synthesis
validator: codellama-13b    # Local - code validation
```

### Premium (All External)
Best for: Critical decisions, maximum quality
```yaml
analyst: claude-opus-4.5
critic: gpt-5.2-pro
synthesizer: gemini-1.5-pro
validator: claude-sonnet-4.5
```

---

## Agent Interaction Template

When a user requests something that could benefit from Kitchen Brigade, the agent should:

### Step 1: Recognize the Opportunity
Triggers:
- "reconcile these documents"
- "review this architecture"
- "let's discuss this design"
- "get multiple perspectives on"
- "have the LLMs debate"

### Step 2: Offer Configuration (if not trivial)

```markdown
I can run a Kitchen Brigade protocol for this. Before starting:

**Protocol Type:**
1. 🔄 Round Table - All LLMs discuss every round (exploration)
2. 📋 Design Review - Structured with synthesis phases (validation)
3. ⚔️ Debate - Adversarial discussion (decision-making)
4. 🔗 Pipeline - Sequential handoff (code generation)

**Brigade Tier:**
- 🆓 Local Only - Zero API cost (deepseek-r1-7b, qwen3-8b, phi-4, codellama)
- ⚖️ Balanced - Mix of local + external APIs
- 💎 Premium - All external (Claude, GPT, Gemini) - API costs apply

Or just say "you decide" and I'll pick based on your task.
```

### Step 3: If User Says "You Decide"

Agent should:
1. Analyze the task type
2. Consider available hardware/resources
3. Select appropriate configuration
4. Show what was selected
5. Ask for confirmation before executing

```markdown
Based on your task (architecture reconciliation), I recommend:
- **Protocol**: Design Review (structured synthesis)
- **Brigade**: Local Only (zero cost)
- **Rounds**: 4

This will take approximately 5-10 minutes with local models.
Proceed? (yes/no/modify)
```

### Step 4: Execute and Report

During execution:
- Show progress for each round
- Indicate which model is responding
- Show [local] or [external] tags
- Report any errors with retry option

After execution:
- Summarize key findings
- Highlight consensus points
- List dissenting opinions
- Provide actionable recommendations

---

## CLI Usage

```bash
# Interactive mode - prompts for all options
python -m src.protocols.kitchen_brigade_executor \
  --protocol ROUNDTABLE_DISCUSSION \
  --input topic="How should we structure the API gateway?" \
  --interactive

# Direct execution with tier and infrastructure mode
export INFRASTRUCTURE_MODE=hybrid
python -m src.protocols.kitchen_brigade_executor \
  --protocol ARCHITECTURE_RECONCILIATION \
  --input 'documents=["doc1.md", "doc2.md"]' \
  --tier local_only

# Custom brigade override
python -m src.protocols.kitchen_brigade_executor \
  --protocol ROUNDTABLE_DISCUSSION \
  --input topic="Code review strategy" \
  --brigade '{"analyst": "claude-opus-4.5", "critic": "qwen3-8b"}'

# With Stage 2 cross-reference evidence collection
python -m src.protocols.kitchen_brigade_executor \
  --protocol ARCHITECTURE_RECONCILIATION \
  --enable-cross-reference \
  --input 'documents=["ADR.md", "ROUNDTABLE_FINDINGS.md"]'

# Resume from a previous trace
python -m src.protocols.kitchen_brigade_executor \
  --protocol ARCHITECTURE_RECONCILIATION \
  --resume-from trace_ARCHITECTURE_RECONCILIATION_20260107_100256.json
```

---

## Infrastructure Modes

Set the deployment mode for service endpoint resolution:

```bash
# Explicit mode (recommended)
export INFRASTRUCTURE_MODE=hybrid  # docker | hybrid | native

# The executor auto-detects if not set:
# - Running in Docker container → docker mode
# - Otherwise → hybrid mode (default)
```

| Mode | Service URLs | Database URLs | Use Case |
|------|--------------|---------------|----------|
| **docker** | Docker DNS (`llm-gateway:8080`) | Docker DNS | Full containerized |
| **hybrid** | localhost (`localhost:8080`) | Docker DNS or localhost | Dev: DBs in Docker |
| **native** | localhost (`localhost:8080`) | localhost | Fully native |

---

## Configuration Files

- `config/protocols/` - Protocol definitions (JSON)
- `config/brigade_recommendations.yaml` - Scenario-to-preset mappings
- `config/prompts/kitchen_brigade/` - Prompt templates
- `src/infrastructure_config.py` - Dynamic endpoint resolution

---

## Hardware Considerations

| Profile | RAM | Recommendation |
|---------|-----|----------------|
| mac_16gb | 16GB | 2-3 local models max, prefer 7B |
| mac_32gb | 32GB | 4 local models, up to 8B |
| mac_64gb | 64GB | Full brigade, 13B models |
| server_24gb_vram | 24GB VRAM | Any local configuration |

When hardware is limited, automatically recommend:
1. Fewer parallel participants
2. Smaller models
3. Mixed local/external to offload

---

## Error Handling

If a model fails:
1. Log the error with context
2. Offer to retry with different model
3. Continue with remaining participants if parallel round
4. Save partial trace for resume capability
