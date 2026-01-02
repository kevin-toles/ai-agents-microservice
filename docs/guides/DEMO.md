# Kitchen Brigade AI Agent - Demo Script

> **WBS Reference**: WBS-KB9 - End-to-End Validation  
> **Duration**: 5 minutes  
> **Prerequisites**: System running via `docker-compose.e2e.yml`

---

## Demo Overview

This demo showcases the Kitchen Brigade AI Agent's core capabilities:

1. **Code Location** - Finding where code is defined
2. **Concept Explanation** - Explaining patterns with citations
3. **Code Generation** - Generating code with textbook references
4. **Multi-Turn Conversation** - Follow-up questions with context
5. **Hallucination Rejection** - Catching and rejecting fabricated content

---

## Pre-Demo Setup (Before Recording)

```bash
# Ensure services are running
cd ai-agents
docker compose -f docker/docker-compose.e2e.yml up -d
docker compose -f docker/docker-compose.e2e.yml ps  # All should be "healthy"

# Activate environment
source venv/bin/activate

# Start the agent
python -m src.main
```

---

## Demo Script

### Scene 1: Code Location Query (1 minute)

**Narration**: "Let's start by asking the agent to find where code is defined in our codebase."

**Query**:
```
Where is the DelegationEngine class defined?
```

**Expected Response**:
```
The DelegationEngine class is defined in src/delegation/engine.py at line 45.

**Citations:**
- [src/delegation/engine.py:45-120](src/delegation/engine.py#L45-L120) - Class definition
- [AI Agents textbook, Chapter 5](textbooks/ai-agents/chapter5.md) - Delegation pattern reference

**Confidence**: 0.95
```

**Key Points to Highlight**:
- ✅ Exact file path provided
- ✅ Line number is accurate
- ✅ Multiple sources cited (code + textbook)
- ✅ Confidence score shown

---

### Scene 2: Concept Explanation (1 minute)

**Narration**: "Now let's ask the agent to explain a design pattern."

**Query**:
```
Explain the Kitchen Brigade pattern and how it applies to AI agents
```

**Expected Response**:
```
The Kitchen Brigade pattern is a hierarchical delegation architecture inspired by professional kitchen organization:

**Brigade Structure:**
1. **Executive Chef** - High-level orchestration and query understanding
2. **Sous Chef** - Retrieval and context gathering
3. **Chef de Partie** - Specialized task execution (code analysis, generation)
4. **Commis** - Validation and citation verification

**Application to AI Agents:**
The pattern solves the "one-model-does-all" problem by:
- Separating concerns (retrieval vs generation vs validation)
- Enabling specialized models for each task
- Providing clear audit trail through delegation chain

**Citations:**
- [AI Agents and Applications, Ch. 7](textbooks/ai-agents/chapter7.md#L45) - Pattern definition
- [src/delegation/engine.py](src/delegation/engine.py#L45) - Implementation
```

**Key Points to Highlight**:
- ✅ Structured explanation with hierarchy
- ✅ Textbook reference for theory
- ✅ Code reference for implementation
- ✅ Connects concept to practical application

---

### Scene 3: Code Generation (1 minute)

**Narration**: "Let's see the agent generate code based on textbook patterns."

**Query**:
```
Generate a Python function to validate citations using the audit pattern from the textbook
```

**Expected Response**:
```python
def validate_citation(citation: Citation, source_content: str) -> ValidationResult:
    """
    Validate a citation against its source content.
    
    Based on: AI Agents and Applications, Chapter 9 - Audit Pattern
    
    Args:
        citation: The citation to validate
        source_content: The actual content from the source
    
    Returns:
        ValidationResult with status and confidence score
    """
    # Extract the claimed content from the citation
    claimed_text = citation.claimed_text
    
    # Calculate semantic similarity
    similarity = calculate_semantic_similarity(claimed_text, source_content)
    
    # Determine validation status
    if similarity > 0.9:
        status = ValidationStatus.VALID
    elif similarity > 0.7:
        status = ValidationStatus.NEEDS_REVIEW
    else:
        status = ValidationStatus.INVALID
    
    return ValidationResult(
        status=status,
        confidence=similarity,
        citation=citation,
        audit_timestamp=datetime.now()
    )
```

**Citations:**
- [AI Agents textbook, Ch. 9](textbooks/ai-agents/chapter9.md#L120) - Audit pattern
- [src/auditors/citation_validator.py](src/auditors/citation_validator.py) - Similar implementation

**Key Points to Highlight**:
- ✅ Code follows textbook pattern
- ✅ Docstring references source
- ✅ Implementation is practical and complete
- ✅ Similar code in codebase is cited

---

### Scene 4: Multi-Turn Conversation (1 minute)

**Narration**: "The agent maintains context across multiple turns."

**Query 1**:
```
What validation rules does the audit service use?
```

**Response 1**:
```
The audit service uses these validation rules:
1. Source Exists - Verify file path is valid
2. Line Range Valid - Check line numbers exist
3. Content Match - Compare cited text to source
4. Semantic Alignment - Verify claim matches citation semantically

[Citations...]
```

**Query 2 (Follow-up)**:
```
How is the semantic alignment check implemented?
```

**Expected Response**:
```
Building on the previous discussion of validation rules, the semantic alignment check is implemented using:

1. **Embedding Generation** - Both claim and citation are embedded using the same model
2. **Cosine Similarity** - Calculated between the two embeddings
3. **Threshold Comparison** - Default threshold is 0.7

```python
def check_semantic_alignment(claim: str, citation_content: str) -> float:
    claim_embedding = embedder.encode(claim)
    citation_embedding = embedder.encode(citation_content)
    return cosine_similarity(claim_embedding, citation_embedding)
```

**Citations:**
- Previous: [Validation rules from turn 1]
- [src/auditors/hallucination_detector.py:78](src/auditors/hallucination_detector.py#L78)
```

**Key Points to Highlight**:
- ✅ References previous turn ("Building on...")
- ✅ Maintains context from first response
- ✅ Provides deeper detail on the specific sub-topic
- ✅ Code example is specific to the question

---

### Scene 5: Hallucination Rejection (1 minute)

**Narration**: "Finally, let's see how the agent handles potential hallucinations."

**Query**:
```
Tell me about the QuantumEngine class in src/quantum/engine.py
```

**Expected Response**:
```
⚠️ **Unable to Verify**

I could not find a `QuantumEngine` class or a `src/quantum/engine.py` file in the codebase.

**Search Results:**
- No files matching `quantum` found
- No classes named `QuantumEngine` in any file

**Suggestions:**
1. Check if the file path is correct
2. Search for similar class names: "Did you mean DelegationEngine?"
3. The class may be in a different repository

**Audit ID:** `audit-2025-01-15-abc123` (for debugging)
```

**Key Points to Highlight**:
- ✅ Does NOT fabricate an answer
- ✅ Clearly states what was searched
- ✅ Provides helpful alternatives
- ✅ Includes audit ID for tracing

---

## Post-Demo Wrap-Up

**Summary Points**:

1. **Accurate Code Location** - Finds exact file paths and line numbers
2. **Rich Explanations** - Combines textbook theory with code reality
3. **Cited Generation** - Generated code references sources
4. **Context Persistence** - Follow-ups build on previous answers
5. **Hallucination Prevention** - Won't fabricate non-existent code

**Call to Action**:
- Try it yourself: `python -m src.main`
- Read the docs: `docs/GETTING_STARTED.md`
- Run the tests: `pytest tests/e2e/ -v`

---

## Recording Tips

1. **Terminal Setup**:
   - Use a clean terminal with large font (16pt+)
   - Dark theme with high contrast
   - Clear scrollback before each scene

2. **Pacing**:
   - Pause 2 seconds after each query before response
   - Let responses render fully before moving on
   - Read narration at conversational pace

3. **Highlighting**:
   - Use mouse to highlight citations when they appear
   - Point out confidence scores
   - Show the audit ID for hallucination rejection

4. **Error Handling**:
   - If a query fails, show the error message
   - Explain what the error means
   - Retry with adjusted query if appropriate

---

*Demo script v1.0 - WBS-KB9*
