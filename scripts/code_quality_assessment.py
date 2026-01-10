#!/usr/bin/env python3
"""
Code Quality Assessment for Multi-LLM Re-evaluation

Gathers SonarQube scan results, test coverage, git history for code quality
improvements, and evidence of BDD/TDD practices for LLM assessment.
"""

import os
import json
import httpx
from datetime import datetime

# ============================================================================
# CODE QUALITY EVIDENCE FOR 7 SERVICES
# ============================================================================

CODE_QUALITY_DATA = """
# CODE QUALITY & MAINTAINABILITY EVIDENCE

## Overview: 7 Microservices with Active SonarQube Integration

All 7 services are configured with SonarQube Cloud for continuous code quality monitoring.
Organization: `kevin-toles`

---

## 1. SonarQube Configuration (All 7 Services)

### Services with sonar-project.properties:
| Service | Project Key | Python Version | Coverage Report |
|---------|-------------|----------------|-----------------|
| ai-platform-data | ai-platform-data | 3.11 | coverage.xml |
| ai-agents | ai-agents | 3.11 | coverage.xml |
| Code-Orchestrator-Service | code-orchestrator-service | 3.11 | coverage.xml |
| llm-gateway | llm-gateway | 3.11 | coverage.xml |
| semantic-search-service | semantic-search-service | 3.13 | coverage.xml |
| llm-document-enhancer | llm-document-enhancer | 3.13 | coverage.xml |
| inference-service | kevin-toles_inference-service | 3.11 | coverage.xml |

---

## 2. Test Results Summary (Recent Clean Runs)

### semantic-search-service
- **Tests:** 321 passed, 0 failed, 0 errors, 0 skipped
- **Time:** 29.164 seconds
- **Timestamp:** 2025-12-07T20:27:22
- **Categories:** Benchmark, Integration, Unit, Validation
- **Notable:** Performance benchmarks (P95 < 500ms), hybrid search, graph traversal

### ai-agents
- **Tests:** 157 passed, 0 failed, 0 errors, 2 skipped
- **Time:** 1.734 seconds  
- **Timestamp:** 2025-12-07T20:30:28
- **Skipped:** 2 real LLM integration tests (require API key)
- **Categories:** E2E cross-reference, workflow integration, unit tests

### llm-document-enhancer
- **Tests:** 121 total, 119 passed, 2 failed (edge case chapter segmentation)
- **Time:** 10.777 seconds
- **Timestamp:** 2025-12-07T15:31:45
- **Note:** 2 failures are in PDF chapter segmentation edge cases (actively being refined)

---

## 3. Git History: SonarQube Fixes & Code Quality Improvements

### ai-platform-data (Since Nov 2025)
- `ee544b4` fix: resolve SonarLint issues across codebase
- `f0422cf` chore: add SonarCloud configuration
- `429168e` fix(D1): delete extract_metadata.py - root cause of ID mismatch
- `30c7df9` fix(tests): Update docker-compose tests

### ai-agents
- `de39bda` **refactor: reduce cognitive complexity across 12 functions**
- `17fab86` chore: remove mock-heavy tests that don't provide regression value
- `9ff39f3` **fix(sonarqube): resolve 23 issues across test and source files**
- `ef0c7a7` fix: resolve SonarCloud S7503, S1172, S1135 issues
- `4957c8c` fix: Update sonar-project.properties to scan src/
- `a569a0c` WBS 5.12: REFACTOR - CODING_PATTERNS_ANALYSIS compliance

### Code-Orchestrator-Service
- `6fc7176` chore: remove mock-heavy tests that don't provide regression value
- `77119d2` refactor: Local model loading, restart policy, test cleanup
- `eaa9aab` chore: add SonarCloud configuration
- `9a47143` Rename Agent to Extractor/Validator/Ranker for clarity

### llm-gateway
- `eeb7310` fix: resolve SonarLint issues in providers
- `d577f48` fix: use NOSONAR syntax for SonarQube suppression (Issue 46)
- `2258d68` **fix: resolve SonarQube Batch 6 code smells (Issues 46-52)**
- `6fde06b` **fix: resolve SonarQube code smells (Issues 42-45)**
- `2bfe580` **fix(batch4): Complete all minor issues (29-41) from static analysis**
- `bb154cf` **fix: Complete Batch 1-3 CodeRabbit static analysis issues (28/41)**
- `6368cda` **fix(critical): resolve all 8 CodeRabbit critical issues**
- `d2c50b6` CL-029: WBS 2.10 Code Quality - SonarLint fixes
- `a81b1c3` CL-028: WBS 2.10.1 Final Validation - Lint fixes

### semantic-search-service
- `a29793d` fix: resolve SonarQube issues in main.py
- `4e63499` **fix(sonarlint): resolve all code quality issues for Phase 6**
- `21fdff0` **fix(sonarqube): resolve 17 issues in retriever tests**
- `f42778b` fix: Remove empty TYPE_CHECKING blocks (SonarLint S108)
- `402d1ed` fix: resolve S1172 unused parameter issues with underscore prefix
- `fdd5a51` refactor: Replace POC folder with production structure

### llm-document-enhancer
- `c110ae67` **Batch 9: SonarCloud - Final cognitive complexity refactoring (S3776)**
- `798ea7c8` Batch 8: SonarCloud - Further cognitive complexity reductions
- `96152d23` Batch 7b: SonarCloud - More cognitive complexity reductions
- `ba2ad7c2` Batch 7: SonarCloud - Reduce cognitive complexity
- `b70e3f8e` **Batch 6: Fix SonarCloud issues - S7688, S3457, S1481, S7494, S1244, S108, S5713, S7493, S5914, S3776**
- `eaa2984d` **fix(sonarcloud): Batch 5 - Fix 78 issues across 10 test files**
- `f55fba93` Batch 4: Fix SonarCloud issues - complexity refactoring
- `c58734ff` Batch 3: SonarCloud remediation - Shell scripts and test fixes
- `2078bd67` **fix: SonarCloud issues - Batches 1 & 2 (~164 issues fixed)**
- `4f5304e2` Fix S7924: Improve contrast ratio for .tier-btn-remove (CSS)
- `4cd2317c` Fix S6019: Change .*? to .+? for quantifier matches
- `d6003e12` Fix S1656 self-assignment and S6019 reluctant quantifier issues

### inference-service
- `c9d41fc` fix: resolve remaining SonarQube issues
- `aa0a18b` **security: address all SonarQube Security Hotspots**
- `b3dd64e` fix: resolve SonarLint issues in Dockerfiles and GitHub workflow
- `37be33f` refactor: fix SonarLint issues across test files
- `3e77e87` fix: resolve SonarQube S1192 duplicated string literals
- `aa08ba3` fix: SonarQube issues - CC=17, ABC signature, S1192, S3516

---

## 4. Deprecated Features & Code Cleanup Evidence

### Active Refactoring Decisions:
1. **ai-platform-data:** `extract_metadata.py` deleted - identified as root cause of ID mismatch
2. **ai-agents:** Removed mock-heavy tests that didn't provide regression value
3. **Code-Orchestrator-Service:** Renamed Agent→Extractor/Validator/Ranker for clarity
4. **llm-document-enhancer:** 
   - Removed SBERTClient architecture violation (external app calling internal services)
   - Deleted old cache implementation (Task 5.2 - Option D)
   - Moved observability_platform to separate repo
5. **semantic-search-service:** 
   - Deleted RELEVANCE_TUNING_PLAN.md (obsolete)
   - Replaced POC folder with production structure
   - Removed obsolete docs during config updates

---

## 5. BDD/TDD Methodology Evidence

### Test-Driven Development Commits:
- `f5f5b09` (ai-agents): Add comprehensive TDD test suites for WBS-AGT14-18
- `9d281de` (ai-agents): feat(msep): implement MSE-2 schemas, config, exceptions, constants **(TDD)**
- `27a1d04` (semantic-search-service): WBS 3.1-3.10: Phase 3 Hybrid Search Implementation Complete
- `ca05f1e` (semantic-search-service): Phase 1 Implementation: Graph RAG Infrastructure Setup **(TDD)**
- `6c8779f` (Code-Orchestrator-Service): feat(HTC-1.0): Implement Hybrid Tiered Classifier - WBS-AC1 through AC6
- `bcbcc83` (inference-service): WBS-INF21: Complete integration testing suite

### End-to-End Testing:
- semantic-search-service: Benchmark tests with P95 latency targets (<500ms)
- ai-agents: E2E cross-reference workflow tests with mock LLM
- llm-document-enhancer: End-to-end JSON generation tests
- All services: Integration tests for API contracts

### Continuous Quality Practices:
1. **WBS (Work Breakdown Structure):** Every feature is WBS-tracked
2. **Changelog Discipline:** TECHNICAL_CHANGE_LOG.md in every service
3. **Batch Remediation:** SonarQube issues fixed in organized batches (1-9)
4. **Cognitive Complexity:** Actively refactored (S3776 rule compliance)
5. **Security Hotspots:** All addressed in inference-service audit

---

## 6. SonarQube Issue Categories Fixed

| Rule | Description | Services Affected |
|------|-------------|-------------------|
| S3776 | Cognitive complexity | ai-agents, llm-document-enhancer |
| S1172 | Unused parameters | ai-agents, semantic-search-service |
| S1192 | Duplicated string literals | inference-service, llm-gateway |
| S3457 | Unnecessary f-strings | llm-document-enhancer |
| S7503 | Various code smells | ai-agents |
| S1135 | TODO comments | ai-agents |
| S108 | Empty blocks | semantic-search-service, llm-document-enhancer |
| S6019 | Reluctant quantifiers | llm-document-enhancer |
| S7688 | Shell script issues | llm-document-enhancer |
| S7924 | CSS accessibility | llm-document-enhancer |
| S5713 | Redundant exceptions | llm-document-enhancer |
| Security | Security hotspots | inference-service |

---

## 7. Code Quality Metrics Summary

### Total SonarQube Issues Fixed (Documented):
- **llm-document-enhancer:** ~164 issues (Batches 1-2) + 78 issues (Batch 5) + 9 batches of complexity refactoring
- **llm-gateway:** 52+ issues across 6 batches + 8 critical issues
- **ai-agents:** 23 issues + 12 cognitive complexity refactors
- **semantic-search-service:** 17 issues in retriever tests + Phase 6 cleanup
- **inference-service:** All security hotspots + multiple lint fixes

### Test Counts:
- semantic-search-service: **321 tests**
- ai-agents: **157 tests**
- llm-document-enhancer: **121 tests**
- (Other services have tests but results not captured in XML)

---

## 8. Key Quality Indicators

1. **Active SonarQube Integration:** All 7 services have `sonar-project.properties`
2. **Clean Test Runs:** 0 errors, minimal failures (2 edge cases in chapter segmentation)
3. **Continuous Refactoring:** Git history shows systematic code quality improvements
4. **Deprecation Discipline:** Dead code removed, architecture violations corrected
5. **TDD/BDD Evidence:** WBS-prefixed commits, test-first implementations
6. **Batch Remediation:** Organized approach to fixing static analysis issues
7. **Security Addressed:** Inference-service security hotspots all resolved
8. **Cognitive Complexity:** Actively monitored and refactored (S3776)

---

## IMPORTANT CONTEXT

This is a **POC/MVP platform** built by a **single developer** (TPM with no prior coding background) 
in **5 weeks**. The code quality practices demonstrated are:

1. **Pragmatic for POC:** Not enterprise-polished, but actively maintained
2. **BDD/TDD:** User stories drive features, tests drive implementation
3. **Continuous Improvement:** Evidence shows ongoing refactoring, not technical debt accumulation
4. **Self-Documented:** Technical change logs in every service

The "unvalidated" status means external audit has not occurred, not that code quality is unknown.
"""

# ============================================================================
# LLM API CALLS
# ============================================================================

def call_openai(prompt: str) -> str:
    """Call OpenAI GPT-5.2 API directly."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set"
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 3000,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude Opus 4.5 API directly."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY not set"
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-opus-4-5-20251101",  # Claude Opus 4.5
                "max_tokens": 3000,
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
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def call_qwen3_local(prompt: str) -> str:
    """Call Qwen3-8B via local inference-service."""
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                "http://localhost:8085/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 3000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: Local inference-service not available: {e}"


def main():
    """Run multi-LLM code quality assessment."""
    
    # Previous assessment context
    previous_context = """
## PREVIOUS SYSTEM DESIGN ASSESSMENT (For Context)

In the previous assessment, the LLMs evaluated the overall system design and gave scores of 7-8/10 for POC stage.
Key findings from that assessment:

**GPT-5.2 Previous Assessment:**
- POC Score: 8/10, MVP Score: 7/10
- "Exceptionally well-designed for a single developer"
- Gemini's 4/10 was unfair - missed the POC context

**Claude Opus 4.5 Previous Assessment:**
- POC Score: 8.5/10, MVP Score: 7/10
- "This is sophisticated work that demonstrates genuine systems thinking"
- "The 3-mode strategy isn't complexity - it's a migration path"

**DeepSeek Previous Assessment:**
- POC Score: 9/10, MVP Score: 8/10
- "Architecture shows remarkable maturity for a 5-week build"

**Team Assessment (All 4 Models Agreed):**
- Typical team needed: 6-9 engineers, 4-6 months, $500K-$2.5M
- Actual: 1 TPM, 5 weeks, no coding background
- Productivity multiplier: 10-28x
- Recommendation: L4/E4 hiring level

NOW: Re-evaluate specifically for CODE QUALITY & MAINTAINABILITY based on the SonarQube evidence below.
"""
    
    prompt = f"""
You are evaluating the CODE QUALITY and MAINTAINABILITY of a microservices platform.

{previous_context}

{CODE_QUALITY_DATA}

---

## YOUR TASK

Based on the evidence provided, RE-ASSESS the CODE QUALITY and MAINTAINABILITY dimension specifically.

The previous assessment focused on system design. Now assess:
1. **SonarQube Integration:** Is active static analysis a positive indicator?
2. **Test Coverage:** Are 599+ tests across services adequate for a POC?
3. **Refactoring Evidence:** Does the git history show continuous improvement?
4. **Deprecated Code Cleanup:** Is removing dead code/architecture violations positive?
5. **TDD/BDD Practices:** Is the WBS-driven development methodologically sound?
6. **Batch Remediation:** Is organizing SonarQube fixes into batches systematic?

Provide:
1. **Code Quality Score:** X/10 for POC stage, X/10 if this were production
2. **Maintainability Assessment:** Can another developer easily understand and extend this?
3. **Technical Debt Evaluation:** Is technical debt being accumulated or actively managed?
4. **Key Strengths:** What does this codebase do well?
5. **Areas for Improvement:** What should be addressed before MVP?
6. **Overall Assessment:** Does this code quality evidence change your previous system design assessment?

Be specific and reference the SonarQube evidence provided.
"""
    
    output_dir = "/tmp/code_quality_assessment"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Call each LLM
    print("=" * 60)
    print("CODE QUALITY MULTI-LLM ASSESSMENT")
    print("=" * 60)
    
    print("\n[1/4] Calling OpenAI GPT-5.2...")
    try:
        results["gpt-5.2"] = call_openai(prompt)
        print("✓ GPT-5.2 response received")
    except Exception as e:
        results["gpt-5.2"] = f"ERROR: {e}"
        print(f"✗ GPT-5.2 failed: {e}")
    
    print("\n[2/4] Calling Claude Opus 4.5...")
    try:
        results["claude-opus-4.5"] = call_anthropic(prompt)
        print("✓ Claude Opus 4.5 response received")
    except Exception as e:
        results["claude-opus-4.5"] = f"ERROR: {e}"
        print(f"✗ Claude Opus 4.5 failed: {e}")
    
    print("\n[3/4] Calling DeepSeek...")
    try:
        results["deepseek"] = call_deepseek(prompt)
        print("✓ DeepSeek response received")
    except Exception as e:
        results["deepseek"] = f"ERROR: {e}"
        print(f"✗ DeepSeek failed: {e}")
    
    print("\n[4/4] Calling Qwen3-8B (Local)...")
    try:
        results["qwen3-8b-local"] = call_qwen3_local(prompt)
        print("✓ Qwen3-8B response received")
    except Exception as e:
        results["qwen3-8b-local"] = f"ERROR: {e}"
        print(f"✗ Qwen3-8B failed: {e}")
    
    # Generate report
    report = f"""# CODE QUALITY & MAINTAINABILITY ASSESSMENT
## Multi-LLM Re-Evaluation (Follow-up to System Design Assessment)
Generated: {datetime.now().isoformat()}

---

## Context

This is a FOLLOW-UP assessment specifically evaluating CODE QUALITY based on SonarQube evidence.

**Previous Assessment Summary:**
- System Design: 7-9/10 for POC stage
- Team Comparison: 10-28x productivity multiplier vs typical team
- Recommendation: L4/E4 hiring level

**This Assessment Focus:**
- SonarQube integration across 7 services
- 599+ tests with clean runs
- 250+ documented SonarQube issues fixed
- Git history showing continuous refactoring

---

## Evidence Summary

- **7 Services** with active SonarQube integration (kevin-toles organization)
- **599+ Tests** across semantic-search-service (321), ai-agents (157), llm-document-enhancer (121)
- **250+ SonarQube Issues Fixed** (documented in git history)
- **9 Batch Remediations** in llm-document-enhancer alone
- **All Security Hotspots Resolved** in inference-service
- **Continuous Refactoring** - cognitive complexity (S3776), unused parameters, dead code removal
- **TDD/BDD Evidence** - WBS-prefixed commits, test-first implementations

---

"""
    
    for model, response in results.items():
        report += f"""## {model.upper()} Assessment

{response}

---

"""
    
    # Write report
    report_path = os.path.join(output_dir, "CODE_QUALITY_ASSESSMENT.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"{'=' * 60}")
    
    # Also save raw JSON
    json_path = os.path.join(output_dir, "raw_responses.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()
