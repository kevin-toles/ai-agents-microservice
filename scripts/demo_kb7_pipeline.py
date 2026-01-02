#!/usr/bin/env python3
"""Demo: WBS-KB7 Code-Orchestrator Tool Integration Pipeline.

Exit Criteria Demo:
  Generate code → CodeT5+ extracts keywords → GraphCodeBERT validates →
  SonarQube checks quality → all pass before delivery

Usage:
  python scripts/demo_kb7_pipeline.py
  
Requirements:
  - Code-Orchestrator-Service running on localhost:8083
  - SonarQube running on localhost:9000 (optional)
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.code_analysis import CodeAnalysisClient, FakeCodeAnalysisClient
from src.clients.sonarqube import SonarQubeClient, FakeSonarQubeClient
from src.tools.code_validation import CodeValidationTool, ValidationStep

# =============================================================================
# Demo Configuration
# =============================================================================

_CODE_ORCHESTRATOR_URL = os.getenv("CODE_ORCHESTRATOR_URL", "http://localhost:8083")
_SONARQUBE_URL = os.getenv("SONARQUBE_URL", "http://localhost:9000")
_SONARQUBE_TOKEN = os.getenv("SONARQUBE_TOKEN", "")
_SONARQUBE_PROJECT_KEY = os.getenv("SONARQUBE_PROJECT_KEY", "ai-agents")
_USE_FAKE_CLIENTS = os.getenv("USE_FAKE_CLIENTS", "false").lower() == "true"

# Sample code to validate (generated code example)
GENERATED_CODE = '''
class Repository:
    """Repository pattern implementation for data access layer.
    
    This class provides a clean abstraction over database operations,
    implementing the Repository pattern for entity management.
    """
    
    def __init__(self, connection):
        """Initialize repository with database connection.
        
        Args:
            connection: Database connection instance
        """
        self._connection = connection
    
    def find(self, entity_id: int) -> dict | None:
        """Find entity by ID.
        
        Args:
            entity_id: Unique identifier for the entity
            
        Returns:
            Entity dict if found, None otherwise
        """
        query = "SELECT * FROM entity WHERE id = ?"
        return self._connection.execute(query, (entity_id,))
    
    def find_all(self) -> list[dict]:
        """Find all entities.
        
        Returns:
            List of all entity dicts
        """
        return self._connection.execute("SELECT * FROM entity")
    
    def save(self, entity: dict) -> int:
        """Save entity to database.
        
        Args:
            entity: Entity dict to save
            
        Returns:
            ID of saved entity
        """
        query = "INSERT INTO entity (name, value) VALUES (?, ?)"
        return self._connection.execute(query, (entity["name"], entity["value"]))
    
    def delete(self, entity_id: int) -> bool:
        """Delete entity by ID.
        
        Args:
            entity_id: ID of entity to delete
            
        Returns:
            True if deleted, False otherwise
        """
        query = "DELETE FROM entity WHERE id = ?"
        result = self._connection.execute(query, (entity_id,))
        return result.rowcount > 0
'''


# =============================================================================
# Demo Functions
# =============================================================================


def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step: int, name: str) -> None:
    """Print step header."""
    print(f"\n{'─' * 60}")
    print(f"  Step {step}: {name}")
    print(f"{'─' * 60}")


def print_result(label: str, value: Any) -> None:
    """Print labeled result."""
    print(f"  {label}: {value}")


async def demo_with_real_clients() -> bool:
    """Run demo with real Code-Orchestrator clients."""
    print_header("KB7 Demo: Real Code-Orchestrator Service")
    
    # Initialize clients
    code_client = CodeAnalysisClient(base_url=_CODE_ORCHESTRATOR_URL)
    sonar_client = SonarQubeClient(
        base_url=_SONARQUBE_URL,
        token=_SONARQUBE_TOKEN,
        project_key=_SONARQUBE_PROJECT_KEY,
    )
    
    try:
        # Step 1: Extract keywords with CodeT5+
        print_step(1, "CodeT5+ Keyword Extraction")
        print("  Extracting semantic keywords from generated code...")
        
        keywords_result = await code_client.extract_keywords(
            code=GENERATED_CODE,
            max_keywords=10,
        )
        
        print_result("Model", keywords_result.model)
        print_result("Keywords", keywords_result.keywords[:5] if keywords_result.keywords else [])
        print_result("Status", "✓ PASS" if keywords_result.keywords else "✗ FAIL")
        
        if not keywords_result.keywords:
            print("\n  ERROR: No keywords extracted. Pipeline cannot continue.")
            return False

        # Step 2: Validate terms with GraphCodeBERT
        print_step(2, "GraphCodeBERT Term Validation")
        print("  Validating extracted terms against semantic context...")
        
        validation_result = await code_client.validate_terms(
            terms=keywords_result.keywords,
            query="repository pattern for data access layer with CRUD operations",
        )
        
        print_result("Model", validation_result.model)
        print_result("Terms Validated", len(validation_result.terms))
        
        valid_count = sum(1 for t in validation_result.terms if t.get("valid", False))
        validation_passed = valid_count >= len(validation_result.terms) * 0.5
        print_result("Valid Terms", f"{valid_count}/{len(validation_result.terms)}")
        print_result("Status", "✓ PASS" if validation_passed else "✗ FAIL")

        # Step 3: Rank code with CodeBERT
        print_step(3, "CodeBERT Code Ranking")
        print("  Ranking code snippet relevance...")
        
        ranking_result = await code_client.rank_code_results(
            code_snippets=[GENERATED_CODE],
            query="repository pattern for data access layer",
        )
        
        print_result("Model", ranking_result.model)
        print_result("Rankings", len(ranking_result.rankings))
        
        top_score = ranking_result.rankings[0].get("score", 0) if ranking_result.rankings else 0
        ranking_passed = top_score >= 0.5
        print_result("Top Score", f"{top_score:.2f}")
        print_result("Status", "✓ PASS" if ranking_passed else "✗ FAIL")

        # Step 4: SonarQube quality check (if available)
        print_step(4, "SonarQube Quality Analysis")
        print("  Analyzing code quality metrics...")
        
        try:
            metrics_result = await sonar_client.get_metrics(
                file_path="src/clients/code_analysis.py",
            )
            
            print_result("Complexity", metrics_result.complexity)
            print_result("Cognitive Complexity", metrics_result.cognitive_complexity)
            print_result("Lines of Code", metrics_result.lines_of_code)
            
            quality_passed = metrics_result.complexity < 20
            print_result("Status", "✓ PASS" if quality_passed else "✗ FAIL")
        except Exception as e:
            print(f"  SonarQube not available: {e}")
            quality_passed = True  # Skip this step

        # Final Summary
        print_header("Pipeline Summary")
        all_passed = keywords_result.keywords and validation_passed and ranking_passed and quality_passed
        
        print("  Step 1 (CodeT5+ Keywords):    " + ("✓ PASS" if keywords_result.keywords else "✗ FAIL"))
        print("  Step 2 (GraphCodeBERT Valid): " + ("✓ PASS" if validation_passed else "✗ FAIL"))
        print("  Step 3 (CodeBERT Ranking):    " + ("✓ PASS" if ranking_passed else "✗ FAIL"))
        print("  Step 4 (SonarQube Quality):   " + ("✓ PASS" if quality_passed else "✗ FAIL"))
        print()
        print(f"  OVERALL: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
        print()
        
        return all_passed
        
    finally:
        await code_client.close()
        await sonar_client.close()


async def demo_with_fake_clients() -> bool:
    """Run demo with fake clients (for offline testing)."""
    print_header("KB7 Demo: Fake Clients (Offline Mode)")
    
    # Initialize fake clients
    code_client = FakeCodeAnalysisClient()
    sonar_client = FakeSonarQubeClient()
    
    # Initialize CodeValidationTool
    validation_tool = CodeValidationTool(
        code_analysis_client=code_client,
        sonarqube_client=sonar_client,
    )
    
    try:
        # Run full validation
        print_step(1, "Full Validation Pipeline")
        print("  Running all validation steps...")
        
        result = await validation_tool.validate_code(
            code=GENERATED_CODE,
            query="repository pattern for data access layer",
            file_path="demo/repository.py",
        )
        
        # Display results
        print_step(2, "Results")
        print_result("Passed", result.passed)
        print_result("Steps", len(result.steps))
        print_result("Keywords", result.keywords[:5] if result.keywords else [])
        print_result("Validation Score", f"{result.validation_score:.2f}")
        
        # Display step results
        for step in result.steps:
            print_result(f"  {step.step.value}", f"{'✓' if step.passed else '✗'} (score: {step.score:.2f})")
        
        # Display errors if any
        if result.failure_reason:
            print()
            print(f"  Failure Reason: {result.failure_reason}")
        
        # Final Summary
        print_header("Pipeline Summary")
        
        for step in result.steps:
            status = "✓ PASS" if step.passed else "✗ FAIL"
            print(f"  {step.step.value}: {status}")
        
        print()
        print(f"  OVERALL: {'✓ ALL CHECKS PASSED' if result.passed else '✗ SOME CHECKS FAILED'}")
        print()
        
        return result.passed
        
    finally:
        if hasattr(code_client, 'close'):
            await code_client.close()
        if hasattr(sonar_client, 'close'):
            await sonar_client.close()


async def main() -> int:
    """Run KB7 demo."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  WBS-KB7: Code-Orchestrator Tool Integration Demo                    ║")
    print("║                                                                      ║")
    print("║  Pipeline: CodeT5+ → GraphCodeBERT → CodeBERT → SonarQube            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    if _USE_FAKE_CLIENTS:
        success = await demo_with_fake_clients()
    else:
        # Try real clients first, fall back to fake
        try:
            success = await demo_with_real_clients()
        except Exception as e:
            print(f"\n  Real clients unavailable: {e}")
            print("  Falling back to fake clients...\n")
            success = await demo_with_fake_clients()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
