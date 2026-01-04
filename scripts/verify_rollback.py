#!/usr/bin/env python3
"""Rollback Verification Script.

WBS-PI7: End-to-End Protocol Testing
AC-PI7.10: Rollback script verifies disabled state

This script verifies that all Phase 2 protocol features are disabled
and that Phase 1 endpoints remain functional.

Usage:
    python scripts/verify_rollback.py [--base-url http://localhost:8082]

Environment:
    All AGENTS_* environment variables should be false (default).

Exit Codes:
    0: Rollback verification passed
    1: Rollback verification failed

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → Rollback Plan
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import NamedTuple

import httpx


class VerificationResult(NamedTuple):
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str


async def verify_agent_card_disabled(client: httpx.AsyncClient) -> VerificationResult:
    """Verify Agent Card is not exposed (404)."""
    try:
        response = await client.get("/.well-known/agent-card.json")
        if response.status_code == 404:
            return VerificationResult(
                name="Agent Card Disabled",
                passed=True,
                message="/.well-known/agent-card.json returns 404",
            )
        else:
            return VerificationResult(
                name="Agent Card Disabled",
                passed=False,
                message=f"Expected 404, got {response.status_code}",
            )
    except Exception as e:
        return VerificationResult(
            name="Agent Card Disabled",
            passed=False,
            message=f"Error: {e}",
        )


async def verify_a2a_endpoints_disabled(client: httpx.AsyncClient) -> list[VerificationResult]:
    """Verify A2A endpoints return 501 (Not Implemented)."""
    results = []
    
    # Test message:send
    try:
        response = await client.post(
            "/a2a/v1/message:send",
            json={
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "test"}],
                }
            },
        )
        if response.status_code == 501:
            results.append(VerificationResult(
                name="A2A message:send Disabled",
                passed=True,
                message="POST /a2a/v1/message:send returns 501",
            ))
        else:
            results.append(VerificationResult(
                name="A2A message:send Disabled",
                passed=False,
                message=f"Expected 501, got {response.status_code}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="A2A message:send Disabled",
            passed=False,
            message=f"Error: {e}",
        ))
    
    # Test message:stream
    try:
        response = await client.post(
            "/a2a/v1/message:stream",
            json={
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "test"}],
                }
            },
        )
        if response.status_code == 501:
            results.append(VerificationResult(
                name="A2A message:stream Disabled",
                passed=True,
                message="POST /a2a/v1/message:stream returns 501",
            ))
        else:
            results.append(VerificationResult(
                name="A2A message:stream Disabled",
                passed=False,
                message=f"Expected 501, got {response.status_code}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="A2A message:stream Disabled",
            passed=False,
            message=f"Error: {e}",
        ))
    
    # Test task status
    try:
        response = await client.get("/a2a/v1/tasks/test-task-id")
        if response.status_code == 501:
            results.append(VerificationResult(
                name="A2A Task Status Disabled",
                passed=True,
                message="GET /a2a/v1/tasks/{id} returns 501",
            ))
        else:
            results.append(VerificationResult(
                name="A2A Task Status Disabled",
                passed=False,
                message=f"Expected 501, got {response.status_code}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="A2A Task Status Disabled",
            passed=False,
            message=f"Error: {e}",
        ))
    
    return results


async def verify_phase1_functional(client: httpx.AsyncClient) -> list[VerificationResult]:
    """Verify Phase 1 endpoints remain functional."""
    results = []
    
    # Test /v1/functions list
    try:
        response = await client.get("/v1/functions")
        if response.status_code == 200:
            data = response.json()
            if "functions" in data:
                results.append(VerificationResult(
                    name="Phase 1 Functions List",
                    passed=True,
                    message="GET /v1/functions returns 200 with functions",
                ))
            else:
                results.append(VerificationResult(
                    name="Phase 1 Functions List",
                    passed=False,
                    message="Response missing 'functions' key",
                ))
        else:
            results.append(VerificationResult(
                name="Phase 1 Functions List",
                passed=False,
                message=f"Expected 200, got {response.status_code}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="Phase 1 Functions List",
            passed=False,
            message=f"Error: {e}",
        ))
    
    # Test extract-structure function
    try:
        response = await client.post(
            "/v1/functions/extract-structure/run",
            json={
                "input": {
                    "content": "# Test\n\nThis is a test document.",
                    "extraction_type": "outline",
                }
            },
        )
        if response.status_code == 200:
            results.append(VerificationResult(
                name="Phase 1 Extract Structure",
                passed=True,
                message="POST /v1/functions/extract-structure/run returns 200",
            ))
        else:
            results.append(VerificationResult(
                name="Phase 1 Extract Structure",
                passed=False,
                message=f"Expected 200, got {response.status_code}: {response.text[:100]}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="Phase 1 Extract Structure",
            passed=False,
            message=f"Error: {e}",
        ))
    
    # Test health endpoint
    try:
        response = await client.get("/health")
        if response.status_code == 200:
            results.append(VerificationResult(
                name="Health Endpoint",
                passed=True,
                message="GET /health returns 200",
            ))
        else:
            results.append(VerificationResult(
                name="Health Endpoint",
                passed=False,
                message=f"Expected 200, got {response.status_code}",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="Health Endpoint",
            passed=False,
            message=f"Error: {e}",
        ))
    
    return results


async def verify_rollback(base_url: str = "http://localhost:8082") -> bool:
    """Run all rollback verification checks.
    
    Args:
        base_url: Base URL of the service to verify
        
    Returns:
        True if all checks pass, False otherwise
    """
    print(f"🔍 Verifying rollback at {base_url}...\n")
    
    all_results: list[VerificationResult] = []
    
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Verify Agent Card disabled
        result = await verify_agent_card_disabled(client)
        all_results.append(result)
        
        # Verify A2A endpoints disabled
        a2a_results = await verify_a2a_endpoints_disabled(client)
        all_results.extend(a2a_results)
        
        # Verify Phase 1 functional
        phase1_results = await verify_phase1_functional(client)
        all_results.extend(phase1_results)
    
    # Print results
    print("=" * 60)
    print("ROLLBACK VERIFICATION RESULTS")
    print("=" * 60)
    
    passed_count = 0
    failed_count = 0
    
    for result in all_results:
        status = "✅" if result.passed else "❌"
        print(f"{status} {result.name}")
        print(f"   {result.message}")
        
        if result.passed:
            passed_count += 1
        else:
            failed_count += 1
    
    print("=" * 60)
    print(f"TOTAL: {passed_count} passed, {failed_count} failed")
    print("=" * 60)
    
    if failed_count == 0:
        print("\n✅ Rollback verification passed")
        return True
    else:
        print("\n❌ Rollback verification failed")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify Phase 2 protocol rollback",
        epilog="Exit 0 if passed, 1 if failed",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8082",
        help="Base URL of the service (default: http://localhost:8082)",
    )
    
    args = parser.parse_args()
    
    passed = asyncio.run(verify_rollback(args.base_url))
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
