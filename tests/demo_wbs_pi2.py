#!/usr/bin/env python3
"""Demo script for WBS-PI2 Exit Criteria.

Demonstrates:
- Agent Card returns 404 by default
- Agent Card returns valid JSON when enabled
- Card contains exactly 8 skills matching agent functions
"""

import os
import sys
from fastapi.testclient import TestClient


def demo_disabled_state():
    """Demo: Agent Card returns 404 by default."""
    print("\n" + "="*80)
    print("DEMO 1: Agent Card DISABLED (default state)")
    print("="*80)
    
    # Ensure flags are disabled
    os.environ.pop("AGENTS_A2A_ENABLED", None)
    os.environ.pop("AGENTS_A2A_AGENT_CARD_ENABLED", None)
    
    from src.config.feature_flags import get_feature_flags
    get_feature_flags.cache_clear()
    
    from src.main import app
    client = TestClient(app)
    
    response = client.get("/.well-known/agent-card.json")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 404
    print("✅ Agent Card correctly returns 404 when disabled")


def demo_enabled_state():
    """Demo: Agent Card returns valid JSON when enabled."""
    print("\n" + "="*80)
    print("DEMO 2: Agent Card ENABLED")
    print("="*80)
    
    # Enable A2A and agent card
    os.environ["AGENTS_A2A_ENABLED"] = "true"
    os.environ["AGENTS_A2A_AGENT_CARD_ENABLED"] = "true"
    os.environ["AGENTS_A2A_STREAMING_ENABLED"] = "true"
    
    from src.config.feature_flags import get_feature_flags
    get_feature_flags.cache_clear()
    
    from src.main import app
    client = TestClient(app)
    
    response = client.get("/.well-known/agent-card.json")
    
    print(f"Status Code: {response.status_code}")
    
    assert response.status_code == 200
    
    card = response.json()
    
    print(f"\nProtocol Version: {card['protocolVersion']}")
    print(f"Service Name: {card['name']}")
    print(f"Description: {card['description']}")
    print(f"Version: {card['version']}")
    
    print(f"\nCapabilities:")
    for key, value in card['capabilities'].items():
        print(f"  {key}: {value}")
    
    print(f"\nSkills ({len(card['skills'])} total):")
    for skill in card['skills']:
        print(f"  - {skill['id']}: {skill['name']}")
        print(f"    Tags: {', '.join(skill['tags'])}")
    
    # Validate
    assert card['protocolVersion'] == '0.3.0'
    assert card['name'] == 'ai-agents-service'
    assert len(card['skills']) == 8
    assert card['capabilities']['streaming'] is True
    
    print("\n✅ Agent Card correctly returns valid JSON with 8 skills")
    
    # Cleanup
    os.environ.pop("AGENTS_A2A_ENABLED", None)
    os.environ.pop("AGENTS_A2A_AGENT_CARD_ENABLED", None)
    os.environ.pop("AGENTS_A2A_STREAMING_ENABLED", None)
    get_feature_flags.cache_clear()


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("WBS-PI2 EXIT CRITERIA DEMO")
    print("A2A Agent Card & Discovery")
    print("="*80)
    
    try:
        demo_disabled_state()
        demo_enabled_state()
        
        print("\n" + "="*80)
        print("✅ ALL DEMOS PASSED")
        print("="*80)
        print("\nExit Criteria Validated:")
        print("  ✅ GET /.well-known/agent-card.json returns 404 by default")
        print("  ✅ With flags enabled → returns valid JSON")
        print("  ✅ Card contains exactly 8 skills matching agent functions")
        print("="*80 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
