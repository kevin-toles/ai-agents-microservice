#!/bin/bash
# verify_platform_health.sh
# PCON-8.7: Platform Health Verification Script
# Created: 2026-01-01

set -e

echo "=========================================="
echo "  Platform Health Check (PCON-8.7)"
echo "=========================================="
echo ""

services=(
  "http://localhost:8080/health|llm-gateway"
  "http://localhost:8081/health|semantic-search"
  "http://localhost:8082/health|ai-agents"
  "http://localhost:8083/health|code-orchestrator"
  "http://localhost:8084/health|audit-service"
)

all_healthy=true
healthy_count=0
total_count=${#services[@]}

for service in "${services[@]}"; do
  IFS='|' read -r url name <<< "$service"
  
  # Try to get health status
  response=$(curl -s --connect-timeout 5 "$url" 2>/dev/null || echo '{"status":"unreachable"}')
  status=$(echo "$response" | jq -r '.status // "error"')
  
  if [[ "$status" == "healthy" || "$status" == "ok" ]]; then
    echo "✅ $name: $status"
    ((healthy_count++))
  elif [[ "$status" == "unreachable" ]]; then
    echo "❌ $name: unreachable (service not running)"
    all_healthy=false
  else
    echo "❌ $name: $status"
    # Show dependencies if available
    deps=$(echo "$response" | jq -r '.dependencies // empty' 2>/dev/null)
    if [[ -n "$deps" && "$deps" != "null" ]]; then
      echo "   Dependencies: $deps"
    fi
    all_healthy=false
  fi
done

echo ""
echo "=========================================="
echo "  Summary: $healthy_count/$total_count services healthy"
echo "=========================================="

if $all_healthy; then
  echo ""
  echo "🎉 All services healthy!"
  exit 0
else
  echo ""
  echo "⚠️  Some services unhealthy or unreachable"
  exit 1
fi
