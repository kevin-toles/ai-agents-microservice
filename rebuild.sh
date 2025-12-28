#!/bin/bash
# Rebuild and restart the container with new code
set -e
cd "$(dirname "$0")"
echo "🔄 Rebuilding ai-agents..."
docker-compose up -d --build
echo "✅ Done. Verifying..."
docker exec ai-agents python -c "from src.agents.msep import schemas; print('Container code loaded successfully')"
