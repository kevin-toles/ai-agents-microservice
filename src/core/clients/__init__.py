"""HTTP clients for external services.

PCON-4: Neo4j client consolidated to src/clients/neo4j_client.py
Re-exports from consolidated location for backward compatibility.
"""

# PCON-4: Re-export from consolidated client location
from src.clients.neo4j_client import (
    Neo4jClient,
    Neo4jClientConfig,
    create_neo4j_client_from_env,
    get_neo4j_client,
    set_neo4j_client,
)
from src.core.clients.semantic_search import SemanticSearchClient

# Backward compatibility alias
RealNeo4jClient = Neo4jClient

__all__ = [
    "Neo4jClient",
    "Neo4jClientConfig",
    "RealNeo4jClient",  # Backward compatibility
    "SemanticSearchClient",
    "create_neo4j_client_from_env",
    "get_neo4j_client",
    "set_neo4j_client",
]
