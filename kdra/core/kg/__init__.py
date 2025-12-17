"""
Knowledge Graph Module
======================

Manages the construction and querying of the knowledge graph.

Responsibilities:
- Build graph nodes (Paper, Method, Dataset, Metric, Concept)
- Build graph edges (USES, EVALUATED_ON, REPORTS, RELATED_TO)
- Support JSON-based and backend-based graph storage
"""

from kdra.core.kg.builder import GraphBuilder
from kdra.core.kg.storage import GraphStorage

__all__ = [
    "GraphBuilder",
    "GraphStorage"
]
