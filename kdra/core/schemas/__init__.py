"""
Schemas Module
==============

Defines the strict JSON schemas (Pydantic models) used throughout the system.
Ensures type safety and structural integrity of extracted data.
"""

from kdra.core.schemas.base import (
    PaperMetadata,
    PaperChunk,
    MetricValue,
    PaperExtraction,
    ComparativeInsight,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    KnowledgeGraph,
    NodeType,
    EdgeType
)

__all__ = [
    "PaperMetadata",
    "PaperChunk",
    "MetricValue",
    "PaperExtraction",
    "ComparativeInsight",
    "KnowledgeGraphNode",
    "KnowledgeGraphEdge",
    "KnowledgeGraph",
    "NodeType",
    "EdgeType"
]
