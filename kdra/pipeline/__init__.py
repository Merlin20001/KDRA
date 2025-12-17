"""
Pipeline Module
===============

Defines the end-to-end execution flow of the KDRA system.
Orchestrates the data flow from ingestion to visualization.

Stages:
1. Topic Input
2. Retrieval
3. Grouping
4. Extraction
5. Alignment
6. Reasoning
7. KG Construction
8. Visualization
"""

from kdra.pipeline.orchestrator import KDRAPipeline
from kdra.pipeline.run_topic import run_topic

__all__ = ["KDRAPipeline", "run_topic"]

__all__ = ["KDRAPipeline"]
