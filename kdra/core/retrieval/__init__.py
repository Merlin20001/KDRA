"""
Retrieval Module
================

Handles vector embedding and similarity search to find relevant
paper chunks for a given query.

Responsibilities:
- Embed queries and chunks
- Vector similarity search
- Return top-K relevant chunks with scores
"""

from kdra.core.retrieval.base import BaseRetriever
from kdra.core.retrieval.simple import MockRetriever
from kdra.core.retrieval.dummy import DummyRetriever

__all__ = [
    "BaseRetriever",
    "MockRetriever",
    "DummyRetriever"
]
