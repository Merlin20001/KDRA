"""
Ingestion Module
================

Responsible for loading raw paper data (PDF, text, URLs) and
segmenting it into semantically meaningful chunks with metadata.

Responsibilities:
- Load papers
- Chunk text
- Attach metadata (paper_id, section, year, venue)
"""

from kdra.core.ingestion.chunker import BaseChunker, SimpleTextChunker
from kdra.core.ingestion.loader import PaperIngestor

__all__ = [
    "BaseChunker",
    "SimpleTextChunker",
    "PaperIngestor"
]
