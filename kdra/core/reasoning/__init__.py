"""
Reasoning Module
================

Orchestrates LLM interactions to extract structured information
and perform comparative analysis.

Responsibilities:
- LLM-based structured extraction
- Schema compliance enforcement
- Self-verification
"""

from kdra.core.reasoning.engine import BaseReasoningEngine, MockReasoningEngine, OpenAIEngine
from kdra.core.reasoning.extractor import PaperExtractor
from kdra.core.reasoning.comparator import ComparativeAnalyst
from kdra.core.reasoning.qa import ResearchAssistant
from kdra.core.reasoning.dummy import DummyExtractor, DummyComparator

__all__ = [
    "BaseReasoningEngine",
    "MockReasoningEngine",
    "OpenAIEngine",
    "PaperExtractor",
    "ComparativeAnalyst",
    "ResearchAssistant",
    "DummyExtractor",
    "DummyComparator"
]
