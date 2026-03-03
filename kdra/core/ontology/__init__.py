"""
Ontology Module
===============

Handles concept alignment and normalization against external or internal ontologies.

Responsibilities:
- Map extracted terms to normalized concepts
- Resolve synonyms
- Track concept frequency
"""

from kdra.core.ontology.normalizer import VectorOntologyNormalizer

__all__ = ["VectorOntologyNormalizer"]
