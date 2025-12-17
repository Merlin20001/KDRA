from typing import Dict, Optional, List

class ConceptNormalizer:
    """
    Handles normalization of concepts and alignment with an ontology.
    Maintains a registry of synonyms and mappings to external ontology IDs.
    """
    
    def __init__(self):
        # Map of alias/synonym (lowercase) -> canonical concept name
        self._synonym_map: Dict[str, str] = {}
        # Map of canonical name -> external ontology ID
        self._ontology_ids: Dict[str, str] = {}
        
        # Pre-populate with some common AI research terms for demonstration
        self._seed_knowledge()

    def _seed_knowledge(self):
        """Add some initial synonyms and IDs."""
        # Synonyms
        self.register_synonym("large language model", "LLM")
        self.register_synonym("gpt-3", "GPT-3")
        self.register_synonym("gpt-4", "GPT-4")
        self.register_synonym("convolutional neural network", "CNN")
        self.register_synonym("transformer network", "Transformer")
        self.register_synonym("chain-of-thought", "CoT")
        
        # Ontology IDs (Mock IDs based on Computer Science Ontology or similar)
        self.register_concept("LLM", "CSO:12345")
        self.register_concept("CNN", "CSO:67890")
        self.register_concept("Transformer", "CSO:54321")
        self.register_concept("CoT", "CSO:99887")

    def normalize(self, term: str) -> str:
        """
        Normalize a term to its canonical form.
        
        Args:
            term: The raw term string.
            
        Returns:
            The canonical term if found, otherwise the original term (cleaned).
        """
        if not term:
            return ""
            
        clean_term = term.strip()
        lower_term = clean_term.lower()
        
        # Check direct match in synonym map
        if lower_term in self._synonym_map:
            return self._synonym_map[lower_term]
            
        # Return original with standard casing if no mapping found
        # (In a real system, we might use fuzzy matching or stemming here)
        return clean_term

    def get_id(self, term: str) -> Optional[str]:
        """
        Get the external ontology ID for a term.
        
        Args:
            term: The term to look up (will be normalized first).
            
        Returns:
            The ontology ID string if found, else None.
        """
        canonical = self.normalize(term)
        return self._ontology_ids.get(canonical)

    def register_synonym(self, alias: str, canonical: str):
        """
        Register a new synonym.
        
        Args:
            alias: The alternative name (e.g., "Large Language Model").
            canonical: The standard name (e.g., "LLM").
        """
        self._synonym_map[alias.lower()] = canonical

    def register_concept(self, canonical: str, ontology_id: str):
        """
        Register a concept with an ID.
        
        Args:
            canonical: The standard name.
            ontology_id: The external ID.
        """
        self._ontology_ids[canonical] = ontology_id
