import re
import logging
from typing import Dict, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    FUZZ_AVAILABLE = True
except ImportError:
    FUZZ_AVAILABLE = False

class VectorOntologyNormalizer:
    """
    Normalizes entity IDs using a hybrid approach of rapid fuzzy string matching
    and vector embeddings to merge semantically similar concepts.
    For example, "Transformer" and "Transform architecture" will be mapped to the same ID.
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', vector_threshold: float = 0.88, fuzz_threshold: float = 90.0):
        self.vector_threshold = vector_threshold
        self.fuzz_threshold = fuzz_threshold
        self.entity_memory: Dict[str, object] = {} # { "entity_id": embedding_tensor }
        self.entity_names: Dict[str, str] = {}     # { "entity_id": "Original Name" }
        
        if ST_AVAILABLE:
            logging.info(f"Loading Vector Ontology Model: {model_name}... (First time may take a moment)")
            # Set trust_remote_code=True if needed, but BAAI models usually don't need it.
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
            logging.warning("sentence-transformers not installed. Falling back to basic string normalization for Ontology.")

    def normalize_and_dedupe(self, node_type_value: str, name: str) -> str:
        """
        Takes an entity name and its type, returns a unified ID.
        If a similar entity already exists, returns the existing ID to merge them.
        Otherwise, registers a new one.
        """
        # 1. Base clean name
        clean_name = self._basic_clean(name)
        type_prefix = f"{node_type_value.lower()}:"
        
        # 2. Fast Fuzzy String Matching (Lexical alignment)
        if FUZZ_AVAILABLE:
            for e_id, existing_name in self.entity_names.items():
                if e_id.startswith(type_prefix):
                    score = fuzz.WRatio(name.lower(), existing_name.lower())
                    if score >= self.fuzz_threshold:
                        return e_id
        else:
            # Fallback to exact match if fuzz is not available
            for e_id, existing_name in self.entity_names.items():
                if e_id.startswith(type_prefix) and name.lower() == existing_name.lower():
                    return e_id

        # 3. Vector-based Semantic Deduplication (if string matching fails)
        if not self.model:
            new_id = self._generate_new_id(type_prefix, clean_name)
            self.entity_names[new_id] = name
            return new_id
            
        new_emb = self.model.encode(name, convert_to_tensor=True)
        
        best_match_id = None
        highest_score = 0.0
        
        for e_id, e_emb in self.entity_memory.items():
            if e_id.startswith(type_prefix):
                # Calculate cosine similarity
                score = util.cos_sim(new_emb, e_emb).item()
                if score > highest_score:
                    highest_score = score
                    best_match_id = e_id
                    
        if highest_score >= self.vector_threshold and best_match_id:
            # Match found semantically
            return best_match_id
        else:
            # Create a new canonical ID
            new_id = self._generate_new_id(type_prefix, clean_name)
            self.entity_memory[new_id] = new_emb
            self.entity_names[new_id] = name
            return new_id

    def _generate_new_id(self, type_prefix: str, clean_name: str) -> str:
        new_id = f"{type_prefix}{clean_name}"
        counter = 1
        original_new_id = new_id
        while new_id in self.entity_names:
            new_id = f"{original_new_id}_{counter}"
            counter += 1
        return new_id

    def _basic_clean(self, name: str) -> str:
        """Fallback basic string cleaning."""
        clean_name = name.strip().lower()
        clean_name = re.sub(r'[^a-z0-9]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        return clean_name.strip('_')

    def sync_from_graph(self, knowledge_graph) -> None:
        """
        Seed the normalizer's memory from an existing knowledge graph
        to ensure incremental runs keep deduplicating text properly.
        """
        if not self.model or not knowledge_graph:
            return
            
        for node in knowledge_graph.nodes:
            # Re-seed memory using the canonical node ID and its original name property
            node_id = node.id
            name = node.properties.get("name", node.id.split(":", 1)[-1])
            
            if node_id not in self.entity_memory:
                emb = self.model.encode(name, convert_to_tensor=True)
                self.entity_memory[node_id] = emb
                self.entity_names[node_id] = name
                logging.info(f"Seeded normalizer memory: {node_id} -> {name}")

