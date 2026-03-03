import re
import logging
from typing import Dict, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

class VectorOntologyNormalizer:
    """
    Normalizes entity IDs using vector embeddings to merge semantically similar concepts.
    For example, "Transformer" and "Transform architecture" will be mapped to the same ID.
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5', threshold: float = 0.88):
        self.threshold = threshold
        self.entity_memory: Dict[str, object] = {} # { "entity_id": embedding_tensor }
        self.entity_names: Dict[str, str] = {}     # { "entity_id": "Original Name" }
        
        if ST_AVAILABLE:
            logging.info(f"Loading Vector Ontology Model: {model_name}... (First time may take a moment)")
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
        
        # 2. If vector model is not available, just use basic ID
        if not self.model:
            return f"{node_type_value.lower()}:{clean_name}"
            
        # 3. Vector-based Deduplication
        new_emb = self.model.encode(name, convert_to_tensor=True)
        
        best_match_id = None
        highest_score = 0.0
        
        # Compare against existing entities of the SAME type
        type_prefix = f"{node_type_value.lower()}:"
        
        for e_id, e_emb in self.entity_memory.items():
            if e_id.startswith(type_prefix):
                # Calculate cosine similarity
                score = util.cos_sim(new_emb, e_emb).item()
                if score > highest_score:
                    highest_score = score
                    best_match_id = e_id
                    
        if highest_score >= self.threshold and best_match_id:
            # Match found
            return best_match_id
        else:
            # Create a new canonical ID
            new_id = f"{type_prefix}{clean_name}"
            
            # Handle potential exact hash collisions from clean_name but different semantics
            counter = 1
            original_new_id = new_id
            while new_id in self.entity_memory:
                new_id = f"{original_new_id}_{counter}"
                counter += 1
                
            self.entity_memory[new_id] = new_emb
            self.entity_names[new_id] = name
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

