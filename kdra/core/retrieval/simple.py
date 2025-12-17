from typing import List, Tuple, Dict
import random
from kdra.core.schemas import PaperChunk
from kdra.core.retrieval.base import BaseRetriever

class MockRetriever(BaseRetriever):
    """
    A mock retriever implementation for testing and architectural validation.
    Does not use actual embeddings, but simulates the interface.
    """
    
    def __init__(self):
        self.chunks: List[PaperChunk] = []

    def index(self, chunks: List[PaperChunk]) -> None:
        """
        Store chunks in memory.
        """
        self.chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PaperChunk, float]]:
        """
        Return random chunks with fake scores, or simple keyword matching if possible.
        """
        if not self.chunks:
            return []
            
        # Simple keyword matching simulation
        scored_chunks = []
        query_terms = set(query.lower().split())
        
        for chunk in self.chunks:
            score = 0.0
            text_lower = chunk.text.lower()
            
            # Basic term frequency scoring
            for term in query_terms:
                if term in text_lower:
                    score += 1.0
            
            # Add a tiny bit of noise to avoid strict ties and simulate vector nuances
            score += random.random() * 0.1
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # If no matches found, return random ones (fallback behavior for mock)
        if not scored_chunks:
             # Return random sample if no keyword match (just to ensure pipeline continuity in dev)
             # In production, this would return empty.
             sample_size = min(len(self.chunks), top_k)
             random_sample = random.sample(self.chunks, sample_size)
             return [(chunk, random.random()) for chunk in random_sample]

        return scored_chunks[:top_k]
