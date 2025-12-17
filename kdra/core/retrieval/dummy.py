from typing import List, Tuple
from kdra.core.schemas import PaperChunk
from kdra.core.retrieval.base import BaseRetriever

class DummyRetriever(BaseRetriever):
    """
    A dummy retriever that stores chunks in memory and returns them based on simple keyword matching.
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
        Retrieve chunks. For dummy purposes, just returns the first few chunks 
        that contain any query word, or just the first few if no match.
        """
        if not self.chunks:
            return []
            
        query_words = set(query.lower().split())
        scored = []
        
        for chunk in self.chunks:
            score = 0.0
            text_lower = chunk.text.lower()
            for word in query_words:
                if word in text_lower:
                    score += 1.0
            
            if score > 0:
                scored.append((chunk, score))
                
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Fallback if no matches
        if not scored:
            return [(c, 0.1) for c in self.chunks[:top_k]]
            
        return scored[:top_k]
